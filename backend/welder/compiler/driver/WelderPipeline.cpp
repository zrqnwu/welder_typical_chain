#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;

namespace {

static Operation *predicateAsyncCopyOnly(RewriterBase &rewriter, Operation *op,
                                         Value predicate) {
  if (isMemoryEffectFree(op) ||
      isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp,
          nvgpu::DeviceAsyncWaitOp>(op))
    return op;

  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  if (!asyncCopyOp)
    return nullptr;

  Location loc = asyncCopyOp.getLoc();
  Value dstElements =
      arith::ConstantOp::create(rewriter, loc, asyncCopyOp.getDstElementsAttr());
  Value originalSrcElements =
      asyncCopyOp.getSrcElements() ? asyncCopyOp.getSrcElements() : dstElements;
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value srcElements =
      arith::SelectOp::create(rewriter, loc, predicate, originalSrcElements, c0);

  auto predCopy = nvgpu::DeviceAsyncCopyOp::create(
      rewriter, loc, nvgpu::DeviceAsyncTokenType::get(op->getContext()),
      asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), asyncCopyOp.getSrc(),
      asyncCopyOp.getSrcIndices(), asyncCopyOp.getDstElementsAttr(), srcElements,
      asyncCopyOp.getBypassL1Attr());
  rewriter.replaceOp(asyncCopyOp, predCopy);
  return predCopy;
}

LogicalResult writeModuleToFile(Operation *op, StringRef path) {
  if (path.empty())
    return success();
  std::string errorMessage;
  auto outFile = openOutputFile(path, &errorMessage);
  if (!outFile) {
    llvm::errs() << "error: cannot open output file: " << path << "\n";
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  op->print(outFile->os());
  outFile->os() << "\n";
  outFile->keep();
  return success();
}

static LogicalResult flattenAsyncCopyIfOps(scf::ForOp loop) {
  Block *body = loop.getBody();
  if (!body)
    return success();

  auto makeZeroLike = [](OpBuilder &b, Location loc, Type ty) -> Value {
    if (ty.isIndex())
      return arith::ConstantIndexOp::create(b, loc, 0);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(b, loc, intTy, 0);
    return {};
  };

  auto makeConstLike = [&](OpBuilder &b, Location loc, Type ty,
                           int64_t v) -> Value {
    if (ty.isIndex())
      return arith::ConstantIndexOp::create(b, loc, v);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(b, loc, intTy, v);
    return {};
  };

  auto asValue = [](OpBuilder &b, Location loc, OpFoldResult ofr) -> Value {
    if (Value v = ofr.dyn_cast<Value>())
      return v;
    Attribute attr = ofr.dyn_cast<Attribute>();
    if (!attr)
      return {};
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      if (intAttr.getType().isIndex())
        return arith::ConstantIndexOp::create(b, loc, intAttr.getInt());
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType()))
        return arith::ConstantIntOp::create(b, loc, intTy, intAttr.getInt());
    }
    return {};
  };

  for (Operation *op = &body->front(); op;) {
    Operation *next = op->getNextNode();
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp.getElseRegion().empty()) {
      op = next;
      continue;
    }

    bool hasAsync = false;
    ifOp.getThenRegion().walk([&](Operation *nested) {
      if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp,
              nvgpu::DeviceAsyncWaitOp>(nested))
        hasAsync = true;
    });
    if (!hasAsync) {
      op = next;
      continue;
    }

    Value pred = ifOp.getCondition();
    Block &thenBlock = ifOp.getThenRegion().front();

  // 在 if 之前将 then-block 内操作外提。
    //
  // 仅外提以下操作：
  // - async-copy 相关操作（device_async_copy/create_group/wait）
  // - 无内存副作用操作（subview、affine/arithmetic 等）
  // - barrier（必须保持统一执行）
    //
  // 任意带副作用的回退路径（如同步 linalg.copy）仍保留在
  // 原 if 中，确保继续受 `pred` 条件控制。
    SmallVector<Operation *, 32> movedOps;
    SmallVector<nvgpu::DeviceAsyncCopyOp, 8> movedCopies;
    for (Operation &nested :
         llvm::make_early_inc_range(thenBlock.getOperations())) {
      if (isa<scf::YieldOp>(nested))
        continue;
      bool isAsync =
          isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp,
              nvgpu::DeviceAsyncWaitOp>(nested);
      bool isBarrier = isa<gpu::BarrierOp>(nested);
      bool movable = isAsync || isBarrier ||
                     (nested.getNumRegions() == 0 && isMemoryEffectFree(&nested));
      if (!movable)
        continue;
      if (auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(nested))
        movedCopies.push_back(copy);
      nested.moveBefore(ifOp);
      movedOps.push_back(&nested);
    }

  // 对被谓词关闭 lane 外提后的 subview 做截断/重定向：
  // - 全局源（global src）：offsets -> 0
  // - shared dst：offsets -> [dim0, 0, 0, ...]（每 stage 需要 padding）
    for (Operation *moved : movedOps) {
      auto sv = dyn_cast<memref::SubViewOp>(moved);
      if (!sv)
        continue;

      OpBuilder b(sv);
      Location loc = sv.getLoc();

      SmallVector<int64_t, 4> sinkOffsets(sv.getMixedOffsets().size(), 0);
      if (auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType())) {
        if (std::optional<int64_t> ms = srcTy.getMemorySpaceAsInt()) {
          if (*ms == 3 && srcTy.hasStaticShape() && srcTy.getRank() >= 1) {
            sinkOffsets[0] = srcTy.getDimSize(0);
          }
        }
      }

      SmallVector<OpFoldResult, 4> newOffsets;
      newOffsets.reserve(sv.getMixedOffsets().size());
      for (auto [idx, off] : llvm::enumerate(sv.getMixedOffsets())) {
        Value offV = asValue(b, loc, off);
        if (!offV) {
          newOffsets.push_back(off);
          continue;
        }
        Value sink =
            makeConstLike(b, loc, offV.getType(), sinkOffsets[idx]);
        if (!sink) {
          newOffsets.push_back(off);
          continue;
        }
        Value safeOff = arith::SelectOp::create(b, loc, pred, offV, sink);
        newOffsets.push_back(safeOff);
      }

      auto newSv = memref::SubViewOp::create(b, loc, sv.getType(),
                                            sv.getSource(), newOffsets,
                                            sv.getMixedSizes(),
                                            sv.getMixedStrides());
      sv.replaceAllUsesWith(newSv.getResult());
      sv.erase();
    }

  // 通过设置 srcElements = pred ? dstElements : 0 对 async copy 做谓词化，
  // 并在 pred==false 时强制索引为 0（使被屏蔽 lane 仅写入
  // sink 位置）。
    for (nvgpu::DeviceAsyncCopyOp copy : movedCopies) {
      OpBuilder b(copy);
      Location loc = copy.getLoc();

      int64_t dstElems = copy.getDstElementsAttr().getInt();
      Value dstElemsV = arith::ConstantIndexOp::create(b, loc, dstElems);
      Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value srcElems = arith::SelectOp::create(b, loc, pred, dstElemsV, c0);

      SmallVector<Value, 4> predDstIdx;
      predDstIdx.reserve(copy.getDstIndices().size());
      for (Value idx : copy.getDstIndices()) {
        Value z = makeZeroLike(b, loc, idx.getType());
        predDstIdx.push_back(arith::SelectOp::create(b, loc, pred, idx, z));
      }
      SmallVector<Value, 4> predSrcIdx;
      predSrcIdx.reserve(copy.getSrcIndices().size());
      for (Value idx : copy.getSrcIndices()) {
        Value z = makeZeroLike(b, loc, idx.getType());
        predSrcIdx.push_back(arith::SelectOp::create(b, loc, pred, idx, z));
      }

      auto newCopy = nvgpu::DeviceAsyncCopyOp::create(
          b, loc, nvgpu::DeviceAsyncTokenType::get(loop.getContext()),
          /* dst=*/copy.getDst(), /*dstIndices=*/predDstIdx,
          /* src=*/copy.getSrc(), /*srcIndices=*/predSrcIdx,
          /* dstElements=*/copy.getDstElementsAttr(),
          /* srcElements=*/srcElems,
          /* bypassL1=*/copy.getBypassL1Attr());
      copy.getResult().replaceAllUsesWith(newCopy.getResult());
      copy.erase();
    }

    bool hasRemaining = false;
    for (Operation &nested : thenBlock.getOperations()) {
      if (isa<scf::YieldOp>(nested))
        continue;
      hasRemaining = true;
      break;
    }
    if (!hasRemaining)
      ifOp.erase();
    op = next;
  }

  return success();
}

static LogicalResult collectStage0PipeliningOps(
    scf::ForOp forOp, llvm::SmallPtrSetImpl<Operation *> &stage0Ops) {
  llvm::SmallPtrSet<Operation *, 4> barriers;
  for (Operation &op : *forOp.getBody()) {
  // 若存在嵌套操作，统一按 non-stage0 处理。loop pipeliner
  // 会按调度需要自行克隆/移动操作。
    if (op.getNumRegions() > 0)
      continue;

    if (isa<gpu::BarrierOp>(op)) {
      barriers.insert(&op);
      continue;
    }

    if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
      stage0Ops.insert(&op);
      stage0Ops.insert(std::make_move_iterator(barriers.begin()),
                       std::make_move_iterator(barriers.end()));
      barriers.clear();
      continue;
    }
  }
  return success();
}

static void setAsyncWaitGroupsInFlight(OpBuilder &b, Operation *op,
                                       scf::PipeliningOption::PipelinerPart part,
                                       unsigned iteration, unsigned depth) {
  auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op);
  if (!waitOp || waitOp.getNumGroups())
    return;

  int numGroupInFlight = 0;
  if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
      part == scf::PipeliningOption::PipelinerPart::Prologue) {
    numGroupInFlight = static_cast<int>(depth) - 1;
  } else {
    assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
    numGroupInFlight = static_cast<int>(depth) - 1 - static_cast<int>(iteration);
  }
  waitOp.setNumGroups(numGroupInFlight);
}

static void getPipelineStages(
    scf::ForOp forOp,
    std::vector<std::pair<Operation *, unsigned>> &opsWithPipelineStages,
    unsigned lastStage, llvm::SmallPtrSetImpl<Operation *> &stage0Ops) {
  SetVector<Operation *> dependencies;
  BackwardSliceOptions options([&](Operation *visited) {
    return visited->getBlock() == forOp.getBody();
  });
  options.inclusive = true;

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (stage0Ops.contains(&op)) {
      LogicalResult result = getBackwardSlice(&op, &dependencies, options);
      (void)result;
      assert(succeeded(result) && "expected a backward slice");
    }
  }

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (!dependencies.contains(&op) && !isa<scf::YieldOp>(op))
      opsWithPipelineStages.emplace_back(&op, lastStage);
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (dependencies.contains(&op))
      opsWithPipelineStages.emplace_back(&op, 0);
  }
}

static LogicalResult pipelineAsyncCopiesInLaunch(gpu::LaunchOp launch,
                                                int64_t depth,
                                                bool peelEpilogue,
                                                bool setAsyncWaitGroups) {
  IRRewriter rewriter(launch.getContext());

  SmallVector<scf::ForOp, 4> loops;
  launch.walk([&](scf::ForOp loop) {
    bool hasAsync = false;
    loop.getBody()->walk([&](nvgpu::DeviceAsyncCopyOp) { hasAsync = true; });
    if (hasAsync)
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(flattenAsyncCopyIfOps(loop)))
      return failure();

    llvm::SmallPtrSet<Operation *, 16> stage0Ops;
    (void)collectStage0PipeliningOps(loop, stage0Ops);
    if (stage0Ops.empty())
      continue;

    scf::PipeliningOption options;
    unsigned maxDepth = static_cast<unsigned>(std::max<int64_t>(2, depth));
    unsigned lastStage = maxDepth - 1;
    options.getScheduleFn =
        [&](scf::ForOp schedulingFor,
            std::vector<std::pair<Operation *, unsigned>> &ops) {
          if (schedulingFor != loop)
            return;
          getPipelineStages(loop, ops, lastStage, stage0Ops);
        };
    if (setAsyncWaitGroups) {
      options.annotateFn = [&](Operation *op,
                               scf::PipeliningOption::PipelinerPart part,
                               unsigned iteration) {
        setAsyncWaitGroupsInFlight(rewriter, op, part, iteration, maxDepth);
      };
    }
    options.peelEpilogue = peelEpilogue;
    options.supportDynamicLoops = true;
    options.predicateFn = predicateAsyncCopyOnly;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);
    bool modifiedIR = false;
    FailureOr<scf::ForOp> pipelined =
        scf::pipelineForLoop(rewriter, loop, options, &modifiedIR);
    if (failed(pipelined)) {
      auto diag = loop.emitError()
                  << "software pipelining failed (scf.pipelineForLoop)";
      if (modifiedIR)
        diag.attachNote() << "IR was partially modified before failure";
      return failure();
    }
  }
  return success();
}

} // 命名空间

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input.mlir>"), llvm::cl::Required);
  llvm::cl::opt<std::string> outputFilename(
      "output", llvm::cl::desc("Output MLIR path ('-' for stdout)"),
      llvm::cl::init("-"));
  llvm::cl::opt<int64_t> pipelineDepth(
      "pipeline-depth",
      llvm::cl::desc("Software pipeline depth (stages, default 2)"),
      llvm::cl::init(2));
  llvm::cl::opt<bool> pipelinePeelEpilogue(
      "pipeline-peel-epilogue",
      llvm::cl::desc("Peel epilogue when pipelining (more robust)"),
      llvm::cl::init(true));
  llvm::cl::opt<bool> pipelineSetAsyncWaitGroups(
      "pipeline-set-async-wait-groups",
      llvm::cl::desc(
          "Set cp.async wait groups in-flight (emit wait_group N>0) instead of "
          "the conservative wait_group 0 default"),
      llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv, "welder-pipeline\n");

  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  bufferization::BufferizationDialect, cf::ControlFlowDialect,
                  func::FuncDialect, gpu::GPUDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, nvgpu::NVGPUDialect,
                  scf::SCFDialect, tensor::TensorDialect,
                  vector::VectorDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "error: cannot open input file: " << inputFilename << "\n";
    llvm::errs() << errorMessage << "\n";
    return 2;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR: " << inputFilename << "\n";
    return 2;
  }

  WalkResult wr = module->walk([&](gpu::LaunchOp launch) -> WalkResult {
    if (failed(pipelineAsyncCopiesInLaunch(launch, pipelineDepth,
                                          pipelinePeelEpilogue,
                                          pipelineSetAsyncWaitGroups)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (wr.wasInterrupted()) {
    llvm::errs() << "error: software pipelining failed\n";
    return 1;
  }

  PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "error: canonicalization after pipelining failed\n";
    return 1;
  }

  if (outputFilename == "-") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  if (failed(writeModuleToFile(*module, outputFilename)))
    return 2;
  return 0;
}

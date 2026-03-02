#include "WelderCompilerAsyncPipelineHelpers.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

LogicalResult flattenAsyncCopyIfOps(scf::ForOp loop) {
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

    // 将 then 块中的可移动操作提升到 scf.if 之前。
    //
    // 仅提升以下操作：
    // - async-copy 相关操作（device_async_copy/create_group/wait）
    // - 无内存副作用操作（subview、affine/arith 等）
    // - barrier（必须保持所有线程一致执行）
    //
    // 带副作用的回退路径（例如同步 linalg.copy）必须留在原始 if 内，
    // 继续受 `pred` 谓词控制。
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

    // 安全性：去掉 scf.if 后，被谓词屏蔽的线程 lane 仍会计算子视图偏移，
    // 可能越界。某些后端即使在 srcElements=0 时也要求地址合法，而且
    // cp.async 在 srcElements=0 时仍会向目标写零。
    // 因此，这里把被屏蔽 lane 的索引重定向到 tile 外的“沉降区”。
    for (Operation *moved : movedOps) {
      auto sv = dyn_cast<memref::SubViewOp>(moved);
      if (!sv)
        continue;

      OpBuilder b(sv);
      Location loc = sv.getLoc();

      // 默认沉降区偏移为 0（保证边界内安全）。
      SmallVector<int64_t, 4> sinkOffsets(sv.getMixedOffsets().size(), 0);

      // 对 shared-memory 目标，沉降区放在 tile 之后（即 [dim0, 0, 0, ...]）。
      // 这要求底层 workgroup 分配预留额外 padding。
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
        Value sink = makeConstLike(b, loc, offV.getType(), sinkOffsets[idx]);
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

    // 通过设置 srcElements = pred ? srcElements : 0 来谓词化 async copy。
    for (nvgpu::DeviceAsyncCopyOp copy : movedCopies) {
      OpBuilder b(copy);
      Location loc = copy.getLoc();
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);

      int64_t dstElems = copy.getDstElementsAttr().getInt();
      Value dstElemsV = b.create<arith::ConstantIndexOp>(loc, dstElems);
      Value origSrcElems =
          copy.getSrcElements() ? copy.getSrcElements() : dstElemsV;
      Value srcElems = b.create<arith::SelectOp>(loc, pred, origSrcElems, zero);

      SmallVector<Value, 4> predDstIdx;
      predDstIdx.reserve(copy.getDstIndices().size());
      for (Value idx : copy.getDstIndices()) {
        Value z = makeZeroLike(b, loc, idx.getType());
        predDstIdx.push_back(b.create<arith::SelectOp>(loc, pred, idx, z));
      }
      SmallVector<Value, 4> predSrcIdx;
      predSrcIdx.reserve(copy.getSrcIndices().size());
      for (Value idx : copy.getSrcIndices()) {
        Value z = makeZeroLike(b, loc, idx.getType());
        predSrcIdx.push_back(b.create<arith::SelectOp>(loc, pred, idx, z));
      }

      auto newCopy = b.create<nvgpu::DeviceAsyncCopyOp>(
          loc, nvgpu::DeviceAsyncTokenType::get(loop.getContext()),
          /*dst=*/copy.getDst(), /*dstIndices=*/predDstIdx,
          /*src=*/copy.getSrc(), /*srcIndices=*/predSrcIdx,
          /*dstElements=*/copy.getDstElementsAttr(),
          /*srcElements=*/srcElems,
          /*bypassL1=*/copy.getBypassL1Attr());
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

LogicalResult collectStage0PipeliningOps(
    scf::ForOp forOp, llvm::SmallPtrSetImpl<Operation *> &stage0Ops) {
  llvm::SmallPtrSet<Operation *, 4> barriers;
  for (Operation &op : *forOp.getBody()) {
    if (op.getNumRegions() > 0)
      return failure();

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

void setAsyncWaitGroupsInFlight(OpBuilder &b, Operation *op,
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
    numGroupInFlight =
        static_cast<int>(depth) - 1 - static_cast<int>(iteration);
  }
  waitOp.setNumGroups(numGroupInFlight);
}

void getPipelineStages(
    scf::ForOp forOp,
    std::vector<std::pair<Operation *, unsigned>> &opsWithPipelineStages,
    unsigned depth, llvm::SmallPtrSetImpl<Operation *> &stage0Ops) {
  llvm::SetVector<Operation *> dependencies;
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
      opsWithPipelineStages.emplace_back(&op, depth);
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (dependencies.contains(&op))
      opsWithPipelineStages.emplace_back(&op, 0);
  }
}

Operation *predicateAsyncCopyOnly(RewriterBase &rewriter, Operation *op,
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

[[maybe_unused]] LogicalResult pipelineAsyncCopiesInLaunch(
    gpu::LaunchOp launch, int64_t depth, bool peelEpilogue,
    bool setAsyncWaitGroups) {
  IRRewriter rewriter(launch.getContext());

  SmallVector<scf::ForOp, 4> loops;
  launch.walk([&](scf::ForOp loop) {
    bool hasAsync = false;
    loop.getBody()->walk([&](nvgpu::DeviceAsyncCopyOp) { hasAsync = true; });
    if (hasAsync)
      loops.push_back(loop);
  });

  auto hasWorkgroupAllocOrDealloc = [](scf::ForOp loop) -> bool {
    for (Operation &op : *loop.getBody()) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        if (auto mt = dyn_cast<MemRefType>(alloc.getType())) {
          if (std::optional<int64_t> ms = mt.getMemorySpaceAsInt())
            if (*ms == 3)
              return true;
        }
      }
      if (auto dealloc = dyn_cast<memref::DeallocOp>(op)) {
        if (auto mt = dyn_cast<MemRefType>(dealloc.getMemref().getType())) {
          if (std::optional<int64_t> ms = mt.getMemorySpaceAsInt())
            if (*ms == 3)
              return true;
        }
      }
    }
    return false;
  };

  for (scf::ForOp loop : loops) {
    // workgroup 分配的多缓冲/提升会在后续阶段处理（当前通过 pass 插件）。
    // 若循环体内仍存在 shared buffer 的 alloc/dealloc，则不能做流水化，
    // 否则语义不正确。
    if (hasWorkgroupAllocOrDealloc(loop))
      continue;

    if (failed(flattenAsyncCopyIfOps(loop)))
      return failure();

    llvm::SmallPtrSet<Operation *, 16> stage0Ops;
    if (failed(collectStage0PipeliningOps(loop, stage0Ops)))
      continue;
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
    // 行级 staging 循环常采用“线程分发”形式，其下界/步长可能由
    // gpu.thread_id/gpu.block_dim 动态推导。这里开启动态循环支持，
    // 并提供谓词回调，让 scf.pipelineForLoop 正确处理此类场景
    // （通过 srcElements=0 谓词化 async copy）。
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
      if (!peelEpilogue)
        diag.attachNote() << "try enabling --pipeline-peel-epilogue";
      if (modifiedIR)
        diag.attachNote() << "IR was partially modified before failure";
      return failure();
    }
  }

  return success();
}

} // namespace

namespace welder::compiler {

void padWorkgroupAllocs(ModuleOp module, int64_t padBytes) {
  if (padBytes <= 0)
    return;

  SmallVector<memref::AllocOp, 8> allocs;
  module.walk([&](memref::AllocOp alloc) {
    auto mt = dyn_cast<MemRefType>(alloc.getType());
    if (!mt || mt.getRank() != 1 || !mt.hasStaticShape())
      return;
    if (!mt.getElementType().isInteger(8))
      return;
    std::optional<int64_t> ms = mt.getMemorySpaceAsInt();
    if (!ms || *ms != 3)
      return;
    allocs.push_back(alloc);
  });

  for (memref::AllocOp alloc : allocs) {
    auto mt = cast<MemRefType>(alloc.getType());
    int64_t n = mt.getNumElements();
    auto newTy = MemRefType::get({n + padBytes}, mt.getElementType(),
                                 mt.getLayout(), mt.getMemorySpace());

    OpBuilder b(alloc);
    auto newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), newTy);
    newAlloc->setAttrs(alloc->getAttrs());
    alloc.replaceAllUsesWith(newAlloc.getResult());
    alloc.erase();
  }
}

} // namespace welder::compiler

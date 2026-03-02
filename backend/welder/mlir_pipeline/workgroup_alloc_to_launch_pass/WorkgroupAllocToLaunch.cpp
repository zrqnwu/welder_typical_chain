// ===- WorkgroupAllocToLaunch.cpp（工作组内存改写）--------------*- C++ -*-===//
//
// 一个轻量级 MLIR pass 插件，主要做三件事：
// - 在 gpu.launch 内找到分配于 GPU workgroup（shared）地址空间的 memref.alloc；
// - 将其改写为 gpu.launch 的 `workgroup(...)` 归属参数（region argument）；
// - 删除对应的 memref.dealloc（语义上是 no-op，并规避 DeallocOpLowering
//   降成 free(ptr<addrspace>) 时的地址空间不匹配）。
// 另外还会把 gpu.launch 内的 memref.copy 展开成显式 scf.for 循环，
// 这样 NVVM pipeline 就不需要依赖运行时 `memrefCopy` 符号（该符号在设备模块不可见）。
// 同时会对部分分配做额外改写，保证生成 kernel 可运行：
// - Host 侧 `memref.alloc` -> `gpu.alloc host_shared`（传入 gpu.launch 的指针对 GPU 可见，
//   避免把 host malloc 指针直接传给设备导致 illegal-address）；
// - Device 侧静态 `memref.alloc` -> `memref.alloca`（避免设备 malloc）。
//
// 这是一套偏工程化的折中方案：在继续使用
// structured.promote(... memory_space = #gpu.address_space<workgroup>) 的同时，
// 让最终 GPU 二进制里真正得到可用的 shared 内存布局。
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

#include <limits>

using namespace mlir;

namespace {

static bool isWorkgroupMemory(MemRefType memrefType);

static std::optional<int64_t> getConstIndexValue(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static int64_t getElementByteWidth(Type elemType) {
  if (!elemType)
    return -1;
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return (intTy.getWidth() + 7) / 8;
  if (auto fTy = dyn_cast<FloatType>(elemType))
    return (fTy.getWidth() + 7) / 8;
  return -1;
}

static Value stripViewLike(Value memref) {
  Value v = memref;
  while (v) {
    if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
      v = sub.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto view = v.getDefiningOp<memref::ViewOp>()) {
      v = view.getSource();
      continue;
    }
    if (auto assumed = v.getDefiningOp<memref::AssumeAlignmentOp>()) {
      v = assumed.getMemref();
      continue;
    }
    break;
  }
  return v;
}

static void collectWorkgroupAccesses(Operation *op,
                                     llvm::DenseSet<const void *> &reads,
                                     llvm::DenseSet<const void *> &writes) {
  if (!op)
    return;
  op->walk([&](Operation *nested) {
    auto recordMemref = [&](Value memref, bool isWrite) {
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      if (!memrefType || !isWorkgroupMemory(memrefType))
        return;
      Value root = stripViewLike(memref);
      auto rootType = dyn_cast<MemRefType>(root.getType());
      if (!rootType || !isWorkgroupMemory(rootType))
        return;
      const void *key = root.getAsOpaquePointer();
      if (isWrite)
        writes.insert(key);
      else
        reads.insert(key);
    };

    // 重要：这个阶段 GPU 区域里可能仍有 linalg op（尚未 convert-linalg-to-loops）。
    // 这里按保守读写模型处理：视为“读全部输入、写全部输出”，否则可能误删
    // shared-memory copy 与计算之间必需的 gpu.barrier。
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(nested)) {
      for (Value in : linalgOp.getDpsInputs())
        recordMemref(in, /*isWrite=*/false);
      for (Value out : linalgOp.getDpsInits())
        recordMemref(out, /*isWrite=*/true);
      return;
    }

    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
      recordMemref(load.getMemref(), /*isWrite=*/false);
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(nested)) {
      recordMemref(store.getMemref(), /*isWrite=*/true);
      return;
    }
    if (auto tr = dyn_cast<vector::TransferReadOp>(nested)) {
      recordMemref(tr.getBase(), /*isWrite=*/false);
      return;
    }
    if (auto tw = dyn_cast<vector::TransferWriteOp>(nested)) {
      recordMemref(tw.getBase(), /*isWrite=*/true);
      return;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(nested)) {
      recordMemref(copy.getSource(), /*isWrite=*/false);
      recordMemref(copy.getTarget(), /*isWrite=*/true);
      return;
    }
  });
}

static void mergeRedundantCopyBarriersInBlock(Block &block) {
  // 可删除屏障 B1 的条件：
  // [writes] ... B1 ... [这些 memref 无后续读写] ... B2
  // 即 B2 已足够同步前面的写入。
  if (block.empty())
    return;
  Operation *prevBarrier = nullptr;
  for (Operation *op = &block.front(); op; ) {
    Operation *next = op->getNextNode();
    auto b1 = dyn_cast<gpu::BarrierOp>(op);
    if (!b1) {
      op = next;
      continue;
    }
    if (b1->hasAttr("welder.keep_barrier")) {
      prevBarrier = op;
      op = next;
      continue;
    }

    // 在同一个 block 内寻找下一个 barrier。
    gpu::BarrierOp b2;
    for (Operation *scan = op->getNextNode(); scan; scan = scan->getNextNode()) {
      if (auto bb = dyn_cast<gpu::BarrierOp>(scan)) {
        b2 = bb;
        break;
      }
    }
    if (!b2) {
      prevBarrier = op;
      op = next;
      continue;
    }

    llvm::DenseSet<const void *> writesBefore;
    {
      Operation *begin =
          prevBarrier ? prevBarrier->getNextNode() : &block.front();
      for (Operation *it = begin; it && it != op; it = it->getNextNode()) {
        llvm::DenseSet<const void *> r, w;
        collectWorkgroupAccesses(it, r, w);
        writesBefore.insert(w.begin(), w.end());
      }
    }

    llvm::DenseSet<const void *> readsBetween, writesBetween;
    for (Operation *it = op->getNextNode(); it && it != b2; it = it->getNextNode())
      collectWorkgroupAccesses(it, readsBetween, writesBetween);

    auto intersects = [](const llvm::DenseSet<const void *> &a,
                         const llvm::DenseSet<const void *> &b) -> bool {
      for (const void *k : a)
        if (b.contains(k))
          return true;
      return false;
    };

    if (!intersects(readsBetween, writesBefore) &&
        !intersects(writesBetween, writesBefore)) {
      b1.erase();
      op = next;
      continue;
    }

    prevBarrier = op;
    op = next;
  }
}

// 一个小型 LLVM dialect 修正 pass：
// - 强制 vector<4xf32> load/store 使用 alignment=16；
// - 强制降级后的 workgroup global（__wg_*）也使用 alignment=16，
//   便于 shared 内存向量化访问。
//
// 这里是工程化处理：我们已知 tiled copy 地址满足 16-byte 对齐（float4），
// 但默认 lowering 偏保守会发出 alignment=4，从而阻断 PTX 的 ld/st v4 生成。
struct LLVMVector4AlignPass
    : public PassWrapper<LLVMVector4AlignPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMVector4AlignPass)

  StringRef getArgument() const final { return "llvm-vector4-align"; }
  StringRef getDescription() const final {
    return "Force alignment=16 for vector<4xf32> loads/stores and __wg_* shared "
           "globals to encourage ld/st v4 in NVPTX";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // 本 pass 仅调整已有 LLVM dialect op 的属性，不改动控制流/语义。
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    auto isVector4F32 = [](Type ty) -> bool {
      auto vecTy = dyn_cast<VectorType>(ty);
      if (!vecTy || vecTy.getRank() != 1)
        return false;
      if (vecTy.getNumElements() != 4)
        return false;
      return vecTy.getElementType().isF32();
    };

    auto align16 = b.getI64IntegerAttr(16);

    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();

      if (opName == "llvm.mlir.global") {
        auto addrSpaceAttr = op->getAttrOfType<IntegerAttr>("addr_space");
        if (!addrSpaceAttr || addrSpaceAttr.getInt() != 3)
          return;
        auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name");
        if (!symNameAttr || !symNameAttr.getValue().starts_with("__wg_"))
          return;
        auto alignAttr = op->getAttrOfType<IntegerAttr>("alignment");
        int64_t cur = alignAttr ? alignAttr.getInt() : 0;
        if (cur < 16)
          op->setAttr("alignment", align16);
        return;
      }

      if (opName == "llvm.load") {
        if (op->getNumResults() != 1)
          return;
        if (!isVector4F32(op->getResult(0).getType()))
          return;
        auto alignAttr = op->getAttrOfType<IntegerAttr>("alignment");
        int64_t cur = alignAttr ? alignAttr.getInt() : 0;
        if (cur < 16)
          op->setAttr("alignment", align16);
        return;
      }

      if (opName == "llvm.store") {
        if (op->getNumOperands() < 1)
          return;
        if (!isVector4F32(op->getOperand(0).getType()))
          return;
        auto alignAttr = op->getAttrOfType<IntegerAttr>("alignment");
        int64_t cur = alignAttr ? alignAttr.getInt() : 0;
        if (cur < 16)
          op->setAttr("alignment", align16);
        return;
      }
    });
  }
};

static bool expandStaticMemrefCopy(memref::CopyOp copy) {
  auto srcType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!srcType || !dstType)
    return false;
  if (srcType.getRank() != dstType.getRank())
    return false;
  if (srcType.getElementType() != dstType.getElementType())
    return false;

  const int64_t rank = srcType.getRank();
  for (int64_t d = 0; d < rank; ++d) {
    if (srcType.isDynamicDim(d) || dstType.isDynamicDim(d))
      return false;
    if (srcType.getDimSize(d) != dstType.getDimSize(d))
      return false;
  }

  // 平凡情形：源和目标是同一个 SSA 值。
  if (copy.getSource() == copy.getTarget()) {
    copy.erase();
    return true;
  }

  OpBuilder b(copy);
  Location loc = copy.getLoc();

  // 在 gpu.launch 内，memref.copy 默认会被每个 GPU 线程执行。
  // 对 shared/workgroup 缓冲区这通常不是我们想要的：会产生冗余流量，
  // 甚至引入数据竞争。只要 copy 涉及 workgroup 内存，就保守地改为
  // 每个 block 仅执行一次（thread0）并做同步。
  if (copy->getParentOfType<gpu::LaunchOp>() &&
      (isWorkgroupMemory(srcType) || isWorkgroupMemory(dstType))) {
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value tx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value ty = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value tz = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
    Value isTx0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, zero);
    Value isTy0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, ty, zero);
    Value isTz0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tz, zero);
    Value pred = b.create<arith::AndIOp>(loc, isTx0, isTy0);
    pred = b.create<arith::AndIOp>(loc, pred, isTz0);

    auto ifOp = b.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
    OpBuilder innerBuilder = ifOp.getThenBodyBuilder();

    Value one = innerBuilder.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> upperBounds;
    upperBounds.reserve(rank);
    for (int64_t d = 0; d < rank; ++d)
      upperBounds.push_back(
          innerBuilder.create<arith::ConstantIndexOp>(loc, srcType.getDimSize(d)));

    SmallVector<Value> ivs;
    ivs.reserve(rank);
    std::function<void(OpBuilder &, int64_t)> emitLoopNest =
        [&](OpBuilder &builder, int64_t dim) {
          if (dim == rank) {
            Value v =
                builder.create<memref::LoadOp>(loc, copy.getSource(), ivs);
            builder.create<memref::StoreOp>(loc, v, copy.getTarget(), ivs);
            return;
          }

          scf::ForOp forOp =
              builder.create<scf::ForOp>(loc, zero, upperBounds[dim], one);
          OpBuilder nestedBuilder(forOp.getBody(), forOp.getBody()->begin());
          ivs.push_back(forOp.getInductionVar());
          emitLoopNest(nestedBuilder, dim + 1);
          ivs.pop_back();
        };
    emitLoopNest(innerBuilder, 0);

    // 同步：确保复制后的 workgroup 缓冲区对所有线程可见。
    // 注意：copy 可能已位于非一致分支 scf.if（如 thread0 guard）内部。
    // 若直接在内层 if 后插 barrier，屏障会落在外层条件内，导致并非所有线程
    // 都命中屏障，可能触发 GPU 死锁。这里会把屏障上提到包围的 scf.if 之外，
    // 同时保持在当前循环层级内。
    Operation *barrierAfter = ifOp.getOperation();
    while (auto parentIf =
               dyn_cast_or_null<scf::IfOp>(barrierAfter->getParentOp()))
      barrierAfter = parentIf.getOperation();
    b.setInsertionPointAfter(barrierAfter);
    b.create<gpu::BarrierOp>(loc);

    copy.erase();
    return true;
  }

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> upperBounds;
  upperBounds.reserve(rank);
  for (int64_t d = 0; d < rank; ++d)
    upperBounds.push_back(
        b.create<arith::ConstantIndexOp>(loc, srcType.getDimSize(d)));

  SmallVector<Value> ivs;
  ivs.reserve(rank);

  std::function<void(OpBuilder &, int64_t)> emitLoopNest =
      [&](OpBuilder &builder, int64_t dim) {
        if (dim == rank) {
          Value v = builder.create<memref::LoadOp>(loc, copy.getSource(), ivs);
          builder.create<memref::StoreOp>(loc, v, copy.getTarget(), ivs);
          return;
        }

        scf::ForOp forOp =
            builder.create<scf::ForOp>(loc, zero, upperBounds[dim], one);
        OpBuilder innerBuilder(forOp.getBody(), forOp.getBody()->begin());
        ivs.push_back(forOp.getInductionVar());
        emitLoopNest(innerBuilder, dim + 1);
        ivs.pop_back();
      };

  emitLoopNest(b, 0);
  copy.erase();
  return true;
}

static bool isIdenticalSubView(memref::SubViewOp a, memref::SubViewOp b) {
  if (!a || !b)
    return false;
  if (a.getSource() != b.getSource())
    return false;
  if (!llvm::equal(a.getOffsets(), b.getOffsets()))
    return false;
  if (!llvm::equal(a.getSizes(), b.getSizes()))
    return false;
  if (!llvm::equal(a.getStrides(), b.getStrides()))
    return false;
  return true;
}

static bool eraseTrivialSubviewCopy(memref::CopyOp copy) {
  auto srcSv = copy.getSource().getDefiningOp<memref::SubViewOp>();
  auto dstSv = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!srcSv || !dstSv)
    return false;
  if (!isIdenticalSubView(srcSv, dstSv))
    return false;
  copy.erase();
  return true;
}

static bool rewritePrivateBlockTileToThreadTile(memref::CopyOp copy) {
  // 模式（来自融合 matmul epilogue）：
  // block 级私有 tile（alloca）仅通过固定大小的线程 subview 被使用。
  // 若仍把完整 block tile copy 到每个线程，会带来灾难性冗余。
  // 这里改写为“真正的每线程累加 tile”：
  // - 为线程分配 `memref<ttm x ttn x f32>` 的 acc；
  // - 用零初始化 acc；
  // - 将原 subview 使用替换为 acc；
  // - 删除大 alloca 与初始化 copy。
  // 这对应论文调度里的 “shared -> register” 语义层，但不需要完整寄存器层 IR 流水。
  if (!copy)
    return false;
  if (!copy->getParentOfType<gpu::LaunchOp>())
    return false;

  Value dst = copy.getTarget();
  auto dstAlloca = dst.getDefiningOp<memref::AllocaOp>();
  if (!dstAlloca)
    return false;
  auto dstType = dyn_cast<MemRefType>(dstAlloca.getType());
  if (!dstType || !dstType.hasStaticShape())
    return false;
  if (dstType.getRank() != 2)
    return false;
  if (isWorkgroupMemory(dstType))
    return false;

  // 仅对“足够大”的 tile 应用（太小时改写收益不明显）。
  if (dstType.getNumElements() < 256)
    return false;

  // dst alloca 除 init copy 以外的所有使用，必须都是固定线程 tile 尺寸的 subview。
  SmallVector<memref::SubViewOp, 8> subviews;
  SmallVector<int64_t, 2> threadTile;
  for (Operation *user : dst.getUsers()) {
    // 忽略初始化 copy 本身。
    if (user == copy.getOperation())
      continue;
    auto sv = dyn_cast<memref::SubViewOp>(user);
    if (!sv)
      return false;
    auto svTy = dyn_cast<MemRefType>(sv.getResult().getType());
    if (!svTy || !svTy.hasStaticShape())
      return false;
    if (svTy.getRank() != 2)
      return false;

    // 注意：memref.subview 的静态 size/stride 在属性里，
    // getSizes()/getStrides() 只返回动态部分；形状校验应以结果类型为准。
    if (svTy.getDimSize(0) <= 0 || svTy.getDimSize(1) <= 0)
      return false;
    if (dstType.getDimSize(0) <= 0 || dstType.getDimSize(1) <= 0)
      return false;
    if (svTy.getDimSize(0) > dstType.getDimSize(0) ||
        svTy.getDimSize(1) > dstType.getDimSize(1))
      return false;

    if (threadTile.empty()) {
      threadTile = {svTy.getDimSize(0), svTy.getDimSize(1)};
    } else if (threadTile[0] != svTy.getDimSize(0) ||
               threadTile[1] != svTy.getDimSize(1)) {
      return false;
    }

    // 线程 tile subview 仅允许被 linalg op 使用。
    for (OpOperand &use : sv.getResult().getUses()) {
      Operation *owner = use.getOwner();
      if (!isa<linalg::LinalgOp>(owner))
        return false;
    }

    subviews.push_back(sv);
  }

  if (subviews.empty() || threadTile.empty())
    return false;

  // 为每个线程分配累加 tile。
  OpBuilder b(copy);
  Location loc = copy.getLoc();
  auto accTy = MemRefType::get(threadTile, dstType.getElementType());
  auto acc = b.create<memref::AllocaOp>(loc, accTy);
  if (auto alignAttr = dstAlloca.getAlignmentAttr())
    acc.setAlignmentAttr(alignAttr);

  // 将累加 tile 清零初始化。
  Value zeroIdx = b.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub0 = b.create<arith::ConstantIndexOp>(loc, threadTile[0]);
  Value ub1 = b.create<arith::ConstantIndexOp>(loc, threadTile[1]);
  auto zeroAttr = cast<TypedAttr>(b.getZeroAttr(dstType.getElementType()));
  Value zeroVal = b.create<arith::ConstantOp>(loc, zeroAttr);

  scf::ForOp for0 = b.create<scf::ForOp>(loc, zeroIdx, ub0, oneIdx);
  OpBuilder b0(for0.getBody(), for0.getBody()->begin());
  scf::ForOp for1 =
      b0.create<scf::ForOp>(loc, zeroIdx, ub1, oneIdx);
  OpBuilder b1(for1.getBody(), for1.getBody()->begin());
  b1.create<memref::StoreOp>(loc, zeroVal, acc,
                             ValueRange{for0.getInductionVar(),
                                        for1.getInductionVar()});

  // 将所有线程 tile subview 使用重定向到新的累加器。
  for (memref::SubViewOp sv : subviews) {
    sv.getResult().replaceAllUsesWith(acc.getResult());
    sv.erase();
  }

  // 删除昂贵的整块初始化 copy 和巨大的私有 tile 缓冲区。
  copy.erase();
  if (dstAlloca.getResult().use_empty())
    dstAlloca.erase();
  return true;
}

static void convertHostAllocToGpuHostShared(memref::AllocOp alloc) {
  // 仅处理 host 侧 alloc。gpu.launch 区域内的分配由 GPU 线程执行，
  // 不能直接改成 gpu.alloc。
  if (alloc->getParentOfType<gpu::LaunchOp>())
    return;

  auto memrefType = dyn_cast<MemRefType>(alloc.getType());
  if (!memrefType)
    return;
  // 简化处理：仅支持静态形状，不处理 dynamic size/symbol 操作数。
  if (!memrefType.hasStaticShape())
    return;
  if (!alloc.getDynamicSizes().empty())
    return;
  if (!alloc.getSymbolOperands().empty())
    return;

  OpBuilder b(alloc);
  Location loc = alloc.getLoc();
  auto gpuAlloc = b.create<gpu::AllocOp>(
      loc, memrefType,
      /* asyncToken=*/Type(), /*asyncDependencies=*/ValueRange{},
      /* dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{},
      /* hostShared=*/true);
  alloc.getResult().replaceAllUsesWith(gpuAlloc.getMemref());
  alloc.erase();
}

static void promoteDeviceAllocToAlloca(memref::AllocOp alloc) {
  // 仅处理会在设备端执行的分配。
  if (!alloc->getParentOfType<gpu::LaunchOp>())
    return;

  auto memrefType = dyn_cast<MemRefType>(alloc.getType());
  if (!memrefType)
    return;
  // 这里不要处理 workgroup 分配：它们由后续 workgroup-attribution 改写统一处理。
  if (isWorkgroupMemory(memrefType))
    return;
  // 当前仅处理静态形状缓冲区。
  if (!memrefType.hasStaticShape())
    return;
  if (!alloc.getDynamicSizes().empty())
    return;

  // 注意：这里避免把带类型的 device alloc（如 f32 tile buffer）直接改到
  // workgroup 内存。后续“打包到单一 i8 workgroup 存储”的逻辑只假设
  // structured.promote 生成的 i8 workgroup alloc；若额外引入 typed alloc，
  // 可能破坏打包（layout/rank 不匹配）并明显增加动态 shared 内存占用。

  OpBuilder b(alloc);
  auto alloca = b.create<memref::AllocaOp>(alloc.getLoc(), memrefType,
                                           alloc.getDynamicSizes(),
                                           alloc.getSymbolOperands(),
                                           alloc.getAlignmentAttr());
  alloc.getResult().replaceAllUsesWith(alloca.getMemref());
  alloc.erase();
}

static bool isWorkgroupMemory(MemRefType memrefType) {
  // 标准 MLIR GPU dialect 形式。
  if (gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memrefType))
    return true;
  // 工程上常见的变体：为打通 vector->LLVM lowering，
  // 也接受 NVVM shared 地址空间整数值 3。
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
    return intAttr.getInt() == 3;
  return false;
}

static bool isPowerOfTwo(int64_t x) {
  return x > 0 && (x & (x - 1)) == 0;
}

static bool canApplyWorkgroupXorSwizzle(MemRefType memrefType, int64_t swizzle) {
  if (swizzle <= 1)
    return false;
  if (!isPowerOfTwo(swizzle))
    return false;
  if (!memrefType || !isWorkgroupMemory(memrefType))
    return false;
  if (!memrefType.hasStaticShape())
    return false;
  int64_t rank = memrefType.getRank();
  if (rank < 2)
    return false;
  int64_t last = memrefType.getShape()[rank - 1];
  if (last <= 0)
    return false;
  if (!isPowerOfTwo(last))
    return false;
  return swizzle <= last;
}

static Value buildWorkgroupXorSwizzledLastIndex(OpBuilder &b, Location loc,
                                                Value rowIdx, Value colIdx,
                                                int64_t swizzle) {
  if (!rowIdx || !colIdx || swizzle <= 1)
    return colIdx;

  // 计算：col' = col ^ (row & (重排-1))
  // 为安全起见先用 i64 计算，再转回 index。
  Value rowI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), rowIdx);
  Value colI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), colIdx);
  Value mask = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(swizzle - 1));
  Value lane = b.create<arith::AndIOp>(loc, rowI64, mask);
  Value swz = b.create<arith::XOrIOp>(loc, colI64, lane);
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), swz);
}

static void applyWorkgroupXorSwizzle(gpu::LaunchOp launch, int64_t swizzle) {
  if (!launch || swizzle <= 1)
    return;

  static constexpr StringLiteral kSwizzledAttrName =
      "welder.workgroup_swizzled";

  auto maybeRewriteIndices = [&](Value memref, SmallVectorImpl<Value> &indices,
                                 OpBuilder &b, Location loc) -> bool {
    if (!memref || indices.size() < 2)
      return false;
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || !canApplyWorkgroupXorSwizzle(memrefType, swizzle))
      return false;
    int64_t rank = memrefType.getRank();
    if (static_cast<int64_t>(indices.size()) != rank)
      return false;
    Value row = indices[rank - 2];
    Value col = indices[rank - 1];
    indices[rank - 1] =
        buildWorkgroupXorSwizzledLastIndex(b, loc, row, col, swizzle);
    return true;
  };

  // 通过克隆 op 并替换索引来原地改写，
  // 保证在使用重排 shared 布局做 bank-conflict 缓解时语义仍正确。
  launch.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (load->hasAttr(kSwizzledAttrName))
        return;
      OpBuilder b(load);
      SmallVector<Value, 4> idx(load.getIndices().begin(),
                                load.getIndices().end());
      if (!maybeRewriteIndices(load.getMemref(), idx, b, load.getLoc()))
        return;
      auto repl = b.create<memref::LoadOp>(load.getLoc(), load.getMemref(), idx);
      repl->setAttrs(load->getAttrs());
      repl->setAttr(kSwizzledAttrName, UnitAttr::get(load.getContext()));
      load.replaceAllUsesWith(repl.getResult());
      load.erase();
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (store->hasAttr(kSwizzledAttrName))
        return;
      OpBuilder b(store);
      SmallVector<Value, 4> idx(store.getIndices().begin(),
                                store.getIndices().end());
      if (!maybeRewriteIndices(store.getMemref(), idx, b, store.getLoc()))
        return;
      auto repl = b.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                            store.getMemref(), idx);
      repl->setAttrs(store->getAttrs());
      repl->setAttr(kSwizzledAttrName, UnitAttr::get(store.getContext()));
      store.erase();
      return;
    }
    if (auto tr = dyn_cast<vector::TransferReadOp>(op)) {
      if (tr->hasAttr(kSwizzledAttrName))
        return;
      OpBuilder b(tr);
      SmallVector<Value, 4> idx(tr.getIndices().begin(), tr.getIndices().end());
      if (!maybeRewriteIndices(tr.getBase(), idx, b, tr.getLoc()))
        return;
      auto repl = b.create<vector::TransferReadOp>(
          tr.getLoc(), tr.getVectorType(), tr.getBase(), idx,
          tr.getPermutationMapAttr(), tr.getPadding(), tr.getMask(),
          tr.getInBoundsAttr());
      repl->setAttrs(tr->getAttrs());
      repl->setAttr(kSwizzledAttrName, UnitAttr::get(tr.getContext()));
      tr.replaceAllUsesWith(repl.getResult());
      tr.erase();
      return;
    }
    if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      if (tw->hasAttr(kSwizzledAttrName))
        return;
      OpBuilder b(tw);
      SmallVector<Value, 4> idx(tw.getIndices().begin(), tw.getIndices().end());
      if (!maybeRewriteIndices(tw.getBase(), idx, b, tw.getLoc()))
        return;
      auto repl = b.create<vector::TransferWriteOp>(
          tw.getLoc(), tw.getVector(), tw.getBase(), idx,
          tw.getPermutationMapAttr(), tw.getMask(), tw.getInBoundsAttr());
      repl->setAttrs(tw->getAttrs());
      repl->setAttr(kSwizzledAttrName, UnitAttr::get(tw.getContext()));
      tw.erase();
      return;
    }
    if (auto ac = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op)) {
      if (ac->hasAttr(kSwizzledAttrName))
        return;
      OpBuilder b(ac);
      SmallVector<Value, 4> dstIdx(ac.getDstIndices().begin(),
                                  ac.getDstIndices().end());
      // Only 重排 the destination (shared) layout.
      if (!maybeRewriteIndices(ac.getDst(), dstIdx, b, ac.getLoc()))
        return;
      auto newCopy = b.create<nvgpu::DeviceAsyncCopyOp>(
          ac.getLoc(), ac.getResult().getType(), ac.getDst(), dstIdx, ac.getSrc(),
          ac.getSrcIndices(), ac.getDstElementsAttr(), ac.getSrcElements(),
          ac.getBypassL1Attr());
      newCopy->setAttrs(ac->getAttrs());
      newCopy->setAttr(kSwizzledAttrName, UnitAttr::get(ac.getContext()));
      ac.replaceAllUsesWith(newCopy.getResult());
      ac.erase();
      return;
    }
  });
}

static bool isPowerOfTwoI64(int64_t v) {
  if (v <= 0)
    return false;
  return (v & (v - 1)) == 0;
}

static void applyBlockRasterizeXor(gpu::LaunchOp launch, int64_t swizzle) {
  if (!launch || swizzle <= 1)
    return;
  if (!isPowerOfTwoI64(swizzle))
    return;

  // 仅在 gridX 为已知 2 次幂且 >= 重排因子时应用；
  // 否则 XOR 映射不保证双射，可能改变程序语义。
  auto gridXOpt = getConstIndexValue(launch.getGridSizeX());
  if (!gridXOpt)
    return;
  int64_t gridX = *gridXOpt;
  if (!isPowerOfTwoI64(gridX) || swizzle > gridX)
    return;

  static constexpr StringLiteral kRasterizedAttrName =
      "welder.block_rasterized";

  // 在 launch body 中找到一组 block_id.x 和 block_id.y。
  gpu::BlockIdOp bxOp;
  gpu::BlockIdOp byOp;
  launch.walk([&](gpu::BlockIdOp bid) {
    if (bid->hasAttr(kRasterizedAttrName))
      return;
    if (!bxOp && bid.getDimension() == gpu::Dimension::x)
      bxOp = bid;
    if (!byOp && bid.getDimension() == gpu::Dimension::y)
      byOp = bid;
  });
  if (!bxOp || !byOp)
    return;

  Location loc = bxOp.getLoc();
  Operation *insertAfter = bxOp.getOperation();
  if (insertAfter->isBeforeInBlock(byOp.getOperation()))
    insertAfter = byOp.getOperation();
  OpBuilder b(insertAfter);
  b.setInsertionPointAfter(insertAfter);
  Value bx = bxOp.getResult();
  Value by = byOp.getResult();

  // Compute: bx' = bx ^ (by & (重排-1))
  Value byI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), by);
  Value bxI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), bx);
  Value mask = b.create<arith::ConstantOp>(
      loc, b.getI64Type(), b.getI64IntegerAttr(swizzle - 1));
  Value lane = b.create<arith::AndIOp>(loc, byI64, mask);
  Value swz = b.create<arith::XOrIOp>(loc, bxI64, lane);
  Value bxSwz = b.create<arith::IndexCastOp>(loc, b.getIndexType(), swz);

  // 替换 bx 的使用，但保留重排计算读取原始 bx，
  // 避免引入自环依赖或支配关系违规。
  llvm::SmallPtrSet<Operation *, 8> created;
  created.insert(byI64.getDefiningOp());
  created.insert(bxI64.getDefiningOp());
  created.insert(mask.getDefiningOp());
  created.insert(lane.getDefiningOp());
  created.insert(swz.getDefiningOp());
  created.insert(bxSwz.getDefiningOp());
  bx.replaceUsesWithIf(bxSwz, [&](OpOperand &use) {
    return !created.contains(use.getOwner());
  });
  bxOp->setAttr(kRasterizedAttrName, UnitAttr::get(bxOp.getContext()));
}

enum class RasterizeMode { None = 0, Row = 1, Column = 2 };

static Value buildRasterization2DRow(OpBuilder &b, Location loc, Value idx,
                                     int64_t rowSize, int64_t colSize,
                                     int64_t panelWidth) {
  // Direct MLIR translation of Welder's header.py::光栅化2DRow.
  Value row = b.create<arith::ConstantIndexOp>(loc, rowSize);
  Value col = b.create<arith::ConstantIndexOp>(loc, colSize);
  Value pw = b.create<arith::ConstantIndexOp>(loc, panelWidth);
  Value blockSize = b.create<arith::MulIOp>(loc, row, col);
  Value panelSize = b.create<arith::MulIOp>(loc, pw, col);
  Value blockOffset = b.create<arith::RemSIOp>(loc, idx, blockSize);
  Value blockIdx = b.create<arith::DivSIOp>(loc, idx, blockSize);
  Value panelOffset = b.create<arith::RemSIOp>(loc, blockOffset, panelSize);
  Value panelIdx = b.create<arith::DivSIOp>(loc, blockOffset, panelSize);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value totalPanel = b.create<arith::DivSIOp>(
      loc,
      b.create<arith::AddIOp>(loc, blockSize,
                              b.create<arith::SubIOp>(loc, panelSize, one)),
      panelSize);
  Value predNotLast = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt,
      b.create<arith::AddIOp>(loc, panelIdx, one), totalPanel);
  Value remBlocks = b.create<arith::SubIOp>(
      loc, blockSize, b.create<arith::MulIOp>(loc, panelIdx, panelSize));
  Value strideLast = b.create<arith::DivSIOp>(loc, remBlocks, col);
  Value stride = b.create<arith::SelectOp>(loc, predNotLast, pw, strideLast);

  Value panelOdd = b.create<arith::AndIOp>(loc, panelIdx, one);
  Value predOdd = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, panelOdd,
      b.create<arith::ConstantIndexOp>(loc, 0));

  Value panelOffsetDivStride = b.create<arith::DivSIOp>(loc, panelOffset, stride);
  Value colMinus1 = b.create<arith::SubIOp>(loc, col, one);
  Value colFlip = b.create<arith::SubIOp>(loc, colMinus1, panelOffsetDivStride);
  Value colIdx = b.create<arith::SelectOp>(loc, predOdd, colFlip,
                                           panelOffsetDivStride);

  Value rowIdxInPanel = b.create<arith::RemSIOp>(loc, panelOffset, stride);
  Value rowBase = b.create<arith::MulIOp>(loc, panelIdx, pw);
  Value rowIdx = b.create<arith::AddIOp>(loc, rowIdxInPanel, rowBase);

  Value lin = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, rowIdx, col), colIdx);
  return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, blockIdx, blockSize),
                                 lin);
}

static Value buildRasterization2DColumn(OpBuilder &b, Location loc, Value idx,
                                        int64_t rowSize, int64_t colSize,
                                        int64_t panelWidth) {
  // Direct MLIR translation of Welder's header.py::光栅化2DColumn.
  Value row = b.create<arith::ConstantIndexOp>(loc, rowSize);
  Value col = b.create<arith::ConstantIndexOp>(loc, colSize);
  Value pw = b.create<arith::ConstantIndexOp>(loc, panelWidth);
  Value blockSize = b.create<arith::MulIOp>(loc, row, col);
  Value panelSize = b.create<arith::MulIOp>(loc, pw, row);
  Value blockOffset = b.create<arith::RemSIOp>(loc, idx, blockSize);
  Value blockIdx = b.create<arith::DivSIOp>(loc, idx, blockSize);
  Value panelOffset = b.create<arith::RemSIOp>(loc, blockOffset, panelSize);
  Value panelIdx = b.create<arith::DivSIOp>(loc, blockOffset, panelSize);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value totalPanel = b.create<arith::DivSIOp>(
      loc,
      b.create<arith::AddIOp>(loc, blockSize,
                              b.create<arith::SubIOp>(loc, panelSize, one)),
      panelSize);
  Value predNotLast = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt,
      b.create<arith::AddIOp>(loc, panelIdx, one), totalPanel);
  Value remBlocks = b.create<arith::SubIOp>(
      loc, blockSize, b.create<arith::MulIOp>(loc, panelIdx, panelSize));
  Value strideLast = b.create<arith::DivSIOp>(loc, remBlocks, row);
  Value stride = b.create<arith::SelectOp>(loc, predNotLast, pw, strideLast);

  Value panelOdd = b.create<arith::AndIOp>(loc, panelIdx, one);
  Value predOdd = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, panelOdd,
      b.create<arith::ConstantIndexOp>(loc, 0));

  Value panelOffsetDivStride = b.create<arith::DivSIOp>(loc, panelOffset, stride);
  Value rowMinus1 = b.create<arith::SubIOp>(loc, row, one);
  Value rowFlip = b.create<arith::SubIOp>(loc, rowMinus1, panelOffsetDivStride);
  Value rowIdx = b.create<arith::SelectOp>(loc, predOdd, rowFlip,
                                           panelOffsetDivStride);

  Value colIdxInPanel = b.create<arith::RemSIOp>(loc, panelOffset, stride);
  Value colBase = b.create<arith::MulIOp>(loc, panelIdx, pw);
  Value colIdx = b.create<arith::AddIOp>(loc, colIdxInPanel, colBase);

  Value lin = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, rowIdx, col), colIdx);
  return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, blockIdx, blockSize),
                                 lin);
}

static void applyBlockRasterization2D(gpu::LaunchOp launch, RasterizeMode mode,
                                      int64_t panelWidth) {
  if (!launch || mode == RasterizeMode::None)
    return;
  if (panelWidth <= 0)
    return;

  auto gx = getConstIndexValue(launch.getGridSizeX());
  auto gy = getConstIndexValue(launch.getGridSizeY());
  if (!gx || !gy)
    return;
  int64_t colSize = *gx;
  int64_t rowSize = *gy;
  if (rowSize <= 0 || colSize <= 0)
    return;

  static constexpr StringLiteral kRasterizedAttrName =
      "welder.block_rasterized_2d";

  if (launch->hasAttr(kRasterizedAttrName))
    return;

  Block &entry = launch.getBody().front();
  OpBuilder b(launch.getContext());
  b.setInsertionPointToStart(&entry);
  Location loc = launch.getLoc();

  auto bxNew = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  auto byNew = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);

  Value bx = bxNew.getResult();
  Value by = byNew.getResult();
  Value gridX = b.create<arith::ConstantIndexOp>(loc, colSize);

  // 将当前 2D block id 展平为线性 id。
  Value bid =
      b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, by, gridX), bx);

  Value bid2 = (mode == RasterizeMode::Row)
                   ? buildRasterization2DRow(b, loc, bid, rowSize, colSize,
                                             panelWidth)
                   : buildRasterization2DColumn(b, loc, bid, rowSize, colSize,
                                                panelWidth);

  // 再解码回 (by, bx)。
  Value bx2 = b.create<arith::RemSIOp>(loc, bid2, gridX);
  Value by2 = b.create<arith::DivSIOp>(loc, bid2, gridX);

  SmallVector<gpu::BlockIdOp, 8> oldBx;
  SmallVector<gpu::BlockIdOp, 8> oldBy;
  launch.walk([&](gpu::BlockIdOp bidOp) {
    if (bidOp.getDimension() == gpu::Dimension::x)
      oldBx.push_back(bidOp);
    else if (bidOp.getDimension() == gpu::Dimension::y)
      oldBy.push_back(bidOp);
  });

  for (gpu::BlockIdOp op : oldBx) {
    if (op == bxNew)
      continue;
    op.getResult().replaceAllUsesWith(bx2);
  }
  for (gpu::BlockIdOp op : oldBy) {
    if (op == byNew)
      continue;
    op.getResult().replaceAllUsesWith(by2);
  }
  for (gpu::BlockIdOp op : oldBx) {
    if (op != bxNew && op->use_empty())
      op.erase();
  }
  for (gpu::BlockIdOp op : oldBy) {
    if (op != byNew && op->use_empty())
      op.erase();
  }

  launch->setAttr(kRasterizedAttrName, UnitAttr::get(launch.getContext()));
}

struct WorkgroupAllocToLaunchWorkgroupPass
    : public PassWrapper<WorkgroupAllocToLaunchWorkgroupPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      WorkgroupAllocToLaunchWorkgroupPass)

  Option<int64_t> workgroupPadLastDim{
      * this, "workgroup-pad-last-dim",
      llvm::cl::desc("Pad the last dimension (in elements) of shared/workgroup "
                     "views by N to mitigate shared-memory bank conflicts; "
                     "only applied when the buffer is immediately subviewed "
                     "back to the original shape"),
      llvm::cl::init(0)};
  Option<bool> workgroupPadLastDimMatmulOnly{
      * this, "workgroup-pad-last-dim-matmul-only",
      llvm::cl::desc("When padding workgroup views, only apply padding to "
                     "workgroup buffers that feed linalg.matmul inputs (A/B). "
                     "This more closely matches Welder's stride-map offset "
                     "behavior for matmul schedules."),
      llvm::cl::init(false)};

  Option<int64_t> workgroupMultiBufferDepth{
      * this, "workgroup-multibuffer-depth",
      llvm::cl::desc(
          "Enable multi-buffering for workgroup buffers by allocating "
          "DEPTH copies of the underlying i8 storage and rewriting the "
          "memref.view byte_shift to select a stage based on the nearest "
          "enclosing scf.for induction variable. Intended to be used together "
          "with transform.nvgpu.pipeline_shared_memory_copies."),
      llvm::cl::init(1)};

  Option<int64_t> workgroupSwizzleXor{
      *this, "workgroup-重排-xor",
      llvm::cl::desc(
          "Apply an XOR swizzle on the last dimension for eligible shared/workgroup "
          "memref accesses: col' = col ^ (row & (swizzle-1)). Only applied to "
          "static-shape workgroup memrefs with rank>=2 and power-of-two last dim."),
      llvm::cl::init(0)};

  Option<int64_t> blockRasterizeXor{
      * this, "block-rasterize-xor",
      llvm::cl::desc(
          "Paper-aligned (approx): apply an XOR-based block rasterization by "
          "rewriting uses of gpu.block_id.x: bx' = bx ^ (by & (swizzle-1)). "
          "Only applied when grid_size_x is a known power-of-two >= swizzle."),
      llvm::cl::init(0)};

  Option<int> blockRasterizeMode{
      * this, "block-rasterize-mode",
      llvm::cl::desc("Paper/Welder parity: 2D rasterization mode "
                     "(0=off, 1=row, 2=column)."),
      llvm::cl::init(0)};

  Option<int64_t> blockRasterizePanelWidth{
      * this, "block-rasterize-panel-width",
      llvm::cl::desc("Panel width for 2D rasterization (typical: 1..16)."),
      llvm::cl::init(0)};

  WorkgroupAllocToLaunchWorkgroupPass() = default;
  WorkgroupAllocToLaunchWorkgroupPass(
      const WorkgroupAllocToLaunchWorkgroupPass &pass)
      : PassWrapper(pass),
        workgroupPadLastDim(
            * this, "workgroup-pad-last-dim",
            llvm::cl::desc("Pad the last dimension (in elements) of "
                           "shared/workgroup views by N to mitigate "
                           "shared-memory bank conflicts; only applied when "
                           "the buffer is immediately subviewed back to the "
                           "original shape"),
            llvm::cl::init(0)) {
		    workgroupPadLastDim = pass.workgroupPadLastDim;
		    workgroupPadLastDimMatmulOnly = pass.workgroupPadLastDimMatmulOnly;
		    workgroupMultiBufferDepth = pass.workgroupMultiBufferDepth;
		    workgroupSwizzleXor = pass.workgroupSwizzleXor;
		    blockRasterizeXor = pass.blockRasterizeXor;
            blockRasterizeMode = pass.blockRasterizeMode;
            blockRasterizePanelWidth = pass.blockRasterizePanelWidth;
		  }

  StringRef getArgument() const final {
    return "workgroup-alloc-to-launch-workgroup";
  }

  StringRef getDescription() const final {
    return "Make structured.promote(workgroup) + GPU lowering runnable by "
           "converting workgroup alloc/dealloc to gpu.launch workgroup "
           "attributions, expanding memref.copy in gpu.launch, and rewriting "
           "allocations to GPU-friendly forms.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    nvgpu::NVGPUDialect, scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Host 侧：将 memref.alloc 替换为 gpu.alloc host_shared，
    // 确保传入 gpu.launch 的缓冲区对 GPU 可访问。
    SmallVector<memref::AllocOp> hostAllocsToConvert;
    module.walk([&](memref::AllocOp alloc) {
      if (!alloc->getParentOfType<gpu::LaunchOp>())
        hostAllocsToConvert.push_back(alloc);
    });
    for (memref::AllocOp alloc : hostAllocsToConvert)
      convertHostAllocToGpuHostShared(alloc);

		    module.walk([&](gpu::LaunchOp launch) {
      // Device 侧：避免由 memref.alloc 触发设备 malloc。
      SmallVector<memref::AllocOp> deviceAllocsToAlloca;
      launch.walk([&](memref::AllocOp alloc) { deviceAllocsToAlloca.push_back(alloc); });
      for (memref::AllocOp alloc : deviceAllocsToAlloca)
        promoteDeviceAllocToAlloca(alloc);

      SmallVector<memref::AllocOp> allocsToConvert;
	      launch.walk([&](memref::AllocOp alloc) {
	        auto memrefType = dyn_cast<MemRefType>(alloc.getType());
	        if (!memrefType)
	          return;
        if (!isWorkgroupMemory(memrefType))
          return;
        // 当前仅处理静态形状缓冲区。
        if (!memrefType.hasStaticShape())
          return;
        if (!alloc.getDynamicSizes().empty())
          return;
        allocsToConvert.push_back(alloc);
	      });

	      if (!allocsToConvert.empty()) {
	        DenseMap<const void *, int64_t> padByAlloc;
          DenseSet<const void *> multiBufferByAlloc;

	        auto getPadFromMatmulAttrsOrDefault =
	            [&](linalg::MatmulOp mm) -> std::pair<int64_t, bool> {
	          int64_t pad = workgroupPadLastDim;
	          bool matmulOnly = workgroupPadLastDimMatmulOnly;
	          if (auto padAttr = mm->getAttrOfType<IntegerAttr>(
	                  "welder.workgroup_pad_last_dim")) {
	            pad = padAttr.getInt();
	          }
	          if (auto onlyAttr = mm->getAttrOfType<BoolAttr>(
	                  "welder.workgroup_pad_last_dim_matmul_only")) {
	            matmulOnly = onlyAttr.getValue();
	          }
	          return {std::max<int64_t>(0, pad), matmulOnly};
	        };

	        // 默认行为：若非 matmul-only，则给每个 workgroup 缓冲区做 pad。
	        if (workgroupPadLastDim > 0 && !workgroupPadLastDimMatmulOnly) {
	          for (memref::AllocOp a : allocsToConvert)
	            padByAlloc[a.getResult().getAsOpaquePointer()] = workgroupPadLastDim;
	        }

	        // 论文/Welder 对齐: matmul stride-map offset is per-matmul; if a matmul
	        // 若 matmul 显式携带 pad 属性，则将其应用到 A/B workgroup 缓冲区。
	        // Otherwise, fall back to the pass-level 默认.
	        //
	        // 注意：TensorCore 调度下，上游 transform 常在本 pass 前把
	        // linalg.matmul 改写为 nvgpu.mma.sync。此时不再有 matmul op 可作为
	        // pad 锚点。为保持预期 stride-map 行为，这里也会识别喂给
	        // nvgpu.mma.sync 操作数的 workgroup 缓冲区。
        launch.walk([&](linalg::MatmulOp mm) {
          if (!mm)
            return;
          if (mm.getNumDpsInputs() < 2)
            return;
          auto [pad, matmulOnly] = getPadFromMatmulAttrsOrDefault(mm);
          if (pad <= 0)
            return;
          if (!matmulOnly)
            return;

          Value a = stripViewLike(mm.getDpsInputOperand(0)->get());
          Value b = stripViewLike(mm.getDpsInputOperand(1)->get());
          Value c;
          if (mm.getNumDpsInits() > 0)
            c = stripViewLike(mm.getDpsInitOperand(0)->get());
          for (Value v : {a, b, c}) {
            if (!v)
              continue;
            auto mt = dyn_cast<MemRefType>(v.getType());
            if (!mt || !isWorkgroupMemory(mt))
              continue;
	            if (auto alloc = v.getDefiningOp<memref::AllocOp>()) {
	              const void *k = alloc.getResult().getAsOpaquePointer();
	              auto it = padByAlloc.find(k);
	              int64_t cur = (it == padByAlloc.end()) ? 0 : it->second;
	              padByAlloc[k] = std::max<int64_t>(cur, pad);
	            }
	          }
	        });

	        auto recordPadAllocFromMemref = [&](Value memref, int64_t pad) {
	          if (!memref)
	            return;
	          Value root = stripViewLike(memref);
	          auto mt = dyn_cast<MemRefType>(root.getType());
	          if (!mt || !isWorkgroupMemory(mt))
	            return;
	          if (auto alloc = root.getDefiningOp<memref::AllocOp>()) {
	            const void *k = alloc.getResult().getAsOpaquePointer();
	            auto it = padByAlloc.find(k);
	            int64_t cur = (it == padByAlloc.end()) ? 0 : it->second;
	            padByAlloc[k] = std::max<int64_t>(cur, std::max<int64_t>(0, pad));
	          }
	        };

	        auto recordPadAllocsFromValueSlice = [&](Value v, int64_t pad) {
	          if (!v || pad <= 0)
	            return;
	          llvm::DenseSet<const void *> seen;
	          SmallVector<Value, 16> work;
	          work.push_back(v);
	          int steps = 0;
	          while (!work.empty() && steps++ < 256) {
	            Value cur = work.pop_back_val();
	            const void *vk = cur.getAsOpaquePointer();
	            if (!seen.insert(vk).second)
	              continue;
	            Operation *def = cur.getDefiningOp();
	            if (!def)
	              continue;

	            if (auto load = dyn_cast<memref::LoadOp>(def)) {
	              recordPadAllocFromMemref(load.getMemref(), pad);
	              continue;
	            }
	            if (auto tr = dyn_cast<vector::TransferReadOp>(def)) {
	              recordPadAllocFromMemref(tr.getBase(), pad);
	              continue;
	            }
	            // 避免跨越控制流/region 边界；mma fragment 由本地 SSA 值构建。
	            if (def->getNumRegions() != 0)
	              continue;
	            for (Value opnd : def->getOperands())
	              work.push_back(opnd);
	          }
	        };

	        // 若请求 matmul-only pad 但 linalg.matmul 已被降级，
	        // 则回退为识别 nvgpu.mma.sync 使用的 A/B workgroup 缓冲区。
	        if (workgroupPadLastDim > 0 && workgroupPadLastDimMatmulOnly) {
	          launch.walk([&](nvgpu::MmaSyncOp mma) {
	            auto [pad, matmulOnly] =
	                std::pair<int64_t, bool>(workgroupPadLastDim,
	                                         workgroupPadLastDimMatmulOnly);
	            if (pad <= 0 || !matmulOnly)
	              return;
	            for (Value opnd : mma->getOperands())
	              recordPadAllocsFromValueSlice(opnd, pad);
	          });
	        }

          // Multi-buffer 策略：
          // 仅对将参与异步 shared copy（cp.async）且位于可软件流水化 scf.for
          // 内的缓冲区启用多缓冲。若对所有 workgroup 缓冲区统一套 DEPTH 倍
          // 多缓冲，很容易超过硬件 shared 内存上限（尤其 kernel 还缓存了
          // matmul->softmax 这类大中间 tile 时）。
          // 因此这里只对（传递意义上）作为 nvgpu.device_async_copy 目标的
          // 缓冲区启用 multi-buffer。
          if (workgroupMultiBufferDepth > 1) {
            launch.walk([&](nvgpu::DeviceAsyncCopyOp asyncCopy) {
              Value dstRoot = stripViewLike(asyncCopy.getDst());
              auto mt = dyn_cast<MemRefType>(dstRoot.getType());
              if (!mt || !isWorkgroupMemory(mt))
                return;
              if (auto alloc = dstRoot.getDefiningOp<memref::AllocOp>()) {
                multiBufferByAlloc.insert(
                    alloc.getResult().getAsOpaquePointer());
              }
            });
          }

		        // 论文/Welder 对齐：将所有 workgroup(shared) 缓冲区打包到单一
		        // gpu.launch workgroup attribution，并基于简单的“最后使用点”扫描
		        // 做存储复用（best-fit），近似参考实现里的 BestFit 分配器行为。
	        DenseMap<Operation *, int64_t> opOrder;
	        int64_t nextId = 0;
	        launch.walk([&](Operation *op) { opOrder[op] = nextId++; });

	        auto computeTransitiveLastUse = [&](Value root) -> int64_t {
	          DenseSet<const void *> seen;
	          SmallVector<Value, 16> work;
	          work.push_back(root);
	          int64_t maxUse = -1;
	          while (!work.empty()) {
	            Value v = work.pop_back_val();
	            const void *k = v.getAsOpaquePointer();
	            if (!seen.insert(k).second)
	              continue;
	            for (OpOperand &use : v.getUses()) {
	              Operation *uop = use.getOwner();
	              if (!launch->isAncestor(uop))
	                continue;
	              auto it = opOrder.find(uop);
	              if (it != opOrder.end())
	                maxUse = std::max<int64_t>(maxUse, it->second);
	              for (Value r : uop->getResults())
	                work.push_back(r);
	            }
	          }
	          return maxUse;
	        };

        struct Plan {
          memref::AllocOp alloc;
          MemRefType oldI8Type;
          int64_t newI8Bytes = 0;
          int64_t sliceI8Bytes = 0;
          int64_t multiDepth = 1;
          int64_t padLastDim = 0;
          memref::ViewOp viewToRewrite;
          SmallVector<int64_t, 4> viewStaticSizes;
	          int64_t elemBytes = -1;
	          int64_t order = 0;
	          int64_t lastUse = 0;
	          int64_t offset = 0;
	          Value allocReplacement;
	        };

        SmallVector<Plan, 8> plans;
        plans.reserve(allocsToConvert.size());

        // 仅打包 i8 memref 且 layout/memory space 兼容的缓冲区。
        // 其余不兼容缓冲区（layout/addrspace 不匹配）回退为独立
        // gpu.launch workgroup attribution（优先保证稳健性）。
        MemRefType refType;
        SmallVector<memref::AllocOp, 8> unpackedAllocs;
        bool anyPadPacked = false;

	        for (memref::AllocOp alloc : allocsToConvert) {
	          auto oldI8Type = cast<MemRefType>(alloc.getType());
          if (!refType)
            refType = oldI8Type;
          if (oldI8Type.getLayout() != refType.getLayout() ||
              oldI8Type.getMemorySpace() != refType.getMemorySpace()) {
            unpackedAllocs.push_back(alloc);
            continue;
          }

	          Plan p;
	          p.alloc = alloc;
	          p.oldI8Type = oldI8Type;
	          p.multiDepth = 1;
	          if (workgroupMultiBufferDepth > 1) {
	            const void *k = alloc.getResult().getAsOpaquePointer();
	            if (multiBufferByAlloc.contains(k))
	              p.multiDepth =
	                  std::max<int64_t>(1, workgroupMultiBufferDepth);
	          }
	          p.newI8Bytes = oldI8Type.getNumElements();

	          int64_t padForThis = 0;
	          if (auto it = padByAlloc.find(alloc.getResult().getAsOpaquePointer());
	              it != padByAlloc.end())
	            padForThis = std::max<int64_t>(0, it->second);
	          p.padLastDim = padForThis;

	          if (padForThis > 0 || p.multiDepth > 1) {
	            SmallVector<memref::ViewOp, 4> userViews;
	            for (Operation *user : alloc.getResult().getUsers()) {
	              if (auto view = dyn_cast<memref::ViewOp>(user))
	                userViews.push_back(view);
	            }

	            bool ok = true;
	            if (padForThis > 0) {
	              if (userViews.empty())
	                ok = false;
	              else
	                p.viewToRewrite = userViews.front();
	            }
	            if (ok && padForThis == 0 && p.multiDepth > 1) {
	              if (userViews.empty())
	                ok = false;
	              else
	                p.viewToRewrite = userViews.front();
	            }

            if (ok && p.viewToRewrite) {
              auto viewTy =
                  dyn_cast<MemRefType>(p.viewToRewrite.getResult().getType());
              if (!viewTy || viewTy.getRank() < 2)
                ok = false;
              if (ok) {
                p.elemBytes = getElementByteWidth(viewTy.getElementType());
                if (p.elemBytes <= 0)
                  ok = false;
              }
              if (ok && padForThis > 0) {
                // Padding 依赖可确定的逻辑视图形状，优先要求静态形状
                //（这是 tiled matmul shared buffer 的常见情况）。
                if (!viewTy.hasStaticShape()) {
                  ok = false;
                } else {
                  // 要求所有 view 的元素类型与形状一致；
                  // 否则无法安全地给底层存储做 pad。
                  for (memref::ViewOp v : userViews) {
                    auto ty = dyn_cast<MemRefType>(v.getResult().getType());
                    if (!ty || ty != viewTy) {
                      ok = false;
                      break;
                    }
                  }
                }
                if (ok) {
                  p.viewStaticSizes.assign(viewTy.getShape().begin(),
                                           viewTy.getShape().end());
                  int64_t expected = p.elemBytes;
                  for (int64_t d : p.viewStaticSizes)
                    expected *= d;
                  if (expected != p.newI8Bytes) {
                    ok = false;
                  } else if (!p.viewStaticSizes.empty()) {
                    int64_t last = p.viewStaticSizes.back();
                    int64_t newLast = last + std::max<int64_t>(0, padForThis);
                    int64_t outer = 1;
                    for (size_t i = 0, e = p.viewStaticSizes.size() - 1; i < e; ++i)
                      outer *= p.viewStaticSizes[i];
                    p.newI8Bytes = outer * newLast * p.elemBytes;
                  } else {
                    ok = false;
                  }
                }
              }

              if (!ok) {
                p.viewToRewrite = memref::ViewOp();
                p.viewStaticSizes.clear();
                p.elemBytes = -1;
                // 若没有一致且静态的 view，就无法安全做 pad。
                p.padLastDim = 0;
                p.newI8Bytes = oldI8Type.getNumElements();
              }
            }
          }

          p.sliceI8Bytes = p.newI8Bytes;
          if (p.multiDepth > 1) {
            if (!p.viewToRewrite) {
              // 放宽策略：若 view 无法改写，则回退到 depth=1。
              p.multiDepth = 1;
            }
            if (p.sliceI8Bytes <= 0 ||
                p.sliceI8Bytes >
                    (std::numeric_limits<int64_t>::max() / p.multiDepth)) {
              alloc.emitError() << "workgroup-multibuffer overflow: sliceBytes="
                                << p.sliceI8Bytes << " depth=" << p.multiDepth;
              signalPassFailure();
              return;
            }
            if (p.multiDepth > 1)
              p.newI8Bytes = p.sliceI8Bytes * p.multiDepth;
          }

		          p.order = opOrder.lookup(alloc.getOperation());
		          p.lastUse = computeTransitiveLastUse(alloc.getResult());
		          if (p.lastUse < p.order)
		            p.lastUse = p.order;
		          anyPadPacked |= (p.padLastDim > 0);
		          plans.push_back(std::move(p));
		        }

	        llvm::sort(plans, [](const Plan &a, const Plan &b) {
	          return a.order < b.order;
	        });

        auto alignTo = [](int64_t v, int64_t a) -> int64_t {
          if (a <= 1)
            return v;
          int64_t r = v % a;
          return r == 0 ? v : (v + (a - r));
        };

        struct Block {
          int64_t off = 0;
          int64_t size = 0;
        };

        SmallVector<Block, 16> freeList;
        struct Active {
          int64_t last = 0;
          int64_t off = 0;
          int64_t size = 0;
        };
        SmallVector<Active, 16> active;

	        int64_t totalBytes = 0;
		          for (Plan &p : plans) {
		          int64_t curPos = p.order;
	          for (int i = 0; i < static_cast<int>(active.size());) {
	            if (active[i].last < curPos) {
	              freeList.push_back(Block{active[i].off, active[i].size});
	              active.erase(active.begin() + i);
              continue;
            }
            ++i;
          }

          int64_t need = alignTo(std::max<int64_t>(1, p.newI8Bytes), 32);

          int bestIdx = -1;
          int64_t bestSize = 0;
          for (int i = 0; i < static_cast<int>(freeList.size()); ++i) {
            if (freeList[i].size < need)
              continue;
            if (bestIdx < 0 || freeList[i].size < bestSize) {
              bestIdx = i;
              bestSize = freeList[i].size;
            }
          }

          if (bestIdx >= 0) {
            p.offset = freeList[bestIdx].off;
            int64_t remaining = freeList[bestIdx].size - need;
            if (remaining > 0) {
              freeList[bestIdx].off += need;
              freeList[bestIdx].size = remaining;
            } else {
              freeList.erase(freeList.begin() + bestIdx);
            }
          } else {
            p.offset = totalBytes;
            totalBytes += need;
          }

          active.push_back(Active{p.lastUse, p.offset, need});
        }

        if (totalBytes <= 0) {
          signalPassFailure();
          return;
        }

        // 一次性创建打包后的 workgroup attribution。
        OpBuilder b0(plans.front().alloc);
        auto packedType = MemRefType::get({totalBytes}, b0.getI8Type(),
                                          refType.getLayout(),
                                          refType.getMemorySpace());
        BlockArgument wgPacked =
            launch.addWorkgroupAttribution(packedType, plans.front().alloc.getLoc());

        // 通过重写各 alloc 的 memref.view 使用点来物化映射：
        // 统一改为引用 packed buffer + byte_shift 偏移。
        // 这里故意使用 memref.view（而非 memref.subview），因为静态 offset 的
        // subview 需要把 offset 编码进结果类型，可能与原 alloc 使用点类型不匹配。
        for (Plan &p : plans) {
          memref::AllocOp alloc = p.alloc;
          OpBuilder pb(alloc);
          Location ploc = alloc.getLoc();
          auto oldTy = p.oldI8Type;
          SmallVector<Value, 4> dynSizes;
          dynSizes.reserve(oldTy.getNumDynamicDims());
          for (int64_t i = 0, e = oldTy.getRank(); i < e; ++i) {
            if (oldTy.isDynamicDim(i))
              dynSizes.push_back(pb.create<memref::DimOp>(ploc, alloc, i));
          }
          Value byteShift = pb.create<arith::ConstantIndexOp>(ploc, p.offset);
          auto view =
              pb.create<memref::ViewOp>(ploc, oldTy, wgPacked, byteShift, dynSizes);
          Value allocReplacement = view.getResult();
          p.allocReplacement = allocReplacement;
          Value wg = wgPacked;

          SmallVector<memref::ViewOp, 4> viewsToRewrite;
          for (Operation *user : alloc.getResult().getUsers()) {
            if (auto view = dyn_cast<memref::ViewOp>(user))
              viewsToRewrite.push_back(view);
          }

		          for (memref::ViewOp view : viewsToRewrite) {
		            OpBuilder vb(view);
		            Location vloc = view.getLoc();

            Value byteShift = view.getByteShift();
            if (p.offset != 0) {
              Value baseShift = vb.create<arith::ConstantIndexOp>(vloc, p.offset);
              byteShift = vb.create<arith::AddIOp>(vloc, byteShift, baseShift);
            }
            if (p.multiDepth > 1) {
              auto forOp = view->getParentOfType<scf::ForOp>();
              if (!forOp) {
                // 放宽策略：若不在 scf.for 内，则禁用 multi-buffer。
                p.multiDepth = 1;
              } else {
                Value ivI64 = vb.create<arith::IndexCastOp>(
                    vloc, vb.getI64Type(), forOp.getInductionVar());
                Value lbI64 = vb.create<arith::IndexCastOp>(
                    vloc, vb.getI64Type(), forOp.getLowerBound());
                Value stepI64 = vb.create<arith::IndexCastOp>(
                    vloc, vb.getI64Type(), forOp.getStep());
                Value delta = vb.create<arith::SubIOp>(vloc, ivI64, lbI64);
                Value iter = vb.create<arith::DivUIOp>(vloc, delta, stepI64);
                Value depthI64 =
                    vb.create<arith::ConstantIntOp>(vloc, p.multiDepth, 64);
                Value stage = vb.create<arith::RemUIOp>(vloc, iter, depthI64);
                Value sliceBytesI64 =
                    vb.create<arith::ConstantIntOp>(vloc, p.sliceI8Bytes, 64);
                Value shiftI64 = vb.create<arith::MulIOp>(vloc, stage, sliceBytesI64);
                Value stageShift = vb.create<arith::IndexCastOp>(
                    vloc, vb.getIndexType(), shiftI64);
                byteShift = vb.create<arith::AddIOp>(vloc, byteShift, stageShift);
              }
            }

            int64_t padForThis = p.padLastDim;
            auto viewTy = dyn_cast<MemRefType>(view.getResult().getType());
            if (padForThis > 0 && viewTy && viewTy.hasStaticShape() &&
                viewTy.getRank() >= 2) {
              // 先构造带 pad 的 view 类型，再 subview 回原类型，
              // 以保持所有既有使用点的静态形状不变。
              SmallVector<int64_t, 4> paddedShape(viewTy.getShape().begin(),
                                                  viewTy.getShape().end());
              paddedShape.back() += std::max<int64_t>(0, padForThis);
              auto paddedTy =
                  MemRefType::get(paddedShape, viewTy.getElementType(),
                                  viewTy.getLayout(), viewTy.getMemorySpace());

              auto paddedView =
                  vb.create<memref::ViewOp>(vloc, paddedTy, wg, byteShift,
                                            /* sizes=*/ValueRange{});

              SmallVector<OpFoldResult, 4> offsets;
              SmallVector<OpFoldResult, 4> sizes;
              SmallVector<OpFoldResult, 4> strides;
              offsets.reserve(viewTy.getRank());
              sizes.reserve(viewTy.getRank());
              strides.reserve(viewTy.getRank());
              for (int64_t d : viewTy.getShape()) {
                offsets.push_back(vb.getIndexAttr(0));
                sizes.push_back(vb.getIndexAttr(d));
                strides.push_back(vb.getIndexAttr(1));
              }
              auto sliced = memref::SubViewOp::create(
                  vb, vloc, paddedView.getResult(), offsets, sizes, strides);
              view.getResult().replaceAllUsesWith(sliced.getResult());
            } else {
              auto newView = vb.create<memref::ViewOp>(
                  vloc, view.getType(), wg, byteShift, view.getSizes());
              view.getResult().replaceAllUsesWith(newView.getResult());
            }
            view.erase();
          }

          for (Operation *user :
               llvm::make_early_inc_range(alloc.getResult().getUsers())) {
            if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
              dealloc.erase();
              continue;
            }
            user->replaceUsesOfWith(alloc.getResult(), p.allocReplacement);
          }
          alloc.erase();
        }

        // 处理剩余未参与打包的 workgroup alloc。
        for (memref::AllocOp alloc : unpackedAllocs) {
          OpBuilder b(alloc);
          auto oldTy = cast<MemRefType>(alloc.getType());
          BlockArgument wg = launch.addWorkgroupAttribution(oldTy, alloc.getLoc());
          for (Operation *user :
               llvm::make_early_inc_range(alloc.getResult().getUsers())) {
            if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
              dealloc.erase();
              continue;
            }
            user->replaceUsesOfWith(alloc.getResult(), wg);
          }
          alloc.erase();
        }

        // 若某些 workgroup view 做过 pad，其推导 stride 会变化
        //（例如 [oldLast+pad, 1]）。此前基于未 pad 布局构造的 SubView
        // 结果类型可能已过期，需要按更新后的 memref 类型重建以保证可验证。
        if (anyPadPacked) {
          SmallVector<memref::SubViewOp, 32> subviewsToFix;
          launch.walk([&](memref::SubViewOp sv) {
            auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
            if (!srcTy || !isWorkgroupMemory(srcTy))
              return;
            subviewsToFix.push_back(sv);
          });
          for (memref::SubViewOp sv : subviewsToFix) {
            auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
            auto oldResTy = dyn_cast<MemRefType>(sv.getResult().getType());
            if (!srcTy || !oldResTy)
              continue;
            MemRefType newResTy = memref::SubViewOp::inferRankReducedResultType(
                oldResTy.getShape(), srcTy, sv.getMixedOffsets(),
                sv.getMixedSizes(), sv.getMixedStrides());
            if (newResTy == oldResTy)
              continue;
            OpBuilder b(sv);
            auto newSv = memref::SubViewOp::create(
                b, sv.getLoc(), newResTy, sv.getSource(), sv.getMixedOffsets(),
                sv.getMixedSizes(), sv.getMixedStrides());
            sv.getResult().replaceAllUsesWith(newSv.getResult());
            sv.erase();
          }
        }
      }

      SmallVector<memref::DeallocOp> deallocsToErase;
      launch.walk([&](memref::DeallocOp dealloc) {
        auto memrefType = dyn_cast<MemRefType>(dealloc.getMemref().getType());
        if (!memrefType)
          return;
        if (!isWorkgroupMemory(memrefType))
          return;
        deallocsToErase.push_back(dealloc);
      });

      for (memref::DeallocOp dealloc : deallocsToErase)
        dealloc.erase();

      // === Phase 4 (Vectorization) 的对齐提示 ===
      //
      // 在我们的 copy 向量化路径里，会出现 vector.transfer_read/write。
      // 即使源/目的地址“事实上”是 16B 对齐的，后端也经常保守地只给 alignment=4，
      // 从而把 float4 访问拆成 4 次标量 ld/st。
      //
      // 这里对 transfer 的 base memref 加一个 assume_alignment(16)，
      // 让后续 vector->LLVM 更容易给出 alignment=16，促使生成 ld/st 的 v4 形式。
      launch.walk([&](vector::TransferReadOp read) {
        Value base = read.getBase();
        auto memrefType = dyn_cast<MemRefType>(base.getType());
        if (!memrefType)
          return;
        // 只对 f32 的 transfer 做这个假设（我们当前实验的 float4 向量化目标）。
        if (!memrefType.getElementType().isF32())
          return;
        if (auto def = base.getDefiningOp<memref::AssumeAlignmentOp>())
          if (def.getAlignment() >= 16)
            return;
        OpBuilder b(read);
        auto assumed = b.create<memref::AssumeAlignmentOp>(
            read.getLoc(), base, b.getI32IntegerAttr(16));
        read->setOperand(0, assumed.getResult());
      });
      launch.walk([&](vector::TransferWriteOp write) {
        Value base = write.getBase();
        auto memrefType = dyn_cast<MemRefType>(base.getType());
        if (!memrefType)
          return;
        if (!memrefType.getElementType().isF32())
          return;
        if (auto def = base.getDefiningOp<memref::AssumeAlignmentOp>())
          if (def.getAlignment() >= 16)
            return;
        OpBuilder b(write);
        auto assumed = b.create<memref::AssumeAlignmentOp>(
            write.getLoc(), base, b.getI32IntegerAttr(16));
        // transfer_write 操作数顺序：(valueToStore, base, indices...)
        write->setOperand(1, assumed.getResult());
      });

      // Optional shared-memory 重排 rewrite (bank-conflict mitigation).
      if (workgroupSwizzleXor > 1)
        applyWorkgroupXorSwizzle(launch, workgroupSwizzleXor);
      // Optional block 光栅化 (重排 block-id mapping).
      if (blockRasterizeXor > 1)
        applyBlockRasterizeXor(launch, blockRasterizeXor);
      // Optional 论文对齐 2D block 光栅化 (Row/Column).
      if (blockRasterizeMode != 0 && blockRasterizePanelWidth > 0) {
        RasterizeMode m = RasterizeMode::None;
        if (blockRasterizeMode == 1)
          m = RasterizeMode::Row;
        else if (blockRasterizeMode == 2)
          m = RasterizeMode::Column;
        applyBlockRasterization2D(launch, m, blockRasterizePanelWidth);
      }

      // 将 gpu.launch 内的 memref.copy 展开成显式循环，
      // 避免设备模块依赖 `memrefCopy` 运行时符号。
      SmallVector<memref::CopyOp> copies;
      launch.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
      SmallVector<memref::CopyOp> copiesToExpand;
      copiesToExpand.reserve(copies.size());
      for (memref::CopyOp copy : copies) {
        if (rewritePrivateBlockTileToThreadTile(copy))
          continue;
        if (eraseTrivialSubviewCopy(copy))
          continue;
        copiesToExpand.push_back(copy);
      }
      for (memref::CopyOp copy : copiesToExpand)
        (void)expandStaticMemrefCopy(copy);

      // A small 时延-hiding tweak: remove redundant barriers between two
      // independent shared-memory copy 阶段 (common in promoted matmul).
      launch.walk([&](Operation *nested) {
        for (Region &r : nested->getRegions())
          for (Block &b : r)
            mergeRedundantCopyBarriersInBlock(b);
      });
    });
  }
};

} // 命名空间

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "WorkgroupAllocToLaunchWorkgroup",
          LLVM_VERSION_STRING,
          []() {
            PassRegistration<WorkgroupAllocToLaunchWorkgroupPass>();
            PassRegistration<LLVMVector4AlignPass>();
          }};
}

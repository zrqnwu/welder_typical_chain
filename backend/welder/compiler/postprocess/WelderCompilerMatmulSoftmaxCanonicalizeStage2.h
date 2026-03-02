#pragma once

static bool hasAnyWritesToMemrefBaseInLaunch(gpu::LaunchOp launch, Value base) {
  if (!launch || !base)
    return false;
  Value baseRoot = stripToTypedMemrefBase(base);
  if (!baseRoot)
    return false;

  bool written = false;
  launch.walk([&](Operation *op) {
    if (written)
      return;

    auto isTargetBase = [&](Value memref) -> bool {
      Value root = stripToTypedMemrefBase(memref);
      return root && root == baseRoot;
    };

    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (isTargetBase(store.getMemref()))
        written = true;
      return;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (isTargetBase(copy.getTarget()))
        written = true;
      return;
    }
    if (auto tw = dyn_cast<vector::TransferWriteOp>(op)) {
      if (isTargetBase(tw.getBase()))
        written = true;
      return;
    }
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      for (Value out : linalgOp.getDpsInits()) {
        if (isTargetBase(out)) {
          written = true;
          return;
        }
      }
      return;
    }
  });
  return written;
}

static bool tryReplaceInitCopyFromFilledMemref(memref::CopyOp copy,
                                              gpu::LaunchOp launch,
                                              UnitAttr keepBarrier) {
  if (!copy || !launch)
    return false;
  auto dstTy = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!dstTy || !dstTy.hasStaticShape())
    return false;
  if (!isWorkgroupMemoryType(dstTy))
    return false;

  Value srcBase = stripToTypedMemrefBase(copy.getSource());
  if (!srcBase)
    return false;

  // 若源 base 在 launch 内被写入，则它不是“统一 fill”来源。
  if (hasAnyWritesToMemrefBaseInLaunch(launch, srcBase))
    return false;

  // 查找唯一一个初始化 srcBase 的 host 侧 linalg.fill。
  linalg::FillOp fillOp;
  Value fillVal;
  for (OpOperand &use : llvm::make_early_inc_range(srcBase.getUses())) {
    Operation *owner = use.getOwner();
    if (!owner)
      continue;
    if (isUsedInLaunch(owner, launch))
      continue;

    if (auto fill = dyn_cast<linalg::FillOp>(owner)) {
      if (fill.getNumDpsInits() == 1 &&
          fill.getDpsInitOperand(0)->get() == srcBase) {
        if (fillOp)
          return false; // ambiguous
        fillOp = fill;
        if (fill.getNumDpsInputs() == 1)
          fillVal = fill.getDpsInputOperand(0)->get();
        continue;
      }
    }

    // 忽略安全的良性使用。
    if (isa<memref::DeallocOp, memref::SubViewOp, memref::ViewOp, memref::CastOp,
            memref::ReinterpretCastOp, memref::AssumeAlignmentOp>(owner)) {
      continue;
    }

    // 出现未知逃逸/写入使用，直接放弃。
    return false;
  }
  if (!fillOp || !fillVal)
    return false;

  // 改写为线程协作 fill 到 workgroup 目标，并保留 barrier。
  OpBuilder b(copy);
  Location loc = copy.getLoc();

  Value tidx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value tidy = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value tidz = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
  Value bdx = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
  Value bdy = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
  Value bdz = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::z);

  Value tidXY = b.create<arith::MulIOp>(loc, tidy, bdx);
  Value bdxBdy = b.create<arith::MulIOp>(loc, bdx, bdy);
  Value tidXYZ = b.create<arith::MulIOp>(loc, tidz, bdxBdy);
  Value tid = b.create<arith::AddIOp>(
      loc, b.create<arith::AddIOp>(loc, tidx, tidXY), tidXYZ);
  Value totalThreads = b.create<arith::MulIOp>(loc, bdxBdy, bdz);

  const int64_t rank = dstTy.getRank();
  int64_t totalElemsI64 = 1;
  for (int64_t d = 0; d < rank; ++d) {
    int64_t dim = dstTy.getDimSize(d);
    if (dim <= 0 ||
        totalElemsI64 > (std::numeric_limits<int64_t>::max() / dim)) {
      totalElemsI64 = 0;
      break;
    }
    totalElemsI64 *= dim;
  }
  if (totalElemsI64 <= 0)
    return false;

  Value totalElems = b.create<arith::ConstantIndexOp>(loc, totalElemsI64);
  auto loop = b.create<scf::ForOp>(loc, tid, totalElems, totalThreads);
  OpBuilder lb(loop.getBody(), loop.getBody()->begin());
  Value lin = loop.getInductionVar();

  SmallVector<Value, 4> idx(rank);
  Value q = lin;
  for (int64_t d = rank - 1; d >= 1; --d) {
    Value dim = lb.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(d));
    Value r = lb.create<arith::RemSIOp>(loc, q, dim);
    q = lb.create<arith::DivSIOp>(loc, q, dim);
    idx[d] = r;
  }
  idx[0] = q;

  lb.create<memref::StoreOp>(loc, fillVal, copy.getTarget(), idx);

  b.setInsertionPointAfter(loop);
  auto barrier = b.create<gpu::BarrierOp>(loc);
  if (keepBarrier)
    barrier->setAttr("welder.keep_barrier", keepBarrier);

  copy.erase();
  return true;
}


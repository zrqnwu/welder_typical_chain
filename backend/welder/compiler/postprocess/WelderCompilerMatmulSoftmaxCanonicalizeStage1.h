#pragma once

static bool regionHasDivF(linalg::GenericOp op) {
  if (!op)
    return false;
  bool found = false;
  op->walk([&](arith::DivFOp) { found = true; });
  return found;
}

static bool regionHasExp(linalg::GenericOp op) {
  if (!op)
    return false;
  bool found = false;
  op->walk([&](math::ExpOp) { found = true; });
  return found;
}

static bool regionHasMaxF(linalg::GenericOp op) {
  if (!op)
    return false;
  bool found = false;
  op->walk([&](arith::MaximumFOp) { found = true; });
  return found;
}

static bool regionHasMaxF(linalg::ReduceOp op) {
  if (!op)
    return false;
  bool found = false;
  op->walk([&](arith::MaximumFOp) { found = true; });
  return found;
}

static bool regionHasAddF(linalg::GenericOp op) {
  if (!op)
    return false;
  bool found = false;
  op->walk([&](arith::AddFOp) { found = true; });
  return found;
}

// 类似 `stripToBaseMemref`，但会保留表示 workgroup 分配的“带类型 memref view”
//（i8 alloc + memref.view）。在融合修复里，我们希望 subview 指向该 typed view，
// 而不是底层 i8 原始缓冲区。
static Value stripToTypedMemrefBase(Value v) {
  while (v) {
    if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
      v = sub.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto rcast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
      v = rcast.getSource();
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

static Value promoteAllocLikeMemrefToWorkgroup(Value baseMemref) {
  if (!baseMemref)
    return baseMemref;
  auto memrefType = dyn_cast<MemRefType>(baseMemref.getType());
  if (!memrefType || !memrefType.hasStaticShape())
    return baseMemref;
  if (isWorkgroupMemoryType(memrefType))
    return baseMemref;

  Operation *def = baseMemref.getDefiningOp();
  if (!def)
    return baseMemref;

  memref::AllocOp alloc = dyn_cast<memref::AllocOp>(def);
  memref::AllocaOp alloca = dyn_cast<memref::AllocaOp>(def);
  if (!alloc && !alloca)
    return baseMemref;

  int64_t elemBytes = getElementByteWidth(memrefType.getElementType());
  if (elemBytes <= 0)
    return baseMemref;
  int64_t elems = memrefType.getNumElements();
  if (elems <= 0)
    return baseMemref;
  if (elems > (std::numeric_limits<int64_t>::max() / elemBytes))
    return baseMemref;
  int64_t bytes = elems * elemBytes;
  if (bytes <= 0)
    return baseMemref;

  OpBuilder b(def);
  Location loc = def->getLoc();
  IntegerAttr wgSpace = b.getI64IntegerAttr(3);
  auto i8Type = b.getI8Type();
  MemRefLayoutAttrInterface defaultLayout;
  auto wgI8Type = MemRefType::get({bytes}, i8Type, defaultLayout, wgSpace);
  auto newAlloc = b.create<memref::AllocOp>(loc, wgI8Type, /*dynamicSizes=*/ValueRange{},
                                            /* symbolOperands=*/ValueRange{},
                                            alloc ? alloc.getAlignmentAttr()
                                                  : alloca.getAlignmentAttr());
  Value zeroShift = b.create<arith::ConstantIndexOp>(loc, 0);
  auto wgViewType = MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                                    memrefType.getLayout(), wgSpace);
  auto view = b.create<memref::ViewOp>(loc, wgViewType, newAlloc, zeroShift,
                                       /* sizes=*/ValueRange{});

  baseMemref.replaceAllUsesWith(view.getResult());
  if (alloc)
    alloc.erase();
  else if (alloca)
    alloca.erase();

  return view.getResult();
}

static void predicateWorkgroupCopyToThread0(memref::CopyOp copy,
                                           UnitAttr keepBarrier) {
  if (!copy)
    return;
  auto dstTy = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!dstTy || !isWorkgroupMemoryType(dstTy))
    return;
  if (copy->getParentOfType<scf::IfOp>())
    return;
  gpu::LaunchOp launch = copy->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return;
  // 仅对顶层初始化 copy 做谓词化，避免干扰已分发的 staging copy。
  if (copy->getBlock() != &launch.getBody().front())
    return;

  // 优先采用线程协作 copy，而不是把初始化 copy 串行化到 thread0。
  // 对 block tile（如 32x128 f32）而言，串行初始化开销极高，可能主导融合 kernel 时延。
  auto srcTy = dyn_cast<MemRefType>(copy.getSource().getType());
  if (srcTy && dstTy && srcTy.hasStaticShape() && dstTy.hasStaticShape() &&
      srcTy.getRank() == dstTy.getRank() &&
      srcTy.getElementType() == dstTy.getElementType() &&
      srcTy.getShape() == dstTy.getShape()) {
    const int64_t rank = dstTy.getRank();
    if (rank >= 1) {
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

      int64_t totalElemsI64 = 1;
      for (int64_t d = 0; d < rank; ++d) {
        int64_t dim = dstTy.getDimSize(d);
        if (dim <= 0 || totalElemsI64 > (std::numeric_limits<int64_t>::max() / dim)) {
          totalElemsI64 = 0;
          break;
        }
        totalElemsI64 *= dim;
      }
      if (totalElemsI64 > 0) {
        Value totalElems =
            b.create<arith::ConstantIndexOp>(loc, totalElemsI64);

        auto loop = b.create<scf::ForOp>(loc, tid, totalElems, totalThreads);
        OpBuilder lb(loop.getBody(), loop.getBody()->begin());
        Value lin = loop.getInductionVar();

        SmallVector<Value, 4> idx(rank);
        Value q = lin;
        for (int64_t d = rank - 1; d >= 1; --d) {
          Value dim =
              lb.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(d));
          Value r = lb.create<arith::RemSIOp>(loc, q, dim);
          q = lb.create<arith::DivSIOp>(loc, q, dim);
          idx[d] = r;
        }
        idx[0] = q;

        Value v = lb.create<memref::LoadOp>(loc, copy.getSource(), idx);
        lb.create<memref::StoreOp>(loc, v, copy.getTarget(), idx);

        b.setInsertionPointAfter(loop);
        auto barrier = b.create<gpu::BarrierOp>(loc);
        if (keepBarrier)
          barrier->setAttr("welder.keep_barrier", keepBarrier);

        copy.erase();
        return;
      }
    }
  }

  OpBuilder b(copy);
  Location loc = copy.getLoc();
  Value tidx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value tidy = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value is0x = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidx, c0);
  Value is0y = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidy, c0);
  Value cond = b.create<arith::AndIOp>(loc, is0x, is0y);

  auto ifOp = b.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);
  Block &thenBlock = ifOp.getThenRegion().front();
  copy->moveBefore(thenBlock.getTerminator());

  b.setInsertionPointAfter(ifOp);
  // 确保其他线程不会在单线程初始化 copy 完成前提前消费 shared 缓冲区。
  Operation *next = ifOp->getNextNode();
  if (!isa_and_present<gpu::BarrierOp>(next)) {
    auto barrier = b.create<gpu::BarrierOp>(loc);
    if (keepBarrier)
      barrier->setAttr("welder.keep_barrier", keepBarrier);
  }
}


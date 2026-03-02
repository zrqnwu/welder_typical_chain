#pragma once

// 归约链代码生成：将小型 1D 中间结果（如 max/mean/sum）保留在 gpu.launch 内，
// 通过 workgroup(shared) 分配避免走 host 侧 `memref.alloc + fill` 造成的
// global-memory 往返。
static void promoteLaunchLocal1DBuffersToWorkgroup(ModuleOp module) {
  if (!module)
    return;
  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;

  Builder b(ctx);
  IntegerAttr wgSpace = b.getI64IntegerAttr(3);
  UnitAttr keepBarrier = UnitAttr::get(ctx);

  struct Item {
    memref::AllocOp alloc;
    gpu::LaunchOp launch;
    linalg::FillOp fill;
    Value fillVal;
    SmallVector<memref::CopyOp, 2> copiesToErase;
  };

  SmallVector<Item, 8> items;

  module.walk([&](memref::AllocOp alloc) {
    if (!alloc || alloc->getParentOfType<gpu::LaunchOp>())
      return;
    auto memrefType = dyn_cast<MemRefType>(alloc.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return;
    if (isWorkgroupMemoryType(memrefType))
      return;
    if (memrefType.getRank() != 1)
      return;
    int64_t elemBytes = getElementByteWidth(memrefType.getElementType());
    if (elemBytes <= 0)
      return;

    auto launchOpt = findUniqueLaunchUser(alloc.getResult());
    if (!launchOpt)
      return;
    gpu::LaunchOp launch = *launchOpt;
    if (!launch)
      return;

    linalg::FillOp fillOp;
    Value fillVal;
    SmallVector<memref::CopyOp, 2> copies;

    for (OpOperand &use : llvm::make_early_inc_range(alloc.getResult().getUses())) {
      Operation *owner = use.getOwner();
      if (!owner)
        continue;
      if (isUsedInLaunch(owner, launch))
        continue;

      if (auto fill = dyn_cast<linalg::FillOp>(owner)) {
        if (fill.getNumDpsInits() == 1 &&
            fill.getDpsInitOperand(0)->get() == alloc.getResult()) {
          fillOp = fill;
          if (fill.getNumDpsInputs() == 1)
            fillVal = fill.getDpsInputOperand(0)->get();
          continue;
        }
      }
      if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
        Value src = copy.getSource();
        Value dst = copy.getTarget();
        if (src == alloc.getResult() || dst == alloc.getResult()) {
          Value other = (src == alloc.getResult()) ? dst : src;
          if (other) {
            for (OpOperand &u2 : other.getUses()) {
              Operation *o2 = u2.getOwner();
              if (o2 && isUsedInLaunch(o2, launch)) {
                // 保守策略：若初始化 copy 通过其他缓冲区喂给 launch，则不改写。
                return;
              }
            }
          }
          copies.push_back(copy);
          continue;
        }
      }

      // 出现未知逃逸使用，跳过。
      return;
    }

    Item it;
    it.alloc = alloc;
    it.launch = launch;
    it.fill = fillOp;
    it.fillVal = fillVal;
    it.copiesToErase = std::move(copies);
    items.push_back(std::move(it));
  });

  if (items.empty())
    return;

  for (Item &it : items) {
    memref::AllocOp alloc = it.alloc;
    gpu::LaunchOp launch = it.launch;
    if (!alloc || !launch)
      continue;

    auto memrefType = dyn_cast<MemRefType>(alloc.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      continue;

    int64_t elemBytes = getElementByteWidth(memrefType.getElementType());
    if (elemBytes <= 0)
      continue;
    int64_t elems = memrefType.getNumElements();
    if (elems <= 0)
      continue;
    if (elems >
        (std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, elemBytes)))
      continue;
    int64_t bytes = elems * elemBytes;
    if (bytes <= 0)
      continue;

    Block &body = launch.getBody().front();
    OpBuilder lb = OpBuilder::atBlockBegin(&body);
    Location loc = alloc.getLoc();

    MemRefLayoutAttrInterface defaultLayout;
    auto wgI8Type =
        MemRefType::get({bytes}, lb.getI8Type(), defaultLayout, wgSpace);
    auto wgAlloc = lb.create<memref::AllocOp>(loc, wgI8Type, ValueRange{},
                                              ValueRange{},
                                              alloc.getAlignmentAttr());
    Value zeroShift = lb.create<arith::ConstantIndexOp>(loc, 0);
    auto wgViewType = MemRefType::get(memrefType.getShape(),
                                      memrefType.getElementType(),
                                      memrefType.getLayout(), wgSpace);
    auto wgView =
        lb.create<memref::ViewOp>(loc, wgViewType, wgAlloc, zeroShift,
                                  /* sizes=*/ValueRange{});

    // 可选初始化：全线程并行填充常量。
    if (it.fill && it.fillVal) {
      Value tidx = lb.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value tidy = lb.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
      Value bdx = lb.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
      Value bdy = lb.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
      Value linear = lb.create<arith::AddIOp>(
          loc, tidx, lb.create<arith::MulIOp>(loc, tidy, bdx));
      Value threads = lb.create<arith::MulIOp>(loc, bdx, bdy);
      Value upper = lb.create<arith::ConstantIndexOp>(loc, elems);

      auto forOp = lb.create<scf::ForOp>(loc, linear, upper, threads);
      Block *forBody = forOp.getBody();
      Operation *term = forBody ? forBody->getTerminator() : nullptr;
      OpBuilder fb(term);
      Value iv = forOp.getInductionVar();
      fb.create<memref::StoreOp>(loc, it.fillVal, wgView, ValueRange{iv});
      lb.setInsertionPointAfter(forOp);
      auto barrier = lb.create<gpu::BarrierOp>(loc);
      barrier->setAttr("welder.keep_barrier", keepBarrier);
    }

    // 将 launch 内使用点改写为 workgroup view。
    SmallVector<OpOperand *, 16> uses;
    for (OpOperand &use : alloc.getResult().getUses())
      uses.push_back(&use);
    for (OpOperand *u : uses) {
      if (!u)
        continue;
      Operation *owner = u->getOwner();
      if (!owner)
        continue;
      if (!isUsedInLaunch(owner, launch))
        continue;
      u->set(wgView.getResult());
    }

    // 删除 host 侧初始化操作。
    for (memref::CopyOp c : it.copiesToErase) {
      if (c)
        c.erase();
    }
    if (it.fill)
      it.fill.erase();

    // 原 alloc 若已无用则删除。
    if (alloc.getResult().use_empty())
      alloc.erase();
  }

  // base memref 改写到 workgroup 地址空间后，修正过期的 subview 结果类型。
  fixSubviewResultTypesAfterWorkgroupPromotion(module);
}

#pragma once

static bool isWorkgroupMemoryType(MemRefType memrefType) {
  if (!memrefType)
    return false;
  if (gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memrefType))
    return true;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
    return intAttr.getInt() == 3;
  return false;
}

static int64_t getElementByteWidth(Type elemType);

static void fixSubviewResultTypesAfterWorkgroupPromotion(ModuleOp module) {
  if (!module)
    return;

  SmallVector<memref::SubViewOp, 16> subviewsToFix;
  module.walk([&](memref::SubViewOp sv) {
    auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
    auto resTy = dyn_cast<MemRefType>(sv.getResult().getType());
    if (!srcTy || !resTy)
      return;
    if (!isWorkgroupMemoryType(srcTy))
      return;
    if (isWorkgroupMemoryType(resTy))
      return;
    subviewsToFix.push_back(sv);
  });

  for (memref::SubViewOp sv : subviewsToFix) {
    auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
    auto oldResTy = dyn_cast<MemRefType>(sv.getResult().getType());
    if (!srcTy || !oldResTy)
      continue;

    MemRefType newResTy = memref::SubViewOp::inferRankReducedResultType(
        oldResTy.getShape(), srcTy, sv.getMixedOffsets(), sv.getMixedSizes(),
        sv.getMixedStrides());
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

static bool isRowReductionScratchAlloc(memref::AllocOp alloc) {
  if (!alloc)
    return false;
  if (!alloc->getParentOfType<gpu::LaunchOp>())
    return false;
  auto memrefType = dyn_cast<MemRefType>(alloc.getType());
  if (!memrefType || !memrefType.hasStaticShape())
    return false;
  if (isWorkgroupMemoryType(memrefType))
    return false;
  if (memrefType.getRank() != 2)
    return false;
  // 启发式：scratch 缓冲区通常形如 `[rows, threads_x]`，
  // 并参与最后一维上的 combine `linalg.reduce`。
  bool hasRowReductionWriter = false;
  bool hasRowReductionReader = false;
  bool hasElementwiseWriter = false;
  bool hasCombiningReducer = false;

  auto scanUse = [&](Value v) {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (!owner)
        continue;
      if (auto gen = dyn_cast<linalg::GenericOp>(owner)) {
        if (gen->hasAttr("welder.row_reduction")) {
          // 通过 output/init 操作数写入。
          for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
            OpOperand *init = gen.getDpsInitOperand(i);
            if (init && init->get() == v) {
              hasRowReductionWriter = true;
              break;
            }
          }
          // 通过 input 操作数读取。
          for (unsigned i = 0, e = gen.getNumDpsInputs(); i < e; ++i) {
            OpOperand *in = gen.getDpsInputOperand(i);
            if (in && in->get() == v) {
              hasRowReductionReader = true;
              break;
            }
          }
        }
        if (gen->hasAttr("welder.elementwise_nd") ||
            gen->hasAttr("welder.elementwise")) {
          for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
            OpOperand *init = gen.getDpsInitOperand(i);
            if (init && init->get() == v) {
              hasElementwiseWriter = true;
              break;
            }
          }
        }
      }
      if (auto red = dyn_cast<linalg::ReduceOp>(owner)) {
        // 通过 input 操作数读取，并在 dim=1 上归约。
        for (unsigned i = 0, e = red.getNumDpsInputs(); i < e; ++i) {
          OpOperand *in = red.getDpsInputOperand(i);
          if (in && in->get() == v) {
            auto dims = red.getDimensions();
            for (int64_t d : dims) {
              if (d == 1) {
                hasCombiningReducer = true;
                break;
              }
            }
          }
        }
      }
    }
  };

  for (Operation *user : alloc.getResult().getUsers()) {
    if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
      scanUse(sv.getResult());
    } else {
      // 直接使用（无 subview）。
      scanUse(alloc.getResult());
    }
    if ((hasCombiningReducer || hasRowReductionReader) &&
        (hasRowReductionWriter || hasElementwiseWriter))
      return true;
  }
  return false;
}

static void promoteRowReductionScratchToWorkgroup(ModuleOp module) {
  if (!module)
    return;
  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;

  SmallVector<memref::AllocOp, 8> scratchAllocs;
  module.walk([&](memref::AllocOp alloc) {
    if (isRowReductionScratchAlloc(alloc))
      scratchAllocs.push_back(alloc);
  });

  if (scratchAllocs.empty())
    return;

  for (memref::AllocOp alloc : scratchAllocs) {
    auto memrefType = dyn_cast<MemRefType>(alloc.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      continue;

    int64_t elemBytes = -1;
    Type elemType = memrefType.getElementType();
    if (auto it = dyn_cast<IntegerType>(elemType))
      elemBytes = (it.getWidth() + 7) / 8;
    else if (auto ft = dyn_cast<FloatType>(elemType))
      elemBytes = (ft.getWidth() + 7) / 8;
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

    OpBuilder b(alloc);
    Location loc = alloc.getLoc();
    IntegerAttr wgSpace = b.getI64IntegerAttr(3);
    auto i8Type = b.getI8Type();
    // workgroup lowering pass 期望 i8 分配采用默认（identity）layout，
    // 以匹配 structured.promote 的表示形式。
    MemRefLayoutAttrInterface defaultLayout;
    auto wgI8Type = MemRefType::get({bytes}, i8Type, defaultLayout, wgSpace);
    auto wgAlloc = b.create<memref::AllocOp>(
        loc, wgI8Type, /*dynamicSizes=*/ValueRange{},
        /* symbolOperands=*/ValueRange{}, alloc.getAlignmentAttr());
    Value zeroShift = b.create<arith::ConstantIndexOp>(loc, 0);

    auto wgViewType = MemRefType::get(memrefType.getShape(), elemType,
                                      memrefType.getLayout(), wgSpace);
    auto view = b.create<memref::ViewOp>(loc, wgViewType, wgAlloc, zeroShift,
                                         /* sizes=*/ValueRange{});
    alloc.getResult().replaceAllUsesWith(view.getResult());
    alloc.erase();
  }

  fixSubviewResultTypesAfterWorkgroupPromotion(module);
}

// 归约链代码生成中，部分行归约结果（如 max/mean/sum）来自带谓词的 combine 归约
//（仅部分线程执行），后续又会被所有线程参与的逐元素阶段读取。
// 这类中间值必须放在 workgroup(shared) 内存；若使用 memref.alloca，
// 会退化成线程私有副本并导致结果错误（只有写入线程子集能看到值）。
//
// 本 pass 会把这些“共享 1D 归约结果”从 memref.alloca 提升为
// workgroup i8 allocation + view，并补充分布式中性元初始化，
// 使 combine 归约不依赖外部（host/profiler）初始化。
static void promoteSharedRowReductionResultAllocasToWorkgroup(ModuleOp module) {
  if (!module)
    return;
  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;

  Builder b(ctx);
  IntegerAttr wgSpace = b.getI64IntegerAttr(3);
  UnitAttr keepBarrier = UnitAttr::get(ctx);

  enum class InitKind { kUnknown, kZero, kNegInf };

  auto getInitKindFromReduce = [&](linalg::ReduceOp red) -> InitKind {
    if (!red)
      return InitKind::kUnknown;
    bool sawAdd = false;
    bool sawMax = false;
    red.getRegion().walk([&](Operation *op) {
      if (!op)
        return WalkResult::advance();
      if (isa<arith::AddFOp, arith::AddIOp>(op)) {
        sawAdd = true;
      } else if (isa<arith::MaximumFOp>(op)) {
        sawMax = true;
      }
      return WalkResult::advance();
    });
    if (sawMax)
      return InitKind::kNegInf;
    if (sawAdd)
      return InitKind::kZero;
    return InitKind::kUnknown;
  };

  auto isElementwiseConsumer = [&](Operation *op) -> bool {
    auto gen = dyn_cast_or_null<linalg::GenericOp>(op);
    if (!gen)
      return false;
    return gen->hasAttr("welder.elementwise") ||
           gen->hasAttr("welder.elementwise_nd") ||
           gen->hasAttr("welder.elementwise_1d");
  };

  auto isDerivedFrom = [&](Value v, Value base) -> bool {
    if (!v || !base)
      return false;
    Value cur = v;
    while (cur) {
      if (cur == base)
        return true;
      if (auto sv = cur.getDefiningOp<memref::SubViewOp>()) {
        cur = sv.getSource();
        continue;
      }
      if (auto cast = cur.getDefiningOp<memref::CastOp>()) {
        cur = cast.getSource();
        continue;
      }
      if (auto rcast = cur.getDefiningOp<memref::ReinterpretCastOp>()) {
        cur = rcast.getSource();
        continue;
      }
      if (auto view = cur.getDefiningOp<memref::ViewOp>()) {
        cur = view.getSource();
        continue;
      }
      if (auto assumed = cur.getDefiningOp<memref::AssumeAlignmentOp>()) {
        cur = assumed.getMemref();
        continue;
      }
      break;
    }
    return false;
  };

  auto isTrivialInitCopyLoop = [&](scf::ForOp forOp, Value dstBase,
                                  int64_t expectedElems) -> bool {
    if (!forOp || !dstBase)
      return false;
    auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (!lb || lb.value() != 0)
      return false;
    if (!step || step.value() != 1)
      return false;
    if (ub && expectedElems > 0 && ub.value() != expectedElems)
      return false;

    Block *body = forOp.getBody();
    if (!body)
      return false;
    Value iv = forOp.getInductionVar();

    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation &op : body->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      if (auto l = dyn_cast<memref::LoadOp>(&op)) {
        if (load)
          return false;
        load = l;
        continue;
      }
      if (auto s = dyn_cast<memref::StoreOp>(&op)) {
        if (store)
          return false;
        store = s;
        continue;
      }
      // 初始化循环中允许无副作用的 cast。
      if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncIOp, memref::CastOp>(&op))
        continue;
      return false;
    }
    if (!load || !store)
      return false;

    if (store.getValue() != load.getResult())
      return false;
    if (!isDerivedFrom(store.getMemref(), dstBase))
      return false;

    if (store.getIndices().size() != 1 || load.getIndices().size() != 1)
      return false;
    if (store.getIndices()[0] != iv || load.getIndices()[0] != iv)
      return false;

    return true;
  };

  struct Item {
    Operation *allocOp = nullptr;
    Value buffer;
    gpu::LaunchOp launch;
    InitKind initKind = InitKind::kUnknown;
    llvm::SmallPtrSet<Operation *, 4> initOpsToErase;
  };
  SmallVector<Item, 8> items;

  auto considerBuffer = [&](Operation *allocOp, Value buffer,
                            gpu::LaunchOp launch) {
    if (!allocOp || !buffer || !launch)
      return;
    auto memrefType = dyn_cast<MemRefType>(buffer.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return;
    if (isWorkgroupMemoryType(memrefType))
      return;
    if (memrefType.getRank() != 1)
      return;
    if (memrefType.getNumElements() <= 0)
      return;

    // 必须先由 combine `linalg.reduce` 写入，再被逐元素广播算子读取。
    InitKind initKind = InitKind::kUnknown;
    bool hasCombiningWriter = false;
    bool hasElementwiseReader = false;
    llvm::SmallPtrSet<Operation *, 4> initOps;

    // 穿过 alloca 的 subview/cast 链，查找真实用户。
    llvm::SmallDenseSet<const void *, 64> seen;
    llvm::SmallVector<Value, 16> worklist;
    worklist.push_back(buffer);
    seen.insert(buffer.getAsOpaquePointer());

    auto push = [&](Value v) {
      if (!v)
        return;
      if (!isa<MemRefType>(v.getType()))
        return;
      const void *k = v.getAsOpaquePointer();
      if (!seen.insert(k).second)
        return;
      worklist.push_back(v);
    };

    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      for (Operation *user : cur.getUsers()) {
        if (!user)
          continue;
        if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
          push(sv.getResult());
          continue;
        }
        if (auto cast = dyn_cast<memref::CastOp>(user)) {
          push(cast.getResult());
          continue;
        }
        if (auto rcast = dyn_cast<memref::ReinterpretCastOp>(user)) {
          push(rcast.getResult());
          continue;
        }
        if (auto view = dyn_cast<memref::ViewOp>(user)) {
          push(view.getResult());
          continue;
        }
        if (auto assumed = dyn_cast<memref::AssumeAlignmentOp>(user)) {
          push(assumed.getResult());
          continue;
        }

        if (auto red = dyn_cast<linalg::ReduceOp>(user)) {
          // 写者：作为 output/init 操作数使用。
          for (unsigned i = 0, e = red.getNumDpsInits(); i < e; ++i) {
            OpOperand *init = red.getDpsInitOperand(i);
            if (init && isDerivedFrom(init->get(), buffer)) {
              hasCombiningWriter = true;
              InitKind k = getInitKindFromReduce(red);
              if (k != InitKind::kUnknown)
                initKind = k;
              break;
            }
          }
          continue;
        }

        if (isElementwiseConsumer(user)) {
          // 读者：作为 input 操作数使用。
          auto gen = dyn_cast<linalg::GenericOp>(user);
          if (!gen)
            continue;
          for (unsigned i = 0, e = gen.getNumDpsInputs(); i < e; ++i) {
            OpOperand *in = gen.getDpsInputOperand(i);
            if (in && isDerivedFrom(in->get(), buffer)) {
              hasElementwiseReader = true;
              break;
            }
          }
          continue;
        }

        // 初始化循环写者：简单 copy 循环把其他 memref 拷到 alloca[idx]。
        // 提升到 workgroup 后会删除这些循环，避免跨线程竞争与外部初始化依赖。
        if (auto store = dyn_cast<memref::StoreOp>(user)) {
          if (isDerivedFrom(store.getMemref(), buffer)) {
            if (auto forOp = store->getParentOfType<scf::ForOp>()) {
              int64_t elems = memrefType.getNumElements();
              if (isTrivialInitCopyLoop(forOp, buffer, elems))
                initOps.insert(forOp.getOperation());
            }
          }
          continue;
        }

        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          Value dst = copy.getTarget();
          if (dst && isDerivedFrom(dst, buffer)) {
            // 将 workgroup->private 的 copy 视作初始化脚手架（归约切分后常见）。
            // 这里改为在 workgroup 内显式写入中性元初始化。
            initOps.insert(copy.getOperation());
          }
          continue;
        }
      }
    }

    if (!hasCombiningWriter || !hasElementwiseReader)
      return;
    if (initKind == InitKind::kUnknown)
      return;

    Item it;
    it.allocOp = allocOp;
    it.buffer = buffer;
    it.launch = launch;
    it.initKind = initKind;
    it.initOpsToErase = std::move(initOps);
    items.push_back(std::move(it));
  };

  module.walk([&](memref::AllocaOp alloca) {
    if (!alloca)
      return;
    gpu::LaunchOp launch = alloca->getParentOfType<gpu::LaunchOp>();
    if (!launch)
      return;
    considerBuffer(alloca.getOperation(), alloca.getResult(), launch);
  });
  module.walk([&](memref::AllocOp alloc) {
    if (!alloc)
      return;
    gpu::LaunchOp launch = alloc->getParentOfType<gpu::LaunchOp>();
    if (!launch)
      return;
    considerBuffer(alloc.getOperation(), alloc.getResult(), launch);
  });

  if (items.empty())
    return;

  for (Item &it : items) {
    Operation *allocOp = it.allocOp;
    gpu::LaunchOp launch = it.launch;
    Value buffer = it.buffer;
    if (!allocOp || !launch || !buffer)
      continue;

    auto memrefType = dyn_cast<MemRefType>(buffer.getType());
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
    Location loc = allocOp->getLoc();

    MemRefLayoutAttrInterface defaultLayout;
    auto wgI8Type =
        MemRefType::get({bytes}, lb.getI8Type(), defaultLayout, wgSpace);
    IntegerAttr alignAttr;
    if (auto a = dyn_cast<memref::AllocOp>(allocOp))
      alignAttr = a.getAlignmentAttr();
    else if (auto a = dyn_cast<memref::AllocaOp>(allocOp))
      alignAttr = a.getAlignmentAttr();
    auto wgAlloc =
        lb.create<memref::AllocOp>(loc, wgI8Type, ValueRange{}, ValueRange{},
                                   /* alignment=*/alignAttr);
    Value zeroShift = lb.create<arith::ConstantIndexOp>(loc, 0);
    auto wgViewType = MemRefType::get(memrefType.getShape(),
                                      memrefType.getElementType(),
                                      memrefType.getLayout(), wgSpace);
    auto wgView =
        lb.create<memref::ViewOp>(loc, wgViewType, wgAlloc, zeroShift,
                                  /* sizes=*/ValueRange{});

    // 分布式初始化为归约中性元。
    Value initVal;
    if (it.initKind == InitKind::kZero) {
      initVal = lb.create<arith::ConstantOp>(
          loc, memrefType.getElementType(),
          lb.getZeroAttr(memrefType.getElementType()));
    } else if (it.initKind == InitKind::kNegInf) {
      Type elemTy = memrefType.getElementType();
      if (auto ft = dyn_cast<FloatType>(elemTy)) {
        // 最大值归约的中性元使用 -inf。
        APFloat apf = APFloat::getInf(ft.getFloatSemantics(), /*negative=*/true);
        initVal =
            lb.create<arith::ConstantOp>(loc, elemTy, lb.getFloatAttr(elemTy, apf));
      } else {
        // 不支持作为 -inf 中性元的元素类型。
        initVal = Value();
      }
    }

    if (initVal) {
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
      fb.create<memref::StoreOp>(loc, initVal, wgView, ValueRange{iv});
      lb.setInsertionPointAfter(forOp);
      auto barrier = lb.create<gpu::BarrierOp>(loc);
      barrier->setAttr("welder.keep_barrier", keepBarrier);
    }

    // 删除初始化 copy（提升后会引入跨线程竞争，且我们已显式初始化）。
    for (Operation *op : it.initOpsToErase) {
      if (op)
        op->erase();
    }

    // 将 launch 内使用改写为 workgroup view。
    buffer.replaceAllUsesWith(wgView.getResult());
    allocOp->erase();
  }

  // base memref 改到 workgroup 地址空间后，修正可能过期的 subview 结果类型。
  fixSubviewResultTypesAfterWorkgroupPromotion(module);
}


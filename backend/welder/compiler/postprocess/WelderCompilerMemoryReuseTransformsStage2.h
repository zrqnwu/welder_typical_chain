#pragma once

static bool isUsedInLaunch(Operation *op, gpu::LaunchOp launch) {
  if (!op || !launch)
    return false;
  return launch->isAncestor(op);
}

static std::optional<gpu::LaunchOp> findUniqueLaunchUser(Value v) {
  if (!v)
    return std::nullopt;
  gpu::LaunchOp found;
  for (OpOperand &use : v.getUses()) {
    Operation *owner = use.getOwner();
    if (!owner)
      continue;
    gpu::LaunchOp launch = owner->getParentOfType<gpu::LaunchOp>();
    if (!launch)
      continue;
    if (!found)
      found = launch;
    else if (launch != found)
      return std::nullopt;
  }
  if (!found)
    return std::nullopt;
  return found;
}

static int64_t getElementByteWidth(Type elemType) {
  if (!elemType)
    return -1;
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return (intTy.getWidth() + 7) / 8;
  if (auto floatTy = dyn_cast<FloatType>(elemType))
    return (floatTy.getWidth() + 7) / 8;
  return -1;
}

static Value stripToBaseMemref(Value v);

// 行归约链的协作式 block 级 staging：
// - 分配与 2D 输入 tile memref 对应的 workgroup tile 缓冲区；
// - 全线程协作把 tile 从 global 复制到 shared（仅拷一次）；
// - 重写下游 subview，使其从 shared 读取；
// - 可选地让单个 2D 中间 producer（如 exp_sub/sq）复用该 shared tile，
//   减少 global scratch 流量。
//
// 该策略故意保守，仅对可静态确定且能放入 `(smemBytes - reserveBytes)` 的 tile 触发。
static void stageRowReduction2DTileToWorkgroup(ModuleOp module,
                                               int64_t smemBytes,
                                               int64_t reserveBytes = 16 * 1024,
                                               bool enableInPlace2DReuse = true,
                                               bool enableVectorize = false,
                                               bool enableAsyncCopy = false,
                                               bool enableSoftwarePipelining = false,
                                               bool asyncBypassL1 = true,
                                               bool relaxBarriers = false,
                                               int64_t vectorWidth = 0) {
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;

  Builder b(ctx);
  IntegerAttr wgSpace = b.getI64IntegerAttr(3);
  UnitAttr keepBarrier = relaxBarriers ? UnitAttr() : UnitAttr::get(ctx);
  bool wantBypassL1 = asyncBypassL1;

  auto isFuncArgMemref = [&](Value v, func::FuncOp func) -> bool {
    auto barg = dyn_cast<BlockArgument>(v);
    if (!barg)
      return false;
    Block *entry = &func.getBody().front();
    return barg.getOwner() == entry && barg.getArgNumber() < func.getNumArguments();
  };

  auto isReturnedByFunc = [&](Value v, func::FuncOp func) -> bool {
    if (!v || !func)
      return false;
    Value baseV = stripToBaseMemref(v);
    for (func::ReturnOp ret : func.getOps<func::ReturnOp>()) {
      for (Value rv : ret.getOperands()) {
        if (stripToBaseMemref(rv) == baseV)
          return true;
      }
    }
    return false;
  };

  module.walk([&](gpu::LaunchOp launch) {
    func::FuncOp func = launch->getParentOfType<func::FuncOp>();
    if (!func)
      return;

    // 仅处理包含行归约的 launch。
    bool hasRowReduction = false;
    launch.walk([&](linalg::GenericOp op) {
      if (op->hasAttr("welder.row_reduction")) {
        hasRowReduction = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasRowReduction)
      return;

    Block &body = launch.getBody().front();

    auto feedsRowReduction = [&](Value v) -> bool {
      if (!v)
        return false;
      if (!isa<MemRefType>(v.getType()))
        return false;
      llvm::SmallDenseSet<const void *, 64> seen;
      llvm::SmallVector<Value, 16> worklist;
      worklist.push_back(v);
      seen.insert(v.getAsOpaquePointer());

      auto push = [&](Value x) {
        if (!x)
          return;
        if (!isa<MemRefType>(x.getType()))
          return;
        const void *k = x.getAsOpaquePointer();
        if (!seen.insert(k).second)
          return;
        worklist.push_back(x);
      };

      while (!worklist.empty()) {
        Value cur = worklist.pop_back_val();
        for (Operation *user : cur.getUsers()) {
          if (!user)
            continue;
          if (auto gen = dyn_cast<linalg::GenericOp>(user)) {
            if (gen->hasAttr("welder.row_reduction")) {
              for (unsigned i = 0, e = gen.getNumDpsInputs(); i < e; ++i) {
                OpOperand *in = gen.getDpsInputOperand(i);
                if (in && in->get() == cur)
                  return true;
              }
            }
          }
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
        }
      }
      return false;
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

    // 尝试寻找函数参数上的 block-tile subview：即一个 2D subview，
    // 且其结果还会继续被 subview（作为基准 tile memref）。
    Value tileIn;
    memref::SubViewOp tileSubviewOp;
    int64_t bestElems = -1;
    launch.walk([&](memref::SubViewOp sv) {
      auto ty = dyn_cast<MemRefType>(sv.getResult().getType());
      if (!ty || ty.getRank() != 2 || !ty.hasStaticShape())
        return;
      if (isWorkgroupMemoryType(ty))
        return;

      bool isTileBase = false;
      for (Operation *u : sv.getResult().getUsers()) {
        if (isa<memref::SubViewOp>(u)) {
          isTileBase = true;
          break;
        }
      }
      if (!isTileBase)
        return;
      if (!feedsRowReduction(sv.getResult()))
        return;

      int64_t elems = ty.getNumElements();
      if (elems > bestElems) {
        bestElems = elems;
        tileIn = sv.getResult();
        tileSubviewOp = sv;
      }
    });

    // 若不存在 tile subview（单 block 场景），则回退到被 subview 使用的
    // 2D 函数参数 memref。
    if (!tileIn) {
      Value bestArg;
      int64_t bestArgElems = -1;
      for (Operation &op : body.getOperations()) {
        auto sv = dyn_cast<memref::SubViewOp>(&op);
        if (!sv)
          continue;
        Value src = stripToBaseMemref(sv.getSource());
        if (!src)
          continue;
        auto ty = dyn_cast<MemRefType>(src.getType());
        if (!ty || ty.getRank() != 2 || !ty.hasStaticShape())
          continue;
        if (isWorkgroupMemoryType(ty))
          continue;
        int64_t elems = ty.getNumElements();
        if (elems > bestArgElems) {
          bestArgElems = elems;
          bestArg = src;
        }
      }
      tileIn = bestArg;
    }

    if (!tileIn)
      return;

    auto tileTy = dyn_cast<MemRefType>(tileIn.getType());
    if (!tileTy || tileTy.getRank() != 2 || !tileTy.hasStaticShape())
      return;
    if (isWorkgroupMemoryType(tileTy))
      return;

    int64_t elemBytes = getElementByteWidth(tileTy.getElementType());
    if (elemBytes <= 0)
      return;
    int64_t elems = tileTy.getNumElements();
    if (elems <= 0)
      return;
    if (elems > (std::numeric_limits<int64_t>::max() / elemBytes))
      return;
    int64_t tileBytes = elems * elemBytes;

    int64_t budget = std::max<int64_t>(
        0, smemBytes - std::max<int64_t>(0, reserveBytes));
    if (tileBytes > budget)
      return;

    Value tileBaseMemref = stripToBaseMemref(tileIn);
    bool tileBaseIsFuncArg =
        tileBaseMemref && isFuncArgMemref(tileBaseMemref, func);
    bool hasWriter = false;
    if (!tileBaseIsFuncArg) {
      launch.walk([&](linalg::LinalgOp op) {
        if (hasWriter)
          return WalkResult::interrupt();
        if (!op)
          return WalkResult::advance();
        for (unsigned i = 0, e = op.getNumDpsInits(); i < e; ++i) {
          OpOperand *init = op.getDpsInitOperand(i);
          if (!init)
            continue;
          if (isDerivedFrom(init->get(), tileIn)) {
            hasWriter = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }
    const bool doStagingCopy = tileBaseIsFuncArg || !hasWriter;

    // 插入 staging 位置：优先在 tile subview 之后，否则在 block 起始处。
    OpBuilder ib(ctx);
    if (tileSubviewOp)
      ib.setInsertionPointAfter(tileSubviewOp);
    else
      ib.setInsertionPointToStart(&body);

    Location loc = launch.getLoc();

    // 以 i8 + view 形式分配 workgroup tile 缓冲区（兼容 workgroup lowering）。
    auto i8Type = ib.getI8Type();
    MemRefLayoutAttrInterface defaultLayout;
    auto wgI8Type = MemRefType::get({tileBytes}, i8Type, defaultLayout, wgSpace);
    auto wgAlloc = ib.create<memref::AllocOp>(loc, wgI8Type);
    Value zeroShift = ib.create<arith::ConstantIndexOp>(loc, 0);
    auto wgTileType = MemRefType::get(tileTy.getShape(), tileTy.getElementType(),
                                      defaultLayout, wgSpace);
    auto wgView =
        ib.create<memref::ViewOp>(loc, wgTileType, wgAlloc, zeroShift, ValueRange{});

    // 线性化的协作 copy：每个线程负责一部分数据。
    // 在可行且安全时，使用 `vector.transfer` 批量搬运最内层连续元素，
    // 以减少指令数。
    Value tidx = ib.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value tidy = ib.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value tidz = ib.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
    Value bdx = ib.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value bdy = ib.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
    Value bdz = ib.create<gpu::BlockDimOp>(loc, gpu::Dimension::z);

    Value tidXY = ib.create<arith::MulIOp>(loc, tidy, bdx);
    Value bdxBdy = ib.create<arith::MulIOp>(loc, bdx, bdy);
    Value tidXYZ = ib.create<arith::MulIOp>(loc, tidz, bdxBdy);
    Value tidXyZ = ib.create<arith::AddIOp>(loc, ib.create<arith::AddIOp>(loc, tidx, tidXY),
                                           tidXYZ);
    Value totalThreads = ib.create<arith::MulIOp>(loc, bdxBdy, bdz);

    Value tileN = ib.create<arith::ConstantIndexOp>(loc, tileTy.getShape()[1]);

    int64_t vecElems = 1;
    if (enableVectorize) {
      // 最内层维优先 16-byte 向量（f32 -> 4 元素）。
      int64_t targetElems =
          (elemBytes > 0) ? std::max<int64_t>(1, 16 / elemBytes) : 1;
      if (vectorWidth > 0)
        targetElems = vectorWidth;
      int64_t n = tileTy.getShape()[1];
      if (targetElems > 1 && n > 0) {
        vecElems = std::min<int64_t>(targetElems, n);
        while (vecElems > 1 && (n % vecElems) != 0)
          vecElems /= 2;
      }
    }

    // NVGPU 的 bypass_l1 提示要求 16B 拷贝。
    // 仅当当前向量宽度满足该条件时才启用，否则会触发 op verifier 失败。
    UnitAttr bypassL1 =
        (wantBypassL1 && (vecElems * elemBytes) == 16) ? UnitAttr::get(ctx)
                                                       : UnitAttr();

    bool wantsAsyncCopy = enableAsyncCopy && enableSoftwarePipelining;
    Operation *stagingScopeOp = nullptr;

    if (!doStagingCopy) {
      // 不做 global->shared staging：将整块 tile base 直接重定向到 shared，
      // 并在第一个行归约阶段前插入 barrier。该路径用于 launch 内自产生 tile 的
      // 跨算子复用场景（如 Matmul->Softmax 中 matmul 写出中间 tile）。

      // 将源自 tileIn 的 subview 改写为源自 wgView。
      launch.walk([&](memref::SubViewOp sv) {
        if (sv.getSource() != tileIn)
          return;
        sv->setOperand(0, wgView.getResult());
      });
      // 同时尽力改写本 launch 内 tileIn 的直接使用点。
      {
        SmallVector<OpOperand *, 16> uses;
        for (OpOperand &use : tileIn.getUses())
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
      }

      // 在首个行归约阶段前插入 barrier，使 producer 对 shared 的写入
      // 对所有线程可见。
      Operation *firstStage = nullptr;
      for (Operation &op : body.getOperations()) {
        bool has = false;
        op.walk([&](linalg::GenericOp gen) {
          if (gen->hasAttr("welder.row_reduction")) {
            has = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (has) {
          firstStage = &op;
          break;
        }
      }
      if (firstStage) {
        Operation *prev = firstStage->getPrevNode();
        if (!isa_and_nonnull<gpu::BarrierOp>(prev)) {
          OpBuilder bb(firstStage);
          auto barrier = bb.create<gpu::BarrierOp>(loc);
          barrier->setAttr("welder.keep_barrier", keepBarrier);
        }
      }

      // 可选：对同形状 2D 中间结果进行原地复用。
      if (enableInPlace2DReuse) {
        Value base0 = stripToBaseMemref(tileIn);
        launch.walk([&](memref::SubViewOp sv) {
          Value base = stripToBaseMemref(sv.getSource());
          if (!base || base == base0)
            return;
          auto baseTy = dyn_cast<MemRefType>(base.getType());
          if (!baseTy || baseTy.getRank() != 2 || !baseTy.hasStaticShape())
            return;
          if (isWorkgroupMemoryType(baseTy))
            return;
          if (baseTy.getShape() != tileTy.getShape())
            return;
          if (baseTy.getElementType() != tileTy.getElementType())
            return;
          auto unique = findUniqueLaunchUser(base);
          if (!unique || *unique != launch)
            return;
          if (isReturnedByFunc(base, func))
            return;
          sv->setOperand(0, wgView.getResult());
        });
      }

      return;
    }

    auto buildStagingLoop = [&](OpBuilder &lb, Value src,
                                bool useAsync) -> scf::ForOp {
      if (vecElems > 1) {
        int64_t packs = elems / vecElems;
        Value totalPacks = lb.create<arith::ConstantIndexOp>(loc, packs);
        Value vecElemsV = lb.create<arith::ConstantIndexOp>(loc, vecElems);

        scf::ForOp loop =
            lb.create<scf::ForOp>(loc, tidXyZ, totalPacks, totalThreads);
        OpBuilder fb(loop.getBody(), loop.getBody()->begin());
        Value p = loop.getInductionVar();
        Value base = fb.create<arith::MulIOp>(loc, p, vecElemsV);
        Value row = fb.create<arith::DivUIOp>(loc, base, tileN);
        Value col = fb.create<arith::RemUIOp>(loc, base, tileN);
        if (useAsync) {
          auto tok = nvgpu::DeviceAsyncCopyOp::create(
              fb, loc, nvgpu::DeviceAsyncTokenType::get(ctx),
              /* dst=*/wgView.getResult(), /*dstIndices=*/ValueRange{row, col},
              /* src=*/src, /*srcIndices=*/ValueRange{row, col},
              /* dstElements=*/lb.getIndexAttr(vecElems),
              /* srcElements=*/Value(), /*bypassL1=*/bypassL1);
          auto group = fb.create<nvgpu::DeviceAsyncCreateGroupOp>(
              loc, nvgpu::DeviceAsyncTokenType::get(ctx),
              ValueRange{tok.getAsyncToken()});
          (void)fb.create<nvgpu::DeviceAsyncWaitOp>(
              loc, group.getAsyncToken(), IntegerAttr());
          return loop;
        }

        // 同步路径：沿最内层维读写 `vector<vecElems x elem>`。
        auto vecTy = VectorType::get({vecElems}, tileTy.getElementType());
        Value pad = lb.create<arith::ConstantOp>(
            loc, tileTy.getElementType(), lb.getZeroAttr(tileTy.getElementType()));
        auto permMap =
            AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                           {lb.getAffineDimExpr(1)}, ctx);
        auto permAttr = AffineMapAttr::get(permMap);
        auto inBounds = lb.getBoolArrayAttr({true});
        auto tr = fb.create<vector::TransferReadOp>(
            loc, vecTy, src, ValueRange{row, col}, permAttr, pad,
            /* mask=*/Value(), inBounds);
        (void)fb.create<vector::TransferWriteOp>(loc, tr.getResult(), wgView,
                                                 ValueRange{row, col}, permAttr,
                                                 /* mask=*/Value(), inBounds);
        return loop;
      }

      Value totalElems = lb.create<arith::ConstantIndexOp>(loc, elems);
      scf::ForOp loop =
          lb.create<scf::ForOp>(loc, tidXyZ, totalElems, totalThreads);
      OpBuilder fb(loop.getBody(), loop.getBody()->begin());
      Value idx = loop.getInductionVar();
      Value row = fb.create<arith::DivUIOp>(loc, idx, tileN);
      Value col = fb.create<arith::RemUIOp>(loc, idx, tileN);
      if (useAsync) {
        auto tok = nvgpu::DeviceAsyncCopyOp::create(
            fb, loc, nvgpu::DeviceAsyncTokenType::get(ctx),
            /* dst=*/wgView.getResult(), /*dstIndices=*/ValueRange{row, col},
            /* src=*/src, /*srcIndices=*/ValueRange{row, col},
            /* dstElements=*/lb.getIndexAttr(1),
            /* srcElements=*/Value(), /*bypassL1=*/bypassL1);
        auto group = fb.create<nvgpu::DeviceAsyncCreateGroupOp>(
            loc, nvgpu::DeviceAsyncTokenType::get(ctx),
            ValueRange{tok.getAsyncToken()});
        (void)fb.create<nvgpu::DeviceAsyncWaitOp>(loc, group.getAsyncToken(),
                                                  IntegerAttr());
        return loop;
      }
      Value v = fb.create<memref::LoadOp>(loc, src, ValueRange{row, col});
      fb.create<memref::StoreOp>(loc, v, wgView, ValueRange{row, col});
      return loop;
    };

    if (wantsAsyncCopy && !tileTy.isLastDimUnitStride()) {
      // NVGPU `device_async_copy` 要求源 memref 在最内层维是 unit stride。
      // one-shot bufferize 常产生带动态 stride/offset 的 strided memref；
      // 这里通过运行时 `stride==1` 检查并配合 reinterpret_cast 到
      //“最后一维静态 stride=1”类型来保护 async-copy。
      auto meta = ib.create<memref::ExtractStridedMetadataOp>(loc, tileIn);
      auto strides = meta.getStrides();
      if (strides.size() != 2)
        return;
      Value stride0 = *strides.begin();
      Value stride1 = *std::next(strides.begin(), 1);
      Value one = ib.create<arith::ConstantIndexOp>(loc, 1);
      Value isUnit =
          ib.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, stride1, one);

      auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                           {ShapedType::kDynamic, 1});
      auto asyncSrcTy = MemRefType::get(tileTy.getShape(), tileTy.getElementType(),
                                        layout, tileTy.getMemorySpace());
      Value tileM = ib.create<arith::ConstantIndexOp>(loc, tileTy.getShape()[0]);
      auto asyncSrc = memref::ReinterpretCastOp::create(
          ib, loc, asyncSrcTy, tileIn, meta.getOffset(), ValueRange{tileM, tileN},
          ValueRange{stride0, one});

      auto ifOp = ib.create<scf::IfOp>(loc, isUnit, /*withElse=*/true);
      {
        OpBuilder tb(ifOp.thenBlock(), ifOp.thenBlock()->begin());
        (void)buildStagingLoop(tb, asyncSrc.getResult(), /*useAsync=*/true);
      }
      {
        OpBuilder eb(ifOp.elseBlock(), ifOp.elseBlock()->begin());
        (void)buildStagingLoop(eb, tileIn, /*useAsync=*/false);
      }
      stagingScopeOp = ifOp.getOperation();
      ib.setInsertionPointAfter(ifOp);
    } else {
      scf::ForOp stagingFor = buildStagingLoop(ib, tileIn, wantsAsyncCopy);
      stagingScopeOp = stagingFor.getOperation();
      ib.setInsertionPointAfter(stagingFor);
    }

    auto barrier = ib.create<gpu::BarrierOp>(loc);
    barrier->setAttr("welder.keep_barrier", keepBarrier);

    // 将源自 tileIn 的 subview 改写为源自 wgView，同时跳过 staging 操作。
    launch.walk([&](memref::SubViewOp sv) {
      Operation *op = sv.getOperation();
      if (!op || !op->getParentOfType<gpu::LaunchOp>())
        return;
      if (stagingScopeOp &&
          (stagingScopeOp->isProperAncestor(op) || op == stagingScopeOp))
        return;
      if (sv.getSource() != tileIn)
        return;
      sv->setOperand(0, wgView.getResult());
    });

    if (!enableInPlace2DReuse)
      return;

    // 尽力复用：把同形状、launch 内局部的 2D 中间结果（如 exp_sub/sq）
    // 重定向到已 staging 的 wgView。
    //
    // 该逻辑面向“行归约链”模式：中间结果产生后，原输入通常不再被读取。
    // 将仍残留的同 tile 形状 2D 中间缓冲区（仅属于本 launch，且不返回）
    // 重写为使用 staged workgroup tile，即便中间 producer 已被原地/CSE 处理，
    // 后续消费者（如下一阶段行归约）仍可直接从 shared 读取。
    Value tileBase = tileBaseMemref;
    launch.walk([&](memref::SubViewOp sv) {
      Operation *op = sv.getOperation();
      if (!op || !op->getParentOfType<gpu::LaunchOp>())
        return;
      if (stagingScopeOp &&
          (stagingScopeOp->isProperAncestor(op) || op == stagingScopeOp))
        return;
      Value base = stripToBaseMemref(sv.getSource());
      if (!base)
        return;
      if (base == tileBase)
        return;
      auto baseTy = dyn_cast<MemRefType>(base.getType());
      if (!baseTy || baseTy.getRank() != 2 || !baseTy.hasStaticShape())
        return;
      if (isWorkgroupMemoryType(baseTy))
        return;
      if (baseTy.getShape() != tileTy.getShape())
        return;
      if (baseTy.getElementType() != tileTy.getElementType())
        return;
      auto unique = findUniqueLaunchUser(base);
      if (!unique || *unique != launch)
        return;
      if (isReturnedByFunc(base, func))
        return;
      sv->setOperand(0, wgView.getResult());
    });
  });

  // 将来源重定向到 workgroup 内存后，修正 subview 结果类型。
  fixSubviewResultTypesAfterWorkgroupPromotion(module);
}


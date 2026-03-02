#pragma once

// 将简单的逐元素平方差融合进行归约：
// `sq(x, mean) -> sumsq` 变为 `sum((x-mean)^2)`。
// 这样可避免物化中间 2D scratch 缓冲区。
static void fuseSquareIntoRowReduction(ModuleOp module,
                                       bool enableMeanScaleFusion) {
  if (!module)
    return;

  const bool debug = (std::getenv("WELDER_DEBUG_FUSE_SQUARE") != nullptr);
  auto dbg = [&](const char *msg) {
    if (!debug)
      return;
    llvm::errs() << "[fuseSquare] " << msg << "\n";
    llvm::errs().flush();
  };

  struct FusePlan {
    linalg::GenericOp cons;
    linalg::GenericOp prod;
    linalg::GenericOp meanProd;
    Attribute meanScaleAttr;
  };

  SmallVector<FusePlan, 8> plans;
  llvm::SmallDenseSet<const void *, 32> plannedOps;

  dbg("walk.start");
  module.walk([&](linalg::GenericOp cons) {
    if (debug) {
      llvm::errs() << "[fuseSquare] cons@" << cons.getOperation() << "\n";
      llvm::errs().flush();
    }
    if (!cons->hasAttr("welder.row_reduction"))
      return;

    if (cons.getNumDpsInputs() != 1 || cons.getNumDpsInits() != 1)
      return;

    Value consIn = cons.getDpsInputOperand(0)->get();
    Value consOut = cons.getDpsInitOperand(0)->get();
    Value consInBase = stripToBaseMemref(consIn);
    if (!consInBase)
      return;

    // 查找唯一一个写入 consInBase 的 2D 逐元素 producer。
    linalg::GenericOp prod;
    for (linalg::GenericOp gen : collectGenericOpsWritingToBase(consInBase)) {
      if (!gen || gen == cons)
        continue;
      if (plannedOps.contains(gen.getOperation()))
        continue;
      if (!gen->hasAttr("welder.elementwise_nd"))
        continue;
      if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1)
        continue;
      // 这里避免直接依赖 getNumLoops/getIteratorTypesArray；
      // 仅在 indexing maps 全为 2-D 时将其视为 2D 逐元素算子。
      auto maps = gen.getIndexingMapsArray();
      if (maps.size() != gen.getNumDpsInputs() + gen.getNumDpsInits())
        continue;
      bool is2D = true;
      for (AffineMap m : maps) {
        if (!m || m.getNumDims() != 2) {
          is2D = false;
          break;
        }
      }
      if (!is2D)
        continue;
      if (prod) {
        // 出现多个 producer，直接放弃融合。
        prod = nullptr;
        return;
      }
      prod = gen;
    }
    if (!prod)
      return;

    // 模式匹配：`out = (x - mean) * (x - mean)`。
    Block &body = prod.getRegion().front();
    if (body.getNumArguments() != 3)
      return;
    auto *term = body.getTerminator();
    auto yield = dyn_cast<linalg::YieldOp>(term);
    if (!yield || yield.getNumOperands() != 1)
      return;

    // 收集计算体中的操作（不含 yield）。
    SmallVector<Operation *, 4> ops;
    for (Operation &op : body.getOperations()) {
      if (&op == term)
        break;
      ops.push_back(&op);
    }
    if (ops.size() != 2)
      return;

    auto subf = dyn_cast<arith::SubFOp>(ops[0]);
    auto mulf = dyn_cast<arith::MulFOp>(ops[1]);
    if (!subf || !mulf)
      return;
    if (mulf.getLhs() != subf.getResult() || mulf.getRhs() != subf.getResult())
      return;
    if (yield.getOperand(0) != mulf.getResult())
      return;

    // 构建新的融合行归约，直接读取 `(x, mean)`。
    OpBuilder b(cons);
    Location loc = cons.getLoc();

    auto maps = cons.getIndexingMapsArray();
    if (maps.size() != 2)
      return;
    AffineMap inMap = maps[0];
    AffineMap outMap = maps[1];
    SmallVector<AffineMap, 3> newMaps{inMap, outMap, outMap};

    Value xIn = prod.getDpsInputOperand(0)->get();
    Value meanIn = prod.getDpsInputOperand(1)->get();

    // 可选：把 mean 缩放（`mean = sum * cst`）一并融合到归约中，
    // 直接读取 sum 缓冲区，避免额外的 1D 逐元素阶段。
    linalg::GenericOp meanProd;
    Attribute meanScaleAttr;
    Value meanBase = stripToBaseMemref(meanIn);
    if (meanBase && enableMeanScaleFusion) {
      for (linalg::GenericOp gen : collectGenericOpsWritingToBase(meanBase)) {
        if (!gen || gen == cons || gen == prod)
          continue;
        if (plannedOps.contains(gen.getOperation()))
          continue;
        if (!gen->hasAttr("welder.elementwise_1d"))
          continue;
        if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
          continue;
        if (stripToBaseMemref(gen.getDpsInitOperand(0)->get()) != meanBase)
          continue;

        // 匹配模式：`out = in * cst`（或 `cst * in`）。
        Block &b = gen.getRegion().front();
        auto *t = b.getTerminator();
        auto y = dyn_cast<linalg::YieldOp>(t);
        if (!y || y.getNumOperands() != 1)
          continue;
        SmallVector<Operation *, 4> ops2;
        for (Operation &op : b.getOperations()) {
          if (&op == t)
            break;
          ops2.push_back(&op);
        }
        if (ops2.size() != 2)
          continue;
        auto cstOp = dyn_cast<arith::ConstantOp>(ops2[0]);
        auto mulOp = dyn_cast<arith::MulFOp>(ops2[1]);
        if (!cstOp || !mulOp)
          continue;
        Value inArg = b.getArgument(0);
        if (!((mulOp.getLhs() == inArg && mulOp.getRhs() == cstOp.getResult()) ||
              (mulOp.getRhs() == inArg && mulOp.getLhs() == cstOp.getResult())))
          continue;
        if (y.getOperand(0) != mulOp.getResult())
          continue;

        // 确保 meanBase 只有一个消费者（即 sq producer）。
        int consumers = countGenericOpsReadingBase(meanBase);
        if (consumers != 1)
          break;

        meanProd = gen;
        meanScaleAttr = cstOp.getValueAttr();
        break;
      }
    }

    // 延迟执行 IR 变更，避免破坏 MLIR walk 迭代器。
    FusePlan p;
    p.cons = cons;
    p.prod = prod;
    p.meanProd = meanProd;
    p.meanScaleAttr = meanScaleAttr;
    plans.push_back(std::move(p));
    plannedOps.insert(cons.getOperation());
    plannedOps.insert(prod.getOperation());
    if (meanProd)
      plannedOps.insert(meanProd.getOperation());
  });
  dbg("walk.end");

  if (plans.empty())
    return;

  if (debug) {
    llvm::errs() << "[fuseSquare] plans=" << plans.size() << "\n";
    llvm::errs().flush();
  }

  dbg("apply.start");
  SmallVector<linalg::GenericOp, 8> toErase;
  for (FusePlan &p : plans) {
    if (p.prod)
      toErase.push_back(p.prod);
    if (p.meanProd)
      toErase.push_back(p.meanProd);
  }

  for (FusePlan &p : plans) {
    if (debug) {
      llvm::errs() << "[fuseSquare] apply cons@" << p.cons.getOperation()
                   << " prod@" << p.prod.getOperation() << "\n";
      llvm::errs().flush();
    }
    if (!p.cons || !p.prod)
      continue;
    if (!p.cons->getParentOp() || !p.prod->getParentOp())
      continue;

    linalg::GenericOp cons = p.cons;
    linalg::GenericOp prod = p.prod;

    Value consOut = cons.getDpsInitOperand(0)->get();
    Value xIn = prod.getDpsInputOperand(0)->get();
    Value meanIn = prod.getDpsInputOperand(1)->get();

    auto maps = cons.getIndexingMapsArray();
    if (maps.size() != 2)
      continue;
    AffineMap inMap = maps[0];
    AffineMap outMap = maps[1];
    SmallVector<AffineMap, 3> newMaps{inMap, outMap, outMap};
    // 行归约的 iterator_types 固定为 `["parallel","归约"]`。
    SmallVector<mlir::utils::IteratorType, 2> iterTypes = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
    };

    SmallVector<Value, 2> newInputs;
    if (p.meanProd) {
      Value sumIn = p.meanProd.getDpsInputOperand(0)->get();
      newInputs = {xIn, sumIn};
    } else {
      newInputs = {xIn, meanIn};
    }
    SmallVector<Value, 1> newOutputs{consOut};

    OpBuilder b(cons);
    Location loc = cons.getLoc();
    auto newCons = b.create<linalg::GenericOp>(
        loc, /*resultTypes=*/TypeRange{}, newInputs, newOutputs, newMaps,
        iterTypes,
        [&](OpBuilder &nb, Location loc, ValueRange args) {
          Value inX = args[0];
          Value inMean = args[1];
          Value out = args[2];
          if (p.meanProd && p.meanScaleAttr) {
            auto typed = dyn_cast<TypedAttr>(p.meanScaleAttr);
            if (typed) {
              Value cst = nb.create<arith::ConstantOp>(loc, typed);
              inMean = nb.create<arith::MulFOp>(loc, inMean, cst);
            }
          }
          Value diff = nb.create<arith::SubFOp>(loc, inX, inMean);
          Value sq = nb.create<arith::MulFOp>(loc, diff, diff);
          Value sum = nb.create<arith::AddFOp>(loc, sq, out);
          nb.create<linalg::YieldOp>(loc, sum);
        });
    // 仅保留最小必要的 Welder 标签；结构属性
    //（indexing_maps/iterator_types）由 builder 生成结果保持。
    auto copyIfPresent = [&](StringRef key) {
      if (Attribute a = cons->getAttr(key))
        newCons->setAttr(key, a);
    };
    copyIfPresent("welder.kernel_id");
    copyIfPresent("welder.kernel_root");
    copyIfPresent("welder.kernel_producer");
    copyIfPresent("welder.thread_fuse_into");
    copyIfPresent("welder.thread_fuse_into_operand");
    copyIfPresent("welder.node_id");
    copyIfPresent("welder.row_reduction");
    copyIfPresent("welder.target");
    cons.erase();
  }

  dbg("erase_producers.start");
  // 删除 producer，并清理无用 scratch 缓冲区。
  for (linalg::GenericOp prod : toErase) {
    if (!prod)
      continue;
    if (!prod->getParentOp())
      continue;
    Value out = prod.getDpsInitOperand(0)->get();
    Value base = stripToBaseMemref(out);
    prod.erase();
    if (!base || !base.use_empty())
      continue;
    if (auto sub = base.getDefiningOp<memref::SubViewOp>()) {
      base = stripToBaseMemref(sub.getSource());
      sub.erase();
    }
    if (!base || !base.use_empty())
      continue;
    if (auto view = base.getDefiningOp<memref::ViewOp>()) {
      Value src = view.getSource();
      view.erase();
      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        if (alloc.getResult().use_empty())
          alloc.erase();
      }
    } else if (auto alloc = base.getDefiningOp<memref::AllocOp>()) {
      if (alloc.getResult().use_empty())
        alloc.erase();
    }
  }
}


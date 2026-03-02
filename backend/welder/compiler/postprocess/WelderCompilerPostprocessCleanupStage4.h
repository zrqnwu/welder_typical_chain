#pragma once

// 删除冗余 barrier：
// - barrier 后面紧跟另一个 barrier；
// - barrier 是 gpu.launch 内最后一个非 terminator 操作。
// 保留带 `welder.keep_barrier` 标记的 barrier。
static void removeRedundantBarriers(ModuleOp module) {
  if (!module)
    return;

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
      auto barrier = dyn_cast<gpu::BarrierOp>(&op);
      if (!barrier)
        continue;
      if (barrier->hasAttr("welder.keep_barrier"))
        continue;

      Operation *next = barrier->getNextNode();
      // 删除紧邻 terminator 之前的 barrier。
      if (next && next->hasTrait<OpTrait::IsTerminator>()) {
        barrier.erase();
        continue;
      }
      // 删除连续且未标记 keep 的 barrier。
      if (next && isa<gpu::BarrierOp>(next)) {
        barrier.erase();
        continue;
      }
    }
  });
}

// 面向 1D 广播源的行归约链最小 staging 修复：
// 确保产生 1D 广播缓冲区的阶段在融合 gpu.launch 内先于 2D 消费者执行。
static void reorderBroadcast1DProducersBefore2DConsumers(ModuleOp module) {
  if (!module)
    return;

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();

    llvm::SmallVector<Operation *, 256> ops;
    ops.reserve(body.getOperations().size());
    llvm::DenseMap<Operation *, int64_t> opIndex;
    int64_t idx = 0;
    for (Operation &op : body) {
      ops.push_back(&op);
      opIndex[&op] = idx++;
    }

    // 找到以 1D 缓冲区为输入（广播）的 2D 逐元素算子。
    // 同时记录顶层锚点（算子本身或其外层 if）。
    linalg::GenericOp consumer;
    scf::IfOp consumerIf;
    for (Operation *op : ops) {
      if (auto gen = dyn_cast<linalg::GenericOp>(op)) {
        if (!gen->hasAttr("welder.elementwise_nd"))
          continue;
        for (Value in : gen.getDpsInputs()) {
          auto mt = dyn_cast<MemRefType>(in.getType());
          if (mt && mt.getRank() == 1) {
            consumer = gen;
            break;
          }
        }
        if (consumer)
          break;
      }

      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp)
        continue;
      linalg::GenericOp found;
      ifOp.walk([&](linalg::GenericOp gen) {
        if (!gen->hasAttr("welder.elementwise_nd"))
          return WalkResult::advance();
        for (Value in : gen.getDpsInputs()) {
          auto mt = dyn_cast<MemRefType>(in.getType());
          if (mt && mt.getRank() == 1) {
            found = gen;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (found) {
        consumer = found;
        consumerIf = ifOp;
        break;
      }
    }
    if (!consumer)
      return;

    Operation *consumerAnchor =
        consumerIf ? consumerIf.getOperation() : consumer.getOperation();
    int64_t consumerIndex = opIndex.lookup(consumerAnchor);
    if (consumerIndex < 0)
      return;

    // 以第一个 1D 输入缓冲区作为广播源进行 staging。
    Value broadcastBuf;
    for (Value in : consumer.getDpsInputs()) {
      auto mt = dyn_cast<MemRefType>(in.getType());
      if (mt && mt.getRank() == 1) {
        broadcastBuf = stripToBaseMemref(in);
        break;
      }
    }
    if (!broadcastBuf)
      return;

    // 查找位于消费者之后、带谓词保护的 `逐元素_1d` 写入者（写入 broadcastBuf）。
    scf::IfOp writerIf;
    linalg::GenericOp writerGen;
    for (Operation *op : ops) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp)
        continue;
      if (opIndex.lookup(op) <= consumerIndex)
        continue;
      linalg::GenericOp found;
      ifOp.walk([&](linalg::GenericOp gen) {
        if (!gen->hasAttr("welder.elementwise_1d"))
          return WalkResult::advance();
        if (gen.getNumDpsInits() < 1)
          return WalkResult::advance();
        Value out = gen.getDpsInitOperand(0)->get();
        if (stripToBaseMemref(out) == broadcastBuf) {
          found = gen;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!found)
        continue;
      writerIf = ifOp;
      writerGen = found;
      break;
    }
    if (!writerIf || !writerGen)
      return;

    // 定位产生 `逐元素_1d` 输入的 combine-reduce。
    if (writerGen.getNumDpsInputs() < 1)
      return;
    Value sumBuf = stripToBaseMemref(writerGen.getDpsInputOperand(0)->get());
    if (!sumBuf)
      return;

    scf::IfOp combineIf;
    linalg::ReduceOp combineReduce;
    for (int64_t i = opIndex.lookup(writerIf); i >= 0; --i) {
      auto ifOp = dyn_cast<scf::IfOp>(ops[static_cast<size_t>(i)]);
      if (!ifOp)
        continue;
      linalg::ReduceOp foundReduce;
      ifOp.walk([&](linalg::ReduceOp red) {
        if (red.getNumDpsInits() < 1)
          return WalkResult::advance();
        Value out = red.getDpsInitOperand(0)->get();
        if (stripToBaseMemref(out) == sumBuf) {
          foundReduce = red;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!foundReduce)
        continue;
      combineIf = ifOp;
      combineReduce = foundReduce;
      break;
    }
    if (!combineIf || !combineReduce)
      return;

    Value scratchView =
        stripToBaseMemref(combineReduce.getDpsInputOperand(0)->get());
    if (!scratchView)
      return;
    Operation *scratchViewDef = scratchView.getDefiningOp();
    if (!scratchViewDef)
      return;

    Operation *stageStart = scratchViewDef;
    if (auto viewOp = dyn_cast<memref::ViewOp>(scratchViewDef)) {
      Value src = viewOp.getSource();
      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        if (alloc->getBlock() == &body &&
            opIndex.lookup(alloc) < opIndex.lookup(scratchViewDef))
          stageStart = alloc;
      }
    }

    Operation *stageEnd = writerIf;
    if (auto barrier =
            dyn_cast_or_null<gpu::BarrierOp>(writerIf->getNextNode())) {
      stageEnd = barrier;
    }

    int64_t startIndex = opIndex.lookup(stageStart);
    int64_t endIndex = opIndex.lookup(stageEnd);
    if (startIndex < 0 || endIndex < 0 || startIndex > endIndex)
      return;
    if (startIndex <= consumerIndex)
      return;

    // 按原顺序将 staging 阶段移动到消费者之前。
    Operation *insertBefore = consumerAnchor;
    Operation *op = stageStart;
    while (op) {
      Operation *next = op->getNextNode();
      op->moveBefore(insertBefore);
      if (op == stageEnd)
        break;
      op = next;
    }
  });
}

static void eraseHostDuplicatesOfFusedLaunchOps(ModuleOp module) {
  if (!module)
    return;

  module.walk([&](func::FuncOp func) {
    llvm::DenseMap<uint64_t, Operation *> inLaunchOp;

    func.walk([&](gpu::LaunchOp launch) {
      launch.walk([&](linalg::LinalgOp op) {
        auto kidAttr = dyn_cast_or_null<IntegerAttr>(op->getAttr("welder.kernel_id"));
        auto nidAttr = dyn_cast_or_null<IntegerAttr>(op->getAttr("welder.node_id"));
        if (!kidAttr || !nidAttr)
          return;
        uint64_t kid = static_cast<uint64_t>(kidAttr.getInt());
        uint64_t nid = static_cast<uint64_t>(nidAttr.getInt());
        uint64_t key = (kid << 32) | (nid & 0xffffffffULL);
        inLaunchOp.try_emplace(key, op.getOperation());
      });
    });

    llvm::SmallVector<Operation *, 16> toErase;
    func.walk([&](linalg::LinalgOp op) {
      if (op->getParentOfType<gpu::LaunchOp>())
        return;
      auto kidAttr = dyn_cast_or_null<IntegerAttr>(op->getAttr("welder.kernel_id"));
      auto nidAttr = dyn_cast_or_null<IntegerAttr>(op->getAttr("welder.node_id"));
      if (!kidAttr || !nidAttr)
        return;
      uint64_t kid = static_cast<uint64_t>(kidAttr.getInt());
      uint64_t nid = static_cast<uint64_t>(nidAttr.getInt());
      uint64_t key = (kid << 32) | (nid & 0xffffffffULL);
      auto it = inLaunchOp.find(key);
      if (it == inLaunchOp.end())
        return;

      // 若重复算子写入的缓冲区与 launch 内克隆体不同，则在删除前先重连消费者，
      // 让其改读 launch 内缓冲区。
      //
      // 这是一个面向多消费者链（如 Softmax）的尽力修复：
      // transform library 可能会把同一节点克隆进融合 gpu.launch，
      // 但其余消费者仍读取原始 host 缓冲区。
      auto insideLinalg = dyn_cast_or_null<linalg::LinalgOp>(it->second);
      if (insideLinalg) {
        ValueRange outs0 = op.getDpsInits();
        ValueRange outs1 = insideLinalg.getDpsInits();
        if (outs0.size() == outs1.size()) {
          for (size_t i = 0; i < outs0.size(); ++i) {
            Value base0 = stripToBaseMemref(outs0[i]);
            Value base1 = stripToBaseMemref(outs1[i]);
            if (!base0 || !base1 || base0 == base1)
              continue;
            auto mt0 = dyn_cast<MemRefType>(base0.getType());
            auto mt1 = dyn_cast<MemRefType>(base1.getType());
            if (!mt0 || !mt1)
              continue;
            if (mt0.getRank() != mt1.getRank())
              continue;
            if (mt0.getElementType() != mt1.getElementType())
              continue;
            if (mt0.hasStaticShape() && mt1.hasStaticShape() &&
                mt0.getShape() != mt1.getShape())
              continue;

            Operation *def0 = base0.getDefiningOp();
            Operation *def1 = base1.getDefiningOp();
            if (!def0 || !def1)
              continue;
            if (def0->getParentOfType<gpu::LaunchOp>() ||
                def1->getParentOfType<gpu::LaunchOp>())
              continue;
            if (def0->getBlock() == def1->getBlock() &&
                !def1->isBeforeInBlock(def0))
              continue;
            base0.replaceAllUsesWith(base1);
          }
        }
      }

      toErase.push_back(op.getOperation());
    });

    for (Operation *op : toErase)
      op->erase();

    // 尽力清理：删除显然为空的 forall/if 包裹层。
    func.walk([&](scf::ForallOp forall) {
      if (forall.getNumResults() != 0)
        return;
      Block *b = forall.getBody();
      if (!b)
        return;
      if (b->getOperations().size() == 1)
        forall.erase();
    });
    func.walk([&](scf::IfOp ifOp) {
      if (ifOp.getNumResults() != 0)
        return;
      bool hasNonTerm = false;
      for (Operation &op : ifOp.getThenRegion().front().getOperations()) {
        if (!op.hasTrait<OpTrait::IsTerminator>()) {
          hasNonTerm = true;
          break;
        }
      }
      if (hasNonTerm)
        return;
      if (ifOp.elseBlock() && ifOp.getElseRegion().hasOneBlock()) {
        for (Operation &op : ifOp.getElseRegion().front().getOperations()) {
          if (!op.hasTrait<OpTrait::IsTerminator>())
            return;
        }
      }
      ifOp.erase();
    });
  });
}

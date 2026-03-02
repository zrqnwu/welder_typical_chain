#pragma once

// 带谓词的 if 区域内 barrier 对 GPU 不安全（并非所有线程都会到达）。
// 将 then 区域尾部 barrier 提升到 if 之后。
static void hoistPredicatedBarriers(ModuleOp module) {
  if (!module)
    return;

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
      auto ifOp = dyn_cast<scf::IfOp>(&op);
      if (!ifOp)
        continue;
      Block *thenBlock = ifOp.thenBlock();
      if (!thenBlock)
        continue;
      // 仅在不存在 else 区域时提升（若 else 有副作用，无法保证线程一致性）。
      if (ifOp.elseBlock() && !ifOp.elseBlock()->empty())
        continue;

      gpu::BarrierOp barrier;
      for (Operation &inner : thenBlock->getOperations()) {
        barrier = dyn_cast<gpu::BarrierOp>(&inner);
        if (barrier)
          break;
      }
      if (!barrier)
        continue;

      Operation *afterBarrier = barrier->getNextNode();
      if (afterBarrier && !afterBarrier->hasTrait<OpTrait::IsTerminator>())
        continue;

      OpBuilder b(ifOp);
      b.setInsertionPointAfter(ifOp);
      auto newBarrier = b.create<gpu::BarrierOp>(barrier.getLoc());
      if (auto keep = barrier->getAttrOfType<UnitAttr>("welder.keep_barrier"))
        newBarrier->setAttr("welder.keep_barrier", keep);
      barrier.erase();
    }
  });
}

// 若 `scf.if`（无 else）中间含 barrier，则拆成两个 if，中间放统一 barrier：
// 即：`if (c) { A; barrier; B }  ->  if (c) { A } ; barrier ; if (c) { B }`
// 并将 B 依赖的简单 SSA 定义提升到父 block。
static void splitPredicatedBarrierStages(ModuleOp module) {
  if (!module)
    return;

  auto isHoistable = [&](Operation *op) -> bool {
    return isa<memref::AllocOp, memref::ViewOp, memref::SubViewOp,
               memref::CastOp, memref::ReinterpretCastOp, arith::ConstantOp,
               arith::ConstantIndexOp, affine::AffineApplyOp>(op);
  };

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    SmallVector<scf::IfOp, 8> ifOps;
    for (Operation &op : body.getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        ifOps.push_back(ifOp);
    }
    for (scf::IfOp ifOp : ifOps) {
      if (ifOp.elseBlock() && !ifOp.elseBlock()->empty())
        continue;
      Block *thenBlock = ifOp.thenBlock();
      if (!thenBlock)
        continue;

      // 在 then 区域中查找“非尾部” barrier。
      gpu::BarrierOp barrier;
      for (Operation &inner : thenBlock->getOperations()) {
        auto b = dyn_cast<gpu::BarrierOp>(&inner);
        if (!b)
          continue;
        Operation *after = b->getNextNode();
        if (after && !after->hasTrait<OpTrait::IsTerminator>()) {
          barrier = b;
          break;
        }
      }
      if (!barrier)
        continue;

      // 收集 barrier 之后的操作。
      SmallVector<Operation *, 16> opsAfter;
      for (Operation *cur = barrier->getNextNode();
           cur && !cur->hasTrait<OpTrait::IsTerminator>();
           cur = cur->getNextNode()) {
        opsAfter.push_back(cur);
      }
      if (opsAfter.empty())
        continue;

      // 确定 barrier 前必须提升的定义。
      llvm::SmallDenseSet<Operation *, 16> toHoist;
      bool ok = true;
      for (Operation *cur : opsAfter) {
        for (Value v : cur->getOperands()) {
          Operation *def = v.getDefiningOp();
          if (!def || def->getBlock() != thenBlock)
            continue;
          if (def->isBeforeInBlock(barrier)) {
            if (!isHoistable(def)) {
              ok = false;
              break;
            }
            toHoist.insert(def);
          }
        }
        if (!ok)
          break;
      }
      if (!ok)
        continue;

      // 扩展提升集合，补齐被提升操作的依赖。
      SmallVector<Operation *, 16> worklist;
      for (Operation *opH : toHoist)
        worklist.push_back(opH);
      while (!worklist.empty()) {
        Operation *def = worklist.pop_back_val();
        for (Value v : def->getOperands()) {
          Operation *d = v.getDefiningOp();
          if (!d || d->getBlock() != thenBlock)
            continue;
          if (!d->isBeforeInBlock(barrier))
            continue;
          if (toHoist.contains(d))
            continue;
          if (!isHoistable(d)) {
            ok = false;
            break;
          }
          toHoist.insert(d);
          worklist.push_back(d);
        }
        if (!ok)
          break;
      }
      if (!ok)
        continue;

      // 提升所需定义（保持原顺序）。
      for (Operation &inner :
           llvm::make_early_inc_range(thenBlock->getOperations())) {
        if (toHoist.contains(&inner))
          inner.moveBefore(ifOp);
      }

      // 在第一个 if 之后插入统一 barrier。
      OpBuilder b(ifOp);
      b.setInsertionPointAfter(ifOp);
      auto newBarrier = b.create<gpu::BarrierOp>(barrier.getLoc());
      if (auto keep = barrier->getAttrOfType<UnitAttr>("welder.keep_barrier"))
        newBarrier->setAttr("welder.keep_barrier", keep);

      // 创建第二个 if，并将后半段操作移入其中。
      b.setInsertionPointAfter(newBarrier);
      auto newIf = b.create<scf::IfOp>(ifOp.getLoc(), ifOp.getCondition(), false);
      Block *newThen = newIf.thenBlock();
      Operation *newTerm = newThen->getTerminator();
      for (Operation *cur : opsAfter)
        cur->moveBefore(newTerm);

      barrier.erase();
    }
  });
}


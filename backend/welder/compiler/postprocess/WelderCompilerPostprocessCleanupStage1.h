#pragma once

static void hoistWorkgroupAllocs(ModuleOp module) {
  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    Operation *insertPt = &body.front();
    llvm::SmallVector<memref::AllocOp, 8> allocs;
    llvm::SmallVector<memref::DeallocOp, 8> deallocs;
    auto workgroupAttr =
        IntegerAttr::get(IntegerType::get(launch.getContext(), 64), 3);

    launch.walk([&](memref::AllocOp alloc) {
      auto memrefType = dyn_cast<MemRefType>(alloc.getType());
      if (!memrefType)
        return;
      if (memrefType.getMemorySpace() != workgroupAttr)
        return;
      if (alloc.getNumOperands() != 0)
        return;
      allocs.push_back(alloc);
    });

    launch.walk([&](memref::DeallocOp dealloc) {
      auto memrefType = dyn_cast<MemRefType>(dealloc.getMemref().getType());
      if (!memrefType)
        return;
      if (memrefType.getMemorySpace() != workgroupAttr)
        return;
      deallocs.push_back(dealloc);
    });

    for (memref::AllocOp alloc : allocs) {
      if (alloc->getBlock() != &body)
        alloc->moveBefore(insertPt);
    }
    for (memref::DeallocOp dealloc : deallocs) {
      dealloc.erase();
    }
  });
}

static void insertBarrierAfterCombiningReductions(ModuleOp module,
                                                  bool skipCombineBarrier) {
  if (!module)
    return;
  if (skipCombineBarrier)
    return;

  MLIRContext *ctx = module.getContext();
  UnitAttr keep = UnitAttr::get(ctx);

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
      auto ifOp = dyn_cast<scf::IfOp>(&op);
      if (!ifOp)
        continue;

      bool hasCombiningReduce = false;
      ifOp.walk([&](linalg::ReduceOp) {
        hasCombiningReduce = true;
        return WalkResult::interrupt();
      });
      if (!hasCombiningReduce)
        continue;

      Operation *next = op.getNextNode();
      if (auto b = dyn_cast_or_null<gpu::BarrierOp>(next)) {
        b->setAttr("welder.keep_barrier", keep);
        continue;
      }

      OpBuilder b(ifOp);
      b.setInsertionPointAfter(ifOp);
      auto barrier = b.create<gpu::BarrierOp>(ifOp.getLoc());
      barrier->setAttr("welder.keep_barrier", keep);
    }
  });
}

static Value stripToBaseMemref(Value v) {
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

static SmallVector<Value, 16> collectViewLikeDerivedMemrefs(Value base) {
  SmallVector<Value, 16> out;
  if (!base)
    return out;

  llvm::SmallDenseSet<const void *, 32> seen;
  SmallVector<Value, 16> worklist;
  worklist.push_back(base);
  seen.insert(base.getAsOpaquePointer());

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    out.push_back(cur);

    for (Operation *user : cur.getUsers()) {
      auto pushResult = [&](Value v) {
        if (!v)
          return;
        if (!isa<MemRefType>(v.getType()))
          return;
        const void *k = v.getAsOpaquePointer();
        if (!seen.insert(k).second)
          return;
        worklist.push_back(v);
      };

      if (auto sub = dyn_cast<memref::SubViewOp>(user)) {
        pushResult(sub.getResult());
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(user)) {
        pushResult(cast.getResult());
        continue;
      }
      if (auto rcast = dyn_cast<memref::ReinterpretCastOp>(user)) {
        pushResult(rcast.getResult());
        continue;
      }
      if (auto view = dyn_cast<memref::ViewOp>(user)) {
        pushResult(view.getResult());
        continue;
      }
      if (auto assumed = dyn_cast<memref::AssumeAlignmentOp>(user)) {
        pushResult(assumed.getResult());
        continue;
      }
    }
  }

  return out;
}

static SmallVector<linalg::GenericOp, 4>
collectGenericOpsWritingToBase(Value base) {
  SmallVector<linalg::GenericOp, 4> out;
  if (!base)
    return out;

  llvm::SmallDenseSet<const void *, 64> seenOps;
  for (Value v : collectViewLikeDerivedMemrefs(base)) {
    for (Operation *user : v.getUsers()) {
      auto gen = dyn_cast<linalg::GenericOp>(user);
      if (!gen)
        continue;
      const void *k = gen.getOperation();
      if (!seenOps.insert(k).second)
        continue;

      bool writes = false;
      for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
        if (stripToBaseMemref(gen.getDpsInitOperand(i)->get()) == base) {
          writes = true;
          break;
        }
      }
      if (writes)
        out.push_back(gen);
    }
  }
  return out;
}

static int countGenericOpsReadingBase(Value base) {
  if (!base)
    return 0;
  llvm::SmallDenseSet<const void *, 64> seenOps;
  int c = 0;
  for (Value v : collectViewLikeDerivedMemrefs(base)) {
    for (Operation *user : v.getUsers()) {
      auto gen = dyn_cast<linalg::GenericOp>(user);
      if (!gen)
        continue;
      const void *k = gen.getOperation();
      if (!seenOps.insert(k).second)
        continue;
      for (unsigned i = 0, e = gen.getNumDpsInputs(); i < e; ++i) {
        if (stripToBaseMemref(gen.getDpsInputOperand(i)->get()) == base) {
          ++c;
          break;
        }
      }
    }
  }
  return c;
}

static void insertKeepBarrierAfterPredicatedElementwise1D(ModuleOp module) {
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();
  UnitAttr keep = UnitAttr::get(ctx);

  module.walk([&](gpu::LaunchOp launch) {
    Block &body = launch.getBody().front();
    for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
      auto ifOp = dyn_cast<scf::IfOp>(&op);
      if (!ifOp)
        continue;

      bool hasElemwise1D = false;
      ifOp.walk([&](linalg::GenericOp gen) {
        if (gen->hasAttr("welder.elementwise_1d")) {
          hasElemwise1D = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasElemwise1D)
        continue;

      Operation *next = op.getNextNode();
      if (auto b = dyn_cast_or_null<gpu::BarrierOp>(next)) {
        b->setAttr("welder.keep_barrier", keep);
        continue;
      }

      OpBuilder b(ifOp);
      b.setInsertionPointAfter(ifOp);
      auto barrier = b.create<gpu::BarrierOp>(ifOp.getLoc());
      barrier->setAttr("welder.keep_barrier", keep);
    }
  });
}


#pragma once

// Matmul->Softmax 链修复（对应论文 Figure 2 风格）：
// 当前 generic 融合/切分路径可能复制 matmul/max/exp 链
//（同一 welder.node_id 出现多个克隆），并误让最终 div 阶段读取
// 与 sum 归约不同的 exp 缓冲区。
//
// 该 pass 在每个 gpu.launch 内做规范化，保证：
// - sum 归约消费的 exp tile 缓冲区也作为最终 div 的输入；
// - exp 阶段与 max 归约读取同一个 matmul tile（避免仅为 exp 额外复制 matmul）；
// - 选中的 matmul tile 缓冲区提升到 workgroup(shared) 内存，
//   避免巨大的逐线程本地分配。
static void canonicalizeMatmulSoftmaxSharedReuse(ModuleOp module) {
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;
  UnitAttr keepBarrier = UnitAttr::get(ctx);

  module.walk([&](gpu::LaunchOp launch) {
    linalg::GenericOp divOp;
    launch.walk([&](linalg::GenericOp gen) {
      if (!gen->hasAttr("welder.kernel_root"))
        return;
      if (regionHasDivF(gen))
        divOp = gen;
    });
    if (!divOp)
      return;

    // 找到 sum 行归约阶段（addf），用于定位 exp tile 缓冲区。
    linalg::GenericOp sumRed;
    launch.walk([&](linalg::GenericOp gen) {
      if (!gen->hasAttr("welder.row_reduction"))
        return;
      if (regionHasAddF(gen))
        sumRed = gen;
    });
    if (!sumRed)
      return;

    Value expBase;
    Value sumInput2D;
    for (OpOperand *in : sumRed.getDpsInputOperands()) {
      if (!in)
        continue;
      auto mt = dyn_cast<MemRefType>(in->get().getType());
      if (!mt || mt.getRank() != 2)
        continue;
      sumInput2D = in->get();
      expBase = stripToTypedMemrefBase(sumInput2D);
      break;
    }
    if (!expBase)
      return;
    auto expBaseTy = dyn_cast<MemRefType>(expBase.getType());
    if (!expBaseTy || !isWorkgroupMemoryType(expBaseTy))
      return;

    // 改写 div 分子，统一改为使用 expBase（即 sum 消费的缓冲区）。
    for (OpOperand *in : divOp.getDpsInputOperands()) {
      if (!in)
        continue;
      auto mt = dyn_cast<MemRefType>(in->get().getType());
      if (!mt || mt.getRank() != 2)
        continue;
      Value num = in->get();
      auto sv = num.getDefiningOp<memref::SubViewOp>();
      if (!sv)
        return;

      // 在 div 前新建一个 subview，确保 expBase 在该 use 位置具备支配关系
      //（旧 subview 可能来自更早位置，属于已失效的重复链）。
      OpBuilder b(divOp);
      b.setInsertionPoint(divOp);
      auto srcTy = dyn_cast<MemRefType>(expBase.getType());
      auto oldResTy = dyn_cast<MemRefType>(sv.getResult().getType());
      if (!srcTy || !oldResTy)
        return;
      MemRefType newResTy = memref::SubViewOp::inferRankReducedResultType(
          oldResTy.getShape(), srcTy, sv.getMixedOffsets(), sv.getMixedSizes(),
          sv.getMixedStrides());
      auto newSv = memref::SubViewOp::create(
          b, sv.getLoc(), newResTy, expBase, sv.getMixedOffsets(),
          sv.getMixedSizes(), sv.getMixedStrides());
      in->set(newSv.getResult());
      break;
    }

    // 找到产生 expBase 的 exp 阶段。
    linalg::GenericOp expOp;
    launch.walk([&](linalg::GenericOp gen) {
      if (!regionHasExp(gen))
        return;
      for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
        OpOperand *init = gen.getDpsInitOperand(i);
        if (!init)
          continue;
        if (stripToTypedMemrefBase(init->get()) == expBase) {
          expOp = gen;
          return;
        }
      }
    });
    if (!expOp)
      return;

    // 识别 expOp 消费的 max 缓冲区，以及产生该缓冲区的 max 归约。
    Value maxBase;
    for (OpOperand *in : expOp.getDpsInputOperands()) {
      if (!in)
        continue;
      auto mt = dyn_cast<MemRefType>(in->get().getType());
      if (mt && mt.getRank() == 1) {
        maxBase = stripToTypedMemrefBase(in->get());
        break;
      }
    }
    if (!maxBase)
      return;

    linalg::GenericOp maxRed;
    launch.walk([&](linalg::GenericOp gen) {
      if (!gen->hasAttr("welder.row_reduction"))
        return;
      if (!regionHasMaxF(gen))
        return;
      for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
        OpOperand *init = gen.getDpsInitOperand(i);
        if (!init)
          continue;
        if (stripToTypedMemrefBase(init->get()) == maxBase) {
          maxRed = gen;
          return;
        }
      }
    });
    if (!maxRed) {
      // 典型行归约 lowering 是两段式：
      // (1) `linalg.generic {welder.row_归约}` 写入 2D scratch
      // (2) `linalg.reduce` 合并到最终 1D 缓冲区。
      linalg::ReduceOp maxCombine;
      launch.walk([&](linalg::ReduceOp red) {
        if (!regionHasMaxF(red))
          return;
        if (red.getNumDpsInits() < 1)
          return;
        Value outBase = stripToTypedMemrefBase(red.getDpsInitOperand(0)->get());
        if (outBase == maxBase)
          maxCombine = red;
      });
      if (!maxCombine)
        return;
      if (maxCombine.getNumDpsInputs() < 1)
        return;
      Value scratchBase =
          stripToTypedMemrefBase(maxCombine.getDpsInputOperand(0)->get());
      if (!scratchBase)
        return;
      launch.walk([&](linalg::GenericOp gen) {
        if (maxRed)
          return;
        if (!gen->hasAttr("welder.row_reduction"))
          return;
        if (!regionHasMaxF(gen))
          return;
        for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
          OpOperand *init = gen.getDpsInitOperand(i);
          if (!init)
            continue;
          if (stripToTypedMemrefBase(init->get()) == scratchBase) {
            maxRed = gen;
            return;
          }
        }
      });
      if (!maxRed)
        return;
    }

    Value expInputBaseForMax;
    for (OpOperand *in : maxRed.getDpsInputOperands()) {
      if (!in)
        continue;
      auto mt = dyn_cast<MemRefType>(in->get().getType());
      if (mt && mt.getRank() == 2) {
        expInputBaseForMax = stripToTypedMemrefBase(in->get());
        break;
      }
    }
    if (!expInputBaseForMax)
      return;

    // 追踪与之对应的 pre-cast matmul tile 缓冲区，用于去重重复的 mma/matmul 循环。
    // maxRed 常读 f32（extf 后），而 matmul 输出 tile 往往是 f16。
    Value matmulTileBaseForMax = expInputBaseForMax;
    launch.walk([&](linalg::GenericOp gen) {
      if (!gen || matmulTileBaseForMax != expInputBaseForMax)
        return;
      bool writesExpBase = false;
      for (unsigned i = 0, e = gen.getNumDpsInits(); i < e; ++i) {
        OpOperand *init = gen.getDpsInitOperand(i);
        if (!init)
          continue;
        if (stripToTypedMemrefBase(init->get()) == expInputBaseForMax) {
          writesExpBase = true;
          break;
        }
      }
      if (!writesExpBase)
        return;

      bool hasExtF = false;
      gen->walk([&](arith::ExtFOp) { hasExtF = true; });
      if (!hasExtF)
        return;

      auto expOutTy = dyn_cast<MemRefType>(expInputBaseForMax.getType());
      for (OpOperand *src : gen.getDpsInputOperands()) {
        if (!src)
          continue;
        auto srcTy = dyn_cast<MemRefType>(src->get().getType());
        if (!srcTy || srcTy.getRank() != 2)
          continue;
        Value srcBase = stripToTypedMemrefBase(src->get());
        if (!srcBase || srcBase == expInputBaseForMax)
          continue;
        auto srcBaseTy = dyn_cast<MemRefType>(srcBase.getType());
        if (expOutTy && srcBaseTy &&
            srcBaseTy.getElementType() == expOutTy.getElementType())
          continue;
        matmulTileBaseForMax = srcBase;
        return;
      }
    });

    // 保证 expOp 与 maxRed 读取同一块 matmul tile。
    for (OpOperand *in : expOp.getDpsInputOperands()) {
      if (!in)
        continue;
      auto mt = dyn_cast<MemRefType>(in->get().getType());
      if (!mt || mt.getRank() != 2)
        continue;
      Value curBase = stripToTypedMemrefBase(in->get());
      if (!curBase || curBase == expInputBaseForMax)
        break;
      // 仅当 exp 操作数是 subview（tile）时才改写，确保索引保持不变。
      auto sv = in->get().getDefiningOp<memref::SubViewOp>();
      if (!sv)
        break;
      auto curBaseTy = dyn_cast<MemRefType>(curBase.getType());
      auto maxBaseTy = dyn_cast<MemRefType>(expInputBaseForMax.getType());
      if (!curBaseTy || !maxBaseTy || !curBaseTy.hasStaticShape() ||
          !maxBaseTy.hasStaticShape() || curBaseTy.getRank() != 2 ||
          maxBaseTy.getRank() != 2 || curBaseTy.getShape() != maxBaseTy.getShape() ||
          curBaseTy.getElementType() != maxBaseTy.getElementType())
        break;

      // 在 exp 前创建新 subview，避免原 subview 属于克隆 matmul 链时的支配问题。
      OpBuilder b(expOp);
      b.setInsertionPoint(expOp);
      auto srcTy = dyn_cast<MemRefType>(expInputBaseForMax.getType());
      auto oldResTy = dyn_cast<MemRefType>(sv.getResult().getType());
      if (!srcTy || !oldResTy)
        break;
      MemRefType newResTy = memref::SubViewOp::inferRankReducedResultType(
          oldResTy.getShape(), srcTy, sv.getMixedOffsets(), sv.getMixedSizes(),
          sv.getMixedStrides());
      auto newSv = memref::SubViewOp::create(
          b, sv.getLoc(), newResTy, expInputBaseForMax, sv.getMixedOffsets(),
          sv.getMixedSizes(), sv.getMixedStrides());
      in->set(newSv.getResult());
      break;
    }

    // 尽力做链路去重：仅保留以 `{divOp,sumRed,expOp,maxRed}` 为核心的
    // max->exp->sum 路径，删除同 node_id 的其余重复克隆
    //（Matmul->Softmax 多消费者融合中较常见）。
    auto getNodeId = [&](Operation *op) -> std::optional<int64_t> {
      if (!op)
        return std::nullopt;
      auto nidAttr = dyn_cast_or_null<IntegerAttr>(op->getAttr("welder.node_id"));
      if (!nidAttr)
        return std::nullopt;
      return nidAttr.getInt();
    };

    // 删除同 node_id 的重复 exp 阶段（保留 expOp）。
    if (auto expNid = getNodeId(expOp.getOperation())) {
      llvm::SmallVector<linalg::GenericOp, 4> expClones;
      launch.walk([&](linalg::GenericOp gen) {
        if (gen == expOp)
          return;
        if (!regionHasExp(gen))
          return;
        if (getNodeId(gen.getOperation()) == expNid)
          expClones.push_back(gen);
      });
      for (linalg::GenericOp gen : expClones)
        gen.erase();
    }

    // 删除同 node_id 的重复 max 行归约阶段（保留 maxRed）。
    if (auto maxNid = getNodeId(maxRed.getOperation())) {
      llvm::SmallVector<linalg::GenericOp, 4> maxClones;
      launch.walk([&](linalg::GenericOp gen) {
        if (gen == maxRed)
          return;
        if (!gen->hasAttr("welder.row_reduction"))
          return;
        if (!regionHasMaxF(gen))
          return;
        if (getNodeId(gen.getOperation()) == maxNid)
          maxClones.push_back(gen);
      });
      for (linalg::GenericOp gen : maxClones)
        gen.erase();
    }

    // 删除不产出 maxBase 的多余 max combine reduce。
    llvm::SmallVector<linalg::ReduceOp, 4> deadMaxCombines;
    launch.walk([&](linalg::ReduceOp red) {
      if (!regionHasMaxF(red))
        return;
      if (red.getNumDpsInits() < 1)
        return;
      Value outBase = stripToTypedMemrefBase(red.getDpsInitOperand(0)->get());
      if (outBase && outBase != maxBase)
        deadMaxCombines.push_back(red);
    });
    for (linalg::ReduceOp red : deadMaxCombines)
      red.erase();

    // 删除重复 matmul 链：移除那些写入 2D base 缓冲区（且不同于 matmulTileBaseForMax）
    // 的 K-loop。
    int64_t matmulNodeId = -1;
    launch.walk([&](linalg::MatmulOp mm) {
      if (matmulNodeId >= 0)
        return;
      Value outBase = stripToTypedMemrefBase(mm.getDpsInitOperand(0)->get());
      if (outBase != matmulTileBaseForMax)
        return;
      auto nidAttr = dyn_cast_or_null<IntegerAttr>(mm->getAttr("welder.node_id"));
      if (!nidAttr)
        return;
      matmulNodeId = nidAttr.getInt();
    });
    if (matmulNodeId >= 0) {
      llvm::DenseSet<Operation *> loopsToEraseSet;
      llvm::SmallVector<Operation *, 8> loopsToErase;
      llvm::DenseSet<Value> deadMatmulBases;

      launch.walk([&](linalg::MatmulOp mm) {
        auto nidAttr = dyn_cast_or_null<IntegerAttr>(mm->getAttr("welder.node_id"));
        if (!nidAttr || nidAttr.getInt() != matmulNodeId)
          return;
        Value outBase = stripToTypedMemrefBase(mm.getDpsInitOperand(0)->get());
        if (!outBase || outBase == matmulTileBaseForMax)
          return;
        if (auto forOp = mm->getParentOfType<scf::ForOp>()) {
          Operation *loopOp = forOp.getOperation();
          if (loopsToEraseSet.insert(loopOp).second)
            loopsToErase.push_back(loopOp);
        }
        deadMatmulBases.insert(outBase);
      });

      // 先删除循环体，使其内部 use 一并消失。
      for (Operation *op : loopsToErase) {
        if (op)
          op->erase();
      }

      // 删除写入已失效 matmul 输出缓冲区的初始化 copy。
      llvm::SmallVector<memref::CopyOp, 8> deadCopies;
      launch.walk([&](memref::CopyOp copy) {
        Value dstBase = stripToTypedMemrefBase(copy.getTarget());
        if (deadMatmulBases.contains(dstBase))
          deadCopies.push_back(copy);
      });
      for (memref::CopyOp copy : deadCopies)
        copy.erase();

      // 删除已无用的 dead matmul 输出 alloc。
      for (Value base : deadMatmulBases) {
        if (!base || !base.use_empty())
          continue;
        if (auto alloc = base.getDefiningOp<memref::AllocOp>())
          alloc.erase();
        else if (auto alloca = base.getDefiningOp<memref::AllocaOp>())
          alloca.erase();
      }
    } else {
      // TensorCore 路径下，postbufferize 可能已把 linalg.matmul 改写为
      // nvgpu.mma.sync，此时无法再依赖残留 linalg::MatmulOp 做定位。
      // 去重策略改为：删除包含 mma.sync 且写入其他 2D base 缓冲区
      //（不同于 matmulTileBaseForMax）的 K-loop。
      auto maxBaseTy = dyn_cast<MemRefType>(matmulTileBaseForMax.getType());
      if (maxBaseTy && maxBaseTy.hasStaticShape() && maxBaseTy.getRank() == 2) {
        llvm::DenseMap<Operation *, Value> loopToOutBase;
        launch.walk([&](nvgpu::MmaSyncOp mma) {
          auto forOp = mma->getParentOfType<scf::ForOp>();
          if (!forOp)
            return;
          Operation *loopOp = forOp.getOperation();
          if (!loopOp || loopToOutBase.count(loopOp))
            return;

          Value outBase;
          forOp.walk([&](memref::StoreOp store) {
            if (outBase)
              return;
            Value v = store.getValueToStore();
            auto ex = v.getDefiningOp<vector::ExtractOp>();
            if (!ex)
              return;
            auto defMma = ex.getSource().getDefiningOp<nvgpu::MmaSyncOp>();
            if (defMma != mma)
              return;
            outBase = stripToTypedMemrefBase(store.getMemref());
          });
          if (!outBase)
            return;

          auto outTy = dyn_cast<MemRefType>(outBase.getType());
          if (!outTy || !outTy.hasStaticShape() || outTy.getRank() != 2)
            return;
          if (outTy.getShape() != maxBaseTy.getShape() ||
              outTy.getElementType() != maxBaseTy.getElementType())
            return;

          loopToOutBase[loopOp] = outBase;
        });

        llvm::SmallVector<Operation *, 8> loopsToErase;
        llvm::DenseSet<Value> deadMatmulBases;
        for (auto &kv : loopToOutBase) {
          Operation *loopOp = kv.first;
          Value outBase = kv.second;
          if (!loopOp || !outBase || outBase == matmulTileBaseForMax)
            continue;
          loopsToErase.push_back(loopOp);
          deadMatmulBases.insert(outBase);
        }

        for (Operation *loopOp : loopsToErase) {
          if (loopOp)
            loopOp->erase();
        }

        // 删除读写 dead matmul base 的 copy；这些 copy 往往会在 MMA 改写路径上
        // 物化无效的私有输出 tile。
        llvm::SmallVector<memref::CopyOp, 8> deadCopies;
        launch.walk([&](memref::CopyOp copy) {
          Value dstBase = stripToTypedMemrefBase(copy.getTarget());
          Value srcBase = stripToTypedMemrefBase(copy.getSource());
          if (deadMatmulBases.contains(dstBase) ||
              deadMatmulBases.contains(srcBase))
            deadCopies.push_back(copy);
        });
        for (memref::CopyOp copy : deadCopies)
          copy.erase();

        for (Value base : deadMatmulBases) {
          if (!base || !base.use_empty())
            continue;
          if (auto alloc = base.getDefiningOp<memref::AllocOp>())
            alloc.erase();
          else if (auto alloca = base.getDefiningOp<memref::AllocaOp>())
            alloca.erase();
        }
      }
    }

    // 清理：删除链路剪枝后遗留的无用 view/subview/alloc。
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<Operation *, 32> toErase;
      launch.walk([&](Operation *op) {
        if (!op || op->getNumResults() == 0)
          return;
        bool allUnused = llvm::all_of(op->getResults(),
                                      [](Value r) { return r.use_empty(); });
        if (!allUnused)
          return;
        if (isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
                memref::ReinterpretCastOp, memref::AssumeAlignmentOp,
                memref::AllocOp, memref::AllocaOp>(op))
          toErase.push_back(op);
      });
      for (Operation *op : toErase) {
        op->erase();
        changed = true;
      }
    }

    // 将 matmul tile 缓冲区提升到 workgroup 内存，避免逐线程本地分配，
    // 并使其可被归约消费者共享。
    Value promotedMatmulBase =
        promoteAllocLikeMemrefToWorkgroup(matmulTileBaseForMax);
    fixSubviewResultTypesAfterWorkgroupPromotion(module);

    // 对写入提升后 workgroup 缓冲区的剩余初始化 copy 加谓词保护。
    llvm::SmallVector<memref::CopyOp, 8> initCopies;
    launch.walk([&](memref::CopyOp copy) {
      Value dstBase = stripToTypedMemrefBase(copy.getTarget());
      if (dstBase == promotedMatmulBase)
        initCopies.push_back(copy);
    });
    for (memref::CopyOp copy : initCopies) {
      // 若源缓冲区只是统一 linalg.fill 初始化，则直接填充 workgroup tile，
      // 避免额外的 global->shared 初始化 copy。
      if (tryReplaceInitCopyFromFilledMemref(copy, launch, keepBarrier))
        continue;
      predicateWorkgroupCopyToThread0(copy, keepBarrier);
    }
  });

  fixSubviewResultTypesAfterWorkgroupPromotion(module);
}

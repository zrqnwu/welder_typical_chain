#pragma once

// 路径一实现主体（由 WelderCompilerModeDispatchKernelAttrsBranch.cpp 包装调用）。
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.codegen_from_kernel_attrs")
                  : welder::Tracer::Span();

    // 仅代码生成模式：直接依赖现有属性
    // - `{welder.kernel_root = <i32>}`：标记 kernel 根节点
    // - `{welder.kernel_id = <i32>}`：标记该 kernel 可融合的 producer
    //
    // 该模式用于论文对齐的性能测量流程，可在不重跑切边传播的前提下
    // 直接编译任意连通子图。
    if (forceTileM <= 0 || forceTileN <= 0 || forceTileK <= 0) {
      llvm::errs() << "error: --codegen-from-kernel-attrs requires "
                      "--force-tile-m/--force-tile-n/--force-tile-k\n";
      return 2;
    }

    llvm::DenseMap<int64_t, Operation *> roots;
    module->walk([&](linalg::LinalgOp op) {
      Operation *op0 = op.getOperation();
      if (!op0)
        return;
      if (auto idAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_root")) {
        int64_t id = idAttr.getInt();
        if (!roots.count(id))
          roots[id] = op0;
      }
    });

    if (roots.empty()) {
      llvm::errs() << "error: --codegen-from-kernel-attrs found no linalg ops "
                      "with welder.kernel_root\n";
      return 2;
    }

    llvm::SmallVector<KernelSpec, 8> kernels;
    kernels.reserve(roots.size());

    for (const auto &kv : roots) {
      int64_t kernelId = kv.first;
      Operation *op0 = kv.second;
      auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op0);
      if (!linalgOp)
        continue;

      llvm::SmallVector<int64_t, 8> tileSizes(linalgOp.getNumLoops(), 0);
      llvm::SmallVector<int64_t, 8> ranges = linalgOp.getStaticLoopRanges();
      if (static_cast<int64_t>(ranges.size()) != linalgOp.getNumLoops()) {
        llvm::errs() << "error: kernel_root has non-static loops: "
                     << op0->getName().getStringRef() << "\n";
        return 2;
      }

      int64_t pSeen = 0;
      auto iters = linalgOp.getIteratorTypesArray();
      for (int64_t i = 0; i < linalgOp.getNumLoops(); ++i) {
        if (iters[i] != mlir::utils::IteratorType::parallel)
          continue;

        int64_t full = ranges[i];
        if (full == ShapedType::kDynamic || full <= 0)
          continue;

        int64_t t = 0;
        if (pSeen == 0)
          t = forceTileM;
        else if (pSeen == 1)
          t = forceTileN;
        ++pSeen;

        if (t <= 0)
          continue;
        if (t > full) {
          llvm::errs() << "error: forced tile exceeds loop range for kernel "
                          "root: "
                       << op0->getName().getStringRef() << "\n";
          return 2;
        }
        if (solveOpts.requirePerfectTiling && (full % t != 0)) {
          llvm::errs() << "error: forced tile violates perfect-tiling for "
                          "kernel root: "
                       << op0->getName().getStringRef() << "\n";
          return 2;
        }
        tileSizes[i] = t;
      }

      KernelSpec spec;
      spec.kernelId = kernelId;
      spec.opName = op0->getName().getStringRef().str();
      spec.tileSizes = std::move(tileSizes);
      // 每个 kernel 可选的 block 顺序提示（论文/Welder 对齐）。
      if (auto a = op0->getAttrOfType<BoolAttr>("welder.swap_xy"))
        spec.swapXY = a.getValue();
      kernels.push_back(std::move(spec));
    }

	    llvm::sort(kernels, [](const KernelSpec &a, const KernelSpec &b) {
	      return a.kernelId < b.kernelId;
	    });

	    // `--codegen-from-kernel-attrs` 下使用确定性的 producer 融合顺序
	    // （逆拓扑）：按 sink->source 融合，使多跳 producer 链
	    // （如 matmul->max->exp->sum->div）稳定可融合，避免意外留在 host。
	    auto getStableNodeId = [&](Operation *op, int64_t fallback) -> int64_t {
	      if (!op)
	        return fallback;
	      if (auto idAttr = op->getAttrOfType<IntegerAttr>("welder.node_id"))
	        return idAttr.getInt();
	      return fallback;
	    };
	    auto isNonTrivialLinalg = [&](Operation *op) -> bool {
	      return op && isa<linalg::LinalgOp>(op) && !isa<linalg::FillOp, linalg::CopyOp>(op);
	    };
	    for (KernelSpec &spec : kernels) {
	      const int64_t kid = spec.kernelId;
	      Operation *rootOp = roots.lookup(kid);
	      if (!rootOp || !isNonTrivialLinalg(rootOp))
	        continue;

	      llvm::SmallVector<Operation *, 32> nodes;
	      module->walk([&](linalg::LinalgOp op) {
	        Operation *op0 = op.getOperation();
	        if (!isNonTrivialLinalg(op0))
	          return;
	        auto idAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_id");
	        auto prodAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_producer");
	        auto rootAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_root");
	        if ((idAttr && idAttr.getInt() == kid) ||
	            (prodAttr && prodAttr.getInt() == kid) ||
	            (rootAttr && rootAttr.getInt() == kid)) {
	          nodes.push_back(op0);
	        }
	      });
	      if (!llvm::is_contained(nodes, rootOp))
	        nodes.push_back(rootOp);

	      llvm::DenseMap<Operation *, int> opToIdx;
	      opToIdx.reserve(nodes.size());
	      for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
	        opToIdx[nodes[static_cast<size_t>(i)]] = i;

	      llvm::SmallVector<int, 32> indeg(nodes.size(), 0);
	      std::vector<llvm::SmallVector<int, 8>> adj;
	      adj.resize(nodes.size());

	      for (int dst = 0; dst < static_cast<int>(nodes.size()); ++dst) {
	        Operation *consumer = nodes[static_cast<size_t>(dst)];
	        if (!consumer)
	          continue;
	        for (Value operand : consumer->getOperands()) {
	          Operation *def = operand.getDefiningOp();
	          if (!def)
	            continue;
	          auto it = opToIdx.find(def);
	          if (it == opToIdx.end())
	            continue;
	          int src = it->second;
	          if (src == dst)
	            continue;
	          adj[static_cast<size_t>(src)].push_back(dst);
	          indeg[static_cast<size_t>(dst)] += 1;
	        }
	      }

	      llvm::SmallVector<int, 32> q;
	      q.reserve(nodes.size());
	      for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
	        if (indeg[static_cast<size_t>(i)] == 0)
	          q.push_back(i);
	      }
	      llvm::SmallVector<int, 32> topo;
	      topo.reserve(nodes.size());
	      while (!q.empty()) {
	        int n = q.pop_back_val();
	        topo.push_back(n);
	        for (int dst : adj[static_cast<size_t>(n)]) {
	          if (--indeg[static_cast<size_t>(dst)] == 0)
	            q.push_back(dst);
	        }
	      }
	      if (topo.size() != nodes.size()) {
	        topo.clear();
	        topo.reserve(nodes.size());
	        for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
	          topo.push_back(i);
	        llvm::sort(topo, [&](int a, int b) {
	          int64_t ida = getStableNodeId(nodes[static_cast<size_t>(a)], a);
	          int64_t idb = getStableNodeId(nodes[static_cast<size_t>(b)], b);
	          if (ida == idb)
	            return a < b;
	          return ida < idb;
	        });
	      }

	      llvm::SmallVector<int64_t, 32> ordered;
	      ordered.reserve(nodes.size());
	      for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
	        Operation *op0 = nodes[static_cast<size_t>(*it)];
	        if (!op0 || op0 == rootOp)
	          continue;
	        auto prodAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_producer");
	        if (!prodAttr || prodAttr.getInt() != kid)
	          continue;
	        ordered.push_back(getStableNodeId(op0, *it));
	      }
	      spec.orderedProducerNodeIds.assign(ordered.begin(), ordered.end());
	    }
	    // 在 `codegen-from-kernel-attrs`（性能测量流程）模式下推导多 kernel 的
	    // block 维度；该模式总是配合强制 tile 参数使用。
	    int64_t effTileM = std::max<int64_t>(1, forceTileM);
	    int64_t effTileN = std::max<int64_t>(1, forceTileN);
    int64_t effTileK = std::max<int64_t>(1, forceTileK);

    int64_t chosenThreadTileM =
        welder::compiler::pickThreadTileDivisible(effTileM, threadTileM);
    int64_t chosenThreadTileN =
        welder::compiler::pickThreadTileDivisible(effTileN, threadTileN);
    bool isFusedChain = hasRowReduction;
    if (enableTensorCoreTf32 || enableTensorCoreF16) {
      // 论文/Welder 对齐（TensorCore）：先确定 MMA 形状，再推导逐线程 tile，
      // 使 epilogue/copy 的线程布局与 warp tiling 对齐。
      const int64_t mmaM = 16;
      const int64_t mmaN = 8;
      const int64_t mmaK = enableTensorCoreF16 ? 16 : 4;
      if ((effTileM % mmaM) != 0 || (effTileN % mmaN) != 0 ||
          (effTileK % mmaK) != 0) {
        llvm::errs()
            << "error: TensorCore requires TILE_M%16==0, TILE_N%8==0, and TILE_K%"
            << mmaK << "==0, got M=" << effTileM << " N=" << effTileN
            << " K=" << effTileK << "\n";
        return 2;
      }
      chosenThreadTileM = std::max<int64_t>(1, mmaM / 8);
      chosenThreadTileN = std::max<int64_t>(1, mmaN / 4);

      // Matmul->Softmax 这类“典型链”会在同一 kernel 内包含行归约。
      // 当前归约切分策略（`TileReductionUsingForallOp`）会把整行映射到线程，
      // 当单线程负责多行（`threadTileM > 1`）时容易超预算。
      //
      // 因此 TensorCore 路径优先使用 `threadTileM==1`，保证行归约的线程映射
      // 落在 `(blockDimX*blockDimY) <= 1024` 的预算内。
      if (!swapBlockDims) {
        bool hasRowReduction = false;
        module->walk([&](Operation *op) {
          if (isRowWiseReductionOp(op)) {
            hasRowReduction = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (hasRowReduction) {
          const int64_t perThreadElems =
              std::max<int64_t>(1, (mmaM * mmaN) / 32);
          if ((effTileN % perThreadElems) == 0) {
            chosenThreadTileM = 1;
            chosenThreadTileN = perThreadElems;
          }
        }
      }
    }
    auto initialBlockDims = welder::compiler::computeBlockDimsExact(
        effTileM, effTileN, chosenThreadTileM, chosenThreadTileN, swapBlockDims);
    int64_t blockDimX = initialBlockDims.x;
    int64_t blockDimY = initialBlockDims.y;
    if ((enableTensorCoreTf32 || enableTensorCoreF16) && isFusedChain) {
      // 对融合的 matmul->行归约 kernel，将 block 维度上扩到最多 32x32，
      // 以避免线程映射溢出。
      if (blockDimX * blockDimY < 1024) {
        blockDimX = std::max<int64_t>(blockDimX, 32);
        blockDimY = std::max<int64_t>(blockDimY, 32);
        if (blockDimX * blockDimY > 1024) {
          blockDimX = 32;
          blockDimY = std::max<int64_t>(1, 1024 / blockDimX);
        }
      }
    }
    bool hasRowReduction = false;
    module->walk([&](Operation *op) {
      if (isRowWiseReductionOp(op)) {
        hasRowReduction = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (enableTensorCoreTf32 || enableTensorCoreF16) {
      const int64_t warps = (effTileM / 16) * (effTileN / 8);
      if (warps <= 0 || warps * 32 > 1024) {
        llvm::errs() << "error: TensorCore invalid warps-per-block: " << warps
                     << " (threads=" << (warps * 32) << ")\n";
        return 2;
      }
      if (isFusedChain) {
        // 允许融合 matmul->行归约 kernel 使用更大的二维线程网格，
        // 以满足嵌套 forall 的线程映射需求（例如 32x32=1024 线程）。
        if (blockDimX * blockDimY < warps * 32) {
          llvm::errs() << "error: internal: TensorCore thread layout under-provisioned "
                          "(threads="
                       << (blockDimX * blockDimY) << " expected>="
                       << (warps * 32) << ")\n";
          return 2;
        }
      } else {
        if (blockDimX * blockDimY != warps * 32) {
          llvm::errs() << "error: internal: TensorCore thread layout mismatch "
                          "(threads="
                       << (blockDimX * blockDimY) << " expected=" << (warps * 32)
                       << ")\n";
          return 2;
        }
      }
    }
    if (blockDimX * blockDimY > 1024) {
      llvm::errs() << "error: block threads exceed 1024 in "
                      "--codegen-from-kernel-attrs mode: ("
                   << blockDimX << "x" << blockDimY << ")\n";
      return 2;
    }

    const bool disableThreadFuseInto =
        getEnvInt64OrDefault("WELDER_DISABLE_THREAD_FUSE_INTO", 0) != 0;
    std::vector<ThreadFusionPair> threadFusionPairs;
    welder::compiler::ThreadFusionFromAttrsResult threadFusionFromAttrs;
    if (!disableThreadFuseInto) {
      const bool enableInferFallback =
          getEnvInt64OrDefault("WELDER_INFER_THREAD_FUSION_PAIRS", 1) != 0;
      threadFusionFromAttrs =
          welder::compiler::buildThreadFusionPairSpecsFromKernelAttrs(
              *module, maxConnectLevel, enableInferFallback);
      threadFusionPairs.reserve(threadFusionFromAttrs.pairs.size());
      for (const auto &spec : threadFusionFromAttrs.pairs) {
        ThreadFusionPair p;
        p.kernelId = spec.kernelId;
        p.producerNodeId = spec.producerNodeId;
        p.consumerNodeId = spec.consumerNodeId;
        p.consumerOperand = spec.consumerOperand;
        threadFusionPairs.push_back(std::move(p));
      }
    }
    normalizeThreadFusionPairs(threadFusionPairs);
    if (tracerPtr) {
      int64_t pairWithOperand = 0;
      for (const ThreadFusionPair &p : threadFusionPairs) {
        if (p.consumerOperand >= 0)
          ++pairWithOperand;
      }
      llvm::json::Object f;
      f["max_connect_level"] = static_cast<int64_t>(maxConnectLevel);
      f["register_fuse_min_connect_level"] =
          static_cast<int64_t>(maxConnectLevel >= 2 ? 2 : 0);
      f["promote_shared_edges"] = false;
      f["thread_fuse_attr_pairs"] = threadFusionFromAttrs.threadFusionAttrPairs;
      f["thread_fuse_attr_pairs_with_operand"] =
          threadFusionFromAttrs.threadFusionAttrPairsWithOperand;
      f["pair_count_explicit_attrs"] =
          threadFusionFromAttrs.explicitThreadFusionPairs;
      f["pair_count_inferred_fallback"] =
          threadFusionFromAttrs.inferredThreadFusionPairs ? 1 : 0;
      f["pair_count"] = static_cast<int64_t>(threadFusionPairs.size());
      f["pair_with_operand"] = pairWithOperand;
      tracerPtr->event("compiler.thread_fusion_pairs", std::move(f),
                       /* isVerbose=*/true);
    }

    tagLinalgOpsForGenericCodegen(*module);
    bool enableAsyncGroups = enableAsyncCopy;
    transformLib = buildGenericTransformLibraryCutEdges(
        &ctx, kernels, /*fuseElementwiseIntoRowReductions=*/{},
        threadFusionPairs, swapBlockDims,
        effTileK, blockDimX, blockDimY,
        chosenThreadTileM, chosenThreadTileN, enableAsyncCopy, asyncBypassL1,
        enableAsyncGroups,
        enableTensorCoreTf32, enableTensorCoreF16,
        effEnableRowReductionInputPromotion, effEnableRowReductionWarp,
        effEnableRowReductionVectorize, effRowReductionVectorWidth,
        effRowReductionThreadsX, effEnableRowReductionCombineVectorize,
        gSkipMapForallToBlocks, gSkipMapNestedForallToThreads);
    return 0;

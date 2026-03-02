static std::vector<PaperScheduleCandidate>
subGraphTilingPaperGlobalShared(const TileGraph &graph, const PaperSubgraph &sg,
                                linalg::LinalgOp sinkOp, int sinkNodeIdx,
                                const SolveOptions &opts,
                                const FootprintInference &inference) {
  const PaperScheduleResolvedLevels scheduleLevels =
      resolvePaperScheduleResolvedLevels(opts);
  const int sharedMinLevelExclusive = scheduleLevels.shared.minLevelExclusive;
  const int sharedMaxLevelInclusive = scheduleLevels.shared.maxLevelInclusive;
  const int recursiveInnerMinLevelExclusive =
      scheduleLevels.recursiveInnerMinLevelExclusive;
  const int recursiveMaxStages = resolvePaperRecursiveMaxStages(opts);
  const auto recursiveWindows =
      resolvePaperRecursiveLevelWindows(opts, scheduleLevels);
  [[maybe_unused]] auto span = [&]() -> Tracer::Span {
    if (!opts.tracer)
      return Tracer::Span();
    llvm::json::Object f;
    f["sink"] = static_cast<int64_t>(sinkNodeIdx);
    f["sink_op"] = sinkOp ? sinkOp->getName().getStringRef().str() : "None";
    f["subgraph_nodes"] = static_cast<int64_t>(sg.nodes.size());
    f["paper_strict"] = opts.paperStrict;
    f["recursive_register"] = opts.paperRecursiveRegisterLevel;
    f["shared_min_level_exclusive"] =
        static_cast<int64_t>(sharedMinLevelExclusive);
    f["shared_max_level_inclusive"] =
        static_cast<int64_t>(sharedMaxLevelInclusive);
    f["recursive_inner_min_level_exclusive"] = static_cast<int64_t>(
        recursiveInnerMinLevelExclusive);
    f["recursive_max_stages"] = static_cast<int64_t>(recursiveMaxStages);
    f["recursive_window_count"] =
        static_cast<int64_t>(recursiveWindows.size());
    f["profiling"] = opts.profile.enable;
    return opts.tracer->span("paper.subgraph_tiling.inner", std::move(f),
                             /* isVerbose=*/true);
  }();
  std::vector<PaperScheduleCandidate> out;

  // 论文/Welder 对齐：先一次性计算每个算子的归约步长（rstep_map）。
  // 这与 shared 层 tile 选择无关，并会在后续细化。
  // 当启用 opts.paperExpandReductionTile 时按每个 tile 进一步调整。
  std::vector<std::vector<int64_t>> baseReduceTiles =
      assignReduceTilesByCoalescingPaper(graph, opts.arch, inference);
  if (!sinkOp)
    return out;
  if (sinkNodeIdx < 0 || sinkNodeIdx >= static_cast<int>(graph.nodes.size()))
    return out;
  if (!sg.inSet.contains(sinkNodeIdx))
    return out;
  const bool graphHasMatmulSoftmaxForProfile =
      graphHasMatmulSoftmaxLikePattern(graph);
  bool subgraphHasMatmulForProfile = subgraphHasMatmulOp(graph, sg);
  if (!subgraphHasMatmulForProfile)
    subgraphHasMatmulForProfile = isa<linalg::MatmulOp>(graph.nodes[sinkNodeIdx].op);
  const bool subgraphHasRowReductionForProfile =
      subgraphHasRowReduction2DOp(graph, sg, nullptr);
  const bool allowRowReductionOnlyProfileInMatmulSoftmax =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_MM_SM_ALLOW_ROW_REDUCTION_SUBGRAPH", 0) != 0;
  const bool allowNonMatmulProfileInMatmulSoftmax =
      getEnvInt64OrDefault("WELDER_PROFILE_MM_SM_ALLOW_NON_MATMUL_SUBGRAPH", 0) !=
      0;
  const bool profileEnabledForSubgraph =
      opts.profile.enable &&
      (!graphHasMatmulSoftmaxForProfile || subgraphHasMatmulForProfile ||
       (allowRowReductionOnlyProfileInMatmulSoftmax &&
        subgraphHasRowReductionForProfile) ||
       allowNonMatmulProfileInMatmulSoftmax);
  if (opts.profile.enable && !profileEnabledForSubgraph && opts.tracer) {
    llvm::json::Object f;
    f["sink"] = static_cast<int64_t>(sinkNodeIdx);
    f["subgraph_nodes"] = static_cast<int64_t>(sg.nodes.size());
    f["graph_has_mm_sm"] = graphHasMatmulSoftmaxForProfile;
    f["subgraph_has_matmul"] = subgraphHasMatmulForProfile;
    f["subgraph_has_row_reduction"] = subgraphHasRowReductionForProfile;
    f["allow_row_reduction_subgraph"] =
        allowRowReductionOnlyProfileInMatmulSoftmax;
    f["allow_non_matmul_profile"] = allowNonMatmulProfileInMatmulSoftmax;
    opts.tracer->event("paper.subgraph_tiling.profile_skip_non_matmul",
                       std::move(f), /*isVerbose=*/true);
  }

  // 论文/Welder 对齐（remap 推断 v1）：
  // 为整个子图推断一致的 swap/no-swap remap。若启用 codegen
  // 搜索，则遵循该推断以收敛搜索空间。
  // 即该 remap（与论文行为对齐）。
  std::optional<bool> sgSwapHint = inferSwapXYHintForSubgraph(graph, sg);
  SolveOptions codegenOpts = opts;
  if (sgSwapHint.has_value() && codegenOpts.codegenSearch.enable &&
      codegenOpts.codegenSearch.swapBlockDims.size() > 1) {
    codegenOpts.codegenSearch.swapBlockDims.clear();
    codegenOpts.codegenSearch.swapBlockDims.push_back(*sgSwapHint);
  }

  // EnumerateSubtiles：这里复用现有候选列表（opts.candidatesMN/K），后续用全子图
  // MemTraffic/MemFootprint 重新打分。
  GenericProblem gp;
  gp.targetOp = sinkOp.getOperation();
  {
    llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
    auto iters = sinkOp.getIteratorTypesArray();
    gp.loops.reserve(sinkOp.getNumLoops());
    for (int64_t i = 0; i < sinkOp.getNumLoops(); ++i) {
      if (ranges[i] == ShapedType::kDynamic)
        return out;
      gp.loops.push_back(LoopDim{ranges[i], iters[i]});
    }
  }

  // 论文对齐的递归骨架：
  // - level0（global->shared）：只枚举 shared 层 tile；
  // - level1（shared->register）：枚举每线程 tile（threadTileM/N）
  //   在 SubGraphTiling 评估循环内完成。
  SolveOptions sharedLevelOpts = opts;
  sharedLevelOpts.enableRegisterLevelSchedule = false;
  std::vector<Candidate> base;
  if (opts.autoCandidates) {
    base = enumerateSharedTilesPaperDfs2D(graph, sg, sinkOp, sinkNodeIdx, opts,
                                          inference, baseReduceTiles);
  }
  if (base.empty())
    base = enumerateCandidatesGeneric(gp, sharedLevelOpts);
  if (opts.tracer) {
    llvm::json::Object f;
    f["base_candidates"] = static_cast<int64_t>(base.size());
    f["auto_candidates"] = opts.autoCandidates;
    f["schedule_topk"] = static_cast<int64_t>(opts.scheduleTopK);
    opts.tracer->event("paper.subgraph_tiling.base", std::move(f));
  }

  auto applySwapHint = [&](Candidate &c) {
    if (sgSwapHint.has_value()) {
      c.swapBlockDims = *sgSwapHint;
      return;
    }
    if (sinkNodeIdx < 0 || sinkNodeIdx >= static_cast<int>(graph.nodes.size()))
      return;
    const TileGraphNode &node = graph.nodes[sinkNodeIdx];
    if (!node.swapXYHint.has_value())
      return;
    c.swapBlockDims = *node.swapXYHint;
  };

  // 严格论文对齐模式：按参考 DefaultPolicy 的顺序
  // 对 shared 层 tile 排序：
  //   评分公式：prio = (MemTrafficBytes + 1) * num_wave
  // 在 Propagate + MemFootprint 剪枝后，只评估 Top-K 配置。
  if (opts.paperStrict) {
    int64_t K = opts.scheduleTopK > 0 ? opts.scheduleTopK
                                     : static_cast<int64_t>(base.size());
    K = std::max<int64_t>(1, K);

    struct BaseKeep {
      double prio = 0.0;
      double trafficBytes = 0.0;
      int64_t blocksTotal = 1;
      int64_t blocksPerSM = 1;
      int64_t numWave = 1;
      Candidate c0;
      PaperSubgraph sgAfterCuts;
      TileGraph g; // 已传播（requiredTile 已填充）
      Traffic t;
    };
    llvm::SmallVector<BaseKeep, 16> kept;
    llvm::SmallVector<BaseKeep, 128> exploredShared;
    const int64_t sharedPoolLimit = std::max<int64_t>(
        K, getEnvInt64OrDefault("WELDER_RECURSIVE_SHARED_STAGE_POOL_LIMIT",
                                /*default=*/512));

    auto maybeInsert = [&](BaseKeep item) {
      if (static_cast<int64_t>(kept.size()) < K) {
        kept.push_back(std::move(item));
        return;
      }
      int worstIdx = 0;
      for (int i = 1; i < static_cast<int>(kept.size()); ++i) {
        if (kept[i].prio > kept[worstIdx].prio)
          worstIdx = i;
      }
      if (item.prio < kept[worstIdx].prio)
        kept[worstIdx] = std::move(item);
    };
    auto recordShared = [&](BaseKeep item) {
      exploredShared.push_back(std::move(item));
      if (sharedPoolLimit <= 0 ||
          static_cast<int64_t>(exploredShared.size()) <= sharedPoolLimit)
        return;
      int worstIdx = 0;
      for (int i = 1; i < static_cast<int>(exploredShared.size()); ++i) {
        if (exploredShared[i].prio > exploredShared[worstIdx].prio)
          worstIdx = i;
      }
      exploredShared[worstIdx] = std::move(exploredShared.back());
      exploredShared.pop_back();
    };

    auto evalSharedTile =
        [&](const Candidate &rootCandIn) -> std::optional<BaseKeep> {
      Candidate rootCand = rootCandIn;

      auto parExtOpt = buildRootParallelExtents2Level(sinkOp, rootCand, opts);
      if (!parExtOpt)
        return std::nullopt;

      // 从 coalescing 选出的每算子归约步长开始。
      std::vector<std::vector<int64_t>> reduceTilesByNode = baseReduceTiles;
      const int64_t reduceExpandPropCacheMax = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_REDUCE_EXPAND_PROP_CACHE_MAX",
                                  /*default=*/128));
      struct ReduceExpandPropCacheVal {
        TileGraph g;
        int64_t fpBytes = 0;
      };
      std::unordered_map<uint64_t, ReduceExpandPropCacheVal> reduceExpandPropCache;
      reduceExpandPropCache.reserve(64);
      auto reduceExpandMixHash = [](uint64_t h, uint64_t x) -> uint64_t {
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
      };

      // 辅助函数：对当前 rstep map 运行传播并计算 shared footprint。
      auto propagateAndFootprint = [&](std::vector<std::vector<int64_t>> &rmap)
          -> std::optional<std::pair<TileGraph, int64_t>> {
        uint64_t key = reduceExpandMixHash(0, static_cast<uint64_t>(sinkNodeIdx));
        for (int64_t v : *parExtOpt)
          key = reduceExpandMixHash(key, static_cast<uint64_t>(v));
        key = reduceExpandMixHash(
            key, static_cast<uint64_t>(rootCand.workgroupPadLastDim));
        key = reduceExpandMixHash(
            key, static_cast<uint64_t>(rootCand.workgroupPadLastDimMatmulOnly));
        key = reduceExpandMixHash(
            key, static_cast<uint64_t>(rootCand.workgroupMultiBufferDepth));
        key = reduceExpandMixHash(key, static_cast<uint64_t>(rootCand.swapBlockDims));
        for (const TileGraphEdge &e : graph.edges) {
          if (e.src < 0 || e.dst < 0)
            continue;
          if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
            continue;
          key = reduceExpandMixHash(key, static_cast<uint64_t>(e.src));
          key = reduceExpandMixHash(key, static_cast<uint64_t>(e.dst));
          key = reduceExpandMixHash(key, static_cast<uint64_t>(e.connectLevel));
        }
        for (int nodeIdx : sg.nodes) {
          key = reduceExpandMixHash(key, static_cast<uint64_t>(nodeIdx));
          if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= rmap.size())
            continue;
          const auto &rs = rmap[static_cast<size_t>(nodeIdx)];
          key = reduceExpandMixHash(key, static_cast<uint64_t>(rs.size()));
          for (int64_t v : rs)
            key = reduceExpandMixHash(key, static_cast<uint64_t>(v));
        }
        if (auto it = reduceExpandPropCache.find(key);
            it != reduceExpandPropCache.end()) {
          TileGraph g = it->second.g;
          return std::make_optional(std::make_pair(std::move(g), it->second.fpBytes));
        }

        TileGraph g = graph; // 拷贝
        syncCutFlagFromConnectLevel(g);

        llvm::ArrayRef<int64_t> sinkRed;
        if (sinkNodeIdx >= 0 &&
            static_cast<size_t>(sinkNodeIdx) < rmap.size())
          sinkRed = rmap[sinkNodeIdx];

        auto rootTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
            sinkOp, *parExtOpt, sinkRed, /*defaultReductionTile=*/0);
        if (!rootTileOpt)
          return std::nullopt;

        TilePropagationOptions popts;
        popts.defaultReductionTile = 0; // 显式设置每节点归约 tile
        popts.reductionTilesByNode = &rmap;
        // 论文对齐的鲁棒性：若冲突则 cut（回退到 global），而非失败。
        popts.enableCutEdges = true;
        popts.resetCutEdges = false; // cut 由 connectLevel 决定

        TilePropagationResult pr = propagateTilesBackward(
            g, sinkNodeIdx, *rootTileOpt, inference, popts);
        if (!pr.success)
          return std::nullopt;

        // 应用 block-order（swapXY）cut，并在 MemFootprint 前重算 kernel 子图，
        // 使代价模型与真实融合组件一致。
        cutEdgesOnSwapXYConflict(g, sg, sharedMinLevelExclusive);
        PaperSubgraph sgAfterCuts =
            extractSubgraphByConnectLevel(g, sinkNodeIdx,
                                          sharedMinLevelExclusive);

			        int64_t fpBytes = computeSharedFootprintBestFitPaper(
			            g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
			            sharedMinLevelExclusive, sharedMaxLevelInclusive,
			            rootCand.workgroupPadLastDim,
			            rootCand.workgroupPadLastDimMatmulOnly,
			            rootCand.workgroupMultiBufferDepth, &rootCand);
		        if (fpBytes < 0)
		          return std::nullopt;
		        if (reduceExpandPropCacheMax > 0 &&
		            static_cast<int64_t>(reduceExpandPropCache.size()) <
		                reduceExpandPropCacheMax) {
		          ReduceExpandPropCacheVal cacheVal;
		          cacheVal.g = g;
		          cacheVal.fpBytes = fpBytes;
		          reduceExpandPropCache.emplace(key, std::move(cacheVal));
		        }
		        return std::make_optional(std::make_pair(std::move(g), fpBytes));
		      };

      auto curOpt = propagateAndFootprint(reduceTilesByNode);
      if (!curOpt)
        return std::nullopt;
      TileGraph curG = std::move(curOpt->first);
      int64_t curFpBytes = curOpt->second;
      if (curFpBytes > opts.arch.smemBytes)
        return std::nullopt;

      // 注意：swapXY cut 已在 propagateAndFootprint 内应用。
      PaperSubgraph sgAfterCuts =
          extractSubgraphByConnectLevel(curG, sinkNodeIdx,
                                        sharedMinLevelExclusive);

      // 论文/Welder 对齐：调度子图内所有算子必须满足
      // 相同 grid 大小（DefaultPolicy.check_tile_shape_isvalid）。
      // 我们通过要求各节点 blocksTotal 一致来强制该约束。
      int64_t sinkBlocksTotal = 1;
      {
        llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
        if (static_cast<int64_t>(ranges.size()) != sinkOp.getNumLoops())
          return std::nullopt;
        auto iters = sinkOp.getIteratorTypesArray();
        int64_t pSeen = 0;
        for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
          if (iters[i] != utils::IteratorType::parallel)
            continue;
          int64_t full = ranges[i];
          if (full == ShapedType::kDynamic || full <= 0)
            return std::nullopt;
          if (pSeen < 0 || pSeen >= static_cast<int64_t>(parExtOpt->size()))
            return std::nullopt;
          int64_t t = (*parExtOpt)[pSeen++];
          if (t <= 0)
            return std::nullopt;
          int64_t tiles =
              opts.requirePerfectTiling ? (full / t) : ceilDiv(full, t);
          sinkBlocksTotal *= std::max<int64_t>(1, tiles);
        }
      }
      sinkBlocksTotal = std::max<int64_t>(1, sinkBlocksTotal);
      for (int nodeIdx : sgAfterCuts.nodes) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(curG.nodes.size()))
          continue;
        if (nodeIdx == sinkNodeIdx)
          continue;
        const TileGraphNode &n = curG.nodes[nodeIdx];
        if (!n.op || !n.hasRequiredTile)
          continue;
        if (isTrivialOpFor2LevelFootprint(n.op))
          continue;
        auto op = dyn_cast_or_null<linalg::LinalgOp>(n.op);
        if (!op)
          continue;
        llvm::SmallVector<int64_t, 8> ranges = op.getStaticLoopRanges();
        if (static_cast<int64_t>(ranges.size()) != op.getNumLoops())
          return std::nullopt;
        auto iters = op.getIteratorTypesArray();
        int64_t blocks = 1;
        for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
          if (iters[i] != utils::IteratorType::parallel)
            continue;
          int64_t full = ranges[i];
          if (full == ShapedType::kDynamic || full <= 0)
            return std::nullopt;
          if (i < 0 || i >= static_cast<int>(n.requiredTile.loopExtents.size()))
            return std::nullopt;
          int64_t t = n.requiredTile.loopExtents[i];
          if (t <= 0)
            return std::nullopt;
          int64_t tiles =
              opts.requirePerfectTiling ? (full / t) : ceilDiv(full, t);
          blocks *= std::max<int64_t>(1, tiles);
        }
        blocks = std::max<int64_t>(1, blocks);
        if (blocks != sinkBlocksTotal)
          return std::nullopt;
      }

      // 论文/Welder 对齐：可在 shared
      // footprint 约束下扩展 reduce 步长（DefaultPolicy._expand_reduce_axis）。
      if (opts.paperExpandReductionTile) {
        const int64_t reduceExpandScoreCacheEnable = getEnvInt64OrDefault(
            "WELDER_REDUCE_EXPAND_SCORE_CACHE_ENABLE", /*default=*/1);
        llvm::DenseMap<uint64_t, double> reduceExpandScoreCache;
        auto buildReduceExpandScoreKey =
            [&](int nodeIdx, ArrayRef<int64_t> parExt,
                ArrayRef<int64_t> redExt) -> uint64_t {
          uint64_t key = reduceExpandMixHash(0, static_cast<uint64_t>(nodeIdx));
          key = reduceExpandMixHash(key, static_cast<uint64_t>(parExt.size()));
          for (int64_t v : parExt)
            key = reduceExpandMixHash(key, static_cast<uint64_t>(v));
          key = reduceExpandMixHash(key, static_cast<uint64_t>(redExt.size()));
          for (int64_t v : redExt)
            key = reduceExpandMixHash(key, static_cast<uint64_t>(v));
          return key;
        };
        llvm::SmallVector<int, 16> nodeOrder =
            topoSortSubgraphByConnectedEdges(curG, sgAfterCuts,
                                             sharedMinLevelExclusive);

        auto isGlobalReadOperand = [&](const TileGraph &g, int nodeIdx,
                                       int operandIdx) -> bool {
          // 若该操作数没有已连接入边，则按 global read 处理。
          for (int edgeIdx : g.nodes[nodeIdx].inEdges) {
            if (edgeIdx < 0 || edgeIdx >= static_cast<int>(g.edges.size()))
              continue;
            const TileGraphEdge &e = g.edges[edgeIdx];
            if (e.dstOperand != operandIdx)
              continue;
            if (e.connectLevel > sharedMinLevelExclusive)
              return false;
          }
          return true;
        };

        auto externalReadCoalescedScore = [&](linalg::LinalgOp op, int nodeIdx,
                                              ArrayRef<int64_t> parExt,
                                              ArrayRef<int64_t> redExt) -> double {
          if (reduceExpandScoreCacheEnable != 0) {
            uint64_t key = buildReduceExpandScoreKey(nodeIdx, parExt, redExt);
            if (auto it = reduceExpandScoreCache.find(key);
                it != reduceExpandScoreCache.end())
              return it->second;
          }
          auto tileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
              op, parExt, redExt, /*defaultReductionTile=*/0);
          if (!tileOpt)
            return 0.0;
          auto fpOpt = inference.infer(op.getOperation(), *tileOpt);
          if (!fpOpt)
            return 0.0;

          double s = 0.0;
          int numInputs = op.getNumDpsInputs();
          for (int i = 0; i < numInputs; ++i) {
            if (!isGlobalReadOperand(curG, nodeIdx, i))
              continue;
            if (i < 0 || i >= static_cast<int>(fpOpt->perOperand.size()))
              continue;
            llvm::SmallVector<int64_t, 4> fullShape =
                getStaticShapeOrUnknown(op.getDpsInputs()[i]);
            if (fullShape.empty())
              continue;
            int64_t cf = coalescedFactor(
                ArrayRef<int64_t>(fpOpt->perOperand[i].shape), fullShape);
            if (cf <= 0)
              continue;
            s += static_cast<double>(cf);
          }
          if (reduceExpandScoreCacheEnable != 0) {
            uint64_t key = buildReduceExpandScoreKey(nodeIdx, parExt, redExt);
            reduceExpandScoreCache[key] = s;
          }
          return s;
        };

        for (int nodeIdx : nodeOrder) {
          if (nodeIdx < 0 || nodeIdx >= static_cast<int>(curG.nodes.size()))
            continue;
          if (!sgAfterCuts.inSet.contains(nodeIdx))
            continue;
          Operation *op0 = curG.nodes[nodeIdx].op;
          auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
          if (!op)
            continue;
          if (op.getNumReductionLoops() <= 0)
            continue;
          if (static_cast<size_t>(nodeIdx) >= reduceTilesByNode.size())
            continue;

          std::vector<int64_t> baseRed = reduceTilesByNode[nodeIdx];
          if (baseRed.empty())
            continue;

          llvm::SmallVector<int64_t, 4> redFull = getReductionLoopFullRanges(op);
          if (redFull.size() != baseRed.size())
            continue;

          std::vector<int64_t> curRed = baseRed;

          // 从已传播 required tile 中提取当前并行 extent。
          llvm::SmallVector<int64_t, 8> parExt;
          parExt.reserve(op.getNumParallelLoops());
          if (!curG.nodes[nodeIdx].hasRequiredTile)
            continue;
          const OpTile &rt = curG.nodes[nodeIdx].requiredTile;
          auto iters = op.getIteratorTypesArray();
          for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
            if (iters[i] == utils::IteratorType::parallel)
              parExt.push_back(rt.loopExtents[i]);
          }
          if (static_cast<int64_t>(parExt.size()) != op.getNumParallelLoops())
            continue;

          // 构建每个轴的步长列表（当前步长的倍数）。
          llvm::SmallVector<llvm::SmallVector<int64_t, 64>, 4> steps;
          steps.reserve(redFull.size());
          for (size_t ax = 0; ax < redFull.size(); ++ax) {
            llvm::SmallVector<int64_t, 64> fs =
                factorsOrPowersForReduceStep(redFull[ax]);
            int64_t base = std::max<int64_t>(1, baseRed[ax]);
            fs.erase(std::remove_if(fs.begin(), fs.end(),
                                    [&](int64_t v) { return (v % base) != 0; }),
                     fs.end());
            if (fs.empty())
              fs.push_back(base);
            steps.push_back(std::move(fs));
          }

          llvm::SmallVector<int, 4> ids;
          ids.reserve(steps.size());
          for (size_t ax = 0; ax < steps.size(); ++ax) {
            auto it = llvm::find(steps[ax], curRed[ax]);
            ids.push_back(it == steps[ax].end() ? 0
                                                : static_cast<int>(it - steps[ax].begin()));
          }

          auto currentScore =
              externalReadCoalescedScore(op, nodeIdx, parExt, curRed);

          while (true) {
            llvm::SmallVector<int, 4> bestIds = ids;
            double bestScore = currentScore;

            for (size_t ax = 0; ax < ids.size(); ++ax) {
              if (ids[ax] + 1 >= static_cast<int>(steps[ax].size()))
                continue;
              llvm::SmallVector<int64_t, 4> trialRed(curRed.begin(), curRed.end());
              trialRed[ax] = steps[ax][ids[ax] + 1];
              if (opts.requirePerfectTiling && (redFull[ax] % trialRed[ax] != 0))
                continue;
              double s = externalReadCoalescedScore(op, nodeIdx, parExt, trialRed);
              if (s > bestScore) {
                bestScore = s;
                bestIds = ids;
                bestIds[ax] += 1;
              }
            }

            if (bestScore <= currentScore)
              break;

            // 暂时应用最佳扩展并重新执行传播 + footprint 计算。
            std::vector<std::vector<int64_t>> trialMap = reduceTilesByNode;
            for (size_t ax = 0; ax < bestIds.size(); ++ax)
              trialMap[nodeIdx][ax] = steps[ax][bestIds[ax]];

            auto trialOpt = propagateAndFootprint(trialMap);
            if (!trialOpt)
              break;
            if (trialOpt->second > opts.arch.smemBytes)
              break;

            // 接受。
            reduceTilesByNode = std::move(trialMap);
            curG = std::move(trialOpt->first);
            curFpBytes = trialOpt->second;
            sgAfterCuts =
                extractSubgraphByConnectLevel(curG, sinkNodeIdx,
                                              sharedMinLevelExclusive);
            ids = bestIds;
            curRed = reduceTilesByNode[nodeIdx];
            currentScore = bestScore;
          }
        }
      }

      // 用主计算算子的首个归约 tile 回填 legacy tileK
      // （论文/Welder 对齐于 epilogue：sink 可能是逐元素算子）。
      int reduceNodeIdx = sinkNodeIdx;
      if (sinkOp.getNumReductionLoops() == 0) {
        reduceNodeIdx = -1;
        llvm::SmallVector<int, 16> stack;
        llvm::SmallDenseSet<int, 32> visitedNodes;
        stack.push_back(sinkNodeIdx);
        visitedNodes.insert(sinkNodeIdx);
        while (!stack.empty() && visitedNodes.size() < 32) {
          int cur = stack.pop_back_val();
          if (cur < 0 || cur >= static_cast<int>(curG.nodes.size()))
            continue;
          for (int edgeIdx : curG.nodes[cur].inEdges) {
            if (edgeIdx < 0 || edgeIdx >= static_cast<int>(curG.edges.size()))
              continue;
            int src = curG.edges[edgeIdx].src;
            if (src < 0 || src >= static_cast<int>(curG.nodes.size()))
              continue;
            if (!visitedNodes.insert(src).second)
              continue;
            Operation *op0 = curG.nodes[src].op;
            auto lop = dyn_cast_or_null<linalg::LinalgOp>(op0);
            if (lop && lop.getNumReductionLoops() > 0) {
              reduceNodeIdx = src;
              stack.clear();
              break;
            }
            stack.push_back(src);
          }
        }
        if (reduceNodeIdx < 0)
          reduceNodeIdx = sinkNodeIdx;
      }
      rootCand.tileK = 1;
      if (reduceNodeIdx >= 0 &&
          static_cast<size_t>(reduceNodeIdx) < reduceTilesByNode.size() &&
          !reduceTilesByNode[reduceNodeIdx].empty() &&
          reduceTilesByNode[reduceNodeIdx].front() > 0) {
        rootCand.tileK = reduceTilesByNode[reduceNodeIdx].front();
      } else if (rootCandIn.tileK > 0) {
        rootCand.tileK = rootCandIn.tileK;
      }

      // MemTraffic（论文对齐：考虑事务宽度/合并访问惩罚）。
      SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
          curG, sg, sharedMinLevelExclusive, sharedMaxLevelInclusive,
          rootCand.workgroupPadLastDim, rootCand.workgroupPadLastDimMatmulOnly,
          rootCand.workgroupSwizzleXor);
      Traffic t0 = computeGlobalTrafficForSubgraph(
          curG, sg, opts.arch, inference, opts.requirePerfectTiling,
          sharedMinLevelExclusive,
          /* applyCoalescingPenalty=*/opts.enableCoalescingPenalty, &layout);
      double trafficBytes = t0.totalBytes();

      // 计算 grid_size 与 num_wave（DefaultPolicy.compute_tile_dict）。
      int64_t blocksTotal = 1;
      {
        llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
        if (static_cast<int64_t>(ranges.size()) != sinkOp.getNumLoops())
          return std::nullopt;
        auto iters = sinkOp.getIteratorTypesArray();
        int64_t pSeen = 0;
        for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
          if (iters[i] != utils::IteratorType::parallel)
            continue;
          int64_t full = ranges[i];
          if (full == ShapedType::kDynamic || full <= 0)
            return std::nullopt;
          if (pSeen < 0 || pSeen >= static_cast<int64_t>(parExtOpt->size()))
            return std::nullopt;
          int64_t t = (*parExtOpt)[pSeen++];
          if (t <= 0)
            return std::nullopt;
          int64_t tiles = opts.requirePerfectTiling ? (full / t) : ceilDiv(full, t);
          blocksTotal *= std::max<int64_t>(1, tiles);
        }
      }
      blocksTotal = std::max<int64_t>(1, blocksTotal);
      // 与上面的 grid 一致性检查保持一致。
      if (blocksTotal != sinkBlocksTotal)
        blocksTotal = sinkBlocksTotal;

      // 估算寄存器使用（DefaultPolicy）：2 * max(prod(tile(parallel))*bits/32)。
      int64_t elemBits = std::max<int64_t>(1, opts.arch.elementBytes) * 8;
      int64_t worstRegs = 0;
      for (int n : sgAfterCuts.nodes) {
        if (n < 0 || n >= static_cast<int>(curG.nodes.size()))
          continue;
        if (!curG.nodes[n].hasRequiredTile)
          continue;
        Operation *op0 = curG.nodes[n].op;
        auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
        if (!op)
          continue;
        auto it = op.getIteratorTypesArray();
        int64_t parElems = 1;
        for (int i = 0; i < static_cast<int>(it.size()); ++i) {
          if (it[i] != utils::IteratorType::parallel)
            continue;
          if (i < 0 ||
              i >= static_cast<int>(curG.nodes[n].requiredTile.loopExtents.size()))
            continue;
          int64_t e = curG.nodes[n].requiredTile.loopExtents[i];
          if (e <= 0) {
            parElems = 0;
            break;
          }
          if (parElems > (std::numeric_limits<int64_t>::max() / e)) {
            parElems = std::numeric_limits<int64_t>::max();
            break;
          }
          parElems *= e;
        }
        if (parElems <= 0)
          continue;
        int64_t regs = (parElems * elemBits + 31) / 32;
        worstRegs = std::max<int64_t>(worstRegs, regs);
      }
      int64_t regUsage = std::max<int64_t>(1, 2 * worstRegs);
      if (regUsage > opts.arch.maxRegistersPerSM)
        return std::nullopt;

      int64_t blocksBySmem = std::max<int64_t>(1, opts.arch.maxBlocksPerSM);
      if (curFpBytes > 0) {
        blocksBySmem =
            std::max<int64_t>(1, getMaxSmemUsageBytes(opts.arch) / curFpBytes);
      }
      int64_t blocksByRegs =
          std::max<int64_t>(1, opts.arch.maxRegistersPerSM / regUsage);
      int64_t blocksByPartition =
          std::max<int64_t>(1, std::max<int64_t>(1, opts.arch.smPartition));
      int64_t blocksPerSM = std::max<int64_t>(
          1, std::min<int64_t>(opts.arch.maxBlocksPerSM,
                               std::min<int64_t>(
                                   blocksByPartition,
                                   std::min<int64_t>(blocksBySmem, blocksByRegs))));
      int64_t concurrentBlocks =
          std::max<int64_t>(1, blocksPerSM * std::max<int64_t>(1, opts.arch.numSM));
      int64_t waves = ceilDiv(std::max<int64_t>(1, blocksTotal), concurrentBlocks);
      waves = std::max<int64_t>(1, waves);

      double prio = (trafficBytes + 1.0) * static_cast<double>(waves);

      BaseKeep item;
      item.prio = prio;
      item.trafficBytes = trafficBytes;
      item.blocksTotal = blocksTotal;
      item.blocksPerSM = blocksPerSM;
      item.numWave = waves;
      rootCand.blocksTotal = blocksTotal;
      rootCand.blocksPerSM = blocksPerSM;
      rootCand.numWave = waves;
      item.c0 = rootCand;
      item.sgAfterCuts = std::move(sgAfterCuts);
      item.g = std::move(curG);
      item.t = t0;
      return item;
    };

    if (opts.autoCandidates) {
      auto getAllFactorsSorted = [](int64_t n) -> llvm::SmallVector<int64_t, 64> {
        llvm::SmallVector<int64_t, 64> fs;
        if (n <= 0)
          return fs;
        for (int64_t i = 1; i * i <= n; ++i) {
          if (n % i != 0)
            continue;
          fs.push_back(i);
          if (i != n / i)
            fs.push_back(n / i);
        }
        llvm::sort(fs);
        fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
        return fs;
      };

      auto buildSteps = [&](int64_t full) -> llvm::SmallVector<int64_t, 64> {
        llvm::SmallVector<int64_t, 64> steps = getAllFactorsSorted(full);
        // 补充少量 2 的幂步长以加速搜索（Welder Python 也这样做）。
        const int64_t extras[] = {2, 4, 8, 16, 32};
        for (int64_t v : extras) {
          if (v > 0 && v < full && llvm::find(steps, v) == steps.end())
            steps.push_back(v);
        }
        llvm::sort(steps);
        steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
        if (steps.empty())
          steps.push_back(1);
        return steps;
      };

      // 对 sink 执行论文对齐的 DFS_smem_tile 风格枚举：
      // - 只枚举前 2 个并行维（shared 层 tile）；
      // - 让 evalSharedTile() 扩展归约 tile（paperExpandReductionTile）。
      int64_t fullM = 0, fullN = 0;
      {
        llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
        auto iters = sinkOp.getIteratorTypesArray();
        int pSeen = 0;
        for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
          if (ranges[i] == ShapedType::kDynamic)
            continue;
          if (iters[i] == utils::IteratorType::parallel) {
            if (pSeen == 0)
              fullM = ranges[i];
            else if (pSeen == 1)
              fullN = ranges[i];
            ++pSeen;
          }
        }
      }
      if (fullM <= 0 || fullN <= 0) {
        // 回退到现有枚举逻辑。
      for (const Candidate &c0 : base) {
        Candidate c1 = c0;
        applySwapHint(c1);
        if (auto itemOpt = evalSharedTile(c1))
          recordShared(std::move(*itemOpt));
      }
      } else {
        llvm::SmallVector<int64_t, 64> stepsM = buildSteps(fullM);
        llvm::SmallVector<int64_t, 64> stepsN = buildSteps(fullN);

        struct QItem {
          double prio = 0.0;
          int64_t tm = 1;
          int64_t tn = 1;
        };
        struct QCmp {
          bool operator()(const QItem &a, const QItem &b) const {
            return a.prio > b.prio; // 小顶堆
          }
        };

        auto keyOf = [](int64_t tm, int64_t tn) -> uint64_t {
          // 打包两个 32 位值。
          return (static_cast<uint64_t>(static_cast<uint32_t>(tm)) << 32) |
                 static_cast<uint64_t>(static_cast<uint32_t>(tn));
        };

        std::priority_queue<QItem, std::vector<QItem>, QCmp> pq;
        llvm::DenseSet<uint64_t> visited;

        auto indexIn = [](ArrayRef<int64_t> xs, int64_t v) -> int {
          auto it = llvm::find(xs, v);
          if (it == xs.end())
            return -1;
          return static_cast<int>(it - xs.begin());
        };

        // 论文 get_base_tile：若能降低
        // 单位输出平均计算量（“workload per item”），则贪心扩展各维。
        auto computeWpi = [&](int64_t tm, int64_t tn) -> double {
          if (tm <= 0 || tn <= 0)
            return std::numeric_limits<double>::infinity();
          if (opts.requirePerfectTiling) {
            if (fullM % tm != 0 || fullN % tn != 0)
              return std::numeric_limits<double>::infinity();
          }

          TileGraph g = graph; // 拷贝
          syncCutFlagFromConnectLevel(g);

          Candidate rootCand;
          rootCand.tileM = tm;
          rootCand.tileN = tn;
          rootCand.tileK = 1;

          auto parExtOpt = buildRootParallelExtents2Level(sinkOp, rootCand, opts);
          if (!parExtOpt)
            return std::numeric_limits<double>::infinity();
          auto rootTileOpt = buildOpTileFromParallelExtents(
              sinkOp, *parExtOpt, /*defaultReductionTile=*/rootCand.tileK);
          if (!rootTileOpt)
            return std::numeric_limits<double>::infinity();

          TilePropagationOptions popts;
          popts.defaultReductionTile = rootCand.tileK;
          // 论文对齐的鲁棒性：若冲突则 cut（回退到 global），而非失败。
          popts.enableCutEdges = true;
          popts.resetCutEdges = false;

          TilePropagationResult pr =
              propagateTilesBackward(g, sinkNodeIdx, *rootTileOpt, inference, popts);
          if (!pr.success)
            return std::numeric_limits<double>::infinity();

          double compute = 0.0;
          for (int n : sg.nodes) {
            auto lop = dyn_cast_or_null<linalg::LinalgOp>(g.nodes[n].op);
            if (!lop)
              continue;
            if (!g.nodes[n].hasRequiredTile)
              continue;
            const OpTile &ot = g.nodes[n].requiredTile;
            auto iters = lop.getIteratorTypesArray();
            if (static_cast<int64_t>(iters.size()) !=
                static_cast<int64_t>(ot.loopExtents.size()))
              continue;
            double vol = 1.0;
            for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
              if (iters[i] != utils::IteratorType::parallel)
                continue;
              int64_t ext = ot.loopExtents[i];
              if (ext <= 0)
                continue;
              vol *= static_cast<double>(ext);
            }
            compute += vol;
          }

          double items = static_cast<double>(tm) * static_cast<double>(tn);
          if (items <= 0.0)
            return std::numeric_limits<double>::infinity();
          return compute / items;
        };

        int64_t baseM = 1, baseN = 1;
        double bestWpi = computeWpi(baseM, baseN);
        for (int dim = 0; dim < 2; ++dim) {
          ArrayRef<int64_t> steps = (dim == 0) ? ArrayRef<int64_t>(stepsM)
                                               : ArrayRef<int64_t>(stepsN);
          for (int64_t v : steps) {
            int64_t tm = (dim == 0) ? v : baseM;
            int64_t tn = (dim == 1) ? v : baseN;
            double wpi = computeWpi(tm, tn);
            if (wpi < bestWpi) {
              bestWpi = wpi;
              baseM = tm;
              baseN = tn;
            }
          }
        }

        auto add = [&](int64_t tm, int64_t tn) {
          if (tm <= 0 || tn <= 0)
            return;
          if (opts.requirePerfectTiling) {
            if (fullM % tm != 0 || fullN % tn != 0)
              return;
          }
          uint64_t k = keyOf(tm, tn);
          if (!visited.insert(k).second)
            return;

          Candidate c0;
          c0.tileM = tm;
          c0.tileN = tn;
          c0.tileK = 1;
          applySwapHint(c0);
          if (auto itemOpt = evalSharedTile(c0)) {
            pq.push(QItem{itemOpt->trafficBytes, tm, tn});
            recordShared(std::move(*itemOpt));
          }
        };

        // 从 base tile 开始扩展。
        add(baseM, baseN);

        // 与 Python 实现类似，限制搜索爆炸。
        while (!pq.empty() && visited.size() <= 2000) {
          QItem cur = pq.top();
          pq.pop();

          int im = indexIn(stepsM, cur.tm);
          int in = indexIn(stepsN, cur.tn);
          if (im < 0 || in < 0)
            continue;

          // 扩展各维（倒序优先最后维，
          // 与 Welder 的 DFS_smem_tile 循环一致）。
          if (in + 1 < static_cast<int>(stepsN.size()))
            add(cur.tm, stepsN[in + 1]);
          if (im + 1 < static_cast<int>(stepsM.size()))
            add(stepsM[im + 1], cur.tn);
        }
      }
    } else {
      for (const Candidate &c0 : base) {
        Candidate c1 = c0;
        applySwapHint(c1);
        if (auto itemOpt = evalSharedTile(c1))
          recordShared(std::move(*itemOpt));
      }
    }
    if (!exploredShared.empty()) {
      llvm::SmallVector<int64_t, 64> keepSharedIdxs =
          selectRecursiveStageTopKSharedCandidateIndices(
              static_cast<int64_t>(exploredShared.size()),
              [&](int64_t i) -> const Candidate & {
                return exploredShared[static_cast<size_t>(i)].c0;
              },
              [&](int64_t i) -> const TileGraph & {
                return exploredShared[static_cast<size_t>(i)].g;
              },
              [&](int64_t i) -> const PaperSubgraph & {
                return exploredShared[static_cast<size_t>(i)].sgAfterCuts;
              },
              [&](int64_t i) -> double {
                return exploredShared[static_cast<size_t>(i)].prio;
              },
              opts, inference, scheduleLevels, K);
      if (keepSharedIdxs.empty()) {
        keepSharedIdxs.reserve(exploredShared.size());
        for (int64_t i = 0; i < static_cast<int64_t>(exploredShared.size()); ++i)
          keepSharedIdxs.push_back(i);
      }
      llvm::SmallDenseSet<int64_t, 128> inserted;
      for (int64_t idx : keepSharedIdxs) {
        if (idx < 0 || idx >= static_cast<int64_t>(exploredShared.size()))
          continue;
        if (!inserted.insert(idx).second)
          continue;
        maybeInsert(std::move(exploredShared[static_cast<size_t>(idx)]));
      }
    }

    llvm::sort(kept, [](const BaseKeep &a, const BaseKeep &b) {
      return a.prio < b.prio;
    });

	    for (BaseKeep &bk : kept) {
	      Candidate c0 = bk.c0;
	      applySwapHint(c0);
		      TileGraph g = std::move(bk.g);
		      PaperSubgraph sgAfterCuts =
		          extractSubgraphByConnectLevel(g, sinkNodeIdx,
		                                        sharedMinLevelExclusive);
      SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
          g, sgAfterCuts, sharedMinLevelExclusive, sharedMaxLevelInclusive,
          c0.workgroupPadLastDim, c0.workgroupPadLastDimMatmulOnly,
          c0.workgroupSwizzleXor);
      MemTrafficBreakdown mt = computeMemTrafficForSubgraph(
          g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
          sharedMinLevelExclusive,
          /* applyCoalescingPenalty=*/opts.enableCoalescingPenalty, &layout);
	      Traffic t = opts.enableCoalescingPenalty ? mt.mem : mt.raw;

      int64_t blocksTotal = std::max<int64_t>(1, c0.blocksTotal);

      // 将 register 层 tile（每线程 tile）作为内层递归枚举
      // 层。
      llvm::SmallVector<std::pair<int64_t, int64_t>, 16> regTiles;
      if (opts.enableRegisterLevelSchedule) {
        // 论文/Welder 对齐：通过 DefaultPolicy 推荐 block_size 候选
        // 并映射为可整除 shared tile 的 2D（blockDimX,blockDimY）
        // 尺寸（tileN,tileM）。
        auto gcdI64 = [](int64_t a, int64_t b) -> int64_t {
          a = std::abs(a);
          b = std::abs(b);
          while (b != 0) {
            int64_t t = a % b;
            a = b;
            b = t;
          }
          return a;
        };
        auto getAllFactorsUpTo = [&](int64_t n, int64_t limit)
            -> llvm::SmallVector<int64_t, 64> {
          llvm::SmallVector<int64_t, 64> fs;
          if (n <= 0)
            return fs;
          for (int64_t i = 1; i * i <= n; ++i) {
            if (n % i != 0)
              continue;
            if (i <= limit)
              fs.push_back(i);
            int64_t j = n / i;
            if (j != i && j <= limit)
              fs.push_back(j);
          }
          llvm::sort(fs);
          fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
          return fs;
        };

        llvm::SmallVector<int64_t, 16> nodeSpaceSizes;
        nodeSpaceSizes.reserve(sgAfterCuts.nodes.size());
        llvm::SmallVector<int64_t, 16> nodeReduceSizes;
        nodeReduceSizes.reserve(sgAfterCuts.nodes.size());
        for (int n : sgAfterCuts.nodes) {
          if (n < 0 || n >= static_cast<int>(g.nodes.size()))
            continue;
          if (!g.nodes[n].hasRequiredTile)
            continue;
          Operation *op0 = g.nodes[n].op;
          auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
          if (!op)
            continue;
          if (isTrivialOpFor2LevelFootprint(op0))
            continue;
          auto it = op.getIteratorTypesArray();
          int64_t space = 1;
          for (int i = 0; i < static_cast<int>(it.size()); ++i) {
            if (it[i] != utils::IteratorType::parallel)
              continue;
            if (i < 0 ||
                i >= static_cast<int>(g.nodes[n].requiredTile.loopExtents.size()))
              continue;
            int64_t e = g.nodes[n].requiredTile.loopExtents[i];
            if (e <= 0) {
              space = 0;
              break;
            }
            if (space > (std::numeric_limits<int64_t>::max() / e)) {
              space = std::numeric_limits<int64_t>::max();
              break;
            }
            space *= e;
          }
          space = std::max<int64_t>(1, space);
          nodeSpaceSizes.push_back(space);

          int64_t red = 1;
          if (n >= 0 && static_cast<size_t>(n) < baseReduceTiles.size()) {
            for (int64_t r : baseReduceTiles[n]) {
              if (r <= 0)
                continue;
              if (red > (std::numeric_limits<int64_t>::max() / r)) {
                red = std::numeric_limits<int64_t>::max();
                break;
              }
              red *= r;
            }
          }
          red = std::max<int64_t>(1, red);
          nodeReduceSizes.push_back(red);
        }

        int64_t maxBlockSize = 0;
        for (int64_t s : nodeSpaceSizes)
          maxBlockSize = (maxBlockSize == 0) ? s : gcdI64(maxBlockSize, s);
        if (maxBlockSize <= 0)
          maxBlockSize = 1;

        // DefaultPolicy 特例：若过小且等于 min(space sizes)，
        // 用 space*reduce size 的 gcd 扩展候选。
        int64_t minSpace = nodeSpaceSizes.empty() ? maxBlockSize
                                                  : *llvm::min_element(nodeSpaceSizes);
        llvm::SmallVector<int64_t, 64> blockSizes;
        if (maxBlockSize < opts.arch.warpSize * opts.arch.smPartition &&
            maxBlockSize == minSpace && !nodeReduceSizes.empty()) {
          int64_t maxPossible = 0;
          for (size_t i = 0; i < nodeSpaceSizes.size() && i < nodeReduceSizes.size();
               ++i) {
            int64_t total = nodeSpaceSizes[i];
            if (nodeReduceSizes[i] > 0 &&
                total <= (std::numeric_limits<int64_t>::max() / nodeReduceSizes[i]))
              total *= nodeReduceSizes[i];
            maxPossible = (maxPossible == 0) ? total : gcdI64(maxPossible, total);
          }
          auto factors = getAllFactorsUpTo(maxPossible, /*limit=*/1024);
          for (int64_t x : factors) {
            if (x % maxBlockSize != 0)
              continue;
            bool ok = true;
            for (int64_t s : nodeSpaceSizes) {
              if (!((x % s) == 0 || (s % x) == 0)) {
                ok = false;
                break;
              }
            }
            if (ok)
              blockSizes.push_back(x);
          }
        } else {
          blockSizes = getAllFactorsUpTo(maxBlockSize, /*limit=*/1024);
        }
        if (blockSizes.empty())
          blockSizes.push_back(std::min<int64_t>(1024, maxBlockSize));

        auto scoreBlockSize = [&](int64_t n) -> std::pair<double, double> {
          int64_t warp = std::max<int64_t>(1, opts.arch.warpSize);
          int64_t part = std::max<int64_t>(1, opts.arch.smPartition);
          int64_t numWarp = ceilDiv(std::max<int64_t>(1, n), warp);
          double r1 = std::max(static_cast<double>(numWarp) / static_cast<double>(part),
                               static_cast<double>(part) / static_cast<double>(numWarp));
          double r2 =
              static_cast<double>(numWarp * warp - std::max<int64_t>(1, n)) /
              static_cast<double>(std::max<int64_t>(1, n));
          return {r1, r2};
        };
        llvm::sort(blockSizes, [&](int64_t a, int64_t b) {
          return scoreBlockSize(a) < scoreBlockSize(b);
        });

        auto addRegTileFromBlockDims = [&](int64_t blockDimX, int64_t blockDimY) {
          if (blockDimX <= 0 || blockDimY <= 0)
            return;
          if (blockDimX * blockDimY > 1024)
            return;
          if (c0.tileM <= 0 || c0.tileN <= 0)
            return;
          int64_t xTile = c0.swapBlockDims ? c0.tileM : c0.tileN;
          int64_t yTile = c0.swapBlockDims ? c0.tileN : c0.tileM;
          if (xTile <= 0 || yTile <= 0)
            return;
          if (xTile % blockDimX != 0 || yTile % blockDimY != 0)
            return;
          int64_t ttx = xTile / blockDimX;
          int64_t tty = yTile / blockDimY;
          int64_t ttm = c0.swapBlockDims ? ttx : tty;
          int64_t ttn = c0.swapBlockDims ? tty : ttx;
          if (ttm <= 0 || ttn <= 0)
            return;
          regTiles.push_back({ttm, ttn});
        };

        int64_t xTile = c0.swapBlockDims ? c0.tileM : c0.tileN;
        int64_t yTile = c0.swapBlockDims ? c0.tileN : c0.tileM;

        // 将 1D block_size 映射为可整除逻辑 (X,Y) 的 2D 因子对
        // tile 维度（X 对应 blockDim.x）。当 swapBlockDims=true 时，X
        // 映射到 tileM；否则映射到 tileN。
        for (int64_t bs : blockSizes) {
          if (bs <= 0 || bs > 1024)
            continue;
          bool added = false;
          // 优先更大的 X 以提升连续性/合并访问。
          for (int64_t bx = std::min<int64_t>(bs, xTile); bx >= 1; --bx) {
            if (bs % bx != 0)
              continue;
            if (xTile % bx != 0)
              continue;
            int64_t by = bs / bx;
            if (by <= 0 || yTile % by != 0)
              continue;
            addRegTileFromBlockDims(bx, by);
            added = true;
            break;
          }
          if (added && regTiles.size() >= 32)
            break;
        }

        llvm::sort(regTiles);
        regTiles.erase(std::unique(regTiles.begin(), regTiles.end()),
                       regTiles.end());
	      } else {
	        regTiles.push_back({0, 0});
	      }
	      const int64_t maxRowReductionExtentForTc =
	          computeTcRowReductionExtentForThreadMapping(g, sgAfterCuts);
      const bool subgraphHasMatmul = subgraphHasMatmulOp(g, sgAfterCuts);
      const bool matmulSoftmaxLikeSubgraph =
          isMatmulSoftmaxLikeSubgraph(g, sgAfterCuts) ||
          (subgraphHasMatmul && graphHasMatmulSoftmaxLikePattern(g));
      const bool tensorCoreLayoutFeasible =
          isTensorCoreStrideLayoutFeasibleForSubgraph(g, sgAfterCuts);
	      if (!regTiles.empty()) {
	        regTiles = selectRecursiveStageTopKRegisterTiles(
	            regTiles, c0, g, sgAfterCuts, opts, inference, scheduleLevels,
	            maxRowReductionExtentForTc);
	      }

	      struct Pending {
	        PaperScheduleCandidate pc;
	        bool drop = false;
	      };
	      std::vector<Pending> pending;

	      for (auto [ttm, ttn] : regTiles) {
	        Candidate baseReg = c0;
	        baseReg.threadTileM = ttm;
	        baseReg.threadTileN = ttn;

	        std::vector<Candidate> variants =
	            expandCandidatesWithCodegenSearch(
                  baseReg, codegenOpts, maxRowReductionExtentForTc,
                  tensorCoreLayoutFeasible, matmulSoftmaxLikeSubgraph,
                  subgraphHasMatmul);
	        if (variants.empty())
	          continue;
	        if (variants.size() > 1) {
	          llvm::SmallVector<int64_t, 64> keepVariantIdxs =
	              selectRecursiveStageTopKCandidateIndices(
	                  static_cast<int64_t>(variants.size()),
	                  [&](int64_t i) -> const Candidate & {
	                    return variants[static_cast<size_t>(i)];
	                  },
	                  g, sgAfterCuts, opts, inference, scheduleLevels,
	                  maxRowReductionExtentForTc);
	          if (!keepVariantIdxs.empty() &&
	              keepVariantIdxs.size() < variants.size()) {
	            std::vector<Candidate> filteredVariants;
	            filteredVariants.reserve(keepVariantIdxs.size());
	            for (int64_t i : keepVariantIdxs) {
	              if (i < 0 || i >= static_cast<int64_t>(variants.size()))
	                continue;
	              filteredVariants.push_back(
	                  std::move(variants[static_cast<size_t>(i)]));
	            }
	            variants = std::move(filteredVariants);
	          }
	        }

			        for (Candidate cand : variants) {
            applySwapHint(cand);
					          int64_t fpBytes = computeSharedFootprintBestFitPaper(
					              g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
					              sharedMinLevelExclusive, sharedMaxLevelInclusive,
					              cand.workgroupPadLastDim,
					              cand.workgroupPadLastDimMatmulOnly,
					              cand.workgroupMultiBufferDepth, &cand);
	          if (fpBytes < 0 || fpBytes > opts.arch.smemBytes)
	            continue;

	          cand.estSharedBankConflict =
	              estimateMatmulSharedBankConflictFactor(cand, opts.arch);

          // 即使在严格模式下，也填充便于调试的 occupancy 估计。
          int64_t blockThreads = estimateBlockThreadsForCandidate(
              cand, maxRowReductionExtentForTc);

          int64_t blocksPerSM = std::max<int64_t>(1, opts.arch.maxBlocksPerSM);
          int64_t byPartition =
              std::max<int64_t>(1, std::max<int64_t>(1, opts.arch.smPartition));
          int64_t byThreads =
              std::max<int64_t>(1, opts.arch.maxThreadsPerSM /
                                       std::max<int64_t>(1, blockThreads));
          // 当 blockThreads 不是 32 的倍数时，warp 限制的 occupancy
          // 比线程限制更严格。
          int64_t warpSize = std::max<int64_t>(1, opts.arch.warpSize);
          int64_t warpsPerSM =
              std::max<int64_t>(1, opts.arch.maxThreadsPerSM / warpSize);
          int64_t warpsPerBlock =
              (std::max<int64_t>(1, blockThreads) + (warpSize - 1)) / warpSize;
          int64_t byWarps =
              std::max<int64_t>(1, warpsPerSM / std::max<int64_t>(1, warpsPerBlock));
          blocksPerSM =
              std::min<int64_t>(blocksPerSM,
                                std::min<int64_t>(byPartition,
                                                  std::min(byThreads, byWarps)));
          int64_t bySmem = blocksPerSM;
          if (fpBytes > 0)
            bySmem =
                std::max<int64_t>(1, getMaxSmemUsageBytes(opts.arch) / fpBytes);
          blocksPerSM =
              std::max<int64_t>(1, std::min<int64_t>(blocksPerSM, bySmem));

	          int64_t regsAcc = 0;
	          if (cand.threadTileM > 0 && cand.threadTileN > 0)
	            regsAcc = cand.threadTileM * cand.threadTileN;
	          const RecursiveStageAggregate recursiveStageAgg =
	              estimateRecursiveStageAggregateForCandidate(
	                  g, sgAfterCuts, opts, inference, scheduleLevels,
	                  blockThreads, cand.workgroupPadLastDim,
	                  cand.workgroupPadLastDimMatmulOnly,
	                  cand.workgroupSwizzleXor);
	          if (!recursiveStageAgg.feasible) {
	            if (cand.feasibilityCode == 0)
	              cand.feasibilityCode = 25; // 递归阶段溢出
	            continue;
	          }
	          int64_t regsOverhead = 32;
	          regsOverhead += recursiveStageAgg.regReuseRegsPerThread;
	          if (cand.enableAsyncCopy)
	            regsOverhead += 8;
	          if (cand.enableSoftwarePipelining)
	            regsOverhead += 16;
          int64_t regsPerThread = std::max<int64_t>(1, regsAcc + regsOverhead);
          if (regsPerThread > opts.arch.maxRegistersPerThread) {
            if (cand.feasibilityCode == 0)
              cand.feasibilityCode = 20; // 寄存器溢出
            continue;
          }
          cand.estRegsPerThread = regsPerThread;
          bool isTensorCoreCand =
              cand.enableTensorCoreTf32 || cand.enableTensorCoreF16;
          if (cand.enableSoftwarePipelining) {
            int64_t regSoftCap =
                std::max<int64_t>(96, (opts.arch.maxRegistersPerThread * 3) / 8);
            if (isTensorCoreCand) {
              regSoftCap = std::max<int64_t>(
                  80, (opts.arch.maxRegistersPerThread * 7) / 20);
            }
            if (regsPerThread >= regSoftCap &&
                (blocksPerSM <= 1 || isTensorCoreCand)) {
              if (cand.feasibilityCode == 0)
                cand.feasibilityCode = 21; // 流水线低占用/寄存器压力
              continue;
            }
          }
          int64_t regsPerBlock =
              regsPerThread * std::max<int64_t>(1, blockThreads);
          int64_t byRegs = blocksPerSM;
          if (regsPerBlock > 0)
            byRegs =
                std::max<int64_t>(1, opts.arch.maxRegistersPerSM / regsPerBlock);
          blocksPerSM =
              std::max<int64_t>(1, std::min<int64_t>(blocksPerSM, byRegs));

          if (cand.enableSoftwarePipelining) {
            int64_t minPipeBlocksPerSM = isTensorCoreCand ? 2 : 1;
            if (isTensorCoreCand && cand.pipelineDepth >= 4)
              minPipeBlocksPerSM = 3;
            if (blocksPerSM < minPipeBlocksPerSM) {
              if (cand.feasibilityCode == 0)
                cand.feasibilityCode = 22; // 流水线 TensorCore 占用压力
              continue;
            }
          }
          if (cand.enableSoftwarePipelining &&
              cand.enableMatmulSoftmaxSharedReuseFusion &&
              cand.enableRowReductionChainReuseFusion) {
            const int64_t minMmSmPipeBlocks = std::max<int64_t>(
                1, getEnvInt64OrDefault(
                       isTensorCoreCand
                           ? "WELDER_MM_SM_TC_PIPE_MIN_BLOCKS_PER_SM"
                           : "WELDER_MM_SM_PIPE_MIN_BLOCKS_PER_SM",
                       isTensorCoreCand ? 3 : 2));
          const int64_t maxMmSmPipeRegs = std::max<int64_t>(
              1, getEnvInt64OrDefault(
                     isTensorCoreCand
                         ? "WELDER_MM_SM_TC_PIPE_MAX_REGS_PER_THREAD"
                         : "WELDER_MM_SM_PIPE_MAX_REGS_PER_THREAD",
                     isTensorCoreCand ? 96 : 112));
          int64_t effectiveMinMmSmPipeBlocks = minMmSmPipeBlocks;
          int64_t effectiveMaxMmSmPipeRegs = maxMmSmPipeRegs;
          const bool relaxTcPipeGuardForWaitGroup =
              isTensorCoreCand && cand.enableAsyncCopy &&
              cand.pipelineSetAsyncWaitGroups &&
              (getEnvInt64OrDefault(
                   "WELDER_MM_SM_TC_PIPE_RELAX_FOR_WAIT_GROUP", 1) != 0);
          if (relaxTcPipeGuardForWaitGroup) {
            const int64_t relaxedMinBlocks = std::max<int64_t>(
                1, getEnvInt64OrDefault(
                       "WELDER_MM_SM_TC_PIPE_MIN_BLOCKS_PER_SM_WAIT_GROUP",
                       2));
            const int64_t relaxedMaxRegs = std::max<int64_t>(
                1, getEnvInt64OrDefault(
                       "WELDER_MM_SM_TC_PIPE_MAX_REGS_PER_THREAD_WAIT_GROUP",
                       112));
            effectiveMinMmSmPipeBlocks =
                std::min(effectiveMinMmSmPipeBlocks, relaxedMinBlocks);
            effectiveMaxMmSmPipeRegs =
                std::max(effectiveMaxMmSmPipeRegs, relaxedMaxRegs);
          }
          if (blocksPerSM < effectiveMinMmSmPipeBlocks ||
              regsPerThread > effectiveMaxMmSmPipeRegs) {
            if (cand.feasibilityCode == 0)
              cand.feasibilityCode = 24; // mm-sm 防溢写流水线拒绝
            continue;
          }
        }

	          int64_t concurrentBlocks =
	              std::max<int64_t>(1, blocksPerSM * opts.arch.numSM);
	          int64_t waves = ceilDiv(blocksTotal, concurrentBlocks);
		          cand.blocksPerSM = blocksPerSM;
		          cand.numWave = waves;

	          // 论文对齐的光栅化（tensorcore、单算子、大 num_wave）。
		          maybeApplyRasterizationTcPolicyPaper(sgAfterCuts, t, cand);

		          double sh2reg = recursiveStageAgg.sharedToRegBytes;
		          fillCostAndScoreFromPaperModel(cand, fpBytes, mt, sh2reg, opts);

		          PaperScheduleCandidate pc;
		          pc.cand = cand;
		          pc.sharedFootprintBytes = fpBytes;
		          pc.traffic = cand.traffic;
	          pc.estimatedLatency = cand.score;
			          pending.push_back(Pending{std::move(pc), /*drop=*/false});
	        }
		      }
		      if (pending.size() > 1) {
		        llvm::SmallVector<int64_t, 64> keepIdxs =
		            selectRecursiveStageTopKCandidateIndices(
		                static_cast<int64_t>(pending.size()),
		                [&](int64_t i) -> const Candidate & {
		                  return pending[static_cast<size_t>(i)].pc.cand;
		                },
		                g, sgAfterCuts, opts, inference, scheduleLevels,
		                maxRowReductionExtentForTc);
		        if (!keepIdxs.empty() &&
		            keepIdxs.size() < static_cast<size_t>(pending.size())) {
		          std::vector<Pending> filtered;
		          filtered.reserve(keepIdxs.size());
		          for (int64_t i : keepIdxs) {
		            if (i < 0 || i >= static_cast<int64_t>(pending.size()))
		              continue;
		            filtered.push_back(std::move(pending[static_cast<size_t>(i)]));
		          }
		          pending = std::move(filtered);
		        }
		      }

			      if (!pending.empty()) {
			        llvm::SmallVector<int64_t, 64> keepIdxs =
			            selectRecursiveStageTopKCandidateIndices(
			                static_cast<int64_t>(pending.size()),
		                [&](int64_t i) -> const Candidate & {
		                  return pending[static_cast<size_t>(i)].pc.cand;
		                },
		                g, sgAfterCuts, opts, inference, scheduleLevels,
		                maxRowReductionExtentForTc);
	        if (!keepIdxs.empty() &&
	            keepIdxs.size() < static_cast<size_t>(pending.size())) {
	          std::vector<Pending> filtered;
	          filtered.reserve(keepIdxs.size());
	          for (int64_t i : keepIdxs) {
	            if (i < 0 || i >= static_cast<int64_t>(pending.size()))
	              continue;
	            filtered.push_back(std::move(pending[static_cast<size_t>(i)]));
	          }
	          pending = std::move(filtered);
	        }
	      }

	      if (profileEnabledForSubgraph && !pending.empty()) {
	        const int64_t strictMaxProfileAttempts = std::max<int64_t>(
	            0, getEnvInt64OrDefault("WELDER_PROFILE_MAX_ATTEMPTS_PER_SHARED_TILE",
	                                    opts.codegenSearch.enable ? 24 : 12));
        if (strictMaxProfileAttempts > 0 &&
            static_cast<int64_t>(pending.size()) > strictMaxProfileAttempts) {
          const int64_t totalBeforeCap = static_cast<int64_t>(pending.size());
          llvm::sort(pending, [](const Pending &a, const Pending &b) {
            return betterPaperCandidateByProfilePriority(a.pc, b.pc);
          });
          const auto isTensorCoreCand = [](const Pending &p) {
            return p.pc.cand.enableTensorCoreF16 || p.pc.cand.enableTensorCoreTf32;
          };
          bool hasTcAny = false;
          bool hasNonTcAny = false;
          for (const Pending &p : pending) {
            if (isTensorCoreCand(p))
              hasTcAny = true;
            else
              hasNonTcAny = true;
          }
          const int64_t cap =
              std::max<int64_t>(1, std::min<int64_t>(
                                       strictMaxProfileAttempts,
                                       static_cast<int64_t>(pending.size())));
          if (hasTcAny && hasNonTcAny && cap >= 2) {
            auto hasClassInPrefix = [&](bool wantTc) -> bool {
              for (int64_t i = 0; i < cap; ++i) {
                if (isTensorCoreCand(pending[static_cast<size_t>(i)]) == wantTc)
                  return true;
              }
              return false;
            };
            auto findClassInTail = [&](bool wantTc) -> int64_t {
              for (int64_t i = cap; i < static_cast<int64_t>(pending.size());
                   ++i) {
                if (isTensorCoreCand(pending[static_cast<size_t>(i)]) == wantTc)
                  return i;
              }
              return -1;
            };
            auto ensurePrefixClass = [&](bool wantTc) {
              if (hasClassInPrefix(wantTc))
                return;
              int64_t tail = findClassInTail(wantTc);
              if (tail < 0)
                return;
              int64_t replace = cap - 1;
              for (int64_t i = cap - 1; i >= 1; --i) {
                if (isTensorCoreCand(pending[static_cast<size_t>(i)]) != wantTc) {
                  replace = i;
                  break;
                }
              }
              std::swap(pending[static_cast<size_t>(replace)],
                        pending[static_cast<size_t>(tail)]);
            };
            ensurePrefixClass(/*wantTc=*/true);
            ensurePrefixClass(/*wantTc=*/false);
          }
          pending.resize(static_cast<size_t>(cap));
          if (opts.tracer) {
            llvm::json::Object f;
            f["total_before_cap"] = totalBeforeCap;
            f["capped_to"] = cap;
            f["cap"] = strictMaxProfileAttempts;
            opts.tracer->event("profile.strict_attempt_cap", std::move(f),
                               /* isVerbose=*/true);
          }
        }
        if (opts.tracer) {
          llvm::json::Object f;
          f["total"] = static_cast<int64_t>(pending.size());
          f["max_jobs"] = static_cast<int64_t>(opts.profile.maxParallelJobs);
          opts.tracer->event("profile.batch.start", std::move(f));
        }
        const size_t total = pending.size();
        const size_t step = std::max<size_t>(1, total / 20);
        std::atomic<size_t> done{0};
        std::atomic<size_t> ok{0};
        std::atomic<size_t> fail{0};
        std::atomic<size_t> cached{0};
        runParallelBounded(
            opts.profile.maxParallelJobs, pending.size(), [&](size_t i) {
              Pending &item = pending[i];
              if (item.drop)
                return;
              bool wasCached = false;
              bool profOk = false;
		              if (auto msOpt = profileSubgraphByCompilingToNvvm(
		                      g, sgAfterCuts, sinkNodeIdx, item.pc.cand, opts,
		                      /* outWasCached=*/&wasCached)) {
		                item.pc.cand.cost.profiledMs = *msOpt;
		                item.pc.cand.score = *msOpt;
		                item.pc.estimatedLatency = *msOpt;
                profOk = true;
                ok.fetch_add(1);
                if (wasCached)
                  cached.fetch_add(1);
		              } else {
                fail.fetch_add(1);
              }

              size_t d = done.fetch_add(1) + 1;
              if (opts.tracer && (d == total || (d % step) == 0)) {
                llvm::json::Object f;
                f["done"] = static_cast<int64_t>(d);
                f["total"] = static_cast<int64_t>(total);
                f["ok"] = static_cast<int64_t>(ok.load());
                f["fail"] = static_cast<int64_t>(fail.load());
                f["cached"] = static_cast<int64_t>(cached.load());
                opts.tracer->event("profile.batch.progress", std::move(f));
              }
              if (!profOk && opts.pruneOnProfileFailure)
                item.drop = true;
            });
        if (opts.tracer) {
          llvm::json::Object f;
          f["done"] = static_cast<int64_t>(done.load());
          f["total"] = static_cast<int64_t>(total);
          f["ok"] = static_cast<int64_t>(ok.load());
          f["fail"] = static_cast<int64_t>(fail.load());
          f["cached"] = static_cast<int64_t>(cached.load());
          opts.tracer->event("profile.batch.end", std::move(f));
        }
      }

      if (opts.paperRecursiveRegisterLevel) {
        bool hasBest = false;
        PaperScheduleCandidate best;
        for (Pending &item : pending) {
          if (item.drop)
            continue;
          if (!hasBest || betterPaperCandidateByProfilePriority(item.pc, best)) {
            best = std::move(item.pc);
            hasBest = true;
          }
        }
        if (hasBest)
          out.push_back(std::move(best));
      } else {
        for (Pending &item : pending) {
          if (!item.drop)
            out.push_back(std::move(item.pc));
        }
      }
    }

    llvm::sort(out, betterPaperCandidateByProfilePriority);
    return out;
  }

  // 按估计时延排序的优先队列（越小越好）。
  struct ScoredIdx {
    double score;
    int idx;
  };
  llvm::SmallVector<ScoredIdx, 64> scored;
  scored.reserve(base.size());

  for (int i = 0; i < static_cast<int>(base.size()); ++i)
    scored.push_back(ScoredIdx{base[i].score, i});

  // 只保留前 K 个基础候选（减少传播开销）。
  llvm::sort(scored, [](const ScoredIdx &a, const ScoredIdx &b) {
    if (a.score == b.score)
      return a.idx < b.idx;
    return a.score < b.score;
  });
  int64_t desiredOut =
      (opts.scheduleTopK > 0) ? opts.scheduleTopK
                              : static_cast<int64_t>(scored.size());
  desiredOut = std::max<int64_t>(1, desiredOut);
  const bool wantTensorCoreClassSlack =
      opts.codegenSearch.enable &&
      (llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true) ||
       llvm::is_contained(opts.codegenSearch.enableTensorCoreTf32, true));
  const bool wantAsyncClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
  const bool wantPipeClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
  const bool wantWaitGroupClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
  const bool wantRowReuseClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionChainReuseFusion,
                         true);
  const bool wantRowPromoClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionInputPromotion,
                         true);
  const bool wantRowPromoVecClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(
          opts.codegenSearch.enableRowReductionInputPromotionVectorize, true);
  const bool wantRowWarpClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionWarp, true);
  const bool wantRowVecClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionVectorize, true);
  const bool wantRowCombineVecClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionCombineVectorize,
                         true);
  const bool wantRowRelaxBarrierClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionRelaxBarriers,
                         true);
  const bool wantRowSkipCombineBarrierClassSlack =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionSkipCombineBarrier,
                         true);
  const bool wantExtraCodegenClassSlack =
      wantTensorCoreClassSlack || wantAsyncClassSlack || wantPipeClassSlack ||
      wantWaitGroupClassSlack || wantRowReuseClassSlack ||
      wantRowPromoClassSlack ||
      wantRowPromoVecClassSlack || wantRowWarpClassSlack ||
      wantRowVecClassSlack || wantRowCombineVecClassSlack ||
      wantRowRelaxBarrierClassSlack || wantRowSkipCombineBarrierClassSlack;
  // 启用 profiling 时，编译/测量过多配置会
  // 非常慢。保留少量余量以维持可选项。
  int64_t maxOut = desiredOut;
	  if (profileEnabledForSubgraph || wantExtraCodegenClassSlack)
    maxOut = std::max<int64_t>(desiredOut, desiredOut * 4);
  // 当 codegen 旋钮（如多缓冲）抬高 shared footprint 时，很多
  // shared 层 Top 候选可能失效。保留更大的基础预算，
  // 以确保仍能找到有效配置（论文对齐：TopK 在剪枝后确定）。
  int64_t baseBudget = static_cast<int64_t>(scored.size());
  if (opts.scheduleTopK > 0) {
    baseBudget = std::min<int64_t>(
        baseBudget, std::max<int64_t>(desiredOut * 16, 32));
  }
  baseBudget = std::min<int64_t>(baseBudget, 256);

  const bool subgraphHasMatmul = subgraphHasMatmulOp(graph, sg);
  const bool wantTensorCoreF16 = opts.codegenSearch.enable
                                     ? llvm::is_contained(
                                           opts.codegenSearch.enableTensorCoreF16, true)
                                     : opts.profile.enableTensorCoreF16;
  const bool wantTensorCoreTf32 =
      opts.codegenSearch.enable
          ? llvm::is_contained(opts.codegenSearch.enableTensorCoreTf32, true)
          : false;
  const bool wantTensorCoreBaseDiversity =
      subgraphHasMatmul && (wantTensorCoreF16 || wantTensorCoreTf32);
  const int64_t sgMaxRowReductionExtent =
      computeMaxRowReductionExtentForSubgraph(graph, sg);
  const int64_t sgTcRowReductionExtentForThreads =
      computeTcRowReductionExtentForThreadMapping(graph, sg);
  const bool tensorCoreLayoutFeasibleForBase =
      isTensorCoreStrideLayoutFeasibleForSubgraph(graph, sg);

  auto isTensorCoreFriendlyBaseCandidate = [&](const Candidate &c0) -> bool {
    if (!subgraphHasMatmul || !tensorCoreLayoutFeasibleForBase)
      return false;
    Candidate probe = c0;
    probe.enableTensorCoreF16 = wantTensorCoreF16;
    probe.enableTensorCoreTf32 = !probe.enableTensorCoreF16 && wantTensorCoreTf32;
    if (!(probe.enableTensorCoreF16 || probe.enableTensorCoreTf32))
      return false;
    chooseMmaShapeForCandidate(probe);
    if (probe.mmaM <= 0 || probe.mmaN <= 0 || probe.mmaK <= 0)
      return false;
    auto tcThreads = computeTensorCoreBlockThreadsForCodegen(
        probe, sgTcRowReductionExtentForThreads);
    return tcThreads.has_value() && *tcThreads > 0 && *tcThreads <= 1024;
  };

  llvm::SmallVector<int64_t, 64> selectedScoredPos;
  selectedScoredPos.reserve(static_cast<size_t>(baseBudget) + 4);
  for (int64_t si = 0; si < baseBudget; ++si)
    selectedScoredPos.push_back(si);

  auto pushScoredPosUnique = [&](int64_t pos) {
    if (pos < 0 || pos >= static_cast<int64_t>(scored.size()))
      return;
    if (!llvm::is_contained(selectedScoredPos, pos))
      selectedScoredPos.push_back(pos);
  };

  if (wantTensorCoreBaseDiversity && !scored.empty()) {
    auto pickTcScoredPosBy = [&](auto keyFn, bool preferSmaller) -> int64_t {
      int64_t bestPos = -1;
      double bestKey = preferSmaller ? std::numeric_limits<double>::infinity()
                                     : -std::numeric_limits<double>::infinity();
      for (int64_t pos = 0; pos < static_cast<int64_t>(scored.size()); ++pos) {
        int idx = scored[static_cast<size_t>(pos)].idx;
        if (idx < 0 || idx >= static_cast<int>(base.size()))
          continue;
        const Candidate &cand = base[static_cast<size_t>(idx)];
        if (!isTensorCoreFriendlyBaseCandidate(cand))
          continue;
        double k = keyFn(cand);
        if (bestPos < 0 ||
            (preferSmaller ? (k < bestKey) : (k > bestKey))) {
          bestPos = pos;
          bestKey = k;
        }
      }
      return bestPos;
    };

    int64_t tcByEstimate =
        pickTcScoredPosBy([](const Candidate &c) { return c.score; },
                          /* preferSmaller=*/true);
    int64_t tcByArea = pickTcScoredPosBy(
        [](const Candidate &c) {
          if (c.tileM <= 0 || c.tileN <= 0)
            return 0.0;
          return static_cast<double>(c.tileM) * static_cast<double>(c.tileN);
        },
        /* preferSmaller=*/false);
    int64_t tcByRowExtent = pickTcScoredPosBy(
        [&](const Candidate &c) {
          if (c.tileN <= 0)
            return std::numeric_limits<double>::infinity();
          if (c.tileN >= sgMaxRowReductionExtent)
            return 0.0;
          return static_cast<double>(sgMaxRowReductionExtent - c.tileN);
        },
        /* preferSmaller=*/true);

    pushScoredPosUnique(tcByEstimate);
    pushScoredPosUnique(tcByArea);
    pushScoredPosUnique(tcByRowExtent);

    if (opts.tracer) {
      llvm::json::Object f;
      f["base_budget"] = baseBudget;
      f["selected_scored_positions"] =
          static_cast<int64_t>(selectedScoredPos.size());
      f["tc_by_estimate"] = tcByEstimate;
      f["tc_by_area"] = tcByArea;
      f["tc_by_row_extent"] = tcByRowExtent;
      f["row_reduction_extent"] = sgMaxRowReductionExtent;
      f["tc_thread_row_extent"] = sgTcRowReductionExtentForThreads;
      opts.tracer->event("paper.subgraph_tiling.base_tc_diversity",
                         std::move(f), /*isVerbose=*/true);
    }
  }

  // SubGraphTiling 缓存：避免对相同 root tile + rstep map 重复传播
  // 当 GraphConnecting 反复评估相似子图时。
  struct PropCacheVal {
    TileGraph g;
    int64_t fpBytes = 0;
  };
  std::unordered_map<uint64_t, PropCacheVal> propCache;
  propCache.reserve(128);
  const int64_t reduceExpandPropCacheMax = std::max<int64_t>(
      0, getEnvInt64OrDefault("WELDER_REDUCE_EXPAND_PROP_CACHE_MAX",
                              /*default=*/256));
  auto mixHash = [](uint64_t h, uint64_t x) -> uint64_t {
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  };

  // 对每个候选：Propagate -> MemFootprint(shared) 过滤 -> MemTraffic(global) -> 时延估计
  bool stop = false;
  const bool matmulSoftmaxLikeOuter = isMatmulSoftmaxLikeSubgraph(graph, sg);
  const bool matmulSoftmaxLikeOuterF16 =
      matmulSoftmaxLikeOuter && opts.arch.elementBytes <= 2;
  const int64_t defaultMaxSharedTilesProfiled =
      matmulSoftmaxLikeOuterF16
          ? ((opts.codegenSearch.enable || opts.enableRegisterLevelSchedule) ? 8
                                                                              : 6)
          : (matmulSoftmaxLikeOuter
                 ? 12
                 : ((opts.codegenSearch.enable || opts.enableRegisterLevelSchedule)
                        ? 8
                        : 4));
	  const int64_t maxSharedTilesProfiled = profileEnabledForSubgraph
	                                             ? std::max<int64_t>(
	                                                   0, getEnvInt64OrDefault(
	                                                          "WELDER_PROFILE_MAX_SHARED_TILES",
	                                                          defaultMaxSharedTilesProfiled))
	                                             : 0;
  const bool ensureGlobalClassAnchors =
      opts.codegenSearch.enable &&
      (getEnvInt64OrDefault("WELDER_PROFILE_ENSURE_GLOBAL_CLASS_ANCHORS", 1) !=
       0);
  const bool wantGlobalAsyncAnchor =
      ensureGlobalClassAnchors &&
      llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
  const bool wantGlobalPipeAnchor =
      ensureGlobalClassAnchors &&
      llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
  const bool wantGlobalWaitGroupAnchor =
      ensureGlobalClassAnchors &&
      llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
  const bool wantGlobalNonTensorCoreAnchor =
      ensureGlobalClassAnchors &&
      (getEnvInt64OrDefault("WELDER_PROFILE_ENSURE_GLOBAL_NON_TC_ANCHOR", 1) !=
       0) &&
      (llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true) ||
       llvm::is_contained(opts.codegenSearch.enableTensorCoreTf32, true));
  const bool wantGlobalTensorCoreF16Anchor =
      ensureGlobalClassAnchors &&
      llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true);
  bool hasGlobalAsyncAnchor = false;
  bool hasGlobalPipeAnchor = false;
  bool hasGlobalWaitGroupAnchor = false;
  bool hasGlobalNonTensorCoreAnchor = false;
  bool hasGlobalTensorCoreF16Anchor = false;
  PaperScheduleCandidate globalAsyncAnchor;
  PaperScheduleCandidate globalPipeAnchor;
  PaperScheduleCandidate globalWaitGroupAnchor;
  PaperScheduleCandidate globalNonTensorCoreAnchor;
  PaperScheduleCandidate globalTensorCoreF16Anchor;
  struct SharedStagePreProbeCacheEntry {
    Candidate rootCand;
    TileGraph g;
    PaperSubgraph sgAfterCuts;
    int64_t fpBytes = 0;
  };
  llvm::DenseMap<int64_t, SharedStagePreProbeCacheEntry> sharedStagePreProbeCache;
  const int64_t nonStrictSharedStageProbeBudget = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_NONSTRICT_PREPRUNE_PROBE_BUDGET",
      /*default=*/96);
  int64_t nonStrictSharedStageHardKeepCap = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_NONSTRICT_PREPRUNE_CAP",
      /*default=*/0);
  if (nonStrictSharedStageHardKeepCap <= 0) {
    int64_t derivedCap = std::max<int64_t>(maxOut, desiredOut * 6);
	    if (profileEnabledForSubgraph && maxSharedTilesProfiled > 0)
	      derivedCap = std::max<int64_t>(derivedCap, maxSharedTilesProfiled * 2);
    nonStrictSharedStageHardKeepCap =
        std::max<int64_t>(16, std::min<int64_t>(baseBudget, derivedCap));
  }
  const int64_t enableSharedStagePrePruneNonStrict = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_NONSTRICT_PREPRUNE_ENABLE", /*default=*/1);
  if (enableSharedStagePrePruneNonStrict != 0 &&
      selectedScoredPos.size() > 1 && opts.maxConnectLevel >= 3) {
    struct SharedStageProbe {
      int64_t scoredPos = -1;
      Candidate cand;
      TileGraph g;
      PaperSubgraph sgAfterCuts;
      double prio = std::numeric_limits<double>::infinity();
    };
    llvm::SmallVector<SharedStageProbe, 128> sharedStageProbes;
    sharedStageProbes.reserve(selectedScoredPos.size());
    int64_t probedScoredPos = 0;
    for (int64_t scoredPos : selectedScoredPos) {
      if (nonStrictSharedStageProbeBudget > 0 &&
          probedScoredPos >= nonStrictSharedStageProbeBudget)
        break;
      ++probedScoredPos;
      if (scoredPos < 0 || scoredPos >= static_cast<int64_t>(scored.size()))
        continue;
      const ScoredIdx &sidx = scored[static_cast<size_t>(scoredPos)];
      if (sidx.idx < 0 || sidx.idx >= static_cast<int>(base.size()))
        continue;
      Candidate rootCand = base[static_cast<size_t>(sidx.idx)];
      applySwapHint(rootCand);
      auto parExtOpt = buildRootParallelExtents2Level(sinkOp, rootCand, opts);
      if (!parExtOpt)
        continue;

      std::vector<std::vector<int64_t>> reduceTilesByNode = baseReduceTiles;
      TileGraph g = graph;
      syncCutFlagFromConnectLevel(g);

      llvm::ArrayRef<int64_t> sinkRed;
      if (sinkNodeIdx >= 0 &&
          static_cast<size_t>(sinkNodeIdx) < reduceTilesByNode.size())
        sinkRed = reduceTilesByNode[static_cast<size_t>(sinkNodeIdx)];

      auto rootTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
          sinkOp, *parExtOpt, sinkRed, /*defaultReductionTile=*/0);
      if (!rootTileOpt)
        continue;

      TilePropagationOptions popts;
      popts.defaultReductionTile = 0;
      popts.reductionTilesByNode = &reduceTilesByNode;
      popts.enableCutEdges = true;
      popts.resetCutEdges = false;

      TilePropagationResult pr =
          propagateTilesBackward(g, sinkNodeIdx, *rootTileOpt, inference, popts);
      if (!pr.success)
        continue;

      cutEdgesOnSwapXYConflict(g, sg, sharedMinLevelExclusive);
      PaperSubgraph sgAfterCuts =
          extractSubgraphByConnectLevel(g, sinkNodeIdx, sharedMinLevelExclusive);

      int64_t fpBytes = computeSharedFootprintBestFitPaper(
          g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
          sharedMinLevelExclusive, sharedMaxLevelInclusive,
          rootCand.workgroupPadLastDim, rootCand.workgroupPadLastDimMatmulOnly,
          rootCand.workgroupMultiBufferDepth, &rootCand);
      if (fpBytes < 0 || fpBytes > opts.arch.smemBytes)
        continue;

      rootCand.estFootprintBytes = std::max<int64_t>(int64_t{0}, fpBytes);
      SharedStageProbe probe;
      probe.scoredPos = scoredPos;
      probe.cand = rootCand;
      probe.g = std::move(g);
      probe.sgAfterCuts = std::move(sgAfterCuts);
      probe.prio = sidx.score;
      sharedStageProbes.push_back(std::move(probe));
    }

    if (sharedStageProbes.size() > 1) {
      llvm::SmallVector<int64_t, 64> keepProbeIdxs =
          selectRecursiveStageTopKSharedCandidateIndices(
              static_cast<int64_t>(sharedStageProbes.size()),
              [&](int64_t i) -> const Candidate & {
                return sharedStageProbes[static_cast<size_t>(i)].cand;
              },
              [&](int64_t i) -> const TileGraph & {
                return sharedStageProbes[static_cast<size_t>(i)].g;
              },
              [&](int64_t i) -> const PaperSubgraph & {
                return sharedStageProbes[static_cast<size_t>(i)].sgAfterCuts;
              },
              [&](int64_t i) -> double {
                return sharedStageProbes[static_cast<size_t>(i)].prio;
              },
              opts, inference, scheduleLevels,
              nonStrictSharedStageHardKeepCap);

      if (!keepProbeIdxs.empty() &&
          keepProbeIdxs.size() < sharedStageProbes.size()) {
        llvm::SmallDenseSet<int64_t, 128> keepScoredPosSet;
        for (int64_t probeIdx : keepProbeIdxs) {
          if (probeIdx < 0 ||
              probeIdx >= static_cast<int64_t>(sharedStageProbes.size()))
            continue;
          keepScoredPosSet.insert(
              sharedStageProbes[static_cast<size_t>(probeIdx)].scoredPos);
        }
        llvm::SmallVector<int64_t, 64> filteredScoredPos;
        filteredScoredPos.reserve(selectedScoredPos.size());
        for (int64_t scoredPos : selectedScoredPos) {
          if (keepScoredPosSet.count(scoredPos) > 0)
            filteredScoredPos.push_back(scoredPos);
        }
        if (!filteredScoredPos.empty()) {
          if (opts.tracer) {
            llvm::json::Object f;
            f["before"] = static_cast<int64_t>(selectedScoredPos.size());
            f["probed"] = static_cast<int64_t>(sharedStageProbes.size());
            f["after"] = static_cast<int64_t>(filteredScoredPos.size());
            f["probe_budget"] = nonStrictSharedStageProbeBudget;
            f["hard_keep_cap"] = nonStrictSharedStageHardKeepCap;
            opts.tracer->event("paper.recursive_shared_stage_nonstrict_preprune",
                               std::move(f), /*isVerbose=*/true);
          }
          selectedScoredPos = std::move(filteredScoredPos);
        }
      }
    }
    for (SharedStageProbe &probe : sharedStageProbes) {
      SharedStagePreProbeCacheEntry cacheEntry;
      cacheEntry.rootCand = std::move(probe.cand);
      cacheEntry.g = std::move(probe.g);
      cacheEntry.sgAfterCuts = std::move(probe.sgAfterCuts);
      cacheEntry.fpBytes = cacheEntry.rootCand.estFootprintBytes;
      sharedStagePreProbeCache[probe.scoredPos] = std::move(cacheEntry);
    }
  }
  int64_t profiledSharedTiles = 0;
  bool emittedSharedTileCapEvent = false;
  const int64_t defaultMaxProfileCompilesTotal =
      matmulSoftmaxLikeOuterF16
          ? ((opts.codegenSearch.enable || opts.enableRegisterLevelSchedule) ? 64
                                                                              : 40)
          : (matmulSoftmaxLikeOuter
                 ? ((opts.codegenSearch.enable ||
                     opts.enableRegisterLevelSchedule)
                        ? 96
                        : 48)
                 : ((opts.codegenSearch.enable ||
                     opts.enableRegisterLevelSchedule)
                        ? 64
                        : 32));
	  const int64_t maxProfileCompilesTotal =
	      profileEnabledForSubgraph
	          ? std::max<int64_t>(0, getEnvInt64OrDefault(
	                                     "WELDER_PROFILE_MAX_COMPILES_TOTAL",
	                                     defaultMaxProfileCompilesTotal))
	          : 0;
  int64_t profileCompilesTotal = 0;
  bool emittedProfileCompileTotalCapEvent = false;
  bool stopOnProfileCompileTotalCap = false;
  for (int64_t scoredPos : selectedScoredPos) {
    if (stopOnProfileCompileTotalCap)
      break;
	    if (profileEnabledForSubgraph && maxProfileCompilesTotal > 0 &&
	        profileCompilesTotal >= maxProfileCompilesTotal) {
      if (opts.tracer && !emittedProfileCompileTotalCapEvent) {
        llvm::json::Object f;
        f["cap"] = maxProfileCompilesTotal;
        f["compiles"] = profileCompilesTotal;
        f["selected_scored_positions"] =
            static_cast<int64_t>(selectedScoredPos.size());
        opts.tracer->event("profile.compile_total_cap", std::move(f),
                           /* isVerbose=*/true);
        emittedProfileCompileTotalCapEvent = true;
      }
      break;
    }
	    if (profileEnabledForSubgraph && maxSharedTilesProfiled > 0 &&
	        profiledSharedTiles >= maxSharedTilesProfiled) {
      if (opts.tracer && !emittedSharedTileCapEvent) {
        llvm::json::Object f;
        f["cap"] = maxSharedTilesProfiled;
        f["processed"] = profiledSharedTiles;
        f["selected_scored_positions"] =
            static_cast<int64_t>(selectedScoredPos.size());
        opts.tracer->event("profile.shared_tile_cap", std::move(f),
                           /* isVerbose=*/true);
        emittedSharedTileCapEvent = true;
      }
      break;
    }
    if (scoredPos < 0 || scoredPos >= static_cast<int64_t>(scored.size()))
      continue;
    const ScoredIdx &sidx = scored[static_cast<size_t>(scoredPos)];
    const Candidate &c0 = base[sidx.idx];

    // root tile：将 tileM/tileN 映射到 sink op 的前两维并行轴。
    Candidate rootCand = c0;
    applySwapHint(rootCand);
    const SharedStagePreProbeCacheEntry *preProbeEntry = nullptr;
    if (auto it = sharedStagePreProbeCache.find(scoredPos);
        it != sharedStagePreProbeCache.end()) {
      rootCand = it->second.rootCand;
      preProbeEntry = &it->second;
    }
    auto parExtOpt = buildRootParallelExtents2Level(sinkOp, rootCand, opts);
    if (!parExtOpt)
      continue;

    // 从 coalescing 选出的每算子归约步长开始（论文 rstep_map）。
    std::vector<std::vector<int64_t>> reduceTilesByNode = baseReduceTiles;

    // 辅助函数：对当前 rstep map 运行传播并计算 shared footprint。
    auto propagateAndFootprint = [&](std::vector<std::vector<int64_t>> &rmap)
        -> std::optional<std::pair<TileGraph, int64_t>> {
      // 缓存键：sink + root 并行 extent + 每节点归约步长。
      uint64_t key = mixHash(0, static_cast<uint64_t>(sinkNodeIdx));
      for (int64_t v : *parExtOpt)
        key = mixHash(key, static_cast<uint64_t>(v));
      key = mixHash(key, static_cast<uint64_t>(rootCand.workgroupPadLastDim));
      key = mixHash(key, static_cast<uint64_t>(rootCand.workgroupPadLastDimMatmulOnly));
      key = mixHash(key, static_cast<uint64_t>(rootCand.workgroupMultiBufferDepth));
      key = mixHash(key, static_cast<uint64_t>(rootCand.swapBlockDims));
      // 将子图内边的 connectLevel 状态纳入缓存键，避免不同
      // GraphConnecting 试探发生冲突。
      for (const TileGraphEdge &e : graph.edges) {
        if (e.src < 0 || e.dst < 0)
          continue;
        if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
          continue;
        key = mixHash(key, static_cast<uint64_t>(e.src));
        key = mixHash(key, static_cast<uint64_t>(e.dst));
        key = mixHash(key, static_cast<uint64_t>(e.connectLevel));
      }
      for (int nodeIdx : sg.nodes) {
        key = mixHash(key, static_cast<uint64_t>(nodeIdx));
        if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= rmap.size())
          continue;
        const auto &rs = rmap[static_cast<size_t>(nodeIdx)];
        key = mixHash(key, static_cast<uint64_t>(rs.size()));
        for (int64_t v : rs)
          key = mixHash(key, static_cast<uint64_t>(v));
      }
      if (auto it = propCache.find(key); it != propCache.end()) {
        TileGraph g = it->second.g;
        return std::make_optional(std::make_pair(std::move(g), it->second.fpBytes));
      }

      TileGraph g = graph; // 拷贝
      syncCutFlagFromConnectLevel(g);

      llvm::ArrayRef<int64_t> sinkRed;
      if (sinkNodeIdx >= 0 && static_cast<size_t>(sinkNodeIdx) < rmap.size())
        sinkRed = rmap[sinkNodeIdx];

      auto rootTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
          sinkOp, *parExtOpt, sinkRed, /*defaultReductionTile=*/0);
      if (!rootTileOpt)
        return std::nullopt;

      TilePropagationOptions popts;
      popts.defaultReductionTile = 0; // 显式设置每节点归约 tile
      popts.reductionTilesByNode = &rmap;
      // 论文对齐的鲁棒性：若冲突则 cut（回退到 global），而非失败。
      popts.enableCutEdges = true;
      popts.resetCutEdges = false; // cut 由 connectLevel 决定

      TilePropagationResult pr =
          propagateTilesBackward(g, sinkNodeIdx, *rootTileOpt, inference, popts);
      if (!pr.success)
        return std::nullopt;

      cutEdgesOnSwapXYConflict(g, sg, sharedMinLevelExclusive);
      PaperSubgraph sgAfterCuts =
          extractSubgraphByConnectLevel(g, sinkNodeIdx,
                                        sharedMinLevelExclusive);

		      int64_t fpBytes = computeSharedFootprintBestFitPaper(
		          g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
		          sharedMinLevelExclusive, sharedMaxLevelInclusive,
		          rootCand.workgroupPadLastDim, rootCand.workgroupPadLastDimMatmulOnly,
		          rootCand.workgroupMultiBufferDepth, &rootCand);
	      if (fpBytes < 0)
	        return std::nullopt;
      if (reduceExpandPropCacheMax > 0 &&
          static_cast<int64_t>(propCache.size()) < reduceExpandPropCacheMax) {
        PropCacheVal val;
        val.g = g;
        val.fpBytes = fpBytes;
        propCache.emplace(key, std::move(val));
      }
      return std::make_optional(std::make_pair(std::move(g), fpBytes));
    };

    std::optional<std::pair<TileGraph, int64_t>> curOpt;
    if (preProbeEntry) {
      TileGraph g0 = preProbeEntry->g;
      curOpt = std::make_optional(
          std::make_pair(std::move(g0), preProbeEntry->fpBytes));
    } else {
      curOpt = propagateAndFootprint(reduceTilesByNode);
    }
    if (!curOpt)
      continue;
    TileGraph g = std::move(curOpt->first);
    int64_t baseFpBytes = curOpt->second;
    if (baseFpBytes > opts.arch.smemBytes)
      continue;

    PaperSubgraph sgAfterCuts;
    if (preProbeEntry)
      sgAfterCuts = preProbeEntry->sgAfterCuts;
    else
      sgAfterCuts =
          extractSubgraphByConnectLevel(g, sinkNodeIdx, sharedMinLevelExclusive);

    // 可选：在 shared footprint 约束下扩展 reduce 步长。
    if (opts.paperExpandReductionTile) {
      const int64_t reduceExpandScoreCacheEnable = getEnvInt64OrDefault(
          "WELDER_REDUCE_EXPAND_SCORE_CACHE_ENABLE", /*default=*/1);
      llvm::DenseMap<uint64_t, double> reduceExpandScoreCache;
      auto buildReduceExpandScoreKey = [&](int nodeIdx, ArrayRef<int64_t> parExt,
                                           ArrayRef<int64_t> redExt) -> uint64_t {
        uint64_t key = mixHash(0, static_cast<uint64_t>(nodeIdx));
        key = mixHash(key, static_cast<uint64_t>(parExt.size()));
        for (int64_t v : parExt)
          key = mixHash(key, static_cast<uint64_t>(v));
        key = mixHash(key, static_cast<uint64_t>(redExt.size()));
        for (int64_t v : redExt)
          key = mixHash(key, static_cast<uint64_t>(v));
        return key;
      };
      TileGraph curG = g;
      int64_t curFpBytes = baseFpBytes;

      llvm::SmallVector<int, 16> nodeOrder =
          topoSortSubgraphByConnectedEdges(curG, sgAfterCuts,
                                           sharedMinLevelExclusive);

      auto isGlobalReadOperand = [&](const TileGraph &g0, int nodeIdx,
                                     int operandIdx) -> bool {
        for (int edgeIdx : g0.nodes[nodeIdx].inEdges) {
          if (edgeIdx < 0 || edgeIdx >= static_cast<int>(g0.edges.size()))
            continue;
          const TileGraphEdge &e = g0.edges[edgeIdx];
          if (e.dstOperand != operandIdx)
            continue;
          if (e.connectLevel > sharedMinLevelExclusive)
            return false;
        }
        return true;
      };

      auto externalReadCoalescedScore = [&](linalg::LinalgOp op, int nodeIdx,
                                            ArrayRef<int64_t> parExt,
                                            ArrayRef<int64_t> redExt) -> double {
        if (reduceExpandScoreCacheEnable != 0) {
          uint64_t key = buildReduceExpandScoreKey(nodeIdx, parExt, redExt);
          if (auto it = reduceExpandScoreCache.find(key);
              it != reduceExpandScoreCache.end())
            return it->second;
        }
        auto tileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
            op, parExt, redExt, /*defaultReductionTile=*/0);
        if (!tileOpt)
          return 0.0;
        auto fpOpt = inference.infer(op.getOperation(), *tileOpt);
        if (!fpOpt)
          return 0.0;

        double s = 0.0;
        int numInputs = op.getNumDpsInputs();
        for (int i = 0; i < numInputs; ++i) {
          if (!isGlobalReadOperand(curG, nodeIdx, i))
            continue;
          if (i < 0 || i >= static_cast<int>(fpOpt->perOperand.size()))
            continue;
          llvm::SmallVector<int64_t, 4> fullShape =
              getStaticShapeOrUnknown(op.getDpsInputs()[i]);
          if (fullShape.empty())
            continue;
          int64_t cf = coalescedFactor(
              ArrayRef<int64_t>(fpOpt->perOperand[i].shape), fullShape);
          if (cf <= 0)
            continue;
          s += static_cast<double>(cf);
        }
        if (reduceExpandScoreCacheEnable != 0) {
          uint64_t key = buildReduceExpandScoreKey(nodeIdx, parExt, redExt);
          reduceExpandScoreCache[key] = s;
        }
        return s;
      };

      for (int nodeIdx : nodeOrder) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(curG.nodes.size()))
          continue;
        if (!sgAfterCuts.inSet.contains(nodeIdx))
          continue;
        Operation *op0 = curG.nodes[nodeIdx].op;
        auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
        if (!op)
          continue;
        if (op.getNumReductionLoops() <= 0)
          continue;
        if (static_cast<size_t>(nodeIdx) >= reduceTilesByNode.size())
          continue;

        std::vector<int64_t> baseRed = reduceTilesByNode[nodeIdx];
        if (baseRed.empty())
          continue;

        llvm::SmallVector<int64_t, 4> redFull = getReductionLoopFullRanges(op);
        if (redFull.size() != baseRed.size())
          continue;

        std::vector<int64_t> curRed = baseRed;

        // 从已传播 required tile 中提取当前并行 extent。
        llvm::SmallVector<int64_t, 8> parExt;
        parExt.reserve(op.getNumParallelLoops());
        if (!curG.nodes[nodeIdx].hasRequiredTile)
          continue;
        const OpTile &rt = curG.nodes[nodeIdx].requiredTile;
        auto iters = op.getIteratorTypesArray();
        for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
          if (iters[i] == utils::IteratorType::parallel)
            parExt.push_back(rt.loopExtents[i]);
        }
        if (static_cast<int64_t>(parExt.size()) != op.getNumParallelLoops())
          continue;

        // 构建每个轴的步长列表（当前步长的倍数）。
        llvm::SmallVector<llvm::SmallVector<int64_t, 64>, 4> steps;
        steps.reserve(redFull.size());
        for (size_t ax = 0; ax < redFull.size(); ++ax) {
          llvm::SmallVector<int64_t, 64> fs =
              factorsOrPowersForReduceStep(redFull[ax]);
          int64_t base = std::max<int64_t>(1, baseRed[ax]);
          fs.erase(std::remove_if(fs.begin(), fs.end(),
                                  [&](int64_t v) { return (v % base) != 0; }),
                   fs.end());
          if (fs.empty())
            fs.push_back(base);
          steps.push_back(std::move(fs));
        }

        llvm::SmallVector<int, 4> ids;
        ids.reserve(steps.size());
        for (size_t ax = 0; ax < steps.size(); ++ax) {
          auto it = llvm::find(steps[ax], curRed[ax]);
          ids.push_back(it == steps[ax].end()
                            ? 0
                            : static_cast<int>(it - steps[ax].begin()));
        }

        auto currentScore = externalReadCoalescedScore(op, nodeIdx, parExt, curRed);

        while (true) {
          llvm::SmallVector<int, 4> bestIds = ids;
          double bestScore = currentScore;

          for (size_t ax = 0; ax < ids.size(); ++ax) {
            if (ids[ax] + 1 >= static_cast<int>(steps[ax].size()))
              continue;
            llvm::SmallVector<int64_t, 4> trialRed(curRed.begin(), curRed.end());
            trialRed[ax] = steps[ax][ids[ax] + 1];
            if (opts.requirePerfectTiling && (redFull[ax] % trialRed[ax] != 0))
              continue;
            double s = externalReadCoalescedScore(op, nodeIdx, parExt, trialRed);
            if (s > bestScore) {
              bestScore = s;
              bestIds = ids;
              bestIds[ax] += 1;
            }
          }

          if (bestScore <= currentScore)
            break;

          // 暂时应用最佳扩展并重新执行传播 + footprint 计算。
          std::vector<std::vector<int64_t>> trialMap = reduceTilesByNode;
          for (size_t ax = 0; ax < bestIds.size(); ++ax)
            trialMap[nodeIdx][ax] = steps[ax][bestIds[ax]];

          auto trialOpt = propagateAndFootprint(trialMap);
          if (!trialOpt)
            break;
          if (trialOpt->second > opts.arch.smemBytes)
            break;

          // 接受。
          reduceTilesByNode = std::move(trialMap);
          curG = std::move(trialOpt->first);
          curFpBytes = trialOpt->second;
          sgAfterCuts = extractSubgraphByConnectLevel(
              curG, sinkNodeIdx, sharedMinLevelExclusive);
          ids = bestIds;
          curRed = reduceTilesByNode[nodeIdx];
          currentScore = bestScore;
        }
      }

      g = std::move(curG);
      baseFpBytes = curFpBytes;
    }

    // 在可能的归约 tile 变更/cut 后刷新子图。
    sgAfterCuts =
        extractSubgraphByConnectLevel(g, sinkNodeIdx, sharedMinLevelExclusive);

    // 对齐 legacy tileK 以用于 profiling/编译（sink 可能是 epilogue）。
    int reduceNodeIdx = sinkNodeIdx;
    if (sinkOp.getNumReductionLoops() == 0) {
      reduceNodeIdx = -1;
      llvm::SmallVector<int, 16> stack;
      llvm::SmallDenseSet<int, 32> visitedNodes;
      stack.push_back(sinkNodeIdx);
      visitedNodes.insert(sinkNodeIdx);
      while (!stack.empty() && visitedNodes.size() < 32) {
        int cur = stack.pop_back_val();
        if (cur < 0 || cur >= static_cast<int>(g.nodes.size()))
          continue;
        for (int edgeIdx : g.nodes[cur].inEdges) {
          if (edgeIdx < 0 || edgeIdx >= static_cast<int>(g.edges.size()))
            continue;
          int src = g.edges[edgeIdx].src;
          if (src < 0 || src >= static_cast<int>(g.nodes.size()))
            continue;
          if (!visitedNodes.insert(src).second)
            continue;
          Operation *op0 = g.nodes[src].op;
          auto lop = dyn_cast_or_null<linalg::LinalgOp>(op0);
          if (lop && lop.getNumReductionLoops() > 0) {
            reduceNodeIdx = src;
            stack.clear();
            break;
          }
          stack.push_back(src);
        }
      }
      if (reduceNodeIdx < 0)
        reduceNodeIdx = sinkNodeIdx;
    }

    Candidate baseShared = c0;
    applySwapHint(baseShared);
    baseShared.estFootprintBytes = std::max<int64_t>(0, baseFpBytes);
    baseShared.tileK = 1;
    if (reduceNodeIdx >= 0 &&
        static_cast<size_t>(reduceNodeIdx) < reduceTilesByNode.size() &&
        !reduceTilesByNode[reduceNodeIdx].empty() &&
        reduceTilesByNode[reduceNodeIdx].front() > 0) {
      baseShared.tileK = reduceTilesByNode[reduceNodeIdx].front();
    } else if (c0.tileK > 0) {
      baseShared.tileK = c0.tileK;
    }

    // MemTraffic(global)：子图对 global 的读写（图外 + 切边）。
    SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
        g, sgAfterCuts, sharedMinLevelExclusive, sharedMaxLevelInclusive,
        baseShared.workgroupPadLastDim, baseShared.workgroupPadLastDimMatmulOnly,
        baseShared.workgroupSwizzleXor);
    MemTrafficBreakdown mt = computeMemTrafficForSubgraph(
        g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
        sharedMinLevelExclusive,
        /* applyCoalescingPenalty=*/opts.enableCoalescingPenalty, &layout);
    Traffic t = opts.enableCoalescingPenalty ? mt.mem : mt.raw;

    int64_t blocksTotal = std::max<int64_t>(1, baseShared.blocksTotal);
    if (sinkNodeIdx >= 0 && sinkNodeIdx < static_cast<int>(g.nodes.size()) &&
        g.nodes[sinkNodeIdx].hasRequiredTile) {
      llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
      if (static_cast<int64_t>(ranges.size()) == sinkOp.getNumLoops() &&
          static_cast<int64_t>(
              g.nodes[sinkNodeIdx].requiredTile.loopExtents.size()) ==
              sinkOp.getNumLoops()) {
        auto iters = sinkOp.getIteratorTypesArray();
        int64_t bt = 1;
        for (int64_t i = 0; i < sinkOp.getNumLoops(); ++i) {
          if (iters[i] != utils::IteratorType::parallel)
            continue;
          int64_t full = ranges[i];
          int64_t t0 = g.nodes[sinkNodeIdx].requiredTile.loopExtents[i];
          if (full == ShapedType::kDynamic || full <= 0 || t0 <= 0) {
            bt = 1;
            break;
          }
          int64_t tiles =
              opts.requirePerfectTiling ? (full / t0) : ceilDiv(full, t0);
          tiles = std::max<int64_t>(1, tiles);
          if (bt <= (std::numeric_limits<int64_t>::max() / tiles))
            bt *= tiles;
          else {
            bt = std::numeric_limits<int64_t>::max();
            break;
          }
        }
        blocksTotal = std::max<int64_t>(1, bt);
        baseShared.blocksTotal = blocksTotal;
      }
    }

    // 将 register 层 tile（每线程 tile）作为内层递归枚举
    // 层（论文“SubGraphTiling”递归骨架）。
    llvm::SmallVector<std::pair<int64_t, int64_t>, 16> regTiles;
    if (opts.enableRegisterLevelSchedule) {
      std::vector<int64_t> threadList = opts.candidatesThreadMN;
      if (threadList.empty())
        threadList.push_back(1);
      llvm::sort(threadList);
      threadList.erase(std::unique(threadList.begin(), threadList.end()),
                       threadList.end());
      for (int64_t ttm : threadList) {
        for (int64_t ttn : threadList) {
          if (ttm <= 0 || ttn <= 0)
            continue;
          if (c0.tileM <= 0 || c0.tileN <= 0)
            continue;
          if (c0.tileM % ttm != 0 || c0.tileN % ttn != 0)
            continue;
          int64_t blockDimX =
              baseShared.swapBlockDims ? (c0.tileM / ttm) : (c0.tileN / ttn);
          int64_t blockDimY =
              baseShared.swapBlockDims ? (c0.tileN / ttn) : (c0.tileM / ttm);
          if (blockDimX <= 0 || blockDimY <= 0 ||
              blockDimX * blockDimY > 1024)
            continue;
          regTiles.push_back({ttm, ttn});
        }
      }
	    } else {
	      regTiles.push_back({0, 0});
	    }
	    const int64_t maxRowReductionExtentForTc =
	        computeTcRowReductionExtentForThreadMapping(g, sgAfterCuts);
    const bool subgraphHasMatmulForVariants =
        subgraphHasMatmulOp(g, sgAfterCuts);
    const bool matmulSoftmaxLikeSubgraph =
        isMatmulSoftmaxLikeSubgraph(g, sgAfterCuts) ||
        (subgraphHasMatmulForVariants && graphHasMatmulSoftmaxLikePattern(g));
    const bool tensorCoreLayoutFeasible =
        isTensorCoreStrideLayoutFeasibleForSubgraph(g, sgAfterCuts);
	    if (!regTiles.empty()) {
	      regTiles = selectRecursiveStageTopKRegisterTiles(
	          regTiles, baseShared, g, sgAfterCuts, opts, inference,
	          scheduleLevels, maxRowReductionExtentForTc);
	    }

	    // 在相同 shared 层 tile 的传播结果上评估 (shared->register) 变体
	    // 。
    //
    // 注意：在非严格模式下，若启用 register 层 tile 或 codegen 搜索，
    // 对每个内层变体都做 profiling 会非常慢，因此我们：
    //   1）用论文代价模型给所有内层变体打分，
	    //   2）只 profile 一小部分尽力而为的子集，
	    //   3）保留该 shared 层 tile 的最佳结果（递归骨架）。
	    std::vector<PaperScheduleCandidate> pending;
	    pending.reserve(regTiles.size());

	    for (auto [ttm, ttn] : regTiles) {
	      Candidate baseReg = baseShared;
	      baseReg.threadTileM = ttm;
      baseReg.threadTileN = ttn;

	      std::vector<Candidate> variants =
	          expandCandidatesWithCodegenSearch(
                baseReg, codegenOpts, maxRowReductionExtentForTc,
                tensorCoreLayoutFeasible, matmulSoftmaxLikeSubgraph,
                subgraphHasMatmulForVariants);
	      if (variants.empty())
	        continue;
	      if (variants.size() > 1) {
	        llvm::SmallVector<int64_t, 64> keepVariantIdxs =
	            selectRecursiveStageTopKCandidateIndices(
	                static_cast<int64_t>(variants.size()),
	                [&](int64_t i) -> const Candidate & {
	                  return variants[static_cast<size_t>(i)];
	                },
	                g, sgAfterCuts, opts, inference, scheduleLevels,
	                maxRowReductionExtentForTc);
	        if (!keepVariantIdxs.empty() &&
	            keepVariantIdxs.size() < variants.size()) {
	          std::vector<Candidate> filteredVariants;
	          filteredVariants.reserve(keepVariantIdxs.size());
	          for (int64_t i : keepVariantIdxs) {
	            if (i < 0 || i >= static_cast<int64_t>(variants.size()))
	              continue;
	            filteredVariants.push_back(
	                std::move(variants[static_cast<size_t>(i)]));
	          }
	          variants = std::move(filteredVariants);
	        }
	      }

      for (Candidate cand : variants) {
        applySwapHint(cand);
        // MemFootprint(shared)：用 bestfit 计算子图在 shared 中的 footprint（对齐论文 §3.1）。
        // 子图连边阈值由 sharedMinLevelExclusive/sharedMaxLevelInclusive 决定。
		        int64_t fpBytes = computeSharedFootprintBestFitPaper(
		            g, sgAfterCuts, opts.arch, inference, opts.requirePerfectTiling,
		            sharedMinLevelExclusive, sharedMaxLevelInclusive,
		            cand.workgroupPadLastDim,
		            cand.workgroupPadLastDimMatmulOnly,
		            cand.workgroupMultiBufferDepth, &cand);
	        if (fpBytes < 0 || fpBytes > opts.arch.smemBytes)
	          continue;

        // Occupancy / waves 估计（论文对齐骨架）：
        // - 受 blocks-per-SM（上界）、threads-per-SM 以及 shared
        //   每 block 的内存 footprint（含 padding/multi-buffering）和
        //   简化寄存器文件模型限制。
        int64_t blockThreads = estimateBlockThreadsForCandidate(
            cand, maxRowReductionExtentForTc);

        int64_t blocksPerSM = std::max<int64_t>(1, opts.arch.maxBlocksPerSM);
        int64_t byPartition =
            std::max<int64_t>(1, std::max<int64_t>(1, opts.arch.smPartition));
        int64_t byThreads =
            std::max<int64_t>(1, opts.arch.maxThreadsPerSM /
                                     std::max<int64_t>(1, blockThreads));
        int64_t warpSize = std::max<int64_t>(1, opts.arch.warpSize);
        int64_t warpsPerSM =
            std::max<int64_t>(1, opts.arch.maxThreadsPerSM / warpSize);
        int64_t warpsPerBlock =
            (std::max<int64_t>(1, blockThreads) + (warpSize - 1)) / warpSize;
        int64_t byWarps =
            std::max<int64_t>(1, warpsPerSM / std::max<int64_t>(1, warpsPerBlock));
        blocksPerSM =
            std::min<int64_t>(blocksPerSM,
                              std::min<int64_t>(byPartition,
                                                std::min(byThreads, byWarps)));

        // 寄存器压力模型（较粗略）：
        // - 累加器寄存器 ~ threadTileM*threadTileN
        // - 地址计算/加载/谓词等开销寄存器
	        int64_t regsAcc = 0;
	        if (cand.threadTileM > 0 && cand.threadTileN > 0)
	          regsAcc = cand.threadTileM * cand.threadTileN;
	        const RecursiveStageAggregate recursiveStageAgg =
	            estimateRecursiveStageAggregateForCandidate(
	                g, sgAfterCuts, opts, inference, scheduleLevels,
	                blockThreads, cand.workgroupPadLastDim,
	                cand.workgroupPadLastDimMatmulOnly,
	                cand.workgroupSwizzleXor);
	        if (!recursiveStageAgg.feasible) {
	          if (cand.feasibilityCode == 0)
	            cand.feasibilityCode = 25; // 递归阶段溢出
	          continue;
	        }
	        int64_t regsOverhead = 32;
	        regsOverhead += recursiveStageAgg.regReuseRegsPerThread;
	        if (cand.enableAsyncCopy)
	          regsOverhead += 8;
	        if (cand.enableSoftwarePipelining)
	          regsOverhead += 16;
        int64_t regsPerThread = std::max<int64_t>(1, regsAcc + regsOverhead);
        if (regsPerThread > opts.arch.maxRegistersPerThread) {
          if (cand.feasibilityCode == 0)
            cand.feasibilityCode = 20; // 寄存器溢出
          continue;
        }
        cand.estRegsPerThread = regsPerThread;
        bool isTensorCoreCand =
            cand.enableTensorCoreTf32 || cand.enableTensorCoreF16;
        if (cand.enableSoftwarePipelining) {
          int64_t regSoftCap =
              std::max<int64_t>(96, (opts.arch.maxRegistersPerThread * 3) / 8);
          if (isTensorCoreCand) {
            regSoftCap = std::max<int64_t>(
                80, (opts.arch.maxRegistersPerThread * 7) / 20);
          }
          if (regsPerThread >= regSoftCap &&
              (blocksPerSM <= 1 || isTensorCoreCand)) {
            if (cand.feasibilityCode == 0)
              cand.feasibilityCode = 21; // 流水线低占用/寄存器压力
            continue;
          }
        }
        int64_t regsPerBlock =
            regsPerThread * std::max<int64_t>(1, blockThreads);
        int64_t byRegs = blocksPerSM;
        if (regsPerBlock > 0)
          byRegs =
              std::max<int64_t>(1, opts.arch.maxRegistersPerSM / regsPerBlock);
        blocksPerSM =
            std::max<int64_t>(1, std::min<int64_t>(blocksPerSM, byRegs));

        int64_t bySmem = blocksPerSM;
        if (fpBytes > 0)
          bySmem =
              std::max<int64_t>(1, getMaxSmemUsageBytes(opts.arch) / fpBytes);
        blocksPerSM =
            std::max<int64_t>(1, std::min<int64_t>(blocksPerSM, bySmem));

        if (cand.enableSoftwarePipelining) {
          int64_t minPipeBlocksPerSM = isTensorCoreCand ? 2 : 1;
          if (isTensorCoreCand && cand.pipelineDepth >= 4)
            minPipeBlocksPerSM = 3;
          if (blocksPerSM < minPipeBlocksPerSM) {
            if (cand.feasibilityCode == 0)
              cand.feasibilityCode = 22; // 流水线 TensorCore 占用压力
            continue;
          }
        }
        if (cand.enableSoftwarePipelining &&
            cand.enableMatmulSoftmaxSharedReuseFusion &&
            cand.enableRowReductionChainReuseFusion) {
          const int64_t minMmSmPipeBlocks = std::max<int64_t>(
              1, getEnvInt64OrDefault(
                     isTensorCoreCand
                         ? "WELDER_MM_SM_TC_PIPE_MIN_BLOCKS_PER_SM"
                         : "WELDER_MM_SM_PIPE_MIN_BLOCKS_PER_SM",
                     isTensorCoreCand ? 3 : 2));
          const int64_t maxMmSmPipeRegs = std::max<int64_t>(
              1, getEnvInt64OrDefault(
                     isTensorCoreCand
                         ? "WELDER_MM_SM_TC_PIPE_MAX_REGS_PER_THREAD"
                         : "WELDER_MM_SM_PIPE_MAX_REGS_PER_THREAD",
                     isTensorCoreCand ? 96 : 112));
          int64_t effectiveMinMmSmPipeBlocks = minMmSmPipeBlocks;
          int64_t effectiveMaxMmSmPipeRegs = maxMmSmPipeRegs;
          const bool relaxTcPipeGuardForWaitGroup =
              isTensorCoreCand && cand.enableAsyncCopy &&
              cand.pipelineSetAsyncWaitGroups &&
              (getEnvInt64OrDefault(
                   "WELDER_MM_SM_TC_PIPE_RELAX_FOR_WAIT_GROUP", 1) != 0);
          if (relaxTcPipeGuardForWaitGroup) {
            const int64_t relaxedMinBlocks = std::max<int64_t>(
                1, getEnvInt64OrDefault(
                       "WELDER_MM_SM_TC_PIPE_MIN_BLOCKS_PER_SM_WAIT_GROUP",
                       2));
            const int64_t relaxedMaxRegs = std::max<int64_t>(
                1, getEnvInt64OrDefault(
                       "WELDER_MM_SM_TC_PIPE_MAX_REGS_PER_THREAD_WAIT_GROUP",
                       112));
            effectiveMinMmSmPipeBlocks =
                std::min(effectiveMinMmSmPipeBlocks, relaxedMinBlocks);
            effectiveMaxMmSmPipeRegs =
                std::max(effectiveMaxMmSmPipeRegs, relaxedMaxRegs);
          }
          if (blocksPerSM < effectiveMinMmSmPipeBlocks ||
              regsPerThread > effectiveMaxMmSmPipeRegs) {
            if (cand.feasibilityCode == 0)
              cand.feasibilityCode = 24; // mm-sm 防溢写流水线拒绝
            continue;
          }
        }

        int64_t concurrentBlocks =
            std::max<int64_t>(1, blocksPerSM * opts.arch.numSM);
        int64_t waves = ceilDiv(blocksTotal, concurrentBlocks);

        cand.blocksPerSM = blocksPerSM;
        cand.numWave = waves;

        // 论文对齐的光栅化（tensorcore、单算子、大 num_wave）。
        maybeApplyRasterizationTcPolicyPaper(sgAfterCuts, t, cand);

	        double sh2reg = recursiveStageAgg.sharedToRegBytes;
	        fillCostAndScoreFromPaperModel(cand, fpBytes, mt, sh2reg, opts);

        PaperScheduleCandidate pc;
        pc.cand = cand;
	        pc.sharedFootprintBytes = fpBytes;
	        pc.traffic = cand.traffic;
	        pc.estimatedLatency = cand.score;
	        pending.push_back(std::move(pc));
	      }
	    }
	    if (pending.size() > 1) {
	      llvm::SmallVector<int64_t, 64> keepIdxs =
	          selectRecursiveStageTopKCandidateIndices(
	              static_cast<int64_t>(pending.size()),
	              [&](int64_t i) -> const Candidate & {
	                return pending[static_cast<size_t>(i)].cand;
	              },
	              g, sgAfterCuts, opts, inference, scheduleLevels,
	              maxRowReductionExtentForTc);
	      if (!keepIdxs.empty() &&
	          keepIdxs.size() < static_cast<size_t>(pending.size())) {
	        std::vector<PaperScheduleCandidate> filtered;
	        filtered.reserve(keepIdxs.size());
	        for (int64_t i : keepIdxs) {
	          if (i < 0 || i >= static_cast<int64_t>(pending.size()))
	            continue;
	          filtered.push_back(std::move(pending[static_cast<size_t>(i)]));
	        }
	        pending = std::move(filtered);
	      }
	    }
	    if (pending.empty())
	      continue;
	    if (profileEnabledForSubgraph)
	      ++profiledSharedTiles;

    bool subgraphHasMatmul = false;
    if (sinkNodeIdx >= 0 && sinkNodeIdx < static_cast<int>(g.nodes.size()))
      subgraphHasMatmul = isa<linalg::MatmulOp>(g.nodes[sinkNodeIdx].op);
    if (!subgraphHasMatmul && subgraphHasMatmulOp(g, sgAfterCuts))
      subgraphHasMatmul = true;

    const bool searchWantsTensorCore =
        opts.codegenSearch.enable &&
        (llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true) ||
         llvm::is_contained(opts.codegenSearch.enableTensorCoreTf32, true));
	    const bool profileWantsTensorCore =
	        profileEnabledForSubgraph && opts.profile.enableTensorCoreF16;
    const bool allowTensorCoreClassPreference =
        subgraphHasMatmul && (searchWantsTensorCore || profileWantsTensorCore);
	    const bool allowTensorCoreProfile =
	        profileEnabledForSubgraph && allowTensorCoreClassPreference;
    const bool wantMatmulSoftmaxReuse =
        opts.profile.enableMatmulSoftmaxSharedReuseFusion ||
        (opts.codegenSearch.enable &&
         llvm::is_contained(
             opts.codegenSearch.enableMatmulSoftmaxSharedReuseFusion, true));
    const bool retainTensorCoreClassForMatmul = allowTensorCoreClassPreference;
    const bool graphHasMatmulSoftmaxContext =
        graphHasMatmulSoftmaxLikePattern(g);
    const bool matmulSoftmaxLikeContext =
        isMatmulSoftmaxLikeSubgraph(g, sgAfterCuts) ||
        (wantMatmulSoftmaxReuse && graphHasMatmulSoftmaxContext);
    const bool preferTensorCoreForMatmulSoftmax =
        allowTensorCoreClassPreference && wantMatmulSoftmaxReuse &&
        matmulSoftmaxLikeContext;
    const double tcPreferRatio = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_PREFER_RATIO",
                                   /*default=*/1.15));

    auto betterEstimate = [&](const PaperScheduleCandidate &a,
                              const PaperScheduleCandidate &b) {
      double al = getPaperCandidateSortLatencyProfileFirst(a);
      double bl = getPaperCandidateSortLatencyProfileFirst(b);
      int latCmp = compareProfilePriorityLatency(a.cand, al, b.cand, bl);
      if (latCmp != 0)
        return latCmp < 0;
      bool aTc = a.cand.enableTensorCoreF16 || a.cand.enableTensorCoreTf32;
      bool bTc = b.cand.enableTensorCoreF16 || b.cand.enableTensorCoreTf32;
      if (preferTensorCoreForMatmulSoftmax && aTc != bTc &&
          std::isfinite(al) && std::isfinite(bl)) {
        if (aTc && al <= bl * tcPreferRatio)
          return true;
        if (bTc && bl <= al * tcPreferRatio)
          return false;
      }
      if (a.cand.blocksPerSM != b.cand.blocksPerSM)
        return a.cand.blocksPerSM > b.cand.blocksPerSM;
      if (a.cand.estRegsPerThread != b.cand.estRegsPerThread)
        return a.cand.estRegsPerThread < b.cand.estRegsPerThread;
      if (aTc != bTc)
        return aTc;
      if (a.sharedFootprintBytes != b.sharedFootprintBytes)
        return a.sharedFootprintBytes < b.sharedFootprintBytes;
      return a.cand.score < b.cand.score;
    };

	    llvm::sort(pending, betterEstimate);
		    if (!pending.empty()) {
		      llvm::SmallVector<int64_t, 64> keepIdxs =
		          selectRecursiveStageTopKCandidateIndices(
		              static_cast<int64_t>(pending.size()),
		              [&](int64_t i) -> const Candidate & {
		                return pending[static_cast<size_t>(i)].cand;
		              },
		              g, sgAfterCuts, opts, inference, scheduleLevels,
		              maxRowReductionExtentForTc);
	      if (!keepIdxs.empty() &&
	          keepIdxs.size() < static_cast<size_t>(pending.size())) {
	        std::vector<PaperScheduleCandidate> filtered;
	        filtered.reserve(keepIdxs.size());
	        for (int64_t i : keepIdxs) {
	          if (i < 0 || i >= static_cast<int64_t>(pending.size()))
	            continue;
	          filtered.push_back(std::move(pending[static_cast<size_t>(i)]));
	        }
	        pending = std::move(filtered);
	      }
	    }

	    // 启用 profiling 时，只对少量内层变体做 profiling
	    // （compile+profile 开销很高）。但一旦启用 register 层
    // tile 和/或 codegen 搜索，基于公式的 top-1~2 估计
    // 很容易“脆弱”（编译/测量失败）。因此需要
    // 稍大的预算，并尽量包含一个简单 baseline 旋钮变体
    // （若存在），避免论文对齐剪枝把整个 shared
    // tile 因早期偶发失败而整体丢弃。
    int64_t keepN = 1;
	    if (profileEnabledForSubgraph) {
      keepN = std::min<int64_t>(
          (opts.codegenSearch.enable || opts.enableRegisterLevelSchedule) ? 4
                                                                          : 2,
          static_cast<int64_t>(pending.size()));
    } else if (opts.codegenSearch.enable || opts.enableRegisterLevelSchedule) {
      keepN = std::min<int64_t>(4, static_cast<int64_t>(pending.size()));
    }
    const int64_t minKeepNMatmulSoftmaxProfile =
        std::max<int64_t>(1, getEnvInt64OrDefault(
                                 "WELDER_PROFILE_MM_SM_MIN_KEEPN",
                                 /*default=*/12));
	    if (retainTensorCoreClassForMatmul && !profileEnabledForSubgraph)
	      keepN = std::min<int64_t>(
	          static_cast<int64_t>(pending.size()), std::max<int64_t>(keepN, 8));
	    if (profileEnabledForSubgraph && matmulSoftmaxLikeContext)
	      keepN = std::min<int64_t>(
	          static_cast<int64_t>(pending.size()),
	          std::max<int64_t>(keepN, minKeepNMatmulSoftmaxProfile));
	    if (!profileEnabledForSubgraph && opts.codegenSearch.enable && keepN > 1) {
      const bool wantAsyncClass =
          llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
      const bool wantPipeClass =
          llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
      const bool wantWaitGroupClass = llvm::is_contained(
          opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
      const bool wantRowReuseClass = llvm::is_contained(
          opts.codegenSearch.enableRowReductionChainReuseFusion, true);
      const bool wantRowPromoClass = llvm::is_contained(
          opts.codegenSearch.enableRowReductionInputPromotion, true);
      const bool wantRowWarpClass = llvm::is_contained(
          opts.codegenSearch.enableRowReductionWarp, true);
      int64_t replacePos = keepN - 1;
      auto ensureClassInKeepN = [&](auto pred) {
        bool hasInTop = false;
        for (int64_t i = 0; i < keepN; ++i) {
          if (pred(pending[static_cast<size_t>(i)].cand)) {
            hasInTop = true;
            break;
          }
        }
        if (hasInTop)
          return;
        int64_t tailIdx = -1;
        for (int64_t i = keepN; i < static_cast<int64_t>(pending.size()); ++i) {
          if (pred(pending[static_cast<size_t>(i)].cand)) {
            tailIdx = i;
            break;
          }
        }
        if (tailIdx < 0)
          return;
        while (replacePos >= 0 &&
               pred(pending[static_cast<size_t>(replacePos)].cand)) {
          --replacePos;
        }
        if (replacePos < 0)
          return;
        std::swap(pending[static_cast<size_t>(replacePos)],
                  pending[static_cast<size_t>(tailIdx)]);
        --replacePos;
      };
      if (retainTensorCoreClassForMatmul) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableTensorCoreF16 || c.enableTensorCoreTf32;
        });
      }
      if (wantRowPromoClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableRowReductionInputPromotion;
        });
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableRowReductionInputPromotion &&
                 !c.enableRowReductionWarp;
        });
      }
      if (wantRowWarpClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableRowReductionWarp;
        });
      }
      if (wantRowReuseClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableRowReductionChainReuseFusion;
        });
      }
      if (wantRowReuseClass && wantAsyncClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableAsyncCopy && c.enableRowReductionChainReuseFusion;
        });
      }
      if (wantRowReuseClass && retainTensorCoreClassForMatmul) {
        ensureClassInKeepN([](const Candidate &c) {
          return (c.enableTensorCoreF16 || c.enableTensorCoreTf32) &&
                 c.enableRowReductionChainReuseFusion;
        });
      }
      if (wantWaitGroupClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
        });
      }
      if (wantPipeClass) {
        ensureClassInKeepN([](const Candidate &c) {
          return c.enableSoftwarePipelining;
        });
      }
      if (wantAsyncClass) {
        ensureClassInKeepN([](const Candidate &c) { return c.enableAsyncCopy; });
      }
    }

    auto isBaselineKnobs = [](const Candidate &c) -> bool {
      if (c.enableTensorCoreF16 || c.enableTensorCoreTf32)
        return false;
      if (c.enableAsyncCopy || c.enableSoftwarePipelining)
        return false;
      if (c.enableRowReductionInputPromotion)
        return false;
      if (c.workgroupMultiBufferDepth > 1)
        return false;
      if (c.workgroupPadLastDim != 0)
        return false;
      if (c.workgroupSwizzleXor != 0)
        return false;
      if (c.blockRasterizeXor != 0)
        return false;
      if (c.blockRasterizeMode != 0 || c.blockRasterizePanelWidth != 0)
        return false;
      return true;
    };

    llvm::SmallVector<int64_t, 8> profileIdxs;
	    if (profileEnabledForSubgraph && !pending.empty()) {
      profileIdxs.push_back(0); // 按估计最优
      // 若存在，也加入一个 baseline 旋钮变体。
      for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
        if (i == 0)
          continue;
        if (isBaselineKnobs(pending[static_cast<size_t>(i)].cand)) {
          profileIdxs.push_back(i);
          break;
        }
      }
      // 论文对齐的调优循环：当 codegen 搜索扩展旋钮族时，
      // 公式估计常看不到这些优化（例如
      // async copy + pipelining）。为避免漏掉仅在
      // profiling 中出现的“收益”，每个旋钮族至少采样
      // 一个候选（若该族在变体集合中存在）。
      auto pushProfileIdx = [&](int64_t idx) {
        if (idx < 0 || idx >= static_cast<int64_t>(pending.size()))
          return;
        if (!llvm::is_contained(profileIdxs, idx))
          profileIdxs.push_back(idx);
      };
      auto maybeAddFirst = [&](auto pred) {
        for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
          if (llvm::is_contained(profileIdxs, i))
            continue;
          const Candidate &c = pending[static_cast<size_t>(i)].cand;
          if (!pred(c))
            continue;
          pushProfileIdx(i);
          return;
        }
      };
      auto maybeAddBest = [&](auto pred, auto keyFn, bool preferSmaller) {
        int64_t bestIdx = -1;
        double bestKey = preferSmaller ? std::numeric_limits<double>::infinity()
                                       : -std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
          if (llvm::is_contained(profileIdxs, i))
            continue;
          const Candidate &c = pending[static_cast<size_t>(i)].cand;
          if (!pred(c))
            continue;
          double key = keyFn(c);
          if ((preferSmaller && key < bestKey) ||
              (!preferSmaller && key > bestKey)) {
            bestKey = key;
            bestIdx = i;
          }
        }
        pushProfileIdx(bestIdx);
      };
      // Async/pipelining：优先最“激进”的变体，以覆盖
      // 可用时的 pipelining 与 wait_group 行为。
      for (int rep = 0; rep < 2; ++rep) {
        maybeAddFirst([](const Candidate &c) {
          return c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
        });
      }
      for (int rep = 0; rep < 2; ++rep)
        maybeAddFirst(
            [](const Candidate &c) { return c.enableSoftwarePipelining; });
      for (int rep = 0; rep < 2; ++rep) {
        maybeAddFirst([](const Candidate &c) {
          return c.enableAsyncCopy && !c.enableSoftwarePipelining;
        });
      }
      // Layout / stride-map 旋钮（padding/swizzle）与 rasterization/remap。
      maybeAddFirst([](const Candidate &c) {
        return c.workgroupPadLastDim != 0 || c.workgroupSwizzleXor != 0;
      });
      maybeAddFirst([](const Candidate &c) {
        return c.blockRasterizeXor != 0 || c.blockRasterizeMode != 0 ||
               c.blockRasterizePanelWidth != 0;
      });
      maybeAddFirst([](const Candidate &c) { return c.swapBlockDims; });
      // 行归约微内核变体（Softmax/LayerNorm 风格）。
      maybeAddFirst([](const Candidate &c) {
        return c.enableRowReductionInputPromotion;
      });
      maybeAddFirst([](const Candidate &c) {
        return c.enableRowReductionChainReuseFusion;
      });
      maybeAddFirst([](const Candidate &c) {
        return c.enableAsyncCopy && c.enableRowReductionChainReuseFusion;
      });
      maybeAddFirst([](const Candidate &c) {
        return c.enableTensorCoreF16 &&
               c.enableRowReductionChainReuseFusion;
      });
      maybeAddFirst([](const Candidate &c) {
        return c.enableRowReductionInputPromotion &&
               !c.enableRowReductionWarp;
      });
      maybeAddFirst(
          [](const Candidate &c) { return c.enableRowReductionWarp; });
      maybeAddFirst(
          [](const Candidate &c) { return c.enableRowReductionVectorize; });
      maybeAddFirst(
          [](const Candidate &c) { return c.enableRowReductionCombineVectorize; });
      maybeAddFirst([](const Candidate &c) {
        return c.enableRowReductionSkipCombineBarrier ||
               c.enableRowReductionRelaxBarriers;
      });
      maybeAddBest(
          [](const Candidate &c) { return c.enableRowReductionInputPromotion; },
          [](const Candidate &c) {
            return static_cast<double>(
                std::max<int64_t>(1, c.estRegsPerThread));
          },
          /* preferSmaller=*/true);
      maybeAddBest(
          [](const Candidate &c) { return c.enableRowReductionInputPromotion; },
          [](const Candidate &c) {
            return static_cast<double>(std::max<int64_t>(1, c.blocksPerSM));
          },
          /* preferSmaller=*/false);
      maybeAddBest(
          [](const Candidate &c) {
            return c.enableRowReductionWarp || c.enableRowReductionVectorize;
          },
          [](const Candidate &c) {
            return static_cast<double>(std::max<int64_t>(1, c.blocksPerSM));
          },
          /* preferSmaller=*/false);
      auto isTensorCoreCand = [](const Candidate &c) {
        return c.enableTensorCoreF16 || c.enableTensorCoreTf32;
      };
      const bool enforceCodegenClassCoverage =
          opts.codegenSearch.enable &&
          (getEnvInt64OrDefault(
               "WELDER_PROFILE_ENFORCE_CODEGEN_CLASS_COVERAGE", 1) != 0);
      const bool wantAsyncProfileClass =
          enforceCodegenClassCoverage &&
          llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
      const bool wantPipeProfileClass =
          enforceCodegenClassCoverage &&
          llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
      const bool wantWaitGroupProfileClass =
          enforceCodegenClassCoverage &&
          llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups,
                             true);
      bool hasTcCandAny = false;
      bool hasNonTcCandAny = false;
      bool hasAsyncCandAny = false;
      bool hasPipeCandAny = false;
      bool hasWaitGroupCandAny = false;
      auto hasPendingCand = [&](auto pred) {
        for (const PaperScheduleCandidate &pc : pending) {
          if (pred(pc.cand))
            return true;
        }
        return false;
      };
      auto hasProfileCand = [&](auto pred) {
        for (int64_t i : profileIdxs) {
          if (i < 0 || i >= static_cast<int64_t>(pending.size()))
            continue;
          if (pred(pending[static_cast<size_t>(i)].cand))
            return true;
        }
        return false;
      };
      // 若正在探索 TensorCore，需确保至少测到一个
      // TensorCore 变体（当 sink 是 matmul 时，编译器的
      // tensorcore 流水线才适用）。
      if (allowTensorCoreProfile) {
        hasTcCandAny = hasPendingCand(isTensorCoreCand);
        hasNonTcCandAny = hasPendingCand(
            [&](const Candidate &c) { return !isTensorCoreCand(c); });
        maybeAddBest(
            [&](const Candidate &c) { return isTensorCoreCand(c); },
            [](const Candidate &c) {
              return static_cast<double>(
                  std::max<int64_t>(1, c.estRegsPerThread));
            },
            /* preferSmaller=*/true);
        maybeAddBest(
            [&](const Candidate &c) { return isTensorCoreCand(c); },
            [](const Candidate &c) {
              return static_cast<double>(std::max<int64_t>(1, c.blocksPerSM));
            },
            /* preferSmaller=*/false);
        maybeAddFirst([&](const Candidate &c) {
          return isTensorCoreCand(c) && c.enableAsyncCopy;
        });
        maybeAddFirst([&](const Candidate &c) {
          return isTensorCoreCand(c) && c.enableSoftwarePipelining;
        });
        maybeAddBest(
            [&](const Candidate &c) {
              return isTensorCoreCand(c) && c.enableSoftwarePipelining &&
                     c.pipelineSetAsyncWaitGroups;
            },
            [](const Candidate &c) {
              return static_cast<double>(std::max<int64_t>(1, c.blocksPerSM));
            },
            /* preferSmaller=*/false);
        maybeAddBest(
            [&](const Candidate &c) {
              return isTensorCoreCand(c) && c.enableSoftwarePipelining &&
                     c.pipelineSetAsyncWaitGroups;
            },
            [](const Candidate &c) {
              return static_cast<double>(
                  std::max<int64_t>(1, c.estRegsPerThread));
            },
            /* preferSmaller=*/true);
        if (hasNonTcCandAny) {
          // 至少保留一个强非 TC 锚点，确保 profile 选择在
          // TC 候选虽可编译但性能不稳定时仍稳健。
          maybeAddBest(
              [&](const Candidate &c) { return !isTensorCoreCand(c); },
              [](const Candidate &c) {
                return static_cast<double>(std::max<int64_t>(1, c.blocksPerSM));
              },
              /* preferSmaller=*/false);
          maybeAddBest(
              [&](const Candidate &c) { return !isTensorCoreCand(c); },
              [](const Candidate &c) {
                return static_cast<double>(
                    std::max<int64_t>(1, c.estRegsPerThread));
              },
              /* preferSmaller=*/true);
        }
        // 最终保护：若两类都存在，强制每类至少采样一个
        // profile 样本。
        if (hasTcCandAny && !hasProfileCand(isTensorCoreCand))
          maybeAddFirst([&](const Candidate &c) { return isTensorCoreCand(c); });
        if (hasNonTcCandAny &&
            !hasProfileCand([&](const Candidate &c) {
              return !isTensorCoreCand(c);
            })) {
          maybeAddFirst([&](const Candidate &c) {
            return !isTensorCoreCand(c);
          });
        }
      }
      hasAsyncCandAny = wantAsyncProfileClass &&
                        hasPendingCand([](const Candidate &c) {
                          return c.enableAsyncCopy;
                        });
      hasPipeCandAny = wantPipeProfileClass &&
                       hasPendingCand([](const Candidate &c) {
                         return c.enableSoftwarePipelining;
                       });
      hasWaitGroupCandAny = wantWaitGroupProfileClass &&
                            hasPendingCand([](const Candidate &c) {
                              return c.enableSoftwarePipelining &&
                                     c.pipelineSetAsyncWaitGroups;
                            });
      int64_t diversityCap = std::min<int64_t>(
          static_cast<int64_t>(pending.size()),
          (opts.codegenSearch.enable || opts.enableRegisterLevelSchedule) ? 12
                                                                          : 6);
      if (preferTensorCoreForMatmulSoftmax)
        diversityCap = std::max<int64_t>(diversityCap, 20);
      diversityCap = std::max<int64_t>(keepN, diversityCap);
      if (static_cast<int64_t>(profileIdxs.size()) > diversityCap)
        profileIdxs.resize(static_cast<size_t>(diversityCap));
      auto forceIncludeClass = [&](auto pred) {
        if (hasProfileCand(pred))
          return;
        int64_t idx = -1;
        for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
          if (pred(pending[static_cast<size_t>(i)].cand)) {
            idx = i;
            break;
          }
        }
        if (idx < 0)
          return;
        if (static_cast<int64_t>(profileIdxs.size()) < diversityCap) {
          profileIdxs.push_back(idx);
          return;
        }
        for (int64_t k = static_cast<int64_t>(profileIdxs.size()) - 1; k >= 0;
             --k) {
          int64_t cur = profileIdxs[static_cast<size_t>(k)];
          if (cur < 0 || cur >= static_cast<int64_t>(pending.size()))
            continue;
          if (!pred(pending[static_cast<size_t>(cur)].cand)) {
            profileIdxs[static_cast<size_t>(k)] = idx;
            return;
          }
        }
      };
      if (allowTensorCoreProfile && hasTcCandAny && hasNonTcCandAny) {
        forceIncludeClass(isTensorCoreCand);
        forceIncludeClass([&](const Candidate &c) { return !isTensorCoreCand(c); });
      }
      if (hasWaitGroupCandAny) {
        forceIncludeClass([](const Candidate &c) {
          return c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
        });
      }
      if (hasPipeCandAny) {
        forceIncludeClass(
            [](const Candidate &c) { return c.enableSoftwarePipelining; });
      }
      if (hasAsyncCandAny) {
        forceIncludeClass(
            [](const Candidate &c) { return c.enableAsyncCopy; });
      }
      // 确保 keepN 覆盖强制多样性样本。
      keepN =
          std::max<int64_t>(keepN, static_cast<int64_t>(profileIdxs.size()));
      // 按估计顺序填满剩余预算。
      for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
        if (static_cast<int64_t>(profileIdxs.size()) >= keepN)
          break;
        if (llvm::is_contained(profileIdxs, i))
          continue;
        profileIdxs.push_back(i);
      }
      const bool matmulSoftmaxLikeContextF16 =
          matmulSoftmaxLikeContext && opts.arch.elementBytes <= 2;
      const int64_t defaultMaxProfileAttemptsPerTile =
          matmulSoftmaxLikeContextF16 && preferTensorCoreForMatmulSoftmax
              ? 12
              : (preferTensorCoreForMatmulSoftmax
                     ? 20
                     : ((opts.codegenSearch.enable ||
                         opts.enableRegisterLevelSchedule)
                            ? 16
                            : 8));
      const int64_t maxProfileAttemptsPerSharedTile =
          std::max<int64_t>(0, getEnvInt64OrDefault(
                                   "WELDER_PROFILE_MAX_ATTEMPTS_PER_SHARED_TILE",
                                   defaultMaxProfileAttemptsPerTile));
      if (maxProfileAttemptsPerSharedTile > 0 &&
          static_cast<int64_t>(profileIdxs.size()) >
              maxProfileAttemptsPerSharedTile) {
        const int64_t totalBeforeCap = static_cast<int64_t>(profileIdxs.size());
        const int64_t cap = std::max<int64_t>(
            1, std::min<int64_t>(maxProfileAttemptsPerSharedTile, totalBeforeCap));
        if (cap >= 1) {
          auto hasClassInPrefix = [&](auto pred) -> bool {
            for (int64_t p = 0; p < cap; ++p) {
              int64_t idx = profileIdxs[static_cast<size_t>(p)];
              if (idx < 0 || idx >= static_cast<int64_t>(pending.size()))
                continue;
              if (pred(pending[static_cast<size_t>(idx)].cand))
                return true;
            }
            return false;
          };
          auto findClassInTail = [&](auto pred) -> int64_t {
            for (int64_t p = cap; p < static_cast<int64_t>(profileIdxs.size());
                 ++p) {
              int64_t idx = profileIdxs[static_cast<size_t>(p)];
              if (idx < 0 || idx >= static_cast<int64_t>(pending.size()))
                continue;
              if (pred(pending[static_cast<size_t>(idx)].cand))
                return p;
            }
            return -1;
          };
          auto ensurePrefixClass = [&](auto pred) {
            if (hasClassInPrefix(pred))
              return;
            int64_t tailPos = findClassInTail(pred);
            if (tailPos < 0)
              return;
            int64_t replacePos = cap - 1;
            for (int64_t p = cap - 1; p >= 0; --p) {
              int64_t idx = profileIdxs[static_cast<size_t>(p)];
              if (idx < 0 || idx >= static_cast<int64_t>(pending.size()))
                continue;
              if (!pred(pending[static_cast<size_t>(idx)].cand)) {
                replacePos = p;
                break;
              }
            }
            std::swap(profileIdxs[static_cast<size_t>(replacePos)],
                      profileIdxs[static_cast<size_t>(tailPos)]);
          };
          if (allowTensorCoreProfile && hasTcCandAny && hasNonTcCandAny &&
              cap >= 2) {
            ensurePrefixClass(
                [&](const Candidate &c) { return isTensorCoreCand(c); });
            ensurePrefixClass(
                [&](const Candidate &c) { return !isTensorCoreCand(c); });
          }
          if (hasWaitGroupCandAny) {
            ensurePrefixClass([](const Candidate &c) {
              return c.enableSoftwarePipelining &&
                     c.pipelineSetAsyncWaitGroups;
            });
          }
          if (hasPipeCandAny) {
            ensurePrefixClass(
                [](const Candidate &c) { return c.enableSoftwarePipelining; });
          }
          if (hasAsyncCandAny) {
            ensurePrefixClass(
                [](const Candidate &c) { return c.enableAsyncCopy; });
          }
        }
        profileIdxs.resize(static_cast<size_t>(cap));
        keepN = std::min<int64_t>(keepN, cap);
        if (opts.tracer) {
          llvm::json::Object f;
          f["total_before_cap"] = totalBeforeCap;
          f["capped_to"] = cap;
          f["cap"] = maxProfileAttemptsPerSharedTile;
          opts.tracer->event("profile.attempt_cap", std::move(f),
                             /* isVerbose=*/true);
        }
      }
      keepN = std::min<int64_t>(keepN, static_cast<int64_t>(profileIdxs.size()));
    }

    const bool matmulSoftmaxLikeContextF16 =
        matmulSoftmaxLikeContext && opts.arch.elementBytes <= 2;
    const int64_t defaultMaxProfileAttemptsPerTile =
        matmulSoftmaxLikeContextF16 && preferTensorCoreForMatmulSoftmax
            ? 12
            : (preferTensorCoreForMatmulSoftmax
                   ? 20
                   : ((opts.codegenSearch.enable ||
                       opts.enableRegisterLevelSchedule)
                          ? 16
                          : 8));
	    const int64_t maxProfileAttemptsPerSharedTile =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(0, getEnvInt64OrDefault(
	                                       "WELDER_PROFILE_MAX_ATTEMPTS_PER_SHARED_TILE",
	                                       defaultMaxProfileAttemptsPerTile))
	            : 0;
	    const int64_t maxProfileCompilesPerSharedTile =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(
	                  0, getEnvInt64OrDefault(
	                         "WELDER_PROFILE_MAX_COMPILES_PER_SHARED_TILE",
	                         maxProfileAttemptsPerSharedTile))
	            : 0;
	    const int64_t maxConsecutiveProfileFailures =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(
	                  0, getEnvInt64OrDefault("WELDER_PROFILE_MAX_CONSECUTIVE_FAILURES",
	                                          4))
	            : 0;
	    const int64_t noImprovementProfileWindow =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(
	                  0, getEnvInt64OrDefault("WELDER_PROFILE_NO_IMPROVEMENT_WINDOW",
	                                          8))
	            : 0;
	    const int64_t minProfileAttemptsBeforeEarlyStop =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(
	                  1, getEnvInt64OrDefault("WELDER_PROFILE_EARLY_STOP_MIN_ATTEMPTS",
	                                          6))
	            : 0;
	    const double profileImprovementRatio =
	        profileEnabledForSubgraph
	            ? std::max(0.0, std::min(
	                                  0.95,
	                                  getEnvDoubleOrDefault(
	                                      "WELDER_PROFILE_EARLY_STOP_IMPROVEMENT_RATIO",
	                                      0.005)))
	            : 0.0;
	    const int64_t recursiveKeepTopMeasured =
	        profileEnabledForSubgraph
	            ? std::max<int64_t>(
	                  0, getEnvInt64OrDefault(
	                         "WELDER_PROFILE_RECURSIVE_KEEP_TOP_MEASURED",
	                         preferTensorCoreForMatmulSoftmax ? 8 : 4))
	            : 0;

    if (ensureGlobalClassAnchors && !pending.empty()) {
      auto updateGlobalAnchorFromPending = [&](auto pred, bool &hasAnchor,
                                               PaperScheduleCandidate &anchor) {
        for (const PaperScheduleCandidate &pc : pending) {
          if (!pred(pc.cand))
            continue;
          if (!hasAnchor || betterPaperCandidateByProfilePriority(pc, anchor)) {
            anchor = pc;
            hasAnchor = true;
          }
        }
      };
      if (wantGlobalAsyncAnchor) {
        updateGlobalAnchorFromPending(
            [](const Candidate &c) { return c.enableAsyncCopy; },
            hasGlobalAsyncAnchor, globalAsyncAnchor);
      }
      if (wantGlobalPipeAnchor) {
        updateGlobalAnchorFromPending(
            [](const Candidate &c) { return c.enableSoftwarePipelining; },
            hasGlobalPipeAnchor, globalPipeAnchor);
      }
      if (wantGlobalWaitGroupAnchor) {
        updateGlobalAnchorFromPending(
            [](const Candidate &c) {
              return c.enableSoftwarePipelining &&
                     c.pipelineSetAsyncWaitGroups;
            },
            hasGlobalWaitGroupAnchor, globalWaitGroupAnchor);
      }
      if (wantGlobalNonTensorCoreAnchor) {
        updateGlobalAnchorFromPending(
            [](const Candidate &c) {
              return !(c.enableTensorCoreF16 || c.enableTensorCoreTf32);
            },
            hasGlobalNonTensorCoreAnchor, globalNonTensorCoreAnchor);
      }
      if (wantGlobalTensorCoreF16Anchor) {
        updateGlobalAnchorFromPending(
            [](const Candidate &c) { return c.enableTensorCoreF16; },
            hasGlobalTensorCoreF16Anchor, globalTensorCoreF16Anchor);
      }
    }

    if (opts.paperRecursiveRegisterLevel) {
      bool hasBest = false;
      PaperScheduleCandidate best;
      llvm::SmallVector<PaperScheduleCandidate, 16> extraMeasuredForOut;
      bool hasBestTensorCoreAnchor = false;
      PaperScheduleCandidate bestTensorCoreAnchor;
      const bool wantNonTensorCoreAnchor =
          retainTensorCoreClassForMatmul &&
          (getEnvInt64OrDefault("WELDER_RETAIN_NON_TC_ANCHOR", 1) != 0);
      bool hasBestNonTensorCoreAnchor = false;
      PaperScheduleCandidate bestNonTensorCoreAnchor;
      const bool wantAsyncAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
      const bool wantPipeAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
      const bool wantWaitGroupAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
      const bool wantRowReuseAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.enableRowReductionChainReuseFusion,
                             true);
      const bool wantRowPromoAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.enableRowReductionInputPromotion,
                             true);
      const bool wantRowWarpAnchor =
          opts.codegenSearch.enable &&
          llvm::is_contained(opts.codegenSearch.enableRowReductionWarp, true);
      const bool wantRowThreadsXAnchor =
          opts.codegenSearch.enable &&
          (getEnvInt64OrDefault("WELDER_PROFILE_ENFORCE_ROW_THREADS_X_CLASS",
                                1) != 0);
      bool hasBestAsyncAnchor = false;
      PaperScheduleCandidate bestAsyncAnchor;
      bool hasBestPipeAnchor = false;
      PaperScheduleCandidate bestPipeAnchor;
      bool hasBestWaitGroupAnchor = false;
      PaperScheduleCandidate bestWaitGroupAnchor;
      bool hasBestRowReuseAnchor = false;
      PaperScheduleCandidate bestRowReuseAnchor;
      bool hasBestRowPromoAnchor = false;
      PaperScheduleCandidate bestRowPromoAnchor;
      bool hasBestRowWarpAnchor = false;
      PaperScheduleCandidate bestRowWarpAnchor;
      bool hasBestRowThreadsXLowAnchor = false;
      PaperScheduleCandidate bestRowThreadsXLowAnchor;
      bool hasBestRowThreadsXHighAnchor = false;
      PaperScheduleCandidate bestRowThreadsXHighAnchor;
      int64_t minRowThreadsXInPending = std::numeric_limits<int64_t>::max();
      int64_t maxRowThreadsXInPending = 0;
      if (wantRowThreadsXAnchor) {
        for (const PaperScheduleCandidate &pc : pending) {
          int64_t tx = pc.cand.rowReductionThreadsX;
          if (tx <= 0)
            continue;
          minRowThreadsXInPending = std::min(minRowThreadsXInPending, tx);
          maxRowThreadsXInPending = std::max(maxRowThreadsXInPending, tx);
        }
      }
      const bool hasRowThreadsXSplit =
          wantRowThreadsXAnchor &&
          minRowThreadsXInPending < std::numeric_limits<int64_t>::max() &&
          maxRowThreadsXInPending > minRowThreadsXInPending;
      auto isTensorCoreCand = [](const Candidate &c) {
        return c.enableTensorCoreF16 || c.enableTensorCoreTf32;
      };
	      if (!profileEnabledForSubgraph) {
        for (int64_t i = 0; i < keepN; ++i) {
          PaperScheduleCandidate pc = pending[static_cast<size_t>(i)];
          if (!hasBest || pc.estimatedLatency < best.estimatedLatency) {
            best = std::move(pc);
            hasBest = true;
          }
        }
        if (retainTensorCoreClassForMatmul) {
          for (const PaperScheduleCandidate &pc : pending) {
            if (isTensorCoreCand(pc.cand)) {
              if (!hasBestTensorCoreAnchor ||
                  pc.estimatedLatency <
                      bestTensorCoreAnchor.estimatedLatency) {
                bestTensorCoreAnchor = pc;
                hasBestTensorCoreAnchor = true;
              }
              continue;
            }
            if (wantNonTensorCoreAnchor &&
                (!hasBestNonTensorCoreAnchor ||
                 pc.estimatedLatency <
                     bestNonTensorCoreAnchor.estimatedLatency)) {
              bestNonTensorCoreAnchor = pc;
              hasBestNonTensorCoreAnchor = true;
            }
          }
        }
        if (wantAsyncAnchor || wantPipeAnchor || wantWaitGroupAnchor ||
            wantRowReuseAnchor || wantRowPromoAnchor || wantRowWarpAnchor ||
            hasRowThreadsXSplit) {
          for (const PaperScheduleCandidate &pc : pending) {
            if (wantAsyncAnchor && pc.cand.enableAsyncCopy) {
              if (!hasBestAsyncAnchor ||
                  pc.estimatedLatency < bestAsyncAnchor.estimatedLatency) {
                bestAsyncAnchor = pc;
                hasBestAsyncAnchor = true;
              }
            }
            if (wantRowReuseAnchor &&
                pc.cand.enableRowReductionChainReuseFusion) {
              if (!hasBestRowReuseAnchor ||
                  pc.estimatedLatency < bestRowReuseAnchor.estimatedLatency) {
                bestRowReuseAnchor = pc;
                hasBestRowReuseAnchor = true;
              }
            }
            if (wantPipeAnchor && pc.cand.enableSoftwarePipelining) {
              if (!hasBestPipeAnchor ||
                  pc.estimatedLatency < bestPipeAnchor.estimatedLatency) {
                bestPipeAnchor = pc;
                hasBestPipeAnchor = true;
              }
            }
            if (wantWaitGroupAnchor && pc.cand.enableSoftwarePipelining &&
                pc.cand.pipelineSetAsyncWaitGroups) {
              if (!hasBestWaitGroupAnchor ||
                  pc.estimatedLatency <
                      bestWaitGroupAnchor.estimatedLatency) {
                bestWaitGroupAnchor = pc;
                hasBestWaitGroupAnchor = true;
              }
            }
            if (wantRowPromoAnchor &&
                pc.cand.enableRowReductionInputPromotion) {
              if (!hasBestRowPromoAnchor ||
                  pc.estimatedLatency < bestRowPromoAnchor.estimatedLatency) {
                bestRowPromoAnchor = pc;
                hasBestRowPromoAnchor = true;
              }
            }
            if (wantRowWarpAnchor && pc.cand.enableRowReductionWarp) {
              if (!hasBestRowWarpAnchor ||
                  pc.estimatedLatency < bestRowWarpAnchor.estimatedLatency) {
                bestRowWarpAnchor = pc;
                hasBestRowWarpAnchor = true;
              }
            }
            if (hasRowThreadsXSplit &&
                pc.cand.rowReductionThreadsX == minRowThreadsXInPending) {
              if (!hasBestRowThreadsXLowAnchor ||
                  pc.estimatedLatency <
                      bestRowThreadsXLowAnchor.estimatedLatency) {
                bestRowThreadsXLowAnchor = pc;
                hasBestRowThreadsXLowAnchor = true;
              }
            }
            if (hasRowThreadsXSplit &&
                pc.cand.rowReductionThreadsX == maxRowThreadsXInPending) {
              if (!hasBestRowThreadsXHighAnchor ||
                  pc.estimatedLatency <
                      bestRowThreadsXHighAnchor.estimatedLatency) {
                bestRowThreadsXHighAnchor = pc;
                hasBestRowThreadsXHighAnchor = true;
              }
            }
          }
        }
      } else {
        llvm::SmallVector<PaperScheduleCandidate, 16> measuredCandidates;
        bool hasBestFallback = false;
        PaperScheduleCandidate bestFallback;
        int64_t profileAttempts = 0;
        int64_t profileCompiles = 0;
        int64_t consecutiveProfileFailures = 0;
        int64_t noImproveProfileStreak = 0;
        double bestMeasuredProfileMs = std::numeric_limits<double>::infinity();
        auto isTcAsyncWaitCandidateEarlyStop = [](const Candidate &c) {
          return c.enableTensorCoreF16 && c.enableAsyncCopy &&
                 c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
        };
        const bool enableTargetClassEarlyStop =
            matmulSoftmaxLikeContext && allowTensorCoreClassPreference &&
            (getEnvInt64OrDefault("WELDER_PROFILE_MM_SM_TARGET_CLASS_EARLY_STOP",
                                  1) != 0);
        const int64_t targetClassMinAttempts = std::max<int64_t>(
            int64_t(1), getEnvInt64OrDefault(
                            "WELDER_PROFILE_MM_SM_TARGET_CLASS_MIN_ATTEMPTS", 6));
        const double targetClassMaxRatio = std::max(
            1.0, getEnvDoubleOrDefault(
                     "WELDER_PROFILE_MM_SM_TARGET_CLASS_MAX_RATIO", 1.30));
        const bool hasNonTcClassInPending = llvm::any_of(
            pending, [&](const PaperScheduleCandidate &pc) {
              return !(pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32);
            });
        bool hasMeasuredTcAsyncWait = false;
        bool hasMeasuredNonTc = false;
        double bestMeasuredTcAsyncWaitMs = std::numeric_limits<double>::infinity();
        for (int64_t i : profileIdxs) {
          if (maxProfileAttemptsPerSharedTile > 0 &&
              profileAttempts >= maxProfileAttemptsPerSharedTile) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = "attempt_cap";
              f["phase"] = "recursive";
              f["attempts"] = profileAttempts;
              f["cap"] = maxProfileAttemptsPerSharedTile;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
          if (maxProfileCompilesPerSharedTile > 0 &&
              profileCompiles >= maxProfileCompilesPerSharedTile) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = "compile_cap";
              f["phase"] = "recursive";
              f["compiles"] = profileCompiles;
              f["cap"] = maxProfileCompilesPerSharedTile;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
          if (maxProfileCompilesTotal > 0 &&
              profileCompilesTotal >= maxProfileCompilesTotal) {
            stopOnProfileCompileTotalCap = true;
            if (opts.tracer && !emittedProfileCompileTotalCapEvent) {
              llvm::json::Object f;
              f["reason"] = "compile_total_cap";
              f["phase"] = "recursive";
              f["compiles"] = profileCompilesTotal;
              f["cap"] = maxProfileCompilesTotal;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
              emittedProfileCompileTotalCapEvent = true;
            }
            break;
          }
          if (i < 0 || i >= static_cast<int64_t>(pending.size()))
            continue;
          PaperScheduleCandidate pc = pending[static_cast<size_t>(i)];
          bool wasMeasured = false;
          bool dropOnFailure = false;
          bool wasCached = false;
          bool skipByGlobalCompileCap = false;
          auto msOpt = profileSubgraphByCompilingToNvvm(
              g, sgAfterCuts, sinkNodeIdx, pc.cand, opts,
              /* outWasCached=*/&wasCached);
          if (msOpt) {
            pc.cand.cost.profiledMs = *msOpt;
            pc.cand.score = *msOpt;
            pc.estimatedLatency = *msOpt;
            wasMeasured = true;
          } else if (wasCached) {
            // 出现“缓存命中失败”不可能；这里编码的是全局编译预算跳过。
            skipByGlobalCompileCap = true;
          } else if (opts.pruneOnProfileFailure || pc.cand.enableTensorCoreF16 ||
                     pc.cand.enableTensorCoreTf32) {
            dropOnFailure = true;
          }

          if (!wasCached) {
            ++profileCompiles;
            ++profileCompilesTotal;
          }
          if (skipByGlobalCompileCap) {
            if (!hasBestFallback ||
                pc.estimatedLatency < bestFallback.estimatedLatency) {
              bestFallback = pc;
              hasBestFallback = true;
            }
            break;
          }
          ++profileAttempts;
          if (wasMeasured && std::isfinite(pc.estimatedLatency)) {
            consecutiveProfileFailures = 0;
            bool improved =
                !std::isfinite(bestMeasuredProfileMs) ||
                pc.estimatedLatency <
                    bestMeasuredProfileMs * (1.0 - profileImprovementRatio);
            if (improved) {
              bestMeasuredProfileMs = pc.estimatedLatency;
              noImproveProfileStreak = 0;
            } else {
              ++noImproveProfileStreak;
            }
            const bool isTc =
                pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32;
            if (!isTc)
              hasMeasuredNonTc = true;
            if (isTcAsyncWaitCandidateEarlyStop(pc.cand)) {
              hasMeasuredTcAsyncWait = true;
              bestMeasuredTcAsyncWaitMs =
                  std::min(bestMeasuredTcAsyncWaitMs, pc.estimatedLatency);
            }
          } else {
            ++consecutiveProfileFailures;
            ++noImproveProfileStreak;
          }

          bool shouldEarlyStop = false;
          const char *earlyStopReason = "";
          if (profileAttempts >= minProfileAttemptsBeforeEarlyStop) {
            if (maxConsecutiveProfileFailures > 0 &&
                consecutiveProfileFailures >= maxConsecutiveProfileFailures) {
              shouldEarlyStop = true;
              earlyStopReason = "consecutive_failures";
            } else if (noImprovementProfileWindow > 0 &&
                       noImproveProfileStreak >= noImprovementProfileWindow) {
              shouldEarlyStop = true;
              earlyStopReason = "no_improvement";
            }
          }
          if (!shouldEarlyStop && enableTargetClassEarlyStop &&
              profileAttempts >= targetClassMinAttempts &&
              hasMeasuredTcAsyncWait &&
              (!hasNonTcClassInPending || hasMeasuredNonTc) &&
              std::isfinite(bestMeasuredProfileMs) &&
              bestMeasuredTcAsyncWaitMs <=
                  bestMeasuredProfileMs * targetClassMaxRatio) {
            shouldEarlyStop = true;
            earlyStopReason = "target_class_hit";
          }

          if (wasMeasured) {
            measuredCandidates.push_back(std::move(pc));
          } else if (!dropOnFailure &&
                     (!hasBestFallback ||
                      pc.estimatedLatency < bestFallback.estimatedLatency)) {
            bestFallback = std::move(pc);
            hasBestFallback = true;
          }
          if (shouldEarlyStop) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = earlyStopReason;
              f["phase"] = "recursive";
              f["attempts"] = profileAttempts;
              f["consecutive_failures"] = consecutiveProfileFailures;
              f["no_improvement_streak"] = noImproveProfileStreak;
              f["best_measured_ms"] =
                  std::isfinite(bestMeasuredProfileMs) ? bestMeasuredProfileMs
                                                       : -1.0;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
        }
        if (!measuredCandidates.empty()) {
          if (recursiveKeepTopMeasured > 0) {
            llvm::SmallVector<PaperScheduleCandidate, 16> measuredSorted(
                measuredCandidates.begin(), measuredCandidates.end());
            llvm::sort(measuredSorted, betterPaperCandidateByProfilePriority);
            int64_t keepTop = std::min<int64_t>(
                recursiveKeepTopMeasured,
                static_cast<int64_t>(measuredSorted.size()));
            for (int64_t i = 0; i < keepTop; ++i)
              extraMeasuredForOut.push_back(
                  measuredSorted[static_cast<size_t>(i)]);
          }
          auto isTcAsyncWaitCandidate = [](const Candidate &c) {
            return c.enableTensorCoreF16 && c.enableAsyncCopy &&
                   c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
          };
          const bool preferTcAsyncWaitForMatmulSoftmax =
              matmulSoftmaxLikeContext && allowTensorCoreClassPreference;
          double waitGroupRequireRatio = std::max(
              0.5, getEnvDoubleOrDefault("WELDER_PROFILE_WAIT_GROUP_REQUIRE_RATIO",
                                         /*default=*/1.0));
          if (preferTcAsyncWaitForMatmulSoftmax) {
            const double mmSmWaitGroupMinRatio = std::max(
                1.0,
                getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_WAIT_GROUP_MIN_RATIO",
                                      /*default=*/1.20));
            waitGroupRequireRatio =
                std::max(waitGroupRequireRatio, mmSmWaitGroupMinRatio);
          }
          const double tcAsyncWaitPreferRatio = std::max(
              1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_ASYNC_WAIT_PREFER_RATIO",
                                         /*default=*/1.20));
          double bestNonPipeMeasured =
              std::numeric_limits<double>::infinity();
          for (const PaperScheduleCandidate &pc : measuredCandidates) {
            bool isWaitGroupPipe =
                pc.cand.enableSoftwarePipelining &&
                pc.cand.pipelineSetAsyncWaitGroups;
            if (isWaitGroupPipe)
              continue;
            if (std::isfinite(pc.estimatedLatency))
              bestNonPipeMeasured =
                  std::min(bestNonPipeMeasured, pc.estimatedLatency);
          }

          bool hasBestMeasured = false;
          PaperScheduleCandidate bestMeasured;
          for (PaperScheduleCandidate &pc : measuredCandidates) {
            bool isWaitGroupPipe =
                pc.cand.enableSoftwarePipelining &&
                pc.cand.pipelineSetAsyncWaitGroups;
            if (isWaitGroupPipe && std::isfinite(bestNonPipeMeasured) &&
                pc.estimatedLatency >=
                    bestNonPipeMeasured * waitGroupRequireRatio) {
              if (opts.tracer) {
                llvm::json::Object f;
                f["wait_group_ms"] = pc.estimatedLatency;
                f["anchor_non_pipe_ms"] = bestNonPipeMeasured;
                f["ratio"] = waitGroupRequireRatio;
                opts.tracer->event("profile.wait_group_anchor_reject",
                                   std::move(f), /*isVerbose=*/true);
              }
              continue;
            }
            if (!hasBestMeasured ||
                pc.estimatedLatency < bestMeasured.estimatedLatency) {
              bestMeasured = pc;
              hasBestMeasured = true;
            }
          }
          if (!hasBestMeasured) {
            for (PaperScheduleCandidate &pc : measuredCandidates) {
              if (!hasBestMeasured ||
                  pc.estimatedLatency < bestMeasured.estimatedLatency) {
                bestMeasured = pc;
                hasBestMeasured = true;
              }
            }
          }
          if (hasBestMeasured) {
            if (preferTcAsyncWaitForMatmulSoftmax &&
                !isTcAsyncWaitCandidate(bestMeasured.cand) &&
                std::isfinite(bestMeasured.estimatedLatency)) {
              bool hasTcAsyncWaitMeasured = false;
              PaperScheduleCandidate bestTcAsyncWaitMeasured;
              for (const PaperScheduleCandidate &pc : measuredCandidates) {
                if (!isTcAsyncWaitCandidate(pc.cand) ||
                    !std::isfinite(pc.estimatedLatency))
                  continue;
                if (!hasTcAsyncWaitMeasured ||
                    pc.estimatedLatency <
                        bestTcAsyncWaitMeasured.estimatedLatency) {
                  bestTcAsyncWaitMeasured = pc;
                  hasTcAsyncWaitMeasured = true;
                }
              }
              if (hasTcAsyncWaitMeasured &&
                  bestTcAsyncWaitMeasured.estimatedLatency <=
                      bestMeasured.estimatedLatency * tcAsyncWaitPreferRatio) {
                if (opts.tracer) {
                  llvm::json::Object f;
                  f["base_ms"] = bestMeasured.estimatedLatency;
                  f["tc_async_wait_ms"] =
                      bestTcAsyncWaitMeasured.estimatedLatency;
                  f["ratio"] = tcAsyncWaitPreferRatio;
                  opts.tracer->event("profile.tc_async_wait_prefer",
                                     std::move(f), /*isVerbose=*/true);
                }
                bestMeasured = bestTcAsyncWaitMeasured;
              }
            }
            best = std::move(bestMeasured);
            hasBest = true;
          }
          if (retainTensorCoreClassForMatmul || wantAsyncAnchor ||
              wantPipeAnchor || wantWaitGroupAnchor || wantRowPromoAnchor ||
              wantRowWarpAnchor || wantRowReuseAnchor ||
              hasRowThreadsXSplit) {
            for (const PaperScheduleCandidate &pc : measuredCandidates) {
              if (retainTensorCoreClassForMatmul && isTensorCoreCand(pc.cand)) {
                if (!hasBestTensorCoreAnchor ||
                    pc.estimatedLatency <
                        bestTensorCoreAnchor.estimatedLatency) {
                  bestTensorCoreAnchor = pc;
                  hasBestTensorCoreAnchor = true;
                }
              }
              if (wantNonTensorCoreAnchor && !isTensorCoreCand(pc.cand)) {
                if (!hasBestNonTensorCoreAnchor ||
                    pc.estimatedLatency <
                        bestNonTensorCoreAnchor.estimatedLatency) {
                  bestNonTensorCoreAnchor = pc;
                  hasBestNonTensorCoreAnchor = true;
                }
              }
              if (wantAsyncAnchor && pc.cand.enableAsyncCopy) {
                if (!hasBestAsyncAnchor ||
                    pc.estimatedLatency < bestAsyncAnchor.estimatedLatency) {
                  bestAsyncAnchor = pc;
                  hasBestAsyncAnchor = true;
                }
              }
              if (wantRowReuseAnchor &&
                  pc.cand.enableRowReductionChainReuseFusion) {
                if (!hasBestRowReuseAnchor ||
                    pc.estimatedLatency < bestRowReuseAnchor.estimatedLatency) {
                  bestRowReuseAnchor = pc;
                  hasBestRowReuseAnchor = true;
                }
              }
              if (wantPipeAnchor && pc.cand.enableSoftwarePipelining) {
                if (!hasBestPipeAnchor ||
                    pc.estimatedLatency < bestPipeAnchor.estimatedLatency) {
                  bestPipeAnchor = pc;
                  hasBestPipeAnchor = true;
                }
              }
              if (wantWaitGroupAnchor && pc.cand.enableSoftwarePipelining &&
                  pc.cand.pipelineSetAsyncWaitGroups) {
                if (!hasBestWaitGroupAnchor ||
                    pc.estimatedLatency <
                        bestWaitGroupAnchor.estimatedLatency) {
                  bestWaitGroupAnchor = pc;
                  hasBestWaitGroupAnchor = true;
                }
              }
              if (wantRowPromoAnchor &&
                  pc.cand.enableRowReductionInputPromotion) {
                if (!hasBestRowPromoAnchor ||
                    pc.estimatedLatency <
                        bestRowPromoAnchor.estimatedLatency) {
                  bestRowPromoAnchor = pc;
                  hasBestRowPromoAnchor = true;
                }
              }
              if (wantRowWarpAnchor && pc.cand.enableRowReductionWarp) {
                if (!hasBestRowWarpAnchor ||
                    pc.estimatedLatency < bestRowWarpAnchor.estimatedLatency) {
                  bestRowWarpAnchor = pc;
                  hasBestRowWarpAnchor = true;
                }
              }
              if (hasRowThreadsXSplit &&
                  pc.cand.rowReductionThreadsX == minRowThreadsXInPending) {
                if (!hasBestRowThreadsXLowAnchor ||
                    pc.estimatedLatency <
                        bestRowThreadsXLowAnchor.estimatedLatency) {
                  bestRowThreadsXLowAnchor = pc;
                  hasBestRowThreadsXLowAnchor = true;
                }
              }
              if (hasRowThreadsXSplit &&
                  pc.cand.rowReductionThreadsX == maxRowThreadsXInPending) {
                if (!hasBestRowThreadsXHighAnchor ||
                    pc.estimatedLatency <
                        bestRowThreadsXHighAnchor.estimatedLatency) {
                  bestRowThreadsXHighAnchor = pc;
                  hasBestRowThreadsXHighAnchor = true;
                }
              }
            }
          }
        } else if (hasBestFallback) {
          best = std::move(bestFallback);
          hasBest = true;
        }
        // 若 profiling 提前停止且未覆盖所有旋钮类，则保留一个
        // 按模型排序的后备锚点（来自当前 pending 集），以保持
        // 输出的类别多样性（论文式探索意图）。
        auto fillMissingAnchorFromPending = [&](auto pred, bool &hasAnchor,
                                                PaperScheduleCandidate &anchor) {
          if (hasAnchor)
            return;
          bool found = false;
          for (const PaperScheduleCandidate &pc : pending) {
            if (!pred(pc.cand))
              continue;
            if (!found || pc.estimatedLatency < anchor.estimatedLatency) {
              anchor = pc;
              found = true;
            }
          }
          hasAnchor = found;
        };
        if (retainTensorCoreClassForMatmul) {
          fillMissingAnchorFromPending(
              [&](const Candidate &c) { return isTensorCoreCand(c); },
              hasBestTensorCoreAnchor, bestTensorCoreAnchor);
        }
        if (wantNonTensorCoreAnchor) {
          fillMissingAnchorFromPending(
              [&](const Candidate &c) { return !isTensorCoreCand(c); },
              hasBestNonTensorCoreAnchor, bestNonTensorCoreAnchor);
        }
        if (wantAsyncAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) { return c.enableAsyncCopy; },
              hasBestAsyncAnchor, bestAsyncAnchor);
        }
        if (wantPipeAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) { return c.enableSoftwarePipelining; },
              hasBestPipeAnchor, bestPipeAnchor);
        }
        if (wantWaitGroupAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) {
                return c.enableSoftwarePipelining &&
                       c.pipelineSetAsyncWaitGroups;
              },
              hasBestWaitGroupAnchor, bestWaitGroupAnchor);
        }
        if (wantRowReuseAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) {
                return c.enableRowReductionChainReuseFusion;
              },
              hasBestRowReuseAnchor, bestRowReuseAnchor);
        }
        if (wantRowPromoAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) {
                return c.enableRowReductionInputPromotion;
              },
              hasBestRowPromoAnchor, bestRowPromoAnchor);
        }
        if (wantRowWarpAnchor) {
          fillMissingAnchorFromPending(
              [](const Candidate &c) { return c.enableRowReductionWarp; },
              hasBestRowWarpAnchor, bestRowWarpAnchor);
        }
        if (hasRowThreadsXSplit) {
          fillMissingAnchorFromPending(
              [&](const Candidate &c) {
                return c.rowReductionThreadsX == minRowThreadsXInPending;
              },
              hasBestRowThreadsXLowAnchor, bestRowThreadsXLowAnchor);
          fillMissingAnchorFromPending(
              [&](const Candidate &c) {
                return c.rowReductionThreadsX == maxRowThreadsXInPending;
              },
              hasBestRowThreadsXHighAnchor, bestRowThreadsXHighAnchor);
        }
      }
      if (hasBest) {
        bool pushedAnchor = false;
        bool pushedExtraMeasured = false;
        std::unordered_set<std::string> pushedKeys;
        pushedKeys.reserve(16);
        auto makeCandidateKey = [&](const Candidate &cand) {
          return buildProfileKeyForSubgraph(g, sgAfterCuts, sinkNodeIdx, cand);
        };
        bool hasAsyncClass = best.cand.enableAsyncCopy;
        bool hasPipeClass = best.cand.enableSoftwarePipelining;
        bool hasWaitGroupClass =
            best.cand.enableSoftwarePipelining &&
            best.cand.pipelineSetAsyncWaitGroups;
        bool hasRowReuseClass = best.cand.enableRowReductionChainReuseFusion;
        bool hasRowPromoClass = best.cand.enableRowReductionInputPromotion;
        bool hasRowWarpClass = best.cand.enableRowReductionWarp;
        bool hasRowThreadsXLowClass =
            hasRowThreadsXSplit &&
            best.cand.rowReductionThreadsX == minRowThreadsXInPending;
        bool hasRowThreadsXHighClass =
            hasRowThreadsXSplit &&
            best.cand.rowReductionThreadsX == maxRowThreadsXInPending;
        auto pushAnchor = [&](const PaperScheduleCandidate &pc) -> bool {
          if (static_cast<int64_t>(out.size()) + 1 >= maxOut)
            return false;
          std::string key = makeCandidateKey(pc.cand);
          if (!pushedKeys.insert(key).second)
            return false;
          out.push_back(pc);
          pushedAnchor = true;
          hasAsyncClass = hasAsyncClass || pc.cand.enableAsyncCopy;
          hasPipeClass = hasPipeClass || pc.cand.enableSoftwarePipelining;
          hasWaitGroupClass =
              hasWaitGroupClass ||
              (pc.cand.enableSoftwarePipelining &&
               pc.cand.pipelineSetAsyncWaitGroups);
          hasRowReuseClass =
              hasRowReuseClass || pc.cand.enableRowReductionChainReuseFusion;
          hasRowPromoClass =
              hasRowPromoClass || pc.cand.enableRowReductionInputPromotion;
          hasRowWarpClass = hasRowWarpClass || pc.cand.enableRowReductionWarp;
          hasRowThreadsXLowClass =
              hasRowThreadsXLowClass ||
              (hasRowThreadsXSplit &&
               pc.cand.rowReductionThreadsX == minRowThreadsXInPending);
          hasRowThreadsXHighClass =
              hasRowThreadsXHighClass ||
              (hasRowThreadsXSplit &&
               pc.cand.rowReductionThreadsX == maxRowThreadsXInPending);
          return true;
        };
        if (retainTensorCoreClassForMatmul &&
            hasBestTensorCoreAnchor && !isTensorCoreCand(best.cand) &&
            static_cast<int64_t>(out.size()) + 1 < maxOut) {
          pushAnchor(bestTensorCoreAnchor);
        }
        if (wantNonTensorCoreAnchor && hasBestNonTensorCoreAnchor &&
            isTensorCoreCand(best.cand) &&
            static_cast<int64_t>(out.size()) + 1 < maxOut) {
          pushAnchor(bestNonTensorCoreAnchor);
        }
        const bool keepExtraCodegenClassForRecursive = true;
        if (keepExtraCodegenClassForRecursive) {
          if (wantRowReuseAnchor && hasBestRowReuseAnchor && !hasRowReuseClass)
            pushAnchor(bestRowReuseAnchor);
          if (wantRowPromoAnchor && hasBestRowPromoAnchor && !hasRowPromoClass)
            pushAnchor(bestRowPromoAnchor);
          if (wantRowWarpAnchor && hasBestRowWarpAnchor && !hasRowWarpClass)
            pushAnchor(bestRowWarpAnchor);
          if (hasRowThreadsXSplit && hasBestRowThreadsXLowAnchor &&
              !hasRowThreadsXLowClass)
            pushAnchor(bestRowThreadsXLowAnchor);
          if (hasRowThreadsXSplit && hasBestRowThreadsXHighAnchor &&
              !hasRowThreadsXHighClass)
            pushAnchor(bestRowThreadsXHighAnchor);
          if (wantWaitGroupAnchor && hasBestWaitGroupAnchor &&
              !hasWaitGroupClass)
            pushAnchor(bestWaitGroupAnchor);
          if (wantPipeAnchor && hasBestPipeAnchor && !hasPipeClass)
            pushAnchor(bestPipeAnchor);
          if (wantAsyncAnchor && hasBestAsyncAnchor && !hasAsyncClass)
            pushAnchor(bestAsyncAnchor);
        }
        {
          std::string bestKey = makeCandidateKey(best.cand);
          if (pushedKeys.insert(bestKey).second)
            out.push_back(std::move(best));
        }
        if (!extraMeasuredForOut.empty()) {
          for (const PaperScheduleCandidate &pc : extraMeasuredForOut) {
            if (static_cast<int64_t>(out.size()) >= maxOut)
              break;
            std::string key = makeCandidateKey(pc.cand);
            if (!pushedKeys.insert(key).second)
              continue;
            out.push_back(pc);
            pushedExtraMeasured = true;
          }
        }
        if (static_cast<int64_t>(out.size()) >= maxOut)
          stop = true;
        if ((pushedAnchor || pushedExtraMeasured) &&
            static_cast<int64_t>(out.size()) >= maxOut)
          stop = true;
      }
	    } else {
		      if (profileEnabledForSubgraph) {
        llvm::SmallVector<char, 64> profiledOk;
        profiledOk.assign(pending.size(), 0);
        bool hasProfileSuccess = false;
        int64_t profileAttempts = 0;
        int64_t profileCompiles = 0;
        int64_t consecutiveProfileFailures = 0;
        int64_t noImproveProfileStreak = 0;
        double bestMeasuredProfileMs = std::numeric_limits<double>::infinity();
        auto isTcAsyncWaitCandidateEarlyStop = [](const Candidate &c) {
          return c.enableTensorCoreF16 && c.enableAsyncCopy &&
                 c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
        };
        const bool enableTargetClassEarlyStop =
            matmulSoftmaxLikeContext && allowTensorCoreClassPreference &&
            (getEnvInt64OrDefault("WELDER_PROFILE_MM_SM_TARGET_CLASS_EARLY_STOP",
                                  1) != 0);
        const int64_t targetClassMinAttempts = std::max<int64_t>(
            int64_t(1), getEnvInt64OrDefault(
                            "WELDER_PROFILE_MM_SM_TARGET_CLASS_MIN_ATTEMPTS", 6));
        const double targetClassMaxRatio = std::max(
            1.0, getEnvDoubleOrDefault(
                     "WELDER_PROFILE_MM_SM_TARGET_CLASS_MAX_RATIO", 1.30));
        const bool hasNonTcClassInPending = llvm::any_of(
            pending, [&](const PaperScheduleCandidate &pc) {
              return !(pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32);
            });
        bool hasMeasuredTcAsyncWait = false;
        bool hasMeasuredNonTc = false;
        double bestMeasuredTcAsyncWaitMs = std::numeric_limits<double>::infinity();
        for (int64_t i : profileIdxs) {
          if (maxProfileAttemptsPerSharedTile > 0 &&
              profileAttempts >= maxProfileAttemptsPerSharedTile) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = "attempt_cap";
              f["phase"] = "non_recursive";
              f["attempts"] = profileAttempts;
              f["cap"] = maxProfileAttemptsPerSharedTile;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
          if (maxProfileCompilesPerSharedTile > 0 &&
              profileCompiles >= maxProfileCompilesPerSharedTile) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = "compile_cap";
              f["phase"] = "non_recursive";
              f["compiles"] = profileCompiles;
              f["cap"] = maxProfileCompilesPerSharedTile;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
          if (maxProfileCompilesTotal > 0 &&
              profileCompilesTotal >= maxProfileCompilesTotal) {
            stopOnProfileCompileTotalCap = true;
            if (opts.tracer && !emittedProfileCompileTotalCapEvent) {
              llvm::json::Object f;
              f["reason"] = "compile_total_cap";
              f["phase"] = "non_recursive";
              f["compiles"] = profileCompilesTotal;
              f["cap"] = maxProfileCompilesTotal;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
              emittedProfileCompileTotalCapEvent = true;
            }
            break;
          }
          if (i < 0 || i >= static_cast<int64_t>(pending.size()))
            continue;
          PaperScheduleCandidate &pc = pending[static_cast<size_t>(i)];
          bool wasCached = false;
          bool wasMeasured = false;
          bool skipByGlobalCompileCap = false;
          auto msOpt = profileSubgraphByCompilingToNvvm(
              g, sgAfterCuts, sinkNodeIdx, pc.cand, opts,
              /* outWasCached=*/&wasCached);
          if (msOpt) {
            pc.cand.cost.profiledMs = *msOpt;
            pc.cand.score = *msOpt;
            pc.estimatedLatency = *msOpt;
            profiledOk[static_cast<size_t>(i)] = 1;
            hasProfileSuccess = true;
            wasMeasured = true;
          } else if (wasCached) {
            // 出现“缓存命中失败”不可能；这里编码的是全局编译预算跳过。
            skipByGlobalCompileCap = true;
          } else if (opts.pruneOnProfileFailure || pc.cand.enableTensorCoreF16 ||
                     pc.cand.enableTensorCoreTf32) {
            pc.estimatedLatency = std::numeric_limits<double>::infinity();
          }

          if (!wasCached) {
            ++profileCompiles;
            ++profileCompilesTotal;
          }
          if (skipByGlobalCompileCap)
            break;
          ++profileAttempts;
          if (wasMeasured && std::isfinite(pc.estimatedLatency)) {
            consecutiveProfileFailures = 0;
            bool improved =
                !std::isfinite(bestMeasuredProfileMs) ||
                pc.estimatedLatency <
                    bestMeasuredProfileMs * (1.0 - profileImprovementRatio);
            if (improved) {
              bestMeasuredProfileMs = pc.estimatedLatency;
              noImproveProfileStreak = 0;
            } else {
              ++noImproveProfileStreak;
            }
            const bool isTc =
                pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32;
            if (!isTc)
              hasMeasuredNonTc = true;
            if (isTcAsyncWaitCandidateEarlyStop(pc.cand)) {
              hasMeasuredTcAsyncWait = true;
              bestMeasuredTcAsyncWaitMs =
                  std::min(bestMeasuredTcAsyncWaitMs, pc.estimatedLatency);
            }
          } else {
            ++consecutiveProfileFailures;
            ++noImproveProfileStreak;
          }

          bool shouldEarlyStop = false;
          const char *earlyStopReason = "";
          if (profileAttempts >= minProfileAttemptsBeforeEarlyStop) {
            if (maxConsecutiveProfileFailures > 0 &&
                consecutiveProfileFailures >= maxConsecutiveProfileFailures) {
              shouldEarlyStop = true;
              earlyStopReason = "consecutive_failures";
            } else if (noImprovementProfileWindow > 0 &&
                       noImproveProfileStreak >= noImprovementProfileWindow) {
              shouldEarlyStop = true;
              earlyStopReason = "no_improvement";
            }
          }
          if (!shouldEarlyStop && enableTargetClassEarlyStop &&
              profileAttempts >= targetClassMinAttempts &&
              hasMeasuredTcAsyncWait &&
              (!hasNonTcClassInPending || hasMeasuredNonTc) &&
              std::isfinite(bestMeasuredProfileMs) &&
              bestMeasuredTcAsyncWaitMs <=
                  bestMeasuredProfileMs * targetClassMaxRatio) {
            shouldEarlyStop = true;
            earlyStopReason = "target_class_hit";
          }
          if (shouldEarlyStop) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["reason"] = earlyStopReason;
              f["phase"] = "non_recursive";
              f["attempts"] = profileAttempts;
              f["consecutive_failures"] = consecutiveProfileFailures;
              f["no_improvement_streak"] = noImproveProfileStreak;
              f["best_measured_ms"] =
                  std::isfinite(bestMeasuredProfileMs) ? bestMeasuredProfileMs
                                                       : -1.0;
              opts.tracer->event("profile.early_stop", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
        }
        if (hasProfileSuccess) {
          for (size_t i = 0; i < pending.size(); ++i) {
            if (!profiledOk[i])
              pending[i].estimatedLatency = std::numeric_limits<double>::infinity();
          }

          auto isTcAsyncWaitCandidate = [](const Candidate &c) {
            return c.enableTensorCoreF16 && c.enableAsyncCopy &&
                   c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
          };
          const bool preferTcAsyncWaitForMatmulSoftmax =
              matmulSoftmaxLikeContext && allowTensorCoreClassPreference;
          double waitGroupRequireRatio = std::max(
              0.5, getEnvDoubleOrDefault("WELDER_PROFILE_WAIT_GROUP_REQUIRE_RATIO",
                                         /*default=*/1.0));
          if (preferTcAsyncWaitForMatmulSoftmax) {
            const double mmSmWaitGroupMinRatio = std::max(
                1.0,
                getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_WAIT_GROUP_MIN_RATIO",
                                      /*default=*/1.20));
            waitGroupRequireRatio =
                std::max(waitGroupRequireRatio, mmSmWaitGroupMinRatio);
          }
          const double tcAsyncWaitPreferRatio = std::max(
              1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_ASYNC_WAIT_PREFER_RATIO",
                                         /*default=*/1.20));
          double bestNonPipeMeasured = std::numeric_limits<double>::infinity();
          for (size_t i = 0; i < pending.size(); ++i) {
            if (!profiledOk[i])
              continue;
            const Candidate &cand = pending[i].cand;
            bool isWaitGroupPipe =
                cand.enableSoftwarePipelining && cand.pipelineSetAsyncWaitGroups;
            if (isWaitGroupPipe)
              continue;
            if (std::isfinite(pending[i].estimatedLatency))
              bestNonPipeMeasured =
                  std::min(bestNonPipeMeasured, pending[i].estimatedLatency);
          }
          if (std::isfinite(bestNonPipeMeasured)) {
            for (size_t i = 0; i < pending.size(); ++i) {
              if (!profiledOk[i])
                continue;
              Candidate &cand = pending[i].cand;
              bool isWaitGroupPipe =
                  cand.enableSoftwarePipelining &&
                  cand.pipelineSetAsyncWaitGroups;
              if (!isWaitGroupPipe)
                continue;
              if (pending[i].estimatedLatency <
                  bestNonPipeMeasured * waitGroupRequireRatio) {
                continue;
              }
              if (opts.tracer) {
                llvm::json::Object f;
                f["wait_group_ms"] = pending[i].estimatedLatency;
                f["anchor_non_pipe_ms"] = bestNonPipeMeasured;
                f["ratio"] = waitGroupRequireRatio;
                opts.tracer->event("profile.wait_group_anchor_reject",
                                   std::move(f), /*isVerbose=*/true);
              }
              pending[i].estimatedLatency =
                  std::numeric_limits<double>::infinity();
            }
          }
          if (preferTcAsyncWaitForMatmulSoftmax) {
            int64_t bestMeasuredIdx = -1;
            int64_t bestTcAsyncWaitIdx = -1;
            for (int64_t i = 0; i < static_cast<int64_t>(pending.size()); ++i) {
              if (!profiledOk[static_cast<size_t>(i)] ||
                  !std::isfinite(pending[static_cast<size_t>(i)].estimatedLatency))
                continue;
              if (bestMeasuredIdx < 0 ||
                  pending[static_cast<size_t>(i)].estimatedLatency <
                      pending[static_cast<size_t>(bestMeasuredIdx)]
                          .estimatedLatency) {
                bestMeasuredIdx = i;
              }
              if (isTcAsyncWaitCandidate(pending[static_cast<size_t>(i)].cand) &&
                  (bestTcAsyncWaitIdx < 0 ||
                   pending[static_cast<size_t>(i)].estimatedLatency <
                       pending[static_cast<size_t>(bestTcAsyncWaitIdx)]
                           .estimatedLatency)) {
                bestTcAsyncWaitIdx = i;
              }
            }
            if (bestMeasuredIdx >= 0 && bestTcAsyncWaitIdx >= 0) {
              double baseMs =
                  pending[static_cast<size_t>(bestMeasuredIdx)].estimatedLatency;
              double tcAsyncWaitMs =
                  pending[static_cast<size_t>(bestTcAsyncWaitIdx)]
                      .estimatedLatency;
              if (tcAsyncWaitMs <= baseMs * tcAsyncWaitPreferRatio) {
                pending[static_cast<size_t>(bestTcAsyncWaitIdx)].estimatedLatency =
                    std::min(tcAsyncWaitMs, baseMs * 0.999999);
                if (opts.tracer) {
                  llvm::json::Object f;
                  f["base_ms"] = baseMs;
                  f["tc_async_wait_ms"] = tcAsyncWaitMs;
                  f["ratio"] = tcAsyncWaitPreferRatio;
                  opts.tracer->event("profile.tc_async_wait_prefer",
                                     std::move(f), /*isVerbose=*/true);
                }
              }
            }
          }
        }
        llvm::sort(pending, betterEstimate);
      }
      for (int64_t i = 0; i < keepN; ++i) {
        PaperScheduleCandidate pc = pending[static_cast<size_t>(i)];
        if (opts.pruneOnProfileFailure &&
            !std::isfinite(pc.estimatedLatency))
          continue;
        out.push_back(std::move(pc));
        if (static_cast<int64_t>(out.size()) >= maxOut) {
          stop = true;
          break;
        }
      }
    }

    if (stop)
      break;
  }

  if (ensureGlobalClassAnchors && !out.empty()) {
    std::unordered_set<std::string> outKeys;
    outKeys.reserve(out.size() * 2 + 8);
    const PaperScheduleCandidate outBestRef = out.front();
    for (const PaperScheduleCandidate &pc : out) {
      outKeys.insert(
          buildProfileKeyForSubgraph(graph, sg, sinkNodeIdx, pc.cand));
    }
    auto appendGlobalAnchorIfMissing =
        [&](auto pred, bool hasAnchor, const PaperScheduleCandidate &anchor) {
          if (!hasAnchor)
            return;
          for (const PaperScheduleCandidate &pc : out) {
            if (pred(pc.cand))
              return;
          }
          std::string key =
              buildProfileKeyForSubgraph(graph, sg, sinkNodeIdx, anchor.cand);
          if (!outKeys.insert(key).second)
            return;
          out.push_back(anchor);
        };
    auto appendSyntheticAnchorIfMissing = [&](auto pred, auto mutate) {
      for (const PaperScheduleCandidate &pc : out) {
        if (pred(pc.cand))
          return;
      }
      PaperScheduleCandidate synth = outBestRef;
      mutate(synth.cand);
      synth.cand.cost.profiledMs = std::nullopt;
      if (std::isfinite(outBestRef.estimatedLatency))
        synth.estimatedLatency = outBestRef.estimatedLatency * 1.25;
      else if (std::isfinite(outBestRef.cand.score))
        synth.estimatedLatency = outBestRef.cand.score * 1.25;
      else
        synth.estimatedLatency = std::numeric_limits<double>::infinity();
      synth.cand.score = std::isfinite(synth.estimatedLatency)
                             ? synth.estimatedLatency
                             : outBestRef.cand.score;
      std::string key =
          buildProfileKeyForSubgraph(graph, sg, sinkNodeIdx, synth.cand);
      if (!outKeys.insert(key).second)
        return;
      out.push_back(std::move(synth));
    };
    if (wantGlobalWaitGroupAnchor) {
      appendGlobalAnchorIfMissing(
          [](const Candidate &c) {
            return c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
          },
          hasGlobalWaitGroupAnchor, globalWaitGroupAnchor);
      appendSyntheticAnchorIfMissing(
          [](const Candidate &c) {
            return c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
          },
          [](Candidate &c) {
            c.enableAsyncCopy = true;
            c.asyncBypassL1 = true;
            c.enableSoftwarePipelining = true;
            c.pipelineSetAsyncWaitGroups = true;
            c.pipelineDepth = std::max<int64_t>(2, c.pipelineDepth);
            c.workgroupMultiBufferDepth =
                std::max<int64_t>(2, c.workgroupMultiBufferDepth);
          });
    }
    if (wantGlobalPipeAnchor) {
      appendGlobalAnchorIfMissing(
          [](const Candidate &c) { return c.enableSoftwarePipelining; },
          hasGlobalPipeAnchor, globalPipeAnchor);
      appendSyntheticAnchorIfMissing(
          [](const Candidate &c) { return c.enableSoftwarePipelining; },
          [](Candidate &c) {
            c.enableAsyncCopy = true;
            c.asyncBypassL1 = true;
            c.enableSoftwarePipelining = true;
            c.pipelineSetAsyncWaitGroups = false;
            c.pipelineDepth = std::max<int64_t>(2, c.pipelineDepth);
            c.workgroupMultiBufferDepth =
                std::max<int64_t>(2, c.workgroupMultiBufferDepth);
          });
    }
    if (wantGlobalAsyncAnchor) {
      appendGlobalAnchorIfMissing(
          [](const Candidate &c) { return c.enableAsyncCopy; },
          hasGlobalAsyncAnchor, globalAsyncAnchor);
      appendSyntheticAnchorIfMissing(
          [](const Candidate &c) { return c.enableAsyncCopy; },
          [](Candidate &c) {
            c.enableAsyncCopy = true;
            c.asyncBypassL1 = true;
          });
    }
    if (wantGlobalTensorCoreF16Anchor) {
      appendGlobalAnchorIfMissing(
          [](const Candidate &c) { return c.enableTensorCoreF16; },
          hasGlobalTensorCoreF16Anchor, globalTensorCoreF16Anchor);
      appendSyntheticAnchorIfMissing(
          [](const Candidate &c) { return c.enableTensorCoreF16; },
          [](Candidate &c) {
            c.enableTensorCoreF16 = true;
            c.enableTensorCoreTf32 = false;
            c.useCutlassMma = true;
            if (c.mmaM <= 0)
              c.mmaM = 16;
            if (c.mmaN <= 0)
              c.mmaN = 8;
            if (c.mmaK <= 0)
              c.mmaK = 16;
          });
    }
    if (wantGlobalNonTensorCoreAnchor) {
      appendGlobalAnchorIfMissing(
          [](const Candidate &c) {
            return !(c.enableTensorCoreF16 || c.enableTensorCoreTf32);
          },
          hasGlobalNonTensorCoreAnchor, globalNonTensorCoreAnchor);
      appendSyntheticAnchorIfMissing(
          [](const Candidate &c) {
            return !(c.enableTensorCoreF16 || c.enableTensorCoreTf32);
          },
          [](Candidate &c) {
            c.enableTensorCoreF16 = false;
            c.enableTensorCoreTf32 = false;
            c.useCutlassMma = false;
            c.mmaM = 0;
            c.mmaN = 0;
            c.mmaK = 0;
          });
    }
  }

  const bool graphHasMatmulSoftmaxForFinalSort =
      graphHasMatmulSoftmaxLikePattern(graph);
  bool subgraphHasMatmulForFinalSort = subgraphHasMatmulOp(graph, sg);
  if (!subgraphHasMatmulForFinalSort && sinkNodeIdx >= 0 &&
      sinkNodeIdx < static_cast<int>(graph.nodes.size())) {
    subgraphHasMatmulForFinalSort =
        isa<linalg::MatmulOp>(graph.nodes[sinkNodeIdx].op);
  }
  const bool searchWantsTensorCoreForFinalSort =
      opts.codegenSearch.enable &&
      (llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true) ||
       llvm::is_contained(opts.codegenSearch.enableTensorCoreTf32, true));
	  const bool profileWantsTensorCoreForFinalSort =
	      profileEnabledForSubgraph && opts.profile.enableTensorCoreF16;
  const bool allowTensorCoreForFinalSort =
      (subgraphHasMatmulForFinalSort || graphHasMatmulSoftmaxForFinalSort) &&
      (searchWantsTensorCoreForFinalSort || profileWantsTensorCoreForFinalSort);
  const bool retainTensorCoreClassForFinalSort = allowTensorCoreForFinalSort;
  const bool wantMatmulSoftmaxReuseForFinalSort =
      opts.profile.enableMatmulSoftmaxSharedReuseFusion ||
      (opts.codegenSearch.enable &&
       llvm::is_contained(
           opts.codegenSearch.enableMatmulSoftmaxSharedReuseFusion, true));
  const bool matmulSoftmaxLikeContextForFinalSort =
      isMatmulSoftmaxLikeSubgraph(graph, sg) ||
      (wantMatmulSoftmaxReuseForFinalSort &&
       graphHasMatmulSoftmaxForFinalSort);
  const bool preferTensorCoreForMatmulSoftmaxFinalSort =
      allowTensorCoreForFinalSort &&
      wantMatmulSoftmaxReuseForFinalSort &&
      matmulSoftmaxLikeContextForFinalSort;

  if (profileEnabledForSubgraph && matmulSoftmaxLikeContextForFinalSort &&
      !out.empty()) {
    int64_t targetedProfileBudget = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TARGETED_MM_SM_BUDGET", 2));
    auto isTcAsyncWaitCandidate = [](const Candidate &c) {
      return c.enableTensorCoreF16 && c.enableAsyncCopy &&
             c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
    };
    auto isTcAsyncCandidate = [](const Candidate &c) {
      return c.enableTensorCoreF16 && c.enableAsyncCopy;
    };
    auto isProfileMissing = [](const PaperScheduleCandidate &pc) {
      return !pc.cand.cost.profiledMs.has_value() ||
             !std::isfinite(*pc.cand.cost.profiledMs) ||
             *pc.cand.cost.profiledMs <= 0.0;
    };
    bool hasTcAsyncWaitClass = false;
    bool hasTcAsyncClass = false;
    for (const PaperScheduleCandidate &pc : out) {
      hasTcAsyncWaitClass |= isTcAsyncWaitCandidate(pc.cand);
      hasTcAsyncClass |= isTcAsyncCandidate(pc.cand);
    }
    if (!hasTcAsyncWaitClass && targetedProfileBudget > 0)
      targetedProfileBudget = std::max<int64_t>(targetedProfileBudget, 6);
    else if (!hasTcAsyncClass && targetedProfileBudget > 0)
      targetedProfileBudget = std::max<int64_t>(targetedProfileBudget, 4);
    int64_t targetedPerIndexBudget = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TARGETED_MM_SM_PER_INDEX_BUDGET",
                                !hasTcAsyncWaitClass ? 3 : 0));
    llvm::SmallVector<int64_t, 8> targetedIdxs;
    llvm::DenseSet<int64_t> forceRetargetProfiledIdxs;
    auto pushTargetedIdx = [&](auto pred) {
      int64_t bestIdx = -1;
      double bestEst = std::numeric_limits<double>::infinity();
      for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        const PaperScheduleCandidate &pc = out[static_cast<size_t>(i)];
        if (!pred(pc.cand) || !isProfileMissing(pc))
          continue;
        if (!std::isfinite(pc.estimatedLatency))
          continue;
        if (bestIdx < 0 || pc.estimatedLatency < bestEst) {
          bestIdx = i;
          bestEst = pc.estimatedLatency;
        }
      }
      if (bestIdx >= 0 && !llvm::is_contained(targetedIdxs, bestIdx))
        targetedIdxs.push_back(bestIdx);
    };
    auto pushTargetedIdxAllowProfiled = [&](auto pred, bool forceRetargetProfiled) {
      int64_t bestIdx = -1;
      double bestEst = std::numeric_limits<double>::infinity();
      for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        const PaperScheduleCandidate &pc = out[static_cast<size_t>(i)];
        if (!pred(pc.cand))
          continue;
        if (!std::isfinite(pc.estimatedLatency))
          continue;
        if (bestIdx < 0 || pc.estimatedLatency < bestEst) {
          bestIdx = i;
          bestEst = pc.estimatedLatency;
        }
      }
      if (bestIdx >= 0 && !llvm::is_contained(targetedIdxs, bestIdx)) {
        targetedIdxs.push_back(bestIdx);
        if (forceRetargetProfiled)
          forceRetargetProfiledIdxs.insert(bestIdx);
      }
    };
    pushTargetedIdx(isTcAsyncWaitCandidate);
    pushTargetedIdx(isTcAsyncCandidate);
    if (!hasTcAsyncWaitClass) {
      pushTargetedIdxAllowProfiled(isTcAsyncCandidate,
                                   /* forceRetargetProfiled=*/true);
      pushTargetedIdxAllowProfiled(
          [](const Candidate &c) { return c.enableTensorCoreF16; },
          /* forceRetargetProfiled=*/true);
      pushTargetedIdxAllowProfiled(
          [](const Candidate &) { return true; },
          /* forceRetargetProfiled=*/true);
    }
    if (!hasTcAsyncClass) {
      pushTargetedIdxAllowProfiled(
          [](const Candidate &c) { return c.enableTensorCoreF16; },
          /* forceRetargetProfiled=*/true);
    }
    if (targetedIdxs.empty()) {
      pushTargetedIdx([](const Candidate &c) { return c.enableTensorCoreF16; });
    }
    int64_t profiledTargets = 0;
    int64_t targetedAttempts = 0;
    auto tryTargetedProfile = [&](PaperScheduleCandidate &pc, int64_t idx,
                                  const Candidate &attemptCand,
                                  llvm::StringRef variant) -> bool {
      if (targetedProfileBudget > 0 && targetedAttempts >= targetedProfileBudget)
        return false;
      if (maxProfileCompilesTotal > 0 &&
          profileCompilesTotal >= maxProfileCompilesTotal)
        return false;
      ++targetedAttempts;
      bool wasCached = false;
      auto msOpt = profileSubgraphByCompilingToNvvm(
          graph, sg, sinkNodeIdx, attemptCand, opts, /*outWasCached=*/&wasCached);
      if (!wasCached)
        ++profileCompilesTotal;
      if (msOpt) {
        pc.cand = attemptCand;
        pc.cand.cost.profiledMs = *msOpt;
        pc.cand.score = *msOpt;
        pc.estimatedLatency = *msOpt;
        ++profiledTargets;
        if (opts.tracer) {
          llvm::json::Object f;
          f["idx"] = idx;
          f["ms"] = *msOpt;
          f["variant"] = variant.str();
          f["attempt"] = targetedAttempts;
          f["tc_async_wait"] = isTcAsyncWaitCandidate(pc.cand);
          f["tc_async"] = isTcAsyncCandidate(pc.cand);
          opts.tracer->event("profile.targeted_mm_sm_ok", std::move(f),
                             /* isVerbose=*/true);
        }
        return true;
      }
      if (opts.tracer) {
        llvm::json::Object f;
        f["idx"] = idx;
        f["variant"] = variant.str();
        f["attempt"] = targetedAttempts;
        f["cached_skip"] = wasCached;
        f["tc_async_wait"] = isTcAsyncWaitCandidate(attemptCand);
        f["tc_async"] = isTcAsyncCandidate(attemptCand);
        opts.tracer->event("profile.targeted_mm_sm_fail", std::move(f),
                           /* isVerbose=*/true);
      }
      return false;
    };
    for (int64_t idx : targetedIdxs) {
      if (targetedProfileBudget > 0 && targetedAttempts >= targetedProfileBudget)
        break;
      if (idx < 0 || idx >= static_cast<int64_t>(out.size()))
        continue;
      PaperScheduleCandidate &pc = out[static_cast<size_t>(idx)];
      const bool forceRetargetProfiled =
          forceRetargetProfiledIdxs.contains(idx);
      if (!isProfileMissing(pc) && !forceRetargetProfiled)
        continue;
      Candidate baseCand = pc.cand;
      int64_t attemptsAtIdx = 0;
      auto tryTargetedProfileAtIdx = [&](const Candidate &attemptCand,
                                         llvm::StringRef variant) {
        if (targetedPerIndexBudget > 0 && attemptsAtIdx >= targetedPerIndexBudget)
          return false;
        const int64_t attemptsBefore = targetedAttempts;
        bool ok = tryTargetedProfile(pc, idx, attemptCand, variant);
        if (targetedAttempts > attemptsBefore)
          ++attemptsAtIdx;
        return ok;
      };
      if (isProfileMissing(pc) &&
          tryTargetedProfileAtIdx(baseCand, "original"))
        continue;

      const bool isTcAsyncLike =
          isTcAsyncWaitCandidate(baseCand) || isTcAsyncCandidate(baseCand);
      auto buildSafeRowCandidate = [&](const Candidate &src, int64_t rowThreadsX,
                                       bool forceWaitGroups,
                                       bool dropRowReuse) {
        Candidate c = src;
        c.enableMatmulSoftmaxSharedReuseFusion = true;
        c.enableAsyncCopy = true;
        c.enableSoftwarePipelining = true;
        c.pipelineDepth = 2;
        c.workgroupMultiBufferDepth =
            std::max<int64_t>(2, c.workgroupMultiBufferDepth);
        c.pipelineSetAsyncWaitGroups = forceWaitGroups;
        c.workgroupPadLastDim = std::max<int64_t>(c.workgroupPadLastDim, 8);
        c.workgroupPadLastDimMatmulOnly = true;
        c.workgroupSwizzleXor = 0;
        c.enableRowReductionWarp = false;
        c.enableRowReductionVectorize = false;
        c.enableRowReductionCombineVectorize = false;
        c.enableRowReductionInputPromotionVectorize = false;
        c.enableRowReductionRelaxBarriers = false;
        c.enableRowReductionSkipCombineBarrier = false;
        c.rowReductionVectorWidth = 0;
        c.rowReductionInputVectorWidth = 0;
        c.rowReductionThreadsX = std::max<int64_t>(1, rowThreadsX);
        c.threadTileM = 1;
        c.threadTileN = 1;
        if (dropRowReuse) {
          c.enableRowReductionChainReuseFusion = false;
          c.enableRowReductionInputPromotion = false;
          c.enableRowReductionInputPromotionVectorize = false;
          c.rowReductionInputVectorWidth = 0;
        }
        return c;
      };

      llvm::SmallVector<int64_t, 4> rowThreadsOrder;
      auto pushRowThreads = [&](int64_t tx) {
        tx = std::max<int64_t>(1, tx);
        if (!llvm::is_contained(rowThreadsOrder, tx))
          rowThreadsOrder.push_back(tx);
      };
      pushRowThreads(8);
      pushRowThreads(16);
      if (baseCand.rowReductionThreadsX > 0)
        pushRowThreads(baseCand.rowReductionThreadsX);
      if (rowThreadsOrder.empty())
        rowThreadsOrder.push_back(8);
      bool baseWantsWaitGroups = isTcAsyncWaitCandidate(baseCand);
      Candidate seedCand = baseCand;
      if (!isTcAsyncLike && !seedCand.enableTensorCoreF16 && !hasTcAsyncWaitClass) {
        seedCand.enableTensorCoreF16 = true;
        seedCand.enableTensorCoreTf32 = false;
        seedCand.useCutlassMma = true;
        if (seedCand.mmaM <= 0)
          seedCand.mmaM = 16;
        if (seedCand.mmaN <= 0)
          seedCand.mmaN = 8;
        if (seedCand.mmaK <= 0)
          seedCand.mmaK = 16;
        seedCand.enableAsyncCopy = true;
        seedCand.enableSoftwarePipelining = true;
        seedCand.pipelineDepth = 2;
        seedCand.workgroupMultiBufferDepth =
            std::max<int64_t>(2, seedCand.workgroupMultiBufferDepth);
        seedCand.pipelineSetAsyncWaitGroups = true;
        seedCand.enableMatmulSoftmaxSharedReuseFusion = true;
        baseWantsWaitGroups = true;
      } else if (!isTcAsyncLike && seedCand.enableTensorCoreF16) {
        seedCand.enableAsyncCopy = true;
        seedCand.enableSoftwarePipelining = true;
        seedCand.pipelineDepth = 2;
        seedCand.workgroupMultiBufferDepth =
            std::max<int64_t>(2, seedCand.workgroupMultiBufferDepth);
        seedCand.pipelineSetAsyncWaitGroups = true;
        seedCand.enableMatmulSoftmaxSharedReuseFusion = true;
        baseWantsWaitGroups = true;
      } else if (!isTcAsyncLike) {
        continue;
      }
      if (!hasTcAsyncWaitClass && seedCand.enableTensorCoreF16 &&
          seedCand.enableAsyncCopy) {
        seedCand.enableSoftwarePipelining = true;
        seedCand.pipelineDepth = std::max<int64_t>(2, seedCand.pipelineDepth);
        seedCand.workgroupMultiBufferDepth =
            std::max<int64_t>(2, seedCand.workgroupMultiBufferDepth);
        seedCand.pipelineSetAsyncWaitGroups = true;
        baseWantsWaitGroups = true;
      }
      bool targetedRecovered = false;
      for (int64_t rowThreadsX : rowThreadsOrder) {
        Candidate safeRowCand = buildSafeRowCandidate(
            seedCand, rowThreadsX, /*forceWaitGroups=*/baseWantsWaitGroups,
            /* dropRowReuse=*/false);
        const std::string variant = "safe_row_tx" + std::to_string(rowThreadsX) +
                                    (baseWantsWaitGroups ? "_wait" : "");
        if (tryTargetedProfileAtIdx(safeRowCand, variant)) {
          targetedRecovered = true;
          break;
        }
      }
      if (targetedRecovered)
        continue;
      if (baseWantsWaitGroups) {
        for (int64_t rowThreadsX : rowThreadsOrder) {
          Candidate noWaitCand = buildSafeRowCandidate(
              seedCand, rowThreadsX, /*forceWaitGroups=*/false,
              /* dropRowReuse=*/false);
          const std::string variant =
              "safe_row_tx" + std::to_string(rowThreadsX) + "_wait_off";
          if (tryTargetedProfileAtIdx(noWaitCand, variant)) {
            targetedRecovered = true;
            break;
          }
        }
      }
      if (targetedRecovered)
        continue;
      for (int64_t rowThreadsX : rowThreadsOrder) {
        Candidate dropRowReuseCand = buildSafeRowCandidate(
            seedCand, rowThreadsX, /*forceWaitGroups=*/baseWantsWaitGroups,
            /* dropRowReuse=*/true);
        const std::string variant =
            "drop_row_reuse_tx" + std::to_string(rowThreadsX) +
            (baseWantsWaitGroups ? "_wait" : "");
        if (tryTargetedProfileAtIdx(dropRowReuseCand, variant)) {
          targetedRecovered = true;
          break;
        }
      }
      if (!targetedRecovered && baseWantsWaitGroups) {
        Candidate dropRowReuseNoWaitCand = buildSafeRowCandidate(
            seedCand, rowThreadsOrder.empty() ? 8 : rowThreadsOrder.front(),
            /* forceWaitGroups=*/false, /*dropRowReuse=*/true);
        (void)tryTargetedProfileAtIdx(dropRowReuseNoWaitCand,
                                      "drop_row_reuse_wait_off");
      }
    }
  }

  const bool preferTcAsyncWaitForMatmulSoftmaxFinalSort =
      preferTensorCoreForMatmulSoftmaxFinalSort && opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true) &&
      llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
  const double tcPreferRatioForFinalSort = std::max(
      1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_PREFER_RATIO",
                                 /*default=*/1.15));
  const double tcAsyncWaitPreferRatioForFinalSort = std::max(
      1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_ASYNC_WAIT_PREFER_RATIO",
                                 /*default=*/1.20));
  auto isTcAsyncWaitForFinalSort = [](const Candidate &c) {
    return c.enableTensorCoreF16 && c.enableAsyncCopy &&
           c.enableSoftwarePipelining && c.pipelineSetAsyncWaitGroups;
  };

  llvm::sort(out, [&](const PaperScheduleCandidate &a,
                      const PaperScheduleCandidate &b) {
    double al = getPaperCandidateSortLatencyProfileFirst(a);
    double bl = getPaperCandidateSortLatencyProfileFirst(b);
    int latCmp = compareProfilePriorityLatency(a.cand, al, b.cand, bl);
    if (latCmp != 0)
      return latCmp < 0;
    bool aTc = a.cand.enableTensorCoreF16 || a.cand.enableTensorCoreTf32;
    bool bTc = b.cand.enableTensorCoreF16 || b.cand.enableTensorCoreTf32;
    bool aTcAsyncWait = isTcAsyncWaitForFinalSort(a.cand);
    bool bTcAsyncWait = isTcAsyncWaitForFinalSort(b.cand);
    if (preferTcAsyncWaitForMatmulSoftmaxFinalSort &&
        aTcAsyncWait != bTcAsyncWait && std::isfinite(al) &&
        std::isfinite(bl)) {
      if (aTcAsyncWait && al <= bl * tcAsyncWaitPreferRatioForFinalSort)
        return true;
      if (bTcAsyncWait && bl <= al * tcAsyncWaitPreferRatioForFinalSort)
        return false;
    }
    if (preferTensorCoreForMatmulSoftmaxFinalSort && aTc != bTc &&
        std::isfinite(al) && std::isfinite(bl)) {
      if (aTc && al <= bl * tcPreferRatioForFinalSort)
        return true;
      if (bTc && bl <= al * tcPreferRatioForFinalSort)
        return false;
    }
    if (a.cand.blocksPerSM != b.cand.blocksPerSM)
      return a.cand.blocksPerSM > b.cand.blocksPerSM;
    if (a.cand.estRegsPerThread != b.cand.estRegsPerThread)
      return a.cand.estRegsPerThread < b.cand.estRegsPerThread;
    if (aTc != bTc)
      return aTc;
    if (a.sharedFootprintBytes != b.sharedFootprintBytes)
      return a.sharedFootprintBytes < b.sharedFootprintBytes;
    return a.cand.score < b.cand.score;
  });
	  const bool wantTensorCoreF16ForFinalOut =
	      (profileEnabledForSubgraph && opts.profile.enableTensorCoreF16) ||
	      (opts.codegenSearch.enable &&
	       llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true));
  int64_t finalDesiredOut = desiredOut;
  const bool forceMinOutForF16MatmulSoftmax =
      matmulSoftmaxLikeContextForFinalSort &&
      wantTensorCoreF16ForFinalOut;
  if (forceMinOutForF16MatmulSoftmax) {
    const int64_t minFinalOutForF16MatmulSoftmax = std::max<int64_t>(
	        1, getEnvInt64OrDefault("WELDER_MM_SM_F16_MIN_FINAL_OUTPUT",
	                                profileEnabledForSubgraph ? 8 : 6));
    if (finalDesiredOut < minFinalOutForF16MatmulSoftmax) {
      finalDesiredOut = minFinalOutForF16MatmulSoftmax;
      if (opts.tracer) {
        llvm::json::Object f;
        f["requested"] = desiredOut;
        f["raised_to"] = finalDesiredOut;
        f["min_required"] = minFinalOutForF16MatmulSoftmax;
        opts.tracer->event("paper.mm_sm_f16_min_final_output", std::move(f),
                           /* isVerbose=*/true);
      }
    }
  }
  if (retainTensorCoreClassForFinalSort &&
      static_cast<int64_t>(out.size()) > finalDesiredOut && finalDesiredOut > 0) {
    auto isTensorCoreCand = [](const PaperScheduleCandidate &pc) {
      return pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32;
    };
    bool hasTensorCoreInTop = false;
    for (int64_t i = 0;
         i < finalDesiredOut && i < static_cast<int64_t>(out.size());
         ++i) {
      if (isTensorCoreCand(out[static_cast<size_t>(i)])) {
        hasTensorCoreInTop = true;
        break;
      }
    }
    if (!hasTensorCoreInTop) {
      int64_t tcTailIdx = -1;
      for (int64_t i = finalDesiredOut; i < static_cast<int64_t>(out.size());
           ++i) {
        if (isTensorCoreCand(out[static_cast<size_t>(i)])) {
          tcTailIdx = i;
          break;
        }
      }
      if (tcTailIdx >= 0) {
        std::swap(out[static_cast<size_t>(finalDesiredOut - 1)],
                  out[static_cast<size_t>(tcTailIdx)]);
      }
    }
    if (finalDesiredOut > 1) {
      bool hasNonTensorCoreInTop = false;
      for (int64_t i = 0;
           i < finalDesiredOut && i < static_cast<int64_t>(out.size()); ++i) {
        if (!isTensorCoreCand(out[static_cast<size_t>(i)])) {
          hasNonTensorCoreInTop = true;
          break;
        }
      }
      if (!hasNonTensorCoreInTop) {
        int64_t nonTcTailIdx = -1;
        for (int64_t i = finalDesiredOut;
             i < static_cast<int64_t>(out.size()); ++i) {
          if (!isTensorCoreCand(out[static_cast<size_t>(i)])) {
            nonTcTailIdx = i;
            break;
          }
        }
        if (nonTcTailIdx >= 0) {
          std::swap(out[static_cast<size_t>(finalDesiredOut - 1)],
                    out[static_cast<size_t>(nonTcTailIdx)]);
        }
      }
    }
  }
  const bool wantAsyncClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableAsyncCopy, true);
  const bool wantPipeClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableSoftwarePipelining, true);
  const bool wantWaitGroupClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.pipelineSetAsyncWaitGroups, true);
  const bool wantTensorCoreF16ClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true);
  const bool wantRowReuseClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionChainReuseFusion,
                         true);
  const bool wantRowPromoClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionInputPromotion,
                         true);
  const bool wantRowPromoVecClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(
          opts.codegenSearch.enableRowReductionInputPromotionVectorize, true);
  const bool wantRowWarpClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionWarp, true);
  const bool wantRowVecClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionVectorize, true);
  const bool wantRowCombineVecClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionCombineVectorize,
                         true);
  const bool wantRowRelaxBarrierClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionRelaxBarriers,
                         true);
  const bool wantRowSkipCombineBarrierClassForFinalSort =
      opts.codegenSearch.enable &&
      llvm::is_contained(opts.codegenSearch.enableRowReductionSkipCombineBarrier,
                         true);
  int64_t minRowThreadsXInOut = std::numeric_limits<int64_t>::max();
  int64_t maxRowThreadsXInOut = 0;
  if (opts.codegenSearch.enable &&
      (getEnvInt64OrDefault("WELDER_PROFILE_ENFORCE_ROW_THREADS_X_CLASS", 1) !=
       0)) {
    for (const PaperScheduleCandidate &pc : out) {
      int64_t tx = pc.cand.rowReductionThreadsX;
      if (tx <= 0)
        continue;
      minRowThreadsXInOut = std::min(minRowThreadsXInOut, tx);
      maxRowThreadsXInOut = std::max(maxRowThreadsXInOut, tx);
    }
  }
  const bool wantRowThreadsXClassForFinalSort =
      minRowThreadsXInOut < std::numeric_limits<int64_t>::max() &&
      maxRowThreadsXInOut > minRowThreadsXInOut;
  const bool keepExtraCodegenClassForFinalSort =
      static_cast<int64_t>(out.size()) > finalDesiredOut && finalDesiredOut > 1;
  if (keepExtraCodegenClassForFinalSort &&
      (wantAsyncClassForFinalSort || wantPipeClassForFinalSort ||
       wantWaitGroupClassForFinalSort || wantRowReuseClassForFinalSort ||
       wantRowPromoClassForFinalSort ||
       wantRowPromoVecClassForFinalSort || wantRowWarpClassForFinalSort ||
       wantRowVecClassForFinalSort || wantRowCombineVecClassForFinalSort ||
       wantRowRelaxBarrierClassForFinalSort ||
       wantRowThreadsXClassForFinalSort ||
       wantRowSkipCombineBarrierClassForFinalSort)) {
    int64_t replacePos = finalDesiredOut - 1;
    auto ensureClassInTopK = [&](auto pred) {
      bool hasInTop = false;
      for (int64_t i = 0;
           i < finalDesiredOut && i < static_cast<int64_t>(out.size()); ++i) {
        if (pred(out[static_cast<size_t>(i)])) {
          hasInTop = true;
          break;
        }
      }
      if (hasInTop)
        return;

      int64_t tailIdx = -1;
      for (int64_t i = finalDesiredOut; i < static_cast<int64_t>(out.size());
           ++i) {
        if (pred(out[static_cast<size_t>(i)])) {
          tailIdx = i;
          break;
        }
      }
      if (tailIdx < 0)
        return;

      while (replacePos >= 0 &&
             pred(out[static_cast<size_t>(replacePos)])) {
        --replacePos;
      }
      if (replacePos < 0)
        return;
      std::swap(out[static_cast<size_t>(replacePos)],
                out[static_cast<size_t>(tailIdx)]);
      --replacePos;
    };

    if (wantWaitGroupClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableSoftwarePipelining &&
               pc.cand.pipelineSetAsyncWaitGroups;
      });
    }
    if (wantRowReuseClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionChainReuseFusion;
      });
    }
    if (wantRowReuseClassForFinalSort && wantAsyncClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableAsyncCopy &&
               pc.cand.enableRowReductionChainReuseFusion;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort && wantWaitGroupClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16 && pc.cand.enableAsyncCopy &&
               pc.cand.enableSoftwarePipelining &&
               pc.cand.pipelineSetAsyncWaitGroups;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort && wantRowReuseClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16 &&
               pc.cand.enableRowReductionChainReuseFusion;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort && wantAsyncClassForFinalSort &&
        wantRowReuseClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16 && pc.cand.enableAsyncCopy &&
               pc.cand.enableRowReductionChainReuseFusion;
      });
    }
    if (wantPipeClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableSoftwarePipelining;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort && wantPipeClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16 && pc.cand.enableAsyncCopy &&
               pc.cand.enableSoftwarePipelining;
      });
    }
    if (wantAsyncClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableAsyncCopy;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort && wantAsyncClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16 && pc.cand.enableAsyncCopy;
      });
    }
    if (wantTensorCoreF16ClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableTensorCoreF16;
      });
    }
    if (wantRowPromoClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionInputPromotion;
      });
    }
    if (wantRowPromoVecClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionInputPromotionVectorize;
      });
    }
    if (wantRowWarpClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionWarp;
      });
    }
    if (wantRowVecClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionVectorize;
      });
    }
    if (wantRowCombineVecClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionCombineVectorize;
      });
    }
    if (wantRowRelaxBarrierClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionRelaxBarriers;
      });
    }
    if (wantRowSkipCombineBarrierClassForFinalSort) {
      ensureClassInTopK([](const PaperScheduleCandidate &pc) {
        return pc.cand.enableRowReductionSkipCombineBarrier;
      });
    }
    if (wantRowThreadsXClassForFinalSort) {
      ensureClassInTopK([&](const PaperScheduleCandidate &pc) {
        return pc.cand.rowReductionThreadsX == minRowThreadsXInOut;
      });
      ensureClassInTopK([&](const PaperScheduleCandidate &pc) {
        return pc.cand.rowReductionThreadsX == maxRowThreadsXInOut;
      });
    }
  }
  if (retainTensorCoreClassForFinalSort && finalDesiredOut > 1 && !out.empty()) {
    auto isTensorCoreCandFinal = [](const PaperScheduleCandidate &pc) {
      return pc.cand.enableTensorCoreF16 || pc.cand.enableTensorCoreTf32;
    };
    const int64_t topN =
        std::min<int64_t>(finalDesiredOut, static_cast<int64_t>(out.size()));
    bool hasNonTensorCoreInTop = false;
    for (int64_t i = 0; i < topN; ++i) {
      if (!isTensorCoreCandFinal(out[static_cast<size_t>(i)])) {
        hasNonTensorCoreInTop = true;
        break;
      }
    }
    if (!hasNonTensorCoreInTop) {
      int64_t nonTcTailIdx = -1;
      for (int64_t i = topN; i < static_cast<int64_t>(out.size()); ++i) {
        if (!isTensorCoreCandFinal(out[static_cast<size_t>(i)])) {
          nonTcTailIdx = i;
          break;
        }
      }
      if (nonTcTailIdx >= 0) {
        std::swap(out[static_cast<size_t>(topN - 1)],
                  out[static_cast<size_t>(nonTcTailIdx)]);
      } else if (getEnvInt64OrDefault(
                     "WELDER_PROFILE_SYNTHESIZE_NON_TC_FINAL_ANCHOR", 1) != 0) {
        PaperScheduleCandidate synth = out.front();
        synth.cand.enableTensorCoreF16 = false;
        synth.cand.enableTensorCoreTf32 = false;
        synth.cand.useCutlassMma = false;
        synth.cand.mmaM = 0;
        synth.cand.mmaN = 0;
        synth.cand.mmaK = 0;
        synth.cand.cost.profiledMs = std::nullopt;
        if (std::isfinite(synth.estimatedLatency))
          synth.estimatedLatency = synth.estimatedLatency * 1.25;
        else if (std::isfinite(synth.cand.score))
          synth.estimatedLatency = synth.cand.score * 1.25;
        synth.cand.score = std::isfinite(synth.estimatedLatency)
                               ? synth.estimatedLatency
                               : synth.cand.score;
        if (topN > 0) {
          out[static_cast<size_t>(topN - 1)] = std::move(synth);
        } else {
          out.push_back(std::move(synth));
        }
      }
    }
  }
  if (static_cast<int64_t>(out.size()) > finalDesiredOut)
    out.resize(finalDesiredOut);
  return out;
}

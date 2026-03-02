static bool graphConnectingPaperGlobalShared(TileGraph &graph,
                                             const SolveOptions &opts,
                                             const FootprintInference &inference,
                                             bool requirePerfectTiling) {
  // GraphConnecting：按拓扑顺序遍历每条 out edge，对每条 edge 试探 connectLevel∈[0,max]，
  // 用 SubGraphTiling 的最优 latency 估计做选择。
  //
  // 当前实现只针对 “global<->shared”：
  // - connectLevel==0：cut（落地到 global）
  // - connectLevel>0：connect（融合，复用在 shared 或更高层）

  (void)requirePerfectTiling;
  [[maybe_unused]] auto span = [&]() -> Tracer::Span {
    if (!opts.tracer)
      return Tracer::Span();
    llvm::json::Object f;
    f["nodes"] = static_cast<int64_t>(graph.nodes.size());
    f["edges"] = static_cast<int64_t>(graph.edges.size());
    f["max_connect_level"] = static_cast<int64_t>(opts.maxConnectLevel);
    f["profiling"] = opts.profile.enable;
    return opts.tracer->span("paper.graph_connecting.inner", std::move(f),
                             /* isVerbose=*/true);
  }();

  // 全图 topo order（按 SSA producer->consumer）。
  llvm::SmallVector<int, 32> topo;
  topo.reserve(graph.nodes.size());

  llvm::SmallVector<int, 64> indeg(graph.nodes.size(), 0);
  for (const TileGraphEdge &e : graph.edges) {
    if (e.src < 0 || e.dst < 0)
      continue;
    if (e.src >= static_cast<int>(graph.nodes.size()) ||
        e.dst >= static_cast<int>(graph.nodes.size()))
      continue;
    indeg[e.dst] += 1;
  }
  llvm::SmallVector<int, 64> q;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    if (indeg[i] == 0)
      q.push_back(i);
  }
  while (!q.empty()) {
    int n = q.pop_back_val();
    topo.push_back(n);
    for (int edgeIdx : graph.nodes[n].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.dst < 0 || e.dst >= static_cast<int>(graph.nodes.size()))
        continue;
      if (--indeg[e.dst] == 0)
        q.push_back(e.dst);
    }
  }
  if (topo.size() != graph.nodes.size()) {
    topo.clear();
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i)
      topo.push_back(i);
  }

  // 初始化：默认 lowest（global），与论文“lowest level by default”一致。
  for (auto &e : graph.edges)
    e.connectLevel = 0;
  syncCutFlagFromConnectLevel(graph);

  // 论文对齐的性能测量语义：GraphConnecting 比较的是 SubGraphTiling 产出的
  // 端到端调度结果。本仓库的性能测量是按 kernel 进行，因此为了近似完整调度
  //（边被切断时会变成多 kernel），这里会评估所有 level-0 连通分量，
  // 并累加各自最佳时延。
  //
  // 这样可避免 profiler 仅测到单个抽取子图时，
  // 出现“总是切边=op 更少”这种偏置。
  struct SubgraphEvalKey {
    int sinkNodeIdx = -1;
    uint64_t nodesHash = 0;
    uint64_t edgesHash = 0;
    bool profiling = false;
    bool profileRunAll = false;
    bool paperStrict = false;
    bool paperExpandReductionTile = false;
    bool autoCandidates = false;
    bool registerLevel = false;
    bool codegenSearch = false;
    bool enableCoalescingPenalty = false;
    bool requirePerfectTiling = false;
    int64_t scheduleTopK = 0;
    int64_t maxConnectLevel = 0;
    bool profileAsyncCopy = false;
    bool profileSoftwarePipelining = false;
    int64_t profilePipelineDepth = 0;
    bool profilePipelinePeelEpilogue = false;
    bool profilePipelineWaitGroups = false;
    bool profileAsyncBypassL1 = false;
    int64_t profileMultiBufferDepth = 0;
    int64_t profilePadLastDim = 0;
    bool profilePadMatmulOnly = false;
    int64_t profileSwizzleXor = 0;
    bool profileSwapBlockDims = false;
    bool profileEnableTensorCoreF16 = false;
    int profileBlockRasterizeMode = 0;
    int profileBlockRasterizePanelWidth = 0;
    bool profileEnableRowReductionChainReuse = false;
    bool profileEnableRowReductionInputPromotion = false;
    bool profileEnableRowReductionInputPromotionVec = false;
    bool profileEnableRowReductionWarp = false;
    bool profileEnableRowReductionVectorize = false;
    int64_t profileRowReductionVectorWidth = 0;
    int64_t profileRowReductionThreadsX = 0;
    bool profileEnableRowReductionRelaxBarriers = false;
    bool profileEnableRowReductionSkipCombineBarrier = false;
    int64_t profileRowReductionInputVectorWidth = 0;
    bool profileEnableRowReductionCombineVectorize = false;
    bool profileEnableMatmulSoftmaxSharedReuseFusion = false;

    bool operator==(const SubgraphEvalKey &o) const {
      return sinkNodeIdx == o.sinkNodeIdx && nodesHash == o.nodesHash &&
             edgesHash == o.edgesHash && profiling == o.profiling &&
             profileRunAll == o.profileRunAll &&
             paperStrict == o.paperStrict &&
             paperExpandReductionTile == o.paperExpandReductionTile &&
             autoCandidates == o.autoCandidates && registerLevel == o.registerLevel &&
             codegenSearch == o.codegenSearch &&
             enableCoalescingPenalty == o.enableCoalescingPenalty &&
             requirePerfectTiling == o.requirePerfectTiling &&
             scheduleTopK == o.scheduleTopK && maxConnectLevel == o.maxConnectLevel &&
             profileAsyncCopy == o.profileAsyncCopy &&
             profileSoftwarePipelining == o.profileSoftwarePipelining &&
             profilePipelineDepth == o.profilePipelineDepth &&
             profilePipelinePeelEpilogue == o.profilePipelinePeelEpilogue &&
             profilePipelineWaitGroups == o.profilePipelineWaitGroups &&
             profileAsyncBypassL1 == o.profileAsyncBypassL1 &&
             profileMultiBufferDepth == o.profileMultiBufferDepth &&
             profilePadLastDim == o.profilePadLastDim &&
             profilePadMatmulOnly == o.profilePadMatmulOnly &&
             profileSwizzleXor == o.profileSwizzleXor &&
             profileSwapBlockDims == o.profileSwapBlockDims &&
             profileEnableTensorCoreF16 == o.profileEnableTensorCoreF16 &&
             profileBlockRasterizeMode == o.profileBlockRasterizeMode &&
             profileBlockRasterizePanelWidth == o.profileBlockRasterizePanelWidth &&
             profileEnableRowReductionChainReuse ==
                 o.profileEnableRowReductionChainReuse &&
             profileEnableRowReductionInputPromotion ==
                 o.profileEnableRowReductionInputPromotion &&
             profileEnableRowReductionInputPromotionVec ==
                 o.profileEnableRowReductionInputPromotionVec &&
             profileEnableRowReductionWarp == o.profileEnableRowReductionWarp &&
             profileEnableRowReductionVectorize ==
                 o.profileEnableRowReductionVectorize &&
             profileRowReductionVectorWidth ==
                 o.profileRowReductionVectorWidth &&
             profileRowReductionThreadsX ==
                 o.profileRowReductionThreadsX &&
             profileEnableRowReductionRelaxBarriers ==
                 o.profileEnableRowReductionRelaxBarriers &&
             profileEnableRowReductionSkipCombineBarrier ==
                 o.profileEnableRowReductionSkipCombineBarrier &&
             profileRowReductionInputVectorWidth ==
                 o.profileRowReductionInputVectorWidth &&
             profileEnableRowReductionCombineVectorize ==
                 o.profileEnableRowReductionCombineVectorize &&
             profileEnableMatmulSoftmaxSharedReuseFusion ==
                 o.profileEnableMatmulSoftmaxSharedReuseFusion;
    }
  };
  struct SubgraphEvalKeyHash {
    size_t operator()(const SubgraphEvalKey &k) const noexcept {
      auto mix = [](uint64_t h, uint64_t x) -> uint64_t {
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
      };
      uint64_t h = 0;
      h = mix(h, static_cast<uint64_t>(k.sinkNodeIdx));
      h = mix(h, k.nodesHash);
      h = mix(h, k.edgesHash);
      h = mix(h, static_cast<uint64_t>(k.profiling));
      h = mix(h, static_cast<uint64_t>(k.profileRunAll));
      h = mix(h, static_cast<uint64_t>(k.paperStrict));
      h = mix(h, static_cast<uint64_t>(k.paperExpandReductionTile));
      h = mix(h, static_cast<uint64_t>(k.autoCandidates));
      h = mix(h, static_cast<uint64_t>(k.registerLevel));
      h = mix(h, static_cast<uint64_t>(k.codegenSearch));
      h = mix(h, static_cast<uint64_t>(k.enableCoalescingPenalty));
      h = mix(h, static_cast<uint64_t>(k.requirePerfectTiling));
      h = mix(h, static_cast<uint64_t>(k.scheduleTopK));
      h = mix(h, static_cast<uint64_t>(k.maxConnectLevel));
      h = mix(h, static_cast<uint64_t>(k.profileAsyncCopy));
      h = mix(h, static_cast<uint64_t>(k.profileSoftwarePipelining));
      h = mix(h, static_cast<uint64_t>(k.profilePipelineDepth));
      h = mix(h, static_cast<uint64_t>(k.profilePipelinePeelEpilogue));
      h = mix(h, static_cast<uint64_t>(k.profilePipelineWaitGroups));
      h = mix(h, static_cast<uint64_t>(k.profileAsyncBypassL1));
      h = mix(h, static_cast<uint64_t>(k.profileMultiBufferDepth));
      h = mix(h, static_cast<uint64_t>(k.profilePadLastDim));
      h = mix(h, static_cast<uint64_t>(k.profilePadMatmulOnly));
      h = mix(h, static_cast<uint64_t>(k.profileSwizzleXor));
      h = mix(h, static_cast<uint64_t>(k.profileSwapBlockDims));
      h = mix(h, static_cast<uint64_t>(k.profileEnableTensorCoreF16));
      h = mix(h, static_cast<uint64_t>(k.profileBlockRasterizeMode));
      h = mix(h, static_cast<uint64_t>(k.profileBlockRasterizePanelWidth));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionChainReuse));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionInputPromotion));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionInputPromotionVec));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionWarp));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionVectorize));
      h = mix(h, static_cast<uint64_t>(k.profileRowReductionVectorWidth));
      h = mix(h, static_cast<uint64_t>(k.profileRowReductionThreadsX));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionRelaxBarriers));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionSkipCombineBarrier));
      h = mix(h, static_cast<uint64_t>(k.profileRowReductionInputVectorWidth));
      h = mix(h, static_cast<uint64_t>(k.profileEnableRowReductionCombineVectorize));
      h = mix(h, static_cast<uint64_t>(k.profileEnableMatmulSoftmaxSharedReuseFusion));
      return static_cast<size_t>(h);
    }
  };

  struct EvalMetrics {
    double latency = std::numeric_limits<double>::infinity();
    double bytesCut = std::numeric_limits<double>::infinity();
  };

  std::unordered_map<SubgraphEvalKey, std::optional<EvalMetrics>, SubgraphEvalKeyHash>
      evalCache;
  evalCache.reserve(512);

  PaperSubgraph allNodes;
  allNodes.nodes.reserve(graph.nodes.size());
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    allNodes.nodes.push_back(i);
    allNodes.inSet.insert(i);
  }

  auto evalComponentBest =
      [&](const TileGraph &gTrial, const PaperSubgraph &sg,
          int sinkNodeIdx) -> EvalMetrics {
    auto sinkOp = dyn_cast_or_null<linalg::LinalgOp>(gTrial.nodes[sinkNodeIdx].op);
    if (!sinkOp)
      return EvalMetrics();

    auto mix = [](uint64_t h, uint64_t x) -> uint64_t {
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    };
    uint64_t nodesHash = 0;
    for (int n : sg.nodes)
      nodesHash = mix(nodesHash, static_cast<uint64_t>(n));

    llvm::SmallDenseSet<int, 64> incidentEdges;
    for (int n : sg.nodes) {
      if (n < 0 || n >= static_cast<int>(gTrial.nodes.size()))
        continue;
      for (int e : gTrial.nodes[n].inEdges)
        incidentEdges.insert(e);
      for (int e : gTrial.nodes[n].outEdges)
        incidentEdges.insert(e);
    }
    llvm::SmallVector<int, 64> edgesVec;
    edgesVec.reserve(incidentEdges.size());
    for (int e : incidentEdges)
      edgesVec.push_back(e);
    llvm::sort(edgesVec);

    uint64_t edgesHash = 0;
    for (int edgeIdx : edgesVec) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(gTrial.edges.size()))
        continue;
      const TileGraphEdge &e = gTrial.edges[edgeIdx];
      edgesHash = mix(edgesHash, static_cast<uint64_t>(edgeIdx));
      edgesHash = mix(edgesHash, static_cast<uint64_t>(e.connectLevel));
      edgesHash = mix(edgesHash, static_cast<uint64_t>(e.src));
      edgesHash = mix(edgesHash, static_cast<uint64_t>(e.dst));
      edgesHash = mix(edgesHash, static_cast<uint64_t>(e.dstOperand));
    }

    SubgraphEvalKey key;
    key.sinkNodeIdx = sinkNodeIdx;
    key.nodesHash = nodesHash;
    key.edgesHash = edgesHash;
    key.profiling = opts.profile.enable;
    key.profileRunAll = opts.profile.runAllKernels;
    key.paperStrict = opts.paperStrict;
    key.paperExpandReductionTile = opts.paperExpandReductionTile;
    key.autoCandidates = opts.autoCandidates;
    key.registerLevel = opts.enableRegisterLevelSchedule;
    key.codegenSearch = opts.codegenSearch.enable;
    key.enableCoalescingPenalty = opts.enableCoalescingPenalty;
    key.requirePerfectTiling = opts.requirePerfectTiling;
    key.scheduleTopK = opts.scheduleTopK;
    key.maxConnectLevel = opts.maxConnectLevel;
    key.profileAsyncCopy = opts.profile.enableAsyncCopy;
    key.profileSoftwarePipelining = opts.profile.enableSoftwarePipelining;
    key.profilePipelineDepth = opts.profile.pipelineDepth;
    key.profilePipelinePeelEpilogue = opts.profile.pipelinePeelEpilogue;
    key.profilePipelineWaitGroups = opts.profile.pipelineSetAsyncWaitGroups;
    key.profileAsyncBypassL1 = opts.profile.asyncBypassL1;
    key.profileMultiBufferDepth = opts.profile.workgroupMultiBufferDepth;
    key.profilePadLastDim = opts.profile.workgroupPadLastDim;
    key.profilePadMatmulOnly = opts.profile.workgroupPadLastDimMatmulOnly;
    key.profileSwizzleXor = opts.profile.workgroupSwizzleXor;
    key.profileSwapBlockDims = opts.profile.swapBlockDims;
    key.profileEnableTensorCoreF16 = opts.profile.enableTensorCoreF16;
    key.profileBlockRasterizeMode = opts.profile.blockRasterizeMode;
    key.profileBlockRasterizePanelWidth = opts.profile.blockRasterizePanelWidth;
    key.profileEnableRowReductionChainReuse =
        opts.profile.enableRowReductionChainReuseFusion;
    key.profileEnableRowReductionInputPromotion =
        opts.profile.enableRowReductionInputPromotion;
    key.profileEnableRowReductionInputPromotionVec =
        opts.profile.enableRowReductionInputPromotionVectorize;
    key.profileEnableRowReductionWarp = opts.profile.enableRowReductionWarp;
    key.profileEnableRowReductionVectorize =
        opts.profile.enableRowReductionVectorize;
    key.profileRowReductionVectorWidth = opts.profile.rowReductionVectorWidth;
    key.profileRowReductionThreadsX = opts.profile.rowReductionThreadsX;
    key.profileEnableRowReductionRelaxBarriers =
        opts.profile.enableRowReductionRelaxBarriers;
    key.profileEnableRowReductionSkipCombineBarrier =
        opts.profile.enableRowReductionSkipCombineBarrier;
    key.profileRowReductionInputVectorWidth =
        opts.profile.rowReductionInputVectorWidth;
    key.profileEnableRowReductionCombineVectorize =
        opts.profile.enableRowReductionCombineVectorize;
    key.profileEnableMatmulSoftmaxSharedReuseFusion =
        opts.profile.enableMatmulSoftmaxSharedReuseFusion;

    if (auto it = evalCache.find(key); it != evalCache.end()) {
      if (it->second.has_value())
        return *it->second;
      return EvalMetrics();
    }

    auto configs =
        subGraphTilingPaperGlobalShared(gTrial, sg, sinkOp, sinkNodeIdx, opts,
                                        inference);
    EvalMetrics best;
    const double candTieEps = opts.profile.enable ? 5e-3 : 1e-9;
    for (const auto &pc : configs) {
      double lat = pc.estimatedLatency;
      if (opts.profile.enable && pc.cand.cost.profiledMs.has_value())
        lat = *pc.cand.cost.profiledMs;
      if (!std::isfinite(lat))
        continue;
      double cut = pc.traffic.bytesCut;
      if ((lat + candTieEps) < best.latency ||
          (std::abs(lat - best.latency) <= candTieEps && cut < best.bytesCut)) {
        best.latency = lat;
        best.bytesCut = cut;
      }
    }

    if (std::isfinite(best.latency))
      evalCache.emplace(key, best);
    else
      evalCache.emplace(key, std::nullopt);
    return best;
  };

  auto evalWholeGraph = [&](const TileGraph &gTrial) -> EvalMetrics {
    EvalMetrics total;
    total.latency = 0.0;
    total.bytesCut = 0.0;
    const int n = static_cast<int>(gTrial.nodes.size());
    llvm::SmallVector<char, 64> visited;
    visited.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      if (visited[static_cast<size_t>(i)])
        continue;
      PaperSubgraph comp = extractSubgraphByConnectLevel(gTrial, i,
                                                        /* minLevelExclusive=*/0);
      for (int n0 : comp.nodes) {
        if (n0 >= 0 && n0 < n)
          visited[static_cast<size_t>(n0)] = 1;
      }

      auto sinkOpt =
          pickConnectedSinkInSubgraph(gTrial, comp, /*minLevelExclusive=*/0);
      if (!sinkOpt)
        continue;
      int sinkNodeIdx = *sinkOpt;

      EvalMetrics best = evalComponentBest(gTrial, comp, sinkNodeIdx);
      if (!std::isfinite(best.latency))
        return EvalMetrics();
      total.latency += best.latency;
      total.bytesCut += best.bytesCut;
    }
    return total;
  };

  const bool shouldForceMatmulSoftmaxReuse =
      opts.maxConnectLevel >= 1 &&
      (opts.profile.enableMatmulSoftmaxSharedReuseFusion ||
       (opts.codegenSearch.enable &&
        llvm::is_contained(
            opts.codegenSearch.enableMatmulSoftmaxSharedReuseFusion, true)));
  int64_t forceMinLevel = 1;
  int64_t forceElemwiseMinLevel = forceMinLevel;
  llvm::DenseMap<int, int64_t> forcedMmSmEdgeMinLevel;
  if (shouldForceMatmulSoftmaxReuse) {
    if (opts.maxConnectLevel >= kConnectLevelRegister &&
        opts.enableRegisterLevelSchedule) {
      forceElemwiseMinLevel = kConnectLevelRegister;
      if (opts.maxConnectLevel > kConnectLevelRegister &&
          opts.paperRecursiveRegisterLevel) {
        int recursiveInnerMinLevelExclusive =
            opts.paperRecursiveInnerMinLevelExclusive;
        const int recursiveMaxStages =
            opts.paperRecursiveMaxStages > 0
                ? std::max(1, opts.paperRecursiveMaxStages)
                : 0;
        if (recursiveInnerMinLevelExclusive <= kConnectLevelGlobal) {
          if (recursiveMaxStages > 0) {
            recursiveInnerMinLevelExclusive =
                std::max(1, opts.maxConnectLevel - recursiveMaxStages);
          } else {
            recursiveInnerMinLevelExclusive =
                std::max<int>(kConnectLevelShared, opts.maxConnectLevel - 1);
          }
        }
        if (recursiveMaxStages > 0) {
          const int minBoundaryForStageCap =
              std::max(1, opts.maxConnectLevel - recursiveMaxStages);
          recursiveInnerMinLevelExclusive =
              std::max(recursiveInnerMinLevelExclusive, minBoundaryForStageCap);
        }
        recursiveInnerMinLevelExclusive = std::max(
            kConnectLevelShared,
            std::min(recursiveInnerMinLevelExclusive, opts.maxConnectLevel - 1));
        forceElemwiseMinLevel = std::max<int64_t>(
            forceElemwiseMinLevel, recursiveInnerMinLevelExclusive + 1);
      }
    }
    forceMinLevel = std::max<int64_t>(
        1, std::min<int64_t>(forceMinLevel, opts.maxConnectLevel));
    forceElemwiseMinLevel = std::max<int64_t>(
        forceMinLevel,
        std::min<int64_t>(forceElemwiseMinLevel, opts.maxConnectLevel));
    const int64_t envForceMinLevel =
        getEnvInt64OrDefault("WELDER_MM_SM_FORCE_CONNECT_MIN_LEVEL", 0);
    if (envForceMinLevel > 0) {
      forceMinLevel = std::max<int64_t>(
          1, std::min<int64_t>(envForceMinLevel, opts.maxConnectLevel));
      forceElemwiseMinLevel = std::max<int64_t>(forceMinLevel,
                                                forceElemwiseMinLevel);
    }
    const int64_t envForceElemwiseMinLevel = getEnvInt64OrDefault(
        "WELDER_MM_SM_FORCE_ELEMWISE_CONNECT_MIN_LEVEL", 0);
    if (envForceElemwiseMinLevel > 0) {
      forceElemwiseMinLevel = std::max<int64_t>(
          forceMinLevel,
          std::min<int64_t>(envForceElemwiseMinLevel, opts.maxConnectLevel));
    }
    forcedMmSmEdgeMinLevel = collectMatmulSoftmaxChainMinConnectLevels(
        graph, forceMinLevel, forceElemwiseMinLevel);
    if (opts.tracer && !forcedMmSmEdgeMinLevel.empty()) {
      llvm::json::Object f;
      f["max_connect_level"] = static_cast<int64_t>(opts.maxConnectLevel);
      f["force_min_level"] = forceMinLevel;
      f["force_elemwise_min_level"] = forceElemwiseMinLevel;
      f["target_edges"] =
          static_cast<int64_t>(forcedMmSmEdgeMinLevel.size());
      opts.tracer->event("paper.graph_connecting.force_targets", std::move(f),
                         /* isVerbose=*/true);
    }
  }

  // 论文对齐：按“从 sink 到 source”的顺序处理边，
  // 让每次边决策都尽量看到更多下游调度上下文。
  // 这也能提高基于性能测量评估的鲁棒性：部分连通图抽出的 kernel
  // 否则可能缺失必要 producer 链而导致编译失败。
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    int nodeIdx = *it;
    for (int edgeIdx : graph.nodes[nodeIdx].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      TileGraphEdge &edge = graph.edges[edgeIdx];
      if (edge.src < 0 || edge.dst < 0)
        continue;

      // 尝试不同 connect level。
      EvalMetrics best;
      int bestLevel = 0;

      int maxLevel = std::max<int>(0, opts.maxConnectLevel);
      int minTrialLevel = 0;
      if (!forcedMmSmEdgeMinLevel.empty()) {
        if (auto it = forcedMmSmEdgeMinLevel.find(edgeIdx);
            it != forcedMmSmEdgeMinLevel.end()) {
          minTrialLevel = std::max(
              0, std::min(maxLevel, static_cast<int>(it->second)));
        }
      }
      for (int trial = minTrialLevel; trial <= maxLevel; ++trial) {
        int saved = edge.connectLevel;
        edge.connectLevel = trial;
        syncCutFlagFromConnectLevel(graph);

        // Evaluate on a temporary graph so we can apply additional 论文对齐
        // 约束 (e.g. block-order conflicts => cut) without mutating the
        // 以免试探过程污染全局 GraphConnecting 状态。
        TileGraph tmp = graph;
        // 将 block-order 冲突按切边处理（鲁棒变体）。
        cutEdgesOnSwapXYConflict(tmp, allNodes, /*minLevelExclusive=*/0);
        syncCutFlagFromConnectLevel(tmp);

        EvalMetrics trialEv = evalWholeGraph(tmp);

        // 目标相同（或近似相同）时优先更高 connect level，
        // 让调度更偏向 kernel 内复用（符合论文意图），并避免在已融合连通分量内
        // 残留冗余切边。
        //
        // 低迭代数下 profiling 噪声可能较大。启用性能测量时，
        // 将小差异视为平局，再用 bytesCut 作为确定性的二级决策器，
        // 向更高复用/connect level 倾斜。
        const double tieEps = opts.profile.enable ? 5e-3 : 1e-9;
        const double bytesTieEps = 1e-6;
        if ((trialEv.latency + tieEps) < best.latency ||
            (std::abs(trialEv.latency - best.latency) <= tieEps &&
             ((trialEv.bytesCut + bytesTieEps) < best.bytesCut ||
              (std::abs(trialEv.bytesCut - best.bytesCut) <= bytesTieEps &&
               trial > bestLevel)))) {
          best = trialEv;
          bestLevel = trial;
        }

        edge.connectLevel = saved;
        syncCutFlagFromConnectLevel(graph);
      }

      edge.connectLevel = bestLevel;
      syncCutFlagFromConnectLevel(graph);
      if (opts.tracer) {
        llvm::json::Object f;
        f["edge"] = static_cast<int64_t>(edgeIdx);
        f["src"] = static_cast<int64_t>(edge.src);
        f["dst"] = static_cast<int64_t>(edge.dst);
        f["best_level"] = static_cast<int64_t>(bestLevel);
        if (std::isfinite(best.latency))
          f["best_latency"] = best.latency;
        if (std::isfinite(best.bytesCut))
          f["best_bytes_cut"] = best.bytesCut;
        opts.tracer->event("paper.graph_connecting.edge", std::move(f),
                           /* isVerbose=*/true);
      }
      // 论文对齐 robustness: enforce block-order conflicts as permanent cuts
      // 在全局 connecting 结果中生效。
      {
        cutEdgesOnSwapXYConflict(graph, allNodes, /*minLevelExclusive=*/0);
        syncCutFlagFromConnectLevel(graph);
      }
    }
  }

  // 鲁棒性处理：避免在已融合的连通分量内部保留切边。
  // 在当前 MLIR 的 `codegen-from-kernel-attrs` 路径中（性能测量 harness 使用），
  // connectLevel 主要控制 kernel 边界，并不建模“融合 kernel 内部再回写 global”。
  // 因此若保留分量内切边，会扭曲流量记账，也会让调度解释变得困难
  //（例如 Matmul->Softmax 的 shared 复用场景）。
  {
    const int n = static_cast<int>(graph.nodes.size());
    llvm::SmallVector<int, 64> compId(n, -1);
    int nextId = 0;
    for (int i = 0; i < n; ++i) {
      if (compId[static_cast<size_t>(i)] != -1)
        continue;
      llvm::SmallVector<int, 16> stack;
      stack.push_back(i);
      compId[static_cast<size_t>(i)] = nextId;
      while (!stack.empty()) {
        int cur = stack.pop_back_val();
        for (int eidx : graph.nodes[cur].inEdges) {
          if (eidx < 0 || eidx >= static_cast<int>(graph.edges.size()))
            continue;
          const TileGraphEdge &e = graph.edges[eidx];
          if (e.connectLevel <= 0)
            continue;
          int nei = e.src;
          if (nei < 0 || nei >= n)
            continue;
          if (compId[static_cast<size_t>(nei)] != -1)
            continue;
          compId[static_cast<size_t>(nei)] = nextId;
          stack.push_back(nei);
        }
        for (int eidx : graph.nodes[cur].outEdges) {
          if (eidx < 0 || eidx >= static_cast<int>(graph.edges.size()))
            continue;
          const TileGraphEdge &e = graph.edges[eidx];
          if (e.connectLevel <= 0)
            continue;
          int nei = e.dst;
          if (nei < 0 || nei >= n)
            continue;
          if (compId[static_cast<size_t>(nei)] != -1)
            continue;
          compId[static_cast<size_t>(nei)] = nextId;
          stack.push_back(nei);
        }
      }
      ++nextId;
    }

    for (TileGraphEdge &e : graph.edges) {
      if (e.connectLevel != 0)
        continue;
      if (e.src < 0 || e.dst < 0)
        continue;
      if (e.src >= n || e.dst >= n)
        continue;
      int a = compId[static_cast<size_t>(e.src)];
      int b = compId[static_cast<size_t>(e.dst)];
      if (a != -1 && a == b)
        e.connectLevel = 1;
    }

    syncCutFlagFromConnectLevel(graph);
    cutEdgesOnSwapXYConflict(graph, allNodes, /*minLevelExclusive=*/0);
    syncCutFlagFromConnectLevel(graph);
  }

  if (shouldForceMatmulSoftmaxReuse) {
    int64_t forced =
        enforceMatmulSoftmaxChainConnectLevels(graph, forceMinLevel,
                                               forceElemwiseMinLevel);
    if (forced > 0) {
      syncCutFlagFromConnectLevel(graph);
      cutEdgesOnSwapXYConflict(graph, allNodes, /*minLevelExclusive=*/0);
      syncCutFlagFromConnectLevel(graph);
      if (opts.tracer) {
        llvm::json::Object f;
        f["forced_edges"] = forced;
        f["max_connect_level"] = static_cast<int64_t>(opts.maxConnectLevel);
        f["force_min_level"] = forceMinLevel;
        f["force_elemwise_min_level"] = forceElemwiseMinLevel;
        opts.tracer->event("paper.graph_connecting.force_matmul_softmax_reuse",
                           std::move(f), /*isVerbose=*/true);
      }
    }
  }

  return true;
}

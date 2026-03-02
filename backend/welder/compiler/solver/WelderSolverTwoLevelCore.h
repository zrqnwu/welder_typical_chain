static std::optional<llvm::SmallVector<int64_t, 8>>
buildRootParallelExtents2Level(linalg::LinalgOp rootOp, const Candidate &cand,
                               const SolveOptions &opts) {
  if (!rootOp)
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> ranges = rootOp.getStaticLoopRanges();
  if (static_cast<int64_t>(ranges.size()) != rootOp.getNumLoops())
    return std::nullopt;

  auto iters = rootOp.getIteratorTypesArray();
  auto validateExtent = [&](int64_t t, int64_t full) -> bool {
    if (full == ShapedType::kDynamic || full <= 0)
      return false;
    if (t <= 0 || t > full)
      return false;
    if (opts.requirePerfectTiling && (full % t != 0))
      return false;
    return true;
  };

  // 若提供了逐 loop extent，则优先使用（ND EnumerateSubtiles 路径）。
  if (!cand.loopTileExtents.empty()) {
    // 情况 1：`loopTileExtents` 已按“parallel-loop 顺序”给出。
    if (cand.loopTileExtents.size() ==
        static_cast<size_t>(rootOp.getNumParallelLoops())) {
      llvm::SmallVector<int64_t, 8> parallel;
      parallel.reserve(rootOp.getNumParallelLoops());
      int64_t pSeen = 0;
      for (int64_t i = 0; i < rootOp.getNumLoops(); ++i) {
        if (iters[i] != utils::IteratorType::parallel)
          continue;
        if (pSeen < 0 ||
            static_cast<size_t>(pSeen) >= cand.loopTileExtents.size())
          return std::nullopt;
        int64_t t = cand.loopTileExtents[static_cast<size_t>(pSeen++)];
        int64_t full = ranges[i];
        if (!validateExtent(t, full))
          return std::nullopt;
        parallel.push_back(t);
      }
      if (static_cast<int64_t>(parallel.size()) != rootOp.getNumParallelLoops())
        return std::nullopt;
      return parallel;
    }

    // 情况 2：`loopTileExtents` 按“loop 顺序”给出（长度等于 numLoops）。
    if (cand.loopTileExtents.size() ==
        static_cast<size_t>(rootOp.getNumLoops())) {
      llvm::SmallVector<int64_t, 8> parallel;
      parallel.reserve(rootOp.getNumParallelLoops());
      for (int64_t i = 0; i < rootOp.getNumLoops(); ++i) {
        if (iters[i] != utils::IteratorType::parallel)
          continue;
        int64_t t = cand.loopTileExtents[static_cast<size_t>(i)];
        int64_t full = ranges[i];
        if (!validateExtent(t, full))
          return std::nullopt;
        parallel.push_back(t);
      }
      if (static_cast<int64_t>(parallel.size()) != rootOp.getNumParallelLoops())
        return std::nullopt;
      return parallel;
    }
  }

  // 回退: map (tileM,tileN) to the first two parallel loops, keep others full.
  llvm::SmallVector<int64_t, 8> parallel;
  parallel.reserve(rootOp.getNumParallelLoops());
  int64_t pSeen = 0;
  for (int64_t i = 0; i < rootOp.getNumLoops(); ++i) {
    if (iters[i] != utils::IteratorType::parallel)
      continue;
    int64_t full = ranges[i];
    int64_t t = full;
    if (pSeen == 0)
      t = cand.tileM;
    else if (pSeen == 1)
      t = cand.tileN;
    if (!validateExtent(t, full))
      return std::nullopt;
    parallel.push_back(t);
    ++pSeen;
  }
  if (static_cast<int64_t>(parallel.size()) != rootOp.getNumParallelLoops())
    return std::nullopt;
  return parallel;
}

TilePropagationResult propagateTilesBackwardTwoLevel(
    TileGraph &graph, int rootNode, Operation *rootOpOp, const Candidate &cand,
    const SolveOptions &opts, const FootprintInference &inference,
    int64_t *outEstFootprintBytes) {
  TilePropagationResult result;

  if (outEstFootprintBytes)
    * outEstFootprintBytes = 0;

  auto rootOp = dyn_cast_or_null<linalg::LinalgOp>(rootOpOp);
  if (!rootOp) {
    result.error = "rootOp is not a linalg op";
    return result;
  }

  auto rootParallelExtentsOpt = buildRootParallelExtents2Level(rootOp, cand, opts);
  if (!rootParallelExtentsOpt) {
    result.error = "failed to build root parallel extents";
    return result;
  }

  auto rootTileOpt = buildOpTileFromParallelExtents(
      rootOp, *rootParallelExtentsOpt,
      /*defaultReductionTile=*/cand.tileK);
  if (!rootTileOpt) {
    result.error = "failed to build root OpTile";
    return result;
  }

  TilePropagationOptions popts;
  popts.defaultReductionTile = cand.tileK;
  popts.enableCutEdges = opts.enableCutEdges;
  popts.resetCutEdges = true;

  TilePropagationResult pr =
      propagateTilesBackward(graph, rootNode, *rootTileOpt, inference, popts);
  if (!pr.success)
    return pr;

  // 2-level MemFootprint 过滤/修正（shared capacity）。
  // 注意：matmul 路径已经有硬编码的 smem 模型；这里只对 “没有 smem 模型” 的候选做估算。
  if (opts.enableTwoLevelSchedule && cand.smemBytes == 0) {
    int64_t fpBytes = estimateSharedFootprintBytes2Level(graph, opts.arch);

    // 若 footprint 超限且允许 cut-edge，则尝试通过 cut-edge 把峰值压到 capacity 内。
    // 这对应论文 GraphConnecting 的“连接策略影响 shared footprint”的核心约束。
    if (fpBytes > opts.arch.smemBytes && opts.enableCutEdges) {
      popts.resetCutEdges = false; // 保留既有 cut 决策，多轮传播迭代调整。

      // 最坏情况下每轮至少 cut 1 条边，因此迭代次数 <= 边数。
      int maxIters = static_cast<int>(graph.edges.size());
      for (int iter = 0; iter < maxIters && fpBytes > opts.arch.smemBytes; ++iter) {
        auto edgeToCutOpt = pickCutEdgeForSharedFootprint2Level(graph, opts.arch);
        if (!edgeToCutOpt)
          break;
        int edgeToCut = *edgeToCutOpt;
        if (edgeToCut < 0 || edgeToCut >= static_cast<int>(graph.edges.size()))
          break;

        setEdgeConnectLevel(graph.edges[edgeToCut], kConnectLevelGlobal);

        TilePropagationResult pr2 =
            propagateTilesBackward(graph, rootNode, *rootTileOpt, inference, popts);
        if (!pr2.success)
          return pr2;

        fpBytes = estimateSharedFootprintBytes2Level(graph, opts.arch);
      }
    }

    if (outEstFootprintBytes)
      * outEstFootprintBytes = fpBytes;

    if (fpBytes > opts.arch.smemBytes) {
      TilePropagationResult fail;
      fail.error = "2-level footprint exceeds shared capacity after cuts";
      return fail;
    }
  }

  pr.success = true;
  return pr;
}

static bool applyGraphConnecting2Level(TileGraph &graph, int rootNode,
                                       linalg::LinalgOp rootOp, Candidate &cand,
                                       const SolveOptions &opts,
                                       const FootprintInference &inference) {
  int64_t fpBytes = 0;
  TilePropagationResult pr = propagateTilesBackwardTwoLevel(
      graph, rootNode, rootOp.getOperation(), cand, opts, inference, &fpBytes);
  if (!pr.success)
    return false;

  if (opts.enableTwoLevelSchedule && cand.smemBytes == 0)
    cand.estFootprintBytes = fpBytes;

  // Phase A：两层（global<->shared）下的 MemTraffic 近似（图外输入 read + sink write +
  // cut-edge 的读写）。
  if (opts.enableGlobalTraffic || opts.enableCutEdges || opts.enableTwoLevelSchedule) {
    Traffic global = computeGlobalTrafficAssumingFullyFused(
        graph, opts.arch, inference, opts.requirePerfectTiling);
    cand.traffic = global;
    cand.score = global.totalBytes() * static_cast<double>(cand.numWave);
    cand.cost.rawTraffic = global;
    cand.cost.memTraffic = global;
    cand.cost.waves = std::max<int64_t>(1, cand.numWave);
    cand.cost.blocksTotal = std::max<int64_t>(1, cand.blocksTotal);
    cand.cost.blocksPerSM = std::max<int64_t>(1, cand.blocksPerSM);
    cand.cost.sharedFootprintBytes = std::max<int64_t>(0, cand.estFootprintBytes);
    cand.cost.bankConflictFactor = std::max(1.0, cand.estSharedBankConflict);
    cand.cost.estimatedLatency = cand.score;
  }
  return true;
}

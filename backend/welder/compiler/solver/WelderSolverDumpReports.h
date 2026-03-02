bool dumpPaperExecutionPlan(ModuleOp module, const SolveOptions &optsIn,
                            const std::string &path) {
  SolveOptions opts = optsIn;
  const PaperScheduleResolvedLevels scheduleLevels =
      resolvePaperScheduleResolvedLevels(opts);
  const int sharedMinLevelExclusive = scheduleLevels.shared.minLevelExclusive;
  const int sharedMaxLevelInclusive = scheduleLevels.shared.maxLevelInclusive;
  const int recursiveInnerMinLevelExclusive =
      scheduleLevels.recursiveInnerMinLevelExclusive;
  inferArchElementBytesFromModule(module, opts.arch);
  if (!opts.enablePaperSchedule) {
    llvm::errs() << "error: dumpPaperExecutionPlan requires --enable-paper-schedule\n";
    return false;
  }
  // Keep consistent with welder::solve 默认值.
  opts.autoCandidates = true;

  auto graphOpt = buildLinalgTileGraph(module);
  if (!graphOpt || graphOpt->nodes.empty()) {
    llvm::errs() << "error: cannot build TileGraph\n";
    return false;
  }
  TileGraph g0 = *graphOpt;

  auto sinkOpt = findGraphSinkNode(g0);
  if (!sinkOpt) {
    llvm::errs() << "error: cannot find sink node in TileGraph\n";
    return false;
  }
  int sinkNodeIdx = *sinkOpt;
  auto sinkOp = dyn_cast_or_null<linalg::LinalgOp>(g0.nodes[sinkNodeIdx].op);
  if (!sinkOp) {
    llvm::errs() << "error: sink is not a linalg op\n";
    return false;
  }

  LinalgIndexingMapsFootprintInference infer;
  (void)graphConnectingPaperGlobalShared(g0, opts, infer, opts.requirePerfectTiling);

  PaperSubgraph sg0 =
      extractSubgraphByConnectLevel(g0, sinkNodeIdx, sharedMinLevelExclusive);

  auto bestList = subGraphTilingPaperGlobalShared(g0, sg0, sinkOp, sinkNodeIdx, opts,
                                                  infer);
  if (bestList.empty()) {
    llvm::errs() << "error: SubGraphTiling produced no configs\n";
    return false;
  }
  PaperScheduleCandidate bestPc = bestList.front();
  Candidate best = bestPc.cand;

  // Re-run 传播 for the chosen config to obtain per-op requiredTile for plan dump.
  // This mirrors the 传播 logic inside the paper schedule evaluation.
  std::vector<std::vector<int64_t>> reduceTilesByNode =
      assignReduceTilesByCoalescingPaper(g0, opts.arch, infer);

  auto parExtOpt = buildRootParallelExtents2Level(sinkOp, best, opts);
  if (!parExtOpt) {
    llvm::errs() << "error: cannot build root parallel extents for plan dump\n";
    return false;
  }

  llvm::ArrayRef<int64_t> sinkRed;
  if (sinkNodeIdx >= 0 && static_cast<size_t>(sinkNodeIdx) < reduceTilesByNode.size())
    sinkRed = reduceTilesByNode[sinkNodeIdx];

  auto rootTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
      sinkOp, *parExtOpt, sinkRed, /*defaultReductionTile=*/0);
  if (!rootTileOpt) {
    llvm::errs() << "error: cannot build root OpTile for plan dump\n";
    return false;
  }

  TileGraph g = g0; // copy
  syncCutFlagFromConnectLevel(g);

  TilePropagationOptions popts;
  popts.defaultReductionTile = 0;
  popts.reductionTilesByNode = &reduceTilesByNode;
  popts.enableCutEdges = true;
  popts.resetCutEdges = false;

  TilePropagationResult pr =
      propagateTilesBackward(g, sinkNodeIdx, *rootTileOpt, infer, popts);
  if (!pr.success) {
    llvm::errs() << "error: propagation failed for plan dump: " << pr.error << "\n";
    return false;
  }

  // 施加与评估器一致的鲁棒性切边规则。
  cutEdgesOnSwapXYConflict(g, sg0, sharedMinLevelExclusive);
  PaperSubgraph sg =
      extractSubgraphByConnectLevel(g, sinkNodeIdx, sharedMinLevelExclusive);

  // Shared 层分配计划：仅统计 shared 窗口内的 connectLevel。
  SharedAllocPlan sharedPlan = computeSharedAllocPlanBestFitPaper(
      g, sg, opts.arch, infer, opts.requirePerfectTiling, sharedMinLevelExclusive,
      sharedMaxLevelInclusive, best.workgroupPadLastDim,
      best.workgroupPadLastDimMatmulOnly, best.workgroupMultiBufferDepth,
      best.workgroupSwizzleXor, &best);

  // Register 层复用摘要：统计高于 shared 窗口的 connectLevel。
  struct RegReuseItem {
    int src = -1;
    int dst = -1;
    int level = 0;
    std::string value;
    int64_t bytes = 0;
  };
  llvm::SmallVector<RegReuseItem, 64> regReuse;
  {
    double elemBytesD = static_cast<double>(opts.arch.elementBytes);
    for (const TileGraphEdge &e : g.edges) {
      if (e.connectLevel <= sharedMaxLevelInclusive)
        continue;
      if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
        continue;
      if (isTrivialOpFor2LevelFootprint(g.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(g.nodes[e.dst].op))
        continue;
      double vol = getVolume(e.footprint);
      if (vol == 0.0)
        continue;
      int64_t bytes = static_cast<int64_t>(vol * elemBytesD);
      if (bytes <= 0)
        continue;
      RegReuseItem it;
      it.src = e.src;
      it.dst = e.dst;
      it.level = e.connectLevel;
      it.value = valueToString(e.value);
      it.bytes = bytes;
      regReuse.push_back(std::move(it));
    }
  }

  const int64_t maxRowReductionExtentForTc =
      computeTcRowReductionExtentForThreadMapping(g, sg);
  const int64_t blockThreads =
      estimateBlockThreadsForCandidate(best, maxRowReductionExtentForTc);
  llvm::SmallVector<RecursiveStageMetric, 4> recursiveStageMetrics;
  const RecursiveStageAggregate recursiveStageAgg =
      estimateRecursiveStageAggregateForCandidate(
          g, sg, opts, infer, scheduleLevels, blockThreads,
          best.workgroupPadLastDim, best.workgroupPadLastDimMatmulOnly,
          best.workgroupSwizzleXor, &recursiveStageMetrics,
          /* enforceFeasibility=*/false);
  const int64_t regReuseRegsPerThread = recursiveStageAgg.regReuseRegsPerThread;
  const double sharedToRegBytes = recursiveStageAgg.sharedToRegBytes;

  // 当前调度子图的 MemTraffic 分解结果。
  SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
      g, sg, sharedMinLevelExclusive, sharedMaxLevelInclusive,
      best.workgroupPadLastDim, best.workgroupPadLastDimMatmulOnly,
      best.workgroupSwizzleXor);
  MemTrafficBreakdown mt = computeMemTrafficForSubgraph(
      g, sg, opts.arch, infer, opts.requirePerfectTiling,
      sharedMinLevelExclusive,
      /* applyCoalescingPenalty=*/opts.enableCoalescingPenalty, &layout);

  std::ofstream ofs(path);
  if (!ofs) {
    llvm::errs() << "error: cannot open exec plan output: " << path << "\n";
    return false;
  }

  auto emitTrafficJson = [&](const Traffic &t) {
    ofs << "{\"bytesA\": " << t.bytesA << ", \"bytesB\": " << t.bytesB
        << ", \"bytesC\": " << t.bytesC << ", \"bytesCut\": " << t.bytesCut
        << ", \"total\": " << t.totalBytes() << "}";
  };

  // 最小化 JSON 输出，保证下游解析稳定。
  ofs << "{\n";
  ofs << "  \"arch\": {\"smemBytes\": " << opts.arch.smemBytes
      << ", \"numSM\": " << opts.arch.numSM
      << ", \"warpSize\": " << opts.arch.warpSize
      << ", \"elementBytes\": " << opts.arch.elementBytes << "},\n";
  ofs << "  \"candidate\": {"
      << "\"tileM\": " << best.tileM << ", \"tileN\": " << best.tileN
      << ", \"tileK\": " << best.tileK << ", \"threadTileM\": " << best.threadTileM
      << ", \"threadTileN\": " << best.threadTileN
      << ", \"waves\": " << best.cost.waves
      << ", \"blocksTotal\": " << best.cost.blocksTotal
      << ", \"blocksPerSM\": " << best.cost.blocksPerSM
      << ", \"sharedFootprintBytes\": " << best.cost.sharedFootprintBytes
      << ", \"sharedMinLevelExclusive\": " << sharedMinLevelExclusive
      << ", \"sharedMaxLevelInclusive\": " << sharedMaxLevelInclusive
      << ", \"recursiveInnerMinLevelExclusive\": "
      << recursiveInnerMinLevelExclusive
      << ", \"recursiveMaxStages\": " << opts.paperRecursiveMaxStages
      << ", \"sharedPeakBytes\": " << sharedPlan.peakBytes
      << ", \"recursiveStageCount\": "
      << static_cast<int64_t>(recursiveStageMetrics.size())
      << ", \"sharedToRegBytes\": " << sharedToRegBytes
      << ", \"trafficRaw\": ";
  emitTrafficJson(mt.raw);
  ofs << ", \"trafficMem\": ";
  emitTrafficJson(mt.mem);
  ofs << ", \"coalescingPenalty\": " << (opts.enableCoalescingPenalty ? "true" : "false")
      << ", \"estimatedLatency\": " << best.cost.estimatedLatency
      << ", \"profiledMs\": "
      << (best.cost.profiledMs.has_value() ? *best.cost.profiledMs : -1.0)
      << ", \"enableTensorCoreTf32\": " << (best.enableTensorCoreTf32 ? "true" : "false")
      << ", \"enableTensorCoreF16\": " << (best.enableTensorCoreF16 ? "true" : "false")
      << ", \"enableAsyncCopy\": " << (best.enableAsyncCopy ? "true" : "false")
      << ", \"asyncBypassL1\": " << (best.asyncBypassL1 ? "true" : "false")
      << ", \"enableSoftwarePipelining\": "
      << (best.enableSoftwarePipelining ? "true" : "false")
      << ", \"pipelineDepth\": " << best.pipelineDepth
      << ", \"pipelinePeelEpilogue\": " << (best.pipelinePeelEpilogue ? "true" : "false")
      << ", \"pipelineSetAsyncWaitGroups\": "
      << (best.pipelineSetAsyncWaitGroups ? "true" : "false")
      << ", \"workgroupMultiBufferDepth\": " << best.workgroupMultiBufferDepth
      << ", \"workgroupPadLastDim\": " << best.workgroupPadLastDim
      << ", \"workgroupPadLastDimMatmulOnly\": "
      << (best.workgroupPadLastDimMatmulOnly ? "true" : "false")
      << ", \"workgroupSwizzleXor\": " << best.workgroupSwizzleXor
      << ", \"blockRasterizeXor\": " << best.blockRasterizeXor
      << ", \"blockRasterizeMode\": " << best.blockRasterizeMode
      << ", \"blockRasterizePanelWidth\": " << best.blockRasterizePanelWidth
      << ", \"swapBlockDims\": " << (best.swapBlockDims ? "true" : "false")
      << ", \"enableRowReductionChainReuseFusion\": "
      << (best.enableRowReductionChainReuseFusion ? "true" : "false")
      << ", \"enableRowReductionInputPromotion\": "
      << (best.enableRowReductionInputPromotion ? "true" : "false")
      << ", \"enableRowReductionInputPromotionVectorize\": "
      << (best.enableRowReductionInputPromotionVectorize ? "true" : "false")
      << ", \"enableRowReductionWarp\": "
      << (best.enableRowReductionWarp ? "true" : "false")
      << ", \"enableRowReductionVectorize\": "
      << (best.enableRowReductionVectorize ? "true" : "false")
      << ", \"rowReductionVectorWidth\": " << best.rowReductionVectorWidth
      << ", \"rowReductionThreadsX\": " << best.rowReductionThreadsX
      << ", \"enableRowReductionRelaxBarriers\": "
      << (best.enableRowReductionRelaxBarriers ? "true" : "false")
      << ", \"enableRowReductionSkipCombineBarrier\": "
      << (best.enableRowReductionSkipCombineBarrier ? "true" : "false")
      << ", \"rowReductionInputVectorWidth\": "
      << best.rowReductionInputVectorWidth
      << ", \"enableRowReductionCombineVectorize\": "
      << (best.enableRowReductionCombineVectorize ? "true" : "false")
      << ", \"enableMatmulSoftmaxSharedReuseFusion\": "
      << (best.enableMatmulSoftmaxSharedReuseFusion ? "true" : "false")
      << "},\n";

  ofs << "  \"recursiveStages\": [\n";
  for (size_t i = 0; i < recursiveStageMetrics.size(); ++i) {
    const RecursiveStageMetric &stage = recursiveStageMetrics[i];
    ofs << "    {\"minLevelExclusive\": " << stage.minLevelExclusive
        << ", \"maxLevelInclusive\": " << stage.maxLevelInclusive
        << ", \"regReuseRegsPerThread\": " << stage.regReuseRegsPerThread
        << ", \"sharedToRegBytes\": " << stage.sharedToRegBytes << "}";
    ofs << (i + 1 == recursiveStageMetrics.size() ? "\n" : ",\n");
  }
  ofs << "  ],\n";

  ofs << "  \"sinkNodeIdx\": " << sinkNodeIdx << ",\n";

  ofs << "  \"nodes\": [\n";
  for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i) {
    const TileGraphNode &n = g.nodes[i];
    std::string opName = n.op ? n.op->getName().getStringRef().str() : "<null>";
    int64_t nodeId = -1;
    if (n.op) {
      if (auto idAttr = n.op->getAttrOfType<IntegerAttr>("welder.node_id"))
        nodeId = idAttr.getInt();
    }
    ofs << "    {\"idx\": " << i << ", \"op\": \"" << jsonEscape(opName)
        << "\", \"nodeId\": " << nodeId
        << ", \"swapXYHint\": "
        << (n.swapXYHint.has_value() ? (*n.swapXYHint ? "true" : "false")
                                     : "null")
        << ", \"hasRequiredTile\": " << (n.hasRequiredTile ? "true" : "false");
    if (n.hasRequiredTile) {
      ofs << ", \"loopTileExtents\": [";
      for (size_t k = 0; k < n.requiredTile.loopExtents.size(); ++k) {
        if (k)
          ofs << ", ";
        ofs << n.requiredTile.loopExtents[k];
      }
      ofs << "]";
      ofs << ", \"reductionSteps\": [";
      for (size_t k = 0; k < n.requiredTile.reductionSteps.size(); ++k) {
        if (k)
          ofs << ", ";
        ofs << n.requiredTile.reductionSteps[k];
      }
      ofs << "]";
    }
    ofs << "}";
    ofs << (i + 1 == static_cast<int>(g.nodes.size()) ? "\n" : ",\n");
  }
  ofs << "  ],\n";

  ofs << "  \"edges\": [\n";
  SharedLayoutPolicyV1 layoutPolicy = buildSharedLayoutPolicyV1(
      g, sg, /*minLevelExclusive=*/0, /*maxLevelInclusive=*/1,
      best.workgroupPadLastDim, best.workgroupPadLastDimMatmulOnly,
      best.workgroupSwizzleXor);
  for (int i = 0; i < static_cast<int>(g.edges.size()); ++i) {
    const TileGraphEdge &e = g.edges[i];
    SharedLayoutInfo li = layoutPolicy.get(e.value.getAsOpaquePointer());
    double vol = getVolume(e.footprint);
    int64_t fpBytes = 0;
    if (vol > 0.0 && opts.arch.elementBytes > 0) {
      fpBytes =
          static_cast<int64_t>(vol * static_cast<double>(opts.arch.elementBytes));
    }
    ofs << "    {\"idx\": " << i << ", \"src\": " << e.src << ", \"dst\": " << e.dst
        << ", \"srcResult\": " << e.srcResult
        << ", \"dstOperand\": " << e.dstOperand
        << ", \"connectLevel\": " << e.connectLevel
        << ", \"isCut\": " << (e.isCut ? "true" : "false")
        << ", \"layoutPadLastDim\": " << li.padLastDim
        << ", \"layoutSwizzleXor\": " << li.swizzleXor
        << ", \"viewOps\": [";
    for (size_t k = 0; k < e.viewOps.size(); ++k) {
      if (k)
        ofs << ", ";
      std::string opName =
          e.viewOps[k] ? e.viewOps[k]->getName().getStringRef().str() : "<null>";
      ofs << "\"" << jsonEscape(opName) << "\"";
    }
    ofs << "]"
        << ", \"value\": \"" << jsonEscape(valueToString(e.value)) << "\""
        << ", \"footprintShape\": [";
    for (size_t k = 0; k < e.footprint.shape.size(); ++k) {
      if (k)
        ofs << ", ";
      ofs << e.footprint.shape[k];
    }
    ofs << "], \"footprintBytes\": " << fpBytes << "}";
    ofs << (i + 1 == static_cast<int>(g.edges.size()) ? "\n" : ",\n");
  }
  ofs << "  ],\n";

  ofs << "  \"shared\": {\"peakBytes\": " << sharedPlan.peakBytes
      << ", \"events\": [\n";
  for (size_t i = 0; i < sharedPlan.events.size(); ++i) {
    const SharedAllocEvent &ev = sharedPlan.events[i];
    ofs << "    {\"action\": \"" << jsonEscape(ev.action) << "\", \"kind\": \""
        << jsonEscape(ev.kind) << "\", \"nodeIdx\": " << ev.nodeIdx
        << ", \"bytes\": " << ev.bytes << ", \"offset\": " << ev.offset
        << ", \"padLastDim\": " << ev.padLastDim
        << ", \"swizzleXor\": " << ev.swizzleXor
        << ", \"value\": \"" << jsonEscape(ev.value) << "\"}";
    ofs << (i + 1 == sharedPlan.events.size() ? "\n" : ",\n");
  }
  ofs << "  ]},\n";

  ofs << "  \"register\": {\"reuseRegsPerThread\": " << regReuseRegsPerThread
      << ", \"reuse\": [\n";
  for (size_t i = 0; i < regReuse.size(); ++i) {
    const RegReuseItem &it = regReuse[i];
    ofs << "    {\"src\": " << it.src << ", \"dst\": " << it.dst
        << ", \"connectLevel\": " << it.level << ", \"bytes\": " << it.bytes
        << ", \"value\": \"" << jsonEscape(it.value) << "\"}";
    ofs << (i + 1 == regReuse.size() ? "\n" : ",\n");
  }
  ofs << "  ]}\n";

  ofs << "}\n";
  return true;
}

bool dumpTileGraphJson(ModuleOp module, const SolveOptions &optsIn,
                       const std::string &path) {
  SolveOptions opts = optsIn;
  inferArchElementBytesFromModule(module, opts.arch);

  auto graphOpt = buildLinalgTileGraph(module);
  if (!graphOpt) {
    llvm::errs() << "error: dumpTileGraphJson: cannot build TileGraph\n";
    return false;
  }
  const TileGraph &g = *graphOpt;

  std::ofstream ofs(path);
  if (!ofs) {
    llvm::errs() << "error: dumpTileGraphJson: cannot open output: " << path
                 << "\n";
    return false;
  }

  ofs << "{\n";
  ofs << "  \"arch\": {\"smemBytes\": " << opts.arch.smemBytes
      << ", \"numSM\": " << opts.arch.numSM
      << ", \"warpSize\": " << opts.arch.warpSize
      << ", \"elementBytes\": " << opts.arch.elementBytes << "},\n";

  ofs << "  \"nodes\": [\n";
  for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i) {
    const TileGraphNode &n = g.nodes[i];
    std::string opName = n.op ? n.op->getName().getStringRef().str() : "<null>";
    int64_t nodeId = -1;
    if (n.op) {
      if (auto idAttr = n.op->getAttrOfType<IntegerAttr>("welder.node_id"))
        nodeId = idAttr.getInt();
    }
    ofs << "    {\"idx\": " << i << ", \"op\": \"" << jsonEscape(opName)
        << "\", \"nodeId\": " << nodeId
        << ", \"swapXYHint\": "
        << (n.swapXYHint.has_value() ? (*n.swapXYHint ? "true" : "false")
                                     : "null")
        << ", \"hasRequiredTile\": " << (n.hasRequiredTile ? "true" : "false")
        << ", \"requiredTileLoopExtents\": [";
    if (n.hasRequiredTile) {
      for (size_t k = 0; k < n.requiredTile.loopExtents.size(); ++k) {
        if (k)
          ofs << ", ";
        ofs << n.requiredTile.loopExtents[k];
      }
    }
    ofs << "]}";
    ofs << (i + 1 == static_cast<int>(g.nodes.size()) ? "\n" : ",\n");
  }
  ofs << "  ],\n";

  ofs << "  \"edges\": [\n";
  PaperSubgraph allNodes;
  allNodes.nodes.reserve(g.nodes.size());
  for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i) {
    allNodes.nodes.push_back(i);
    allNodes.inSet.insert(i);
  }
  SharedLayoutPolicyV1 layoutPolicy = buildSharedLayoutPolicyV1(
      g, allNodes, /*minLevelExclusive=*/0, /*maxLevelInclusive=*/1,
      opts.profile.workgroupPadLastDim,
      opts.profile.workgroupPadLastDimMatmulOnly,
      opts.profile.workgroupSwizzleXor);

  for (int i = 0; i < static_cast<int>(g.edges.size()); ++i) {
    const TileGraphEdge &e = g.edges[i];
    SharedLayoutInfo li = layoutPolicy.get(e.value.getAsOpaquePointer());
    ofs << "    {\"idx\": " << i << ", \"src\": " << e.src
        << ", \"dst\": " << e.dst << ", \"srcResult\": " << e.srcResult
        << ", \"dstOperand\": " << e.dstOperand
        << ", \"connectLevel\": " << e.connectLevel
        << ", \"isCut\": " << (e.isCut ? "true" : "false")
        << ", \"layoutPadLastDim\": " << li.padLastDim
        << ", \"layoutSwizzleXor\": " << li.swizzleXor
        << ", \"viewOps\": [";
    for (size_t k = 0; k < e.viewOps.size(); ++k) {
      if (k)
        ofs << ", ";
      std::string opName =
          e.viewOps[k] ? e.viewOps[k]->getName().getStringRef().str() : "<null>";
      ofs << "\"" << jsonEscape(opName) << "\"";
    }
    ofs << "]"
        << ", \"value\": \"" << jsonEscape(valueToString(e.value)) << "\"}";
    ofs << (i + 1 == static_cast<int>(g.edges.size()) ? "\n" : ",\n");
  }
  ofs << "  ]\n";

  ofs << "}\n";
  return true;
}

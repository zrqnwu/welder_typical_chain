static int resolvePaperRecursiveMaxStages(const SolveOptions &opts) {
  if (opts.paperRecursiveMaxStages <= 0)
    return 0;
  return std::max(1, opts.paperRecursiveMaxStages);
}
static int resolvePaperRecursiveInnerMinLevelExclusive(
    const SolveOptions &opts) {
  const int maxConnectLevel = std::max(0, opts.maxConnectLevel);
  if (maxConnectLevel <= 1)
    return 0;
  const int recursiveMaxStages = resolvePaperRecursiveMaxStages(opts);
  const bool fullWindowDefault =
      getEnvInt64OrDefault("WELDER_PAPER_RECURSIVE_FULL_WINDOW_DEFAULT", 1) !=
      0;
  int resolved = std::max(1, maxConnectLevel - 1);
  if (opts.paperRecursiveInnerMinLevelExclusive > 0) {
    resolved = opts.paperRecursiveInnerMinLevelExclusive;
  } else if (recursiveMaxStages > 0) {
    resolved = std::max(1, maxConnectLevel - recursiveMaxStages);
  } else if (fullWindowDefault && maxConnectLevel > kConnectLevelShared + 1) {
    // P2（更完整的递归）：当未显式给出边界/阶段上限时，
    // 从 shared 层边界继续向更深层展开递归，
    // 而不是默认只保留最深的单阶段。
    resolved = kConnectLevelShared;
  }
  if (recursiveMaxStages > 0) {
    const int minBoundaryForStageCap =
        std::max(1, maxConnectLevel - recursiveMaxStages);
    resolved = std::max(resolved, minBoundaryForStageCap);
  }
  resolved = std::max(1, std::min(resolved, maxConnectLevel - 1));
  return resolved;
}

struct PaperScheduleLevelWindow {
  int minLevelExclusive = kConnectLevelGlobal;
  int maxLevelInclusive = kConnectLevelShared;
};

struct PaperScheduleResolvedLevels {
  PaperScheduleLevelWindow shared;
  int recursiveInnerMinLevelExclusive = kConnectLevelGlobal;
};

static PaperScheduleResolvedLevels
resolvePaperScheduleResolvedLevels(const SolveOptions &opts) {
  PaperScheduleResolvedLevels levels;
  levels.shared.minLevelExclusive = kConnectLevelGlobal;
  levels.shared.maxLevelInclusive = std::min(
      kConnectLevelShared, std::max(kConnectLevelGlobal, opts.maxConnectLevel));
  levels.recursiveInnerMinLevelExclusive =
      resolvePaperRecursiveInnerMinLevelExclusive(opts);
  return levels;
}

static llvm::SmallVector<PaperScheduleLevelWindow, 4>
resolvePaperRecursiveLevelWindows(const SolveOptions &opts,
                                  const PaperScheduleResolvedLevels &levels) {
  llvm::SmallVector<PaperScheduleLevelWindow, 4> windows;
  const int maxConnectLevel = std::max(0, opts.maxConnectLevel);
  int lower = std::max(levels.shared.maxLevelInclusive,
                       levels.recursiveInnerMinLevelExclusive);
  lower = std::max(lower, kConnectLevelShared);
  if (maxConnectLevel <= lower)
    return windows;
  for (int level = lower + 1; level <= maxConnectLevel; ++level) {
    PaperScheduleLevelWindow w;
    w.minLevelExclusive = level - 1;
    w.maxLevelInclusive = level;
    windows.push_back(w);
  }
  return windows;
}

struct RecursiveStageAggregate {
  bool feasible = true;
  int failedStage = -1;
  int64_t regReuseRegsPerThread = 0;
  double sharedToRegBytes = 0.0;
};

struct RecursiveStageMetric {
  int minLevelExclusive = 0;
  int maxLevelInclusive = 0;
  int64_t regReuseRegsPerThread = 0;
  double sharedToRegBytes = 0.0;
};

static RecursiveStageAggregate
estimateRecursiveStageAggregateForCandidate(
    const TileGraph &graph, const PaperSubgraph &sg, const SolveOptions &opts,
    const FootprintInference &inference,
    const PaperScheduleResolvedLevels &scheduleLevels, int64_t blockThreads,
    int64_t workgroupPadLastDim, bool workgroupPadLastDimMatmulOnly,
    int64_t workgroupSwizzleXor,
    llvm::SmallVectorImpl<RecursiveStageMetric> *stageMetrics = nullptr,
    bool enforceFeasibility = true) {
  RecursiveStageAggregate agg;
  if (opts.maxConnectLevel < 2)
    return agg;
  if (stageMetrics)
    stageMetrics->clear();
  const auto windows = resolvePaperRecursiveLevelWindows(opts, scheduleLevels);
  const double stageBytesHardLimit =
      getEnvDoubleOrDefault("WELDER_RECURSIVE_STAGE_BYTES_HARD_LIMIT", 0.0);
  for (size_t idx = 0; idx < windows.size(); ++idx) {
    const PaperScheduleLevelWindow &w = windows[idx];
    RecursiveStageMetric stage;
    stage.minLevelExclusive = w.minLevelExclusive;
    stage.maxLevelInclusive = w.maxLevelInclusive;
    if (blockThreads > 0) {
      stage.regReuseRegsPerThread = estimateRegisterReuseRegsPerThreadForSubgraph(
          graph, sg, w.minLevelExclusive, blockThreads, opts.arch,
          w.maxLevelInclusive);
      agg.regReuseRegsPerThread =
          std::max(agg.regReuseRegsPerThread, stage.regReuseRegsPerThread);
    }
    stage.sharedToRegBytes = computeSharedToRegTrafficBytesForSubgraph(
        graph, sg, opts.arch, inference, opts.requirePerfectTiling,
        w.minLevelExclusive, w.maxLevelInclusive, workgroupPadLastDim,
        workgroupPadLastDimMatmulOnly, workgroupSwizzleXor);
    agg.sharedToRegBytes += std::max(0.0, stage.sharedToRegBytes);
    if (stageMetrics)
      stageMetrics->push_back(stage);
    if (!enforceFeasibility)
      continue;
    if (blockThreads > 0 &&
        stage.regReuseRegsPerThread > opts.arch.maxRegistersPerThread) {
      agg.feasible = false;
      agg.failedStage = static_cast<int>(idx);
      return agg;
    }
    if (stageBytesHardLimit > 0.0 && stage.sharedToRegBytes > stageBytesHardLimit) {
      agg.feasible = false;
      agg.failedStage = static_cast<int>(idx);
      return agg;
    }
  }
  return agg;
}

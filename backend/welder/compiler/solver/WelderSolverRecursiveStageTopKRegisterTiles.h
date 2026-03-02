static llvm::SmallVector<std::pair<int64_t, int64_t>, 32>
selectRecursiveStageTopKRegisterTiles(
    llvm::ArrayRef<std::pair<int64_t, int64_t>> regTiles,
    const Candidate &baseCandidate, const TileGraph &graph,
    const PaperSubgraph &sg, const SolveOptions &opts,
    const FootprintInference &inference,
    const PaperScheduleResolvedLevels &scheduleLevels,
    int64_t maxRowReductionExtentForTc) {
  llvm::SmallVector<std::pair<int64_t, int64_t>, 32> out;
  if (regTiles.empty())
    return out;
  out.append(regTiles.begin(), regTiles.end());
  if (regTiles.size() <= 1 || opts.maxConnectLevel < 3)
    return out;
  const int64_t enableRegStageTopK = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_REG_STAGE_TOPK_ENABLE", /*default=*/1);
  if (enableRegStageTopK == 0)
    return out;

  const auto windows = resolvePaperRecursiveLevelWindows(opts, scheduleLevels);
  if (windows.size() <= 1)
    return out;

  const int64_t stageTopK = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_REG_STAGE_TOPK", /*default=*/16);
  const int64_t stageMinKeep = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_RECURSIVE_REG_STAGE_MIN_KEEP",
                              /*default=*/8));
  double stageKeepRatio = getEnvDoubleOrDefault(
      "WELDER_RECURSIVE_REG_STAGE_KEEP_RATIO", /*default=*/0.8);
  if (!std::isfinite(stageKeepRatio))
    stageKeepRatio = 0.8;
  stageKeepRatio = std::max(0.0, std::min(1.0, stageKeepRatio));
  const int64_t incrementalEnable = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_REG_STAGE_INCREMENTAL_ENABLE", /*default=*/1);
  const int64_t incrementalSeeds = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_RECURSIVE_REG_STAGE_INCREMENTAL_SEEDS",
                              /*default=*/8));

  auto keyOf = [](int64_t m, int64_t n) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(m)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(n));
  };
  llvm::DenseMap<uint64_t, int64_t> pairToIdx;
  pairToIdx.reserve(regTiles.size());
  llvm::SmallVector<int64_t, 64> valsM;
  llvm::SmallVector<int64_t, 64> valsN;
  valsM.reserve(regTiles.size());
  valsN.reserve(regTiles.size());
  for (size_t i = 0; i < regTiles.size(); ++i) {
    pairToIdx[keyOf(regTiles[i].first, regTiles[i].second)] =
        static_cast<int64_t>(i);
    valsM.push_back(regTiles[i].first);
    valsN.push_back(regTiles[i].second);
  }
  llvm::sort(valsM);
  valsM.erase(std::unique(valsM.begin(), valsM.end()), valsM.end());
  llvm::sort(valsN);
  valsN.erase(std::unique(valsN.begin(), valsN.end()), valsN.end());
  llvm::DenseMap<int64_t, int> posM;
  llvm::DenseMap<int64_t, int> posN;
  for (int i = 0; i < static_cast<int>(valsM.size()); ++i)
    posM[valsM[static_cast<size_t>(i)]] = i;
  for (int i = 0; i < static_cast<int>(valsN.size()); ++i)
    posN[valsN[static_cast<size_t>(i)]] = i;

  std::vector<llvm::SmallVector<RecursiveStageMetric, 4>> stageCache(
      regTiles.size());
  std::vector<char> stageCacheReady(regTiles.size(), 0);
  auto ensureStageCache = [&](int64_t idx)
      -> const llvm::SmallVector<RecursiveStageMetric, 4> & {
    if (idx < 0 || idx >= static_cast<int64_t>(stageCache.size())) {
      static const llvm::SmallVector<RecursiveStageMetric, 4> kEmpty;
      return kEmpty;
    }
    size_t u = static_cast<size_t>(idx);
    if (!stageCacheReady[u]) {
      Candidate probe = baseCandidate;
      probe.threadTileM = regTiles[u].first;
      probe.threadTileN = regTiles[u].second;
      const int64_t blockThreads =
          estimateBlockThreadsForCandidate(probe, maxRowReductionExtentForTc);
      (void)estimateRecursiveStageAggregateForCandidate(
          graph, sg, opts, inference, scheduleLevels, blockThreads,
          probe.workgroupPadLastDim, probe.workgroupPadLastDimMatmulOnly,
          probe.workgroupSwizzleXor, &stageCache[u],
          /* enforceFeasibility=*/false);
      stageCacheReady[u] = 1;
    }
    return stageCache[u];
  };

  llvm::SmallVector<int64_t, 64> allIdx;
  allIdx.reserve(regTiles.size());
  for (size_t i = 0; i < regTiles.size(); ++i)
    allIdx.push_back(static_cast<int64_t>(i));
  llvm::sort(allIdx, [&](int64_t a, int64_t b) {
    auto score = [&](int64_t idx) -> std::tuple<int64_t, int64_t, int64_t> {
      const auto &tile = regTiles[static_cast<size_t>(idx)];
      int64_t ttm = tile.first;
      int64_t ttn = tile.second;
      int64_t d1 = std::abs(ttm - 1) + std::abs(ttn - 1);
      int64_t d4 = std::abs(ttm - 4) + std::abs(ttn - 4);
      int64_t area = std::max<int64_t>(1, ttm) * std::max<int64_t>(1, ttn);
      return std::make_tuple(d1, d4, area);
    };
    auto sa = score(a);
    auto sb = score(b);
    if (sa != sb)
      return sa < sb;
    return a < b;
  });
  auto computeKeepN = [&](int64_t curN) -> int64_t {
    int64_t keepN = curN;
    if (stageTopK > 0)
      keepN = std::min<int64_t>(keepN, stageTopK);
    if (stageKeepRatio > 0.0 && stageKeepRatio < 1.0) {
      int64_t byRatio =
          std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                 static_cast<double>(curN) * stageKeepRatio)));
      keepN = std::min<int64_t>(keepN, byRatio);
    }
    keepN = std::max<int64_t>(std::min<int64_t>(curN, stageMinKeep), keepN);
    keepN = std::max<int64_t>(1, std::min<int64_t>(curN, keepN));
    return keepN;
  };
  const int64_t enableRegGlobalPrune = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_REG_STAGE_GLOBAL_PRUNE_ENABLE", /*default=*/1);
  if (enableRegGlobalPrune != 0) {
    struct GlobalScoredIdx {
      int64_t idx = -1;
      double cumulativeBytes = std::numeric_limits<double>::infinity();
      double maxStageBytes = std::numeric_limits<double>::infinity();
      int64_t maxStageRegs = std::numeric_limits<int64_t>::max();
      int64_t area = std::numeric_limits<int64_t>::max();
    };
    llvm::SmallVector<GlobalScoredIdx, 64> scored;
    scored.reserve(allIdx.size());
    for (int64_t idx : allIdx) {
      GlobalScoredIdx s;
      s.idx = idx;
      if (idx >= 0 && idx < static_cast<int64_t>(regTiles.size())) {
        const auto &tile = regTiles[static_cast<size_t>(idx)];
        s.area = std::max<int64_t>(1, tile.first) *
                 std::max<int64_t>(1, tile.second);
      }
      const auto &metrics = ensureStageCache(idx);
      double cumulativeBytes = 0.0;
      double maxStageBytes = 0.0;
      int64_t maxStageRegs = 0;
      bool cumulativeFinite = true;
      for (const RecursiveStageMetric &m : metrics) {
        maxStageRegs = std::max(maxStageRegs, m.regReuseRegsPerThread);
        if (!std::isfinite(m.sharedToRegBytes)) {
          cumulativeFinite = false;
          break;
        }
        double stageBytes = std::max(0.0, m.sharedToRegBytes);
        cumulativeBytes += stageBytes;
        maxStageBytes = std::max(maxStageBytes, stageBytes);
      }
      s.maxStageRegs = maxStageRegs;
      if (cumulativeFinite) {
        s.cumulativeBytes = cumulativeBytes;
        s.maxStageBytes = maxStageBytes;
      }
      scored.push_back(std::move(s));
    }
    llvm::sort(scored, [](const GlobalScoredIdx &a, const GlobalScoredIdx &b) {
      const bool aCumFinite = std::isfinite(a.cumulativeBytes);
      const bool bCumFinite = std::isfinite(b.cumulativeBytes);
      if (aCumFinite != bCumFinite)
        return aCumFinite;
      if (a.cumulativeBytes != b.cumulativeBytes)
        return a.cumulativeBytes < b.cumulativeBytes;
      const bool aMaxFinite = std::isfinite(a.maxStageBytes);
      const bool bMaxFinite = std::isfinite(b.maxStageBytes);
      if (aMaxFinite != bMaxFinite)
        return aMaxFinite;
      if (a.maxStageBytes != b.maxStageBytes)
        return a.maxStageBytes < b.maxStageBytes;
      if (a.maxStageRegs != b.maxStageRegs)
        return a.maxStageRegs < b.maxStageRegs;
      if (a.area != b.area)
        return a.area < b.area;
      return a.idx < b.idx;
    });
    const int64_t curN = static_cast<int64_t>(scored.size());
    const int64_t keepN = computeKeepN(curN);
    if (opts.tracer) {
      llvm::json::Object f;
      f["mode"] = "global_joint";
      f["before"] = curN;
      f["after"] = keepN;
      f["window_count"] = static_cast<int64_t>(windows.size());
      f["topk_cap"] = stageTopK;
      f["keep_ratio"] = stageKeepRatio;
      f["min_keep"] = stageMinKeep;
      opts.tracer->event("paper.recursive_regtile_prune_global", std::move(f),
                         /* isVerbose=*/true);
    }
    llvm::SmallDenseSet<int64_t, 64> seen;
    llvm::SmallVector<std::pair<int64_t, int64_t>, 32> prunedGlobal;
    prunedGlobal.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i) {
      int64_t idx = scored[static_cast<size_t>(i)].idx;
      if (idx < 0 || idx >= static_cast<int64_t>(regTiles.size()))
        continue;
      if (!seen.insert(idx).second)
        continue;
      prunedGlobal.push_back(regTiles[static_cast<size_t>(idx)]);
    }
    if (!prunedGlobal.empty())
      return prunedGlobal;
  }

  llvm::SmallVector<int64_t, 64> selected;
  if (incrementalEnable != 0) {
    llvm::SmallDenseSet<int64_t, 64> seedSet;
    for (int64_t i = 0;
         i < std::min<int64_t>(incrementalSeeds,
                               static_cast<int64_t>(allIdx.size()));
         ++i)
      seedSet.insert(allIdx[static_cast<size_t>(i)]);
    int64_t maxAreaIdx = allIdx.front();
    for (int64_t idx : allIdx) {
      const auto &cur = regTiles[static_cast<size_t>(idx)];
      const auto &best = regTiles[static_cast<size_t>(maxAreaIdx)];
      int64_t curArea = std::max<int64_t>(1, cur.first) *
                        std::max<int64_t>(1, cur.second);
      int64_t bestArea = std::max<int64_t>(1, best.first) *
                         std::max<int64_t>(1, best.second);
      if (curArea > bestArea)
        maxAreaIdx = idx;
    }
    seedSet.insert(maxAreaIdx);
    for (int64_t idx : seedSet)
      selected.push_back(idx);
    llvm::sort(selected);
  } else {
    selected = allIdx;
  }
  if (selected.empty())
    selected = allIdx;

  struct StageScoredIdx {
    int64_t idx = -1;
    double stageBytes = std::numeric_limits<double>::infinity();
    double cumulativeBytes = std::numeric_limits<double>::infinity();
    int64_t prefixRegs = std::numeric_limits<int64_t>::max();
  };

  for (size_t stageIdx = 0; stageIdx < windows.size(); ++stageIdx) {
    if (selected.size() <= 1)
      break;
    llvm::SmallDenseSet<int64_t, 128> frontierSet;
    for (int64_t idx : selected) {
      if (idx >= 0 && idx < static_cast<int64_t>(regTiles.size()))
        frontierSet.insert(idx);
    }
    if (incrementalEnable != 0) {
      auto tryAddPair = [&](int64_t ttm, int64_t ttn) {
        auto it = pairToIdx.find(keyOf(ttm, ttn));
        if (it == pairToIdx.end())
          return;
        frontierSet.insert(it->second);
      };
      for (int64_t idx : selected) {
        if (idx < 0 || idx >= static_cast<int64_t>(regTiles.size()))
          continue;
        int64_t ttm = regTiles[static_cast<size_t>(idx)].first;
        int64_t ttn = regTiles[static_cast<size_t>(idx)].second;
        auto itM = posM.find(ttm);
        auto itN = posN.find(ttn);
        if (itM == posM.end() || itN == posN.end())
          continue;
        int pm = itM->second;
        int pn = itN->second;
        const int delta[] = {-1, 1};
        for (int d : delta) {
          int nm = pm + d;
          int nn = pn + d;
          if (nm >= 0 && nm < static_cast<int>(valsM.size()))
            tryAddPair(valsM[static_cast<size_t>(nm)], ttn);
          if (nn >= 0 && nn < static_cast<int>(valsN.size()))
            tryAddPair(ttm, valsN[static_cast<size_t>(nn)]);
          if (nm >= 0 && nm < static_cast<int>(valsM.size()) &&
              nn >= 0 && nn < static_cast<int>(valsN.size()))
            tryAddPair(valsM[static_cast<size_t>(nm)],
                       valsN[static_cast<size_t>(nn)]);
        }
      }
    }
    llvm::SmallVector<int64_t, 128> frontier;
    frontier.reserve(frontierSet.size());
    for (int64_t idx : frontierSet)
      frontier.push_back(idx);
    llvm::sort(frontier);
    if (frontier.empty())
      break;
    llvm::SmallVector<StageScoredIdx, 64> scored;
    scored.reserve(frontier.size());
    for (int64_t idx : frontier) {
      if (idx < 0 || idx >= static_cast<int64_t>(regTiles.size()))
        continue;
      StageScoredIdx s;
      s.idx = idx;
      const auto &metrics = ensureStageCache(idx);
      if (stageIdx < metrics.size()) {
        s.stageBytes = metrics[stageIdx].sharedToRegBytes;
        double cumulativeBytes = 0.0;
        int64_t prefixRegs = 0;
        bool cumulativeFinite = true;
        for (size_t j = 0; j <= stageIdx; ++j) {
          const RecursiveStageMetric &m = metrics[j];
          prefixRegs = std::max(prefixRegs, m.regReuseRegsPerThread);
          if (!std::isfinite(m.sharedToRegBytes)) {
            cumulativeFinite = false;
            break;
          }
          cumulativeBytes += std::max(0.0, m.sharedToRegBytes);
        }
        s.prefixRegs = prefixRegs;
        if (cumulativeFinite)
          s.cumulativeBytes = cumulativeBytes;
      }
      scored.push_back(std::move(s));
    }
    if (scored.empty())
      break;
    llvm::sort(scored, [](const StageScoredIdx &a, const StageScoredIdx &b) {
      const bool aCumFinite = std::isfinite(a.cumulativeBytes);
      const bool bCumFinite = std::isfinite(b.cumulativeBytes);
      if (aCumFinite != bCumFinite)
        return aCumFinite;
      if (a.cumulativeBytes != b.cumulativeBytes)
        return a.cumulativeBytes < b.cumulativeBytes;
      const bool aFinite = std::isfinite(a.stageBytes);
      const bool bFinite = std::isfinite(b.stageBytes);
      if (aFinite != bFinite)
        return aFinite;
      if (a.stageBytes != b.stageBytes)
        return a.stageBytes < b.stageBytes;
      if (a.prefixRegs != b.prefixRegs)
        return a.prefixRegs < b.prefixRegs;
      return a.idx < b.idx;
    });

    int64_t curN = static_cast<int64_t>(scored.size());
    int64_t keepN = computeKeepN(curN);
    if (keepN >= curN)
      continue;

    if (opts.tracer) {
      llvm::json::Object f;
      f["stage_idx"] = static_cast<int64_t>(stageIdx);
      f["min_level_exclusive"] =
          static_cast<int64_t>(windows[stageIdx].minLevelExclusive);
      f["max_level_inclusive"] =
          static_cast<int64_t>(windows[stageIdx].maxLevelInclusive);
      f["before"] = curN;
      f["after"] = keepN;
      f["topk_cap"] = stageTopK;
      f["keep_ratio"] = stageKeepRatio;
      f["min_keep"] = stageMinKeep;
      f["incremental"] = (incrementalEnable != 0);
      f["seed_count"] = static_cast<int64_t>(selected.size());
      f["frontier"] = static_cast<int64_t>(frontier.size());
      opts.tracer->event("paper.recursive_regtile_prune", std::move(f),
                         /* isVerbose=*/true);
    }

    selected.clear();
    selected.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i)
      selected.push_back(scored[static_cast<size_t>(i)].idx);
  }

  llvm::SmallDenseSet<int64_t, 64> seen;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 32> pruned;
  pruned.reserve(selected.size());
  for (int64_t idx : selected) {
    if (idx < 0 || idx >= static_cast<int64_t>(regTiles.size()))
      continue;
    if (!seen.insert(idx).second)
      continue;
    pruned.push_back(regTiles[static_cast<size_t>(idx)]);
  }
  if (!pruned.empty())
    return pruned;
  return out;
}

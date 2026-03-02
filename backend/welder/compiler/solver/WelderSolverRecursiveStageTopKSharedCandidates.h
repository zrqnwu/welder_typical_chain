static llvm::SmallVector<int64_t, 64> selectRecursiveStageTopKSharedCandidateIndices(
    int64_t candidateCount,
    const std::function<const Candidate &(int64_t)> &getCandidate,
    const std::function<const TileGraph &(int64_t)> &getGraph,
    const std::function<const PaperSubgraph &(int64_t)> &getSubgraph,
    const std::function<double(int64_t)> &getPriority,
    const SolveOptions &opts, const FootprintInference &inference,
    const PaperScheduleResolvedLevels &scheduleLevels, int64_t hardKeepCap) {
  llvm::SmallVector<int64_t, 64> selected;
  if (candidateCount <= 0)
    return selected;
  selected.reserve(static_cast<size_t>(candidateCount));
  for (int64_t i = 0; i < candidateCount; ++i)
    selected.push_back(i);
  if (candidateCount <= 1)
    return selected;
  if (opts.maxConnectLevel < 3)
    return selected;

  const int64_t enableSharedStageTopK = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_TOPK_ENABLE", /*default=*/1);
  if (enableSharedStageTopK == 0)
    return selected;

  const auto windows = resolvePaperRecursiveLevelWindows(opts, scheduleLevels);
  if (windows.size() <= 1)
    return selected;

  const int64_t stageTopK = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_TOPK", /*default=*/24);
  const int64_t stageMinKeep = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_RECURSIVE_SHARED_STAGE_MIN_KEEP",
                              /*default=*/8));
  double stageKeepRatio = getEnvDoubleOrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_KEEP_RATIO", /*default=*/0.8);
  if (!std::isfinite(stageKeepRatio))
    stageKeepRatio = 0.8;
  stageKeepRatio = std::max(0.0, std::min(1.0, stageKeepRatio));
  const int64_t incrementalEnable = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_INCREMENTAL_ENABLE", /*default=*/1);
  const int64_t incrementalSeeds = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_RECURSIVE_SHARED_STAGE_INCREMENTAL_SEEDS",
                              /*default=*/8));

  auto keyOf = [](int64_t tm, int64_t tn) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(tm)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(tn));
  };
  llvm::DenseMap<uint64_t, llvm::SmallVector<int64_t, 4>> pairToIndices;
  llvm::SmallVector<int64_t, 64> valsM;
  llvm::SmallVector<int64_t, 64> valsN;
  valsM.reserve(static_cast<size_t>(candidateCount));
  valsN.reserve(static_cast<size_t>(candidateCount));
  for (int64_t idx = 0; idx < candidateCount; ++idx) {
    const Candidate &cand = getCandidate(idx);
    if (cand.tileM <= 0 || cand.tileN <= 0)
      continue;
    pairToIndices[keyOf(cand.tileM, cand.tileN)].push_back(idx);
    valsM.push_back(cand.tileM);
    valsN.push_back(cand.tileN);
  }
  if (pairToIndices.size() <= 1)
    return selected;

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

  llvm::SmallVector<int64_t, 64> allIdx;
  allIdx.reserve(static_cast<size_t>(candidateCount));
  for (int64_t idx = 0; idx < candidateCount; ++idx) {
    const Candidate &cand = getCandidate(idx);
    if (cand.tileM > 0 && cand.tileN > 0)
      allIdx.push_back(idx);
  }
  if (allIdx.size() <= 1)
    return selected;

  llvm::sort(allIdx, [&](int64_t a, int64_t b) {
    const Candidate &ca = getCandidate(a);
    const Candidate &cb = getCandidate(b);
    int64_t da1 = std::abs(ca.tileM - 1) + std::abs(ca.tileN - 1);
    int64_t db1 = std::abs(cb.tileM - 1) + std::abs(cb.tileN - 1);
    if (da1 != db1)
      return da1 < db1;
    int64_t da4 = std::abs(ca.tileM - 4) + std::abs(ca.tileN - 4);
    int64_t db4 = std::abs(cb.tileM - 4) + std::abs(cb.tileN - 4);
    if (da4 != db4)
      return da4 < db4;
    int64_t aa = std::max<int64_t>(1, ca.tileM) * std::max<int64_t>(1, ca.tileN);
    int64_t ab = std::max<int64_t>(1, cb.tileM) * std::max<int64_t>(1, cb.tileN);
    if (aa != ab)
      return aa < ab;
    double pa = getPriority(a);
    double pb = getPriority(b);
    const bool paFinite = std::isfinite(pa);
    const bool pbFinite = std::isfinite(pb);
    if (paFinite != pbFinite)
      return paFinite;
    if (pa != pb)
      return pa < pb;
    return a < b;
  });

  std::vector<llvm::SmallVector<RecursiveStageMetric, 4>> stageCache(
      static_cast<size_t>(candidateCount));
  std::vector<char> stageCacheReady(static_cast<size_t>(candidateCount), 0);
  auto ensureStageCache = [&](int64_t idx)
      -> const llvm::SmallVector<RecursiveStageMetric, 4> & {
    if (idx < 0 || idx >= candidateCount) {
      static const llvm::SmallVector<RecursiveStageMetric, 4> kEmpty;
      return kEmpty;
    }
    size_t u = static_cast<size_t>(idx);
    if (!stageCacheReady[u]) {
      const Candidate &cand = getCandidate(idx);
      const TileGraph &g = getGraph(idx);
      const PaperSubgraph &sg = getSubgraph(idx);
      const int64_t maxRowReductionExtentForTc =
          computeTcRowReductionExtentForThreadMapping(g, sg);
      const int64_t blockThreads =
          estimateBlockThreadsForCandidate(cand, maxRowReductionExtentForTc);
      (void)estimateRecursiveStageAggregateForCandidate(
          g, sg, opts, inference, scheduleLevels, blockThreads,
          cand.workgroupPadLastDim, cand.workgroupPadLastDimMatmulOnly,
          cand.workgroupSwizzleXor, &stageCache[u],
          /* enforceFeasibility=*/false);
      stageCacheReady[u] = 1;
    }
    return stageCache[u];
  };
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
    if (hardKeepCap > 0)
      keepN = std::min<int64_t>(keepN, hardKeepCap);
    keepN = std::max<int64_t>(1, std::min<int64_t>(curN, keepN));
    return keepN;
  };
  const int64_t enableSharedGlobalPrune = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_GLOBAL_PRUNE_ENABLE", /*default=*/1);
  const int64_t stageClassAnchor = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_CLASS_ANCHOR", /*default=*/1);
  const int64_t stageAsyncClassAnchor = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_SHARED_STAGE_ASYNC_CLASS_ANCHOR", /*default=*/1);
  auto isTensorCoreIdx = [&](int64_t idx) -> bool {
    const Candidate &cand = getCandidate(idx);
    return cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
  };
  auto isTcAsyncIdx = [&](int64_t idx) -> bool {
    const Candidate &cand = getCandidate(idx);
    return (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
           (cand.enableAsyncCopy || cand.enableSoftwarePipelining);
  };
  auto isTcAsyncWaitIdx = [&](int64_t idx) -> bool {
    const Candidate &cand = getCandidate(idx);
    return (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
           cand.pipelineSetAsyncWaitGroups;
  };
  if (enableSharedGlobalPrune != 0) {
    struct GlobalScoredIdx {
      int64_t idx = -1;
      double cumulativeBytes = std::numeric_limits<double>::infinity();
      double maxStageBytes = std::numeric_limits<double>::infinity();
      int64_t maxStageRegs = std::numeric_limits<int64_t>::max();
      int64_t estRegs = std::numeric_limits<int64_t>::max();
      double prio = std::numeric_limits<double>::infinity();
      bool tensorCore = false;
    };
    llvm::SmallVector<GlobalScoredIdx, 64> scored;
    scored.reserve(allIdx.size());
    for (int64_t idx : allIdx) {
      GlobalScoredIdx s;
      s.idx = idx;
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
      const Candidate &cand = getCandidate(idx);
      s.estRegs = cand.estRegsPerThread > 0 ? cand.estRegsPerThread
                                            : std::numeric_limits<int64_t>::max();
      s.prio = getPriority(idx);
      s.tensorCore = cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
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
      if (a.estRegs != b.estRegs)
        return a.estRegs < b.estRegs;
      const bool aPrioFinite = std::isfinite(a.prio);
      const bool bPrioFinite = std::isfinite(b.prio);
      if (aPrioFinite != bPrioFinite)
        return aPrioFinite;
      if (a.prio != b.prio)
        return a.prio < b.prio;
      return a.idx < b.idx;
    });
    const int64_t curN = static_cast<int64_t>(scored.size());
    const int64_t keepN = computeKeepN(curN);
    llvm::SmallVector<int64_t, 64> selectedGlobal;
    selectedGlobal.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i)
      selectedGlobal.push_back(scored[static_cast<size_t>(i)].idx);

    if ((stageClassAnchor != 0 || stageAsyncClassAnchor != 0) && keepN >= 2) {
      auto pickBestByPrio = [&](const auto &pred) -> int64_t {
        int64_t bestIdx = -1;
        bool bestFinite = false;
        double bestPrio = std::numeric_limits<double>::infinity();
        for (const GlobalScoredIdx &s : scored) {
          if (!pred(s.idx))
            continue;
          const bool finite = std::isfinite(s.prio);
          if (bestIdx < 0) {
            bestIdx = s.idx;
            bestFinite = finite;
            bestPrio = s.prio;
            continue;
          }
          if (finite && (!bestFinite || s.prio < bestPrio)) {
            bestIdx = s.idx;
            bestFinite = true;
            bestPrio = s.prio;
          }
        }
        return bestIdx;
      };
      auto containsIdx = [&](int64_t idx) -> bool {
        return llvm::is_contained(selectedGlobal, idx);
      };
      auto injectClassAnchor = [&](int64_t anchorIdx,
                                  const auto &sameClassPredicate) {
        if (anchorIdx < 0 || containsIdx(anchorIdx))
          return;
        for (int64_t pos = keepN - 1; pos >= 0; --pos) {
          int64_t curIdx = selectedGlobal[static_cast<size_t>(pos)];
          if (sameClassPredicate(curIdx))
            continue;
          selectedGlobal[static_cast<size_t>(pos)] = anchorIdx;
          return;
        }
      };
      if (stageClassAnchor != 0) {
        const int64_t bestTc = pickBestByPrio(
            [&](int64_t idx) { return isTensorCoreIdx(idx); });
        const int64_t bestSimt = pickBestByPrio(
            [&](int64_t idx) { return !isTensorCoreIdx(idx); });
        injectClassAnchor(bestTc,
                          [&](int64_t idx) { return isTensorCoreIdx(idx); });
        injectClassAnchor(bestSimt,
                          [&](int64_t idx) { return !isTensorCoreIdx(idx); });
      }
      if (stageAsyncClassAnchor != 0) {
        const int64_t bestTcAsync = pickBestByPrio(
            [&](int64_t idx) { return isTcAsyncIdx(idx); });
        const int64_t bestTcAsyncWait = pickBestByPrio(
            [&](int64_t idx) { return isTcAsyncWaitIdx(idx); });
        injectClassAnchor(bestTcAsync,
                          [&](int64_t idx) { return isTcAsyncIdx(idx); });
        injectClassAnchor(bestTcAsyncWait,
                          [&](int64_t idx) { return isTcAsyncWaitIdx(idx); });
      }
      llvm::SmallDenseSet<int64_t, 64> seen;
      llvm::SmallVector<int64_t, 64> deduped;
      deduped.reserve(static_cast<size_t>(keepN));
      for (int64_t idx : selectedGlobal) {
        if (seen.insert(idx).second)
          deduped.push_back(idx);
      }
      if (deduped.size() < static_cast<size_t>(keepN)) {
        for (const GlobalScoredIdx &s : scored) {
          if (seen.insert(s.idx).second)
            deduped.push_back(s.idx);
          if (deduped.size() >= static_cast<size_t>(keepN))
            break;
        }
      }
      selectedGlobal = std::move(deduped);
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["mode"] = "global_joint";
      f["before"] = curN;
      f["after"] = keepN;
      f["window_count"] = static_cast<int64_t>(windows.size());
      f["topk_cap"] = stageTopK;
      f["keep_ratio"] = stageKeepRatio;
      f["min_keep"] = stageMinKeep;
      f["hard_cap"] = hardKeepCap;
      int64_t tcCount = 0;
      int64_t tcAsyncCount = 0;
      int64_t tcAsyncWaitCount = 0;
      for (int64_t idx : selectedGlobal) {
        if (isTensorCoreIdx(idx))
          ++tcCount;
        if (isTcAsyncIdx(idx))
          ++tcAsyncCount;
        if (isTcAsyncWaitIdx(idx))
          ++tcAsyncWaitCount;
      }
      f["class_anchor"] = stageClassAnchor;
      f["async_class_anchor"] = stageAsyncClassAnchor;
      f["tensorcore_after"] = tcCount;
      f["simt_after"] = static_cast<int64_t>(selectedGlobal.size()) - tcCount;
      f["tc_async_after"] = tcAsyncCount;
      f["tc_async_wait_after"] = tcAsyncWaitCount;
      opts.tracer->event("paper.recursive_shared_stage_prune_global",
                         std::move(f), /*isVerbose=*/true);
    }
    llvm::sort(selectedGlobal);
    selectedGlobal.erase(std::unique(selectedGlobal.begin(), selectedGlobal.end()),
                         selectedGlobal.end());
    if (!selectedGlobal.empty()) {
      if (hardKeepCap > 0 &&
          selectedGlobal.size() > static_cast<size_t>(hardKeepCap)) {
        llvm::sort(selectedGlobal, [&](int64_t a, int64_t b) {
          double pa = getPriority(a);
          double pb = getPriority(b);
          const bool paFinite = std::isfinite(pa);
          const bool pbFinite = std::isfinite(pb);
          if (paFinite != pbFinite)
            return paFinite;
          if (pa != pb)
            return pa < pb;
          return a < b;
        });
        selectedGlobal.resize(static_cast<size_t>(hardKeepCap));
      }
      return selectedGlobal;
    }
  }

  if (incrementalEnable != 0) {
    llvm::SmallDenseSet<int64_t, 64> seedSet;
    for (int64_t i = 0;
         i < std::min<int64_t>(incrementalSeeds,
                               static_cast<int64_t>(allIdx.size()));
         ++i)
      seedSet.insert(allIdx[static_cast<size_t>(i)]);
    int64_t bestPrioIdx = allIdx.front();
    int64_t maxAreaIdx = allIdx.front();
    for (int64_t idx : allIdx) {
      const Candidate &cand = getCandidate(idx);
      const Candidate &maxArea = getCandidate(maxAreaIdx);
      double prioCur = getPriority(idx);
      double prioBest = getPriority(bestPrioIdx);
      if (!std::isfinite(prioBest) ||
          (std::isfinite(prioCur) && prioCur < prioBest))
        bestPrioIdx = idx;
      int64_t areaCur =
          std::max<int64_t>(1, cand.tileM) * std::max<int64_t>(1, cand.tileN);
      int64_t areaBest =
          std::max<int64_t>(1, maxArea.tileM) * std::max<int64_t>(1, maxArea.tileN);
      if (areaCur > areaBest)
        maxAreaIdx = idx;
    }
    seedSet.insert(bestPrioIdx);
    seedSet.insert(maxAreaIdx);
    selected.clear();
    for (int64_t idx : seedSet)
      selected.push_back(idx);
    llvm::sort(selected);
    if (selected.empty())
      selected = allIdx;
  } else {
    selected = allIdx;
  }

  struct StageScoredIdx {
    int64_t idx = -1;
    double stageBytes = std::numeric_limits<double>::infinity();
    double cumulativeBytes = std::numeric_limits<double>::infinity();
    int64_t prefixRegs = std::numeric_limits<int64_t>::max();
    int64_t estRegs = std::numeric_limits<int64_t>::max();
    double prio = std::numeric_limits<double>::infinity();
  };

  for (size_t stageIdx = 0; stageIdx < windows.size(); ++stageIdx) {
    if (selected.size() <= 1)
      break;
    llvm::SmallDenseSet<int64_t, 128> frontierSet;
    for (int64_t idx : selected) {
      if (idx >= 0 && idx < candidateCount)
        frontierSet.insert(idx);
    }
    if (incrementalEnable != 0) {
      auto tryAddPair = [&](int64_t tm, int64_t tn) {
        auto it = pairToIndices.find(keyOf(tm, tn));
        if (it == pairToIndices.end())
          return;
        for (int64_t idx : it->second) {
          if (idx >= 0 && idx < candidateCount)
            frontierSet.insert(idx);
        }
      };
      for (int64_t idx : selected) {
        if (idx < 0 || idx >= candidateCount)
          continue;
        const Candidate &cand = getCandidate(idx);
        auto itM = posM.find(cand.tileM);
        auto itN = posN.find(cand.tileN);
        if (itM == posM.end() || itN == posN.end())
          continue;
        int pm = itM->second;
        int pn = itN->second;
        const int delta[] = {-1, 1};
        for (int d : delta) {
          int nm = pm + d;
          int nn = pn + d;
          if (nm >= 0 && nm < static_cast<int>(valsM.size()))
            tryAddPair(valsM[static_cast<size_t>(nm)], cand.tileN);
          if (nn >= 0 && nn < static_cast<int>(valsN.size()))
            tryAddPair(cand.tileM, valsN[static_cast<size_t>(nn)]);
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
      if (idx < 0 || idx >= candidateCount)
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
      const Candidate &cand = getCandidate(idx);
      s.estRegs = cand.estRegsPerThread > 0 ? cand.estRegsPerThread
                                            : std::numeric_limits<int64_t>::max();
      s.prio = getPriority(idx);
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
      if (a.estRegs != b.estRegs)
        return a.estRegs < b.estRegs;
      const bool aPrioFinite = std::isfinite(a.prio);
      const bool bPrioFinite = std::isfinite(b.prio);
      if (aPrioFinite != bPrioFinite)
        return aPrioFinite;
      if (a.prio != b.prio)
        return a.prio < b.prio;
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
      opts.tracer->event("paper.recursive_shared_stage_prune", std::move(f),
                         /* isVerbose=*/true);
    }

    selected.clear();
    selected.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i)
      selected.push_back(scored[static_cast<size_t>(i)].idx);
  }

  llvm::sort(selected);
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  if (selected.empty())
    return allIdx;
  if (hardKeepCap > 0 && selected.size() > static_cast<size_t>(hardKeepCap)) {
    llvm::sort(selected, [&](int64_t a, int64_t b) {
      double pa = getPriority(a);
      double pb = getPriority(b);
      const bool paFinite = std::isfinite(pa);
      const bool pbFinite = std::isfinite(pb);
      if (paFinite != pbFinite)
        return paFinite;
      if (pa != pb)
        return pa < pb;
      return a < b;
    });
    selected.resize(static_cast<size_t>(hardKeepCap));
  }
  return selected;
}

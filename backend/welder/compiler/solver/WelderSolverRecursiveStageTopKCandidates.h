static llvm::SmallVector<int64_t, 64> selectRecursiveStageTopKCandidateIndices(
    int64_t candidateCount,
    const std::function<const Candidate &(int64_t)> &getCandidate,
    const TileGraph &graph, const PaperSubgraph &sg, const SolveOptions &opts,
    const FootprintInference &inference,
    const PaperScheduleResolvedLevels &scheduleLevels,
    int64_t maxRowReductionExtentForTc) {
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
  const int64_t enableStageTopK = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_STAGE_TOPK_ENABLE", /*default=*/1);
  if (enableStageTopK == 0)
    return selected;

  const auto windows = resolvePaperRecursiveLevelWindows(opts, scheduleLevels);
  if (windows.size() <= 1)
    return selected;

  const int64_t stageTopK =
      getEnvInt64OrDefault("WELDER_RECURSIVE_STAGE_TOPK", /*default=*/24);
  const int64_t stageMinKeep = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_RECURSIVE_STAGE_MIN_KEEP", /*default=*/8));
  double stageKeepRatio = getEnvDoubleOrDefault(
      "WELDER_RECURSIVE_STAGE_KEEP_RATIO", /*default=*/0.75);
  if (!std::isfinite(stageKeepRatio))
    stageKeepRatio = 0.75;
  stageKeepRatio = std::max(0.0, std::min(1.0, stageKeepRatio));
  const int64_t stageClassAnchor = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_STAGE_CLASS_ANCHOR", /*default=*/1);
  const int64_t stageAsyncClassAnchor = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_STAGE_ASYNC_CLASS_ANCHOR", /*default=*/1);
  auto isTensorCoreCand = [&](int64_t idx) -> bool {
    const Candidate &cand = getCandidate(idx);
    return cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
  };

  std::vector<llvm::SmallVector<RecursiveStageMetric, 4>> stageCache(
      static_cast<size_t>(candidateCount));
  for (int64_t i = 0; i < candidateCount; ++i) {
    const Candidate &cand = getCandidate(i);
    const int64_t blockThreads =
        estimateBlockThreadsForCandidate(cand, maxRowReductionExtentForTc);
    (void)estimateRecursiveStageAggregateForCandidate(
        graph, sg, opts, inference, scheduleLevels,
        blockThreads, cand.workgroupPadLastDim,
        cand.workgroupPadLastDimMatmulOnly, cand.workgroupSwizzleXor,
        &stageCache[static_cast<size_t>(i)], /*enforceFeasibility=*/false);
  }
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
    keepN =
        std::max<int64_t>(std::min<int64_t>(curN, stageMinKeep), keepN);
    keepN = std::max<int64_t>(1, std::min<int64_t>(curN, keepN));
    return keepN;
  };
  const int64_t enableGlobalJointStagePrune = getEnvInt64OrDefault(
      "WELDER_RECURSIVE_STAGE_GLOBAL_PRUNE_ENABLE", /*default=*/1);
  if (enableGlobalJointStagePrune != 0) {
    struct GlobalScoredIdx {
      int64_t idx = -1;
      double cumulativeBytes = std::numeric_limits<double>::infinity();
      double maxStageBytes = std::numeric_limits<double>::infinity();
      int64_t maxStageRegs = std::numeric_limits<int64_t>::max();
      int64_t estRegs = std::numeric_limits<int64_t>::max();
      double est = std::numeric_limits<double>::infinity();
      bool tensorCore = false;
    };
    llvm::SmallVector<GlobalScoredIdx, 64> scored;
    scored.reserve(selected.size());
    for (int64_t idx : selected) {
      GlobalScoredIdx s;
      s.idx = idx;
      const auto &metrics = stageCache[static_cast<size_t>(idx)];
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
      s.estRegs = cand.estRegsPerThread > 0
                      ? cand.estRegsPerThread
                      : std::numeric_limits<int64_t>::max();
      s.est = cand.score;
      s.tensorCore = isTensorCoreCand(idx);
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
      const bool aEstFinite = std::isfinite(a.est);
      const bool bEstFinite = std::isfinite(b.est);
      if (aEstFinite != bEstFinite)
        return aEstFinite;
      if (a.est != b.est)
        return a.est < b.est;
      return a.idx < b.idx;
    });
    const int64_t curN = static_cast<int64_t>(scored.size());
    const int64_t keepN = computeKeepN(curN);
    llvm::SmallVector<int64_t, 64> selectedGlobal;
    selectedGlobal.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i)
      selectedGlobal.push_back(scored[static_cast<size_t>(i)].idx);

    if ((stageClassAnchor != 0 || stageAsyncClassAnchor != 0) && keepN >= 2) {
      auto isTcByCand = [](const Candidate &cand) -> bool {
        return cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
      };
      auto isTcAsyncByCand = [&](const Candidate &cand) -> bool {
        return isTcByCand(cand) &&
               (cand.enableAsyncCopy || cand.enableSoftwarePipelining);
      };
      auto isTcAsyncWaitByCand = [&](const Candidate &cand) -> bool {
        return isTcByCand(cand) && cand.pipelineSetAsyncWaitGroups;
      };
      auto pickBestByScore = [&](const auto &pred) -> int64_t {
        int64_t bestIdx = -1;
        bool bestFinite = false;
        double bestScore = std::numeric_limits<double>::infinity();
        for (const GlobalScoredIdx &s : scored) {
          const Candidate &cand = getCandidate(s.idx);
          if (!pred(cand))
            continue;
          const bool finite = std::isfinite(cand.score);
          if (bestIdx < 0) {
            bestIdx = s.idx;
            bestFinite = finite;
            bestScore = cand.score;
            continue;
          }
          if (finite && (!bestFinite || cand.score < bestScore)) {
            bestIdx = s.idx;
            bestFinite = true;
            bestScore = cand.score;
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
        const int64_t bestTc = pickBestByScore(
            [&](const Candidate &cand) { return isTcByCand(cand); });
        const int64_t bestSimt = pickBestByScore(
            [&](const Candidate &cand) { return !isTcByCand(cand); });
        injectClassAnchor(bestTc, [&](int64_t idx) { return isTensorCoreCand(idx); });
        injectClassAnchor(bestSimt,
                          [&](int64_t idx) { return !isTensorCoreCand(idx); });
      }
      if (stageAsyncClassAnchor != 0) {
        const int64_t bestTcAsync = pickBestByScore(
            [&](const Candidate &cand) { return isTcAsyncByCand(cand); });
        const int64_t bestTcAsyncWait = pickBestByScore(
            [&](const Candidate &cand) { return isTcAsyncWaitByCand(cand); });
        injectClassAnchor(bestTcAsync, [&](int64_t idx) {
          const Candidate &cand = getCandidate(idx);
          return isTcAsyncByCand(cand);
        });
        injectClassAnchor(bestTcAsyncWait, [&](int64_t idx) {
          const Candidate &cand = getCandidate(idx);
          return isTcAsyncWaitByCand(cand);
        });
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
      int64_t tcCount = 0;
      for (int64_t idx : selectedGlobal) {
        if (isTensorCoreCand(idx))
          ++tcCount;
      }
      f["class_anchor"] = stageClassAnchor;
      f["tensorcore_after"] = tcCount;
      f["simt_after"] =
          static_cast<int64_t>(selectedGlobal.size()) - tcCount;
      opts.tracer->event("paper.recursive_stage_prune_global", std::move(f),
                         /* isVerbose=*/true);
    }

    selected.clear();
    selected.reserve(selectedGlobal.size());
    for (int64_t idx : selectedGlobal)
      selected.push_back(idx);
    llvm::sort(selected);
    return selected;
  }

  struct StageScoredIdx {
    int64_t idx = -1;
    double stageBytes = std::numeric_limits<double>::infinity();
    double cumulativeBytes = std::numeric_limits<double>::infinity();
    int64_t prefixRegs = std::numeric_limits<int64_t>::max();
    int64_t estRegs = std::numeric_limits<int64_t>::max();
    double est = std::numeric_limits<double>::infinity();
    bool tensorCore = false;
  };

  for (size_t stageIdx = 0; stageIdx < windows.size(); ++stageIdx) {
    if (selected.size() <= 1)
      break;
    llvm::SmallVector<StageScoredIdx, 64> scored;
    scored.reserve(selected.size());
    for (int64_t idx : selected) {
      StageScoredIdx s;
      s.idx = idx;
      const auto &metrics = stageCache[static_cast<size_t>(idx)];
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
      s.est = getCandidate(idx).score;
      s.tensorCore = isTensorCoreCand(idx);
      scored.push_back(std::move(s));
    }
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
      const bool aEstFinite = std::isfinite(a.est);
      const bool bEstFinite = std::isfinite(b.est);
      if (aEstFinite != bEstFinite)
        return aEstFinite;
      if (a.est != b.est)
        return a.est < b.est;
      return a.idx < b.idx;
    });

    int64_t curN = static_cast<int64_t>(scored.size());
    int64_t keepN = computeKeepN(curN);
    if (keepN >= curN)
      continue;

    llvm::SmallVector<int64_t, 64> selectedStage;
    selectedStage.reserve(static_cast<size_t>(keepN));
    for (int64_t i = 0; i < keepN; ++i)
      selectedStage.push_back(scored[static_cast<size_t>(i)].idx);

    if ((stageClassAnchor != 0 || stageAsyncClassAnchor != 0) && keepN >= 2) {
      auto isTcByCand = [](const Candidate &cand) -> bool {
        return cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
      };
      auto isTcAsyncByCand = [&](const Candidate &cand) -> bool {
        return isTcByCand(cand) &&
               (cand.enableAsyncCopy || cand.enableSoftwarePipelining);
      };
      auto isTcAsyncWaitByCand = [&](const Candidate &cand) -> bool {
        return isTcByCand(cand) && cand.pipelineSetAsyncWaitGroups;
      };
      auto pickBestByScore = [&](const auto &pred) -> int64_t {
        int64_t bestIdx = -1;
        bool bestFinite = false;
        double bestScore = std::numeric_limits<double>::infinity();
        for (const StageScoredIdx &s : scored) {
          const Candidate &cand = getCandidate(s.idx);
          if (!pred(cand))
            continue;
          const bool finite = std::isfinite(cand.score);
          if (bestIdx < 0) {
            bestIdx = s.idx;
            bestFinite = finite;
            bestScore = cand.score;
            continue;
          }
          if (finite && (!bestFinite || cand.score < bestScore)) {
            bestIdx = s.idx;
            bestFinite = true;
            bestScore = cand.score;
          }
        }
        return bestIdx;
      };
      auto containsIdx = [&](int64_t idx) -> bool {
        return llvm::is_contained(selectedStage, idx);
      };
      auto injectClassAnchor = [&](int64_t anchorIdx,
                                  const auto &sameClassPredicate) {
        if (anchorIdx < 0 || containsIdx(anchorIdx))
          return;
        for (int64_t pos = keepN - 1; pos >= 0; --pos) {
          int64_t curIdx = selectedStage[static_cast<size_t>(pos)];
          if (sameClassPredicate(curIdx))
            continue;
          selectedStage[static_cast<size_t>(pos)] = anchorIdx;
          return;
        }
      };
      if (stageClassAnchor != 0) {
        const int64_t bestTc = pickBestByScore(
            [&](const Candidate &cand) { return isTcByCand(cand); });
        const int64_t bestSimt = pickBestByScore(
            [&](const Candidate &cand) { return !isTcByCand(cand); });
        injectClassAnchor(bestTc, [&](int64_t idx) { return isTensorCoreCand(idx); });
        injectClassAnchor(bestSimt,
                          [&](int64_t idx) { return !isTensorCoreCand(idx); });
      }
      if (stageAsyncClassAnchor != 0) {
        const int64_t bestTcAsync = pickBestByScore(
            [&](const Candidate &cand) { return isTcAsyncByCand(cand); });
        const int64_t bestTcAsyncWait = pickBestByScore(
            [&](const Candidate &cand) { return isTcAsyncWaitByCand(cand); });
        injectClassAnchor(bestTcAsync, [&](int64_t idx) {
          const Candidate &cand = getCandidate(idx);
          return isTcAsyncByCand(cand);
        });
        injectClassAnchor(bestTcAsyncWait, [&](int64_t idx) {
          const Candidate &cand = getCandidate(idx);
          return isTcAsyncWaitByCand(cand);
        });
      }

      llvm::SmallDenseSet<int64_t, 64> seen;
      llvm::SmallVector<int64_t, 64> deduped;
      deduped.reserve(static_cast<size_t>(keepN));
      for (int64_t idx : selectedStage) {
        if (seen.insert(idx).second)
          deduped.push_back(idx);
      }
      if (deduped.size() < static_cast<size_t>(keepN)) {
        for (const StageScoredIdx &s : scored) {
          if (seen.insert(s.idx).second)
            deduped.push_back(s.idx);
          if (deduped.size() >= static_cast<size_t>(keepN))
            break;
        }
      }
      selectedStage = std::move(deduped);
    }

    if (opts.tracer) {
      llvm::json::Object f;
      f["stage_idx"] = static_cast<int64_t>(stageIdx);
      f["min_level_exclusive"] = static_cast<int64_t>(windows[stageIdx].minLevelExclusive);
      f["max_level_inclusive"] = static_cast<int64_t>(windows[stageIdx].maxLevelInclusive);
      f["before"] = curN;
      f["after"] = keepN;
      f["topk_cap"] = stageTopK;
      f["keep_ratio"] = stageKeepRatio;
      f["min_keep"] = stageMinKeep;
      int64_t tcCount = 0;
      for (int64_t idx : selectedStage) {
        if (isTensorCoreCand(idx))
          ++tcCount;
      }
      f["class_anchor"] = stageClassAnchor;
      f["tensorcore_after"] = tcCount;
      f["simt_after"] =
          static_cast<int64_t>(selectedStage.size()) - tcCount;
      opts.tracer->event("paper.recursive_stage_prune", std::move(f),
                         /* isVerbose=*/true);
    }

    selected.clear();
    selected.reserve(static_cast<size_t>(keepN));
    for (int64_t idx : selectedStage)
      selected.push_back(idx);
  }

  llvm::sort(selected);
  return selected;
}

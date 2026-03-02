//===----------------------------------------------------------------------===//
// Phase A：Grid/Reuse 辅助函数（单算子与全图记账共用）
//===----------------------------------------------------------------------===//

struct GridInfo {
  int64_t blocksTotal = 1;
  int64_t reductionTiles = 1;
  llvm::SmallVector<utils::IteratorType, 8> iterators;
};

static std::optional<GridInfo> computeGridInfo(linalg::LinalgOp op,
                                               const OpTile &tile,
                                               bool requirePerfectTiling) {
  if (!op)
    return std::nullopt;
  if (op.getNumLoops() != static_cast<int64_t>(tile.loopExtents.size()))
    return std::nullopt;

  GridInfo info;
  auto iters = op.getIteratorTypesArray();
  info.iterators.assign(iters.begin(), iters.end());

  llvm::SmallVector<int64_t, 8> staticLoopRanges = op.getStaticLoopRanges();
  if (static_cast<int64_t>(staticLoopRanges.size()) != op.getNumLoops())
    return std::nullopt;

  for (int64_t i = 0; i < op.getNumLoops(); ++i) {
    int64_t full = staticLoopRanges[i];
    int64_t t = tile.loopExtents[i];
    if (full == ShapedType::kDynamic || t <= 0)
      return std::nullopt; // Phase A：要求静态 loop range 才能精确对齐 baseline。

    int64_t tiles = requirePerfectTiling ? (full / t) : ceilDiv(full, t);
    if (info.iterators[i] == utils::IteratorType::parallel)
      info.blocksTotal *= tiles;
    else if (info.iterators[i] == utils::IteratorType::reduction)
      info.reductionTiles *= tiles;
  }
  return info;
}

// 论文/Welder 对齐：DefaultPolicy.DFS_smem_tile（ND 版本，由 2D 泛化）。
//
// 参考实现会用优先队列探索 shared 层输出 tile，并从“冗余工作最少”的
// base tile 开始扩展。该路径在 `opts.autoCandidates` 与 paper schedule
// 同时启用时生效。
static std::vector<Candidate> enumerateSharedTilesPaperDfs2D(
    const TileGraph &graph, const PaperSubgraph &sg, linalg::LinalgOp sinkOp,
    int sinkNodeIdx, const SolveOptions &opts, const FootprintInference &inference,
    const std::vector<std::vector<int64_t>> &reduceTilesByNode) {
  // 泛化后的 ND 版本（对应论文中空间维的 EnumerateSubtiles）。
  std::vector<Candidate> out;
  if (!sinkOp)
    return out;
  if (sinkNodeIdx < 0 || sinkNodeIdx >= static_cast<int>(graph.nodes.size()))
    return out;
  if (!sg.inSet.contains(sinkNodeIdx))
    return out;
  if (sinkOp.getNumParallelLoops() < 1)
    return out;

  llvm::SmallVector<int64_t, 8> ranges = sinkOp.getStaticLoopRanges();
  if (static_cast<int64_t>(ranges.size()) != sinkOp.getNumLoops())
    return out;
  auto iters = sinkOp.getIteratorTypesArray();

  // 按 iterator 顺序记录并行循环的完整范围。
  llvm::SmallVector<int64_t, 8> fulls;
  fulls.reserve(sinkOp.getNumParallelLoops());
  for (int64_t i = 0; i < sinkOp.getNumLoops(); ++i) {
    if (iters[i] != utils::IteratorType::parallel)
      continue;
    int64_t full = ranges[i];
    if (full == ShapedType::kDynamic || full <= 0)
      return out;
    fulls.push_back(full);
  }
  if (fulls.empty() ||
      static_cast<int64_t>(fulls.size()) != sinkOp.getNumParallelLoops())
    return out;
  const int nd = static_cast<int>(fulls.size());

  std::vector<std::vector<int64_t>> rmap = reduceTilesByNode;

  struct TileEval {
    bool valid = false;
    double trafficBytes = 0.0;
    int64_t numWave = 0;
    double workloadPerItem = 0.0;
  };

  auto evalTile = [&](ArrayRef<int64_t> outTile) -> TileEval {
    TileEval ev;
    if (outTile.size() != static_cast<size_t>(nd))
      return ev;

    for (int d = 0; d < nd; ++d) {
      int64_t t = outTile[static_cast<size_t>(d)];
      int64_t full = fulls[static_cast<size_t>(d)];
      if (t <= 0 || t > full)
        return ev;
      if (opts.requirePerfectTiling && (full % t != 0))
        return ev;
    }

    llvm::SmallVector<int64_t, 8> sinkPar(outTile.begin(), outTile.end());
    if (static_cast<int64_t>(sinkPar.size()) != sinkOp.getNumParallelLoops())
      return ev;

    llvm::ArrayRef<int64_t> sinkRed;
    if (sinkNodeIdx >= 0 && static_cast<size_t>(sinkNodeIdx) < rmap.size())
      sinkRed = rmap[sinkNodeIdx];

    auto rootTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
        sinkOp, sinkPar, sinkRed, /*defaultReductionTile=*/0);
    if (!rootTileOpt)
      return ev;

    TileGraph g = graph;
    syncCutFlagFromConnectLevel(g);

    TilePropagationOptions popts;
    popts.defaultReductionTile = 0;
    popts.reductionTilesByNode = &rmap;
    // 论文对齐的鲁棒性策略：冲突时切边（回退到 global），而不是直接失败。
    popts.enableCutEdges = true;
    popts.resetCutEdges = false;

    TilePropagationResult pr =
        propagateTilesBackward(g, sinkNodeIdx, *rootTileOpt, inference, popts);
    if (!pr.success)
      return ev;

    int64_t fpBytes = computeSharedFootprintBestFitPaper(
        g, sg, opts.arch, inference, opts.requirePerfectTiling,
        /* minLevelExclusive=*/0, /*maxLevelInclusive=*/1,
        /* workgroupPadLastDim=*/0,
        /* workgroupPadLastDimMatmulOnly=*/false,
        /* workgroupMultiBufferDepth=*/1,
        /* cand=*/nullptr);
    if (fpBytes < 0 || fpBytes > opts.arch.smemBytes)
      return ev;

    Traffic t = computeGlobalTrafficAssumingFullyFused(
        g, opts.arch, inference, opts.requirePerfectTiling);
    double bytes = t.totalBytes();

    // 寄存器占用启发式：`2 * max(prod(tile(parallel)) * bits / 32)`。
    int64_t elemBits = std::max<int64_t>(1, opts.arch.elementBytes) * 8;
    int64_t worstRegs = 0;
    for (int n : sg.nodes) {
      if (n < 0 || n >= static_cast<int>(g.nodes.size()))
        continue;
      if (!g.nodes[n].hasRequiredTile)
        continue;
      Operation *op0 = g.nodes[n].op;
      auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
      if (!op)
        continue;
      auto it = op.getIteratorTypesArray();
      int64_t parElems = 1;
      for (int i = 0; i < static_cast<int>(it.size()); ++i) {
        if (it[i] != utils::IteratorType::parallel)
          continue;
        int64_t e = g.nodes[n].requiredTile.loopExtents[i];
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
      return ev;

    int64_t blocksBySmem = std::max<int64_t>(1, opts.arch.maxBlocksPerSM);
    if (fpBytes > 0) {
      blocksBySmem =
          std::max<int64_t>(1, getMaxSmemUsageBytes(opts.arch) / fpBytes);
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

    auto gridOpt = computeGridInfo(sinkOp, *rootTileOpt, opts.requirePerfectTiling);
    if (!gridOpt)
      return ev;
    int64_t gridSize = gridOpt->blocksTotal;
    int64_t denom = std::max<int64_t>(
        1, blocksPerSM * std::max<int64_t>(1, opts.arch.numSM));
    int64_t waves = ceilDiv(gridSize, denom);

    // 单元素工作量：`sum(prod(tile(parallel))) / prod(output_tile)`。
    double compute = 0.0;
    for (int n : sg.nodes) {
      if (n < 0 || n >= static_cast<int>(g.nodes.size()))
        continue;
      if (!g.nodes[n].hasRequiredTile)
        continue;
      Operation *op0 = g.nodes[n].op;
      auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
      if (!op)
        continue;
      auto it = op.getIteratorTypesArray();
      double parElems = 1.0;
      for (int i = 0; i < static_cast<int>(it.size()); ++i) {
        if (it[i] != utils::IteratorType::parallel)
          continue;
        int64_t e = g.nodes[n].requiredTile.loopExtents[i];
        if (e <= 0) {
          parElems = 0.0;
          break;
        }
        parElems *= static_cast<double>(e);
      }
      compute += parElems;
    }
    double numItem = 1.0;
    for (int d = 0; d < nd; ++d)
      numItem *= static_cast<double>(outTile[static_cast<size_t>(d)]);
    if (numItem <= 0.0)
      numItem = 1.0;

    ev.valid = true;
    ev.trafficBytes = bytes;
    ev.numWave = std::max<int64_t>(1, waves);
    ev.workloadPerItem = compute / numItem;
    return ev;
  };

  std::vector<int64_t> baseTile(static_cast<size_t>(nd), 1);
  double baseWpi = 0.0;
  {
    TileEval ev = evalTile(baseTile);
    baseWpi = ev.valid ? ev.workloadPerItem : 0.0;
  }
  for (int dim = 0; dim < nd; ++dim) {
    std::vector<int64_t> trial = baseTile;
    trial[static_cast<size_t>(dim)] = fulls[static_cast<size_t>(dim)];
    TileEval ev = evalTile(trial);
    if (ev.valid && (baseWpi == 0.0 || ev.workloadPerItem < baseWpi)) {
      baseTile = std::move(trial);
      baseWpi = ev.workloadPerItem;
    }
  }

  std::vector<llvm::SmallVector<int64_t, 64>> stepsNd;
  stepsNd.resize(static_cast<size_t>(nd));
  for (int d = 0; d < nd; ++d) {
    int64_t full = fulls[static_cast<size_t>(d)];
    llvm::SmallVector<int64_t, 64> steps = getAllFactorsSorted(full);
    if (steps.empty())
      steps.push_back(1);
    auto it = llvm::find(steps, baseTile[static_cast<size_t>(d)]);
    if (it != steps.end()) {
      llvm::SmallVector<int64_t, 64> suffix;
      suffix.assign(it, steps.end());
      steps.swap(suffix);
    }
    int64_t lo = steps.front();
    int64_t hi = steps.back();
    const int64_t extra[] = {2, 4, 8, 16, 32};
    for (int64_t v : extra) {
      if (v <= lo || v >= hi)
        continue;
      if (!llvm::is_contained(steps, v))
        steps.push_back(v);
    }
    llvm::sort(steps);
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    stepsNd[static_cast<size_t>(d)] = std::move(steps);
  }

  struct TileVecHash {
    size_t operator()(const std::vector<int64_t> &v) const noexcept {
      size_t h = 0;
      for (int64_t x : v) {
        size_t k = std::hash<int64_t>{}(x);
        h ^= k + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      }
      return h;
    }
  };

  struct PQItem {
    double prio = 0.0;
    std::vector<int64_t> tile;
  };
  auto prioOf = [&](const TileEval &ev) -> double {
    return (ev.trafficBytes + 1.0) *
           static_cast<double>(std::max<int64_t>(1, ev.numWave));
  };
  struct PQCmp {
    bool operator()(const PQItem &a, const PQItem &b) const {
      return a.prio > b.prio;
    }
  };
  std::priority_queue<PQItem, std::vector<PQItem>, PQCmp> pq;
  std::unordered_map<std::vector<int64_t>, TileEval, TileVecHash> visited;
  visited.reserve(2048);

  auto addTile = [&](std::vector<int64_t> tile) {
    if (visited.find(tile) != visited.end())
      return;
    TileEval ev = evalTile(tile);
    visited.emplace(tile, ev);
    if (!ev.valid)
      return;
    pq.push(PQItem{prioOf(ev), std::move(tile)});
  };

  addTile(baseTile);

  while (!pq.empty() && visited.size() <= 2000) {
    PQItem cur = pq.top();
    pq.pop();
    for (int d = nd - 1; d >= 0; --d) {
      int64_t v = cur.tile[static_cast<size_t>(d)];
      auto &steps = stepsNd[static_cast<size_t>(d)];
      auto it = llvm::find(steps, v);
      if (it == steps.end())
        continue;
      ++it;
      if (it == steps.end())
        continue;
      std::vector<int64_t> next = cur.tile;
      next[static_cast<size_t>(d)] = *it;
      addTile(std::move(next));
    }
  }

  std::vector<std::pair<double, std::vector<int64_t>>> all;
  all.reserve(visited.size());
  for (auto &kv : visited) {
    const TileEval &ev = kv.second;
    if (!ev.valid)
      continue;
    all.push_back({prioOf(ev), kv.first});
  }
  llvm::sort(all, [](const auto &a, const auto &b) { return a.first < b.first; });

  // 论文/Welder 对齐（epilogue）：
  // 若 sink 没有归约循环（例如 matmul 后的逐元素 ReLU），`tileK` 仍需反映
  // 主计算 op 的归约 tile，编译器才能正确应用 K 方向分块。
  int reduceNodeIdx = sinkNodeIdx;
  if (sinkOp.getNumReductionLoops() == 0) {
    // 优先选择 sink 真正上游的归约 op，即使它因 connect-level 切边
    // 没被纳入 `sg.nodes`。
    reduceNodeIdx = -1;
    llvm::SmallVector<int, 16> stack;
    llvm::SmallDenseSet<int, 32> visitedNodes;
    stack.push_back(sinkNodeIdx);
    visitedNodes.insert(sinkNodeIdx);
    while (!stack.empty() && visitedNodes.size() < 32) {
      int cur = stack.pop_back_val();
      if (cur < 0 || cur >= static_cast<int>(graph.nodes.size()))
        continue;
      for (int edgeIdx : graph.nodes[cur].inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        int src = graph.edges[edgeIdx].src;
        if (src < 0 || src >= static_cast<int>(graph.nodes.size()))
          continue;
        if (!visitedNodes.insert(src).second)
          continue;
        Operation *op0 = graph.nodes[src].op;
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

  out.reserve(all.size());
  for (auto &it : all) {
    const double prio = it.first;
    const std::vector<int64_t> &tile = it.second;
    Candidate c;
    c.tileM = !tile.empty() ? tile[0] : 1;
    c.tileN = (tile.size() >= 2) ? tile[1] : 1;
    c.score = prio;

    // 让候选的归约分块与论文中的 rstep_map（由合并访问选定）保持一致。
    // 编译器侧性能测量 harness 会把 `tileK` 当作 K-tile 覆盖值，
    // 因此这里必须与传播/代价评估一致。
    c.tileK = 1;
    if (reduceNodeIdx >= 0 &&
        static_cast<size_t>(reduceNodeIdx) < reduceTilesByNode.size() &&
        !reduceTilesByNode[reduceNodeIdx].empty() &&
        reduceTilesByNode[reduceNodeIdx][0] > 0) {
      c.tileK = reduceTilesByNode[reduceNodeIdx][0];
    }

    // 填充 root 的 `loopTileExtents`，便于 `buildRootParallelExtents2Level`
    // 复用 ND tile（论文 EnumerateSubtiles 路径）。
    llvm::ArrayRef<int64_t> sinkRed;
    if (sinkNodeIdx >= 0 &&
        static_cast<size_t>(sinkNodeIdx) < reduceTilesByNode.size())
      sinkRed = reduceTilesByNode[sinkNodeIdx];
    llvm::SmallVector<int64_t, 8> loopTile;
    loopTile.reserve(sinkOp.getNumLoops());
    int64_t pSeen = 0;
    int64_t rSeen = 0;
    for (int64_t i = 0; i < sinkOp.getNumLoops(); ++i) {
      if (iters[i] == utils::IteratorType::parallel) {
        if (pSeen < 0 || static_cast<size_t>(pSeen) >= tile.size())
          return std::vector<Candidate>();
        loopTile.push_back(tile[static_cast<size_t>(pSeen++)]);
        continue;
      }
      if (iters[i] == utils::IteratorType::reduction) {
        int64_t full = ranges[i];
        int64_t t = full;
        if (rSeen >= 0 && static_cast<size_t>(rSeen) < sinkRed.size() &&
            sinkRed[static_cast<size_t>(rSeen)] > 0) {
          t = sinkRed[static_cast<size_t>(rSeen)];
        }
        ++rSeen;
        if (full != ShapedType::kDynamic && full > 0)
          t = std::min<int64_t>(std::max<int64_t>(1, t), full);
        loopTile.push_back(t);
        continue;
      }
      int64_t full = ranges[i];
      if (full == ShapedType::kDynamic || full <= 0)
        full = 1;
      loopTile.push_back(full);
    }
    c.loopTileExtents.assign(loopTile.begin(), loopTile.end());

    // 填充 grid 尺寸，便于后续阶段即使在 non-strict 模式下
    // 也能计算 waves/低利用率惩罚（ND 版本）。
    int64_t blocksTotal = 1;
    for (int d = 0; d < nd; ++d) {
      int64_t full = fulls[static_cast<size_t>(d)];
      int64_t t = tile[static_cast<size_t>(d)];
      if (full <= 0 || t <= 0) {
        blocksTotal = 1;
        break;
      }
      int64_t tiles = opts.requirePerfectTiling ? (full / t) : ceilDiv(full, t);
      tiles = std::max<int64_t>(1, tiles);
      if (d == 0)
        c.blocksM = tiles;
      else if (d == 1)
        c.blocksN = tiles;
      if (blocksTotal <= (std::numeric_limits<int64_t>::max() / tiles))
        blocksTotal *= tiles;
      else
        blocksTotal = std::numeric_limits<int64_t>::max();
    }
    c.blocksTotal = std::max<int64_t>(1, blocksTotal);

    c.threadTileM = 0;
    c.threadTileN = 0;
    applyFixedCodegenKnobsFromProfile(c, opts);
    out.push_back(std::move(c));
  }
  return out;
}

SolveResult solve(ModuleOp module, const SolveOptions &optsIn) {
  SolveOptions opts = optsIn;
  [[maybe_unused]] auto solveSpan =
      opts.tracer ? opts.tracer->span("solver.solve") : Tracer::Span();
  inferArchElementBytesFromModule(module, opts.arch);
  // Paper schedule 假设采用构造式枚举（EnumerateSubtiles）并具备鲁棒性
  //（冲突 => 切边并计入代价）。开启该模式时，把这些行为设为默认，
  // 避免调用方额外记忆开关组合。
  if (opts.enablePaperSchedule) {
    // Paper schedule 会建模多级内存（global/shared/register）。
    // 默认探索到寄存器层复用决策，除非调用方显式限制层级。
    if (opts.maxConnectLevel < 2)
      opts.maxConnectLevel = 2;
  }
  SolveResult out;
  auto probOpt = analyzeMatmulProblem(module);
  if (!probOpt) {
    out.problem = ProblemSize{kUnknown, kUnknown, kUnknown};
    if (opts.tracer) {
      llvm::json::Object f;
      f["mode"] = "matmul";
      f["result"] = "no_problem";
      opts.tracer->event("solver.solve.result", std::move(f));
    }
    return out;
  }
  out.problem = *probOpt;
  out.detectedConsumerChain = detectMatmulConsumerChain(module);
  if (opts.tracer) {
    llvm::json::Object f;
    f["mode"] = "matmul";
    f["M"] = out.problem.m;
    f["N"] = out.problem.n;
    f["K"] = out.problem.k;
    f["paper"] = opts.enablePaperSchedule;
    f["two_level"] = opts.enableTwoLevelSchedule;
    f["cut_edges"] = opts.enableCutEdges;
    f["profiling"] = opts.profile.enable;
    f["codegen_search"] = opts.codegenSearch.enable;
    f["register_level"] = opts.enableRegisterLevelSchedule;
    f["smem_bytes"] = opts.arch.smemBytes;
    f["num_sm"] = opts.arch.numSM;
    opts.tracer->event("solver.solve.problem", std::move(f));
  }

  // Paper schedule mode：走 Figure 7 的 GraphConnecting/SubGraphTiling。
  if (opts.enablePaperSchedule) {
    auto graphOpt = buildLinalgTileGraph(module);
    if (!graphOpt)
      return out;
    if (!runPaperScheduleSolvePath(*graphOpt, opts, "matmul", out))
      return out;
    return out;
  }

  // 找到第一个 matmul op（作为 indexing_maps footprint 推导的入口）。
  // 这是过渡期做法：先把 footprint inference 应用到 matmul，验证与硬编码一致。
  linalg::LinalgOp matmulOp = nullptr;
  module.walk([&](linalg::MatmulOp mm) {
    if (!matmulOp)
      matmulOp = mm;
  });

  auto candidates =
      enumerateCandidates(out.problem, matmulOp, opts.enableFootprintInference,
                          opts, opts.candidatesMN, opts.candidatesK);

  // 第 6/8 阶段：
  // - Tile propagation（consumer-driven）用于验证候选 tile 的“全图一致性”；
  // - Phase A global traffic（全融合假设）用于对齐旧的 MatMul 专用流量公式。
  //
  // 当前最小版本：
  // - 构建 LinalgOp 的 TileGraph；
  // - 选择一个 sink 节点作为 root（无 outEdges 的 node）；
  // - 对每个候选 (tileM,tileN,tileK)，从 root 反向传播；
  // - 若传播失败且 assumeFusedRelu=true，则丢弃该候选（拒绝这条 fusion 假设）；
  // - 若 enableGlobalTraffic=true，则用全图记账覆盖候选的 traffic/score（Phase A）。
  if ((opts.enableTilePropagation || opts.enableGlobalTraffic ||
       opts.enableCutEdges || opts.enableTwoLevelSchedule) &&
      !candidates.empty()) {
    auto graphOpt = buildLinalgTileGraph(module);
    if (graphOpt && !graphOpt->nodes.empty()) {
      int root = -1;
      for (int i = 0; i < static_cast<int>(graphOpt->nodes.size()); ++i) {
        if (graphOpt->nodes[i].outEdges.empty())
          root = i;
      }

      auto rootOp = (root >= 0)
                        ? dyn_cast_or_null<linalg::LinalgOp>(graphOpt->nodes[root].op)
                        : linalg::LinalgOp();

      if (root >= 0 && rootOp) {
        LinalgIndexingMapsFootprintInference infer;
        std::vector<Candidate> filtered;
        filtered.reserve(candidates.size());
        const int64_t before = static_cast<int64_t>(candidates.size());
        if (opts.tracer) {
          llvm::json::Object f;
          f["mode"] = "matmul";
          f["before"] = before;
          f["tile_propagation"] = opts.enableTilePropagation;
          f["global_traffic"] = opts.enableGlobalTraffic;
          f["cut_edges"] = opts.enableCutEdges;
          f["two_level"] = opts.enableTwoLevelSchedule;
          opts.tracer->event("solver.graph_filter.start", std::move(f));
        }

        for (const Candidate &c : candidates) {
          Candidate outCand = c;
          // Phase 14 (paper alignment, 骨架):
          // 用“Graph Connecting + Tile Propagation”统一 matmul/generic 的处理方式。
          //
          // 对 matmul：rootOp 只有 2 维 parallel，因此 buildRootParallelExtents2Level()
          // 等价于 [tileM, tileN]。
          bool ok = applyGraphConnecting2Level(*graphOpt, root, rootOp, outCand,
                                               opts, infer);
          if (ok) {
            filtered.push_back(outCand);
          } else if (!opts.assumeFusedRelu) {
            filtered.push_back(outCand);
          }
        }

        candidates = std::move(filtered);
        if (opts.tracer) {
          llvm::json::Object f;
          f["mode"] = "matmul";
          f["before"] = before;
          f["after"] = static_cast<int64_t>(candidates.size());
          opts.tracer->event("solver.graph_filter.end", std::move(f));
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            betterCandidateByProfilePriority);
  out.sortedCandidates = std::move(candidates);
  if (opts.tracer && !out.sortedCandidates.empty()) {
    const Candidate &best = out.sortedCandidates.front();
    llvm::json::Object f;
    f["mode"] = "matmul";
    f["result"] = "ok";
    f["ranked"] = static_cast<int64_t>(out.sortedCandidates.size());
    f["best_tm"] = best.tileM;
    f["best_tn"] = best.tileN;
    f["best_tk"] = best.tileK;
    f["best_ttm"] = best.threadTileM;
    f["best_ttn"] = best.threadTileN;
    f["best_score"] = best.score;
    opts.tracer->event("solver.solve.result", std::move(f));
  }
  return out;
}

SolveResult solveGeneric(ModuleOp module, const SolveOptions &optsIn) {
  SolveOptions opts = optsIn;
  [[maybe_unused]] auto solveSpan =
      opts.tracer ? opts.tracer->span("solver.solve_generic") : Tracer::Span();
  inferArchElementBytesFromModule(module, opts.arch);
  SolveResult out;
  out.problem = ProblemSize{kUnknown, kUnknown, kUnknown};

  auto probOpt = analyzeGenericProblem(module);
  if (!probOpt)
    return out;
  if (opts.tracer) {
    llvm::json::Object f;
    f["mode"] = "generic";
    f["op"] = probOpt->getOpName();
    f["loops"] = static_cast<int64_t>(probOpt->loops.size());
    f["paper"] = opts.enablePaperSchedule;
    f["two_level"] = opts.enableTwoLevelSchedule;
    f["cut_edges"] = opts.enableCutEdges;
    f["profiling"] = opts.profile.enable;
    f["codegen_search"] = opts.codegenSearch.enable;
    f["register_level"] = opts.enableRegisterLevelSchedule;
    f["smem_bytes"] = opts.arch.smemBytes;
    f["num_sm"] = opts.arch.numSM;
    opts.tracer->event("solver.solve.problem", std::move(f));
  }

  // Paper schedule mode：走 Figure 7 的 GraphConnecting/SubGraphTiling。
  if (opts.enablePaperSchedule) {
    auto graphOpt = buildLinalgTileGraph(module);
    if (!graphOpt)
      return out;
    if (!runPaperScheduleSolvePath(*graphOpt, opts, "generic", out))
      return out;
    return out;
  }

  auto candidates = enumerateCandidatesGeneric(*probOpt, opts);

  // Phase 9.5：把 Phase 6/8 的“图构建 + 传播 + 全图计费”复用到通用路径上。
  // 目标：generic solver 也能做融合一致性检查，并在需要时用全图 traffic 打分。
  if ((opts.enableTilePropagation || opts.enableGlobalTraffic ||
       opts.enableCutEdges || opts.enableTwoLevelSchedule) &&
      !candidates.empty()) {
    auto graphOpt = buildLinalgTileGraph(module);
    if (graphOpt && !graphOpt->nodes.empty()) {
      int root = -1;
      for (int i = 0; i < static_cast<int>(graphOpt->nodes.size()); ++i) {
        if (graphOpt->nodes[i].outEdges.empty())
          root = i;
      }

      auto rootOp = (root >= 0)
                        ? dyn_cast_or_null<linalg::LinalgOp>(graphOpt->nodes[root].op)
                        : linalg::LinalgOp();

      if (root >= 0 && rootOp) {
        LinalgIndexingMapsFootprintInference infer;
        std::vector<Candidate> filtered;
        filtered.reserve(candidates.size());
        const int64_t before = static_cast<int64_t>(candidates.size());
        if (opts.tracer) {
          llvm::json::Object f;
          f["mode"] = "generic";
          f["before"] = before;
          f["tile_propagation"] = opts.enableTilePropagation;
          f["global_traffic"] = opts.enableGlobalTraffic;
          f["cut_edges"] = opts.enableCutEdges;
          f["two_level"] = opts.enableTwoLevelSchedule;
          opts.tracer->event("solver.graph_filter.start", std::move(f));
        }

        for (const Candidate &c : candidates) {
          Candidate outCand = c;
          bool ok = applyGraphConnecting2Level(*graphOpt, root, rootOp, outCand,
                                               opts, infer);
          if (ok) {
            filtered.push_back(outCand);
          } else if (!opts.assumeFusedRelu) {
            filtered.push_back(outCand);
          }
        }

        candidates = std::move(filtered);
        if (opts.tracer) {
          llvm::json::Object f;
          f["mode"] = "generic";
          f["before"] = before;
          f["after"] = static_cast<int64_t>(candidates.size());
          opts.tracer->event("solver.graph_filter.end", std::move(f));
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            betterCandidateByProfilePriority);
  out.sortedCandidates = std::move(candidates);
  if (opts.tracer && !out.sortedCandidates.empty()) {
    const Candidate &best = out.sortedCandidates.front();
    llvm::json::Object f;
    f["mode"] = "generic";
    f["result"] = "ok";
    f["ranked"] = static_cast<int64_t>(out.sortedCandidates.size());
    f["best_score"] = best.score;
    opts.tracer->event("solver.solve.result", std::move(f));
  }
  return out;
}

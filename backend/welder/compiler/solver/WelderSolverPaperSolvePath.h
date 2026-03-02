static bool runPaperScheduleSolvePath(TileGraph &graph, const SolveOptions &opts,
                                      const char *mode, SolveResult &out) {
  if (graph.nodes.empty())
    return false;

  int sink = -1;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    if (graph.nodes[i].outEdges.empty() &&
        !isTrivialOpFor2LevelFootprint(graph.nodes[i].op)) {
      sink = i;
    }
  }
  if (sink < 0)
    return false;

  auto sinkOp = dyn_cast_or_null<linalg::LinalgOp>(graph.nodes[sink].op);
  if (!sinkOp)
    return false;

  LinalgIndexingMapsFootprintInference infer;
  {
    [[maybe_unused]] auto sp =
        opts.tracer ? opts.tracer->span("paper.graph_connecting")
                    : Tracer::Span();
    (void)graphConnectingPaperGlobalShared(graph, opts, infer,
                                           opts.requirePerfectTiling);
  }

  PaperSubgraph sg =
      extractSubgraphByConnectLevel(graph, sink, /*minLevelExclusive=*/0);

  auto bestList = [&]() {
    [[maybe_unused]] auto sp =
        opts.tracer ? opts.tracer->span("paper.subgraph_tiling")
                    : Tracer::Span();
    return subGraphTilingPaperGlobalShared(graph, sg, sinkOp, sink, opts, infer);
  }();
  if (bestList.empty())
    return false;

  const int64_t effectiveTopK = computeEffectivePaperSolveTopK(graph, sg, opts);
  if (opts.tracer && opts.scheduleTopK > 0 && effectiveTopK > opts.scheduleTopK) {
    llvm::json::Object f;
    f["mode"] = mode ? mode : "unknown";
    f["requested"] = opts.scheduleTopK;
    f["raised_to"] = effectiveTopK;
    opts.tracer->event("paper.mm_sm_f16_min_solver_topk", std::move(f),
                       /* isVerbose=*/true);
  }

  out.sortedCandidates.clear();
  for (const PaperScheduleCandidate &pc : bestList) {
    Candidate c = pc.cand;
    c.estFootprintBytes = pc.sharedFootprintBytes;
    c.traffic = pc.traffic;
    c.score = getPaperCandidateSortLatencyProfileFirst(pc);
    out.sortedCandidates.push_back(std::move(c));
    if (effectiveTopK > 0 &&
        static_cast<int64_t>(out.sortedCandidates.size()) >= effectiveTopK)
      break;
  }

  if (opts.verboseCostModel && !out.sortedCandidates.empty()) {
    const Candidate &best = out.sortedCandidates.front();
    llvm::errs() << "welder-solver(best): tile=(" << best.tileM << ","
                 << best.tileN << "," << best.tileK << ")"
                 << " thread=(" << best.threadTileM << "," << best.threadTileN
                 << ")"
                 << " cost={" << best.cost.toString() << "}\n";
  }

  if (opts.tracer && !out.sortedCandidates.empty()) {
    const Candidate &best = out.sortedCandidates.front();
    llvm::json::Object f;
    f["mode"] = mode ? mode : "unknown";
    f["result"] = "ok";
    f["ranked"] = static_cast<int64_t>(out.sortedCandidates.size());
    if (mode && std::string(mode) == "matmul") {
      f["best_tm"] = best.tileM;
      f["best_tn"] = best.tileN;
      f["best_tk"] = best.tileK;
      f["best_ttm"] = best.threadTileM;
      f["best_ttn"] = best.threadTileN;
    }
    f["best_score"] = best.score;
    if (best.cost.profiledMs.has_value())
      f["best_profiled_ms"] = *best.cost.profiledMs;
    opts.tracer->event("solver.solve.result", std::move(f));
  }

  return true;
}

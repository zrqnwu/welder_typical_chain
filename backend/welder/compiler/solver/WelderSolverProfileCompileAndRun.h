struct CompilerThreadFusionEvidence {
  bool seen = false;
  int64_t pairCount = 0;
  int64_t pairWithOperand = 0;
  int64_t attrPairCount = 0;
  int64_t attrPairWithOperand = 0;
  int64_t registerFuseMinConnectLevel = 0;
};

static int64_t jsonValueToInt64(const llvm::json::Value *value,
                                int64_t defaultValue = 0) {
  if (!value)
    return defaultValue;
  if (auto v = value->getAsInteger())
    return static_cast<int64_t>(*v);
  if (auto v = value->getAsNumber())
    return static_cast<int64_t>(*v);
  if (auto v = value->getAsString()) {
    int64_t parsed = defaultValue;
    if (!v->getAsInteger(10, parsed))
      return parsed;
  }
  return defaultValue;
}

static int64_t jsonObjectLookupInt64(const llvm::json::Object &obj,
                                     llvm::StringRef key,
                                     int64_t defaultValue = 0) {
  return jsonValueToInt64(obj.get(key), defaultValue);
}

static CompilerThreadFusionEvidence
collectCompilerThreadFusionEvidenceFromTraceFile(llvm::StringRef tracePath) {
  CompilerThreadFusionEvidence out;
  if (tracePath.empty())
    return out;
  std::string text = readFileOrEmpty(std::string(tracePath));
  if (text.empty())
    return out;
  llvm::SmallVector<llvm::StringRef, 64> lines;
  llvm::StringRef(text).split(lines, '\n', /*MaxSplit=*/-1,
                              /* KeepEmpty=*/false);
  for (llvm::StringRef line : lines) {
    auto parsed = llvm::json::parse(line);
    if (!parsed)
      continue;
    const auto *obj = parsed->getAsObject();
    if (!obj)
      continue;
    auto eventName = obj->getString("event");
    if (!eventName || *eventName != "compiler.thread_fusion_pairs")
      continue;
    const llvm::json::Object *fields = obj;
    if (const auto *nested = obj->getObject("fields"))
      fields = nested;
    out.seen = true;
    out.pairCount = std::max(
        out.pairCount, jsonObjectLookupInt64(*fields, "pair_count", 0));
    out.pairWithOperand =
        std::max(out.pairWithOperand,
                 jsonObjectLookupInt64(*fields, "pair_with_operand", 0));
    out.attrPairCount = std::max(
        out.attrPairCount,
        jsonObjectLookupInt64(*fields, "thread_fuse_attr_pairs", 0));
    out.attrPairWithOperand =
        std::max(out.attrPairWithOperand, jsonObjectLookupInt64(
                                           * fields,
                                           "thread_fuse_attr_pairs_with_operand",
                                           0));
    out.registerFuseMinConnectLevel = std::max(
        out.registerFuseMinConnectLevel,
        jsonObjectLookupInt64(*fields, "register_fuse_min_connect_level", 0));
  }
  return out;
}

static std::optional<double>
profileSubgraphByCompilingToNvvm(const TileGraph &graph, const PaperSubgraph &sg,
                                 int sinkNodeIdx, const Candidate &cand,
                                 const SolveOptions &opts,
                                 bool *outWasCached) {
  if (outWasCached)
    * outWasCached = false;
  if (!opts.profile.enable)
    return std::nullopt;
  if (opts.profile.compilerToNvvmScript.empty() ||
      opts.profile.profilerBin.empty()) {
    return std::nullopt;
  }
  if (cand.tileM <= 0 || cand.tileN <= 0 || cand.tileK <= 0)
    return std::nullopt;

  const int64_t maxRowReductionExtentForTc =
      computeTcRowReductionExtentForThreadMapping(graph, sg);
  const bool subgraphHasMatmul = subgraphHasMatmulOp(graph, sg);
  const bool graphHasMatmulSoftmax = graphHasMatmulSoftmaxLikePattern(graph);
  const bool isMatmulSoftmaxSubgraph =
      isMatmulSoftmaxLikeSubgraph(graph, sg) ||
      (subgraphHasMatmul && cand.enableMatmulSoftmaxSharedReuseFusion &&
       graphHasMatmulSoftmax);
  if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
      !subgraphHasMatmul) {
    if (opts.tracer) {
      llvm::json::Object f;
      f["sink"] = sinkNodeIdx;
      f["tm"] = cand.tileM;
      f["tn"] = cand.tileN;
      f["tk"] = cand.tileK;
      f["reason"] = "tensorcore_no_matmul_in_subgraph";
      opts.tracer->event("profile.skip", std::move(f), /*isVerbose=*/true);
    }
    return std::nullopt;
  }
  if (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) {
    if (auto tcThreads = computeTensorCoreBlockThreadsForCodegen(
            cand, maxRowReductionExtentForTc);
        !tcThreads || *tcThreads <= 0 || *tcThreads > 1024) {
      if (opts.tracer) {
        llvm::json::Object f;
        f["tm"] = cand.tileM;
        f["tn"] = cand.tileN;
        f["tk"] = cand.tileK;
        f["max_row_reduction_extent"] = maxRowReductionExtentForTc;
        opts.tracer->event("profile.reject_illegal_tc_codegen", std::move(f));
      }
      return std::nullopt;
    }
  }

  static std::unordered_map<std::string, double> cache;
  static std::mutex cacheMu;
  static std::mutex diskMu;
  {
    // 保证一次性磁盘加载串行执行，并确保 `cache` 的并发更新安全。
    std::lock_guard<std::mutex> lockDisk(diskMu);
    std::lock_guard<std::mutex> lockCache(cacheMu);
    loadDiskProfileCacheIfNeeded(opts.profile.cachePath, cache);
  }

  std::string key = buildProfileKeyForSubgraph(graph, sg, sinkNodeIdx, cand);
  // 让性能测量缓存键在多次 solver 调用间保持稳定，同时避免不同测量参数的结果混用。
  key.append("|warmup=");
  key.append(std::to_string(std::max(0, opts.profile.warmup)));
  key.append("|iters=");
  key.append(std::to_string(std::max(1, opts.profile.iters)));
  key.append("|run_all_kernels=");
  key.append(opts.profile.runAllKernels ? "1" : "0");
  // v2：性能测量会为多 kernel 图注入 `--init-ptr`，并自动附加
  // `--fill/--fill-each-iter`，以对齐 host 侧 `linalg.fill` 语义。
  key.append("|prof_fill_v2=1");
  {
    std::lock_guard<std::mutex> lock(cacheMu);
    if (auto it = cache.find(key); it != cache.end()) {
      if (outWasCached)
        * outWasCached = true;
      if (opts.tracer) {
        llvm::json::Object f;
        f["sink"] = sinkNodeIdx;
        f["tm"] = cand.tileM;
        f["tn"] = cand.tileN;
        f["tk"] = cand.tileK;
        f["ttm"] = cand.threadTileM;
        f["ttn"] = cand.threadTileN;
        f["avg_ms"] = it->second;
        opts.tracer->event("profile.cache_hit", std::move(f), /*isVerbose=*/true);
      }
      return it->second;
    }
  }

  // 可选的进程级性能测量编译预算（仅统计 cache miss）。
  // 预算耗尽后跳过后续编译尝试，但允许调用方回退到模型估计排序，
  // 以保持 solver 流程继续推进。
  const int64_t maxProfileCompilesGlobal = std::max<int64_t>(
      0, getEnvInt64OrDefault("WELDER_PROFILE_MAX_COMPILES_GLOBAL",
                              /*default=*/0));
  if (maxProfileCompilesGlobal > 0) {
    static std::atomic<int64_t> profileCompileGlobalCounter{0};
    const int64_t compileTicket =
        profileCompileGlobalCounter.fetch_add(1, std::memory_order_relaxed);
    if (compileTicket >= maxProfileCompilesGlobal) {
      if (outWasCached)
        * outWasCached = true;
      if (opts.tracer) {
        static std::atomic<int64_t> skipEventBudget{8};
        int64_t oldBudget = skipEventBudget.fetch_sub(1, std::memory_order_relaxed);
        if (oldBudget > 0) {
          llvm::json::Object f;
          f["cap"] = maxProfileCompilesGlobal;
          f["attempted"] = compileTicket + 1;
          opts.tracer->event("profile.compile_global_cap_skip", std::move(f),
                             /* isVerbose=*/true);
        }
      }
      return std::nullopt;
    }
  }

  if (opts.tracer) {
    llvm::json::Object f;
    f["sink"] = sinkNodeIdx;
    f["tm"] = cand.tileM;
    f["tn"] = cand.tileN;
    f["tk"] = cand.tileK;
    f["ttm"] = cand.threadTileM;
    f["ttn"] = cand.threadTileN;
    f["ac"] = cand.enableAsyncCopy;
    f["pipe"] = cand.enableSoftwarePipelining;
    f["rr_reuse"] = cand.enableRowReductionChainReuseFusion;
    f["rr_promo"] = cand.enableRowReductionInputPromotion;
    f["rr_promo_vec"] = cand.enableRowReductionInputPromotionVectorize;
    f["rr_warp"] = cand.enableRowReductionWarp;
    f["rr_vec"] = cand.enableRowReductionVectorize;
    f["rr_vec_w"] = cand.rowReductionVectorWidth;
    f["rr_tx"] = cand.rowReductionThreadsX;
    f["rr_relax"] = cand.enableRowReductionRelaxBarriers;
    f["rr_skipc"] = cand.enableRowReductionSkipCombineBarrier;
    f["rr_in_vec"] = cand.rowReductionInputVectorWidth;
    f["rr_comb_vec"] = cand.enableRowReductionCombineVectorize;
    f["mm_sm_reuse"] = cand.enableMatmulSoftmaxSharedReuseFusion;
    f["tc_f16"] = cand.enableTensorCoreF16;
    f["tc_tf32"] = cand.enableTensorCoreTf32;
    opts.tracer->event("profile.cache_miss", std::move(f), /*isVerbose=*/true);
  }

  // 复制一份模块，保证并行性能测量时不会原地修改共享 IR。
  Operation *sinkOpRaw = graph.nodes[sinkNodeIdx].op;
  if (!sinkOpRaw)
    return std::nullopt;
  ModuleOp module = sinkOpRaw->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  OwningOpRef<ModuleOp> clonedModule(cast<ModuleOp>(module->clone()));

  Builder b(clonedModule->getContext());
  auto kernelIdAttr = b.getI32IntegerAttr(0);

  // 为克隆模块建立 id->op 映射。
  llvm::DenseMap<int64_t, Operation *> idToOp;
  clonedModule->walk([&](linalg::LinalgOp op) {
    Operation *op0 = op.getOperation();
    if (!op0)
      return;
    // 清理克隆模块中可能残留的属性。
    op0->removeAttr("welder.kernel_id");
    op0->removeAttr("welder.kernel_producer");
    op0->removeAttr("welder.kernel_root");
    if (auto idAttr = op0->getAttrOfType<IntegerAttr>("welder.node_id")) {
      idToOp[idAttr.getInt()] = op0;
    }
  });

  Operation *sinkOp = nullptr;
  if (sinkNodeIdx >= 0)
    sinkOp = idToOp.lookup(static_cast<int64_t>(sinkNodeIdx));
  if (!sinkOp)
    return std::nullopt;

  // 性能测量阶段按“每个子图一个 kernel”打标：id=0。
  for (int n : sg.nodes) {
    if (n < 0)
      continue;
    Operation *op0 = idToOp.lookup(static_cast<int64_t>(n));
    if (!op0)
      continue;
    if (op0 == sinkOp)
      continue; // 根节点不应加入 producersMatch 集合。
    if (isTrivialOpFor2LevelFootprint(op0))
      continue;
    if (!isa<linalg::LinalgOp>(op0))
      continue;
    op0->setAttr("welder.kernel_id", kernelIdAttr);
    // 与编译器的切边多 kernel 打标保持一致：额外打上 producer-only 标记，
    // 让 transform 库能把这些 op 融到 kernel root，同时避免误选 root 本身。
    op0->setAttr("welder.kernel_producer", kernelIdAttr);
  }
  sinkOp->setAttr("welder.kernel_root", kernelIdAttr);
  // 与编译器切边路径保持一致：kernel root 也带上 `welder.kernel_id`，
  // 便于后续 post-bufferize 变换统一定位该 kernel 内的 op。
  sinkOp->setAttr("welder.kernel_id", kernelIdAttr);

  // 论文/Welder 对齐（MatMul epilogue）：
  // 当假设 MatMul->(bias)->ReLU 融合时，tile-graph 的 connect level
  // 仍可能把上游 producer 链排除在 `sg.nodes` 外。性能测量时需要补打
  // 上游 producer 标记，确保编译器能把它们一并融合到 kernel root，
  // profiler 测到的是完整计算 kernel，而非仅逐元素 epilogue。
  if (opts.assumeFusedRelu) {
    auto findUpstreamLinalgProducer =
        [&](Value v) -> Operation * {
      Value cur = v;
      for (int hop = 0; hop < 8; ++hop) {
        Operation *def = cur.getDefiningOp();
        if (!def)
          return nullptr;
        if (isa<linalg::LinalgOp>(def))
          return def;
        if (isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
                tensor::ExpandShapeOp>(def)) {
          if (def->getNumOperands() == 0)
            return nullptr;
          cur = def->getOperand(0);
          continue;
        }
        return nullptr;
      }
      return nullptr;
    };

    if (isa<linalg::GenericOp>(sinkOp)) {
      llvm::SmallVector<Operation *, 8> queue;
      llvm::SmallPtrSet<Operation *, 16> visited;
      queue.push_back(sinkOp);
      visited.insert(sinkOp);

      while (!queue.empty()) {
        Operation *cur = queue.pop_back_val();
        if (!cur)
          continue;
        for (Value operand : cur->getOperands()) {
          Operation *prod = findUpstreamLinalgProducer(operand);
          if (!prod)
            continue;
          if (prod == sinkOp)
            continue;
          if (isTrivialOpFor2LevelFootprint(prod))
            continue;
          if (!isa<linalg::LinalgOp>(prod))
            continue;
          if (!visited.insert(prod).second)
            continue;

          prod->setAttr("welder.kernel_id", kernelIdAttr);
          prod->setAttr("welder.kernel_producer", kernelIdAttr);
          queue.push_back(prod);
          if (visited.size() >= 16)
            break;
        }
        if (visited.size() >= 16)
          break;
      }
    }
  }

  // 寄存器层融合：保持性能测量侧的边标记与编译器路径一致。
  // 对递归调度（max_connect_level>2），仅融合深于递归内层边界的边。
  int64_t threadFuseMarks = 0;
  int64_t threadFuseOperandMarks = 0;
  int64_t promotedSharedEdges = 0;
  int registerFuseMinConnectLevel = 0;
  bool promoteSharedEdgesForRegisterFuse = false;
  if (opts.maxConnectLevel >= 2) {
    registerFuseMinConnectLevel = kConnectLevelRegister;
    if (opts.maxConnectLevel > kConnectLevelRegister) {
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
      registerFuseMinConnectLevel =
          std::max<int>(kConnectLevelRegister,
                        recursiveInnerMinLevelExclusive + 1);
    }
    // 对 connect_level>=2，默认启用更严格的寄存器融合证据约束：
    // shared 边提升必须显式开启（opt-in）。
    const int promoteSharedDefault = 0;
    promoteSharedEdgesForRegisterFuse =
        getEnvInt64OrDefault("WELDER_CONNECT2_PROMOTE_SHARED_EDGES",
                             promoteSharedDefault) != 0;
    const int registerFuseMinLevelExclusive =
        std::max<int>(kConnectLevelGlobal, registerFuseMinConnectLevel - 1);
    for (const TileGraphEdge &e : graph.edges) {
      const bool eligibleAtRegisterLevel = isRegisterFuseEligibleEdge(
          graph, sg, e, registerFuseMinLevelExclusive);
      const bool eligibleAtSharedLevel =
          promoteSharedEdgesForRegisterFuse &&
          isRegisterFuseEligibleEdge(graph, sg, e, kConnectLevelGlobal);
      if (!eligibleAtRegisterLevel && !eligibleAtSharedLevel)
        continue;

      Operation *srcOp = idToOp.lookup(static_cast<int64_t>(e.src));
      if (!srcOp)
        continue;

      if (!srcOp->hasAttr("welder.thread_fuse_into")) {
        srcOp->setAttr("welder.thread_fuse_into",
                       b.getI64IntegerAttr(static_cast<int64_t>(e.dst)));
        if (e.dstOperand >= 0) {
          srcOp->setAttr("welder.thread_fuse_into_operand",
                         b.getI64IntegerAttr(static_cast<int64_t>(e.dstOperand)));
          ++threadFuseOperandMarks;
        }
        ++threadFuseMarks;
        if (!eligibleAtRegisterLevel && eligibleAtSharedLevel)
          ++promotedSharedEdges;
      }
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["max_connect_level"] = std::max<int64_t>(0, opts.maxConnectLevel);
      f["register_fuse_min_connect_level"] =
          static_cast<int64_t>(registerFuseMinConnectLevel);
      f["promote_shared_edges"] = promoteSharedEdgesForRegisterFuse;
      f["thread_fuse_marks"] = threadFuseMarks;
      f["thread_fuse_operand_marks"] = threadFuseOperandMarks;
      f["promoted_shared_edges"] = promotedSharedEdges;
      opts.tracer->event("profile.register_fuse_marks", std::move(f),
                         /* isVerbose=*/true);
    }
  }

  // 在编译前前置执行 connect_level=2 的寄存器融合合法性检查，
  // 避免把编译/重试预算浪费在必然会被 connect2 严格门禁拒绝的候选上。
  const bool useCandKnobsForConnect2Evidence = opts.codegenSearch.enable;
  const bool effRowReductionReuseFusionForConnect2Evidence =
      useCandKnobsForConnect2Evidence
          ? cand.enableRowReductionChainReuseFusion
          : opts.profile.enableRowReductionChainReuseFusion;
  const bool effMatmulSoftmaxSharedReuseFusionForConnect2Evidence =
      useCandKnobsForConnect2Evidence
          ? cand.enableMatmulSoftmaxSharedReuseFusion
          : opts.profile.enableMatmulSoftmaxSharedReuseFusion;
  const bool enableRegisterLevelCodegenForConnect2Evidence =
      (opts.enableRegisterLevelSchedule || opts.maxConnectLevel >= 2 ||
       cand.threadTileM > 0 || cand.threadTileN > 0);
  const int64_t maxConnectLevelForConnect2Evidence =
      std::max<int64_t>(0, opts.maxConnectLevel);
  const bool requireConnect2RegisterFuseEvidencePre =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_REQUIRE_CONNECT2_REG_FUSE_EVIDENCE",
          (isMatmulSoftmaxSubgraph &&
           maxConnectLevelForConnect2Evidence >= 2)
              ? 1
              : 0) != 0;
  const bool requireConnect2ThreadFuseOperandMarksPre =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_THREAD_FUSE_OPERAND_MARKS",
          0) != 0;
  const bool requireConnect2RegisterMinLevelPre =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_REGISTER_MIN_LEVEL",
          maxConnectLevelForConnect2Evidence >= 2 ? 1 : 0) != 0;
  const bool connect2ReuseExpectedPre =
      maxConnectLevelForConnect2Evidence >= 2 &&
      enableRegisterLevelCodegenForConnect2Evidence &&
      (effMatmulSoftmaxSharedReuseFusionForConnect2Evidence ||
       effRowReductionReuseFusionForConnect2Evidence ||
       isMatmulSoftmaxSubgraph);
  if (requireConnect2RegisterFuseEvidencePre && connect2ReuseExpectedPre) {
    bool hasSolverConnect2Evidence = threadFuseMarks > 0;
    if (requireConnect2ThreadFuseOperandMarksPre)
      hasSolverConnect2Evidence =
          hasSolverConnect2Evidence && (threadFuseOperandMarks > 0);
    if (requireConnect2RegisterMinLevelPre)
      hasSolverConnect2Evidence =
          hasSolverConnect2Evidence && (registerFuseMinConnectLevel >= 2);
    if (!hasSolverConnect2Evidence) {
      const double rejectMs = std::max(
          1.0, getEnvDoubleOrDefault("WELDER_PROFILE_FEATURE_REJECT_MS",
                                     /*default=*/1.0e9));
      if (opts.tracer) {
        llvm::json::Object f;
        f["reason"] = "precompile_missing_connect2_register_fuse_evidence";
        f["max_connect_level"] = maxConnectLevelForConnect2Evidence;
        f["enable_register_level_codegen"] =
            enableRegisterLevelCodegenForConnect2Evidence;
        f["thread_fuse_marks"] = threadFuseMarks;
        f["thread_fuse_operand_marks"] = threadFuseOperandMarks;
        f["register_fuse_min_connect_level"] =
            static_cast<int64_t>(registerFuseMinConnectLevel);
        f["require_operand_marks"] = requireConnect2ThreadFuseOperandMarksPre;
        f["require_register_min_level"] = requireConnect2RegisterMinLevelPre;
        f["reuse_expected"] = connect2ReuseExpectedPre;
        f["reject_ms"] = rejectMs;
        opts.tracer->event("profile.reject_connect2_register_fuse_evidence",
                           std::move(f), /*isVerbose=*/true);
      }
      {
        std::lock_guard<std::mutex> lock(cacheMu);
        cache[key] = rejectMs;
      }
      {
        std::lock_guard<std::mutex> lock(diskMu);
        appendDiskProfileCache(opts.profile.cachePath, key, rejectMs);
      }
      return std::optional<double>(rejectMs);
    }
  }

  // 论文/Welder 对齐：从 sink op 的 indexing map 推导每个 kernel 的
  // block 映射顺序提示（例如转置类 op 往往受益于交换 x/y）。
  if (auto sinkLinalg = dyn_cast_or_null<linalg::LinalgOp>(sinkOp)) {
    if (auto hint = inferSwapXYHintForLinalgOp(sinkLinalg)) {
      if (*hint) {
        sinkOp->setAttr("welder.swap_xy", b.getBoolAttr(true));
      } else {
        // 默认即“不交换”；不显式写 `false` 属性，保持 IR 干净。
        sinkOp->removeAttr("welder.swap_xy");
      }
    }
  }

  // 为外部编译脚本落盘临时 MLIR 文件。
  auto dirOpt = makeTempDir("welder_profile");
  if (!dirOpt) {
    return std::nullopt;
  }
  std::string dir = *dirOpt;
  TempDirGuard tmp{dir, opts.profile.verbose};
  std::string inputPath = dir + "/input.mlir";
  std::string outDir = dir + "/out";
  std::string compilerTracePath = dir + "/compiler.trace.jsonl";
  (void)llvm::sys::fs::create_directories(outDir);

  std::string writeErr;
  bool wrote = writeModuleToFile(*clonedModule, inputPath, &writeErr);

  if (!wrote) {
    if (opts.profile.verbose) {
      llvm::errs() << "profile: failed to write temp MLIR: " << writeErr
                   << "\n";
    }
    return std::nullopt;
  }

  // 尽力为 profiler 提供具体的 memref 描述符 i64 值。
  // 某些 NVVM runnable 包装会把原始 tensor 参数降成 `%argN` 形式的 i64
  // token（offset/sizes/strides）。profiler 直接执行 kernel，因此需要这些值
  // 来推断分配大小。
  std::vector<std::pair<std::string, int64_t>> profilerI64Overrides;
  if (auto ovOpt = inferProfilerI64OverridesFromMainFunc(*clonedModule)) {
    profilerI64Overrides = std::move(*ovOpt);
  } else if (opts.tracer) {
    llvm::json::Object f;
    f["note"] = "cannot_infer_profiler_i64_overrides";
    opts.tracer->event("profile.info", std::move(f), /*isVerbose=*/true);
  }

  std::vector<std::string> profilerInitPtrSyms;
  if (auto initOpt = inferProfilerInitPtrSymsFromMainFunc(*clonedModule)) {
    profilerInitPtrSyms = std::move(*initOpt);
  } else if (opts.tracer) {
    llvm::json::Object f;
    f["note"] = "cannot_infer_profiler_init_ptrs";
    opts.tracer->event("profile.info", std::move(f), /*isVerbose=*/true);
  }
  bool hasMaxReduction = moduleHasMaximumFReduction(*clonedModule);

  // 编译到 NVVM ISA MLIR。
  std::string nvvmPath = outDir + "/05.out.nvvm.runnable.mlir";
  std::string compileCmd;
  compileCmd.reserve(512);
  bool useCandKnobs = opts.codegenSearch.enable;
  bool effEnableAsyncCopy =
      useCandKnobs ? cand.enableAsyncCopy : opts.profile.enableAsyncCopy;
  bool effAsyncBypassL1 =
      useCandKnobs ? cand.asyncBypassL1 : opts.profile.asyncBypassL1;
  bool effEnableSoftwarePipelining =
      useCandKnobs ? cand.enableSoftwarePipelining
                   : opts.profile.enableSoftwarePipelining;
  int64_t effPipelineDepth =
      useCandKnobs ? cand.pipelineDepth : opts.profile.pipelineDepth;
  bool effPipelinePeelEpilogue =
      useCandKnobs ? cand.pipelinePeelEpilogue : opts.profile.pipelinePeelEpilogue;
  bool effPipelineSetAsyncWaitGroups =
      useCandKnobs ? cand.pipelineSetAsyncWaitGroups
                   : opts.profile.pipelineSetAsyncWaitGroups;

  int64_t multiDepth =
      useCandKnobs ? std::max<int64_t>(1, cand.workgroupMultiBufferDepth)
                   : std::max<int64_t>(1, opts.profile.workgroupMultiBufferDepth);
  int64_t padLastDim =
      useCandKnobs ? std::max<int64_t>(0, cand.workgroupPadLastDim)
                   : std::max<int64_t>(0, opts.profile.workgroupPadLastDim);
  bool padMatmulOnly =
      useCandKnobs ? cand.workgroupPadLastDimMatmulOnly
                   : opts.profile.workgroupPadLastDimMatmulOnly;
  if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) && padLastDim == 0) {
    // 论文/Welder 对齐：TCPolicy 的 stride 偏移默认使用 8。
    padLastDim = 8;
  }
  if (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) {
    // 参考 TCPolicy 只把 stride 偏移应用到 matmul 输入的 shared tile，
    // 不作用于无关的 shared buffer。
    padMatmulOnly = true;
  }
  int64_t swizzleXor =
      useCandKnobs ? std::max<int64_t>(0, cand.workgroupSwizzleXor)
                   : std::max<int64_t>(0, opts.profile.workgroupSwizzleXor);
  int64_t blockRasterizeXor =
      useCandKnobs ? std::max<int64_t>(0, cand.blockRasterizeXor) : 0;
  int effRasterizeMode =
      useCandKnobs ? std::max(0, cand.blockRasterizeMode)
                   : std::max(0, opts.profile.blockRasterizeMode);
  int effRasterizePanel =
      useCandKnobs ? std::max(0, cand.blockRasterizePanelWidth)
                   : std::max(0, opts.profile.blockRasterizePanelWidth);
  bool effRowReductionReuseFusion =
      useCandKnobs ? cand.enableRowReductionChainReuseFusion
                   : opts.profile.enableRowReductionChainReuseFusion;
  bool effRowReductionInputPromotion =
      useCandKnobs ? cand.enableRowReductionInputPromotion
                   : opts.profile.enableRowReductionInputPromotion;
  bool effRowReductionInputPromotionVectorize =
      effRowReductionInputPromotion &&
      (useCandKnobs ? cand.enableRowReductionInputPromotionVectorize
                   : opts.profile.enableRowReductionInputPromotionVectorize);
  bool effRowReductionWarp =
      useCandKnobs ? cand.enableRowReductionWarp
                   : opts.profile.enableRowReductionWarp;
  bool effRowReductionVectorize =
      useCandKnobs ? cand.enableRowReductionVectorize
                   : opts.profile.enableRowReductionVectorize;
  int64_t effRowReductionVectorWidth =
      useCandKnobs ? cand.rowReductionVectorWidth
                   : opts.profile.rowReductionVectorWidth;
  int64_t effRowReductionThreadsX =
      useCandKnobs ? cand.rowReductionThreadsX
                   : opts.profile.rowReductionThreadsX;
  bool effRowReductionRelaxBarriers =
      useCandKnobs ? cand.enableRowReductionRelaxBarriers
                   : opts.profile.enableRowReductionRelaxBarriers;
  bool effRowReductionSkipCombineBarrier =
      useCandKnobs ? cand.enableRowReductionSkipCombineBarrier
                   : opts.profile.enableRowReductionSkipCombineBarrier;
  int64_t effRowReductionInputVectorWidth =
      useCandKnobs ? cand.rowReductionInputVectorWidth
                   : opts.profile.rowReductionInputVectorWidth;
  bool effRowReductionCombineVectorize =
      useCandKnobs ? cand.enableRowReductionCombineVectorize
                   : opts.profile.enableRowReductionCombineVectorize;
  bool effMatmulSoftmaxSharedReuseFusion =
      useCandKnobs ? cand.enableMatmulSoftmaxSharedReuseFusion
                   : opts.profile.enableMatmulSoftmaxSharedReuseFusion;

  const bool nonMmSmSubgraphInMmSmGraph =
      graphHasMatmulSoftmax && !isMatmulSoftmaxSubgraph;
  const bool conservativeNonMmSmCodegen =
      nonMmSmSubgraphInMmSmGraph &&
      (getEnvInt64OrDefault("WELDER_PROFILE_NON_MM_SM_CONSERVATIVE_CODEGEN", 1) !=
       0);
  if (conservativeNonMmSmCodegen) {
    const bool disableAsync =
        getEnvInt64OrDefault("WELDER_PROFILE_NON_MM_SM_DISABLE_ASYNC", 1) != 0;
    const bool disablePipeline =
        getEnvInt64OrDefault("WELDER_PROFILE_NON_MM_SM_DISABLE_PIPELINE", 1) !=
        0;
    const bool disableRowReuse = getEnvInt64OrDefault(
                                     "WELDER_PROFILE_NON_MM_SM_DISABLE_ROW_REUSE",
                                     1) != 0;
    const bool disableMmSmReuse = getEnvInt64OrDefault(
                                      "WELDER_PROFILE_NON_MM_SM_DISABLE_MM_SM_REUSE",
                                      1) != 0;
    if (disableAsync)
      effEnableAsyncCopy = false;
    if (disablePipeline) {
      effEnableSoftwarePipelining = false;
      effPipelineSetAsyncWaitGroups = false;
    }
    if (disableRowReuse)
      effRowReductionReuseFusion = false;
    if (disableMmSmReuse)
      effMatmulSoftmaxSharedReuseFusion = false;
    effRowReductionInputPromotionVectorize = false;
    effRowReductionWarp = false;
    effRowReductionVectorize = false;
    effRowReductionVectorWidth = 0;
    effRowReductionThreadsX = 0;
    effRowReductionRelaxBarriers = false;
    effRowReductionSkipCombineBarrier = false;
    effRowReductionInputVectorWidth = 0;
    effRowReductionCombineVectorize = false;
    if (opts.tracer) {
      llvm::json::Object f;
      f["sink"] = sinkNodeIdx;
      f["disable_async"] = disableAsync;
      f["disable_pipeline"] = disablePipeline;
      f["disable_row_reuse"] = disableRowReuse;
      f["disable_mm_sm_reuse"] = disableMmSmReuse;
      opts.tracer->event("profile.non_mm_sm_conservative_codegen", std::move(f),
                         /* isVerbose=*/true);
    }
  }

  if (multiDepth > 1) {
    compileCmd.append("WORKGROUP_MULTIBUFFER_DEPTH=");
    compileCmd.append(std::to_string(std::max<int64_t>(1, multiDepth)));
    compileCmd.push_back(' ');
  }
  if (padLastDim != 0) {
    compileCmd.append("WORKGROUP_PAD_LAST_DIM=");
    compileCmd.append(std::to_string(padLastDim));
    compileCmd.push_back(' ');
    if (padMatmulOnly)
      compileCmd.append("WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY=1 ");
  }
  if (swizzleXor != 0) {
    compileCmd.append("WORKGROUP_SWIZZLE_XOR=");
    compileCmd.append(std::to_string(swizzleXor));
    compileCmd.push_back(' ');
  }
  if (blockRasterizeXor != 0) {
    compileCmd.append("BLOCK_RASTERIZE_XOR=");
    compileCmd.append(std::to_string(blockRasterizeXor));
    compileCmd.push_back(' ');
  }
  if (effRasterizeMode != 0) {
    compileCmd.append("BLOCK_RASTERIZE_MODE=");
    compileCmd.append(std::to_string(effRasterizeMode));
    compileCmd.push_back(' ');
  }
  if (effRasterizePanel > 0) {
    compileCmd.append("BLOCK_RASTERIZE_PANEL_WIDTH=");
    compileCmd.append(std::to_string(effRasterizePanel));
    compileCmd.push_back(' ');
  }
  if (effEnableSoftwarePipelining) {
    compileCmd.append("ENABLE_SOFTWARE_PIPELINING=1 ");
    compileCmd.append("PIPELINE_DEPTH=");
    compileCmd.append(std::to_string(std::max<int64_t>(2, effPipelineDepth)));
    compileCmd.push_back(' ');
    compileCmd.append("PIPELINE_PEEL_EPILOGUE=");
    compileCmd.append(effPipelinePeelEpilogue ? "1" : "0");
    compileCmd.push_back(' ');
    if (effPipelineSetAsyncWaitGroups)
      compileCmd.append("PIPELINE_SET_ASYNC_WAIT_GROUPS=1 ");
  }
  compileCmd.append("OUT_DIR=");
  compileCmd.append(shellEscapeSingleQuotes(outDir));
  compileCmd.append(" bash ");
  compileCmd.append(shellEscapeSingleQuotes(opts.profile.compilerToNvvmScript));
  compileCmd.push_back(' ');
  compileCmd.append(shellEscapeSingleQuotes(inputPath));
  compileCmd.append(" --enable-generic-problem");
  compileCmd.append(" --codegen-from-kernel-attrs");
  const bool enableRegisterLevelCodegen =
      (opts.enableRegisterLevelSchedule || opts.maxConnectLevel >= 2 ||
       cand.threadTileM > 0 || cand.threadTileN > 0);
  const int64_t maxConnectLevelForCodegen =
      std::max<int64_t>(0, opts.maxConnectLevel);
  if (enableRegisterLevelCodegen) {
    compileCmd.append(" --enable-register-level-schedule");
  }
  compileCmd.append(" --max-connect-level=");
  compileCmd.append(std::to_string(maxConnectLevelForCodegen));
  compileCmd.append(" --force-tile-m ");
  compileCmd.append(std::to_string(cand.tileM));
  compileCmd.append(" --force-tile-n ");
  compileCmd.append(std::to_string(cand.tileN));
  compileCmd.append(" --force-tile-k ");
  compileCmd.append(std::to_string(cand.tileK));
  compileCmd.append(" --trace-file ");
  compileCmd.append(shellEscapeSingleQuotes(compilerTracePath));

  // 寄存器层 tile（thread tile）：确保编译器能为当前 tile 推导出合法 blockDim。
  // 若候选未显式给出 thread tile，则选择保守回退值。
  auto pickThreadTile = [&](int64_t tile, int64_t prefer) -> int64_t {
    if (prefer > 0 && tile > 0 && tile % prefer == 0)
      return prefer;
    const int64_t fallbacks[] = {4, 2, 1};
    for (int64_t v : fallbacks) {
      if (v > 0 && tile > 0 && tile % v == 0)
        return v;
    }
    return 1;
  };
  int64_t ttm = cand.threadTileM > 0 ? cand.threadTileM
                                     : pickThreadTile(cand.tileM, 4);
  int64_t ttn = cand.threadTileN > 0 ? cand.threadTileN
                                     : pickThreadTile(cand.tileN, 4);
  if (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) {
    if (!cand.swapBlockDims) {
      ttm = 1;
      ttn = 4;
    } else {
      ttm = 4;
      ttn = 1;
    }
  }
  // 确保推导出的 blockDim（tile/threadTile）不超过 1024 线程。
  // 在论文对齐的性能测量模式下可能会探索较大 tile；若未指定每线程 tile，
  // 则自动选择合法回退，避免 `--codegen-from-kernel-attrs` 编译失败。
  {
    auto computeBlockThreads = [&](int64_t thM, int64_t thN)
        -> std::optional<std::pair<int64_t, int64_t>> {
      if (cand.tileM <= 0 || cand.tileN <= 0 || thM <= 0 || thN <= 0)
        return std::nullopt;
      if ((cand.tileM % thM) != 0 || (cand.tileN % thN) != 0)
        return std::nullopt;
      int64_t bx = 1;
      int64_t by = 1;
      if (!cand.swapBlockDims) {
        bx = cand.tileN / thN;
        by = cand.tileM / thM;
      } else {
        bx = cand.tileM / thM;
        by = cand.tileN / thN;
      }
      return std::make_pair(bx, by);
    };

    auto dimsOpt = computeBlockThreads(ttm, ttn);
    int64_t threads =
        dimsOpt ? (dimsOpt->first * dimsOpt->second) : (int64_t)INT64_MAX;
    if (threads > 1024) {
      const int64_t choices[] = {1, 2, 4, 8, 16, 32, 64};
      int64_t bestThreads = -1;
      int64_t bestM = ttm;
      int64_t bestN = ttn;
      for (int64_t m : choices) {
        for (int64_t n : choices) {
          auto d = computeBlockThreads(m, n);
          if (!d)
            continue;
          int64_t t = d->first * d->second;
          if (t <= 1024 && t > bestThreads) {
            bestThreads = t;
            bestM = m;
            bestN = n;
          }
        }
      }
      if (bestThreads > 0) {
        ttm = bestM;
        ttn = bestN;
      }
    }
  }
  compileCmd.append(" --thread-tile-m ");
  compileCmd.append(std::to_string(std::max<int64_t>(1, ttm)));
  compileCmd.append(" --thread-tile-n ");
  compileCmd.append(std::to_string(std::max<int64_t>(1, ttn)));

  if (effRowReductionReuseFusion) {
    compileCmd.append(" --enable-row-reduction-chain-reuse-fusion");
    // 专用 reuse-fusion 期望把 1D broadcast 中间值保留在融合后的 gpu.launch 内。
    // 因此关闭保守的“split broadcast edges”启发式，保持单 kernel 以便调优。
    compileCmd.append(" --reduction-chain-split-broadcast-edges=false");
  }
  if (effRowReductionInputPromotion)
    compileCmd.append(" --enable-row-reduction-input-promotion");
  if (effRowReductionInputPromotionVectorize)
    compileCmd.append(" --enable-row-reduction-input-promotion-vectorize");
  if (effRowReductionWarp)
    compileCmd.append(" --enable-row-reduction-warp");
  if (effRowReductionVectorize)
    compileCmd.append(" --enable-row-reduction-vectorize");
  if (effRowReductionVectorWidth > 0) {
    compileCmd.append(" --row-reduction-vector-width=");
    compileCmd.append(std::to_string(effRowReductionVectorWidth));
  }
  if (effRowReductionThreadsX > 0) {
    compileCmd.append(" --row-reduction-threads-x=");
    compileCmd.append(std::to_string(effRowReductionThreadsX));
  }
  if (effRowReductionRelaxBarriers)
    compileCmd.append(" --enable-row-reduction-relax-barriers");
  if (effRowReductionSkipCombineBarrier)
    compileCmd.append(" --enable-row-reduction-skip-combine-barrier");
  if (effRowReductionInputVectorWidth > 0) {
    compileCmd.append(" --row-reduction-input-vector-width=");
    compileCmd.append(std::to_string(effRowReductionInputVectorWidth));
  }
  if (effRowReductionCombineVectorize)
    compileCmd.append(" --enable-row-reduction-combine-vectorize");
  if (effMatmulSoftmaxSharedReuseFusion)
    compileCmd.append(" --enable-matmul-softmax-shared-reuse-fusion");

  if (effEnableAsyncCopy) {
    compileCmd.append(" --enable-async-copy");
    if (!effAsyncBypassL1)
      compileCmd.append(" --async-bypass-l1=false");
  }
  if (effEnableSoftwarePipelining) {
    compileCmd.append(" --enable-software-pipelining");
    compileCmd.append(" --pipeline-depth ");
    compileCmd.append(std::to_string(std::max<int64_t>(2, effPipelineDepth)));
    if (!effPipelinePeelEpilogue)
      compileCmd.append(" --pipeline-peel-epilogue=false");
    if (effPipelineSetAsyncWaitGroups)
      compileCmd.append(" --pipeline-set-async-wait-groups");
  }
  if (cand.swapBlockDims) {
    compileCmd.append(" --swap-block-dims");
  }
  if (cand.enableTensorCoreTf32) {
    compileCmd.append(" --enable-tensorcore-tf32");
  }
  if (cand.enableTensorCoreF16) {
    compileCmd.append(" --enable-tensorcore-f16");
  }
  compileCmd.append(" > ");
  const std::string compileLogPath = dir + "/compile.log";
  compileCmd.append(shellEscapeSingleQuotes(compileLogPath));
  compileCmd.append(" 2>&1");

  const bool preemptDisableThreadFuseOnMarks =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_PREEMPTIVE_DISABLE_THREAD_FUSE_ON_MARKS", 0) != 0;
  const bool preemptDisableThreadFuseOnRisk =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_PREEMPTIVE_DISABLE_THREAD_FUSE_ON_RISK", 0) != 0;
  const bool allowPreemptDisableThreadFuseWhenConnect2Required =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_PREEMPTIVE_DISABLE_THREAD_FUSE_ALLOW_WHEN_CONNECT2_REQUIRED",
          0) != 0;
  const bool requireConnect2RegisterFuseEvidenceForPreempt =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_REQUIRE_CONNECT2_REG_FUSE_EVIDENCE",
          (isMatmulSoftmaxSubgraph && maxConnectLevelForCodegen >= 2) ? 1 : 0) !=
      0;
  const bool keepThreadFuseForConnect2Evidence =
      !allowPreemptDisableThreadFuseWhenConnect2Required &&
      requireConnect2RegisterFuseEvidenceForPreempt &&
      maxConnectLevelForCodegen >= 2 && enableRegisterLevelCodegen;
  bool preemptDisableThreadFuse = false;
  bool preemptByThreadFuseMarks = false;
  bool preemptByRiskPattern = false;
  int64_t riskMinTileM = 0;
  int64_t riskMinTileN = 0;
  const bool tcCand = cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
  if (threadFuseMarks > 0 && isMatmulSoftmaxSubgraph &&
      !keepThreadFuseForConnect2Evidence) {
    if (preemptDisableThreadFuseOnMarks) {
      preemptDisableThreadFuse = true;
      preemptByThreadFuseMarks = true;
    } else if (preemptDisableThreadFuseOnRisk) {
      riskMinTileM = std::max<int64_t>(
          1, getEnvInt64OrDefault(
                 "WELDER_PROFILE_PREEMPTIVE_THREAD_FUSE_RISK_MIN_TILE_M", 64));
      riskMinTileN = std::max<int64_t>(
          1, getEnvInt64OrDefault(
                 "WELDER_PROFILE_PREEMPTIVE_THREAD_FUSE_RISK_MIN_TILE_N", 64));
      const bool likelyHighRiskNoAsync =
          tcCand && effMatmulSoftmaxSharedReuseFusion && !effEnableAsyncCopy &&
          !effEnableSoftwarePipelining && cand.tileM >= riskMinTileM &&
          cand.tileN >= riskMinTileN;
      if (likelyHighRiskNoAsync) {
        preemptDisableThreadFuse = true;
        preemptByRiskPattern = true;
      }
    }
  }
  if (keepThreadFuseForConnect2Evidence && threadFuseMarks > 0 &&
      isMatmulSoftmaxSubgraph &&
      (preemptDisableThreadFuseOnMarks || preemptDisableThreadFuseOnRisk) &&
      opts.tracer) {
    llvm::json::Object f;
    f["reason"] = "connect2_register_fuse_evidence_required";
    f["thread_fuse_marks"] = threadFuseMarks;
    f["max_connect_level"] = maxConnectLevelForCodegen;
    f["enable_register_level_codegen"] = enableRegisterLevelCodegen;
    f["allow_when_connect2_required"] =
        allowPreemptDisableThreadFuseWhenConnect2Required;
    opts.tracer->event("profile.skip_preemptive_disable_thread_fuse",
                       std::move(f), /*isVerbose=*/true);
  }
  if (preemptDisableThreadFuse) {
    injectEnvBeforeBashInPlace(compileCmd, "WELDER_DISABLE_THREAD_FUSE_INTO=1");
    if (opts.tracer) {
      llvm::json::Object f;
      f["reason"] =
          preemptByThreadFuseMarks ? "thread_fuse_marks" : "risk_tile_no_async";
      f["thread_fuse_marks"] = threadFuseMarks;
      f["tile_m"] = cand.tileM;
      f["tile_n"] = cand.tileN;
      f["tile_k"] = cand.tileK;
      f["ttm"] = cand.threadTileM;
      f["ttn"] = cand.threadTileN;
      f["tc"] = tcCand;
      f["async"] = effEnableAsyncCopy;
      f["pipe"] = effEnableSoftwarePipelining;
      if (preemptByRiskPattern) {
        f["risk_min_tile_m"] = riskMinTileM;
        f["risk_min_tile_n"] = riskMinTileN;
      }
      opts.tracer->event("profile.preemptive_disable_thread_fuse", std::move(f),
                         /* isVerbose=*/true);
    }
  }
  const bool preemptTcAsyncWaitRowSafe =
      getEnvInt64OrDefault("WELDER_PROFILE_PREEMPTIVE_TC_ASYNC_WAIT_ROW_SAFE",
                           opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
  const bool tcAsyncWaitCandidate =
      (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
      effEnableAsyncCopy && effEnableSoftwarePipelining &&
      effPipelineSetAsyncWaitGroups;
  if (preemptTcAsyncWaitRowSafe && isMatmulSoftmaxSubgraph &&
      tcAsyncWaitCandidate) {
    const int64_t preemptRowThreadsX = std::max<int64_t>(
        1,
        getEnvInt64OrDefault(
            "WELDER_PROFILE_PREEMPTIVE_TC_ASYNC_WAIT_ROW_SAFE_THREADS_X", 8));
    const int64_t preemptConnectLevel =
        std::max<int64_t>(1, maxConnectLevelForCodegen);
    compileCmd = buildTcAsyncWaitRowSafeRetryCompileCmd(
        std::move(compileCmd), /*connectLevel=*/preemptConnectLevel,
        /* threadTileM=*/std::max<int64_t>(1, ttm),
        /* threadTileN=*/std::max<int64_t>(1, ttn),
        /* rowReductionThreadsX=*/preemptRowThreadsX,
        /* forceWaitGroups=*/true);
    if (opts.tracer) {
      llvm::json::Object f;
      f["connect_level"] = preemptConnectLevel;
      f["thread_tile_m"] = std::max<int64_t>(1, ttm);
      f["thread_tile_n"] = std::max<int64_t>(1, ttn);
      f["row_threads_x"] = preemptRowThreadsX;
      opts.tracer->event("profile.preemptive_tc_async_wait_row_safe",
                         std::move(f), /*isVerbose=*/true);
    }
  }

  auto compileStart = std::chrono::steady_clock::now();
  int compileRc = runShellCommand(
      wrapWithTimeoutIfRequested(compileCmd, opts.profile.timeoutSec));
  auto compileEnd = std::chrono::steady_clock::now();
  double compileMs =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          compileEnd - compileStart)
          .count();
  const CompilerThreadFusionEvidence compilerThreadFusionEvidence =
      collectCompilerThreadFusionEvidenceFromTraceFile(compilerTracePath);
  if (opts.tracer) {
    llvm::json::Object f;
    f["rc"] = compileRc;
    f["dur_ms"] = compileMs;
    f["enable_register_level_codegen"] = enableRegisterLevelCodegen;
    f["max_connect_level"] = maxConnectLevelForCodegen;
    f["thread_fuse_marks"] = threadFuseMarks;
    f["thread_fuse_operand_marks"] = threadFuseOperandMarks;
    f["register_fuse_min_connect_level"] =
        static_cast<int64_t>(registerFuseMinConnectLevel);
    f["promote_shared_edges"] = promoteSharedEdgesForRegisterFuse;
    f["promoted_shared_edges"] = promotedSharedEdges;
    f["compiler_thread_fusion_seen"] = compilerThreadFusionEvidence.seen;
    f["compiler_thread_fusion_pairs"] = compilerThreadFusionEvidence.pairCount;
    f["compiler_thread_fusion_pairs_with_operand"] =
        compilerThreadFusionEvidence.pairWithOperand;
    f["compiler_thread_fuse_attr_pairs"] =
        compilerThreadFusionEvidence.attrPairCount;
    f["compiler_thread_fuse_attr_pairs_with_operand"] =
        compilerThreadFusionEvidence.attrPairWithOperand;
    f["compiler_register_fuse_min_connect_level"] =
        compilerThreadFusionEvidence.registerFuseMinConnectLevel;
    opts.tracer->event("profile.compile", std::move(f), /*isVerbose=*/true);
  }
  if (compileRc != 0) {
    const std::string compileLogText = readFileOrEmpty(compileLogPath);
    const bool connectLevelRetryByPattern =
        isRetryableConnectLevelCompileFailure(compileLogText);
    const bool retryConnectLevel1OnPostbufferizeFail =
        getEnvInt64OrDefault(
            "WELDER_PROFILE_RETRY_CONNECT_LEVEL1_ON_POSTBUFFERIZE_FAIL", 1) !=
        0;
    const bool retryConnectLevel1OnAnyCompileFailure =
        getEnvInt64OrDefault("WELDER_PROFILE_RETRY_CONNECT_LEVEL1_ON_ANY_FAIL",
                             opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
    const bool shouldRetryConnectLevel1 =
        retryConnectLevel1OnPostbufferizeFail && maxConnectLevelForCodegen > 1 &&
        (effRowReductionReuseFusion || effMatmulSoftmaxSharedReuseFusion) &&
        (connectLevelRetryByPattern || retryConnectLevel1OnAnyCompileFailure);
    if (shouldRetryConnectLevel1) {
      int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
      if (tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
        const std::string retryCompileCmd =
            buildConnectLevelRetryCompileCmd(compileCmd, /*connectLevel=*/1);
        auto retryStart = std::chrono::steady_clock::now();
        int retryRc = runShellCommand(
            wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
        auto retryEnd = std::chrono::steady_clock::now();
        compileMs +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                retryEnd - retryStart)
                .count();
        if (opts.tracer) {
          llvm::json::Object f;
          f["rc"] = retryRc;
          f["retry_kind"] = "connect_level1_postbufferize";
          f["triggered_by_pattern"] = connectLevelRetryByPattern;
          f["triggered_on_any_fail"] = retryConnectLevel1OnAnyCompileFailure;
          f["original_max_connect_level"] = maxConnectLevelForCodegen;
          f["retry_budget_remaining"] = retryBudgetRemaining;
          opts.tracer->event("profile.compile_retry", std::move(f),
                             /* isVerbose=*/true);
        }
        compileRc = retryRc;
        if (compileRc == 0 && opts.profile.verbose) {
          llvm::errs() << "profile: compile retry succeeded with "
                          "connect_level=1 fallback, see: "
                       << compileLogPath << "\n";
        }
      } else if (opts.tracer) {
        llvm::json::Object f;
        f["retry_kind"] = "connect_level1_postbufferize";
        f["reason"] = "retry_budget_exhausted";
        f["retry_budget_remaining"] = std::max<int64_t>(int64_t(0),
                                                        retryBudgetRemaining);
        opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                           /* isVerbose=*/true);
      }
    }
  }
  if (compileRc != 0 && threadFuseMarks > 0 && !preemptDisableThreadFuse) {
    const std::string compileLogText = readFileOrEmpty(compileLogPath);
    const bool retryThreadFuseByPattern =
        isRetryableThreadFuseCompileFailure(compileLogText);
    const bool retryThreadFuseOnPatternFailure =
        getEnvInt64OrDefault(
            "WELDER_PROFILE_RETRY_DISABLE_THREAD_FUSE_ON_PATTERN_FAIL", 1) != 0;
    const bool retryThreadFuseOnAnyCompileFailure = getEnvInt64OrDefault(
                                                        "WELDER_PROFILE_RETRY_DISABLE_THREAD_FUSE_ON_ANY_FAIL",
                                                        opts.arch.elementBytes <= 2 ? 1 : 0) !=
                                                    0;
    const bool shouldRetryThreadFuse =
        (retryThreadFuseByPattern && retryThreadFuseOnPatternFailure) ||
        retryThreadFuseOnAnyCompileFailure;
    if (shouldRetryThreadFuse) {
      int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
      if (tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
        std::string retryCompileCmd = compileCmd;
        injectEnvBeforeBashInPlace(retryCompileCmd,
                                   "WELDER_DISABLE_THREAD_FUSE_INTO=1");
        auto retryStart = std::chrono::steady_clock::now();
        int retryRc = runShellCommand(
            wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
        auto retryEnd = std::chrono::steady_clock::now();
        compileMs +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                retryEnd - retryStart)
                .count();
        if (opts.tracer) {
          llvm::json::Object f;
          f["rc"] = retryRc;
          f["retry_kind"] = "disable_thread_fuse";
          f["triggered_by_pattern"] = retryThreadFuseByPattern;
          f["triggered_on_pattern_fail"] = retryThreadFuseOnPatternFailure;
          f["triggered_on_any_fail"] = retryThreadFuseOnAnyCompileFailure;
          f["thread_fuse_marks"] = threadFuseMarks;
          f["retry_budget_remaining"] = retryBudgetRemaining;
          opts.tracer->event("profile.compile_retry", std::move(f),
                             /* isVerbose=*/true);
        }
        compileRc = retryRc;
        if (compileRc == 0 && opts.profile.verbose) {
          llvm::errs() << "profile: compile retry succeeded with thread-fuse "
                          "fallback disabled, see: "
                       << compileLogPath << "\n";
        }
      } else if (opts.tracer) {
        llvm::json::Object f;
        f["retry_kind"] = "disable_thread_fuse";
        f["reason"] = "retry_budget_exhausted";
        f["retry_budget_remaining"] = std::max<int64_t>(int64_t(0),
                                                        retryBudgetRemaining);
        opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                           /* isVerbose=*/true);
      }
    }
  }
  if (compileRc != 0) {
    const bool tcAsyncRecoveryEnabled =
        getEnvInt64OrDefault("WELDER_PROFILE_RETRY_TC_ASYNC_RECOVERY",
                             opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
    const bool isTcAsyncCand = (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
                               effEnableAsyncCopy;
    if (tcAsyncRecoveryEnabled && isMatmulSoftmaxSubgraph && isTcAsyncCand) {
      const std::string tcAsyncCompileLogText = readFileOrEmpty(compileLogPath);
      const bool retryByConnectHandle =
          isRetryableConnectLevelCompileFailure(tcAsyncCompileLogText);
      const bool retryByParallelOverflow =
          isRetryableParallelResourceOverflowCompileFailure(tcAsyncCompileLogText);
      const bool retryByRowReduction =
          isRetryableRowReductionCompileFailure(tcAsyncCompileLogText);
      const bool retryByWorkgroupPackLayout =
          isRetryableWorkgroupPackLayoutCompileFailure(tcAsyncCompileLogText);
      const bool retryByKernelRootMissing =
          isRetryableKernelRootCompileFailure(tcAsyncCompileLogText);
      const bool retryByUnsupportedTarget =
          isRetryableUnsupportedTargetCompileFailure(tcAsyncCompileLogText);
      const bool defaultRetryOnAnyFailure =
          (opts.arch.elementBytes <= 2) && isMatmulSoftmaxSubgraph;
      const bool retryOnAnyFailure = getEnvInt64OrDefault(
                                         "WELDER_PROFILE_RETRY_TC_ASYNC_ON_ANY_FAIL",
                                         defaultRetryOnAnyFailure ? 1 : 0) != 0;
      if (retryByConnectHandle || retryByParallelOverflow || retryByRowReduction ||
          retryByWorkgroupPackLayout || retryByKernelRootMissing ||
          retryByUnsupportedTarget || retryOnAnyFailure) {
        const int64_t targetRowThreadsX = std::max<int64_t>(
            1, getEnvInt64OrDefault("WELDER_PROFILE_RETRY_TC_ASYNC_ROW_THREADS_X",
                                    16));
        int64_t retryRowThreadsX = effRowReductionThreadsX > 0
                                       ? std::min<int64_t>(effRowReductionThreadsX,
                                                           targetRowThreadsX)
                                       : targetRowThreadsX;
        const bool disableRowWarp =
            getEnvInt64OrDefault(
                "WELDER_PROFILE_RETRY_TC_ASYNC_DISABLE_ROW_WARP", 1) != 0;
        const bool forceTcSafeRowReduction =
            getEnvInt64OrDefault(
                "WELDER_PROFILE_RETRY_TC_ASYNC_FORCE_TC_SAFE_ROW", 1) != 0;
        const int64_t maxBlockThreads = std::max<int64_t>(
            32, getEnvInt64OrDefault("WELDER_PROFILE_RETRY_TC_ASYNC_MAX_THREADS",
                                     256));
        const int64_t maxRetries = std::max<int64_t>(
            1, getEnvInt64OrDefault("WELDER_PROFILE_RETRY_TC_ASYNC_MAX_RETRIES",
                                    4));
        auto computeBlockThreads = [&](int64_t thM, int64_t thN) -> int64_t {
          if (cand.tileM <= 0 || cand.tileN <= 0 || thM <= 0 || thN <= 0)
            return -1;
          if ((cand.tileM % thM) != 0 || (cand.tileN % thN) != 0)
            return -1;
          int64_t bx = 1;
          int64_t by = 1;
          if (!cand.swapBlockDims) {
            bx = cand.tileN / thN;
            by = cand.tileM / thM;
          } else {
            bx = cand.tileM / thM;
            by = cand.tileN / thN;
          }
          if (bx <= 0 || by <= 0)
            return -1;
          return bx * by;
        };
        auto chooseThreadTilesForMaxThreads =
            [&](int64_t limitThreads) -> std::pair<int64_t, int64_t> {
          int64_t curM = std::max<int64_t>(1, ttm);
          int64_t curN = std::max<int64_t>(1, ttn);
          int64_t curThreads = computeBlockThreads(curM, curN);
          if (curThreads > 0 && curThreads <= limitThreads)
            return {curM, curN};
          const int64_t choices[] = {1, 2, 4, 8, 16, 32, 64};
          int64_t bestThreads = -1;
          int64_t bestM = curM;
          int64_t bestN = curN;
          for (int64_t m : choices) {
            for (int64_t n : choices) {
              int64_t th = computeBlockThreads(m, n);
              if (th <= 0 || th > limitThreads)
                continue;
              if (th > bestThreads) {
                bestThreads = th;
                bestM = m;
                bestN = n;
              }
            }
          }
          if (bestThreads > 0)
            return {bestM, bestN};
          return {curM, curN};
        };
        struct TcAsyncRetryCfg {
          int64_t threadTileM;
          int64_t threadTileN;
          int64_t rowThreadsX;
        };
        std::vector<TcAsyncRetryCfg> retryCfgs;
        auto pushRetryCfg = [&](int64_t m, int64_t n, int64_t rowTx) {
          m = std::max<int64_t>(1, m);
          n = std::max<int64_t>(1, n);
          rowTx = std::max<int64_t>(1, rowTx);
          for (const auto &cfg : retryCfgs) {
            if (cfg.threadTileM == m && cfg.threadTileN == n &&
                cfg.rowThreadsX == rowTx)
              return;
          }
          retryCfgs.push_back({m, n, rowTx});
        };
        const auto baseTiles =
            std::make_pair(std::max<int64_t>(1, ttm), std::max<int64_t>(1, ttn));
        pushRetryCfg(baseTiles.first, baseTiles.second, retryRowThreadsX);
        const auto clampedTiles = chooseThreadTilesForMaxThreads(maxBlockThreads);
        pushRetryCfg(clampedTiles.first, clampedTiles.second, retryRowThreadsX);
        if (isMatmulSoftmaxSubgraph) {
          pushRetryCfg(/*threadTileM=*/1, /*threadTileN=*/1, retryRowThreadsX);
          pushRetryCfg(/*threadTileM=*/1, /*threadTileN=*/2, retryRowThreadsX);
          pushRetryCfg(/*threadTileM=*/2, /*threadTileN=*/1, retryRowThreadsX);
        }
        if (retryByParallelOverflow) {
          const int64_t halfThreadsLimit =
              std::max<int64_t>(32, maxBlockThreads / 2);
          const auto halfTiles =
              chooseThreadTilesForMaxThreads(halfThreadsLimit);
          pushRetryCfg(halfTiles.first, halfTiles.second, retryRowThreadsX);
          const int64_t quarterThreadsLimit =
              std::max<int64_t>(32, maxBlockThreads / 4);
          const auto quarterTiles =
              chooseThreadTilesForMaxThreads(quarterThreadsLimit);
          pushRetryCfg(quarterTiles.first, quarterTiles.second, retryRowThreadsX);
          if (retryRowThreadsX > 8) {
            const int64_t tighterRowThreadsX = std::max<int64_t>(
                8, std::min<int64_t>(retryRowThreadsX, retryRowThreadsX / 2));
            pushRetryCfg(baseTiles.first, baseTiles.second, tighterRowThreadsX);
            pushRetryCfg(halfTiles.first, halfTiles.second, tighterRowThreadsX);
            pushRetryCfg(quarterTiles.first, quarterTiles.second,
                         tighterRowThreadsX);
          }
        }

        int64_t retriesTried = 0;
        for (const auto &cfg : retryCfgs) {
          if (compileRc == 0 || retriesTried >= maxRetries)
            break;
          int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
          if (!tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
            if (opts.tracer) {
              llvm::json::Object f;
              f["retry_kind"] = "tc_async_compile_recovery";
              f["reason"] = "retry_budget_exhausted";
              f["attempt"] = retriesTried + 1;
              f["retry_budget_remaining"] =
                  std::max<int64_t>(int64_t(0), retryBudgetRemaining);
              opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                                 /* isVerbose=*/true);
            }
            break;
          }
          ++retriesTried;
          const std::string retryCompileCmd = buildTcAsyncRecoveryRetryCompileCmd(
              compileCmd,
              /* connectLevel=*/1, /*threadTileM=*/cfg.threadTileM,
              /* threadTileN=*/cfg.threadTileN,
              /* rowReductionThreadsX=*/cfg.rowThreadsX, disableRowWarp,
              forceTcSafeRowReduction);
          auto retryStart = std::chrono::steady_clock::now();
          int retryRc = runShellCommand(
              wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
          auto retryEnd = std::chrono::steady_clock::now();
          compileMs += std::chrono::duration_cast<
                           std::chrono::duration<double, std::milli>>(
                           retryEnd - retryStart)
                           .count();
          if (opts.tracer) {
            llvm::json::Object f;
            f["rc"] = retryRc;
            f["retry_kind"] = "tc_async_compile_recovery";
            f["attempt"] = retriesTried;
            f["triggered_by_connect_handle"] = retryByConnectHandle;
            f["triggered_by_parallel_overflow"] = retryByParallelOverflow;
            f["triggered_by_row_reduction"] = retryByRowReduction;
            f["triggered_by_workgroup_pack_layout"] = retryByWorkgroupPackLayout;
            f["triggered_by_kernel_root_missing"] = retryByKernelRootMissing;
            f["triggered_by_unsupported_target"] = retryByUnsupportedTarget;
            f["triggered_on_any_fail"] = retryOnAnyFailure;
            f["thread_tile_m"] = cfg.threadTileM;
            f["thread_tile_n"] = cfg.threadTileN;
            f["row_threads_x"] = cfg.rowThreadsX;
            f["disable_row_warp"] = disableRowWarp;
            f["force_tc_safe_row"] = forceTcSafeRowReduction;
            f["max_block_threads"] = maxBlockThreads;
            f["retry_budget_remaining"] = retryBudgetRemaining;
            opts.tracer->event("profile.compile_retry", std::move(f),
                               /* isVerbose=*/true);
          }
          compileRc = retryRc;
          if (compileRc == 0 && opts.profile.verbose) {
            llvm::errs() << "profile: compile retry succeeded with tc+async "
                            "recovery fallback, see: "
                         << compileLogPath << "\n";
          }
        }
        const bool retryTcAsyncWaitRowSafe =
            getEnvInt64OrDefault(
                "WELDER_PROFILE_RETRY_TC_ASYNC_WAIT_ROW_SAFE",
                opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
        const bool forceWaitGroups =
            effEnableSoftwarePipelining && effPipelineSetAsyncWaitGroups;
        if (compileRc != 0 && retryByRowReduction && retryTcAsyncWaitRowSafe) {
          const int64_t waitRowSafeMaxRetries = std::max<int64_t>(
              1, getEnvInt64OrDefault(
                     "WELDER_PROFILE_RETRY_TC_ASYNC_WAIT_ROW_SAFE_MAX_RETRIES",
                     3));
          llvm::SmallVector<int64_t, 4> rowThreadsChoices;
          auto pushRowThreadsChoice = [&](int64_t rowThreadsX) {
            const int64_t clamped = std::max<int64_t>(1, rowThreadsX);
            if (!llvm::is_contained(rowThreadsChoices, clamped))
              rowThreadsChoices.push_back(clamped);
          };
          const int64_t targetWaitRowThreadsX = std::max<int64_t>(
              1, getEnvInt64OrDefault(
                     "WELDER_PROFILE_RETRY_TC_ASYNC_WAIT_ROW_SAFE_THREADS_X",
                     8));
          const int64_t fallbackWaitRowThreadsX = std::max<int64_t>(
              targetWaitRowThreadsX,
              getEnvInt64OrDefault(
                  "WELDER_PROFILE_RETRY_TC_ASYNC_WAIT_ROW_SAFE_THREADS_X_FALLBACK",
                  16));
          pushRowThreadsChoice(targetWaitRowThreadsX);
          pushRowThreadsChoice(fallbackWaitRowThreadsX);
          if (effRowReductionThreadsX > 0)
            pushRowThreadsChoice(effRowReductionThreadsX);
          pushRowThreadsChoice(8);
          pushRowThreadsChoice(16);
          int64_t waitRowSafeRetriesTried = 0;
          for (int64_t rowThreadsX : rowThreadsChoices) {
            if (compileRc == 0 || waitRowSafeRetriesTried >= waitRowSafeMaxRetries)
              break;
            int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
            if (!tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
              if (opts.tracer) {
                llvm::json::Object f;
                f["retry_kind"] = "tc_async_wait_row_safe";
                f["reason"] = "retry_budget_exhausted";
                f["attempt"] = waitRowSafeRetriesTried + 1;
                f["retry_budget_remaining"] =
                    std::max<int64_t>(int64_t(0), retryBudgetRemaining);
                opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                                   /* isVerbose=*/true);
              }
              break;
            }
            ++waitRowSafeRetriesTried;
            const std::string retryCompileCmd = buildTcAsyncWaitRowSafeRetryCompileCmd(
                compileCmd,
                /* connectLevel=*/1, /*threadTileM=*/1,
                /* threadTileN=*/1, /*rowReductionThreadsX=*/rowThreadsX,
                /* forceWaitGroups=*/forceWaitGroups);
            auto retryStart = std::chrono::steady_clock::now();
            int retryRc = runShellCommand(wrapWithTimeoutIfRequested(
                retryCompileCmd, opts.profile.timeoutSec));
            auto retryEnd = std::chrono::steady_clock::now();
            compileMs += std::chrono::duration_cast<
                             std::chrono::duration<double, std::milli>>(
                             retryEnd - retryStart)
                             .count();
            if (opts.tracer) {
              llvm::json::Object f;
              f["rc"] = retryRc;
              f["retry_kind"] = "tc_async_wait_row_safe";
              f["attempt"] = waitRowSafeRetriesTried;
              f["triggered_by_row_reduction"] = retryByRowReduction;
              f["thread_tile_m"] = 1;
              f["thread_tile_n"] = 1;
              f["row_threads_x"] = rowThreadsX;
              f["force_wait_group"] = forceWaitGroups;
              f["retry_budget_remaining"] = retryBudgetRemaining;
              opts.tracer->event("profile.compile_retry", std::move(f),
                                 /* isVerbose=*/true);
            }
            compileRc = retryRc;
            if (compileRc == 0 && opts.profile.verbose) {
              llvm::errs()
                  << "profile: compile retry succeeded with tc+async+wait "
                     "row-safe fallback, see: "
                  << compileLogPath << "\n";
            }
          }
          if (compileRc != 0 && forceWaitGroups) {
            int64_t waitOffBudgetRemaining = kCompileRetryBudgetUnlimited;
            const bool fallbackWaitGroupOff = getEnvInt64OrDefault(
                                                  "WELDER_PROFILE_RETRY_TC_ASYNC_WAIT_ROW_SAFE_FALLBACK_WAIT_OFF",
                                                  1) != 0;
            if (fallbackWaitGroupOff &&
                tryConsumeCompileRetryBudget(&waitOffBudgetRemaining)) {
              const std::string retryCompileCmd =
                  buildTcAsyncWaitRowSafeRetryCompileCmd(
                      compileCmd,
                      /* connectLevel=*/1, /*threadTileM=*/1,
                      /* threadTileN=*/1,
                      /* rowReductionThreadsX=*/targetWaitRowThreadsX,
                      /* forceWaitGroups=*/false);
              auto retryStart = std::chrono::steady_clock::now();
              int retryRc = runShellCommand(wrapWithTimeoutIfRequested(
                  retryCompileCmd, opts.profile.timeoutSec));
              auto retryEnd = std::chrono::steady_clock::now();
              compileMs += std::chrono::duration_cast<
                               std::chrono::duration<double, std::milli>>(
                               retryEnd - retryStart)
                               .count();
              if (opts.tracer) {
                llvm::json::Object f;
                f["rc"] = retryRc;
                f["retry_kind"] = "tc_async_wait_row_safe_wait_off";
                f["attempt"] = waitRowSafeRetriesTried + 1;
                f["triggered_by_row_reduction"] = retryByRowReduction;
                f["thread_tile_m"] = 1;
                f["thread_tile_n"] = 1;
                f["row_threads_x"] = targetWaitRowThreadsX;
                f["retry_budget_remaining"] = waitOffBudgetRemaining;
                opts.tracer->event("profile.compile_retry", std::move(f),
                                   /* isVerbose=*/true);
              }
              compileRc = retryRc;
            } else if (fallbackWaitGroupOff && opts.tracer) {
              llvm::json::Object f;
              f["retry_kind"] = "tc_async_wait_row_safe_wait_off";
              f["reason"] = "retry_budget_exhausted";
              f["retry_budget_remaining"] =
                  std::max<int64_t>(int64_t(0), waitOffBudgetRemaining);
              opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                                 /* isVerbose=*/true);
            }
          }
        }
      }
    }
  }
  if (compileRc != 0) {
    const std::string compileLogText = readFileOrEmpty(compileLogPath);
    const bool retryByWorkgroupPackLayout =
        isRetryableWorkgroupPackLayoutCompileFailure(compileLogText);
    const bool enableWorkgroupPackLayoutRetry =
        getEnvInt64OrDefault("WELDER_PROFILE_RETRY_WORKGROUP_PACK_LAYOUT_SAFE",
                             opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
    if (enableWorkgroupPackLayoutRetry && retryByWorkgroupPackLayout) {
      int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
      if (tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
        const bool tcCandidate =
            cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
        const int64_t retryConnectLevel =
            maxConnectLevelForCodegen > 1 ? int64_t(1)
                                          : std::max<int64_t>(1, maxConnectLevelForCodegen);
        const std::string retryCompileCmd = buildWorkgroupPackLayoutSafeRetryCompileCmd(
            compileCmd, retryConnectLevel, tcCandidate);
        auto retryStart = std::chrono::steady_clock::now();
        int retryRc = runShellCommand(
            wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
        auto retryEnd = std::chrono::steady_clock::now();
        compileMs +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                retryEnd - retryStart)
                .count();
        if (opts.tracer) {
          llvm::json::Object f;
          f["rc"] = retryRc;
          f["retry_kind"] = "workgroup_pack_layout_safe";
          f["tc_candidate"] = tcCandidate;
          f["retry_connect_level"] = retryConnectLevel;
          f["retry_budget_remaining"] = retryBudgetRemaining;
          opts.tracer->event("profile.compile_retry", std::move(f),
                             /* isVerbose=*/true);
        }
        compileRc = retryRc;
        if (compileRc == 0 && opts.profile.verbose) {
          llvm::errs()
              << "profile: compile retry succeeded with workgroup-pack "
                 "layout-safe fallback, see: "
              << compileLogPath << "\n";
        }
      } else if (opts.tracer) {
        llvm::json::Object f;
        f["retry_kind"] = "workgroup_pack_layout_safe";
        f["reason"] = "retry_budget_exhausted";
        f["retry_budget_remaining"] =
            std::max<int64_t>(int64_t(0), retryBudgetRemaining);
        opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                           /* isVerbose=*/true);
      }
    }
  }
  if (compileRc != 0) {
    const std::string compileLogText = readFileOrEmpty(compileLogPath);
    const bool hasRowReductionKnobs =
        effRowReductionReuseFusion || effRowReductionInputPromotion ||
        effRowReductionInputPromotionVectorize || effRowReductionWarp ||
        effRowReductionVectorize || effRowReductionRelaxBarriers ||
        effRowReductionSkipCombineBarrier || effRowReductionCombineVectorize ||
        effRowReductionVectorWidth > 0 || effRowReductionThreadsX > 0 ||
        effRowReductionInputVectorWidth > 0;
    const bool safeRowRetryByPattern =
        isRetryableRowReductionCompileFailure(compileLogText);
    const bool retrySafeRowOnAnyCompileFailure =
        getEnvInt64OrDefault("WELDER_PROFILE_RETRY_SAFE_ROW_ON_ANY_FAIL",
                             opts.arch.elementBytes <= 2 ? 1 : 0) != 0;
    const bool shouldRetrySafeRowCompile =
        (hasRowReductionKnobs || effMatmulSoftmaxSharedReuseFusion) &&
        (safeRowRetryByPattern || retrySafeRowOnAnyCompileFailure);
    if (shouldRetrySafeRowCompile) {
      const bool dropMatmulSoftmaxSharedReuse =
          getEnvInt64OrDefault("WELDER_PROFILE_SAFE_ROW_DROP_MM_SM_REUSE", 0) !=
          0;
      const bool dropRowReductionReuseFusion =
          getEnvInt64OrDefault("WELDER_PROFILE_SAFE_ROW_DROP_RR_REUSE", 0) != 0;
      int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
      if (tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
        const std::string retryCompileCmd = buildSafeRowReductionRetryCompileCmd(
            compileCmd, dropMatmulSoftmaxSharedReuse,
            dropRowReductionReuseFusion,
            /* forceFastRowReduction=*/false,
            /* forceTcSafeRowReduction=*/true);
        auto retryStart = std::chrono::steady_clock::now();
        int retryRc = runShellCommand(
            wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
        auto retryEnd = std::chrono::steady_clock::now();
        compileMs +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                retryEnd - retryStart)
                .count();
        if (opts.tracer) {
          llvm::json::Object f;
          f["rc"] = retryRc;
          f["retry_kind"] = "safe_row_reduction";
          f["drop_mm_sm_reuse"] = dropMatmulSoftmaxSharedReuse;
          f["drop_rr_reuse"] = dropRowReductionReuseFusion;
          f["triggered_by_pattern"] = safeRowRetryByPattern;
          f["triggered_on_any_fail"] = retrySafeRowOnAnyCompileFailure;
          f["retry_budget_remaining"] = retryBudgetRemaining;
          opts.tracer->event("profile.compile_retry", std::move(f),
                             /* isVerbose=*/true);
        }
        compileRc = retryRc;
      } else if (opts.tracer) {
        llvm::json::Object f;
        f["retry_kind"] = "safe_row_reduction";
        f["reason"] = "retry_budget_exhausted";
        f["retry_budget_remaining"] = std::max<int64_t>(int64_t(0),
                                                        retryBudgetRemaining);
        opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                           /* isVerbose=*/true);
      }
      if (compileRc != 0) {
        const bool retryFastRowOnParallelOverflow =
            getEnvInt64OrDefault(
                "WELDER_PROFILE_RETRY_SAFE_ROW_FASTROW_ON_PAR_OVERFLOW", 1) != 0;
        const std::string retryCompileLogText = readFileOrEmpty(compileLogPath);
        if (retryFastRowOnParallelOverflow &&
            isRetryableParallelResourceOverflowCompileFailure(
                retryCompileLogText)) {
          int64_t retryFastBudgetRemaining = kCompileRetryBudgetUnlimited;
          if (tryConsumeCompileRetryBudget(&retryFastBudgetRemaining)) {
            const std::string retryFastRowCmd = buildSafeRowReductionRetryCompileCmd(
                compileCmd, dropMatmulSoftmaxSharedReuse,
                dropRowReductionReuseFusion,
                /* forceFastRowReduction=*/true,
                /* forceTcSafeRowReduction=*/false);
            auto retryFastStart = std::chrono::steady_clock::now();
            int retryFastRc = runShellCommand(wrapWithTimeoutIfRequested(
                retryFastRowCmd, opts.profile.timeoutSec));
            auto retryFastEnd = std::chrono::steady_clock::now();
            compileMs += std::chrono::duration_cast<
                             std::chrono::duration<double, std::milli>>(
                             retryFastEnd - retryFastStart)
                             .count();
            if (opts.tracer) {
              llvm::json::Object f;
              f["rc"] = retryFastRc;
              f["retry_kind"] = "safe_row_reduction_fastrow_overflow";
              f["retry_budget_remaining"] = retryFastBudgetRemaining;
              opts.tracer->event("profile.compile_retry", std::move(f),
                                 /* isVerbose=*/true);
            }
            compileRc = retryFastRc;
          } else if (opts.tracer) {
            llvm::json::Object f;
            f["retry_kind"] = "safe_row_reduction_fastrow_overflow";
            f["reason"] = "retry_budget_exhausted";
            f["retry_budget_remaining"] = std::max<int64_t>(
                int64_t(0), retryFastBudgetRemaining);
            opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                               /* isVerbose=*/true);
          }
        }
      }
      if (compileRc == 0 && opts.profile.verbose) {
        llvm::errs() << "profile: compile retry succeeded with safe "
                        "row-reduction fallback, see: "
                     << compileLogPath << "\n";
      }
    }
  }
  if (compileRc != 0) {
    const std::string compileLogText = readFileOrEmpty(compileLogPath);
    const bool retryConnectPattern =
        isRetryableConnectLevelCompileFailure(compileLogText);
    const bool retryParallelOverflowPattern =
        isRetryableParallelResourceOverflowCompileFailure(compileLogText);
    const bool retryRowReductionPattern =
        isRetryableRowReductionCompileFailure(compileLogText);
    const bool retryVectorMaskPattern =
        isRetryableVectorMaskCompileFailure(compileLogText);
    const bool retryWorkgroupPackPattern =
        isRetryableWorkgroupPackLayoutCompileFailure(compileLogText);
    const bool retryKernelRootPattern =
        isRetryableKernelRootCompileFailure(compileLogText);
    const bool retryUnsupportedTargetPattern =
        isRetryableUnsupportedTargetCompileFailure(compileLogText);
    const bool tcAsyncCandidate =
        (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
        effEnableAsyncCopy;
    if (opts.profile.verbose) {
      llvm::errs() << "profile: compile failed (rc=" << compileRc
                   << "), see: " << compileLogPath << "\n";
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["rc"] = compileRc;
      f["stage"] = "compile";
      f["tm"] = cand.tileM;
      f["tn"] = cand.tileN;
      f["tk"] = cand.tileK;
      f["ttm"] = ttm;
      f["ttn"] = ttn;
      f["row_threads_x"] = effRowReductionThreadsX;
      f["row_warp"] = effRowReductionWarp;
      f["row_vec"] = effRowReductionVectorize;
      f["row_comb_vec"] = effRowReductionCombineVectorize;
      f["row_input_promo_vec"] = effRowReductionInputPromotionVectorize;
      f["row_relax_barriers"] = effRowReductionRelaxBarriers;
      f["row_skip_combine_barrier"] = effRowReductionSkipCombineBarrier;
      f["pipeline_depth"] = effPipelineDepth;
      f["multibuffer_depth"] = multiDepth;
      f["pipeline_wait_group"] = effPipelineSetAsyncWaitGroups;
      f["retry_connect_pattern"] = retryConnectPattern;
      f["retry_parallel_overflow_pattern"] = retryParallelOverflowPattern;
      f["retry_row_reduction_pattern"] = retryRowReductionPattern;
      f["retry_vector_mask_pattern"] = retryVectorMaskPattern;
      f["retry_workgroup_pack_pattern"] = retryWorkgroupPackPattern;
      f["retry_kernel_root_pattern"] = retryKernelRootPattern;
      f["retry_unsupported_target_pattern"] = retryUnsupportedTargetPattern;
      f["tc_async_candidate"] = tcAsyncCandidate;
      f["thread_fuse_marks"] = threadFuseMarks;
      f["compiler_thread_fusion_seen"] = compilerThreadFusionEvidence.seen;
      f["compiler_thread_fusion_pairs"] =
          compilerThreadFusionEvidence.pairCount;
      f["compiler_thread_fusion_pairs_with_operand"] =
          compilerThreadFusionEvidence.pairWithOperand;
      f["compiler_thread_fuse_attr_pairs"] =
          compilerThreadFusionEvidence.attrPairCount;
      f["compiler_thread_fuse_attr_pairs_with_operand"] =
          compilerThreadFusionEvidence.attrPairWithOperand;
      f["compiler_register_fuse_min_connect_level"] =
          compilerThreadFusionEvidence.registerFuseMinConnectLevel;
      f["mm_sm_subgraph"] = isMatmulSoftmaxSubgraph;
      f["max_connect_level"] = maxConnectLevelForCodegen;
      opts.tracer->event("profile.fail", std::move(f));
    }
    return std::nullopt;
  }

  // 运行 profiler。
  std::string profOutPath = dir + "/profile.out";
  std::string nvvmText = readFileOrEmpty(nvvmPath);
  int kernelCount = 0;
  {
    size_t pos = 0;
    while (true) {
      pos = nvvmText.find("gpu.launch_func", pos);
      if (pos == std::string::npos)
        break;
      ++kernelCount;
      pos += 14;
    }
  }

  NvvmPtxStats ptxStats = collectNvvmPtxStats(nvvmText);
  auto emitPtxFeatures = [&](llvm::StringRef stage) {
    if (!opts.tracer)
      return;
    llvm::json::Object f;
    f["has_mma_sync"] = ptxStats.hasMmaSync;
    f["has_cp_async"] = ptxStats.hasCpAsync;
    f["has_cp_async_wait_group"] = ptxStats.hasCpAsyncWaitGroup;
    f["cp_async_ops"] = ptxStats.cpAsyncOps;
    f["cp_async_wait_group_ops"] = ptxStats.cpAsyncWaitGroupOps;
    f["local_depot"] = ptxStats.maxLocalDepotBytes;
    f["local_ops"] = ptxStats.localMemOps();
    f["reg_b64"] = ptxStats.maxRegB64;
    f["reg_b32"] = ptxStats.maxRegB32;
    f["tc"] = cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
    f["async"] = cand.enableAsyncCopy;
    f["pipe"] = cand.enableSoftwarePipelining;
    f["wait_group"] = cand.pipelineSetAsyncWaitGroups;
    f["mm_sm_subgraph"] = isMatmulSoftmaxSubgraph;
    if (!stage.empty())
      f["stage"] = stage.str();
    opts.tracer->event("profile.ptx_features", std::move(f), /*isVerbose=*/true);
  };
  emitPtxFeatures("");

  auto cacheRejectProfileMs = [&](llvm::StringRef reason, double rejectMs) {
    if (opts.profile.verbose) {
      llvm::errs() << "profile: reject candidate before profiling reason="
                   << reason << " reject_ms=" << rejectMs
                   << " mma=" << (ptxStats.hasMmaSync ? 1 : 0)
                   << " cp_async=" << (ptxStats.hasCpAsync ? 1 : 0)
                   << " cp_async_wait_group="
                   << (ptxStats.hasCpAsyncWaitGroup ? 1 : 0) << "\n";
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["reason"] = reason;
      f["reject_ms"] = rejectMs;
      f["has_mma_sync"] = ptxStats.hasMmaSync;
      f["has_cp_async"] = ptxStats.hasCpAsync;
      f["has_cp_async_wait_group"] = ptxStats.hasCpAsyncWaitGroup;
      f["cp_async_ops"] = ptxStats.cpAsyncOps;
      f["cp_async_wait_group_ops"] = ptxStats.cpAsyncWaitGroupOps;
      opts.tracer->event("profile.reject_missing_codegen_feature", std::move(f));
    }
    {
      std::lock_guard<std::mutex> lock(cacheMu);
      cache[key] = rejectMs;
    }
    {
      std::lock_guard<std::mutex> lock(diskMu);
      appendDiskProfileCache(opts.profile.cachePath, key, rejectMs);
    }
    return std::optional<double>(rejectMs);
  };

  const double featureRejectMs = std::max(
      1.0, getEnvDoubleOrDefault("WELDER_PROFILE_FEATURE_REJECT_MS",
                                 /*default=*/1.0e9));
  const bool requireTcMmaSync =
      getEnvInt64OrDefault("WELDER_PROFILE_TC_REQUIRE_MMA_SYNC", 1) != 0;
  const bool requireAsyncCpAsync =
      getEnvInt64OrDefault("WELDER_PROFILE_REQUIRE_CP_ASYNC_FOR_ASYNC", 1) != 0;
  const bool requireWaitGroup =
      getEnvInt64OrDefault("WELDER_PROFILE_REQUIRE_WAIT_GROUP_FOR_WAIT_KNOB", 1) !=
      0;
  const bool retryTensorCoreMissingMma =
      getEnvInt64OrDefault("WELDER_PROFILE_TC_RETRY_ON_MISSING_MMA", 1) != 0;
  const bool requireConnect2RegisterFuseEvidence =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_REQUIRE_CONNECT2_REG_FUSE_EVIDENCE",
          (isMatmulSoftmaxSubgraph && maxConnectLevelForCodegen >= 2) ? 1 : 0) !=
      0;
  const bool requireConnect2ThreadFuseOperandMarks =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_THREAD_FUSE_OPERAND_MARKS",
          0) != 0;
  const bool requireConnect2RegisterMinLevel =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_REGISTER_MIN_LEVEL",
          maxConnectLevelForCodegen >= 2 ? 1 : 0) != 0;
  const bool requireConnect2CompilerFusionEvidence =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_REQUIRE_CONNECT2_COMPILER_FUSION_EVIDENCE",
          (isMatmulSoftmaxSubgraph && maxConnectLevelForCodegen >= 2) ? 1 : 0) !=
      0;
  const bool requireConnect2CompilerOperandPairs =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_COMPILER_OPERAND_PAIRS", 0) != 0;
  const bool requireConnect2CompilerMinLevel =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_REQUIRE_COMPILER_MIN_LEVEL",
          maxConnectLevelForCodegen >= 2 ? 1 : 0) != 0;
  const bool acceptCompilerAttrPairMarks =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_ACCEPT_COMPILER_ATTR_PAIR_MARKS",
          (isMatmulSoftmaxSubgraph && maxConnectLevelForCodegen >= 2) ? 1 : 0) !=
      0;
  const bool acceptCompilerMinLevelAsFusionEvidence =
      getEnvInt64OrDefault(
          "WELDER_PROFILE_CONNECT2_ACCEPT_COMPILER_MIN_LEVEL_AS_FUSION_EVIDENCE",
          (isMatmulSoftmaxSubgraph && maxConnectLevelForCodegen >= 2) ? 1 : 0) !=
      0;
  const bool connect2ReuseExpected =
      maxConnectLevelForCodegen >= 2 && enableRegisterLevelCodegen &&
      (effMatmulSoftmaxSharedReuseFusion || effRowReductionReuseFusion ||
       isMatmulSoftmaxSubgraph);
  if (requireConnect2RegisterFuseEvidence && connect2ReuseExpected) {
    bool hasRegisterFuseEvidence = threadFuseMarks > 0;
    if (requireConnect2ThreadFuseOperandMarks)
      hasRegisterFuseEvidence = hasRegisterFuseEvidence &&
                                (threadFuseOperandMarks > 0);
    if (requireConnect2RegisterMinLevel)
      hasRegisterFuseEvidence =
          hasRegisterFuseEvidence && (registerFuseMinConnectLevel >= 2);
    int64_t compilerPairCountEvidence = compilerThreadFusionEvidence.pairCount;
    int64_t compilerPairOperandEvidence =
        compilerThreadFusionEvidence.pairWithOperand;
    if (acceptCompilerAttrPairMarks && compilerPairCountEvidence <= 0)
      compilerPairCountEvidence = compilerThreadFusionEvidence.attrPairCount;
    if (acceptCompilerAttrPairMarks && compilerPairOperandEvidence <= 0)
      compilerPairOperandEvidence =
          compilerThreadFusionEvidence.attrPairWithOperand;
    bool hasCompilerFusionEvidence = compilerPairCountEvidence > 0;
    if (requireConnect2CompilerOperandPairs)
      hasCompilerFusionEvidence =
          hasCompilerFusionEvidence && (compilerPairOperandEvidence > 0);
    if (!hasCompilerFusionEvidence && acceptCompilerMinLevelAsFusionEvidence &&
        compilerThreadFusionEvidence.seen &&
        compilerThreadFusionEvidence.registerFuseMinConnectLevel >= 2) {
      hasCompilerFusionEvidence = true;
    }
    if (requireConnect2CompilerMinLevel)
      hasCompilerFusionEvidence =
          hasCompilerFusionEvidence &&
          (compilerThreadFusionEvidence.registerFuseMinConnectLevel >= 2);
    if (requireConnect2CompilerFusionEvidence)
      hasRegisterFuseEvidence =
          hasRegisterFuseEvidence && hasCompilerFusionEvidence;
    if (!hasRegisterFuseEvidence) {
      if (opts.tracer) {
        llvm::json::Object f;
        f["max_connect_level"] = maxConnectLevelForCodegen;
        f["enable_register_level_codegen"] = enableRegisterLevelCodegen;
        f["thread_fuse_marks"] = threadFuseMarks;
        f["thread_fuse_operand_marks"] = threadFuseOperandMarks;
        f["register_fuse_min_connect_level"] = registerFuseMinConnectLevel;
        f["promoted_shared_edges"] = promotedSharedEdges;
        f["compiler_thread_fusion_seen"] = compilerThreadFusionEvidence.seen;
        f["compiler_thread_fusion_pairs"] =
            compilerThreadFusionEvidence.pairCount;
        f["compiler_thread_fusion_pairs_with_operand"] =
            compilerThreadFusionEvidence.pairWithOperand;
        f["compiler_thread_fuse_attr_pairs"] =
            compilerThreadFusionEvidence.attrPairCount;
        f["compiler_thread_fuse_attr_pairs_with_operand"] =
            compilerThreadFusionEvidence.attrPairWithOperand;
        f["compiler_register_fuse_min_connect_level"] =
            compilerThreadFusionEvidence.registerFuseMinConnectLevel;
        f["reuse_expected"] = connect2ReuseExpected;
        f["require_operand_marks"] = requireConnect2ThreadFuseOperandMarks;
        f["require_register_min_level"] = requireConnect2RegisterMinLevel;
        f["require_compiler_fusion_evidence"] =
            requireConnect2CompilerFusionEvidence;
        f["require_compiler_operand_pairs"] =
            requireConnect2CompilerOperandPairs;
        f["require_compiler_min_level"] =
            requireConnect2CompilerMinLevel;
        f["accept_compiler_attr_pair_marks"] =
            acceptCompilerAttrPairMarks;
        f["accept_compiler_min_level_as_fusion_evidence"] =
            acceptCompilerMinLevelAsFusionEvidence;
        opts.tracer->event("profile.reject_connect2_register_fuse_evidence",
                           std::move(f), /*isVerbose=*/true);
      }
      return cacheRejectProfileMs("missing_connect2_register_fuse_evidence",
                                  featureRejectMs);
    }
  }
  if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) && requireTcMmaSync &&
      !ptxStats.hasMmaSync && retryTensorCoreMissingMma) {
    int64_t retryBudgetRemaining = kCompileRetryBudgetUnlimited;
    if (tryConsumeCompileRetryBudget(&retryBudgetRemaining)) {
      const std::string retryCompileCmd =
          buildTensorCoreMissingMmaRetryCompileCmd(compileCmd);
      auto retryStart = std::chrono::steady_clock::now();
      int retryRc = runShellCommand(
          wrapWithTimeoutIfRequested(retryCompileCmd, opts.profile.timeoutSec));
      auto retryEnd = std::chrono::steady_clock::now();
      compileMs +=
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
              retryEnd - retryStart)
              .count();
      if (opts.tracer) {
        llvm::json::Object f;
        f["rc"] = retryRc;
        f["retry_kind"] = "tensorcore_missing_mma_safe";
        f["retry_budget_remaining"] = retryBudgetRemaining;
        opts.tracer->event("profile.compile_retry", std::move(f),
                           /* isVerbose=*/true);
      }
      if (retryRc == 0) {
        nvvmText = readFileOrEmpty(nvvmPath);
        ptxStats = collectNvvmPtxStats(nvvmText);
        emitPtxFeatures("retry_missing_mma_safe");
      }
    } else if (opts.tracer) {
      llvm::json::Object f;
      f["retry_kind"] = "tensorcore_missing_mma_safe";
      f["reason"] = "retry_budget_exhausted";
      f["retry_budget_remaining"] =
          std::max<int64_t>(int64_t(0), retryBudgetRemaining);
      opts.tracer->event("profile.compile_retry_skipped", std::move(f),
                         /* isVerbose=*/true);
    }
  }
  if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) && requireTcMmaSync &&
      !ptxStats.hasMmaSync)
    return cacheRejectProfileMs("missing_mma_sync", featureRejectMs);
  if (cand.enableAsyncCopy && requireAsyncCpAsync && !ptxStats.hasCpAsync)
    return cacheRejectProfileMs("missing_cp_async", featureRejectMs);
  if (cand.enableSoftwarePipelining && cand.enableAsyncCopy && requireAsyncCpAsync &&
      !ptxStats.hasCpAsync)
    return cacheRejectProfileMs("missing_cp_async_for_pipeline", featureRejectMs);
  if (cand.enableSoftwarePipelining && cand.pipelineSetAsyncWaitGroups &&
      requireWaitGroup && !ptxStats.hasCpAsyncWaitGroup)
    return cacheRejectProfileMs("missing_cp_async_wait_group", featureRejectMs);

  // TensorCore 候选保护：部分融合 kernel（尤其 f16/tensorcore 的
  // Matmul->Softmax）可能编译出病态 PTX，表现为本地栈极大、ld/st.local 过重，
  // 最终导致严重降速并干扰性能测量驱动的选优。
  //
  // 与其让这些候选因代价信号不完整而“误胜”，这里直接赋予确定性的高时延并缓存。
  if (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) {
    int64_t maxLocalDepotBytes = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MAX_LOCAL_DEPOT_BYTES",
                                /*default=*/16384));
    int64_t maxLocalMemOps = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MAX_LOCAL_MEM_OPS",
                                /*default=*/24));
    const int64_t maxRegB64 = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MAX_REG_B64",
                                /*default=*/320));
    const int64_t maxRegB32 = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MAX_REG_B32",
                                /*default=*/192));
    int64_t maxLocalMemOpsNoAsync = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MAX_LOCAL_MEM_OPS_NO_ASYNC",
                                /*default=*/12));
    int64_t minLocalDepotNoAsync = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MIN_LOCAL_DEPOT_NO_ASYNC",
                                /*default=*/4096));
    if (isMatmulSoftmaxSubgraph && ptxStats.hasMmaSync) {
      const int64_t mmSmMaxLocalDepotBytes = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MM_SM_MAX_LOCAL_DEPOT_BYTES",
                                  /*default=*/12288));
      const int64_t mmSmMaxLocalMemOps = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_PROFILE_TC_MM_SM_MAX_LOCAL_MEM_OPS",
                                  /*default=*/16));
      const int64_t mmSmMaxLocalMemOpsNoAsync = std::max<int64_t>(
          0,
          getEnvInt64OrDefault("WELDER_PROFILE_TC_MM_SM_MAX_LOCAL_MEM_OPS_NO_ASYNC",
                               /*default=*/8));
      const int64_t mmSmMinLocalDepotNoAsync = std::max<int64_t>(
          0,
          getEnvInt64OrDefault("WELDER_PROFILE_TC_MM_SM_MIN_LOCAL_DEPOT_NO_ASYNC",
                               /*default=*/2048));
      if (mmSmMaxLocalDepotBytes > 0)
        maxLocalDepotBytes =
            std::min<int64_t>(maxLocalDepotBytes, mmSmMaxLocalDepotBytes);
      if (mmSmMaxLocalMemOps > 0)
        maxLocalMemOps = std::min<int64_t>(maxLocalMemOps, mmSmMaxLocalMemOps);
      if (mmSmMaxLocalMemOpsNoAsync > 0)
        maxLocalMemOpsNoAsync =
            std::min<int64_t>(maxLocalMemOpsNoAsync, mmSmMaxLocalMemOpsNoAsync);
      if (mmSmMinLocalDepotNoAsync > 0)
        minLocalDepotNoAsync =
            std::min<int64_t>(minLocalDepotNoAsync, mmSmMinLocalDepotNoAsync);
    }
    const double rejectMs = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_REJECT_MS",
                                   /*default=*/1.0e9));

    const bool localDepotTooLarge =
        (maxLocalDepotBytes > 0 &&
         ptxStats.maxLocalDepotBytes > maxLocalDepotBytes);
    const bool localOpsTooMany =
        (maxLocalMemOps > 0 && ptxStats.localMemOps() > maxLocalMemOps);
    const bool regAndLocalPressureTooHigh =
        (maxRegB64 > 0 && maxRegB32 > 0 && ptxStats.maxRegB64 > maxRegB64 &&
         ptxStats.maxRegB32 > maxRegB32 && ptxStats.localMemOps() > 0);
    const bool strictNoAsyncLocalOpsGuard =
        isMatmulSoftmaxSubgraph && ptxStats.hasMmaSync &&
        (getEnvInt64OrDefault(
             "WELDER_PROFILE_TC_MM_SM_REJECT_NO_ASYNC_LOCAL_OPS_ONLY", 1) != 0);
    const bool localNoAsyncTooHeavy =
        (maxLocalMemOpsNoAsync > 0 &&
         ptxStats.localMemOps() > maxLocalMemOpsNoAsync &&
         (strictNoAsyncLocalOpsGuard ||
          ptxStats.maxLocalDepotBytes >= minLocalDepotNoAsync) &&
         !ptxStats.hasCpAsync);

    if (ptxStats.hasMmaSync &&
        (localDepotTooLarge || localOpsTooMany || regAndLocalPressureTooHigh ||
         localNoAsyncTooHeavy)) {
      if (opts.profile.verbose) {
        llvm::errs()
            << "profile: reject pathological TC candidate before profiling"
            << " local_depot=" << ptxStats.maxLocalDepotBytes
            << " local_ops=" << ptxStats.localMemOps()
            << " reg_b64=" << ptxStats.maxRegB64
            << " reg_b32=" << ptxStats.maxRegB32
            << " cp_async=" << (ptxStats.hasCpAsync ? 1 : 0)
            << " reject_ms=" << rejectMs << "\n";
      }
      if (opts.tracer) {
        llvm::json::Object f;
        f["local_depot"] = ptxStats.maxLocalDepotBytes;
        f["local_ops"] = ptxStats.localMemOps();
        f["ld_local"] = ptxStats.localLoadOps;
        f["st_local"] = ptxStats.localStoreOps;
        f["reg_b64"] = ptxStats.maxRegB64;
        f["reg_b32"] = ptxStats.maxRegB32;
        f["has_cp_async"] = ptxStats.hasCpAsync;
        f["reject_ms"] = rejectMs;
        opts.tracer->event("profile.reject_pathological_tc", std::move(f));
      }
      {
        std::lock_guard<std::mutex> lock(cacheMu);
        cache[key] = rejectMs;
      }
      {
        std::lock_guard<std::mutex> lock(diskMu);
        appendDiskProfileCache(opts.profile.cachePath, key, rejectMs);
      }
      return rejectMs;
    }
  }

  // async + software-pipelining 的附加保护：
  // 即使 PTX 没有明显 ld/st.local 病态，高寄存器压力和低估计占用率
  // 也常导致 Matmul->Softmax 融合 kernel 出现严重回退。
  if (cand.enableSoftwarePipelining) {
    const int64_t maxPipeRegB64 = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_PIPE_MAX_REG_B64",
                                /*default=*/448));
    const int64_t maxPipeRegB32 = std::max<int64_t>(
        0, getEnvInt64OrDefault("WELDER_PROFILE_PIPE_MAX_REG_B32",
                                /*default=*/256));
    const double pipeRejectMs = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_PIPE_REJECT_MS",
                                   /*default=*/1.0e9));
    const bool pipeRegPressureTooHigh =
        (maxPipeRegB64 > 0 && maxPipeRegB32 > 0 &&
         ptxStats.maxRegB64 > maxPipeRegB64 &&
         ptxStats.maxRegB32 > maxPipeRegB32);
    if (pipeRegPressureTooHigh)
      return cacheRejectProfileMs("pipeline_reg_pressure", pipeRejectMs);
    const int64_t maxPipeLocalMemOps =
        std::max<int64_t>(0, getEnvInt64OrDefault(
                                 "WELDER_PROFILE_PIPE_MAX_LOCAL_MEM_OPS",
                                 /*default=*/24));
    if (maxPipeLocalMemOps > 0 &&
        ptxStats.localMemOps() > maxPipeLocalMemOps) {
      return cacheRejectProfileMs("pipeline_local_mem_pressure", pipeRejectMs);
    }

    if (cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) {
      const int64_t maxTcPipeRegB64 = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_PROFILE_TC_PIPE_MAX_REG_B64",
                                  /*default=*/256));
      const int64_t maxTcPipeRegB32 = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_PROFILE_TC_PIPE_MAX_REG_B32",
                                  /*default=*/160));
      const int64_t minTcPipeBlocksPerSM =
          std::max<int64_t>(1, getEnvInt64OrDefault(
                                   "WELDER_PROFILE_TC_PIPE_MIN_BLOCKS_PER_SM",
                                   /*default=*/2));
      const double tcPipeRejectMs = std::max(
          1.0, getEnvDoubleOrDefault("WELDER_PROFILE_TC_PIPE_REJECT_MS",
                                     /*default=*/1.0e9));
      const bool tcPipeRegTooHigh =
          (maxTcPipeRegB64 > 0 && maxTcPipeRegB32 > 0 &&
           ptxStats.maxRegB64 > maxTcPipeRegB64 &&
           ptxStats.maxRegB32 > maxTcPipeRegB32);
      if (ptxStats.hasMmaSync && tcPipeRegTooHigh)
        return cacheRejectProfileMs("tc_pipeline_reg_pressure", tcPipeRejectMs);
      const int64_t maxTcPipeLocalMemOps = std::max<int64_t>(
          0, getEnvInt64OrDefault(
                 isMatmulSoftmaxSubgraph
                     ? "WELDER_PROFILE_TC_MM_SM_PIPE_MAX_LOCAL_MEM_OPS"
                     : "WELDER_PROFILE_TC_PIPE_MAX_LOCAL_MEM_OPS",
                 isMatmulSoftmaxSubgraph ? 12 : 16));
      if (maxTcPipeLocalMemOps > 0 && ptxStats.hasMmaSync &&
          ptxStats.localMemOps() > maxTcPipeLocalMemOps) {
        return cacheRejectProfileMs("tc_pipeline_local_mem_pressure",
                                    tcPipeRejectMs);
      }

      const int64_t estBlocksPerSM = std::max<int64_t>(0, cand.blocksPerSM);
      if (estBlocksPerSM > 0 && estBlocksPerSM < minTcPipeBlocksPerSM)
        return cacheRejectProfileMs("tc_pipeline_low_occupancy", tcPipeRejectMs);
      if (cand.pipelineDepth >= 4 && estBlocksPerSM > 0 && estBlocksPerSM < 3)
        return cacheRejectProfileMs("tc_pipeline_depth_low_occupancy",
                                    tcPipeRejectMs);
    }
  }

  // 当编译结果包含多个 kernel（切边 / split broadcast edges）时，
  // profiler 不会执行 host 侧 `linalg.fill` 初始化。这里通过 `--list-memrefs`
  // 推断中间 memref 并预填充，且每轮迭代重复填充，避免 warmup/iters 间累积。
  //
  // 注意：仅在用户请求 run-all-kernels 性能测量时启用该逻辑；
  // 单 kernel 配置仍由 `--init-ptr` 初始化输入，并假设输出会被完整覆盖。
  std::vector<std::pair<std::string, std::string>> fillSpecs;
  std::unordered_set<std::string> profilerSkipSyms;
  profilerSkipSyms.reserve(profilerInitPtrSyms.size());
  if (!profilerInitPtrSyms.empty())
    profilerSkipSyms.insert(profilerInitPtrSyms.begin(),
                            profilerInitPtrSyms.end());

  // 只要计划传 `--init-ptr`，就始终先查询 `--list-memrefs`，
  // 以过滤当前已编译子图中不存在的符号，避免误报 profiler 失败
  // （例如子图只含独立 fill kernel，或切边后的内部 kernel）。
  std::unordered_map<std::string, ListedMemrefInfo> listedMemrefs;
  bool hasListedMemrefs = false;
  if (!profilerInitPtrSyms.empty() || (opts.profile.runAllKernels && kernelCount > 1)) {
    std::string memrefsOutPath = dir + "/memrefs.txt";
    std::string listCmd;
    listCmd.reserve(512);
    listCmd.append(shellEscapeSingleQuotes(opts.profile.profilerBin));
    listCmd.push_back(' ');
    listCmd.append(shellEscapeSingleQuotes(nvvmPath));
    if (opts.profile.runAllKernels)
      listCmd.append(" --run-all-kernels");
    listCmd.append(" --list-memrefs");
    for (const auto &kv : profilerI64Overrides) {
      listCmd.append(" --i64 ");
      listCmd.append(shellEscapeSingleQuotes(kv.first + "=" +
                                            std::to_string(kv.second)));
    }
    listCmd.append(" > ");
    listCmd.append(shellEscapeSingleQuotes(memrefsOutPath));
    listCmd.append(" 2>&1");

    int listRc = runShellCommand(
        wrapWithTimeoutIfRequested(listCmd, opts.profile.timeoutSec));
    if (listRc == 0) {
      std::string memrefsText = readFileOrEmpty(memrefsOutPath);
      listedMemrefs = parseProfilerListMemrefsOutput(memrefsText);
      hasListedMemrefs = true;
    }
  }

  if (hasListedMemrefs && !profilerInitPtrSyms.empty()) {
    std::vector<std::string> filtered;
    filtered.reserve(profilerInitPtrSyms.size());
    for (const std::string &sym : profilerInitPtrSyms) {
      if (listedMemrefs.count(sym))
        filtered.push_back(sym);
    }
    profilerInitPtrSyms = std::move(filtered);
    profilerSkipSyms.clear();
    profilerSkipSyms.insert(profilerInitPtrSyms.begin(),
                            profilerInitPtrSyms.end());
  }

  if (opts.profile.runAllKernels && kernelCount > 1 && hasListedMemrefs) {
    fillSpecs = inferFillSpecsFromListedMemrefs(listedMemrefs, profilerSkipSyms,
                                                hasMaxReduction);
  }
  if (!fillSpecs.empty()) {
    // 这些参数会在下方追加到 profiler 命令。
  }

  std::string profCmd;
  profCmd.reserve(512);
  profCmd.append(shellEscapeSingleQuotes(opts.profile.profilerBin));
  profCmd.push_back(' ');
  profCmd.append(shellEscapeSingleQuotes(nvvmPath));
  if (opts.profile.runAllKernels)
    profCmd.append(" --run-all-kernels");
  profCmd.append(" --warmup ");
  profCmd.append(std::to_string(std::max(0, opts.profile.warmup)));
  profCmd.append(" --iters ");
  profCmd.append(std::to_string(std::max(1, opts.profile.iters)));
  for (const auto &kv : profilerI64Overrides) {
    profCmd.append(" --i64 ");
    profCmd.append(shellEscapeSingleQuotes(kv.first + "=" +
                                          std::to_string(kv.second)));
  }

  for (const std::string &sym : profilerInitPtrSyms) {
    profCmd.append(" --init-ptr ");
    profCmd.append(shellEscapeSingleQuotes(sym));
  }
  // 保持性能测量运行的初始化可复现。
  if (!profilerInitPtrSyms.empty()) {
    profCmd.append(" --init linear --seed 1");
  }

  if (!fillSpecs.empty()) {
    profCmd.append(" --fill-each-iter");
    for (const auto &kv : fillSpecs) {
      profCmd.append(" --fill ");
      profCmd.append(shellEscapeSingleQuotes(kv.first + "=" + kv.second));
    }
  }

  if (opts.profile.verbose)
    profCmd.append(" -v");
  profCmd.append(" > ");
  profCmd.append(shellEscapeSingleQuotes(profOutPath));
  profCmd.append(" 2>&1");

  auto profStart = std::chrono::steady_clock::now();
  int profRc = runShellCommand(
      wrapWithTimeoutIfRequested(profCmd, opts.profile.timeoutSec));
  auto profEnd = std::chrono::steady_clock::now();
  double profMs =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          profEnd - profStart)
          .count();
  if (opts.tracer) {
    llvm::json::Object f;
    f["rc"] = profRc;
    f["dur_ms"] = profMs;
    opts.tracer->event("profile.run", std::move(f), /*isVerbose=*/true);
  }
  if (profRc != 0) {
    const bool fallbackOnProfilerFail = getEnvInt64OrDefault(
                                            "WELDER_PROFILE_FALLBACK_ON_PROFILER_FAIL",
                                            (opts.arch.elementBytes <= 2 &&
                                             isMatmulSoftmaxSubgraph)
                                                ? 1
                                                : 0) != 0;
    const double profilerFailRejectMs = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_PROFILER_FAIL_REJECT_MS",
                                   featureRejectMs));
    if (opts.profile.verbose) {
      llvm::errs() << "profile: profiler failed (rc=" << profRc
                   << "), see: " << profOutPath << "\n";
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["rc"] = profRc;
      f["stage"] = "profiler";
      opts.tracer->event("profile.fail", std::move(f));
    }
    if (fallbackOnProfilerFail) {
      if (opts.tracer) {
        llvm::json::Object f;
        f["reason"] = "profiler_run_fail";
        f["profiler_rc"] = profRc;
        f["reject_ms"] = profilerFailRejectMs;
        opts.tracer->event("profile.profiler_fail_fallback", std::move(f),
                           /* isVerbose=*/true);
      }
      return cacheRejectProfileMs("profiler_run_fail", profilerFailRejectMs);
    }
    return std::nullopt;
  }

  std::string profOut = readFileOrEmpty(profOutPath);
  auto msOpt = parseAvgMsFromProfilerOutput(profOut);
  if (!msOpt) {
    const bool fallbackOnParseFail = getEnvInt64OrDefault(
                                         "WELDER_PROFILE_FALLBACK_ON_PARSE_FAIL",
                                         (opts.arch.elementBytes <= 2 &&
                                          isMatmulSoftmaxSubgraph)
                                             ? 1
                                             : 0) != 0;
    const double parseFailRejectMs = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_PARSE_FAIL_REJECT_MS",
                                   featureRejectMs));
    if (opts.profile.verbose) {
      llvm::errs() << "profile: cannot parse avg_ms from: " << profOutPath
                   << "\n";
    }
    if (opts.tracer) {
      llvm::json::Object f;
      f["stage"] = "parse";
      opts.tracer->event("profile.fail", std::move(f));
    }
    if (fallbackOnParseFail) {
      if (opts.tracer) {
        llvm::json::Object f;
        f["reason"] = "profiler_parse_fail";
        f["reject_ms"] = parseFailRejectMs;
        opts.tracer->event("profile.profiler_fail_fallback", std::move(f),
                           /* isVerbose=*/true);
      }
      return cacheRejectProfileMs("profiler_parse_fail", parseFailRejectMs);
    }
    return std::nullopt;
  }
  double measuredMs = *msOpt;
  double adjustedMs = measuredMs;
  if (isMatmulSoftmaxSubgraph) {
    const bool isTensorCoreCand =
        cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
    const int64_t softLocalOps = std::max<int64_t>(
        0, getEnvInt64OrDefault(isTensorCoreCand
                                    ? "WELDER_PROFILE_TC_MM_SM_SOFT_LOCAL_MEM_OPS"
                                    : "WELDER_PROFILE_MM_SM_SOFT_LOCAL_MEM_OPS",
                                isTensorCoreCand ? 8 : 12));
    const int64_t softLocalDepot = std::max<int64_t>(
        0, getEnvInt64OrDefault(isTensorCoreCand
                                    ? "WELDER_PROFILE_TC_MM_SM_SOFT_LOCAL_DEPOT_BYTES"
                                    : "WELDER_PROFILE_MM_SM_SOFT_LOCAL_DEPOT_BYTES",
                                isTensorCoreCand ? 2048 : 4096));
    const int64_t softRegB64 = std::max<int64_t>(
        0, getEnvInt64OrDefault(isTensorCoreCand
                                    ? "WELDER_PROFILE_TC_MM_SM_SOFT_REG_B64"
                                    : "WELDER_PROFILE_MM_SM_SOFT_REG_B64",
                                isTensorCoreCand ? 192 : 256));
    const int64_t softRegB32 = std::max<int64_t>(
        0, getEnvInt64OrDefault(isTensorCoreCand
                                    ? "WELDER_PROFILE_TC_MM_SM_SOFT_REG_B32"
                                    : "WELDER_PROFILE_MM_SM_SOFT_REG_B32",
                                isTensorCoreCand ? 96 : 128));
    const double localWeight = std::max(
        0.0, getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_PENALTY_LOCAL_WEIGHT",
                                   /*default=*/0.9));
    const double depotWeight = std::max(
        0.0, getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_PENALTY_DEPOT_WEIGHT",
                                   /*default=*/0.6));
    const double regWeight = std::max(
        0.0, getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_PENALTY_REG_WEIGHT",
                                   /*default=*/0.4));
    const double noAsyncExtra = std::max(
        0.0,
        getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_PENALTY_NO_ASYNC_EXTRA",
                              /*default=*/0.75));
    const double pipeExtra = std::max(
        0.0, getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_PENALTY_PIPE_EXTRA",
                                   /*default=*/0.3));
    const double maxPenaltyRatio = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_PROFILE_MM_SM_MAX_SPILL_PENALTY_RATIO",
                                   /*default=*/6.0));
    auto overRatio = [](int64_t value, int64_t softCap) -> double {
      if (softCap <= 0 || value <= softCap)
        return 0.0;
      return static_cast<double>(value - softCap) /
             static_cast<double>(std::max<int64_t>(1, softCap));
    };
    const double overLocal = overRatio(ptxStats.localMemOps(), softLocalOps);
    const double overDepot =
        overRatio(ptxStats.maxLocalDepotBytes, softLocalDepot);
    const double overReg = std::max(overRatio(ptxStats.maxRegB64, softRegB64),
                                    overRatio(ptxStats.maxRegB32, softRegB32));
    double penalty = localWeight * overLocal + depotWeight * overDepot +
                     regWeight * overReg;
    if (isTensorCoreCand && ptxStats.hasMmaSync && !ptxStats.hasCpAsync &&
        ptxStats.localMemOps() > 0)
      penalty += noAsyncExtra;
    if (cand.enableSoftwarePipelining && ptxStats.localMemOps() > softLocalOps)
      penalty += pipeExtra;
    const double penaltyRatio =
        std::min(maxPenaltyRatio, std::max(1.0, 1.0 + penalty));
    adjustedMs = measuredMs * penaltyRatio;
    if (opts.tracer && penaltyRatio > 1.0001) {
      llvm::json::Object f;
      f["measured_ms"] = measuredMs;
      f["adjusted_ms"] = adjustedMs;
      f["penalty_ratio"] = penaltyRatio;
      f["local_ops"] = ptxStats.localMemOps();
      f["local_depot"] = ptxStats.maxLocalDepotBytes;
      f["reg_b64"] = ptxStats.maxRegB64;
      f["reg_b32"] = ptxStats.maxRegB32;
      f["tc"] = isTensorCoreCand;
      opts.tracer->event("profile.adjust_spill_penalty", std::move(f),
                         /* isVerbose=*/true);
    }
  }

  if (opts.tracer) {
    llvm::json::Object f;
    f["avg_ms"] = measuredMs;
    f["adjusted_ms"] = adjustedMs;
    opts.tracer->event("profile.ok", std::move(f), /*isVerbose=*/true);
  }

  {
    std::lock_guard<std::mutex> lock(cacheMu);
    cache[key] = adjustedMs;
  }
  {
    std::lock_guard<std::mutex> lock(diskMu);
    appendDiskProfileCache(opts.profile.cachePath, key, adjustedMs);
  }
  return adjustedMs;
}

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input.mlir>"), llvm::cl::Required);
  static llvm::cl::opt<std::string> outputFilename(
      "output", llvm::cl::desc("Output MLIR path ('-' for stdout)"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> emitAfterPre(
      "emit-after-prebufferize",
      llvm::cl::desc("Write payload module after prebufferize transform"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> emitAfterBuf(
      "emit-after-bufferize",
      llvm::cl::desc("Write payload module after one-shot-bufferize"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> emitAfterPost(
      "emit-after-postbufferize",
      llvm::cl::desc("Write payload module after postbufferize transform"),
      llvm::cl::init(""));

  // [参数-映射] 线程 tile 和 block/thread 映射控制。
  static llvm::cl::opt<int64_t> threadTileM(
      "thread-tile-m",
      llvm::cl::desc("Thread tile size on M dimension (default 4)"),
      llvm::cl::init(4));
  static llvm::cl::opt<int64_t> threadTileN(
      "thread-tile-n",
      llvm::cl::desc("Thread tile size on N dimension (default 4)"),
      llvm::cl::init(4));
  static llvm::cl::opt<bool> swapBlockDims(
      "swap-block-dims",
      llvm::cl::desc("Swap (x,y) block/thread mapping order (experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> skipMapNestedForallToThreads(
      "skip-map-nested-forall-to-threads",
      llvm::cl::desc("Skip MapNestedForallToThreads in postbufferize (debug)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> skipMapForallToBlocks(
      "skip-map-forall-to-blocks",
      llvm::cl::desc("Skip MapForallToBlocks in postbufferize (debug)"),
      llvm::cl::init(false));

  // [参数-异步流水] async copy 与 software pipelining 开关。
  static llvm::cl::opt<bool> enableAsyncCopy(
      "enable-async-copy",
      llvm::cl::desc(
          "Phase 14: enable NVGPU async copy (cp.async) for global->shared "
          "copies when possible (experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableSoftwarePipelining(
      "enable-software-pipelining",
      llvm::cl::desc(
          "Phase 14: enable software pipelining for scf.for loops with shared "
          "memory copies (requires workgroup multibuffering; experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<int64_t> pipelineDepth(
      "pipeline-depth",
      llvm::cl::desc("Software pipeline depth (stages, default 2)"),
      llvm::cl::init(2));
  static llvm::cl::opt<bool> pipelinePeelEpilogue(
      "pipeline-peel-epilogue",
      llvm::cl::desc("Peel epilogue when pipelining (more robust)"),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> pipelineSetAsyncWaitGroups(
      "pipeline-set-async-wait-groups",
      llvm::cl::desc(
          "Set cp.async wait groups in-flight (emit wait_group N>0) instead of "
          "the conservative wait_group 0 default"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> asyncBypassL1(
      "async-bypass-l1",
      llvm::cl::desc("Set bypass_l1 hint on eligible async copies (16B)"),
      llvm::cl::init(true));

  // [参数-TensorCore/共享内存布局] TC 选择和 shared padding。
  static llvm::cl::opt<int64_t> workgroupPadLastDim(
      "workgroup-pad-last-dim",
      llvm::cl::desc("Paper/Welder parity: shared-memory stride-map padding "
                     "(pad last dim by N elements). This is emitted as "
                     "{welder.workgroup_pad_last_dim} attrs on linalg.matmul "
                     "so the workgroup pass can apply per-matmul padding."),
      llvm::cl::init(0));
  static llvm::cl::opt<bool> workgroupPadLastDimMatmulOnly(
      "workgroup-pad-last-dim-matmul-only",
      llvm::cl::desc("Only apply workgroup-pad-last-dim to linalg.matmul operand "
                     "tiles (A/B) to match TCPolicy stride-map semantics."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableTensorCoreTf32(
      "enable-tensorcore-tf32",
      llvm::cl::desc(
          "Enable a minimal TensorCore (TF32) matmul path via "
          "transform.nvgpu.rewrite_matmul_as_mma_sync (experimental; matmul-only)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableTensorCoreF16(
      "enable-tensorcore-f16",
      llvm::cl::desc(
          "Enable a minimal TensorCore (f16) matmul path via "
          "transform.nvgpu.rewrite_matmul_as_mma_sync (experimental; matmul-only)"),
      llvm::cl::init(false));

  // [参数-Solver硬件模型与候选空间] 控制求解器评分与候选枚举。
  static llvm::cl::opt<int64_t> smemBytes(
      "smem-bytes", llvm::cl::desc("Shared memory capacity per block (bytes)"),
      llvm::cl::init(48 * 1024));
  static llvm::cl::opt<int64_t> numSM("num-sm",
                               llvm::cl::desc("Number of SMs (approx)"),
                               llvm::cl::init(80));
  static llvm::cl::opt<int64_t> maxBlocksPerSM(
      "max-blocks-per-sm",
      llvm::cl::desc("Upper bound for blocks/SM used in wave estimation"),
      llvm::cl::init(4));
  static llvm::cl::opt<int64_t> warpSize(
      "warp-size",
      llvm::cl::desc("Paper/Welder parity: warp size (default 32)"),
      llvm::cl::init(32));
  static llvm::cl::opt<int64_t> smPartition(
      "sm-partition",
      llvm::cl::desc("Paper/Welder parity: SM partition heuristic (default 4)"),
      llvm::cl::init(4));
  static llvm::cl::opt<int64_t> maxSmemUsageBytes(
      "max-smem-usage-bytes",
      llvm::cl::desc("Paper/Welder parity: max shared usage per SM (bytes). "
                     "0 means 2*smem-bytes."),
      llvm::cl::init(0));
  static llvm::cl::opt<int64_t> globalTransactionBytes(
      "global-transaction-bytes",
      llvm::cl::desc("Paper-aligned: global memory transaction width in bytes "
                     "(default 128)"),
      llvm::cl::init(128));
  static llvm::cl::opt<int64_t> globalReadTransactionBytes(
      "global-read-transaction-bytes",
      llvm::cl::desc("Paper/Welder parity: global read transaction width in bytes "
                     "(default 128)"),
      llvm::cl::init(128));
  static llvm::cl::opt<int64_t> globalWriteTransactionBytes(
      "global-write-transaction-bytes",
      llvm::cl::desc("Paper/Welder parity: global write transaction width in bytes "
                     "(default 32)"),
      llvm::cl::init(32));
  static llvm::cl::opt<int64_t> maxThreadsPerSM(
      "max-threads-per-sm",
      llvm::cl::desc("Approx max resident threads per SM (for occupancy heuristic)"),
      llvm::cl::init(2048));
  static llvm::cl::opt<std::string> candidatesMN(
      "candidates-mn",
      llvm::cl::desc("Candidate tile sizes for M/N (csv), e.g. 32,64,128"),
      llvm::cl::init("32,64,128"));
  static llvm::cl::opt<std::string> candidatesK(
      "candidates-k",
      llvm::cl::desc("Candidate tile sizes for K (csv), e.g. 8,16,32"),
      llvm::cl::init("8,16,32"));
  static llvm::cl::opt<bool> autoCandidates(
      "auto-candidates",
      llvm::cl::desc("Paper-aligned: generate candidate tiles from problem/hardware "
                     "instead of relying on candidates-mn/k"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRegisterLevelSchedule(
      "enable-register-level-schedule",
      llvm::cl::desc("Paper-aligned: include per-thread tiles (threadTileM/N) in "
                     "the candidate space (affects occupancy/waves estimate)"),
      llvm::cl::init(false));
  static llvm::cl::opt<std::string> candidatesThreadMN(
      "candidates-thread-mn",
      llvm::cl::desc("Candidate per-thread tiles for M/N (csv), e.g. 1,2,4,8"),
      llvm::cl::init("1,2,4,8"));
  static llvm::cl::opt<bool> requirePerfectTiling(
      "require-perfect-tiling",
      llvm::cl::desc("Require M%tm==0 && N%tn==0 && K%tk==0"),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> assumeFusedRelu(
      "assume-fused-relu",
      llvm::cl::desc("Assume MatMul->consumer is fused in one kernel"),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> enableFootprintInference(
      "enable-footprint-inference",
      llvm::cl::desc("Use indexing_maps-based footprint inference to compute "
                     "traffic (experimental; keeps old hardcoded model as "
                     "fallback)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableTilePropagation(
      "enable-tile-propagation",
      llvm::cl::desc("Build TileGraph and run Welder-style consumer-driven tile "
                     "propagation (experimental; used to validate fusion "
                     "assumptions)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableGlobalTraffic(
      "enable-global-traffic",
      llvm::cl::desc("Phase A: compute whole-graph traffic assuming fully fused "
                     "(counts only graph-input reads and sink writes; "
                     "requires tile propagation; experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableCutEdges(
      "enable-cut-edges",
      llvm::cl::desc("Phase 13A/B (experimental): enable cut-edges in tile "
                     "propagation and codegen (may emit multiple kernels)"),
      llvm::cl::init(false));
  // [参数-行归约链融合] softmax/layernorm 风格链路的专用优化与防护开关。
  static llvm::cl::opt<bool> reductionChainSplitBroadcastEdges(
      "reduction-chain-split-broadcast-edges",
      llvm::cl::desc(
          "Codegen heuristic (experimental): for row-reduction graphs, cut "
          "broadcast edges from reduction-derived 1D values into 2D elementwise "
          "consumers. This avoids accidentally leaving the 1D chain on the host "
          "in the current generic codegen and makes the behavior A/B testable."),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> enableRowReductionChainReuseFusion(
      "enable-row-reduction-chain-reuse-fusion",
      llvm::cl::desc(
          "Codegen (experimental): enable a minimal staged schedule fixup for "
          "row-reduction chains (Softmax/LayerNorm-style) to keep 1D broadcast "
          "intermediates inside the fused gpu.launch with correct barriers."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionMeanScaleFusion(
      "enable-row-reduction-mean-scale-fusion",
      llvm::cl::desc(
          "Codegen (experimental): fuse mean scaling (sum * cst) into the "
          "subsequent row-reduction (e.g., LayerNorm sumsq), avoiding a "
          "separate 1D elementwise stage."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionSquareFusion(
      "enable-row-reduction-square-fusion",
      llvm::cl::desc(
          "Codegen (experimental): fuse a simple (x-mean)^2 elementwise stage "
          "into a subsequent row-reduction (LayerNorm-style). Requires "
          "--enable-row-reduction-chain-reuse-fusion and currently depends on "
          "row-reduction tiling support for multi-input reductions."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionInputPromotion(
      "enable-row-reduction-input-promotion",
      llvm::cl::desc(
          "Codegen (experimental): promote row-reduction input operands to "
          "workgroup (shared) memory to reduce global reads."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionInputPromotionVectorize(
      "enable-row-reduction-input-promotion-vectorize",
      llvm::cl::desc(
          "Codegen (experimental): vectorize the cooperative global->shared "
          "staging copy used by row-reduction input promotion (uses "
          "vector.transfer when safe). Requires --enable-row-reduction-input-promotion."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionWarp(
      "enable-row-reduction-warp",
      llvm::cl::desc(
          "Codegen (experimental): prefer warp-level reduction for row-reduction "
          "kernels (single warp per row) to reduce synchronization overhead."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionVectorize(
      "enable-row-reduction-vectorize",
      llvm::cl::desc(
          "Codegen (experimental): vectorize row-reduction elementwise stages "
          "(e.g., exp/div) on 2D tiles."),
      llvm::cl::init(false));
  static llvm::cl::opt<int64_t> rowReductionVectorWidth(
      "row-reduction-vector-width",
      llvm::cl::desc("Row-reduction elementwise vector width (0=auto)"),
      llvm::cl::init(0));
  static llvm::cl::opt<int64_t> rowReductionThreadsX(
      "row-reduction-threads-x",
      llvm::cl::desc("Row-reduction threads along X (0=auto)"),
      llvm::cl::init(0));
  static llvm::cl::opt<bool> enableRowReductionRelaxBarriers(
      "enable-row-reduction-relax-barriers",
      llvm::cl::desc(
          "Codegen (experimental): allow redundant barrier cleanup for "
          "row-reduction staging."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableRowReductionSkipCombineBarrier(
      "enable-row-reduction-skip-combine-barrier",
      llvm::cl::desc(
          "Codegen (experimental): skip barrier insertion after combining "
          "reductions (unsafe; profiling only)."),
      llvm::cl::init(false));
  static llvm::cl::opt<int64_t> rowReductionInputVectorWidth(
      "row-reduction-input-vector-width",
      llvm::cl::desc("Row-reduction input staging vector width (0=auto)"),
      llvm::cl::init(0));
  static llvm::cl::opt<bool> enableRowReductionCombineVectorize(
      "enable-row-reduction-combine-vectorize",
      llvm::cl::desc(
          "Codegen (experimental): vectorize row-reduction combining op."),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableMatmulSoftmaxSharedReuseFusion(
      "enable-matmul-softmax-shared-reuse-fusion",
      llvm::cl::desc(
          "Codegen (experimental): canonicalize Matmul->Softmax fusion so the "
          "matmul tile is computed once and reused by max/exp/sum/div stages, "
          "avoiding duplicated chains and huge per-thread local buffers."),
      llvm::cl::init(false));
  // [参数-Paper/Profiling] 论文对齐调度、Top-K 与 profile 配置。
  static llvm::cl::opt<bool> enableTwoLevelSchedule(
      "enable-two-level-schedule",
      llvm::cl::desc("Phase 14 (paper alignment, skeleton): enable 2-level "
                     "tile-graph scheduling (global<->shared) in solver-side "
                     "scoring/filters (experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enablePaperSchedule(
      "enable-paper-schedule",
      llvm::cl::desc(
          "Welder paper (Figure 7): enable GraphConnecting + SubGraphTiling "
          "in solver (uses traffic-based latency estimate instead of profiling; "
          "experimental)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> paperRecursiveRegisterLevel(
      "paper-recursive-register-level",
      llvm::cl::desc(
          "Paper-aligned: evaluate shared tiles via an inner register-level "
          "search (threadTileM/N + codegen knobs), and use the best inner config "
          "as the score for GraphConnecting and final ranking"),
      llvm::cl::init(true));
  static llvm::cl::opt<int> paperRecursiveInnerMinLevelExclusive(
      "paper-recursive-inner-min-level-exclusive",
      llvm::cl::desc(
          "Recursive SubGraphTiling inner-stage boundary (minLevelExclusive). "
          "<=0 means auto(max(1,max-connect-level-1)); >0 forces an explicit "
          "boundary."),
      llvm::cl::init(0));
  static llvm::cl::opt<int> paperRecursiveMaxStages(
      "paper-recursive-max-stages",
      llvm::cl::desc(
          "Cap recursive SubGraphTiling stage depth when max-connect-level>2. "
          "<=0 keeps legacy auto behavior; >0 allows at most N recursive "
          "stage windows."),
      llvm::cl::init(0));
  static llvm::cl::opt<bool> paperStrict(
      "paper-strict",
      llvm::cl::desc("Paper-aligned strict mode: rank by MemTraffic bytes only "
                     "and prune configs that fail compile/profile"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> paperExpandReductionTile(
      "paper-expand-reduction-tile",
      llvm::cl::desc(
          "Paper-aligned: greedily enlarge reduction tiles (e.g. K) under shared "
          "memory constraints"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> pruneOnProfileFailure(
      "prune-on-profile-failure",
      llvm::cl::desc("When profiling is enabled, drop configs that fail "
                     "compile/profile instead of falling back to heuristics"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableCoalescingPenalty(
      "enable-coalescing-penalty",
      llvm::cl::desc("Paper-aligned: account for uncoalesced global memory access "
                     "by charging extra memory transactions in MemTraffic "
                     "(default: true)"),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> solverVerboseCost(
      "solver-verbose-cost",
      llvm::cl::desc("Print solver-side cost breakdown for selected configs "
                     "(traffic/footprint/waves, plus profiling if enabled)"),
      llvm::cl::init(false));
  static llvm::cl::opt<int64_t> scheduleTopK(
      "schedule-topk",
      llvm::cl::desc("Top-K configs kept in SubGraphTiling (paper k)"),
      llvm::cl::init(8));
  static llvm::cl::opt<int> maxConnectLevel(
      "max-connect-level",
      llvm::cl::desc(
          "Max connect level tried in GraphConnecting (0=cut to global, 1=fuse "
          "in one kernel; default 1)"),
      llvm::cl::init(1));
  static llvm::cl::opt<bool> codegenFromKernelAttrs(
      "codegen-from-kernel-attrs",
      llvm::cl::desc(
          "Codegen only: use existing {welder.kernel_root, welder.kernel_id} "
          "attrs to decide multi-kernel fusion boundaries (used by paper "
          "schedule profiling harness). This bypasses solver-side cut-edge "
          "propagation."),
      llvm::cl::init(false));
  static llvm::cl::opt<int64_t> forceTileM(
      "force-tile-m",
      llvm::cl::desc(
          "Force TILE_M (used by profiling/codegen-from-kernel-attrs). <=0 "
          "means unset."),
      llvm::cl::init(0));
  static llvm::cl::opt<int64_t> forceTileN(
      "force-tile-n",
      llvm::cl::desc(
          "Force TILE_N (used by profiling/codegen-from-kernel-attrs). <=0 "
          "means unset."),
      llvm::cl::init(0));
  static llvm::cl::opt<int64_t> forceTileK(
      "force-tile-k",
      llvm::cl::desc(
          "Force TILE_K (used by profiling/codegen-from-kernel-attrs). <=0 "
          "means unset."),
      llvm::cl::init(0));
  static llvm::cl::opt<bool> enableProfiling(
      "enable-profiling",
      llvm::cl::desc("Enable paper-aligned hardware profiling (d.Profile)"),
      llvm::cl::init(false));
  static llvm::cl::opt<int> profileWarmup(
      "profile-warmup",
      llvm::cl::desc("CUDA event warmup launches (default 10)"),
      llvm::cl::init(10));
  static llvm::cl::opt<int> profileIters(
      "profile-iters",
      llvm::cl::desc("CUDA event measured launches (default 100)"),
      llvm::cl::init(100));
  static llvm::cl::opt<int> profileMaxParallelJobs(
      "profile-max-parallel-jobs",
      llvm::cl::desc("Max parallel profiling jobs (paper compilation speedup; "
                     "default 1)"),
      llvm::cl::init(1));
  static llvm::cl::opt<int> profileTimeoutSec(
      "profile-timeout-sec",
      llvm::cl::desc("Timeout (seconds) for compile/profile per config "
                     "(default 0 = disabled; best-effort)"),
      llvm::cl::init(0));
  static llvm::cl::opt<std::string> profileCompilerToNvvm(
      "profile-compiler-to-nvvm",
      llvm::cl::desc("Path to compiler/run_welder_to_nvvm_isa.sh"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> profileProfilerBin(
      "profile-profiler-bin",
      llvm::cl::desc("Path to welder-profiler binary"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> profileCachePath(
      "profile-cache",
      llvm::cl::desc("Append-only cache file for profile results"),
      llvm::cl::init(""));
  static llvm::cl::opt<bool> profileVerbose(
      "profile-verbose",
      llvm::cl::desc("Verbose profiling logs (keeps temp artifacts)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableGenericProblem(
      "enable-generic-problem",
      llvm::cl::desc(
          "Use generic loop-based problem analysis/enumeration (experimental; "
          "allows non-matmul ops like conv; keeps old matmul path as default)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> enableGenericFusion(
      "enable-generic-fusion",
      llvm::cl::desc(
          "Phase 11 (experimental): try consumer-driven tile+fuse in generic "
          "mode (assumes a single linalg consumer like ReLU)"),
      llvm::cl::init(false));
  // [参数-观测] 运行时 trace 与 pass 级 trace。
  static llvm::cl::opt<bool> trace(
      "trace",
      llvm::cl::desc("Real-time trace to stderr (stage timing + progress)"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> traceVerbose(
      "trace-verbose",
      llvm::cl::desc("Verbose trace (per-candidate/per-pass events)"),
      llvm::cl::init(false));
  static llvm::cl::opt<std::string> traceFile(
      "trace-file",
      llvm::cl::desc(
          "Write JSONL trace events to a file (one JSON object per line)"),
      llvm::cl::init(""));
  static llvm::cl::opt<bool> traceFileAppend(
      "trace-file-append",
      llvm::cl::desc("Append to --trace-file instead of truncating"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> tracePasses(
      "trace-passes",
      llvm::cl::desc("Trace individual MLIR passes (start/end + duration)"),
      llvm::cl::init(false));

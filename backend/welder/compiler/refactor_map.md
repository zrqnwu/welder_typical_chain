# Welder Compiler Refactor Map

## Purpose
- Record the new module split after `WelderSolverLib.cpp`, `WelderCompiler.cpp`, and `WelderProfiler.cpp` refactor.
- Provide a fast reading path for review/interview.

## Compiler Directory Layout
- `driver/`
  - CLI/CAPI entrypoints, run config, top-level orchestration.
- `dispatch/`
  - mode routing, anchor discovery, tile decisions, fusion pair and dispatch plan.
- `transforms/`
  - async/generic/matmul/row-reduction transform libraries.
- `postprocess/`
  - postbufferize fixups, memory reuse, cleanup, matmul-softmax canonicalization.
- `solver/`
  - solver CLI/CAPI/library, candidate generation, cost model, profiler and report writer.
- `common/`
  - shared infra (`AffineIntervalEvaluator`, `WelderTrace`).

## Entry Points
- `driver/WelderCompiler.cpp`: top-level compiler CLI and orchestration.
- `solver/WelderSolver.cpp`: solver CLI entry that builds `SolveOptions` and calls solver library.
- `solver/WelderProfiler.cpp`: CUDA runtime profiler CLI for generated NVVM runner modules.

## Solver (`solver/WelderSolverLib.cpp`) Module Map
- `WelderSolverTwoLevelCore.h`
  - 2-level GraphConnecting core propagation path.
  - Main symbols: `buildRootParallelExtents2Level`, `propagateTilesBackwardTwoLevel`, `applyGraphConnecting2Level`.
- `WelderSolverCandidatePolicyHelpers.h`
  - candidate legality/policy helpers and codegen search expansion primitives.
  - includes TensorCore feasibility, row-reduction/codegen knob filtering, traffic/coalescing helpers used by candidate construction.
- `WelderSolverTrafficCostCore.h`
  - shared footprint best-fit allocator and traffic/cost model core.
  - main symbols: `BestFitAllocator`, `computeSharedFootprintBestFitPaper`, `computeGlobalTrafficForSubgraph`, `fillCostAndScoreFromPaperModel`.
- Existing split modules retained:
  - `WelderSolverGraphConnectingPaperGlobalShared.h`
  - `WelderSolverSubGraphTilingPaperGlobalShared.h`
  - `WelderSolverRecursiveStagesCore.h`
  - `WelderSolverRecursiveStageTopKCandidates.h`
  - `WelderSolverRecursiveStageTopKRegisterTiles.h`
  - `WelderSolverRecursiveStageTopKSharedCandidates.h`
  - `WelderSolverProfileRetryHelpers.h`
  - `WelderSolverProfileCompileAndRun.h`
  - `WelderSolverGraphAnalysisAndFootprintInfer.h`
  - `WelderSolverPhaseAGridReuseHelpers.h`
  - `WelderSolverPhaseAFootprintTrafficHelpers.h`
  - `WelderSolverDumpPlanHelpers.h`
  - `WelderSolverDumpReports.h`

## Compiler (`driver/WelderCompiler.cpp`) Module Map
- `driver/WelderCompilerPassTraceAndEnv.h/.cpp`
  - env knobs + pass trace instrumentation class（已从 `.h` 迁移）。
- `transforms/WelderCompilerAsyncPipelineHelpers.h/.cpp`
  - async copy control-flow flattening and software pipelining helpers.
  - includes `padWorkgroupAllocs` used by postbufferize hooks（已从 `.h` 迁移）。
- `driver/WelderCompilerDriverHelpers.h/.cpp`
  - Driver 级通用阶段 helper：`parseInputModule` / `buildSolveOptions` /
    `deriveEffectiveRowReductionKnobs` / `loadTransformLibraryIntoDialect` /
    `runLoweringPipelineAndEmit`。
- `driver/WelderCompilerCliOptions.h`
  - 仅承载 CLI 选项定义片段（在 `welderCompilerMain` 内 include），
    避免主流程被大量参数声明淹没。
- `dispatch/WelderCompilerModeDispatch.h/.cpp`
  - 三条 transform library 构建路径（kernel-attrs / generic / matmul）的路由入口（薄路由）。
- `dispatch/WelderCompilerModeDispatchKernelAttrsBranch.cpp`
  - `--codegen-from-kernel-attrs` 路径实现（复用已标注 kernel attrs）。
- `dispatch/WelderCompilerModeDispatchGenericBranch.cpp`
  - generic problem 路径实现（含 cut-edges / generic fusion）。
- `dispatch/WelderCompilerModeDispatchMatmulBranch.cpp`
  - matmul 专项路径实现（solver + anchor + matmul transform library）。
- `dispatch/WelderCompilerModeDispatch*BranchBody.h`
  - 对应三条分支的主体代码块（由各自 `.cpp` 包装调用）。
- `postprocess/WelderCompilerMemoryReuseTransforms.h`
  - row-reduction/shared-memory reuse transforms public API.
  - implementation is isolated in `WelderCompilerMemoryReuseTransforms.cpp`
    + `WelderCompilerMemoryReuseTransformsImpl.h` (internal).
  - internal implementation is further split by sub-stage:
    `WelderCompilerMemoryReuseTransformsStage{1,2,3}.h`.
- `transforms/WelderCompilerAsyncCopyRewrite.h/.cpp`
  - global->workgroup `linalg.copy` 到 NVGPU async copy 的重写（已从 `.h` 迁移）。
- `transforms/WelderCompilerRowReductionInputPromotion.h/.cpp`
  - 行归约输入提升（global/subview -> workgroup）逻辑（已从 `.h` 迁移）。
- `postprocess/WelderCompilerMatmulSoftmaxCanonicalize.h`
  - matmul->softmax shared reuse canonicalization public API.
  - implementation is isolated in `WelderCompilerMatmulSoftmaxCanonicalize.cpp`
    + `WelderCompilerMatmulSoftmaxCanonicalizeImpl.h` (internal).
  - internal implementation is further split by sub-stage:
    `WelderCompilerMatmulSoftmaxCanonicalizeStage{1,2,3}.h`.
- `postprocess/WelderCompilerPostprocessCleanup.h`
  - barrier/cleanup/postprocess public API.
  - implementation is isolated in `WelderCompilerPostprocessCleanup.cpp`
    + `WelderCompilerPostprocessCleanupImpl.h` (internal).
  - internal implementation is further split by sub-stage:
    `WelderCompilerPostprocessCleanupStage{1,2,3,4}.h`.
- `dispatch/WelderCompilerFusionAnchors.h/.cpp`
  - fusion anchor discovery and graph anchoring helpers（已从 `.h` 迁移）。
- `transforms/WelderCompilerMatmulTransformLibrary.h/.cpp`
  - transform library builders for matmul path（已从 `.h` 迁移）。
- `transforms/WelderCompilerGenericTransformAndFusion.h/.cpp`
  - generic transform and fusion library + shared fusion pair normalization
    types/helpers（已从 `.h` 迁移）。
- `transforms/WelderCompilerGenericCutEdgesTransformLibrary.h/.cpp`
  - cut-edge aware generic transform library（已从 `.h` 迁移）。
- `driver/WelderCompiler.cpp`
  - 仅保留主流程编排与 postbufferize hook 装配，阶段边界清晰：
    1) 参数解析与 tracing
    2) dialect 注册与输入解析
    3) solve 选项与 row-reduction 生效开关推导
    4) ModeDispatch 构建 transform library
    5) lowering pipeline 执行与产物输出

## Typical Chain Runtime Sequence (Single Run)
1. `wtc-compiler` 前端：
   - `canonicalize` -> `tagging` -> `search` -> `compile` -> `validate postbufferize artifacts`
2. `search` 阶段：
   - `shell|process_chain|api` 三种 backend 模式选一；
   - 默认 full+api 会回退 `search` 到 `process_chain`，减少同进程 API 风险。
3. `compile` 阶段：
   - 产物按阶段落盘：`03.after_postbufferize.mlir` ->
     `04.after_workgroup_launch.mlir` -> `04c.after_linalg_to_loops.mlir` ->
     `05.out.nvvm.runnable.mlir`。
4. `runtime validate` 阶段：
   - 校验/统计上述产物并写 `postbufferize_report.txt`。

## Per-Stage Sanity Commands
1. 前端可执行构建：
   - `cmake --build /home/zhangruiqi/welder_typical_chain/build -j4`
2. 后端编译器构建：
   - `cmake --build /home/zhangruiqi/welder_typical_chain/backend/welder/compiler/build -j4 --target welder-compiler welder-compile-capi`
3. 全链路回归（含性能护栏）：
   - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash /home/zhangruiqi/welder_typical_chain/bench/run_all_stages.sh`
   - latest check: all PASS, `ab_speedup=1.1753x`.

## Profiler (`solver/WelderProfiler.cpp`) Module Map
- `solver/WelderProfilerParsing.h`
  - MLIR text parsing, launch parsing, constants extraction, PTX/JIT helpers.
- `solver/WelderProfilerEval.h`
  - token evaluators and f32<->f16 conversion helpers.
- `solver/WelderProfilerRuntime.h`
  - memref descriptor inference, dump metadata, host init helpers.

## Suggested Reading Order (Interview)
1. `driver/WelderCompiler.cpp` (`main`) to understand end-to-end orchestration.
2. `solver/WelderSolver.cpp` + `solver/WelderSolverLib.cpp` to understand scheduling search/cost model flow.
3. `solver/WelderSolverTwoLevelCore.h` + `solver/WelderSolverSubGraphTilingPaperGlobalShared.h` for GraphConnecting/SubGraphTiling logic.
4. `solver/WelderSolverTrafficCostCore.h` for footprint/traffic/cost objective.
5. `postprocess/WelderCompilerMemoryReuseTransforms.h` + `postprocess/WelderCompilerMatmulSoftmaxCanonicalize.h` for realized lowering/codegen optimizations.
6. `solver/WelderProfiler.cpp` + `solver/WelderProfiler*.h` for measurement and verification loop.

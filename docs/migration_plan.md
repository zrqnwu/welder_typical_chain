# Migration Plan (from `welder_try`)

## Stage 0 (done)
- New standalone skeleton project.
- Prototype `wtc-compiler` with explicit phase boundaries.

## Stage 1 (done)
- Lock one mode family: generic problem + cut edges.
- Keep typical-chain-only scope.

## Stage 2 (done for in-repo packaging)
- Search backend source is vendored into this repo under `backend/welder/`.
- `scheduler/` exposes stable outputs:
  - `best.json`
  - `candidates.tsv`
- `scheduler/` default path is now `process_chain` orchestration (with `shell`
  fallback mode, compatible with legacy name `inprocess`).
- Added `api` mode for solver: same-process call into backend `welder-solver-capi` shared library (no solver subprocess).

## Stage 3 (done for in-repo packaging)
- Compile/lowering backend source is vendored into this repo.
- `transform/` exposes stable compile artifacts and variant control.
- `transform/` default path is now `process_chain` orchestration (with `shell`
  fallback mode, compatible with legacy name `inprocess`).
- Added `api` mode for compile/lowering via backend `welder-compile-capi`.

## Stage 4 (done for orchestration)
- `pipeline/` supports explicit modes:
  - `--search-only`
  - `--compile-only`
  - full default mode
- Tile source priority is explicit:
  1. `--best-json`
  2. search result
  3. explicit tile overrides
  4. built-in defaults

## Stage 5 (done for bench entrypoints)
- `bench/run_search.sh` calls `wtc-compiler --search-only`.
- `bench/run_ab.sh` calls `wtc-compiler` + vendored profiler.
- No forwarding to legacy bench scripts.

## Stage 6 (done)
- Canonicalize stage now supports named softmax decomposition in-repo:
  - input containing `linalg.softmax` is decomposed to `max/exp/sum/div` generic chain
  - then enters the same tagging/search/compile pipeline as decomposed input

## Stage 7 (done with API compile hardening)
- Compile API now runs compiler stage in-process (`welderCompilerMain`) instead of
  shelling out to `welder-compiler`.
- Hardened repeat-time behavior for pure API full mode:
  - `api + pure-api-full + repeat>1` runs as same-process loop.
  - backend compiler option parsing uses repeat-safe reset semantics.
- Full mode keeps a stability-first default:
  - when `--backend-mode api` and not `--pure-api-full`, search falls back to
    `process_chain` mode before compile(api).

## Stage 8 (done for acceptance automation)
- Added one-command acceptance runner:
  - `bench/run_all_stages.sh`
- Runner covers:
  - regression checks
  - API compile pair flow
  - pure API full repeat stress
  - baseline/fused AB profiling

## Stage 9 (done for transform modularization)
- Split `transform` compile implementation by backend mode:
  - shell path
  - process_chain path
  - api path
- Kept a small dispatch entry (`BuildCutEdgesTransform.cpp`) for validation and mode routing.
- Reduced monolithic transform compile file complexity while keeping runtime behavior unchanged.

## Stage 10 (done for scheduler modularization)
- Split `scheduler` search implementation by backend mode:
  - shell path
  - process_chain path
  - api path
- Kept a small dispatch entry (`Search.cpp`) for artifact normalization and mode routing.
- Reduced monolithic search file complexity while keeping runtime behavior unchanged.

## Stage 11 (done for baseline guardrails)
- Added baseline pin script:
  - `bench/pin_baseline.sh`
- Added performance regression guard script (threshold based):
  - `bench/check_perf_guard.sh`
- Added optional perf guard stage in acceptance runner:
  - `CHECK_PERF_GUARD=1 bash bench/run_all_stages.sh`

## Stage 12 (done for compiler entry slimming - step 1)
- Introduced `AnchorDiscovery` module for cut-edges path:
  - `WelderCompilerAnchorDiscovery.h/.cpp`
- Moved topology and initial root discovery helpers out of
  `WelderCompilerModeDispatchBody.inc`:
  - non-trivial linalg node filter
  - topo order construction
  - topo index construction
  - initial kernel root collection
- Wired `WelderCompiler.cpp` / build targets to use this module, keeping behavior unchanged.

## Stage 13 (done for compiler entry slimming - step 2)
- Introduced `TileDecision` module:
  - `WelderCompilerTileDecision.h/.cpp`
- Introduced `FusionPairBuild` module:
  - `WelderCompilerFusionPairBuild.h/.cpp`
- Moved thread/block tile helpers from `WelderCompilerModeDispatchBody.inc`:
  - divisible thread tile fallback
  - ceil/exact block-dim derivation
- Moved fusion-pair construction out of `WelderCompilerModeDispatchBody.inc`:
  - `codegen-from-kernel-attrs` thread-fusion pair build
  - cut-edges row-reduction fusion pair build
  - cut-edges register-level thread-fusion pair build
- Wired backend build targets (`welder-compiler`, `welder-compile-capi`) to new modules,
  keeping runtime behavior and acceptance checks unchanged.

## Stage 14 (done for compiler entry slimming - step 3 partial)
- Introduced `DispatchPlan` module:
  - `WelderCompilerDispatchPlan.h/.cpp`
- Moved per-kernel producer ordering logic out of `WelderCompilerModeDispatchBody.inc`:
  - deterministic sink->source producer ordering (reverse-topo)
  - assignment back into `KernelSpec::orderedProducerNodeIds`
- Kept existing kernel assignment semantics unchanged.

## Stage 15 (done for solver layering - step 1 partial)
- Introduced `CandidateGenerator` module for solver CLI orchestration:
  - `WelderSolverCandidateGenerator.h/.cpp`
  - unified wrappers for `solveGeneric` / `solve` invocation and failure hints.
- Introduced `ReportWriter` module for solver outputs:
  - `WelderSolverReportWriter.h/.cpp`
  - best summary JSON writer
  - sorted candidates TSV writer
- Updated `WelderSolver.cpp` to delegate candidate-run and report dumping to
  the new modules, reducing monolithic CLI body complexity while preserving
  ranking/cost logic in `welder-solver-lib`.

## Stage 16 (done for solver layering - step 2 partial)
- Introduced `ProfilerRunner` helper for solver profiling tool-path setup:
  - `WelderSolverProfilerRunner.h/.cpp`
  - centralizes profiler binary / script path resolution from CLI argv.
- Introduced `CostModel` output helper module:
  - `WelderSolverCostModel.h/.cpp`
  - centralizes candidate summary and top-k score printing.
- Updated `WelderSolver.cpp` to delegate profiler-path resolution and cost
  summary output responsibilities to dedicated modules, further reducing
  CLI-entry coupling.
- Kept core ranking formulas and recursive scheduling logic unchanged in
  `WelderSolverLib.cpp` and related `.inc` files for low-risk behavior parity.

## Stage 17 (done for perf guard stability hardening)
- Hardened `bench/check_perf_guard.sh` speedup check policy:
  - Added `SPEEDUP_CHECK_MODE=auto|strict|off` (default `auto`).
  - In `auto` mode, skip speedup lower-bound checks when baseline runtime
    improves beyond threshold, preventing false regression alarms caused by
    denominator drift.
- Existing absolute latency checks (`baseline.avg_ms`, `fused.avg_ms`) remain
  mandatory, preserving regression sensitivity on real slowdowns.

## Stage 18 (done for compile(api) sessionization in frontend runtime)
- Introduced compile API session wrapper:
  - `compiler/include/wtc/transform/internal/CompileApiSession.h`
  - `compiler/transform/CompileApiSession.cpp`
- Refactored `BuildCutEdgesTransformApi.cpp` to delegate `dlopen/dlsym` and
  invocation lifecycle to `CompileApiSession`:
  - persistent C API handle reuse
  - serialized compile(api) calls via internal mutex
  - reduced ad-hoc static state in transform entrypoint
- Updated build wiring:
  - `compiler/CMakeLists.txt` now includes `transform/CompileApiSession.cpp`
- Verified by full acceptance suite (including pure-api stress and perf guard).

## Stage 19 (done for backend compile(api) sessionization)
- Introduced backend compile session module:
  - `backend/welder/compiler/WelderCompileSession.h`
  - `backend/welder/compiler/WelderCompileSession.cpp`
- Refactored `WelderCompileCAPI.cpp` into a thin C API adapter that builds
  `CompileSessionRequest` and delegates execution to `CompileSession`.
- Session responsibilities now include:
  - welder-compiler stage invocation
  - pass pipeline execution (`workgroup`, `linalg-to-loops`, `nvvm`)
  - per-run MLIRContext/PassManager construction
  - centralized error propagation
- Updated backend build wiring:
  - `backend/welder/compiler/CMakeLists.txt` adds `WelderCompileSession.cpp`
    into `welder-compile-capi`.

## Stage 20 (done for `.inc` debt reduction in compiler driver)
- Migrated pass-trace/env helper from `.inc` to typed modules:
  - added `backend/welder/compiler/WelderCompilerPassTraceAndEnv.h`
  - added `backend/welder/compiler/WelderCompilerPassTraceAndEnv.cpp`
  - removed `backend/welder/compiler/WelderCompilerPassTraceAndEnv.inc`
- Updated `WelderCompiler.cpp` to use the new module while keeping existing
  global feature toggles and call sites behavior unchanged.
- Updated backend build wiring to compile the new module for both:
  - `welder-compiler`
  - `welder-compile-capi`

## Stage 21 (done for `.inc` debt reduction: postbufferize hook setup)
- Removed `backend/welder/compiler/WelderCompilerPostbufferizeHookSetup.inc`.
- Inlined its two tiny hook-construction helpers back into
  `WelderCompiler.cpp`:
  - `makeRowReductionFixupTogglesFromGlobals`
  - `makePostbufferizeFixupHooks`
- No behavior changes: hook bindings and toggle wiring remain identical.

## Stage 22 (done for `.inc` debt reduction: mapping/io helpers)
- Removed `backend/welder/compiler/WelderCompilerMappingAndIoHelpers.inc`.
- Inlined its helper set into `WelderCompiler.cpp`:
  - output writer (`writeModuleToFile`)
  - GPU block/thread/warp mapping builders
  - utility predicates for row-reduction/broadcast edge classification
- No behavior changes; only code-placement simplification to reduce include
  fragmentation in compiler driver flow.

## Stage 23 (done for `.inc` debt reduction: fusion anchors)
- Migrated fusion-anchor helper block out of include file:
  - added `backend/welder/compiler/WelderCompilerFusionAnchors.h`
  - added `backend/welder/compiler/WelderCompilerFusionAnchors.cpp`
  - removed `backend/welder/compiler/WelderCompilerFusionAnchors.inc`
- Updated `WelderCompiler.cpp` to consume typed symbols from
  `welder::compiler` namespace instead of include-injected static helpers.
- Updated backend build wiring so both targets compile the new module:
  - `welder-compiler`
  - `welder-compile-capi`
- No behavior change expected; this step only reduces `.inc` coupling and
  clarifies module ownership around anchor/tagging logic.

## Stage 24 (done for `.inc` debt reduction: row-reduction input promotion)
- Migrated row-reduction input promotion helper out of include file:
  - added `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.h`
  - added `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.cpp`
  - removed `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.inc`
- Updated `WelderCompilerMemoryReuseTransforms.inc` to stop include-injecting
  this helper and use typed declaration from compiler module boundary.
- Updated backend build wiring so both targets compile the new module:
  - `welder-compiler`
  - `welder-compile-capi`
- No behavior change expected; refactor only reduces `.inc` dependency fan-out.

## Stage 25 (done for `.inc` debt reduction: async copy rewrite)
- Migrated async copy rewrite helper out of include file:
  - added `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.h`
  - added `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.cpp`
  - removed `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.inc`
- Updated `WelderCompilerMemoryReuseTransforms.inc` to stop include-injecting
  async copy rewrite logic.
- Updated `WelderCompiler.cpp` to consume typed symbol from
  `welder::compiler` namespace.
- Updated backend build wiring so both targets compile the new module:
  - `welder-compiler`
  - `welder-compile-capi`
- No behavior change expected; this is module-boundary cleanup only.

## Stage 26 (done for `.inc` debt reduction: async pipeline helpers)
- Migrated async pipeline helper block out of include file:
  - added `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.h`
  - added `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.cpp`
  - removed `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.inc`
- Updated `WelderCompiler.cpp` to consume typed symbol from
  `welder::compiler` namespace (`padWorkgroupAllocs`).
- Updated backend build wiring so both targets compile the new module:
  - `welder-compiler`
  - `welder-compile-capi`

## Stage 27 (done for `.inc` debt reduction: generic transform and fusion)
- Migrated generic transform/fusion helper block out of include file:
  - added `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.h`
  - added `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.cpp`
  - removed `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.inc`
- Updated `WelderCompiler.cpp` to consume typed symbols from
  `welder::compiler` namespace:
  - `buildGenericTransformLibrary`
  - `KernelSpec` / `RowReductionFusionPair` / `ThreadFusionPair`
  - `normalizeThreadFusionPairs`
- Updated generic path callsite to pass explicit `skipMapForallToBlocks`
  toggle instead of reading include-injected global state.
- Updated backend build wiring so both targets compile the new module:
  - `welder-compiler`
  - `welder-compile-capi`

## Stage 28 (done for compiler-side `.inc` debt burn-down)
- Completed migration of compiler transform library blocks to typed modules:
  - added `backend/welder/compiler/WelderCompilerMatmulTransformLibrary.h/.cpp`
  - added `backend/welder/compiler/WelderCompilerGenericCutEdgesTransformLibrary.h/.cpp`
  - removed legacy include files:
    - `WelderCompilerMatmulTransformLibrary.inc`
    - `WelderCompilerGenericCutEdgesTransformLibrary.inc`
- Switched remaining compiler include snippets from `.inc` suffix to `.h` suffix
  to keep include boundaries explicit while preserving behavior parity:
  - `WelderCompilerDriverHelpers.h`
  - `WelderCompilerModeDispatchBody.h`
  - `WelderCompilerMemoryReuseTransforms.h`
  - `WelderCompilerMatmulSoftmaxCanonicalize.h`
  - `WelderCompilerPostprocessCleanup.h`
- Verified backend build targets pass:
  - `welder-compiler`
  - `welder-compile-capi`
- Verified full acceptance suite passes with perf guard:
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash bench/run_all_stages.sh`

## Stage 29 (done for backend include-suffix unification)
- Renamed remaining backend include fragments from `.inc` to `.h` across
  solver/profiler/compiler include blocks under `backend/welder/compiler`.
- Updated include directives in source files accordingly.
- Result: backend compiler directory no longer contains `*.inc` files.
- Rebuilt and validated core targets:
  - `welder-solver`
  - `welder-compiler`
  - `welder-compile-capi`
  - `welder-pipeline`
- Re-ran full acceptance suite with perf guard, all PASS.
- No behavior change expected; this step only reduces include-injection
  coupling in compiler driver flow.

## Stage 30 (done for compiler phase boundary cleanup)
- Introduced explicit mode-dispatch phase module:
  - `backend/welder/compiler/WelderCompilerModeDispatch.h`
  - `backend/welder/compiler/WelderCompilerModeDispatch.cpp`
- Moved mode-branch orchestration out of `WelderCompiler.cpp` into
  `buildTransformLibraryForMode(...)`, keeping three branches behavior-stable:
  - `codegen-from-kernel-attrs`
  - `generic problem`
  - `matmul path`
- Promoted driver helper block to typed module pair:
  - `backend/welder/compiler/WelderCompilerDriverHelpers.h`
  - `backend/welder/compiler/WelderCompilerDriverHelpers.cpp`
- Split large CLI option declarations out of main orchestration file:
  - `backend/welder/compiler/WelderCompilerCliOptions.h`
- `WelderCompiler.cpp` now focuses on phase orchestration:
  1) CLI parse + trace setup
  2) dialect registration + input parse
  3) solve option build + effective row-reduction knob derivation
  4) mode dispatch
  5) transform library load
  6) lowering pipeline execution
- Updated backend build wiring for both targets:
  - `welder-compiler`
  - `welder-compile-capi`
- Verified full acceptance with perf guard:
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash bench/run_all_stages.sh`
  - all stages PASS (`ab_speedup` observed around `1.1751x`).

## Stage 31 (done for mode-dispatch branch split + pipeline naming clarity)
- ModeDispatch 从单体 body 拆为“薄路由 + 三分支实现”：
  - `WelderCompilerModeDispatch.cpp` 仅保留路由
  - `WelderCompilerModeDispatchKernelAttrsBranch.cpp`
  - `WelderCompilerModeDispatchGenericBranch.cpp`
  - `WelderCompilerModeDispatchMatmulBranch.cpp`
  - `WelderCompilerModeDispatchBody.h` 已删除
- 新增 `CompilerRunConfig`（parse+validate 专用模块）：
  - `WelderCompilerRunConfig.h/.cpp`
  - 主流程改为配置对象校验 + `buildModeDispatchContext(...)`。
- 前端命名和阶段契约收敛：
  - backend mode 默认名升级为 `process_chain`
    （兼容旧输入 `inprocess`）
  - runtime 后处理函数命名改为
    `validatePostbufferizeArtifacts(...)`
  - `PipelineRunner` 引入 `StageArtifacts` 统一阶段 I/O 路径。
- 验证结果：
  - `cmake --build /home/zhangruiqi/welder_typical_chain/build -j4`
  - `cmake --build /home/zhangruiqi/welder_typical_chain/backend/welder/compiler/build -j4 --target welder-compiler welder-compile-capi`
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash /home/zhangruiqi/welder_typical_chain/bench/run_all_stages.sh` 全 PASS。

## Stage 32 (done for postprocess giant-header split hardening)
- Completed split of the three large postprocess blocks into public API headers
  + implementation cpp wrappers:
  - `WelderCompilerMemoryReuseTransforms.h/.cpp`
  - `WelderCompilerPostprocessCleanup.h/.cpp`
  - `WelderCompilerMatmulSoftmaxCanonicalize.h/.cpp`
- Kept heavy logic in internal impl headers for behavior parity:
  - `WelderCompilerMemoryReuseTransformsImpl.h`
  - `WelderCompilerPostprocessCleanupImpl.h`
  - `WelderCompilerMatmulSoftmaxCanonicalizeImpl.h`
- Further split internal implementation by sub-stage (readability-first):
  - `WelderCompilerMemoryReuseTransformsStage{1,2,3}.h`
  - `WelderCompilerPostprocessCleanupStage{1,2,3,4}.h`
  - `WelderCompilerMatmulSoftmaxCanonicalizeStage{1,2,3}.h`
- Removed cross-module implicit symbol coupling by localizing helper utilities
  in owning modules (e.g. `stripToBaseMemref`, `fixSubviewResultTypes...`,
  `isUsedInLaunch`) and making required dialect includes explicit.
- Updated backend build wiring so both targets compile these new modules:
  - `welder-compiler`
  - `welder-compile-capi`
- Verification:
  - `cmake --build /home/zhangruiqi/welder_typical_chain/backend/welder/compiler/build -j4 --target welder-compiler welder-compile-capi` PASS
  - `cmake --build /home/zhangruiqi/welder_typical_chain/build -j4` PASS
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash /home/zhangruiqi/welder_typical_chain/bench/run_all_stages.sh` PASS
  - latest measured `ab_speedup=1.1753x`

## Acceptance checklist
1. `wtc-compiler` can run search-only and compile-only on typical-chain MLIR.
2. `bench/run_search.sh` outputs `best.json` and `candidates.tsv`.
3. `bench/run_ab.sh` emits `ab_summary.tsv` and `speedup.tsv`.
4. Codepath remains single-path and interview-explainable.

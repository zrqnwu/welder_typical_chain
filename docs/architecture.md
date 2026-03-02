# Architecture (Typical Chain Only)

## Scope
Only support:
- `linalg.matmul`
- row-wise softmax decomposition (`max / exp / sum / div`)
- generic + cut-edges pipeline

## Pipeline
1. `canonicalize` (MLIR pass stage)
   - runs `mlir-opt --canonicalize`
   - if input has `linalg.softmax`, first run transform-interpreter decomposition
     (`transform.structured.decompose_interface`)
   - emits `ir/01.canonicalized.mlir`
2. `tagging` (typical-chain tagging + sidecar metadata)
   - emits `ir/02.tagged.mlir` and `ir/02.tags.json`
3. `search` (optional)
   - outputs `search/best.json`, `search/candidates.tsv`
4. `build transform + compile`
   - baseline or fused variant
5. `postbufferize` diagnostics
   - emits `postbufferize_report.txt`
6. runnable NVVM artifact (`05.out.nvvm.runnable.mlir`)

## Compiler modes
- `--search-only`: stop after search artifacts
- `--compile-only`: compile using `--best-json` or tile overrides
- default full mode: search + compile
- `--repeat <n>`:
  - all modes run as same-process loop (used for reentrancy stress)
- `--backend-mode process_chain|api|shell`:
  - `process_chain` (default, compatible with legacy alias `inprocess`):
    orchestrate solver/compile/lowering directly with backend binaries and
    `mlir-opt`
  - `api`:
    - search: call backend solver C API in-process
    - compile: call backend compile C API in-process with explicit tool paths
      (`welderCompilerMain` + in-CAPI pass/lowering for `workgroup`, `linalg-to-loops`, `nvvm`)
    - in full mode, default behavior is search fallback to `process_chain`;
      add `--pure-api-full` to force search(api)+compile(api) in one process
  - `shell`: compatibility path that invokes backend wrapper scripts

## Code ownership
- `compiler/ir/`: canonicalize checks + tagging sidecar generation
- `compiler/scheduler/`: search orchestration + best/candidate artifacts
  - `Search.cpp`: search dispatch + artifact normalize/write
  - `Search{BackendShell,BackendProcessChain,BackendApi}.cpp`:
    backend-mode specific search implementations
  - `SearchCommon.cpp`: shared solver flag assembly
- `compiler/transform/`: compile orchestration and variant flags
  - `BuildCutEdgesTransform.cpp`: entry dispatch + artifact checks
  - `BuildCutEdgesTransform{Shell,ProcessChain,Api}.cpp`:
    backend-mode specific compile implementations
  - `BuildCutEdgesTransformCommon.cpp`: shared compiler flag assembly
- `compiler/pipeline/`: end-to-end phase orchestration
- `bench/`: search / A-B profile / verify scripts
  - `run_all_stages.sh`: one-command acceptance runner (regression + api pair + pure-api stress + AB)

## Backend boundary
This repo now vendors backend sources at `backend/welder/`:
- solver binary target: `backend/welder/compiler/build/welder-solver`
- solver API library target: `backend/welder/compiler/build/libwelder-solver-capi.so`
- compiler binary target: `backend/welder/compiler/build/welder-compiler`
- lowering pass plugin: `backend/welder/mlir_pipeline/workgroup_alloc_to_launch_pass/build/WorkgroupAllocToLaunchPass.so`
- shell compatibility scripts:
  - `backend/welder/compiler/run_welder_solver.sh`
  - `backend/welder/compiler/run_welder_to_nvvm_isa.sh`
- profiler binary target: `backend/welder/compiler/build/welder-profiler`

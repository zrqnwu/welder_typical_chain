# welder_typical_chain

A focused project for one path only: `Matmul -> Softmax` typical chain.

## Goal
- Keep only the typical-chain pipeline.
- Keep code compact and interview-explainable.
- Make the project runnable without external `welder_try` path dependency.

## Current status
- `wtc-compiler` supports explicit modes:
  - `--search-only`
  - `--compile-only`
  - full pipeline (default)
- Backend execution mode:
  - `--backend-mode inprocess` (default): direct toolchain orchestration (`welder-solver`, `welder-compiler`, `mlir-opt`) without backend wrapper scripts
  - `--backend-mode api`: `dlopen` backend C API (`welder-solver-capi` + `welder-compile-capi`)
    - search(api): true in-process C API call
    - compile(api): true in-process C API compile (`welderCompilerMain`) + in-CAPI lowering
  - `--backend-mode shell`: compatibility mode via legacy backend scripts
  - stability default: in full mode with `--backend-mode api`, search falls back to `inprocess` mode before compile API
  - experimental pure API full mode: add `--pure-api-full`
- Search artifacts under `output_dir/search/`:
  - `best.json`
  - `best_summary.json`
  - `candidates.tsv`
  - `solver.log`
- IR stage artifacts under `output_dir/ir/`:
  - `01.canonicalized.mlir`
  - `01.canonicalize.log`
  - `02.tagged.mlir`
  - `02.tags.json`
- Compile diagnostics include `postbufferize_report.txt`.
- Backend source is vendored in this repo at:
  - `backend/welder/compiler`
  - `backend/welder/mlir_pipeline`

## Build
```bash
cd /home/zhangruiqi/welder_typical_chain
cmake -S . -B build
cmake --build build -j
```

## Search only
```bash
./build/compiler/wtc-compiler \
  --input mlir/matmul_softmax_chain_f16_native.mlir \
  --output-dir /tmp/wtc_out \
  --backend-root /home/zhangruiqi/welder_typical_chain/backend/welder \
  --backend-mode inprocess \
  --search-only \
  --max-connect-level 1 \
  --verbose
```

## Compile from searched best
```bash
./build/compiler/wtc-compiler \
  --input mlir/matmul_softmax_chain_f16_native.mlir \
  --output-dir /tmp/wtc_out \
  --backend-root /home/zhangruiqi/welder_typical_chain/backend/welder \
  --backend-mode inprocess \
  --compile-only \
  --best-json /tmp/wtc_out/search/best.json \
  --fused \
  --max-connect-level 1 \
  --verbose
```

## Bench
```bash
# Search artifacts
bash bench/run_search.sh

# Compile baseline+fused, profile, and output ab_summary.tsv + speedup.tsv
bash bench/run_ab.sh

# Regression checks: search/api, named-softmax decompose, full/api, pure-api-full repeat
bash bench/run_regression.sh

# Stress pure API full mode (in-process repeat loop)
REPEAT=100 bash bench/run_pure_api_stress.sh

# One-command stage acceptance (regression + api pair + pure-api stress + AB)
bash bench/run_all_stages.sh

# Pin baseline artifacts for fixed shape (03/04/04c/05 + ab summary)
bash bench/pin_baseline.sh

# Run acceptance with performance regression guard (<=3% by default)
CHECK_PERF_GUARD=1 bash bench/run_all_stages.sh
```

`run_ab.sh` defaults to `VERIFY=0`. Enable correctness check with:
```bash
VERIFY=1 bash bench/run_ab.sh
```

## Notes
- `--legacy-root` is still accepted as a compatibility alias of `--backend-root`.
- `BACKEND_MODE=shell` can be used in bench scripts for fallback debugging.
- `--repeat <n>` repeats pipeline in one process (used for API reentrancy stress).
- `--backend-mode api --pure-api-full --repeat>1` is supported as same-process loop (no forced per-iteration subprocess isolation).
- In full mode with `--backend-mode api`, default is still stability-first:
  search falls back to `inprocess` mode unless `--pure-api-full` is set.
- Canonicalize stage supports both input styles:
  - already decomposed softmax chain (`max/exp/sum/div`)
  - named softmax op (`linalg.softmax`) via transform-interpreter decomposition

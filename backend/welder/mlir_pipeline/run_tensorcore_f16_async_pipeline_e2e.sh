#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ART="mlir_pipeline/tensorcore_f16_async_pipeline_artifacts"
mkdir -p "$ART"

echo "[1/3] welder-compiler (tensorcore f16 + async copy + software pipelining) -> $ART/01.after_welder_compiler.mlir"
cmake --build compiler/build -j 8 >/dev/null

echo "[2/3] Lower to nvvm runnable -> $ART/05.out.nvvm.runnable.mlir"
OUT_DIR="$ART" \
WORKGROUP_PAD_LAST_DIM=8 \
WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY=1 \
WORKGROUP_MULTIBUFFER_DEPTH=3 \
ENABLE_SOFTWARE_PIPELINING=1 \
PIPELINE_DEPTH=3 \
PIPELINE_PEEL_EPILOGUE=1 \
./compiler/run_welder_to_nvvm_isa.sh \
  mlir_pipeline/matmul_host_shared_f16.mlir \
  --enable-tensorcore-f16 \
  --enable-async-copy \
  >/dev/null

echo "[3/3] Sanity checks: expect mma.sync, async-copy IR, and pipeline artifacts"

if ! grep -q "mma.sync" "$ART/05.out.nvvm.runnable.mlir"; then
  echo "FAIL: expected mma.sync in nvvm runnable MLIR"
  exit 1
fi

# Async-copy must materialize at IR level when enabled.
if ! grep -q "nvgpu\\.device_async_copy" "$ART/03.after_postbufferize.mlir" && \
   ! grep -q "nvgpu\\.device_async_copy" "$ART/04c.after_linalg_to_loops.mlir"; then
  echo "FAIL: expected nvgpu.device_async_copy in IR (async copy enabled)" >&2
  exit 1
fi

# Software pipelining should introduce a loop-carried iter_args structure.
if ! grep -q "iter_args" "$ART/04b.after_software_pipelining.mlir"; then
  echo "FAIL: expected iter_args in pipelined payload IR" >&2
  exit 1
fi

echo "PASS: tensorcore f16 async+pipelining e2e verified."

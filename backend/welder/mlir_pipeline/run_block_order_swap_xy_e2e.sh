#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BIN="${ROOT_DIR}/compiler/build/welder-compiler"

INPUT="${ROOT_DIR}/mlir_pipeline/transpose_outputmap_swap.mlir"
OUT_DIR="${ROOT_DIR}/mlir_pipeline/block_order_swap_xy_artifacts"
OUT_PRE="${OUT_DIR}/01.after_prebufferize.mlir"

mkdir -p "${OUT_DIR}"

cmake --build "${ROOT_DIR}/compiler/build" -j

"${COMPILER_BIN}" "${INPUT}" \
  --output "${OUT_DIR}/99.final.mlir" \
  --emit-after-prebufferize "${OUT_PRE}" \
  --enable-generic-problem \
  --enable-cut-edges \
  --enable-tile-propagation \
  --enable-footprint-inference \
  --enable-global-traffic \
  --require-perfect-tiling=false \
  --candidates-mn 64,32 \
  --candidates-k 16 \
  >/dev/null

echo "Checking scf.forall block mapping order (expect swapped: x then y)..."

# Prefer explicit LinearDim markers when available.
if rg -n "LinearDim0|linear_dim_0" "${OUT_PRE}" >/dev/null 2>&1; then
  if rg -n "mapping\\s*=\\s*\\[[^\\]]*(LinearDim0|linear_dim_0)[^\\]]*(LinearDim1|linear_dim_1)" "${OUT_PRE}" >/dev/null 2>&1; then
    echo "PASS: Found mapping with LinearDim0 before LinearDim1 (swapXY applied)."
    exit 0
  fi
fi

# 回退 for older printers using x/y names.
if rg -n "mapping\\s*=\\s*\\[[^\\]]*block<[^\\]]*x[^\\]]*>,\\s*#gpu\\.block<[^\\]]*y[^\\]]*>\\]" "${OUT_PRE}" >/dev/null 2>&1; then
  echo "PASS: Found mapping with block<x> before block<y> (swapXY applied)."
  exit 0
fi

echo "FAIL: Could not detect swapped block mapping in ${OUT_PRE}"
echo "Hint: inspect mapping attrs in ${OUT_PRE}"
exit 1


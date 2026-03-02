#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/mlir_pipeline/rasterization_2d_artifacts"
mkdir -p "${OUT_DIR}"

echo "[0/3] build pass plugin"
cmake --build "${ROOT_DIR}/mlir_pipeline/workgroup_alloc_to_launch_pass/build" -j

echo "[1/3] compile to nvvm with 2D rasterization enabled"
OUT_DIR="${OUT_DIR}" \
  BLOCK_RASTERIZE_MODE=1 \
  BLOCK_RASTERIZE_PANEL_WIDTH=4 \
  bash "${ROOT_DIR}/compiler/run_welder_to_nvvm_isa.sh" \
    "${ROOT_DIR}/mlir_pipeline/matmul_relu_host_shared.mlir" \
    --enable-generic-problem \
    --candidates-mn 64 \
    --candidates-k 16 \
    >/dev/null

AFTER_WG="${OUT_DIR}/04.after_workgroup_launch.mlir"
if [[ ! -f "${AFTER_WG}" ]]; then
  echo "FAIL: missing output: ${AFTER_WG}" >&2
  exit 1
fi

echo "[2/3] check rasterization marker attribute"
if ! rg -n "welder\\.block_rasterized_2d" "${AFTER_WG}" >/dev/null; then
  echo "FAIL: did not find welder.block_rasterized_2d in ${AFTER_WG}" >&2
  exit 1
fi

echo "[3/3] sanity: ensure we still have a gpu.launch"
if ! rg -n "gpu\\.launch" "${AFTER_WG}" >/dev/null; then
  echo "FAIL: did not find gpu.launch in ${AFTER_WG}" >&2
  exit 1
fi

echo "PASS: 2D rasterization e2e verified."


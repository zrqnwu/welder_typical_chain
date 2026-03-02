#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/mlir_pipeline/block_rasterize_artifacts"
mkdir -p "${ART_DIR}"

echo "[1/2] compile with block rasterize XOR -> ${ART_DIR}"
OUT_DIR="${ART_DIR}" \
BLOCK_RASTERIZE_XOR=2 \
  bash "${ROOT_DIR}/compiler/run_welder_to_nvvm_isa.sh" \
  "${ROOT_DIR}/mlir_pipeline/matmul_relu_host_shared.mlir" \
  --candidates-mn 64 --candidates-k 16 \
  --require-perfect-tiling=true \
  --thread-tile-m 4 --thread-tile-n 4

WG="${ART_DIR}/04.after_workgroup_launch.mlir"
if [[ ! -f "${WG}" ]]; then
  echo "FAIL: missing output: ${WG}" >&2
  exit 1
fi

echo "[2/2] IR sanity check: expect arith.xori introduced by block rasterization"
if grep -q "arith\\.xori" "${WG}"; then
  echo "PASS: block rasterize e2e verified."
else
  echo "FAIL: did not find arith.xori in ${WG}" >&2
  exit 1
fi

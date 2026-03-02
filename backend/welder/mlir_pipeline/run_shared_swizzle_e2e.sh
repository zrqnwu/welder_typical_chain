#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/mlir_pipeline/shared_swizzle_artifacts"
mkdir -p "${ART_DIR}"

echo "[1/2] compile with workgroup XOR swizzle -> ${ART_DIR}"
# Note: after promotion we often see shared-memory subviews with last-dim=4,
# so use 重排=4 to ensure the rewrite triggers.
OUT_DIR="${ART_DIR}" \
WORKGROUP_SWIZZLE_XOR=4 \
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

echo "[2/2] IR sanity check: expect arith.xori introduced by swizzle"
if grep -q "arith\\.xori" "${WG}"; then
  echo "PASS: shared swizzle e2e verified."
else
  echo "FAIL: did not find arith.xori in ${WG}" >&2
  exit 1
fi

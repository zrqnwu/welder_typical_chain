#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/mlir_pipeline/register_level_matmul_artifacts}"
mkdir -p "${OUT_DIR}"

echo "[1/2] compile matmul with solver-chosen thread tile"
OUT_DIR="${OUT_DIR}" \
  bash "${ROOT_DIR}/compiler/run_welder_to_nvvm_isa.sh" \
    "${ROOT_DIR}/mlir_pipeline/matmul_relu_host_shared.mlir" \
    --force-tile-m 64 \
    --force-tile-n 64 \
    --force-tile-k 16 \
    --enable-register-level-schedule \
    --candidates-thread-mn 8 \
    --candidates-mn 64 \
    --candidates-k 16 \
    >/dev/null

NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"
if [[ ! -f "${NVVM}" ]]; then
  echo "FAIL: missing output: ${NVVM}" >&2
  exit 1
fi

echo "[2/2] PTX sanity check: expect .maxntid 8, 8, 1 (tile 64/64 with thread tile 8)"
if grep -q "\\.maxntid 8, 8, 1" "${NVVM}"; then
  echo "PASS: register-level matmul e2e verified."
else
  echo "FAIL: expected .maxntid 8, 8, 1 in ${NVVM}" >&2
  exit 1
fi

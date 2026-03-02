#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/mlir_pipeline/tensorcore_f16_artifacts"
mkdir -p "${ART_DIR}"

echo "[1/2] compile to nvvm (tensorcore f16) -> ${ART_DIR}"
OUT_DIR="${ART_DIR}" \
  bash "${ROOT_DIR}/compiler/run_welder_to_nvvm_isa.sh" \
  "${ROOT_DIR}/mlir_pipeline/matmul_host_shared_f16.mlir" \
  --enable-tensorcore-f16 \
  --force-tile-m 16 --force-tile-n 8 --force-tile-k 16

NVVM="${ART_DIR}/05.out.nvvm.runnable.mlir"
if [[ ! -f "${NVVM}" ]]; then
  echo "FAIL: missing output: ${NVVM}" >&2
  exit 1
fi

echo "[2/2] PTX sanity check: expect mma.sync m16n8k16"
if grep -q "mma\\.sync.*m16n8k16" "${NVVM}"; then
  echo "PASS: tensorcore f16 e2e verified."
else
  echo "FAIL: did not find mma.sync m16n8k16 in ${NVVM}" >&2
  exit 1
fi


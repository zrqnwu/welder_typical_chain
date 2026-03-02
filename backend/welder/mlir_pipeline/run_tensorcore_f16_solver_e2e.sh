#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/mlir_pipeline/tensorcore_f16_solver_artifacts"
mkdir -p "${ART_DIR}"

echo "[1/2] compile to nvvm via solver (tensorcore f16) -> ${ART_DIR}"
OUT_DIR="${ART_DIR}" \
  bash "${ROOT_DIR}/compiler/run_welder_to_nvvm_isa.sh" \
  "${ROOT_DIR}/mlir_pipeline/matmul_host_shared_f16.mlir" \
  --enable-tensorcore-f16 \
  --candidates-mn 32 \
  --candidates-k 16

NVVM="${ART_DIR}/05.out.nvvm.runnable.mlir"
if [[ ! -f "${NVVM}" ]]; then
  echo "FAIL: missing output: ${NVVM}" >&2
  exit 1
fi

echo "[2/2] PTX sanity checks: expect mma.sync and a valid warp-multiple blockDim"
if ! grep -q "mma\\.sync.*m16n8k16" "${NVVM}"; then
  echo "FAIL: did not find mma.sync m16n8k16 in ${NVVM}" >&2
  exit 1
fi
maxntid_line="$(grep -oE "\\.maxntid [0-9]+, *1, *1" "${NVVM}" | head -n 1 || true)"
threads="$(echo "${maxntid_line}" | sed -n 's/\.maxntid \([0-9][0-9]*\), *1, *1/\1/p')"
if [[ -z "${threads}" ]]; then
  echo "FAIL: could not parse .maxntid x dimension from ${NVVM}" >&2
  exit 1
fi
if (( threads <= 0 || threads > 1024 || threads % 32 != 0 )); then
  echo "FAIL: invalid .maxntid x=${threads} (expect multiple of 32 and <=1024)" >&2
  exit 1
fi
echo "PASS: tensorcore f16 solver e2e verified (.maxntid x=${threads})."

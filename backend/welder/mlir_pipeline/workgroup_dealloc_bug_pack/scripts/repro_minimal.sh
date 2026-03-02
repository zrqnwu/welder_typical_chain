#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd -- "${PACK_DIR}/../.." && pwd)"

BIN="${BIN:-${ROOT_DIR}/llvm-project/build/bin}"
CHIP="${CHIP:-sm_86}"

MLIR_OPT="${BIN}/mlir-opt"
if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  echo "hint: set BIN=.../llvm-project/build/bin" >&2
  exit 2
fi

IN_MLIR="${PACK_DIR}/minimal/repro.mlir"
OUT_DIR="${PACK_DIR}/artifacts"
mkdir -p "${OUT_DIR}"

set +e
"${MLIR_OPT}" "${IN_MLIR}" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CHIP} cubin-format=isa" \
  -o "${OUT_DIR}/out.nvvm.minimal.mlir" \
  2> "${OUT_DIR}/nvvm_lower_err.minimal.log"
STATUS=$?
set -e

echo "exit=${STATUS}"
echo "stderr: ${OUT_DIR}/nvvm_lower_err.minimal.log"
echo
tail -n 50 "${OUT_DIR}/nvvm_lower_err.minimal.log"

if [[ ${STATUS} -eq 0 ]]; then
  echo "unexpected: nvvm pipeline succeeded" >&2
  exit 1
fi


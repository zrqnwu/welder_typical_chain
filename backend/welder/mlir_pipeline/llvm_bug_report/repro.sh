#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BIN="${BIN:-${ROOT_DIR}/llvm-project/build/bin}"
CHIP="${CHIP:-sm_86}"

MLIR_OPT="${BIN}/mlir-opt"

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  echo "hint: set BIN=.../llvm-project/build/bin" >&2
  exit 2
fi

echo "[mlir-opt]"
"${MLIR_OPT}" --version | head -n 20
echo

echo "[repro]"
echo "  file: ${SCRIPT_DIR}/repro.mlir"
echo "  cmd : ${MLIR_OPT} ${SCRIPT_DIR}/repro.mlir --gpu-lower-to-nvvm-pipeline=\"cubin-chip=${CHIP} cubin-format=isa\" -o /tmp/out.nvvm.mlir"
echo

set +e
"${MLIR_OPT}" "${SCRIPT_DIR}/repro.mlir" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CHIP} cubin-format=isa" \
  -o /tmp/out.nvvm.mlir \
  2> /tmp/repro_workgroup_dealloc.stderr
STATUS=$?
set -e

echo "[result]"
echo "  exit: ${STATUS}"
if [[ ${STATUS} -eq 0 ]]; then
  echo "  unexpected: succeeded (see /tmp/out.nvvm.mlir)" >&2
  exit 1
fi
echo "  stderr: /tmp/repro_workgroup_dealloc.stderr"
echo

echo "[stderr (tail)]"
tail -n 50 /tmp/repro_workgroup_dealloc.stderr


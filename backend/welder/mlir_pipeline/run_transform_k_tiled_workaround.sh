#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BIN="${BIN:-${ROOT_DIR}/llvm-project/build/bin}"
CHIP="${CHIP:-sm_86}"

MLIR_OPT="${BIN}/mlir-opt"
if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  echo "hint: set BIN=.../llvm-project/build/bin" >&2
  exit 2
fi

IN_MLIR="${SCRIPT_DIR}/matmul.mlir"
IN_TRANSFORM="${SCRIPT_DIR}/transform_k_tiled.mlir"
OUT_DIR="${SCRIPT_DIR}/transform_k_tiled_artifacts"

mkdir -p "${OUT_DIR}"

echo "[1/4] bufferize -> ${OUT_DIR}/matmul.bufferized.mlir"
"${MLIR_OPT}" "${IN_MLIR}" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o "${OUT_DIR}/matmul.bufferized.mlir"

echo "[2/4] transform + linalg-to-loops -> ${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.mlir"
"${MLIR_OPT}" "${OUT_DIR}/matmul.bufferized.mlir" \
  --transform-preload-library="transform-library-paths=${IN_TRANSFORM}" \
  --transform-interpreter \
  --convert-linalg-to-loops \
  -o "${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.mlir"

echo "[3/4] drop workgroup memref.dealloc (workaround) -> ${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.nodealloc.mlir"
sed '/memref\.dealloc .*#gpu\.address_space<workgroup>/d' \
  "${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.mlir" \
  > "${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.nodealloc.mlir"

echo "[4/4] nvvm pipeline (expected to succeed with workaround) -> ${OUT_DIR}/out.nvvm.workaround.mlir"
"${MLIR_OPT}" "${OUT_DIR}/matmul.bufferized.tiled.promoted.gpu.loops.nodealloc.mlir" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CHIP} cubin-format=isa" \
  -o "${OUT_DIR}/out.nvvm.workaround.mlir"

echo "ok: ${OUT_DIR}/out.nvvm.workaround.mlir"


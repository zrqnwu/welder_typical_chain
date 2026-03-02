#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
OUT_BASE="${OUT_BASE:-/tmp/wtc_pair}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
BACKEND_MODE="${BACKEND_MODE:-process_chain}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input not found: ${INPUT_MLIR}" >&2
  exit 2
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" >/dev/null
cmake --build "${ROOT_DIR}/build" -j >/dev/null

WTC="${ROOT_DIR}/build/compiler/wtc-compiler"
if [[ ! -x "${WTC}" ]]; then
  echo "error: wtc-compiler not found: ${WTC}" >&2
  exit 2
fi

mkdir -p "${OUT_BASE}"

BEST_JSON="${OUT_BASE}/search/best.json"

echo "[wtc] search"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${OUT_BASE}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode "${BACKEND_MODE}" \
  --search-only \
  --max-connect-level "${MAX_CONNECT_LEVEL}" \
  --verbose

echo "[wtc] compile baseline"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${OUT_BASE}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode "${BACKEND_MODE}" \
  --compile-only \
  --best-json "${BEST_JSON}" \
  --baseline \
  --max-connect-level 0 \
  --verbose

echo "[wtc] compile fused"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${OUT_BASE}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode "${BACKEND_MODE}" \
  --compile-only \
  --best-json "${BEST_JSON}" \
  --fused \
  --max-connect-level "${MAX_CONNECT_LEVEL}" \
  --verbose

echo "[wtc] done"
echo "  best:     ${BEST_JSON}"
echo "  baseline: ${OUT_BASE}/baseline/05.out.nvvm.runnable.mlir"
echo "  fused:    ${OUT_BASE}/fused/05.out.nvvm.runnable.mlir"

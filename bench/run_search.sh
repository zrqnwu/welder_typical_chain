#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
OUT_DIR="${OUT_DIR:-/tmp/wtc_search}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
BACKEND_MODE="${BACKEND_MODE:-process_chain}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input mlir not found: ${INPUT_MLIR}" >&2
  exit 2
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" >/dev/null
cmake --build "${ROOT_DIR}/build" -j >/dev/null

WTC="${ROOT_DIR}/build/compiler/wtc-compiler"
if [[ ! -x "${WTC}" ]]; then
  echo "error: wtc-compiler not found: ${WTC}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${OUT_DIR}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode "${BACKEND_MODE}" \
  --search-only \
  --max-connect-level "${MAX_CONNECT_LEVEL}" \
  --verbose

echo "[wtc] search artifacts"
echo "  best:        ${OUT_DIR}/search/best.json"
echo "  best_backend:${OUT_DIR}/search/best_summary.json"
echo "  candidates:  ${OUT_DIR}/search/candidates.tsv"
echo "  log:         ${OUT_DIR}/search/solver.log"

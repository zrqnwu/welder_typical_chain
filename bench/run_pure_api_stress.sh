#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
OUT_BASE="${OUT_BASE:-/tmp/wtc_pure_api_stress}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi

REPEAT="${REPEAT:-100}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"
VERBOSE="${VERBOSE:-0}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input mlir not found: ${INPUT_MLIR}" >&2
  exit 2
fi

if [[ "${REPEAT}" -lt 1 ]]; then
  echo "error: REPEAT must be >= 1" >&2
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
LOG_FILE="${OUT_BASE}/stress.log"

echo "[wtc] pure-api-full stress"
echo "  repeat: ${REPEAT}"
echo "  out:    ${OUT_BASE}"
echo "  log:    ${LOG_FILE}"

: >"${LOG_FILE}"
cmd=(
  "${WTC}"
  --input "${INPUT_MLIR}"
  --output-dir "${OUT_BASE}"
  --backend-root "${BACKEND_ROOT}"
  --backend-mode api
  --pure-api-full
  --max-connect-level "${MAX_CONNECT_LEVEL}"
  --fused
  --repeat "${REPEAT}"
)
if [[ "${VERBOSE}" == "1" ]]; then
  cmd+=(--verbose)
fi

"${cmd[@]}" >"${LOG_FILE}" 2>&1

last_iter=$((REPEAT - 1))
last_artifact="${OUT_BASE}/iter_${last_iter}/fused/05.out.nvvm.runnable.mlir"
if [[ ! -f "${last_artifact}" ]]; then
  echo "error: missing artifact from last iteration: ${last_artifact}" >&2
  exit 2
fi

echo "[wtc] pure-api-full stress passed"
echo "  last artifact: ${last_artifact}"
echo "  full log:      ${LOG_FILE}"

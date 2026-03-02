#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
NAMED_SOFTMAX_INPUT_MLIR="${NAMED_SOFTMAX_INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_named_softmax.mlir}"
OUT_BASE="${OUT_BASE:-/tmp/wtc_regression}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"
PURE_API_REPEAT="${PURE_API_REPEAT:-10}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input mlir not found: ${INPUT_MLIR}" >&2
  exit 2
fi
if [[ ! -f "${NAMED_SOFTMAX_INPUT_MLIR}" ]]; then
  echo "error: named-softmax input mlir not found: ${NAMED_SOFTMAX_INPUT_MLIR}" >&2
  exit 2
fi

mkdir -p "${OUT_BASE}"
SUMMARY="${OUT_BASE}/regression_summary.tsv"
echo -e "check\tstatus\tdetail" >"${SUMMARY}"

pass() {
  echo -e "$1\tPASS\t$2" | tee -a "${SUMMARY}"
}

fail() {
  echo -e "$1\tFAIL\t$2" | tee -a "${SUMMARY}" >&2
  exit 1
}

require_file() {
  local check="$1"
  local path="$2"
  [[ -f "${path}" ]] || fail "${check}" "missing file: ${path}"
  pass "${check}" "exists: ${path}"
}

require_contains() {
  local check="$1"
  local path="$2"
  local pattern="$3"
  rg -q "${pattern}" "${path}" || fail "${check}" "pattern '${pattern}' not found in ${path}"
  pass "${check}" "found '${pattern}' in ${path}"
}

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" >/dev/null
cmake --build "${ROOT_DIR}/build" -j >/dev/null
WTC="${ROOT_DIR}/build/compiler/wtc-compiler"
[[ -x "${WTC}" ]] || fail "build" "wtc-compiler not found: ${WTC}"
pass "build" "${WTC}"

SEARCH_API_OUT="${OUT_BASE}/search_api"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${SEARCH_API_OUT}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode api \
  --search-only \
  --max-connect-level "${MAX_CONNECT_LEVEL}" >/dev/null
require_file "search_api.best" "${SEARCH_API_OUT}/search/best.json"
require_file "search_api.candidates" "${SEARCH_API_OUT}/search/candidates.tsv"

NAMED_SOFTMAX_SEARCH_OUT="${OUT_BASE}/search_named_softmax"
"${WTC}" \
  --input "${NAMED_SOFTMAX_INPUT_MLIR}" \
  --output-dir "${NAMED_SOFTMAX_SEARCH_OUT}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode process_chain \
  --search-only \
  --max-connect-level "${MAX_CONNECT_LEVEL}" >/dev/null
require_file "named_softmax.canonicalized" "${NAMED_SOFTMAX_SEARCH_OUT}/ir/01.canonicalized.mlir"
require_contains "named_softmax.exp" "${NAMED_SOFTMAX_SEARCH_OUT}/ir/01.canonicalized.mlir" "math\\.exp"
require_contains "named_softmax.div" "${NAMED_SOFTMAX_SEARCH_OUT}/ir/01.canonicalized.mlir" "arith\\.divf"
if rg -q "linalg\\.softmax" "${NAMED_SOFTMAX_SEARCH_OUT}/ir/01.canonicalized.mlir"; then
  fail "named_softmax.decompose" "linalg.softmax remains after canonicalize"
else
  pass "named_softmax.decompose" "linalg.softmax removed after canonicalize"
fi

FULL_API_OUT="${OUT_BASE}/full_api_default"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${FULL_API_OUT}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode api \
  --max-connect-level "${MAX_CONNECT_LEVEL}" \
  --fused >/dev/null
require_file "full_api_default.runnable" "${FULL_API_OUT}/fused/05.out.nvvm.runnable.mlir"
require_contains "full_api_default.workgroup" "${FULL_API_OUT}/fused/04.after_workgroup_launch.mlir" "workgroup\\("
require_contains "full_api_default.launchfunc" "${FULL_API_OUT}/fused/05.out.nvvm.runnable.mlir" "gpu\\.launch_func"

PURE_API_OUT="${OUT_BASE}/pure_api_repeat"
"${WTC}" \
  --input "${INPUT_MLIR}" \
  --output-dir "${PURE_API_OUT}" \
  --backend-root "${BACKEND_ROOT}" \
  --backend-mode api \
  --pure-api-full \
  --max-connect-level "${MAX_CONNECT_LEVEL}" \
  --fused \
  --repeat "${PURE_API_REPEAT}" >/dev/null
last_iter=$((PURE_API_REPEAT - 1))
require_file "pure_api_repeat.runnable" "${PURE_API_OUT}/iter_${last_iter}/fused/05.out.nvvm.runnable.mlir"
require_contains "pure_api_repeat.launchfunc" "${PURE_API_OUT}/iter_${last_iter}/fused/05.out.nvvm.runnable.mlir" "gpu\\.launch_func"

echo "[wtc] regression passed"
echo "  summary: ${SUMMARY}"

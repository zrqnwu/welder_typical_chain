#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OUT_BASE="${OUT_BASE:-/tmp/wtc_all_stages}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
PURE_API_REPEAT="${PURE_API_REPEAT:-100}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"
CHECK_PERF_GUARD="${CHECK_PERF_GUARD:-0}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/bench/baselines/default}"
MAX_REGRESSION_PCT="${MAX_REGRESSION_PCT:-3}"

mkdir -p "${OUT_BASE}"
SUMMARY="${OUT_BASE}/all_stages_summary.tsv"
echo -e "stage\tstatus\tdetail" >"${SUMMARY}"

run_stage() {
  local name="$1"
  shift
  echo "[stage] ${name}"
  if "$@"; then
    echo -e "${name}\tPASS\tok" | tee -a "${SUMMARY}"
  else
    local rc=$?
    echo -e "${name}\tFAIL\trc=${rc}" | tee -a "${SUMMARY}" >&2
    exit "${rc}"
  fi
}

run_stage "regression" env \
  OUT_BASE="${OUT_BASE}/regression" \
  BACKEND_ROOT="${BACKEND_ROOT}" \
  MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  PURE_API_REPEAT=10 \
  bash "${SCRIPT_DIR}/run_regression.sh"

run_stage "compile_pair_api" env \
  OUT_BASE="${OUT_BASE}/compile_pair_api" \
  BACKEND_ROOT="${BACKEND_ROOT}" \
  BACKEND_MODE=api \
  MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  bash "${SCRIPT_DIR}/run_compile_pair.sh"

run_stage "pure_api_stress" env \
  OUT_BASE="${OUT_BASE}/pure_api_stress" \
  BACKEND_ROOT="${BACKEND_ROOT}" \
  REPEAT="${PURE_API_REPEAT}" \
  MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  bash "${SCRIPT_DIR}/run_pure_api_stress.sh"

run_stage "ab_process_chain" env \
  OUT_BASE="${OUT_BASE}/ab" \
  BACKEND_ROOT="${BACKEND_ROOT}" \
  BACKEND_MODE=process_chain \
  MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  VERIFY=0 \
  bash "${SCRIPT_DIR}/run_ab.sh"

if [[ "${CHECK_PERF_GUARD}" == "1" ]]; then
  run_stage "perf_guard" env \
    BASELINE_DIR="${BASELINE_DIR}" \
    CURRENT_OUT_BASE="${OUT_BASE}/ab" \
    MAX_REGRESSION_PCT="${MAX_REGRESSION_PCT}" \
    bash "${SCRIPT_DIR}/check_perf_guard.sh"
fi

if [[ -f "${OUT_BASE}/ab/speedup.tsv" ]]; then
  speedup="$(awk 'NR==1{print $2}' "${OUT_BASE}/ab/speedup.tsv" 2>/dev/null || true)"
  if [[ -n "${speedup}" ]]; then
    echo -e "ab_speedup\tPASS\t${speedup}x" | tee -a "${SUMMARY}"
  fi
fi

echo "[wtc] all stages passed"
echo "  out:     ${OUT_BASE}"
echo "  summary: ${SUMMARY}"

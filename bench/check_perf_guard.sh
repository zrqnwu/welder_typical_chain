#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/bench/baselines/default}"
CURRENT_OUT_BASE="${CURRENT_OUT_BASE:-/tmp/wtc_ab}"
MAX_REGRESSION_PCT="${MAX_REGRESSION_PCT:-3}"
# speedup 检查模式：
# - auto(默认): 当当前 baseline 比基线快很多时（超过阈值），跳过 speedup 下限检查，
#               避免“baseline 变快导致 speedup 变小”的误报。
# - strict: 始终执行 speedup 下限检查。
# - off: 不检查 speedup，仅检查 baseline/fused 绝对时延。
SPEEDUP_CHECK_MODE="${SPEEDUP_CHECK_MODE:-auto}"

baseline_ab="${BASELINE_DIR}/ab_summary.tsv"
baseline_speedup="${BASELINE_DIR}/speedup.tsv"
current_ab="${CURRENT_OUT_BASE}/ab_summary.tsv"
current_speedup="${CURRENT_OUT_BASE}/speedup.tsv"

for p in "${baseline_ab}" "${baseline_speedup}" "${current_ab}" "${current_speedup}"; do
  if [[ ! -f "${p}" ]]; then
    echo "error: missing required file: ${p}" >&2
    exit 2
  fi
done

get_avg_ms() {
  local file="$1"
  local case_name="$2"
  awk -F'\t' -v k="${case_name}" '$1==k{print $2}' "${file}"
}

get_speedup() {
  awk 'NR==1{print $2}' "$1"
}

base_baseline_ms="$(get_avg_ms "${baseline_ab}" baseline)"
base_fused_ms="$(get_avg_ms "${baseline_ab}" fused)"
curr_baseline_ms="$(get_avg_ms "${current_ab}" baseline)"
curr_fused_ms="$(get_avg_ms "${current_ab}" fused)"
base_speedup="$(get_speedup "${baseline_speedup}")"
curr_speedup="$(get_speedup "${current_speedup}")"

for v in "${base_baseline_ms}" "${base_fused_ms}" "${curr_baseline_ms}" "${curr_fused_ms}" "${base_speedup}" "${curr_speedup}"; do
  if [[ -z "${v}" ]]; then
    echo "error: failed to parse performance metrics" >&2
    exit 2
  fi
done

check_not_worse_than_pct() {
  local metric_name="$1"
  local baseline_val="$2"
  local current_val="$3"
  local max_reg_pct="$4"

  local ok
  ok="$(awk -v b="${baseline_val}" -v c="${current_val}" -v p="${max_reg_pct}" 'BEGIN{limit=b*(1.0+p/100.0); if(c<=limit) print "1"; else print "0"}')"
  if [[ "${ok}" != "1" ]]; then
    local limit
    limit="$(awk -v b="${baseline_val}" -v p="${max_reg_pct}" 'BEGIN{printf "%.9g", b*(1.0+p/100.0)}')"
    echo "FAIL ${metric_name}: baseline=${baseline_val}, current=${current_val}, limit=${limit} (${max_reg_pct}% regression threshold)" >&2
    return 1
  fi
  echo "PASS ${metric_name}: baseline=${baseline_val}, current=${current_val}, threshold=${max_reg_pct}%"
}

check_not_less_than_pct() {
  local metric_name="$1"
  local baseline_val="$2"
  local current_val="$3"
  local max_drop_pct="$4"

  local ok
  ok="$(awk -v b="${baseline_val}" -v c="${current_val}" -v p="${max_drop_pct}" 'BEGIN{limit=b*(1.0-p/100.0); if(c>=limit) print "1"; else print "0"}')"
  if [[ "${ok}" != "1" ]]; then
    local limit
    limit="$(awk -v b="${baseline_val}" -v p="${max_drop_pct}" 'BEGIN{printf "%.9g", b*(1.0-p/100.0)}')"
    echo "FAIL ${metric_name}: baseline=${baseline_val}, current=${current_val}, min_allowed=${limit} (${max_drop_pct}% drop threshold)" >&2
    return 1
  fi
  echo "PASS ${metric_name}: baseline=${baseline_val}, current=${current_val}, drop_threshold=${max_drop_pct}%"
}

is_less_than_pct() {
  local lhs="$1"
  local rhs="$2"
  local pct="$3"
  awk -v l="${lhs}" -v r="${rhs}" -v p="${pct}" \
      'BEGIN{limit=r*(1.0-p/100.0); if(l<limit) print "1"; else print "0"}'
}

check_not_worse_than_pct "baseline.avg_ms" "${base_baseline_ms}" "${curr_baseline_ms}" "${MAX_REGRESSION_PCT}"
check_not_worse_than_pct "fused.avg_ms" "${base_fused_ms}" "${curr_fused_ms}" "${MAX_REGRESSION_PCT}"

case "${SPEEDUP_CHECK_MODE}" in
  off)
    echo "SKIP speedup: SPEEDUP_CHECK_MODE=off"
    ;;
  strict)
    check_not_less_than_pct "speedup" "${base_speedup}" "${curr_speedup}" "${MAX_REGRESSION_PCT}"
    ;;
  auto)
    baseline_got_faster="$(is_less_than_pct "${curr_baseline_ms}" "${base_baseline_ms}" "${MAX_REGRESSION_PCT}")"
    if [[ "${baseline_got_faster}" == "1" ]]; then
      echo "SKIP speedup: baseline improved beyond ${MAX_REGRESSION_PCT}% (baseline=${base_baseline_ms}, current=${curr_baseline_ms})"
    else
      check_not_less_than_pct "speedup" "${base_speedup}" "${curr_speedup}" "${MAX_REGRESSION_PCT}"
    fi
    ;;
  *)
    echo "error: invalid SPEEDUP_CHECK_MODE=${SPEEDUP_CHECK_MODE} (expected auto|strict|off)" >&2
    exit 2
    ;;
esac

echo "[wtc] perf guard passed"
echo "  baseline_dir: ${BASELINE_DIR}"
echo "  current_out:  ${CURRENT_OUT_BASE}"

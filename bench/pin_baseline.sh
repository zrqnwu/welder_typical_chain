#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
BACKEND_MODE="${BACKEND_MODE:-process_chain}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"
RUN_OUT_BASE="${RUN_OUT_BASE:-/tmp/wtc_pin_baseline_run}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/bench/baselines/default}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input mlir not found: ${INPUT_MLIR}" >&2
  exit 2
fi

mkdir -p "${RUN_OUT_BASE}"
mkdir -p "${BASELINE_DIR}"

echo "[wtc] run AB to pin baseline"
env \
  INPUT_MLIR="${INPUT_MLIR}" \
  OUT_BASE="${RUN_OUT_BASE}" \
  BACKEND_ROOT="${BACKEND_ROOT}" \
  BACKEND_MODE="${BACKEND_MODE}" \
  MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  VERIFY=0 \
  bash "${SCRIPT_DIR}/run_ab.sh"

echo "[wtc] pin artifacts to ${BASELINE_DIR}"
mkdir -p "${BASELINE_DIR}/baseline" "${BASELINE_DIR}/fused" "${BASELINE_DIR}/search"

cp -f "${RUN_OUT_BASE}/ab_summary.tsv" "${BASELINE_DIR}/ab_summary.tsv"
cp -f "${RUN_OUT_BASE}/ab_summary.csv" "${BASELINE_DIR}/ab_summary.csv"
cp -f "${RUN_OUT_BASE}/speedup.tsv" "${BASELINE_DIR}/speedup.tsv"
cp -f "${RUN_OUT_BASE}/search/best.json" "${BASELINE_DIR}/search/best.json"

for variant in baseline fused; do
  cp -f "${RUN_OUT_BASE}/${variant}/03.after_postbufferize.mlir" \
        "${BASELINE_DIR}/${variant}/03.after_postbufferize.mlir"
  cp -f "${RUN_OUT_BASE}/${variant}/04.after_workgroup_launch.mlir" \
        "${BASELINE_DIR}/${variant}/04.after_workgroup_launch.mlir"
  cp -f "${RUN_OUT_BASE}/${variant}/04c.after_linalg_to_loops.mlir" \
        "${BASELINE_DIR}/${variant}/04c.after_linalg_to_loops.mlir"
  cp -f "${RUN_OUT_BASE}/${variant}/05.out.nvvm.runnable.mlir" \
        "${BASELINE_DIR}/${variant}/05.out.nvvm.runnable.mlir"
done

cat >"${BASELINE_DIR}/manifest.txt" <<EOF
timestamp=$(date -Iseconds)
input_mlir=${INPUT_MLIR}
backend_mode=${BACKEND_MODE}
max_connect_level=${MAX_CONNECT_LEVEL}
run_out_base=${RUN_OUT_BASE}
EOF

echo "[wtc] baseline pinned"
echo "  baseline_dir: ${BASELINE_DIR}"
echo "  manifest:     ${BASELINE_DIR}/manifest.txt"

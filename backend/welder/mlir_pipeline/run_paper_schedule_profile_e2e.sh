#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOLVER_DRIVER="${ROOT_DIR}/compiler/run_welder_solver.sh"
if [[ ! -f "${SOLVER_DRIVER}" ]]; then
  echo "error: solver driver not found at: ${SOLVER_DRIVER}" >&2
  exit 2
fi

IN_MLIR="${SCRIPT_DIR}/matmul_relu_host_shared.mlir"
CACHE="$(mktemp -t welder_paper_profile_cache_XXXXXX.tsv)"
trap 'rm -f "${CACHE}"' EXIT

echo "[1/1] paper schedule + profiling -> ${CACHE}"
bash "${SOLVER_DRIVER}" "${IN_MLIR}" \
  --enable-paper-schedule \
  --auto-candidates \
  --enable-register-level-schedule \
  --enable-profiling \
  --profile-enable-async-copy \
  --profile-enable-software-pipelining \
  --profile-workgroup-multibuffer-depth 2 \
  --profile-warmup 1 \
  --profile-iters 3 \
  --schedule-topk 2 \
  --profile-cache "${CACHE}" \
  >/dev/null

if [[ ! -s "${CACHE}" ]]; then
  echo "FAIL: profiling cache was not produced (or empty): ${CACHE}" >&2
  exit 1
fi

echo "PASS: paper-schedule profiling e2e verified."

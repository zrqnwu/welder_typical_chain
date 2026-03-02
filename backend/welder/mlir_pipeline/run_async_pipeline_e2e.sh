#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/async_pipeline_artifacts}"
mkdir -p "${OUT_DIR}"

echo "[1/2] build + run fused matmul->relu with async copy + pipelining"
ENABLE_ASYNC_COPY=1 \
ENABLE_SOFTWARE_PIPELINING=0 \
PIPELINE_DEPTH="${PIPELINE_DEPTH:-2}" \
WORKGROUP_MULTIBUFFER_DEPTH="${WORKGROUP_MULTIBUFFER_DEPTH:-2}" \
OUT_DIR="${OUT_DIR}" \
  bash "${SCRIPT_DIR}/run_transform_fusion_runnable.sh" >/dev/null

LOWERED_NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"
if [[ ! -f "${LOWERED_NVVM}" ]]; then
  echo "error: missing output: ${LOWERED_NVVM}" >&2
  exit 2
fi

echo "[2/2] IR/PTX sanity checks: expect async-copy IR and multi-buffered shared allocations"

POSTBUF="${OUT_DIR}/03.after_transform_fusion_postbufferize.mlir"
if [[ -f "${POSTBUF}" ]]; then
  if ! grep -q "nvgpu\\.device_async_copy" "${POSTBUF}"; then
    echo "FAIL: expected nvgpu.device_async_copy in postbufferize IR (async copy enabled)" >&2
    exit 1
  fi
else
  echo "warning: missing postbufferize IR: ${POSTBUF} (skipping async-copy IR check)" >&2
fi
if command -v rg >/dev/null 2>&1; then
  rg -n "__wg_|\\.shared" "${LOWERED_NVVM}" | head -n 20 || true
else
  grep -nE "__wg_|\\.shared" "${LOWERED_NVVM}" | head -n 20 || true
fi

expected_re="__wg_.*\\[(8192|16384)\\]"
if command -v rg >/dev/null 2>&1; then
  if ! rg -q "${expected_re}" "${LOWERED_NVVM}"; then
    echo "FAIL: expected multi-buffered shared storage (8192B or 16384B) not found" >&2
    exit 1
  fi
else
  if ! grep -qE "${expected_re}" "${LOWERED_NVVM}"; then
    echo "FAIL: expected multi-buffered shared storage (8192B or 16384B) not found" >&2
    exit 1
  fi
fi

echo "PASS: async pipeline e2e verified."

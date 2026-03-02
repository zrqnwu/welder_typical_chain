#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/software_pipelining_artifacts}"
mkdir -p "${OUT_DIR}"

echo "[1/3] build + run fused matmul->relu with async copy + software pipelining"
ENABLE_ASYNC_COPY=1 \
ENABLE_SOFTWARE_PIPELINING=1 \
PIPELINE_DEPTH="${PIPELINE_DEPTH:-2}" \
WORKGROUP_MULTIBUFFER_DEPTH="${WORKGROUP_MULTIBUFFER_DEPTH:-2}" \
OUT_DIR="${OUT_DIR}" \
  bash "${SCRIPT_DIR}/run_transform_fusion_runnable.sh" >/dev/null

PIPELINED_MLIR="${OUT_DIR}/04b.after_software_pipelining.mlir"
LOWERED_NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"

if [[ ! -f "${PIPELINED_MLIR}" ]]; then
  echo "error: missing output: ${PIPELINED_MLIR}" >&2
  exit 2
fi
if [[ ! -f "${LOWERED_NVVM}" ]]; then
  echo "error: missing output: ${LOWERED_NVVM}" >&2
  exit 2
fi

echo "[2/3] IR sanity check: expect scf.pipelineForLoop result (iter_args)"
if command -v rg >/dev/null 2>&1; then
  rg -n "scf\\.for .* iter_args\\(" "${PIPELINED_MLIR}" >/dev/null
else
  grep -nE "scf\\.for .* iter_args\\(" "${PIPELINED_MLIR}" >/dev/null
fi

POSTBUF="${OUT_DIR}/03.after_transform_fusion_postbufferize.mlir"
if [[ -f "${POSTBUF}" ]]; then
  if ! grep -q "nvgpu\\.device_async_copy" "${POSTBUF}"; then
    echo "FAIL: expected nvgpu.device_async_copy in postbufferize IR (async copy enabled)" >&2
    exit 1
  fi
else
  echo "warning: missing postbufferize IR: ${POSTBUF} (skipping async-copy IR check)" >&2
fi

echo "[3/3] PTX sanity check: expect padded multi-buffered shared allocations"
if command -v rg >/dev/null 2>&1; then
  rg -n "__wg_.*\\[(8224|16448)\\]" "${LOWERED_NVVM}" >/dev/null || {
    echo "FAIL: expected padded multi-buffered shared storage (8224B or 16448B) not found" >&2
    exit 1
  }
else
  grep -qE "__wg_.*\\[(8224|16448)\\]" "${LOWERED_NVVM}" || {
    echo "FAIL: expected padded multi-buffered shared storage (8224B or 16448B) not found" >&2
    exit 1
  }
fi

echo "PASS: software pipelining e2e verified."

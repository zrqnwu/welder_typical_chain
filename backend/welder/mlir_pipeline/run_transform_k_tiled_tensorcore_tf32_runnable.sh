#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLVM_BUILD="${LLVM_BUILD:-}"
if [[ -z "${LLVM_BUILD}" ]]; then
  if [[ -n "${BIN:-}" ]]; then
    LLVM_BUILD="$(cd -- "${BIN}/.." && pwd)"
  else
    for cand in "${ROOT_DIR}/../llvm-project/build" "${ROOT_DIR}/llvm-project/build"; do
      if [[ -x "${cand}/bin/mlir-opt" ]]; then
        LLVM_BUILD="${cand}"
        break
      fi
    done
    if [[ -z "${LLVM_BUILD}" ]] && command -v mlir-opt >/dev/null 2>&1; then
      BIN="$(dirname "$(command -v mlir-opt)")"
      LLVM_BUILD="$(cd -- "${BIN}/.." && pwd)"
    fi
  fi
fi

BIN="${BIN:-${LLVM_BUILD}/bin}"
if [[ -z "${LIB:-}" ]]; then
  if [[ -d "${LLVM_BUILD}/lib" ]]; then
    LIB="${LLVM_BUILD}/lib"
  elif [[ -d "${LLVM_BUILD}/lib64" ]]; then
    LIB="${LLVM_BUILD}/lib64"
  else
    LIB="${LLVM_BUILD}/lib"
  fi
fi

CHIP="${CHIP:-sm_86}"

MLIR_OPT="${BIN}/mlir-opt"
MLIR_RUNNER="${BIN}/mlir-runner"

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  exit 2
fi
if [[ ! -x "${MLIR_RUNNER}" ]]; then
  echo "error: mlir-runner not found at: ${MLIR_RUNNER}" >&2
  exit 2
fi

IN_MLIR="${SCRIPT_DIR}/matmul_host_shared.mlir"
IN_TRANSFORM="${SCRIPT_DIR}/transform_k_tiled_tensorcore_tf32.mlir"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/transform_k_tiled_tensorcore_tf32_artifacts}"
mkdir -p "${OUT_DIR}"

AFTER_TRANSFORM="${OUT_DIR}/01.after_transform.mlir"
AFTER_LOOPS="${OUT_DIR}/02.after_linalg_to_loops.mlir"
LOWERED_NVVM="${OUT_DIR}/03.out.nvvm.runnable.mlir"

echo "[1/4] transform (tile + mma.sync rewrite) -> ${AFTER_TRANSFORM}"
"${MLIR_OPT}" "${IN_MLIR}" \
  --transform-preload-library="transform-library-paths=${IN_TRANSFORM}" \
  --transform-interpreter \
  --canonicalize \
  -o "${AFTER_TRANSFORM}"

echo "[2/4] linalg-to-loops -> ${AFTER_LOOPS}"
"${MLIR_OPT}" "${AFTER_TRANSFORM}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops)" \
  -o "${AFTER_LOOPS}"

echo "[3/4] nvvm pipeline -> ${LOWERED_NVVM}"
"${MLIR_OPT}" "${AFTER_LOOPS}" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CHIP} cubin-format=isa" \
  -o "${LOWERED_NVVM}"

echo "PTX sanity check (expect mma.sync):"
if command -v rg >/dev/null 2>&1; then
  rg -n "mma\\.sync|wmma" "${LOWERED_NVVM}" | head -n 20 || true
else
  grep -nE "mma\\.sync|wmma" "${LOWERED_NVVM}" | head -n 20 || true
fi

echo "[4/4] run with mlir-runner (expected output: 128)"
"${MLIR_RUNNER}" "${LOWERED_NVVM}" \
  -e main \
  --entry-point-result=void \
  -shared-libs="${LIB}/libmlir_runner_utils.so,${LIB}/libmlir_c_runner_utils.so,${LIB}/libmlir_cuda_runtime.so"


#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 自动发现 LLVM/MLIR build（和 mlir_pipeline 里的脚本保持一致的习惯）。
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

LLVM_DIR="${LLVM_DIR:-${LIB}/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-${LIB}/cmake/mlir}"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"

need_configure=0
if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  need_configure=1
fi
if [[ ! -f "${BUILD_DIR}/build.ninja" && ! -f "${BUILD_DIR}/Makefile" ]]; then
  need_configure=1
fi

if [[ "${need_configure}" -eq 1 ]]; then
  cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${BUILD_DIR}" -j

SOLVER="${BUILD_DIR}/welder-solver"
if [[ ! -x "${SOLVER}" ]]; then
  echo "error: welder-solver not found at: ${SOLVER}" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  set -- "${ROOT_DIR}/mlir_pipeline/matmul_relu_host_shared.mlir"
fi

exec "${SOLVER}" "$@"


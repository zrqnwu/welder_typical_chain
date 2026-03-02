#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 约定：
# - 你的 LLVM/MLIR build 通常在：../llvm-project/build（Welder 源码/笔记本常见布局）
# - 也有人会把 llvm-project 放到仓库内：./llvm-project/build
# - 或者你直接用环境变量指定：BIN=/path/to/build/bin
#
# 我们在这里做一个“自动发现”，让脚本在大多数布局下都能开箱即用。
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
CHIP="${CHIP:-sm_86}"

MLIR_OPT="${BIN}/mlir-opt"
MLIR_RUNNER="${BIN}/mlir-runner"

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  echo "hint: set BIN=/path/to/llvm-project/build/bin (or LLVM_BUILD=/path/to/llvm-project/build)" >&2
  exit 2
fi
if [[ ! -x "${MLIR_RUNNER}" ]]; then
  echo "error: mlir-runner not found at: ${MLIR_RUNNER}" >&2
  echo "hint: set BIN=/path/to/llvm-project/build/bin (or LLVM_BUILD=/path/to/llvm-project/build)" >&2
  exit 2
fi

PASS_SRC_DIR="${SCRIPT_DIR}/workgroup_alloc_to_launch_pass"
PASS_BUILD_DIR="${PASS_BUILD_DIR:-${PASS_SRC_DIR}/build}"
PASS_LIB=""

find_pass_lib() {
  local build_dir="$1"
  local -a cands=(
    "${build_dir}/WorkgroupAllocToLaunchPass.so"
    "${build_dir}/libWorkgroupAllocToLaunchPass.so"
    "${build_dir}/WorkgroupAllocToLaunchPass.dylib"
    "${build_dir}/libWorkgroupAllocToLaunchPass.dylib"
  )
  for c in "${cands[@]}"; do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

PASS_LIB="$(find_pass_lib "${PASS_BUILD_DIR}" || true)"

need_configure=0
if [[ ! -f "${PASS_BUILD_DIR}/CMakeCache.txt" ]]; then
  need_configure=1
fi
if [[ ! -f "${PASS_BUILD_DIR}/build.ninja" && ! -f "${PASS_BUILD_DIR}/Makefile" ]]; then
  need_configure=1
fi

if [[ "${need_configure}" -eq 1 ]]; then
  echo "[0/4] configure pass plugin"
  cmake -S "${PASS_SRC_DIR}" -B "${PASS_BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
fi
echo "[0/4] build pass plugin"
cmake --build "${PASS_BUILD_DIR}" -j
PASS_LIB="$(find_pass_lib "${PASS_BUILD_DIR}" || true)"

if [[ ! -f "${PASS_LIB}" ]]; then
  echo "error: pass plugin not found under: ${PASS_BUILD_DIR}" >&2
  exit 2
fi

IN_MLIR="${SCRIPT_DIR}/matmul_host_shared.mlir"
IN_TRANSFORM="${SCRIPT_DIR}/transform_k_tiled.mlir"
WORKGROUP_PAD_LAST_DIM="${WORKGROUP_PAD_LAST_DIM:-0}"
WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY="${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY:-0}"
WORKGROUP_MULTIBUFFER_DEPTH="${WORKGROUP_MULTIBUFFER_DEPTH:-1}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/transform_k_tiled_artifacts}"
mkdir -p "${OUT_DIR}"

AFTER_TRANSFORM="${OUT_DIR}/matmul.host_shared.tiled.promoted.gpu.mlir"
AFTER_REWRITE="${OUT_DIR}/matmul.host_shared.tiled.promoted.gpu.workgroup_launch.mlir"
LOWERED_NVVM="${OUT_DIR}/out.nvvm.runnable.mlir"

echo "[1/4] transform -> ${AFTER_TRANSFORM}"
"${MLIR_OPT}" "${IN_MLIR}" \
  --transform-preload-library="transform-library-paths=${IN_TRANSFORM}" \
  --transform-interpreter \
  -o "${AFTER_TRANSFORM}"

echo "[2/4] linalg-to-loops + workgroup-alloc->gpu.launch workgroup(...) -> ${AFTER_REWRITE}"
WG_PASS="workgroup-alloc-to-launch-workgroup"
declare -a WG_OPTS=()
if [[ "${WORKGROUP_PAD_LAST_DIM}" -ne 0 ]]; then
  WG_OPTS+=("workgroup-pad-last-dim=${WORKGROUP_PAD_LAST_DIM}")
fi
if [[ "${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY}" -eq 1 ]]; then
  WG_OPTS+=("workgroup-pad-last-dim-matmul-only=true")
fi
if [[ "${WORKGROUP_MULTIBUFFER_DEPTH}" -ne 1 ]]; then
  WG_OPTS+=("workgroup-multibuffer-depth=${WORKGROUP_MULTIBUFFER_DEPTH}")
fi
if [[ "${#WG_OPTS[@]}" -ne 0 ]]; then
  WG_PASS="workgroup-alloc-to-launch-workgroup{${WG_OPTS[*]// /,}}"
fi
"${MLIR_OPT}" "${AFTER_TRANSFORM}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops,${WG_PASS})" \
  -o "${AFTER_REWRITE}"

echo "sanity: should NOT contain memref.dealloc(workgroup)"
if command -v rg >/dev/null 2>&1; then
  rg -n "memref\\.dealloc.*#gpu\\.address_space<workgroup>" "${AFTER_REWRITE}" && exit 1 || true
else
  grep -nE "memref\\.dealloc.*#gpu\\.address_space<workgroup>" "${AFTER_REWRITE}" && exit 1 || true
fi

echo "[3/4] nvvm pipeline -> ${LOWERED_NVVM}"
"${MLIR_OPT}" "${AFTER_REWRITE}" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=${CHIP} cubin-format=isa" \
  -o "${LOWERED_NVVM}"

echo "shared-memory sanity check (expect .shared / ld.shared / st.shared):"
if command -v rg >/dev/null 2>&1; then
  rg -n "\\.shared|ld\\.shared|st\\.shared|__wg_" "${LOWERED_NVVM}" | head -n 5 || true
else
  grep -nE "\\.shared|ld\\.shared|st\\.shared|__wg_" "${LOWERED_NVVM}" | head -n 5 || true
fi

echo "[4/4] run with mlir-runner (expected output: 128)"
"${MLIR_RUNNER}" "${LOWERED_NVVM}" \
  -e main \
  --entry-point-result=void \
  -shared-libs="${LIB}/libmlir_runner_utils.so,${LIB}/libmlir_c_runner_utils.so,${LIB}/libmlir_cuda_runtime.so"

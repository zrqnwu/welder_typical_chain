#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 复用 run_transform_fusion_runnable.sh 的“自动发现 LLVM/MLIR build”逻辑。
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
if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found at: ${MLIR_OPT}" >&2
  echo "hint: set BIN=/path/to/llvm-project/build/bin (or LLVM_BUILD=/path/to/llvm-project/build)" >&2
  exit 2
fi

IN_MLIR="${1:-${SCRIPT_DIR}/conv2d_relu_generic.mlir}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/conv2d_artifacts}"
mkdir -p "${OUT_DIR}"
WORKGROUP_PAD_LAST_DIM="${WORKGROUP_PAD_LAST_DIM:-0}"

AFTER_T1="${OUT_DIR}/01.after_prebufferize.mlir"
AFTER_BUF="${OUT_DIR}/02.after_bufferize.mlir"
AFTER_T2="${OUT_DIR}/03.after_postbufferize.mlir"
AFTER_LOOPS="${OUT_DIR}/04.after_linalg_to_loops_and_workgroup_launch.mlir"
LOWERED_NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"

# 插件：把 memref.alloc/dealloc(workgroup) 改成 gpu.launch workgroup(...)（并删 dealloc）。
PASS_SRC_DIR="${SCRIPT_DIR}/workgroup_alloc_to_launch_pass"
PASS_BUILD_DIR="${PASS_BUILD_DIR:-${PASS_SRC_DIR}/build}"

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
  echo "[0/5] configure pass plugin"
  cmake -S "${PASS_SRC_DIR}" -B "${PASS_BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
fi
echo "[0/5] build pass plugin"
cmake --build "${PASS_BUILD_DIR}" -j
PASS_LIB="$(find_pass_lib "${PASS_BUILD_DIR}" || true)"
if [[ ! -f "${PASS_LIB}" ]]; then
  echo "error: pass plugin not found under: ${PASS_BUILD_DIR}" >&2
  exit 2
fi

echo "[1/5] welder-compiler (generic solve + fusion) -> ${AFTER_T2}"
COMPILER_DRIVER="${ROOT_DIR}/compiler/run_welder_compiler.sh"
LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
  bash "${COMPILER_DRIVER}" "${IN_MLIR}" \
    --enable-generic-problem \
    --enable-generic-fusion \
    --emit-after-prebufferize "${AFTER_T1}" \
    --emit-after-bufferize "${AFTER_BUF}" \
    --output "${AFTER_T2}"

echo "[2/5] linalg-to-loops + workgroup-alloc->gpu.launch workgroup(...) -> ${AFTER_LOOPS}"
WG_PASS="workgroup-alloc-to-launch-workgroup"
if [[ "${WORKGROUP_PAD_LAST_DIM}" -ne 0 ]]; then
  WG_PASS="workgroup-alloc-to-launch-workgroup{workgroup-pad-last-dim=${WORKGROUP_PAD_LAST_DIM}}"
fi
"${MLIR_OPT}" "${AFTER_T2}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops,${WG_PASS})" \
  -o "${AFTER_LOOPS}"

echo "[3/5] nvvm pipeline -> ${LOWERED_NVVM}"
NVVM_PIPELINE="builtin.module(\
convert-nvgpu-to-nvvm,\
gpu-kernel-outlining,\
convert-vector-to-scf{target-rank=1},\
convert-vector-to-llvm,\
convert-scf-to-cf,\
convert-nvvm-to-llvm,\
convert-func-to-llvm,\
expand-strided-metadata,\
nvvm-attach-target{chip=${CHIP}},\
lower-affine,\
convert-arith-to-llvm,\
convert-index-to-llvm,\
canonicalize,\
cse,\
gpu.module(convert-gpu-to-nvvm,canonicalize,cse,reconcile-unrealized-casts),\
gpu-to-llvm,\
reconcile-unrealized-casts,\
llvm-vector4-align,\
gpu-module-to-binary{format=isa},\
convert-math-to-llvm,\
canonicalize,\
cse,\
reconcile-unrealized-casts)"

"${MLIR_OPT}" "${AFTER_LOOPS}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="${NVVM_PIPELINE}" \
  -o "${LOWERED_NVVM}"

echo "[4/5] PTX sanity check (expect .entry + ld/st + maybe max.* for ReLU):"
if command -v rg >/dev/null 2>&1; then
  rg -n "\\.entry|ld\\.global|st\\.global|max\\.|max\\.rn\\.|\\%ctaid\\.|\\%tid\\." "${LOWERED_NVVM}" | head -n 40 || true
else
  grep -nE "\\.entry|ld\\.global|st\\.global|max\\.|max\\.rn\\.|\\%ctaid\\.|\\%tid\\." "${LOWERED_NVVM}" | head -n 40 || true
fi

echo "[5/5] done: ${LOWERED_NVVM}"

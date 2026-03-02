#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Auto-discover LLVM/MLIR build (same as other scripts).
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

IN_MLIR="${1:-${SCRIPT_DIR}/tensorcore_f16_two_sinks.mlir}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/tensorcore_f16_two_sinks_artifacts}"
mkdir -p "${OUT_DIR}"

AFTER_WELDER="${OUT_DIR}/01.after_welder_compiler.mlir"
AFTER_LOOPS="${OUT_DIR}/02.after_linalg_to_loops_and_workgroup_launch.mlir"
LOWERED_NVVM="${OUT_DIR}/03.out.nvvm.runnable.mlir"

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

echo "[1/4] welder-compiler (generic + cut-edges + tensorcore f16) -> ${AFTER_WELDER}"
COMPILER_DRIVER="${ROOT_DIR}/compiler/run_welder_compiler.sh"
LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
  bash "${COMPILER_DRIVER}" "${IN_MLIR}" \
    --enable-generic-problem \
    --enable-cut-edges \
    --enable-tensorcore-f16 \
    --candidates-mn=16,32 \
    --candidates-k=16 \
    --output "${AFTER_WELDER}"

echo "[2/4] linalg-to-loops + workgroup-alloc->gpu.launch workgroup(...) -> ${AFTER_LOOPS}"
"${MLIR_OPT}" "${AFTER_WELDER}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops,workgroup-alloc-to-launch-workgroup)" \
  -o "${AFTER_LOOPS}"

echo "[3/4] nvvm pipeline -> ${LOWERED_NVVM}"
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

echo "[4/4] sanity checks: expect >=2 PTX .entry and mma.sync"
entry_count="$(grep -c "\\.entry" "${LOWERED_NVVM}" || true)"
mma_count="$(grep -c "mma\\.sync" "${LOWERED_NVVM}" || true)"
if [[ "${entry_count}" -ge 2 && "${mma_count}" -ge 1 ]]; then
  echo "PASS: Found ${entry_count} .entry (>=2) and ${mma_count} mma.sync."
else
  echo "FAIL: Found ${entry_count} .entry and ${mma_count} mma.sync (expected >=2 and >=1)." >&2
  echo "hint: inspect ${LOWERED_NVVM} and search for '.entry' / 'mma.sync'." >&2
  exit 1
fi

echo "done: ${LOWERED_NVVM}"


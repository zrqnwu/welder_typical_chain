#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <input.mlir> [welder-compiler flags...]" >&2
  exit 2
fi

IN_MLIR="$1"
shift

# Auto-discover LLVM/MLIR build (same convention as other scripts).
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

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/nvvm_isa_artifacts}"
mkdir -p "${OUT_DIR}"

AFTER_T2="${OUT_DIR}/03.after_postbufferize.mlir"
AFTER_WG="${OUT_DIR}/04.after_workgroup_launch.mlir"
AFTER_PIPE="${OUT_DIR}/04b.after_software_pipelining.mlir"
AFTER_LOOPS="${OUT_DIR}/04c.after_linalg_to_loops.mlir"
LOWERED_NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"

# Build pass plugin (workgroup_alloc_to_launch_pass).
PASS_SRC_DIR="${ROOT_DIR}/mlir_pipeline/workgroup_alloc_to_launch_pass"
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
  cmake -S "${PASS_SRC_DIR}" -B "${PASS_BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${PASS_BUILD_DIR}" -j
PASS_LIB="$(find_pass_lib "${PASS_BUILD_DIR}" || true)"
if [[ ! -f "${PASS_LIB}" ]]; then
  echo "error: pass plugin not found under: ${PASS_BUILD_DIR}" >&2
  exit 2
fi

echo "[1/3] welder-compiler -> ${AFTER_T2}"
COMPILER_DRIVER="${SCRIPT_DIR}/run_welder_compiler.sh"
# Keep workgroup 填充 默认值 consistent across compiler-side attributes and
# the downstream workgroup pass options.
WORKGROUP_PAD_LAST_DIM="${WORKGROUP_PAD_LAST_DIM:-0}"
WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY="${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY:-0}"

# 论文/Welder 对齐 (TCPolicy): TensorCore schedules typically use a padded
# shared-memory layout with stride offset=8. In this MLIR pipeline, that maps
# to 填充 the last dim by 8 elements. If the user didn't specify a 填充
# value explicitly, 默认 to 8 when TensorCore is enabled.
if [[ "${WORKGROUP_PAD_LAST_DIM}" -eq 0 ]]; then
  for a in "$@"; do
    if [[ "${a}" == "--enable-tensorcore-f16" || "${a}" == "--enable-tensorcore-tf32" ]]; then
      WORKGROUP_PAD_LAST_DIM=8
      # 论文/Welder 对齐: TensorCore stride offset primarily applies to the
      # matmul operand tiles in shared 内存.
      if [[ "${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY}" -eq 0 ]]; then
        WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY=1
      fi
      break
    fi
  done
fi

EXTRA_COMPILER_ARGS=()
if [[ "${WORKGROUP_PAD_LAST_DIM}" -ne 0 ]]; then
  EXTRA_COMPILER_ARGS+=(--workgroup-pad-last-dim "${WORKGROUP_PAD_LAST_DIM}")
  if [[ "${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY}" -eq 1 ]]; then
    EXTRA_COMPILER_ARGS+=(--workgroup-pad-last-dim-matmul-only)
  fi
fi

LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
  bash "${COMPILER_DRIVER}" "${IN_MLIR}" \
    "$@" \
    "${EXTRA_COMPILER_ARGS[@]}" \
    --output "${AFTER_T2}"

echo "[2/3] workgroup-alloc->gpu.launch (+ optional pipelining) + linalg-to-loops -> ${AFTER_LOOPS}"
WORKGROUP_MULTIBUFFER_DEPTH="${WORKGROUP_MULTIBUFFER_DEPTH:-1}"
WORKGROUP_SWIZZLE_XOR="${WORKGROUP_SWIZZLE_XOR:-0}"
BLOCK_RASTERIZE_XOR="${BLOCK_RASTERIZE_XOR:-0}"
BLOCK_RASTERIZE_MODE="${BLOCK_RASTERIZE_MODE:-0}"
BLOCK_RASTERIZE_PANEL_WIDTH="${BLOCK_RASTERIZE_PANEL_WIDTH:-0}"
ENABLE_SOFTWARE_PIPELINING="${ENABLE_SOFTWARE_PIPELINING:-0}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-2}"
PIPELINE_PEEL_EPILOGUE="${PIPELINE_PEEL_EPILOGUE:-1}"

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
if [[ "${WORKGROUP_SWIZZLE_XOR}" -ne 0 ]]; then
  WG_OPTS+=("workgroup-swizzle-xor=${WORKGROUP_SWIZZLE_XOR}")
fi
if [[ "${BLOCK_RASTERIZE_XOR}" -ne 0 ]]; then
  WG_OPTS+=("block-rasterize-xor=${BLOCK_RASTERIZE_XOR}")
fi
if [[ "${BLOCK_RASTERIZE_MODE}" -ne 0 ]]; then
  WG_OPTS+=("block-rasterize-mode=${BLOCK_RASTERIZE_MODE}")
fi
if [[ "${BLOCK_RASTERIZE_PANEL_WIDTH}" -ne 0 ]]; then
  WG_OPTS+=("block-rasterize-panel-width=${BLOCK_RASTERIZE_PANEL_WIDTH}")
fi
if [[ "${#WG_OPTS[@]}" -ne 0 ]]; then
  WG_PASS="workgroup-alloc-to-launch-workgroup{${WG_OPTS[*]// /,}}"
fi

echo "  [2.1] workgroup-alloc->gpu.launch workgroup(...) -> ${AFTER_WG}"
"${MLIR_OPT}" "${AFTER_T2}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="builtin.module(${WG_PASS})" \
  -o "${AFTER_WG}"

PIPELINED_PAYLOAD="${AFTER_WG}"
if [[ "${ENABLE_SOFTWARE_PIPELINING}" -eq 1 ]]; then
  echo "  [2.2] software pipelining (welder-pipeline) -> ${AFTER_PIPE}"
  PIPELINE_DRIVER="${SCRIPT_DIR}/run_welder_pipeline.sh"
  if [[ ! -f "${PIPELINE_DRIVER}" ]]; then
    echo "error: pipeline driver not found at: ${PIPELINE_DRIVER}" >&2
    exit 2
  fi
  PIPELINE_SET_ASYNC_WAIT_GROUPS="${PIPELINE_SET_ASYNC_WAIT_GROUPS:-0}"
  extra_peel=()
  if [[ "${PIPELINE_PEEL_EPILOGUE}" -eq 0 ]]; then
    extra_peel+=(--pipeline-peel-epilogue=false)
  fi
  extra_wait_groups=()
  if [[ "${PIPELINE_SET_ASYNC_WAIT_GROUPS}" -eq 1 ]]; then
    extra_wait_groups+=(--pipeline-set-async-wait-groups)
  fi
  LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
    bash "${PIPELINE_DRIVER}" "${AFTER_WG}" \
      --pipeline-depth "${PIPELINE_DEPTH}" \
      "${extra_peel[@]}" \
      "${extra_wait_groups[@]}" \
      --output "${AFTER_PIPE}"
  PIPELINED_PAYLOAD="${AFTER_PIPE}"
fi

echo "  [2.3] linalg-to-loops -> ${AFTER_LOOPS}"
"${MLIR_OPT}" "${PIPELINED_PAYLOAD}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops)" \
  -o "${AFTER_LOOPS}"

echo "[3/3] nvvm pipeline -> ${LOWERED_NVVM}"
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
gpu.module(convert-vector-to-scf{target-rank=1},convert-vector-to-llvm,convert-scf-to-cf,convert-gpu-to-nvvm,canonicalize,cse,reconcile-unrealized-casts),\
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

echo "${LOWERED_NVVM}"

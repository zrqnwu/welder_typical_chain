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

IN_MLIR="${SCRIPT_DIR}/matmul_relu_host_shared.mlir"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/fusion_artifacts}"
mkdir -p "${OUT_DIR}"
WORKGROUP_PAD_LAST_DIM="${WORKGROUP_PAD_LAST_DIM:-0}"
WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY="${WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY:-0}"
WORKGROUP_MULTIBUFFER_DEPTH="${WORKGROUP_MULTIBUFFER_DEPTH:-1}"

# === Phase 6: Integration ===
# 默认启用 C++ welder-compiler：不再依赖 *.mlir 模板，也不需要 sed。
# 若你想回退到旧的 “mlir-opt + template + sed” 流程：USE_WELDER_COMPILER=0
USE_WELDER_COMPILER="${USE_WELDER_COMPILER:-1}"
USE_SOLVER="${USE_SOLVER:-1}" # 仅在 USE_WELDER_COMPILER=0 时生效（旧流程）。
THREAD_TILE_M="${THREAD_TILE_M:-4}"
THREAD_TILE_N="${THREAD_TILE_N:-4}"
ENABLE_ASYNC_COPY="${ENABLE_ASYNC_COPY:-0}"
ENABLE_SOFTWARE_PIPELINING="${ENABLE_SOFTWARE_PIPELINING:-0}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-2}"

T_FUSION_1="${SCRIPT_DIR}/transform_fusion_prebufferize.mlir"
T_FUSION_2="${SCRIPT_DIR}/transform_fusion_postbufferize.mlir"

AFTER_T1="${OUT_DIR}/01.after_transform_fusion_prebufferize.mlir"
AFTER_BUF="${OUT_DIR}/02.after_bufferize.mlir"
AFTER_T2="${OUT_DIR}/03.after_transform_fusion_postbufferize.mlir"
AFTER_WG="${OUT_DIR}/04.after_workgroup_launch.mlir"
AFTER_LOOPS="${OUT_DIR}/04.after_linalg_to_loops_and_workgroup_launch.mlir"
LOWERED_NVVM="${OUT_DIR}/05.out.nvvm.runnable.mlir"

# 复用我们已有的插件：把 memref.alloc/dealloc(workgroup) 改成 gpu.launch workgroup(...)（并删 dealloc）
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
  echo "[0/6] configure pass plugin"
  cmake -S "${PASS_SRC_DIR}" -B "${PASS_BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
fi
echo "[0/6] build pass plugin"
cmake --build "${PASS_BUILD_DIR}" -j
PASS_LIB="$(find_pass_lib "${PASS_BUILD_DIR}" || true)"
if [[ ! -f "${PASS_LIB}" ]]; then
  echo "error: pass plugin not found under: ${PASS_BUILD_DIR}" >&2
  exit 2
fi

if [[ "${USE_WELDER_COMPILER}" -eq 1 ]]; then
  echo "[1/6] welder-compiler (solve + transform + bufferize) -> ${AFTER_T2}"
  COMPILER_DRIVER="${ROOT_DIR}/compiler/run_welder_compiler.sh"
  if [[ ! -f "${COMPILER_DRIVER}" ]]; then
    echo "error: compiler driver not found at: ${COMPILER_DRIVER}" >&2
    echo "hint: check repo has compiler/run_welder_compiler.sh" >&2
    exit 2
  fi

  EXTRA_COMPILER_FLAGS=()
  if [[ "${ENABLE_ASYNC_COPY}" -eq 1 ]]; then
    EXTRA_COMPILER_FLAGS+=(--enable-async-copy)
  fi
  if [[ "${ENABLE_SOFTWARE_PIPELINING}" -eq 1 ]]; then
    EXTRA_COMPILER_FLAGS+=(--enable-software-pipelining --pipeline-depth "${PIPELINE_DEPTH}")
  fi

  LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
    bash "${COMPILER_DRIVER}" "${IN_MLIR}" \
      --thread-tile-m "${THREAD_TILE_M}" \
      --thread-tile-n "${THREAD_TILE_N}" \
      "${EXTRA_COMPILER_FLAGS[@]}" \
      --emit-after-prebufferize "${AFTER_T1}" \
      --emit-after-bufferize "${AFTER_BUF}" \
      --output "${AFTER_T2}"
else
  # === 旧流程（保留作对照 / debug）===
  # solver -> template(sed) -> mlir-opt (pre) -> bufferize -> mlir-opt (post)
  if [[ "${USE_SOLVER}" -eq 1 ]]; then
    echo "[1/6] run welder-solver -> pick tile sizes"
    SOLVER_DRIVER="${ROOT_DIR}/compiler/run_welder_solver.sh"
    if [[ ! -f "${SOLVER_DRIVER}" ]]; then
      echo "error: solver driver not found at: ${SOLVER_DRIVER}" >&2
      echo "hint: check repo has compiler/run_welder_solver.sh" >&2
      exit 2
    fi

    SOLVER_OUTPUT="$(
      LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
        bash "${SOLVER_DRIVER}" "${IN_MLIR}"
    )"
    echo "${SOLVER_OUTPUT}"

    BEST_LINE="$(echo "${SOLVER_OUTPUT}" | grep -E '^BEST_TILE ' | head -n 1 || true)"
    if [[ -z "${BEST_LINE}" ]]; then
      echo "error: failed to parse BEST_TILE from solver output" >&2
      exit 2
    fi

    TILE_M="$(echo "${BEST_LINE}" | sed -n 's/.*tile_m=\([0-9][0-9]*\).*/\1/p')"
    TILE_N="$(echo "${BEST_LINE}" | sed -n 's/.*tile_n=\([0-9][0-9]*\).*/\1/p')"
    TILE_K="$(echo "${BEST_LINE}" | sed -n 's/.*tile_k=\([0-9][0-9]*\).*/\1/p')"

    if [[ -z "${TILE_M}" || -z "${TILE_N}" || -z "${TILE_K}" ]]; then
      echo "error: parsed empty tile sizes from solver output: ${BEST_LINE}" >&2
      exit 2
    fi

    if (( TILE_M % THREAD_TILE_M != 0 )); then
      echo "error: TILE_M=${TILE_M} is not divisible by THREAD_TILE_M=${THREAD_TILE_M}" >&2
      exit 2
    fi
    if (( TILE_N % THREAD_TILE_N != 0 )); then
      echo "error: TILE_N=${TILE_N} is not divisible by THREAD_TILE_N=${THREAD_TILE_N}" >&2
      exit 2
    fi

    # block_dims 的顺序是 [x, y, z]，而我们把 (M,N) 映射成 (thread<y>, thread<x>)。
    BLOCK_DIM_X=$(( TILE_N / THREAD_TILE_N ))
    BLOCK_DIM_Y=$(( TILE_M / THREAD_TILE_M ))
    if (( BLOCK_DIM_X * BLOCK_DIM_Y > 1024 )); then
      echo "error: block_dims (${BLOCK_DIM_X}x${BLOCK_DIM_Y}) exceeds 1024 threads" >&2
      exit 2
    fi

    echo ">>> Auto-Tuned Config: TILE_M=${TILE_M} TILE_N=${TILE_N} TILE_K=${TILE_K} block_dims=[${BLOCK_DIM_X},${BLOCK_DIM_Y},1]"

    T1_TEMPLATE="${SCRIPT_DIR}/transform_fusion_prebufferize_template.mlir"
    T2_TEMPLATE="${SCRIPT_DIR}/transform_fusion_postbufferize_template.mlir"
    if [[ ! -f "${T1_TEMPLATE}" || ! -f "${T2_TEMPLATE}" ]]; then
      echo "error: transform templates not found: ${T1_TEMPLATE} / ${T2_TEMPLATE}" >&2
      exit 2
    fi

    T1_CONCRETE="${OUT_DIR}/00.transform_fusion_prebufferize.concrete.mlir"
    T2_CONCRETE="${OUT_DIR}/00.transform_fusion_postbufferize.concrete.mlir"

    sed \
      -e "s/\\\${TILE_M}/${TILE_M}/g" \
      -e "s/\\\${TILE_N}/${TILE_N}/g" \
      "${T1_TEMPLATE}" > "${T1_CONCRETE}"

    sed \
      -e "s/\\\${TILE_K}/${TILE_K}/g" \
      -e "s/\\\${BLOCK_DIM_X}/${BLOCK_DIM_X}/g" \
      -e "s/\\\${BLOCK_DIM_Y}/${BLOCK_DIM_Y}/g" \
      "${T2_TEMPLATE}" > "${T2_CONCRETE}"

    echo ">>> Generated concrete transforms:"
    echo "    - ${T1_CONCRETE}"
    echo "    - ${T2_CONCRETE}"
    T_FUSION_1="${T1_CONCRETE}"
    T_FUSION_2="${T2_CONCRETE}"
  fi

  echo "[1/6] transform (fusion, tensor stage) -> ${AFTER_T1}"
  "${MLIR_OPT}" "${IN_MLIR}" \
    --transform-preload-library="transform-library-paths=${T_FUSION_1}" \
    --transform-interpreter \
    --canonicalize \
    -o "${AFTER_T1}"

  echo "[2/6] one-shot-bufferize -> ${AFTER_BUF}"
  "${MLIR_OPT}" "${AFTER_T1}" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --canonicalize \
    -o "${AFTER_BUF}"

  echo "[3/6] transform (K-tiling + shared promotion + gpu mapping) -> ${AFTER_T2}"
  "${MLIR_OPT}" "${AFTER_BUF}" \
    --transform-preload-library="transform-library-paths=${T_FUSION_2}" \
    --transform-interpreter \
    --canonicalize \
    -o "${AFTER_T2}"
fi

echo "[4/6] workgroup-alloc->gpu.launch workgroup(...) -> ${AFTER_WG}"
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
"${MLIR_OPT}" "${AFTER_T2}" \
  --load-pass-plugin="${PASS_LIB}" \
  --pass-pipeline="builtin.module(${WG_PASS})" \
  -o "${AFTER_WG}"

PIPELINED_PAYLOAD="${AFTER_WG}"
if [[ "${ENABLE_SOFTWARE_PIPELINING}" -eq 1 ]]; then
  echo "[4.5/6] software pipelining (scf.pipelineForLoop) -> ${OUT_DIR}/04b.after_software_pipelining.mlir"
  PIPELINED_PAYLOAD="${OUT_DIR}/04b.after_software_pipelining.mlir"
  PIPELINE_DRIVER="${ROOT_DIR}/compiler/run_welder_pipeline.sh"
  if [[ ! -f "${PIPELINE_DRIVER}" ]]; then
    echo "error: pipeline driver not found at: ${PIPELINE_DRIVER}" >&2
    exit 2
  fi
  LLVM_BUILD="${LLVM_BUILD}" BIN="${BIN}" LIB="${LIB}" LLVM_DIR="${LLVM_DIR}" MLIR_DIR="${MLIR_DIR}" \
    bash "${PIPELINE_DRIVER}" "${AFTER_WG}" \
      --pipeline-depth "${PIPELINE_DEPTH}" \
      --output "${PIPELINED_PAYLOAD}"
fi

echo "[4.8/6] linalg-to-loops -> ${AFTER_LOOPS}"
"${MLIR_OPT}" "${PIPELINED_PAYLOAD}" \
  --pass-pipeline="builtin.module(convert-linalg-to-loops)" \
  -o "${AFTER_LOOPS}"

echo "sanity: should NOT contain memref.dealloc(workgroup)"
if command -v rg >/dev/null 2>&1; then
  rg -n "memref\\.dealloc.*(#gpu\\.address_space<workgroup>|,\\s*3\\s*>)" "${AFTER_LOOPS}" && exit 1 || true
else
  grep -nE "memref\\.dealloc.*(#gpu\\.address_space<workgroup>|,[[:space:]]*3[[:space:]]*>)" "${AFTER_LOOPS}" && exit 1 || true
fi

echo "[5/6] nvvm pipeline -> ${LOWERED_NVVM}"
#
# 注意：这里我们不用 --gpu-lower-to-nvvm-pipeline，而是展开成显式 pipeline。
# 原因：当我们对 linalg.copy 做了 vectorize 之后，会在 lowering 过程中引入
# builtin.unrealized_conversion_cast；如果不在 “gpu-module-to-binary(会触发 LLVM 翻译/序列化)”
# 之前 reconcile，这一步会直接失败。
#
# 这个 pipeline 基本等价于 MLIR 源码里的 buildLowerToNVVMPassPipeline，
# 只是把 reconcile-unrealized-casts 提前到了 gpu-module-to-binary 之前。
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

echo "PTX sanity check (expect .shared + tid.x/tid.y):"
if command -v rg >/dev/null 2>&1; then
  rg -n "__wg_|\\.shared|ld\\.shared|st\\.shared|\\%tid\\.x|\\%tid\\.y|max\\.f32|max\\.rn\\.f32" "${LOWERED_NVVM}" | head -n 20 || true
else
  grep -nE "__wg_|\\.shared|ld\\.shared|st\\.shared|\\%tid\\.x|\\%tid\\.y|max\\.f32|max\\.rn\\.f32" "${LOWERED_NVVM}" | head -n 20 || true
fi

echo "[6/6] run with mlir-runner (expected output: 128)"
"${MLIR_RUNNER}" "${LOWERED_NVVM}" \
  -e main \
  --entry-point-result=void \
  -shared-libs="${LIB}/libmlir_runner_utils.so,${LIB}/libmlir_c_runner_utils.so,${LIB}/libmlir_cuda_runtime.so"

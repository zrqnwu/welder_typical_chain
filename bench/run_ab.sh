#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_MLIR="${INPUT_MLIR:-${ROOT_DIR}/mlir/matmul_softmax_chain_f16_native.mlir}"
OUT_BASE="${OUT_BASE:-/tmp/wtc_ab}"
BACKEND_ROOT="${BACKEND_ROOT:-/home/zhangruiqi/welder_typical_chain/backend/welder}"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  BACKEND_ROOT="${LEGACY_ROOT}"
fi
BACKEND_MODE="${BACKEND_MODE:-process_chain}"
MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL:-1}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-200}"
VERIFY="${VERIFY:-0}"
VERIFY_WARMUP="${VERIFY_WARMUP:-0}"
VERIFY_ITERS="${VERIFY_ITERS:-1}"
VERIFY_RTOL="${VERIFY_RTOL:-1e-3}"
VERIFY_ATOL="${VERIFY_ATOL:-1e-3}"

PATTERN="${PATTERN:-linear}"
SEED="${SEED:-1}"

M="${M:-8192}"
K="${K:-64}"
N="${N:-128}"
ASTRIDE0="${ASTRIDE0:-${K}}"
ASTRIDE1="${ASTRIDE1:-1}"
BSTRIDE0="${BSTRIDE0:-${N}}"
BSTRIDE1="${BSTRIDE1:-1}"

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "error: input mlir not found: ${INPUT_MLIR}" >&2
  exit 2
fi

# 1) Build new project compiler.
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" >/dev/null
cmake --build "${ROOT_DIR}/build" -j >/dev/null

# 2) Compile baseline + fused with the same searched best tile.
mkdir -p "${OUT_BASE}"
INPUT_MLIR="${INPUT_MLIR}" OUT_BASE="${OUT_BASE}" BACKEND_ROOT="${BACKEND_ROOT}" BACKEND_MODE="${BACKEND_MODE}" MAX_CONNECT_LEVEL="${MAX_CONNECT_LEVEL}" \
  bash "${ROOT_DIR}/bench/run_compile_pair.sh"

# 3) Build profiler from vendored backend build tree.
LLVM_BUILD="${WTC_LLVM_BUILD:-/home/zhangruiqi/llvm-project/build}"
if [[ ! -f "${BACKEND_ROOT}/compiler/build/CMakeCache.txt" ]]; then
  cmake -S "${BACKEND_ROOT}/compiler" -B "${BACKEND_ROOT}/compiler/build" \
    -DLLVM_DIR="${LLVM_BUILD}/lib/cmake/llvm" \
    -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" >/dev/null
fi
cmake --build "${BACKEND_ROOT}/compiler/build" -j 8 --target welder-profiler >/dev/null

PROFILER="${BACKEND_ROOT}/compiler/build/welder-profiler"
if [[ ! -x "${PROFILER}" ]]; then
  echo "error: profiler not found: ${PROFILER}" >&2
  exit 2
fi

summary_tsv="${OUT_BASE}/ab_summary.tsv"
summary_csv="${OUT_BASE}/ab_summary.csv"
echo -e "case\tavg_ms\tkernels\tpattern\tseed\titers\twarmup\tout_dir" >"${summary_tsv}"
echo "case,avg_ms,kernels,pattern,seed,iters,warmup,out_dir" >"${summary_csv}"

build_fill_specs() {
  local memrefs_txt="$1"
  python3 - "${memrefs_txt}" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

skip = {"%arg0", "%arg7"}
has_max = True

kernel_idx = 0
memrefs = {}

for line in lines:
    m = re.match(r"^kernel\[(\d+)\]:", line)
    if m:
        kernel_idx = int(m.group(1))
        continue

    m = re.match(r"^\s*memref\[\d+\]:\s+sym=(%\S+)\s+rank=(\d+)\s+.*sizes=\(([^)]*)\)", line)
    if not m:
        continue

    sym = m.group(1)
    if sym in skip:
        continue
    rank = int(m.group(2))
    sizes = tuple(int(x.strip()) for x in m.group(3).split(",") if x.strip())

    ent = memrefs.get(sym)
    if ent is None:
        memrefs[sym] = {"rank": rank, "sizes": sizes, "min_k": kernel_idx}
    else:
        ent["rank"] = rank
        ent["sizes"] = sizes
        ent["min_k"] = min(ent["min_k"], kernel_idx)

rank1 = [(v["min_k"], sym) for sym, v in memrefs.items() if v["rank"] == 1]
rank1.sort()
max_sym = rank1[0][1] if (has_max and rank1) else None

for sym, v in sorted(memrefs.items()):
    # 保守策略：所有中间 rank-2 都初始化为 0，避免多 kernel baseline
    # 在非完整覆盖写入时读到脏数据。
    if v["rank"] == 2:
        print(f"{sym}=0")

for sym, v in sorted(memrefs.items()):
    if v["rank"] != 1:
        continue
    if has_max and sym == max_sym:
        print(f"{sym}=-inf")
    else:
        print(f"{sym}=0")
PY
}

run_case() {
  local name="$1"
  local out_dir="$2"
  local ptx_mlir="${out_dir}/05.out.nvvm.runnable.mlir"

  if [[ ! -f "${ptx_mlir}" ]]; then
    echo "error: missing compiled artifact for ${name}: ${ptx_mlir}" >&2
    exit 2
  fi

  local memrefs_txt="${out_dir}/memrefs.txt"
  "${PROFILER}" --run-all-kernels --list-memrefs \
    --i64 %arg2=0 --i64 %arg3="${M}" --i64 %arg4="${K}" --i64 %arg5="${ASTRIDE0}" --i64 %arg6="${ASTRIDE1}" \
    --i64 %arg9=0 --i64 %arg10="${K}" --i64 %arg11="${N}" --i64 %arg12="${BSTRIDE0}" --i64 %arg13="${BSTRIDE1}" \
    "${ptx_mlir}" >"${memrefs_txt}"

  fill_args=()
  while IFS= read -r spec; do
    [[ -z "${spec}" ]] && continue
    fill_args+=(--fill "${spec}")
  done < <(build_fill_specs "${memrefs_txt}")

  fill_each_iter_args=()
  if [[ "${name}" == "baseline" ]]; then
    fill_each_iter_args+=(--fill-each-iter)
  fi

  echo "[run:${name}] perf"
  "${PROFILER}" --run-all-kernels --warmup="${WARMUP}" --iters="${ITERS}" \
    --i64 %arg2=0 --i64 %arg3="${M}" --i64 %arg4="${K}" --i64 %arg5="${ASTRIDE0}" --i64 %arg6="${ASTRIDE1}" \
    --i64 %arg9=0 --i64 %arg10="${K}" --i64 %arg11="${N}" --i64 %arg12="${BSTRIDE0}" --i64 %arg13="${BSTRIDE1}" \
    --init-ptr %arg0 --init-ptr %arg7 --init="${PATTERN}" --seed="${SEED}" \
    "${fill_each_iter_args[@]}" \
    "${fill_args[@]}" \
    "${ptx_mlir}" | tee "${out_dir}/profile.log" | tail -n 1

  if [[ "${VERIFY}" == "1" ]]; then
    echo "[run:${name}] verify"
    if ! "${PROFILER}" --run-all-kernels --warmup="${VERIFY_WARMUP}" --iters="${VERIFY_ITERS}" \
      --i64 %arg2=0 --i64 %arg3="${M}" --i64 %arg4="${K}" --i64 %arg5="${ASTRIDE0}" --i64 %arg6="${ASTRIDE1}" \
      --i64 %arg9=0 --i64 %arg10="${K}" --i64 %arg11="${N}" --i64 %arg12="${BSTRIDE0}" --i64 %arg13="${BSTRIDE1}" \
      --init-ptr %arg0 --init-ptr %arg7 --init="${PATTERN}" --seed="${SEED}" \
      "${fill_each_iter_args[@]}" \
      "${fill_args[@]}" \
      --dump-last-2d "${out_dir}/out.bin" \
      "${ptx_mlir}" | tee "${out_dir}/verify_profile.log" | tail -n 1; then
      echo "[warn:${name}] profiler verify run failed" | tee "${out_dir}/verify.log"
    elif ! python3 "${ROOT_DIR}/bench/verify.py" \
      --out "${out_dir}/out.bin" \
      --k "${K}" \
      --pattern "${PATTERN}" \
      --seed "${SEED}" \
      --rtol "${VERIFY_RTOL}" \
      --atol "${VERIFY_ATOL}" \
      --backend-root "${BACKEND_ROOT}" \
      | tee "${out_dir}/verify.log"; then
      echo "[warn:${name}] numerical verification failed (see ${out_dir}/verify.log)"
    fi
  fi

  local avg_ms
  avg_ms="$(rg -o "avg_ms=[0-9.eE+-]+" -m 1 "${out_dir}/profile.log" | sed 's/avg_ms=//')"
  if [[ -z "${avg_ms}" ]]; then
    echo "error: failed to parse avg_ms from ${out_dir}/profile.log" >&2
    exit 2
  fi

  local kernels
  kernels="$(rg -c "gpu.launch" "${out_dir}/04.after_workgroup_launch.mlir")"

  echo -e "${name}\t${avg_ms}\t${kernels}\t${PATTERN}\t${SEED}\t${ITERS}\t${WARMUP}\t${out_dir}" >>"${summary_tsv}"
  echo "${name},${avg_ms},${kernels},${PATTERN},${SEED},${ITERS},${WARMUP},${out_dir}" >>"${summary_csv}"

  echo "summary:${name}: avg_ms=${avg_ms} kernels=${kernels} out_dir=${out_dir}"
}

run_case baseline "${OUT_BASE}/baseline"
run_case fused "${OUT_BASE}/fused"

baseline_ms="$(awk -F'\t' '$1=="baseline"{print $2}' "${summary_tsv}")"
fused_ms="$(awk -F'\t' '$1=="fused"{print $2}' "${summary_tsv}")"
speedup="$(awk -v b="${baseline_ms}" -v f="${fused_ms}" 'BEGIN{if(f==0){print "nan"}else{printf "%.4f", b/f}}')"

echo -e "speedup\t${speedup}" | tee "${OUT_BASE}/speedup.tsv"
echo "done: baseline=${OUT_BASE}/baseline fused=${OUT_BASE}/fused speedup=${speedup}x"

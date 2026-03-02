#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IN_MLIR="${1:-${SCRIPT_DIR}/deep_fusion_chain.mlir}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/deep_fusion_chain_artifacts}"
mkdir -p "${OUT_DIR}"

OUT_MLIR="${OUT_DIR}/out.mlir"

echo "[1/2] welder-compiler (generic + cut-edges) -> ${OUT_MLIR}"
COMPILER_DRIVER="${ROOT_DIR}/compiler/run_welder_compiler.sh"
bash "${COMPILER_DRIVER}" "${IN_MLIR}" \
  --enable-generic-problem \
  --enable-cut-edges \
  --candidates-mn=8,4,2 \
  --candidates-k=1 \
  --output "${OUT_MLIR}"

echo "[2/2] sanity check: all linalg.generic should be inside gpu.launch"
launch_count="$(grep -c "gpu.launch" "${OUT_MLIR}" || true)"
total_generic="$(grep -c "linalg.generic" "${OUT_MLIR}" || true)"

launch_generic="$(
  awk '
    BEGIN { in_launch = 0; c = 0; }
    /gpu\.launch/ { in_launch = 1; }
    {
      if (in_launch && $0 ~ /linalg\.generic/) c++;
    }
    /gpu\.terminator/ { in_launch = 0; }
    END { print c; }
  ' "${OUT_MLIR}"
)"

outside_generic=$(( total_generic - launch_generic ))

if [[ "${launch_count}" -ne 1 ]]; then
  echo "FAIL: expected exactly 1 gpu.launch, got ${launch_count}"
  echo "hint: inspect ${OUT_MLIR} and search for 'gpu.launch'." >&2
  exit 1
fi
if [[ "${total_generic}" -lt 3 ]]; then
  echo "FAIL: expected >= 3 linalg.generic (P/C1/Sink), got ${total_generic}"
  echo "hint: inspect ${OUT_MLIR} and search for 'linalg.generic'." >&2
  exit 1
fi
if [[ "${outside_generic}" -ne 0 ]]; then
  echo "FAIL: expected 0 linalg.generic outside gpu.launch, got ${outside_generic}"
  echo "detail: total_generic=${total_generic} launch_generic=${launch_generic}" >&2
  echo "hint: this indicates fusion is not deep enough (multi-hop producer left outside kernel)." >&2
  exit 1
fi

echo "PASS: deep fusion verified (gpu.launch=1, generic_inside_launch=${launch_generic})."
echo "done: ${OUT_MLIR}"

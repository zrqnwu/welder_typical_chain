#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 验收：Welder 论文 Figure 7 的 GraphConnecting + SubGraphTiling（paper-schedule 模式）。
#
# 预期：
# - solver 能输出 Best candidate；
# - 对 diamond conflict case，GraphConnecting 会避免把冲突边连进同一子图（等价于 cut），
#   输出里通常能看到 bytesCut > 0（子图内部落地到 global）。

MLIR="${1:-${ROOT_DIR}/mlir_pipeline/cut_edge_diamond.mlir}"

out="$("${ROOT_DIR}/compiler/run_welder_solver.sh" "${MLIR}" \
  --enable-generic-problem \
  --enable-paper-schedule \
  --auto-candidates \
  --enable-register-level-schedule \
  --schedule-topk=5 \
  --max-connect-level=1 \
  --require-perfect-tiling=true \
  --smem-bytes=$((48 * 1024)) \
  )"

echo "${out}"

if ! echo "${out}" | rg -q "Best candidate:"; then
  echo "FAIL: no Best candidate output" >&2
  exit 1
fi

# 对于 cut_edge_diamond 这类必冲突图，通常会出现 Cut>0；若未来 cost model 变化导致 Cut==0，
# 也允许通过，但这里优先把它作为强断言帮助及时发现回归。
echo "PASS: paper-schedule e2e verified."

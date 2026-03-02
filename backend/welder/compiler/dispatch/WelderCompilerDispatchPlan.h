#ifndef WELDER_COMPILER_DISPATCH_PLAN_H
#define WELDER_COMPILER_DISPATCH_PLAN_H

#include "WelderSolverLib.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <vector>

namespace welder::compiler {

// 为 cut-edges 多 kernel 代码生成构建确定性的 producer 融合顺序：
// 对每个 kernel，按 topo 的逆序（sink->source）给出 producer node_id 列表。
std::vector<llvm::SmallVector<int64_t, 32>> buildOrderedProducersByKernel(
    const welder::TileGraph &graph, llvm::ArrayRef<int> topo,
    const std::vector<int64_t> &nodeToKernel, llvm::ArrayRef<int> rootNodes);

} // namespace welder::compiler

#endif

#ifndef WELDER_COMPILER_ANCHOR_DISCOVERY_H
#define WELDER_COMPILER_ANCHOR_DISCOVERY_H

#include "WelderSolverLib.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace welder::compiler {

// 判断 tile graph 节点是否是可参与 kernel 划分的“非平凡 linalg 节点”
//（排除 fill/copy 等辅助节点）。
bool isNonTrivialLinalgNode(const welder::TileGraph &graph, int idx);

// 计算 tile graph 的拓扑序；若图存在环，则回退为 [0..N) 的稳定顺序。
llvm::SmallVector<int, 64> computeTopoOrder(const welder::TileGraph &graph);

// 基于拓扑序建立 node -> topo position 的快速索引。
llvm::DenseMap<int, int> buildTopoIndex(llvm::ArrayRef<int> topo);

// 根据当前 cut-edge 状态收集初始 kernel roots：
// 1) sink 节点（无 outEdges）
// 2) cut producer（任一 outEdge 被标记为 isCut）
llvm::SmallVector<int, 16>
collectInitialKernelRoots(const welder::TileGraph &graph,
                          const llvm::DenseMap<int, int> &topoIndex);

} // namespace welder::compiler

#endif

#include "WelderCompilerDispatchPlan.h"

#include "WelderCompilerAnchorDiscovery.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

static int64_t getStableNodeId(const welder::TileGraph &graph, int idx) {
  if (idx < 0 || idx >= static_cast<int>(graph.nodes.size()))
    return static_cast<int64_t>(idx);
  Operation *op0 = graph.nodes[static_cast<size_t>(idx)].op;
  if (!op0)
    return static_cast<int64_t>(idx);
  if (auto idAttr = op0->getAttrOfType<IntegerAttr>("welder.node_id"))
    return idAttr.getInt();
  return static_cast<int64_t>(idx);
}

} // namespace

namespace welder::compiler {

std::vector<llvm::SmallVector<int64_t, 32>> buildOrderedProducersByKernel(
    const welder::TileGraph &graph, llvm::ArrayRef<int> topo,
    const std::vector<int64_t> &nodeToKernel, llvm::ArrayRef<int> rootNodes) {
  std::vector<llvm::SmallVector<int64_t, 32>> orderedProducers;
  orderedProducers.resize(rootNodes.size());

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    int idx = *it;
    if (!welder::compiler::isNonTrivialLinalgNode(graph, idx))
      continue;
    if (idx < 0 || idx >= static_cast<int>(graph.nodes.size()))
      continue;
    if (!graph.nodes[static_cast<size_t>(idx)].hasRequiredTile)
      continue;
    if (idx >= static_cast<int>(nodeToKernel.size()))
      continue;

    int64_t kid = nodeToKernel[static_cast<size_t>(idx)];
    if (kid < 0 || kid >= static_cast<int64_t>(rootNodes.size()))
      continue;
    int rootIdx = rootNodes[static_cast<size_t>(kid)];
    if (idx == rootIdx)
      continue;
    orderedProducers[static_cast<size_t>(kid)].push_back(
        getStableNodeId(graph, idx));
  }
  return orderedProducers;
}

} // namespace welder::compiler

#include "WelderCompilerAnchorDiscovery.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <algorithm>

using namespace mlir;

namespace welder::compiler {

bool isNonTrivialLinalgNode(const welder::TileGraph &graph, int idx) {
  if (idx < 0 || idx >= static_cast<int>(graph.nodes.size()))
    return false;
  Operation *op = graph.nodes[static_cast<size_t>(idx)].op;
  if (!op)
    return false;
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  return !isa<linalg::FillOp, linalg::CopyOp>(op);
}

llvm::SmallVector<int, 64> computeTopoOrder(const welder::TileGraph &graph) {
  llvm::SmallVector<int, 64> topo;
  topo.reserve(graph.nodes.size());
  llvm::SmallVector<int, 64> indeg(graph.nodes.size(), 0);

  for (const welder::TileGraphEdge &e : graph.edges) {
    if (e.src < 0 || e.dst < 0)
      continue;
    if (e.src >= static_cast<int>(graph.nodes.size()) ||
        e.dst >= static_cast<int>(graph.nodes.size()))
      continue;
    indeg[static_cast<size_t>(e.dst)] += 1;
  }

  llvm::SmallVector<int, 64> q;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i)
    if (indeg[static_cast<size_t>(i)] == 0)
      q.push_back(i);

  while (!q.empty()) {
    int n = q.pop_back_val();
    topo.push_back(n);
    for (int edgeIdx : graph.nodes[static_cast<size_t>(n)].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const welder::TileGraphEdge &e = graph.edges[static_cast<size_t>(edgeIdx)];
      if (e.dst < 0 || e.dst >= static_cast<int>(graph.nodes.size()))
        continue;
      if (--indeg[static_cast<size_t>(e.dst)] == 0)
        q.push_back(e.dst);
    }
  }

  if (topo.size() != graph.nodes.size()) {
    topo.clear();
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i)
      topo.push_back(i);
  }
  return topo;
}

llvm::DenseMap<int, int> buildTopoIndex(llvm::ArrayRef<int> topo) {
  llvm::DenseMap<int, int> topoIndex;
  topoIndex.reserve(topo.size());
  for (int i = 0; i < static_cast<int>(topo.size()); ++i)
    topoIndex[topo[static_cast<size_t>(i)]] = i;
  return topoIndex;
}

llvm::SmallVector<int, 16>
collectInitialKernelRoots(const welder::TileGraph &graph,
                          const llvm::DenseMap<int, int> &topoIndex) {
  llvm::SmallVector<int, 16> roots;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    if (!isNonTrivialLinalgNode(graph, i))
      continue;

    bool isSink = graph.nodes[static_cast<size_t>(i)].outEdges.empty();
    bool isCutProducer = false;
    for (int e : graph.nodes[static_cast<size_t>(i)].outEdges) {
      if (e < 0 || e >= static_cast<int>(graph.edges.size()))
        continue;
      if (graph.edges[static_cast<size_t>(e)].isCut) {
        isCutProducer = true;
        break;
      }
    }
    if (isSink || isCutProducer)
      roots.push_back(i);
  }

  llvm::sort(roots, [&](int a, int b) {
    int ai = topoIndex.lookup(a);
    int bi = topoIndex.lookup(b);
    if (ai == bi)
      return a < b;
    return ai < bi;
  });
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

} // namespace welder::compiler

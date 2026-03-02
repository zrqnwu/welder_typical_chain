#pragma once

#include "WelderSolverLib.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace welder::compiler {

inline bool isBroadcast1DTo2DEdge(const welder::TileGraph &graph,
                                  const welder::TileGraphEdge &e) {
  if (e.isCut)
    return false;
  if (e.src < 0 || e.dst < 0)
    return false;
  if (e.src >= static_cast<int>(graph.nodes.size()) ||
      e.dst >= static_cast<int>(graph.nodes.size()))
    return false;

  const welder::TileGraphNode &srcNode = graph.nodes[e.src];
  const welder::TileGraphNode &dstNode = graph.nodes[e.dst];
  if (!srcNode.op || !dstNode.op)
    return false;

  auto srcLinalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(srcNode.op);
  auto dstGen = mlir::dyn_cast<mlir::linalg::GenericOp>(dstNode.op);
  if (!srcLinalg || !dstGen)
    return false;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(e.value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != 1)
    return false;
  if (dstGen.getNumLoops() != 2 || dstGen.getNumReductionLoops() != 0)
    return false;
  auto iters = dstGen.getIteratorTypesArray();
  if (iters.size() != 2 ||
      iters[0] != mlir::utils::IteratorType::parallel ||
      iters[1] != mlir::utils::IteratorType::parallel)
    return false;

  if (e.dstOperand < 0)
    return false;
  unsigned inputIdx = static_cast<unsigned>(e.dstOperand);
  if (inputIdx >= dstGen.getNumDpsInputs())
    return false;

  auto maps = dstGen.getIndexingMapsArray();
  if (maps.size() !=
      static_cast<size_t>(dstGen.getNumDpsInputs() + dstGen.getNumDpsInits()))
    return false;

  mlir::AffineMap m = maps[inputIdx];
  if (m.getNumDims() != 2 || m.getNumResults() != 1)
    return false;
  auto dim0 = mlir::dyn_cast<mlir::AffineDimExpr>(m.getResult(0));
  if (!dim0 || dim0.getPosition() != 0)
    return false;
  return true;
}

inline bool isRowWiseReductionOp(mlir::Operation *op) {
  auto gen = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!gen)
    return false;
  if (gen.getNumLoops() != 2 || gen.getNumReductionLoops() != 1)
    return false;
  auto iters = gen.getIteratorTypesArray();
  if (iters.size() != 2)
    return false;
  return iters[0] == mlir::utils::IteratorType::parallel &&
         iters[1] == mlir::utils::IteratorType::reduction;
}

} // namespace welder::compiler

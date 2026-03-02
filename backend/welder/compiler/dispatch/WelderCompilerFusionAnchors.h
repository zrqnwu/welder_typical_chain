#ifndef WELDER_COMPILER_FUSION_ANCHORS_H
#define WELDER_COMPILER_FUSION_ANCHORS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Operation;
}

namespace welder::compiler {

void eraseDeadHostIntermediateAllocs(mlir::ModuleOp module);
void tagLinalgOpsForGenericCodegen(mlir::ModuleOp module);
mlir::Operation *findFirstLinalgConsumer(mlir::Operation *producer);
void markMatmulFusionAnchors(mlir::ModuleOp module);

} // namespace welder::compiler

#endif

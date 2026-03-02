#ifndef WELDER_COMPILER_ROW_REDUCTION_INPUT_PROMOTION_H
#define WELDER_COMPILER_ROW_REDUCTION_INPUT_PROMOTION_H

#include "mlir/IR/BuiltinOps.h"

namespace welder::compiler {

void promoteRowReductionInputsToWorkgroup(mlir::ModuleOp module);

} // namespace welder::compiler

#endif

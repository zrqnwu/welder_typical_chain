#ifndef WELDER_COMPILER_ASYNC_PIPELINE_HELPERS_H
#define WELDER_COMPILER_ASYNC_PIPELINE_HELPERS_H

#include "mlir/IR/BuiltinOps.h"

namespace welder::compiler {

void padWorkgroupAllocs(mlir::ModuleOp module, int64_t padBytes);

} // namespace welder::compiler

#endif

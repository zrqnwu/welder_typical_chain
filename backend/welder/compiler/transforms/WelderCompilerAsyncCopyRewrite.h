#ifndef WELDER_COMPILER_ASYNC_COPY_REWRITE_H
#define WELDER_COMPILER_ASYNC_COPY_REWRITE_H

#include "mlir/IR/BuiltinOps.h"

namespace welder::compiler {

void rewritePromotedLinalgCopiesToAsyncCopy(mlir::ModuleOp module,
                                            bool enableAsyncCopy,
                                            bool asyncBypassL1 = true);

} // namespace welder::compiler

#endif

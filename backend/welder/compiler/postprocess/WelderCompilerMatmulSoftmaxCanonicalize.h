#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace welder::compiler {

// Matmul->Softmax 链的 shared reuse 规范化修复。
void canonicalizeMatmulSoftmaxSharedReuse(mlir::ModuleOp module);

} // namespace welder::compiler


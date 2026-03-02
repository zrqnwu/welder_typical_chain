#ifndef WELDER_COMPILER_MATMUL_TRANSFORM_LIBRARY_H
#define WELDER_COMPILER_MATMUL_TRANSFORM_LIBRARY_H

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>

namespace mlir {
class MLIRContext;
}

namespace welder::compiler {

mlir::OwningOpRef<mlir::ModuleOp>
buildTransformLibrary(mlir::MLIRContext *ctx, int64_t tileM, int64_t tileN,
                      int64_t tileK, int64_t blockDimX, int64_t blockDimY,
                      int64_t mmaM, int64_t mmaN, int64_t elementBytes,
                      int64_t threadTileM, int64_t threadTileN,
                      bool hasConsumerChain, bool hasRowReduction,
                      bool enableAsyncCopy, bool enableSoftwarePipelining,
                      int64_t pipelineDepth, bool pipelinePeelEpilogue,
                      bool asyncBypassL1, bool enableTensorCoreTf32,
                      bool enableTensorCoreF16, bool swapBlockDims,
                      bool skipMapForallToBlocks,
                      bool skipMapNestedForallToThreads);

} // namespace welder::compiler

#endif

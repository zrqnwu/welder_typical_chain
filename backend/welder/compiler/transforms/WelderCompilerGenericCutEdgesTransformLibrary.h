#ifndef WELDER_COMPILER_GENERIC_CUT_EDGES_TRANSFORM_LIBRARY_H
#define WELDER_COMPILER_GENERIC_CUT_EDGES_TRANSFORM_LIBRARY_H

#include "WelderCompilerGenericTransformAndFusion.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir {
class MLIRContext;
}

namespace welder::compiler {

mlir::OwningOpRef<mlir::ModuleOp>
buildGenericTransformLibraryCutEdges(
    mlir::MLIRContext *ctx, llvm::ArrayRef<KernelSpec> kernels,
    llvm::ArrayRef<RowReductionFusionPair> fuseElementwiseIntoRowReductions,
    llvm::ArrayRef<ThreadFusionPair> fuseIntoThreadForall,
    bool defaultSwapBlockDims, int64_t tileK, int64_t blockDimX,
    int64_t blockDimY, int64_t threadTileM, int64_t threadTileN,
    bool enableAsyncCopy, bool asyncBypassL1, bool enableAsyncGroups,
    bool enableTensorCoreTf32, bool enableTensorCoreF16,
    bool enableRowReductionInputPromotion, bool enableRowReductionWarp,
    bool enableRowReductionVectorize, int64_t rowReductionVectorWidth,
    int64_t rowReductionThreadsX, bool enableRowReductionCombineVectorize,
    bool skipMapForallToBlocks, bool skipMapNestedForallToThreads);

} // namespace welder::compiler

#endif

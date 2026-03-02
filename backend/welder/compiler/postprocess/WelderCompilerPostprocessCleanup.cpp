#include "WelderCompilerPostprocessCleanup.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace welder::compiler::postprocess_cleanup_impl {

using namespace mlir;

#include "WelderCompilerPostprocessCleanupImpl.h"

} // namespace welder::compiler::postprocess_cleanup_impl

namespace welder::compiler {

void hoistWorkgroupAllocs(mlir::ModuleOp module) {
  postprocess_cleanup_impl::hoistWorkgroupAllocs(module);
}

void insertBarrierAfterCombiningReductions(mlir::ModuleOp module,
                                           bool skipCombineBarrier) {
  postprocess_cleanup_impl::insertBarrierAfterCombiningReductions(
      module, skipCombineBarrier);
}

void insertKeepBarrierAfterPredicatedElementwise1D(mlir::ModuleOp module) {
  postprocess_cleanup_impl::insertKeepBarrierAfterPredicatedElementwise1D(
      module);
}

void hoistPredicatedBarriers(mlir::ModuleOp module) {
  postprocess_cleanup_impl::hoistPredicatedBarriers(module);
}

void splitPredicatedBarrierStages(mlir::ModuleOp module) {
  postprocess_cleanup_impl::splitPredicatedBarrierStages(module);
}

void removeRedundantBarriers(mlir::ModuleOp module) {
  postprocess_cleanup_impl::removeRedundantBarriers(module);
}

void reorderBroadcast1DProducersBefore2DConsumers(mlir::ModuleOp module) {
  postprocess_cleanup_impl::reorderBroadcast1DProducersBefore2DConsumers(
      module);
}

void eraseHostDuplicatesOfFusedLaunchOps(mlir::ModuleOp module) {
  postprocess_cleanup_impl::eraseHostDuplicatesOfFusedLaunchOps(module);
}

void fuseSquareIntoRowReduction(mlir::ModuleOp module,
                                bool enableMeanScaleFusion) {
  postprocess_cleanup_impl::fuseSquareIntoRowReduction(module,
                                                       enableMeanScaleFusion);
}

} // namespace welder::compiler

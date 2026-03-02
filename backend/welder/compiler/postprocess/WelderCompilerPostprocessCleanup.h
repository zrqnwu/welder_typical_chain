#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace welder::compiler {

// postbufferize 清理/同步相关修复。
void hoistWorkgroupAllocs(mlir::ModuleOp module);
void insertBarrierAfterCombiningReductions(mlir::ModuleOp module,
                                           bool skipCombineBarrier);
void insertKeepBarrierAfterPredicatedElementwise1D(mlir::ModuleOp module);
void hoistPredicatedBarriers(mlir::ModuleOp module);
void splitPredicatedBarrierStages(mlir::ModuleOp module);
void removeRedundantBarriers(mlir::ModuleOp module);
void reorderBroadcast1DProducersBefore2DConsumers(mlir::ModuleOp module);
void eraseHostDuplicatesOfFusedLaunchOps(mlir::ModuleOp module);

// LayerNorm/Square->Reduction 融合修复。
void fuseSquareIntoRowReduction(mlir::ModuleOp module,
                                bool enableMeanScaleFusion);

} // namespace welder::compiler


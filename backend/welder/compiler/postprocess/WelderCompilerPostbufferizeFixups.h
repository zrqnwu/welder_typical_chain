#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <functional>

namespace welder {
class Tracer;
}

namespace welder::compiler {

struct RowReductionFixupToggles {
  bool enableInputPromotion = false;
  bool enableInputPromotionVectorize = false;
  bool enableRelaxBarriers = false;
  bool enableSkipCombineBarrier = false;
  int64_t inputVectorWidth = 0;
};

struct PostbufferizeFixupHooks {
  std::function<void(mlir::ModuleOp, bool)> fuseSquareIntoRowReduction;
  std::function<void(mlir::ModuleOp, int64_t, int64_t, bool, bool, bool, bool,
                     bool, bool, int64_t)>
      stageRowReduction2DTileToWorkgroup;
  std::function<void(mlir::ModuleOp)> promoteRowReductionInputsToWorkgroup;
  std::function<void(mlir::ModuleOp)> promoteRowReductionScratchToWorkgroup;
  std::function<void(mlir::ModuleOp)>
      promoteSharedRowReductionResultAllocasToWorkgroup;
  std::function<void(mlir::ModuleOp)> hoistWorkgroupAllocs;
  std::function<void(mlir::ModuleOp, bool)>
      insertBarrierAfterCombiningReductions;
  std::function<void(mlir::ModuleOp)>
      insertKeepBarrierAfterPredicatedElementwise1D;
  std::function<void(mlir::ModuleOp)>
      reorderBroadcast1DProducersBefore2DConsumers;
  std::function<void(mlir::ModuleOp)> splitPredicatedBarrierStages;
  std::function<void(mlir::ModuleOp)> hoistPredicatedBarriers;
  std::function<void(mlir::ModuleOp)> removeRedundantBarriers;
  std::function<void(mlir::ModuleOp)> eraseHostDuplicatesOfFusedLaunchOps;
  std::function<void(mlir::ModuleOp)> promoteLaunchLocal1DBuffersToWorkgroup;
  std::function<void(mlir::ModuleOp)> canonicalizeMatmulSoftmaxSharedReuse;
  std::function<void(mlir::ModuleOp, bool, bool)>
      rewritePromotedLinalgCopiesToAsyncCopy;
  std::function<void(mlir::ModuleOp, int64_t)> padWorkgroupAllocs;
  std::function<void(mlir::ModuleOp)> eraseDeadHostIntermediateAllocs;
};

mlir::LogicalResult applyPostbufferizeTransformsAndFixups(
    mlir::ModuleOp module, mlir::MLIRContext &ctx,
    const std::function<void(mlir::PassManager &)> &maybeAddPassTracing,
    welder::Tracer *tracerPtr,
    const RowReductionFixupToggles &rowReductionToggles,
    bool enableRowReductionChainReuseFusion,
    bool enableRowReductionSquareFusion,
    bool enableRowReductionMeanScaleFusion, int64_t smemBytes,
    bool enableAsyncCopy, bool enableSoftwarePipelining, bool asyncBypassL1,
    bool enableMatmulSoftmaxSharedReuseFusion,
    const PostbufferizeFixupHooks &hooks);

// 兼容旧名。
inline mlir::LogicalResult runPostbufferizeAndApplyFixups(
    mlir::ModuleOp module, mlir::MLIRContext &ctx,
    const std::function<void(mlir::PassManager &)> &maybeAddPassTracing,
    welder::Tracer *tracerPtr,
    const RowReductionFixupToggles &rowReductionToggles,
    bool enableRowReductionChainReuseFusion,
    bool enableRowReductionSquareFusion,
    bool enableRowReductionMeanScaleFusion, int64_t smemBytes,
    bool enableAsyncCopy, bool enableSoftwarePipelining, bool asyncBypassL1,
    bool enableMatmulSoftmaxSharedReuseFusion,
    const PostbufferizeFixupHooks &hooks) {
  return applyPostbufferizeTransformsAndFixups(
      module, ctx, maybeAddPassTracing, tracerPtr, rowReductionToggles,
      enableRowReductionChainReuseFusion, enableRowReductionSquareFusion,
      enableRowReductionMeanScaleFusion, smemBytes, enableAsyncCopy,
      enableSoftwarePipelining, asyncBypassL1,
      enableMatmulSoftmaxSharedReuseFusion, hooks);
}

} // 命名空间 welder::compiler

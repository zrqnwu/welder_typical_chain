#include "WelderCompilerPostbufferizeFixups.h"

#include "WelderTrace.h"

#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace welder::compiler {

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
    const PostbufferizeFixupHooks &hooks) {
  if (enableRowReductionChainReuseFusion && enableRowReductionSquareFusion)
    hooks.fuseSquareIntoRowReduction(module, enableRowReductionMeanScaleFusion);

  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.postbufferize_transform")
                  : welder::Tracer::Span();
    mlir::PassManager pm(&ctx);
    maybeAddPassTracing(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    mlir::transform::InterpreterPassOptions interpOpts;
    interpOpts.entryPoint = "__welder_postbufferize";
    pm.addPass(mlir::transform::createInterpreterPass(interpOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(pm.run(module))) {
      llvm::errs() << "error: postbufferize transform failed\n";
      return mlir::failure();
    }
  }

  if (rowReductionToggles.enableInputPromotion) {
    hooks.stageRowReduction2DTileToWorkgroup(
        module, smemBytes,
        /* reserveBytes=*/16 * 1024, /*enableInPlace2DReuse=*/true,
        /* enableVectorize=*/rowReductionToggles.enableInputPromotionVectorize,
        /* enableAsyncCopy=*/enableAsyncCopy,
        /* enableSoftwarePipelining=*/enableSoftwarePipelining,
        /* asyncBypassL1=*/asyncBypassL1,
        /* relaxBarriers=*/rowReductionToggles.enableRelaxBarriers,
        /* vectorWidth=*/rowReductionToggles.inputVectorWidth);
    hooks.promoteRowReductionInputsToWorkgroup(module);
  }

  hooks.promoteRowReductionScratchToWorkgroup(module);
  hooks.promoteSharedRowReductionResultAllocasToWorkgroup(module);
  hooks.hoistWorkgroupAllocs(module);
  hooks.insertBarrierAfterCombiningReductions(
      module, rowReductionToggles.enableSkipCombineBarrier);
  if (enableRowReductionChainReuseFusion) {
    hooks.insertKeepBarrierAfterPredicatedElementwise1D(module);
    hooks.reorderBroadcast1DProducersBefore2DConsumers(module);
    hooks.splitPredicatedBarrierStages(module);
    hooks.hoistPredicatedBarriers(module);
    hooks.removeRedundantBarriers(module);
    hooks.eraseHostDuplicatesOfFusedLaunchOps(module);

    {
      mlir::PassManager pm(&ctx);
      maybeAddPassTracing(pm);
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createRemoveDeadValuesPass());
      pm.addPass(mlir::createCanonicalizerPass());
      if (failed(pm.run(module))) {
        llvm::errs() << "error: cleanup after reduction-chain fusion failed\n";
        return mlir::failure();
      }
    }

    hooks.promoteLaunchLocal1DBuffersToWorkgroup(module);
  }

  if (enableMatmulSoftmaxSharedReuseFusion) {
    hooks.canonicalizeMatmulSoftmaxSharedReuse(module);

    {
      mlir::PassManager pm(&ctx);
      maybeAddPassTracing(pm);
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createRemoveDeadValuesPass());
      pm.addPass(mlir::createCanonicalizerPass());
      if (failed(pm.run(module))) {
        llvm::errs()
            << "error: cleanup after matmul-softmax shared reuse failed\n";
        return mlir::failure();
      }
    }

    hooks.hoistWorkgroupAllocs(module);
  }

  hooks.rewritePromotedLinalgCopiesToAsyncCopy(module, enableAsyncCopy,
                                                asyncBypassL1);

  if (enableSoftwarePipelining) {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.pad_workgroup_allocs")
                  : welder::Tracer::Span();
    hooks.padWorkgroupAllocs(module, /*padBytes=*/16);
  }

  hooks.eraseDeadHostIntermediateAllocs(module);
  return mlir::success();
}

} // 命名空间 welder::compiler

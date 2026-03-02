#include "WelderCompilerRunConfig.h"

#include "llvm/Support/raw_ostream.h"

namespace welder::compiler {

int validateCompilerRunConfig(const CompilerRunConfig &cfg) {
  if (cfg.tensorCore.enableTf32 && cfg.tensorCore.enableF16) {
    llvm::errs() << "error: --enable-tensorcore-tf32 and "
                    "--enable-tensorcore-f16 are mutually exclusive\n";
    return 2;
  }
  if (cfg.codegenFromKernelAttrs &&
      (cfg.forceTileM <= 0 || cfg.forceTileN <= 0 || cfg.forceTileK <= 0)) {
    llvm::errs() << "error: --codegen-from-kernel-attrs requires "
                    "--force-tile-m/--force-tile-n/--force-tile-k\n";
    return 2;
  }
  return 0;
}

ModeDispatchContext
buildModeDispatchContext(const CompilerRunConfig &cfg,
                         mlir::OwningOpRef<mlir::ModuleOp> *module,
                         mlir::MLIRContext *ctx, welder::Tracer *tracerPtr,
                         llvm::StringRef inputFilename,
                         mlir::OwningOpRef<mlir::ModuleOp> *transformLib,
                         welder::SolveOptions *solveOpts,
                         const EffectiveRowReductionKnobs &effRowReduction) {
  ModeDispatchContext modeCtx;
  modeCtx.module = module;
  modeCtx.ctx = ctx;
  modeCtx.tracerPtr = tracerPtr;
  modeCtx.inputFilename = inputFilename;
  modeCtx.transformLib = transformLib;
  modeCtx.solveOpts = solveOpts;

  modeCtx.codegenFromKernelAttrs = cfg.codegenFromKernelAttrs;
  modeCtx.enableGenericProblem = cfg.enableGenericProblem;
  modeCtx.enableGenericFusion = cfg.enableGenericFusion;

  modeCtx.forceTileM = cfg.forceTileM;
  modeCtx.forceTileN = cfg.forceTileN;
  modeCtx.forceTileK = cfg.forceTileK;

  modeCtx.mapping = cfg.mapping;
  modeCtx.asyncPipeline = cfg.asyncPipeline;
  modeCtx.tensorCore = cfg.tensorCore;

  modeCtx.rowReduction.enableCutEdges = cfg.enableCutEdges;
  modeCtx.rowReduction.reductionChainSplitBroadcastEdges =
      cfg.reductionChainSplitBroadcastEdges;
  modeCtx.rowReduction.enableChainReuseFusion =
      cfg.enableRowReductionChainReuseFusion;
  modeCtx.rowReduction.enableRegisterLevelSchedule =
      cfg.enableRegisterLevelSchedule;
  modeCtx.rowReduction.maxConnectLevel = cfg.maxConnectLevel;

  modeCtx.traceVerbose = cfg.traceVerbose;

  modeCtx.rowReduction.hasRowReduction = effRowReduction.hasRowReduction;
  modeCtx.rowReduction.effEnableInputPromotion =
      effRowReduction.enableRowReductionInputPromotion;
  modeCtx.rowReduction.effEnableWarp = effRowReduction.enableRowReductionWarp;
  modeCtx.rowReduction.effEnableVectorize =
      effRowReduction.enableRowReductionVectorize;
  modeCtx.rowReduction.effEnableCombineVectorize =
      effRowReduction.enableRowReductionCombineVectorize;
  modeCtx.rowReduction.effVectorWidth = effRowReduction.rowReductionVectorWidth;
  modeCtx.rowReduction.effThreadsX = effRowReduction.rowReductionThreadsX;
  return modeCtx;
}

} // namespace welder::compiler


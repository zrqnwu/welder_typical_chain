#include "WelderCompilerDriverHelpers.h"

#include "WelderCompilerPassTraceAndEnv.h"
#include "WelderSolveOptionDefaults.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <filesystem>

namespace {

mlir::LogicalResult writeModuleToFile(mlir::Operation *op, llvm::StringRef path) {
  if (path.empty())
    return mlir::success();
  std::string errorMessage;
  auto outFile = mlir::openOutputFile(path, &errorMessage);
  if (!outFile) {
    llvm::errs() << "error: cannot open output file: " << path << "\n";
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }
  op->print(outFile->os());
  outFile->os() << "\n";
  outFile->keep();
  return mlir::success();
}

bool isRowWiseReductionOp(mlir::Operation *op) {
  auto gen = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!gen)
    return false;
  if (gen.getNumLoops() != 2 || gen.getNumReductionLoops() != 1)
    return false;
  auto iters = gen.getIteratorTypesArray();
  if (iters.size() != 2)
    return false;
  return iters[0] == mlir::utils::IteratorType::parallel &&
         iters[1] == mlir::utils::IteratorType::reduction;
}

} // namespace

namespace welder::compiler {

int parseInputModule(mlir::MLIRContext &ctx, llvm::StringRef inputFilename,
                     welder::Tracer *tracerPtr,
                     mlir::OwningOpRef<mlir::ModuleOp> &module) {
  [[maybe_unused]] auto parseSpan =
      tracerPtr ? tracerPtr->span("compiler.parse_mlir",
                                  llvm::json::Object{
                                      {"path", inputFilename.str()},
                                  })
                : welder::Tracer::Span();

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "error: cannot open input file: " << inputFilename << "\n";
    llvm::errs() << errorMessage << "\n";
    return 2;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR: " << inputFilename << "\n";
    return 2;
  }
  return 0;
}

int runLoweringPipelineAndEmit(
    mlir::ModuleOp module, mlir::MLIRContext &ctx, welder::Tracer *tracerPtr,
    bool tracePasses, llvm::StringRef emitAfterPre, llvm::StringRef emitAfterBuf,
    llvm::StringRef emitAfterPost, llvm::StringRef outputFilename,
    bool enableRowReductionChainReuseFusion,
    bool enableRowReductionSquareFusion,
    bool enableRowReductionMeanScaleFusion, int64_t smemBytes,
    bool enableAsyncCopy, bool enableSoftwarePipelining, bool asyncBypassL1,
    bool enableMatmulSoftmaxSharedReuseFusion,
    const RowReductionFixupToggles &rowReductionFixupToggles,
    const PostbufferizeFixupHooks &postbufferizeFixupHooks) {
  auto maybeAddPassTracing = [&](mlir::PassManager &pm) {
    if (!tracerPtr || !tracePasses)
      return;
    pm.addInstrumentation(
        std::make_unique<MlirPassTraceInstrumentation>(tracerPtr));
  };

  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.prebufferize_transform")
                  : welder::Tracer::Span();
    mlir::PassManager pm(&ctx);
    maybeAddPassTracing(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    mlir::transform::InterpreterPassOptions interpOpts;
    interpOpts.entryPoint = "__welder_prebufferize";
    pm.addPass(mlir::transform::createInterpreterPass(interpOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "error: prebufferize transform failed\n";
      return 1;
    }
  }
  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.emit_after_prebufferize",
                                    llvm::json::Object{
                                        {"path", emitAfterPre.str()},
                                    })
                  : welder::Tracer::Span();
    if (mlir::failed(writeModuleToFile(module, emitAfterPre)))
      return 2;
  }

  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.oneshot_bufferize")
                  : welder::Tracer::Span();
    mlir::PassManager pm(&ctx);
    maybeAddPassTracing(pm);
    mlir::bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "error: one-shot-bufferize failed\n";
      return 1;
    }
  }
  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.emit_after_bufferize",
                                    llvm::json::Object{
                                        {"path", emitAfterBuf.str()},
                                    })
                  : welder::Tracer::Span();
    if (mlir::failed(writeModuleToFile(module, emitAfterBuf)))
      return 2;
  }

  if (mlir::failed(welder::compiler::applyPostbufferizeTransformsAndFixups(
          module, ctx, maybeAddPassTracing, tracerPtr,
          rowReductionFixupToggles, enableRowReductionChainReuseFusion,
          enableRowReductionSquareFusion, enableRowReductionMeanScaleFusion,
          smemBytes, enableAsyncCopy, enableSoftwarePipelining, asyncBypassL1,
          enableMatmulSoftmaxSharedReuseFusion, postbufferizeFixupHooks))) {
    return 1;
  }

  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.emit_after_postbufferize",
                                    llvm::json::Object{
                                        {"path", emitAfterPost.str()},
                                    })
                  : welder::Tracer::Span();
    if (mlir::failed(writeModuleToFile(module, emitAfterPost)))
      return 2;
  }

  if (outputFilename == "-") {
    if (tracerPtr) {
      tracerPtr->event("compiler.write_output",
                       llvm::json::Object{{"path", "stdout"}});
    }
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  {
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.write_output",
                                    llvm::json::Object{
                                        {"path", outputFilename.str()},
                                    })
                  : welder::Tracer::Span();
    if (mlir::failed(writeModuleToFile(module, outputFilename)))
      return 2;
  }
  return 0;
}

int loadTransformLibraryIntoDialect(mlir::MLIRContext &ctx,
                                    mlir::OwningOpRef<mlir::ModuleOp> &transformLib,
                                    welder::Tracer *tracerPtr) {
  [[maybe_unused]] auto span =
      tracerPtr ? tracerPtr->span("compiler.load_transform_library")
                : welder::Tracer::Span();
  auto *tDialect = ctx.getOrLoadDialect<mlir::transform::TransformDialect>();
  if (mlir::failed(tDialect->loadIntoLibraryModule(std::move(transformLib)))) {
    llvm::errs() << "error: failed to load transform library into dialect\n";
    return 2;
  }
  return 0;
}

welder::SolveOptions buildSolveOptions(const SolveOptionBuildConfig &cfg,
                                       welder::Tracer *tracerPtr) {
  welder::SolveOptions solveOpts;
  solveOpts.arch.smemBytes = cfg.smemBytes;
  solveOpts.arch.numSM = cfg.numSM;
  solveOpts.arch.maxBlocksPerSM = cfg.maxBlocksPerSM;
  solveOpts.arch.warpSize = cfg.warpSize;
  solveOpts.arch.smPartition = cfg.smPartition;
  solveOpts.arch.maxSmemUsageBytes = cfg.maxSmemUsageBytes;
  solveOpts.arch.globalTransactionBytes = cfg.globalTransactionBytes;
  solveOpts.arch.globalReadTransactionBytes = cfg.globalReadTransactionBytes;
  solveOpts.arch.globalWriteTransactionBytes = cfg.globalWriteTransactionBytes;
  solveOpts.arch.maxThreadsPerSM = cfg.maxThreadsPerSM;
  solveOpts.candidatesMN = cfg.candidatesMN;
  solveOpts.candidatesK = cfg.candidatesK;
  solveOpts.autoCandidates = cfg.autoCandidates;
  solveOpts.enableRegisterLevelSchedule = cfg.enableRegisterLevelSchedule;
  solveOpts.candidatesThreadMN = cfg.candidatesThreadMN;
  solveOpts.requirePerfectTiling = cfg.requirePerfectTiling;
  solveOpts.assumeFusedRelu = cfg.assumeFusedRelu;
  solveOpts.enableFootprintInference = cfg.enableFootprintInference;
  solveOpts.enableTilePropagation = cfg.enableTilePropagation;
  solveOpts.enableGlobalTraffic = cfg.enableGlobalTraffic;
  solveOpts.enableCutEdges = cfg.enableCutEdges;
  solveOpts.enableTwoLevelSchedule = cfg.enableTwoLevelSchedule;
  solveOpts.enablePaperSchedule = cfg.enablePaperSchedule;
  solveOpts.paperRecursiveRegisterLevel = cfg.paperRecursiveRegisterLevel;
  solveOpts.paperRecursiveInnerMinLevelExclusive =
      cfg.paperRecursiveInnerMinLevelExclusive;
  solveOpts.paperRecursiveMaxStages = cfg.paperRecursiveMaxStages;
  solveOpts.paperStrict = cfg.paperStrict;
  solveOpts.paperExpandReductionTile = cfg.paperExpandReductionTile;
  solveOpts.pruneOnProfileFailure = cfg.pruneOnProfileFailure;
  solveOpts.enableCoalescingPenalty = cfg.enableCoalescingPenalty;
  solveOpts.verboseCostModel = cfg.solverVerboseCost;
  solveOpts.scheduleTopK = cfg.scheduleTopK;
  solveOpts.maxConnectLevel = cfg.maxConnectLevel;

  welder::applyPaperModeDefaults(
      solveOpts, cfg.autoCandidatesExplicit, cfg.maxConnectLevelExplicit,
      cfg.enableProfiling);
  solveOpts.profile.enable = cfg.enableProfiling;
  solveOpts.profile.warmup = cfg.profileWarmup;
  solveOpts.profile.iters = cfg.profileIters;
  solveOpts.profile.maxParallelJobs = std::max(1, cfg.profileMaxParallelJobs);
  solveOpts.profile.timeoutSec = std::max(0, cfg.profileTimeoutSec);
  solveOpts.profile.cachePath = cfg.profileCachePath;
  solveOpts.profile.verbose = cfg.profileVerbose;
  solveOpts.tracer = tracerPtr;

  if (solveOpts.profile.enable) {
    std::filesystem::path exePath(cfg.argv0);
    std::filesystem::path exeDir = exePath.has_parent_path()
                                       ? exePath.parent_path()
                                       : std::filesystem::current_path();

    if (!cfg.profileProfilerBin.empty()) {
      solveOpts.profile.profilerBin = cfg.profileProfilerBin;
    } else {
      std::filesystem::path p = exeDir / "welder-profiler";
      solveOpts.profile.profilerBin = p.string();
    }

    if (!cfg.profileCompilerToNvvm.empty()) {
      solveOpts.profile.compilerToNvvmScript = cfg.profileCompilerToNvvm;
    } else {
      std::filesystem::path p = exeDir / ".." / "run_welder_to_nvvm_isa.sh";
      solveOpts.profile.compilerToNvvmScript = p.string();
    }
  }
  return solveOpts;
}

EffectiveRowReductionKnobs
deriveEffectiveRowReductionKnobs(mlir::ModuleOp module,
                                 const RowReductionKnobTuningRequest &req) {
  EffectiveRowReductionKnobs out;
  out.enableRowReductionInputPromotion = req.enableRowReductionInputPromotion;
  out.enableRowReductionInputPromotionVectorize =
      req.enableRowReductionInputPromotionVectorize;
  out.enableRowReductionWarp = req.enableRowReductionWarp;
  out.enableRowReductionVectorize = req.enableRowReductionVectorize;
  out.enableRowReductionRelaxBarriers = req.enableRowReductionRelaxBarriers;
  out.enableRowReductionSkipCombineBarrier =
      req.enableRowReductionSkipCombineBarrier;
  out.enableRowReductionCombineVectorize =
      req.enableRowReductionCombineVectorize;
  out.rowReductionVectorWidth = req.rowReductionVectorWidth;
  out.rowReductionThreadsX = req.rowReductionThreadsX;
  out.rowReductionInputVectorWidth = req.rowReductionInputVectorWidth;

  module->walk([&](mlir::Operation *op) {
    if (!out.hasMatmul && mlir::isa<mlir::linalg::MatmulOp>(op))
      out.hasMatmul = true;
    if (!out.hasRowReduction && isRowWiseReductionOp(op))
      out.hasRowReduction = true;
  });

  const bool rowReductionKnobsExplicit =
      req.rowReductionInputPromotionExplicit ||
      req.rowReductionInputPromotionVectorizeExplicit ||
      req.rowReductionWarpExplicit || req.rowReductionVectorizeExplicit ||
      req.rowReductionRelaxBarriersExplicit ||
      req.rowReductionSkipCombineBarrierExplicit ||
      req.rowReductionCombineVectorizeExplicit ||
      req.rowReductionVectorWidthExplicit || req.rowReductionThreadsXExplicit ||
      req.rowReductionInputVectorWidthExplicit;

  bool forceFastRowReduction =
      req.enableMatmulSoftmaxSharedReuseFusion &&
      req.enableRowReductionChainReuseFusion && out.hasMatmul &&
      out.hasRowReduction && !rowReductionKnobsExplicit &&
      (getEnvInt64OrDefault("WELDER_FORCE_MM_SM_FAST_ROW", 1) != 0);
  if (forceFastRowReduction) {
    if (!req.rowReductionWarpExplicit)
      out.enableRowReductionWarp = true;
    if (!req.rowReductionVectorizeExplicit)
      out.enableRowReductionVectorize = true;
    if (!req.rowReductionCombineVectorizeExplicit)
      out.enableRowReductionCombineVectorize = true;
    if (!req.rowReductionRelaxBarriersExplicit)
      out.enableRowReductionRelaxBarriers = true;
    if (!req.rowReductionSkipCombineBarrierExplicit)
      out.enableRowReductionSkipCombineBarrier = true;
    if (!req.rowReductionVectorWidthExplicit && out.rowReductionVectorWidth <= 0)
      out.rowReductionVectorWidth = 4;
    if (!req.rowReductionThreadsXExplicit && out.rowReductionThreadsX <= 0)
      out.rowReductionThreadsX = 32;
    if (!req.rowReductionInputVectorWidthExplicit &&
        out.rowReductionInputVectorWidth <= 0)
      out.rowReductionInputVectorWidth = 4;
  }

  bool forceTcSafeRowReduction =
      forceFastRowReduction &&
      (req.enableTensorCoreF16 || req.enableTensorCoreTf32) &&
      !rowReductionKnobsExplicit &&
      (getEnvInt64OrDefault("WELDER_FORCE_MM_SM_TC_SAFE_ROW", 1) != 0);
  if (forceTcSafeRowReduction) {
    out.enableRowReductionInputPromotion = false;
    out.enableRowReductionInputPromotionVectorize = false;
    out.enableRowReductionWarp = false;
    out.enableRowReductionVectorize = false;
    out.enableRowReductionRelaxBarriers = false;
    out.enableRowReductionSkipCombineBarrier = false;
    out.enableRowReductionCombineVectorize = false;
    out.rowReductionVectorWidth = 0;
    out.rowReductionInputVectorWidth = 0;
  }
  return out;
}

} // namespace welder::compiler

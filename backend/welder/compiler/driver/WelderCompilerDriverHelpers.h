#pragma once

#include "WelderCompilerPostbufferizeFixups.h"
#include "WelderSolverLib.h"
#include "WelderTrace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace welder::compiler {

int parseInputModule(mlir::MLIRContext &ctx, llvm::StringRef inputFilename,
                     welder::Tracer *tracerPtr,
                     mlir::OwningOpRef<mlir::ModuleOp> &module);

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
    const PostbufferizeFixupHooks &postbufferizeFixupHooks);

int loadTransformLibraryIntoDialect(mlir::MLIRContext &ctx,
                                    mlir::OwningOpRef<mlir::ModuleOp> &transformLib,
                                    welder::Tracer *tracerPtr);

struct SolveOptionBuildConfig {
  int64_t smemBytes = 0;
  int64_t numSM = 0;
  int64_t maxBlocksPerSM = 0;
  int64_t warpSize = 0;
  int64_t smPartition = 0;
  int64_t maxSmemUsageBytes = 0;
  int64_t globalTransactionBytes = 0;
  int64_t globalReadTransactionBytes = 0;
  int64_t globalWriteTransactionBytes = 0;
  int64_t maxThreadsPerSM = 0;
  std::vector<int64_t> candidatesMN;
  std::vector<int64_t> candidatesK;
  std::vector<int64_t> candidatesThreadMN;
  bool autoCandidates = false;
  bool autoCandidatesExplicit = false;
  bool enableRegisterLevelSchedule = false;
  bool requirePerfectTiling = false;
  bool assumeFusedRelu = false;
  bool enableFootprintInference = false;
  bool enableTilePropagation = false;
  bool enableGlobalTraffic = false;
  bool enableCutEdges = false;
  bool enableTwoLevelSchedule = false;
  bool enablePaperSchedule = false;
  bool paperRecursiveRegisterLevel = false;
  int paperRecursiveInnerMinLevelExclusive = 0;
  int paperRecursiveMaxStages = 0;
  bool paperStrict = false;
  bool paperExpandReductionTile = false;
  bool pruneOnProfileFailure = false;
  bool enableCoalescingPenalty = false;
  bool solverVerboseCost = false;
  int64_t scheduleTopK = 0;
  int maxConnectLevel = 0;
  bool maxConnectLevelExplicit = false;
  bool enableProfiling = false;
  int profileWarmup = 0;
  int profileIters = 0;
  int profileMaxParallelJobs = 1;
  int profileTimeoutSec = 0;
  std::string profileCompilerToNvvm;
  std::string profileProfilerBin;
  std::string profileCachePath;
  bool profileVerbose = false;
  std::string argv0;
};

welder::SolveOptions buildSolveOptions(const SolveOptionBuildConfig &cfg,
                                       welder::Tracer *tracerPtr);

struct RowReductionKnobTuningRequest {
  bool enableMatmulSoftmaxSharedReuseFusion = false;
  bool enableRowReductionChainReuseFusion = false;
  bool enableTensorCoreTf32 = false;
  bool enableTensorCoreF16 = false;
  bool enableRowReductionInputPromotion = false;
  bool enableRowReductionInputPromotionVectorize = false;
  bool enableRowReductionWarp = false;
  bool enableRowReductionVectorize = false;
  bool enableRowReductionRelaxBarriers = false;
  bool enableRowReductionSkipCombineBarrier = false;
  bool enableRowReductionCombineVectorize = false;
  int64_t rowReductionVectorWidth = 0;
  int64_t rowReductionThreadsX = 0;
  int64_t rowReductionInputVectorWidth = 0;
  bool rowReductionInputPromotionExplicit = false;
  bool rowReductionInputPromotionVectorizeExplicit = false;
  bool rowReductionWarpExplicit = false;
  bool rowReductionVectorizeExplicit = false;
  bool rowReductionRelaxBarriersExplicit = false;
  bool rowReductionSkipCombineBarrierExplicit = false;
  bool rowReductionCombineVectorizeExplicit = false;
  bool rowReductionVectorWidthExplicit = false;
  bool rowReductionThreadsXExplicit = false;
  bool rowReductionInputVectorWidthExplicit = false;
};

struct EffectiveRowReductionKnobs {
  bool hasMatmul = false;
  bool hasRowReduction = false;
  bool enableRowReductionInputPromotion = false;
  bool enableRowReductionInputPromotionVectorize = false;
  bool enableRowReductionWarp = false;
  bool enableRowReductionVectorize = false;
  bool enableRowReductionRelaxBarriers = false;
  bool enableRowReductionSkipCombineBarrier = false;
  bool enableRowReductionCombineVectorize = false;
  int64_t rowReductionVectorWidth = 0;
  int64_t rowReductionThreadsX = 0;
  int64_t rowReductionInputVectorWidth = 0;
};

EffectiveRowReductionKnobs
deriveEffectiveRowReductionKnobs(mlir::ModuleOp module,
                                 const RowReductionKnobTuningRequest &req);

} // namespace welder::compiler


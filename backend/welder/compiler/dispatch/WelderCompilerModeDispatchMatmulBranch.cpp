#include "WelderCompilerModeDispatchBranches.h"

#include "WelderCompilerAnchorDiscovery.h"
#include "WelderCompilerDispatchPlan.h"
#include "WelderCompilerFusionAnchors.h"
#include "WelderCompilerFusionPairBuild.h"
#include "WelderCompilerGenericCutEdgesTransformLibrary.h"
#include "WelderCompilerGenericTransformAndFusion.h"
#include "WelderCompilerMatmulTransformLibrary.h"
#include "WelderCompilerPassTraceAndEnv.h"
#include "WelderCompilerTileDecision.h"
#include "WelderCompilerModeDispatchBranchUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <vector>

using namespace mlir;

namespace welder::compiler {

int buildTransformLibraryFromMatmulBranch(const ModeDispatchContext &modeCtx) {
  auto &module = *modeCtx.module;
  auto &ctx = *modeCtx.ctx;
  auto *tracerPtr = modeCtx.tracerPtr;
  llvm::StringRef inputFilename = modeCtx.inputFilename;
  auto &transformLib = *modeCtx.transformLib;
  auto &solveOpts = *modeCtx.solveOpts;

  bool enableGenericFusion = modeCtx.enableGenericFusion;

  int64_t forceTileM = modeCtx.forceTileM;
  int64_t forceTileN = modeCtx.forceTileN;
  int64_t forceTileK = modeCtx.forceTileK;

  int64_t threadTileM = modeCtx.mapping.threadTileM;
  int64_t threadTileN = modeCtx.mapping.threadTileN;
  bool swapBlockDims = modeCtx.mapping.swapBlockDims;
  bool gSkipMapNestedForallToThreads =
      modeCtx.mapping.skipMapNestedForallToThreads;
  bool gSkipMapForallToBlocks = modeCtx.mapping.skipMapForallToBlocks;

  bool enableAsyncCopy = modeCtx.asyncPipeline.enableAsyncCopy;
  bool enableSoftwarePipelining = modeCtx.asyncPipeline.enableSoftwarePipelining;
  int64_t pipelineDepth = modeCtx.asyncPipeline.pipelineDepth;
  bool pipelinePeelEpilogue = modeCtx.asyncPipeline.pipelinePeelEpilogue;
  bool asyncBypassL1 = modeCtx.asyncPipeline.asyncBypassL1;

  bool enableTensorCoreTf32 = modeCtx.tensorCore.enableTf32;
  bool enableTensorCoreF16 = modeCtx.tensorCore.enableF16;

  bool enableCutEdges = modeCtx.rowReduction.enableCutEdges;
  bool reductionChainSplitBroadcastEdges =
      modeCtx.rowReduction.reductionChainSplitBroadcastEdges;
  bool enableRowReductionChainReuseFusion =
      modeCtx.rowReduction.enableChainReuseFusion;
  bool enableRegisterLevelSchedule =
      modeCtx.rowReduction.enableRegisterLevelSchedule;

  int64_t maxConnectLevel = modeCtx.rowReduction.maxConnectLevel;
  bool traceVerbose = modeCtx.traceVerbose;

  bool hasRowReduction = modeCtx.rowReduction.hasRowReduction;
  bool effEnableRowReductionInputPromotion =
      modeCtx.rowReduction.effEnableInputPromotion;
  bool effEnableRowReductionWarp = modeCtx.rowReduction.effEnableWarp;
  bool effEnableRowReductionVectorize =
      modeCtx.rowReduction.effEnableVectorize;
  bool effEnableRowReductionCombineVectorize =
      modeCtx.rowReduction.effEnableCombineVectorize;
  int64_t effRowReductionVectorWidth = modeCtx.rowReduction.effVectorWidth;
  int64_t effRowReductionThreadsX = modeCtx.rowReduction.effThreadsX;

  (void)inputFilename;
  (void)enableGenericFusion;
  (void)enableCutEdges;
  (void)reductionChainSplitBroadcastEdges;
  (void)enableRowReductionChainReuseFusion;
  (void)maxConnectLevel;
  (void)traceVerbose;
  (void)hasRowReduction;
  (void)effEnableRowReductionInputPromotion;
  (void)effEnableRowReductionWarp;
  (void)effEnableRowReductionVectorize;
  (void)effEnableRowReductionCombineVectorize;
  (void)effRowReductionVectorWidth;
  (void)effRowReductionThreadsX;

  auto branchImpl = [&]() -> int {
#include "WelderCompilerModeDispatchMatmulBranchBody.h"
  };
  return branchImpl();
}

} // namespace welder::compiler

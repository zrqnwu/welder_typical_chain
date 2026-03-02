#pragma once

#include "WelderSolverLib.h"
#include "WelderTrace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace welder::compiler {

struct TensorCoreConfig {
  bool enableTf32 = false;
  bool enableF16 = false;
};

struct MappingConfig {
  int64_t threadTileM = 0;
  int64_t threadTileN = 0;
  bool swapBlockDims = false;
  bool skipMapNestedForallToThreads = false;
  bool skipMapForallToBlocks = false;
};

struct AsyncPipelineConfig {
  bool enableAsyncCopy = false;
  bool enableSoftwarePipelining = false;
  int64_t pipelineDepth = 0;
  bool pipelinePeelEpilogue = false;
  bool asyncBypassL1 = false;
};

struct RowReductionConfig {
  bool hasRowReduction = false;
  bool enableCutEdges = false;
  bool reductionChainSplitBroadcastEdges = false;
  bool enableChainReuseFusion = false;
  bool enableRegisterLevelSchedule = false;
  int64_t maxConnectLevel = 0;

  bool effEnableInputPromotion = false;
  bool effEnableWarp = false;
  bool effEnableVectorize = false;
  bool effEnableCombineVectorize = false;
  int64_t effVectorWidth = 0;
  int64_t effThreadsX = 0;
};

struct ModeDispatchContext {
  mlir::OwningOpRef<mlir::ModuleOp> *module = nullptr;
  mlir::MLIRContext *ctx = nullptr;
  welder::Tracer *tracerPtr = nullptr;
  llvm::StringRef inputFilename;
  mlir::OwningOpRef<mlir::ModuleOp> *transformLib = nullptr;
  welder::SolveOptions *solveOpts = nullptr;

  bool codegenFromKernelAttrs = false;
  bool enableGenericProblem = false;
  bool enableGenericFusion = false;

  int64_t forceTileM = 0;
  int64_t forceTileN = 0;
  int64_t forceTileK = 0;

  MappingConfig mapping;
  AsyncPipelineConfig asyncPipeline;
  TensorCoreConfig tensorCore;
  RowReductionConfig rowReduction;

  bool traceVerbose = false;
};

int buildTransformLibraryForMode(const ModeDispatchContext &modeCtx);

} // namespace welder::compiler

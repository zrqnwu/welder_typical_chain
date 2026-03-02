#pragma once

#include "WelderCompilerDriverHelpers.h"
#include "WelderCompilerModeDispatch.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/StringRef.h"

namespace welder {
class Tracer;
struct SolveOptions;
} // namespace welder

namespace welder::compiler {

struct CompilerRunConfig {
  bool codegenFromKernelAttrs = false;
  bool enableGenericProblem = false;
  bool enableGenericFusion = false;

  int64_t forceTileM = 0;
  int64_t forceTileN = 0;
  int64_t forceTileK = 0;

  MappingConfig mapping;
  AsyncPipelineConfig asyncPipeline;
  TensorCoreConfig tensorCore;

  bool enableCutEdges = false;
  bool reductionChainSplitBroadcastEdges = false;
  bool enableRowReductionChainReuseFusion = false;
  bool enableRegisterLevelSchedule = false;
  int64_t maxConnectLevel = 0;

  bool traceVerbose = false;
};

int validateCompilerRunConfig(const CompilerRunConfig &cfg);

ModeDispatchContext
buildModeDispatchContext(const CompilerRunConfig &cfg,
                         mlir::OwningOpRef<mlir::ModuleOp> *module,
                         mlir::MLIRContext *ctx, welder::Tracer *tracerPtr,
                         llvm::StringRef inputFilename,
                         mlir::OwningOpRef<mlir::ModuleOp> *transformLib,
                         welder::SolveOptions *solveOpts,
                         const EffectiveRowReductionKnobs &effRowReduction);

} // namespace welder::compiler


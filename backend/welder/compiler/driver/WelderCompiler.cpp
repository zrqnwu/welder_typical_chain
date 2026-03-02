#include "WelderCompilerMain.h"
#include "WelderCompilerAnchorDiscovery.h"
#include "WelderCompilerAsyncCopyRewrite.h"
#include "WelderCompilerAsyncPipelineHelpers.h"
#include "WelderCompilerDispatchPlan.h"
#include "WelderCompilerFusionPairBuild.h"
#include "WelderCompilerFusionAnchors.h"
#include "WelderCompilerRowReductionInputPromotion.h"
#include "WelderCompilerModeDispatch.h"
#include "WelderCompilerTileDecision.h"
#include "WelderCompilerPostbufferizeFixups.h"
#include "WelderCompilerPassTraceAndEnv.h"
#include "WelderCompilerGenericTransformAndFusion.h"
#include "WelderCompilerMatmulTransformLibrary.h"
#include "WelderCompilerGenericCutEdgesTransformLibrary.h"
#include "WelderCompilerDriverHelpers.h"
#include "WelderCompilerRunConfig.h"
#include "WelderCompilerMemoryReuseTransforms.h"
#include "WelderCompilerMatmulSoftmaxCanonicalize.h"
#include "WelderCompilerPostprocessCleanup.h"
#include "WelderSolverLib.h"
#include "WelderSolveOptionDefaults.h"
#include "WelderTrace.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

using namespace mlir;

namespace {
static welder::compiler::RowReductionFixupToggles
makeRowReductionFixupToggles(
    const welder::compiler::EffectiveRowReductionKnobs &effKnobs) {
  welder::compiler::RowReductionFixupToggles toggles;
  toggles.enableInputPromotion = effKnobs.enableRowReductionInputPromotion;
  toggles.enableInputPromotionVectorize =
      effKnobs.enableRowReductionInputPromotionVectorize;
  toggles.enableRelaxBarriers = effKnobs.enableRowReductionRelaxBarriers;
  toggles.enableSkipCombineBarrier =
      effKnobs.enableRowReductionSkipCombineBarrier;
  toggles.inputVectorWidth = effKnobs.rowReductionInputVectorWidth;
  return toggles;
}

static welder::compiler::PostbufferizeFixupHooks
makePostbufferizeFixupHooks() {
  welder::compiler::PostbufferizeFixupHooks hooks;
  hooks.fuseSquareIntoRowReduction = [](ModuleOp payload, bool meanScaleFusion) {
    welder::compiler::fuseSquareIntoRowReduction(payload, meanScaleFusion);
  };
  hooks.stageRowReduction2DTileToWorkgroup =
      [](ModuleOp payload, int64_t smemCap, int64_t reserveBytes,
         bool enableInPlace2DReuse, bool enableVectorize, bool enableAsync,
         bool enablePipeline, bool bypassL1, bool relaxBarriers,
         int64_t vectorWidth) {
        welder::compiler::stageRowReduction2DTileToWorkgroup(
            payload, smemCap, reserveBytes, enableInPlace2DReuse,
            enableVectorize, enableAsync, enablePipeline, bypassL1,
            relaxBarriers, vectorWidth);
      };
  hooks.promoteRowReductionInputsToWorkgroup =
      [](ModuleOp payload) {
        welder::compiler::promoteRowReductionInputsToWorkgroup(payload);
      };
  hooks.promoteRowReductionScratchToWorkgroup =
      [](ModuleOp payload) {
        welder::compiler::promoteRowReductionScratchToWorkgroup(payload);
      };
  hooks.promoteSharedRowReductionResultAllocasToWorkgroup =
      [](ModuleOp payload) {
        welder::compiler::promoteSharedRowReductionResultAllocasToWorkgroup(
            payload);
      };
  hooks.hoistWorkgroupAllocs =
      [](ModuleOp payload) { welder::compiler::hoistWorkgroupAllocs(payload); };
  hooks.insertBarrierAfterCombiningReductions =
      [](ModuleOp payload, bool skipCombineBarrier) {
        welder::compiler::insertBarrierAfterCombiningReductions(
            payload, skipCombineBarrier);
      };
  hooks.insertKeepBarrierAfterPredicatedElementwise1D =
      [](ModuleOp payload) {
        welder::compiler::insertKeepBarrierAfterPredicatedElementwise1D(
            payload);
      };
  hooks.reorderBroadcast1DProducersBefore2DConsumers = [](ModuleOp payload) {
    welder::compiler::reorderBroadcast1DProducersBefore2DConsumers(payload);
  };
  hooks.splitPredicatedBarrierStages =
      [](ModuleOp payload) {
        welder::compiler::splitPredicatedBarrierStages(payload);
      };
  hooks.hoistPredicatedBarriers =
      [](ModuleOp payload) { welder::compiler::hoistPredicatedBarriers(payload); };
  hooks.removeRedundantBarriers =
      [](ModuleOp payload) { welder::compiler::removeRedundantBarriers(payload); };
  hooks.eraseHostDuplicatesOfFusedLaunchOps =
      [](ModuleOp payload) {
        welder::compiler::eraseHostDuplicatesOfFusedLaunchOps(payload);
      };
  hooks.promoteLaunchLocal1DBuffersToWorkgroup =
      [](ModuleOp payload) {
        welder::compiler::promoteLaunchLocal1DBuffersToWorkgroup(payload);
      };
  hooks.canonicalizeMatmulSoftmaxSharedReuse =
      [](ModuleOp payload) {
        welder::compiler::canonicalizeMatmulSoftmaxSharedReuse(payload);
      };
  hooks.rewritePromotedLinalgCopiesToAsyncCopy =
      [](ModuleOp payload, bool enableAsync, bool bypassL1) {
        welder::compiler::rewritePromotedLinalgCopiesToAsyncCopy(
            payload, enableAsync, bypassL1);
      };
  hooks.padWorkgroupAllocs = [](ModuleOp payload, int64_t padBytes) {
    welder::compiler::padWorkgroupAllocs(payload, padBytes);
  };
  hooks.eraseDeadHostIntermediateAllocs =
      [](ModuleOp payload) {
        welder::compiler::eraseDeadHostIntermediateAllocs(payload);
      };
  return hooks;
}
} // namespace

int welderCompilerMain(int argc, char **argv) {
  // 允许同一进程中重复调用（例如 C API stress/repeat 模式）。
  // 注意：不能用 ResetCommandLineParser()，它会清空选项注册表；
  // 在本函数使用静态 cl::opt 的情况下，应只重置 occurrence/value。
  llvm::cl::ResetAllOptionOccurrences();

  // ==============================================================
  // 运行流程总览（main 只做编排）
  //  0) 解析命令行参数（I/O、调度、融合、profiling 等）
  //  1) 初始化 trace 与全局调试开关
  //  2) 注册 MLIR dialect/interface 并读取输入模块
  //  3) 预注解（如 matmul 的 workgroup padding）
  //  4) 构建 SolveOptions + 推导行归约有效开关
  //  5) 根据模式构建 transform library（三条分支）
  //  6) 加载 transform library 到 TransformDialect
  //  7) 执行 pre/buf/post lowering pipeline 并输出结果
  // ==============================================================

  // [参数-输入输出] 输入/输出路径与中间阶段 IR 导出。
  // [阶段0] CLI 选项定义（拆分到独立文件，主流程保持短小）
#include "WelderCompilerCliOptions.h"

  // 步骤 0：解析参数并做早期一致性检查。
  llvm::cl::ParseCommandLineOptions(argc, argv, "welder-compiler (minimal)\n");
  if (traceVerbose) {
    llvm::errs() << "[welder] opts: generic="
                 << (enableGenericProblem ? "1" : "0")
                 << " cut_edges=" << (enableCutEdges ? "1" : "0")
                 << " tc_f16=" << (enableTensorCoreF16 ? "1" : "0")
                 << " skip_blocks="
                 << (skipMapForallToBlocks ? "1" : "0")
                 << " skip_threads="
                 << (skipMapNestedForallToThreads ? "1" : "0") << "\n";
  }
  // 步骤 1：初始化 tracing（文本/JSONL）。
  welder::TraceConfig traceCfg;
  traceCfg.text = trace;
  traceCfg.verbose = traceVerbose;
  traceCfg.jsonl = !traceFile.getValue().empty();
  traceCfg.jsonlPath = traceFile.getValue();
  traceCfg.jsonlAppend = traceFileAppend;
  welder::Tracer tracer(traceCfg);
  welder::Tracer *tracerPtr = tracer.enabled() ? &tracer : nullptr;
  if (tracerPtr) {
    llvm::json::Object f;
    f["input"] = inputFilename.getValue();
    f["output"] = outputFilename.getValue();
    tracerPtr->event("compiler.main.start", std::move(f));
  }

  // 步骤 2：注册 dialect 与外部接口模型（tiling/bufferize/transform）。
  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  bufferization::BufferizationDialect, func::FuncDialect,
                  cf::ControlFlowDialect,
                  gpu::GPUDialect, linalg::LinalgDialect, nvgpu::NVGPUDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect, tensor::TensorDialect,
                  transform::TransformDialect, vector::VectorDialect>();
  linalg::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  nvgpu::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
  // tile_using_forall / tile_using_for 会依赖 TilingInterface；它在 linalg 里是
  // 通过“外部模型”注册的，所以我们必须显式注册它，否则会触发断言崩溃。
  linalg::registerTilingInterfaceExternalModels(registry);

  // one-shot-bufferize 依赖 BufferizableOpInterface；同样很多 dialect 是通过外部模型注册。
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // 步骤 3：读取输入 payload module。
  OwningOpRef<ModuleOp> module;
  if (int parseRc =
          welder::compiler::parseInputModule(
              ctx, inputFilename.getValue(), tracerPtr, module);
      parseRc != 0)
    return parseRc;

  // 论文/Welder 对齐（stride_map/output_strides_map，matmul 子集）：
  // 将每个 matmul 的 stride-map padding 写入算子属性，供后续 pass 按 buffer 生效，
  // 避免依赖全局环境变量开关。
  {
    // 步骤 4：预注解（例如 matmul 的 workgroup padding 属性）。
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.annotate_workgroup_padding")
                  : welder::Tracer::Span();

    int64_t pad = std::max<int64_t>(0, workgroupPadLastDim.getValue());
    bool matmulOnly = workgroupPadLastDimMatmulOnly.getValue();
    if ((enableTensorCoreTf32 || enableTensorCoreF16) && pad == 0) {
      pad = 8;
      matmulOnly = true;
    }
    if (pad > 0) {
      OpBuilder b(&ctx);
      module->walk([&](linalg::MatmulOp mm) {
        mm->setAttr("welder.workgroup_pad_last_dim", b.getI64IntegerAttr(pad));
        mm->setAttr("welder.workgroup_pad_last_dim_matmul_only",
                    b.getBoolAttr(matmulOnly));
      });
    }
  }

  // --- 求解阶段 ---
  {
  // 步骤 5：组装 solve 配置（硬件模型、候选空间、profiling 参数）。
  [[maybe_unused]] auto solveSpan =
      tracerPtr ? tracerPtr->span("compiler.solve",
                                  llvm::json::Object{
                                      {"generic", enableGenericProblem.getValue()},
                                      {"cut_edges", enableCutEdges.getValue()},
                                      {"paper_schedule", enablePaperSchedule.getValue()},
                                      {"profiling", enableProfiling.getValue()},
                                  })
                : welder::Tracer::Span();

  welder::compiler::SolveOptionBuildConfig solveCfg;
  solveCfg.smemBytes = smemBytes;
  solveCfg.numSM = numSM;
  solveCfg.maxBlocksPerSM = maxBlocksPerSM;
  solveCfg.warpSize = warpSize;
  solveCfg.smPartition = smPartition;
  solveCfg.maxSmemUsageBytes = maxSmemUsageBytes;
  solveCfg.globalTransactionBytes = globalTransactionBytes;
  solveCfg.globalReadTransactionBytes = globalReadTransactionBytes;
  solveCfg.globalWriteTransactionBytes = globalWriteTransactionBytes;
  solveCfg.maxThreadsPerSM = maxThreadsPerSM;
  solveCfg.candidatesMN = welder::parseCsvIntList(candidatesMN);
  solveCfg.candidatesK = welder::parseCsvIntList(candidatesK);
  solveCfg.candidatesThreadMN = welder::parseCsvIntList(candidatesThreadMN);
  solveCfg.autoCandidates = autoCandidates;
  solveCfg.autoCandidatesExplicit = autoCandidates.getNumOccurrences() != 0;
  solveCfg.enableRegisterLevelSchedule = enableRegisterLevelSchedule;
  solveCfg.requirePerfectTiling = requirePerfectTiling;
  solveCfg.assumeFusedRelu = assumeFusedRelu;
  solveCfg.enableFootprintInference = enableFootprintInference;
  solveCfg.enableTilePropagation = enableTilePropagation;
  solveCfg.enableGlobalTraffic = enableGlobalTraffic;
  solveCfg.enableCutEdges = enableCutEdges;
  solveCfg.enableTwoLevelSchedule = enableTwoLevelSchedule;
  solveCfg.enablePaperSchedule = enablePaperSchedule;
  solveCfg.paperRecursiveRegisterLevel = paperRecursiveRegisterLevel;
  solveCfg.paperRecursiveInnerMinLevelExclusive =
      paperRecursiveInnerMinLevelExclusive;
  solveCfg.paperRecursiveMaxStages = paperRecursiveMaxStages;
  solveCfg.paperStrict = paperStrict;
  solveCfg.paperExpandReductionTile = paperExpandReductionTile;
  solveCfg.pruneOnProfileFailure = pruneOnProfileFailure;
  solveCfg.enableCoalescingPenalty = enableCoalescingPenalty;
  solveCfg.solverVerboseCost = solverVerboseCost;
  solveCfg.scheduleTopK = scheduleTopK;
  solveCfg.maxConnectLevel = maxConnectLevel;
  solveCfg.maxConnectLevelExplicit = maxConnectLevel.getNumOccurrences() != 0;
  solveCfg.enableProfiling = enableProfiling;
  solveCfg.profileWarmup = profileWarmup;
  solveCfg.profileIters = profileIters;
  solveCfg.profileMaxParallelJobs = profileMaxParallelJobs.getValue();
  solveCfg.profileTimeoutSec = profileTimeoutSec.getValue();
  solveCfg.profileCompilerToNvvm = profileCompilerToNvvm.getValue();
  solveCfg.profileProfilerBin = profileProfilerBin.getValue();
  solveCfg.profileCachePath = profileCachePath;
  solveCfg.profileVerbose = profileVerbose;
  solveCfg.argv0 = argv[0];
  welder::SolveOptions solveOpts =
      welder::compiler::buildSolveOptions(solveCfg, tracerPtr);

  OwningOpRef<ModuleOp> transformLib;

  // 步骤 6：根据当前图与命令行，推导“生效的行归约策略开关”。
  welder::compiler::RowReductionKnobTuningRequest rowReq;
  rowReq.enableMatmulSoftmaxSharedReuseFusion =
      enableMatmulSoftmaxSharedReuseFusion;
  rowReq.enableRowReductionChainReuseFusion =
      enableRowReductionChainReuseFusion;
  rowReq.enableTensorCoreTf32 = enableTensorCoreTf32;
  rowReq.enableTensorCoreF16 = enableTensorCoreF16;
  rowReq.enableRowReductionInputPromotion = enableRowReductionInputPromotion;
  rowReq.enableRowReductionInputPromotionVectorize =
      enableRowReductionInputPromotionVectorize;
  rowReq.enableRowReductionWarp = enableRowReductionWarp;
  rowReq.enableRowReductionVectorize = enableRowReductionVectorize;
  rowReq.enableRowReductionRelaxBarriers = enableRowReductionRelaxBarriers;
  rowReq.enableRowReductionSkipCombineBarrier =
      enableRowReductionSkipCombineBarrier;
  rowReq.enableRowReductionCombineVectorize =
      enableRowReductionCombineVectorize;
  rowReq.rowReductionVectorWidth = rowReductionVectorWidth;
  rowReq.rowReductionThreadsX = rowReductionThreadsX;
  rowReq.rowReductionInputVectorWidth = rowReductionInputVectorWidth;
  rowReq.rowReductionInputPromotionExplicit =
      enableRowReductionInputPromotion.getNumOccurrences() > 0;
  rowReq.rowReductionInputPromotionVectorizeExplicit =
      enableRowReductionInputPromotionVectorize.getNumOccurrences() > 0;
  rowReq.rowReductionWarpExplicit =
      enableRowReductionWarp.getNumOccurrences() > 0;
  rowReq.rowReductionVectorizeExplicit =
      enableRowReductionVectorize.getNumOccurrences() > 0;
  rowReq.rowReductionRelaxBarriersExplicit =
      enableRowReductionRelaxBarriers.getNumOccurrences() > 0;
  rowReq.rowReductionSkipCombineBarrierExplicit =
      enableRowReductionSkipCombineBarrier.getNumOccurrences() > 0;
  rowReq.rowReductionCombineVectorizeExplicit =
      enableRowReductionCombineVectorize.getNumOccurrences() > 0;
  rowReq.rowReductionVectorWidthExplicit =
      rowReductionVectorWidth.getNumOccurrences() > 0;
  rowReq.rowReductionThreadsXExplicit =
      rowReductionThreadsX.getNumOccurrences() > 0;
  rowReq.rowReductionInputVectorWidthExplicit =
      rowReductionInputVectorWidth.getNumOccurrences() > 0;

  welder::compiler::EffectiveRowReductionKnobs effRowReduction =
      welder::compiler::deriveEffectiveRowReductionKnobs(*module, rowReq);
  // 步骤 7：组装并校验运行配置对象（parse+validate 专用模块）。
  welder::compiler::CompilerRunConfig runCfg;
  runCfg.codegenFromKernelAttrs = codegenFromKernelAttrs.getValue();
  runCfg.enableGenericProblem = enableGenericProblem.getValue();
  runCfg.enableGenericFusion = enableGenericFusion.getValue();
  runCfg.forceTileM = forceTileM.getValue();
  runCfg.forceTileN = forceTileN.getValue();
  runCfg.forceTileK = forceTileK.getValue();
  runCfg.mapping.threadTileM = threadTileM.getValue();
  runCfg.mapping.threadTileN = threadTileN.getValue();
  runCfg.mapping.swapBlockDims = swapBlockDims.getValue();
  runCfg.mapping.skipMapNestedForallToThreads =
      skipMapNestedForallToThreads.getValue();
  runCfg.mapping.skipMapForallToBlocks = skipMapForallToBlocks.getValue();
  runCfg.asyncPipeline.enableAsyncCopy = enableAsyncCopy.getValue();
  runCfg.asyncPipeline.enableSoftwarePipelining =
      enableSoftwarePipelining.getValue();
  runCfg.asyncPipeline.pipelineDepth = pipelineDepth.getValue();
  runCfg.asyncPipeline.pipelinePeelEpilogue = pipelinePeelEpilogue.getValue();
  runCfg.asyncPipeline.asyncBypassL1 = asyncBypassL1.getValue();
  runCfg.tensorCore.enableTf32 = enableTensorCoreTf32.getValue();
  runCfg.tensorCore.enableF16 = enableTensorCoreF16.getValue();
  runCfg.enableCutEdges = enableCutEdges.getValue();
  runCfg.reductionChainSplitBroadcastEdges =
      reductionChainSplitBroadcastEdges.getValue();
  runCfg.enableRowReductionChainReuseFusion =
      enableRowReductionChainReuseFusion.getValue();
  runCfg.enableRegisterLevelSchedule =
      enableRegisterLevelSchedule.getValue();
  runCfg.maxConnectLevel = maxConnectLevel.getValue();
  runCfg.traceVerbose = traceVerbose.getValue();
  if (int cfgRc = welder::compiler::validateCompilerRunConfig(runCfg);
      cfgRc != 0) {
    return cfgRc;
  }

  // 步骤 7：按模式构建 transform library（已独立为 ModeDispatch 阶段）。
  welder::compiler::ModeDispatchContext modeCtx =
      welder::compiler::buildModeDispatchContext(
          runCfg, &module, &ctx, tracerPtr, inputFilename.getValue(),
          &transformLib, &solveOpts, effRowReduction);

  if (int modeRc = welder::compiler::buildTransformLibraryForMode(modeCtx);
      modeRc != 0)
    return modeRc;

  // 步骤 8：将 transform library 注入 TransformDialect。
  if (int loadRc = welder::compiler::loadTransformLibraryIntoDialect(
          ctx, transformLib, tracerPtr);
      loadRc != 0)
    return loadRc;

  // 步骤 9：执行 pre/bufferize/post 流水线并写出结果。
  auto rowReductionFixupToggles = makeRowReductionFixupToggles(effRowReduction);
  auto postbufferizeFixupHooks = makePostbufferizeFixupHooks();
  return welder::compiler::runLoweringPipelineAndEmit(
      * module, ctx, tracerPtr, tracePasses.getValue(), emitAfterPre.getValue(),
      emitAfterBuf.getValue(), emitAfterPost.getValue(),
      outputFilename.getValue(), enableRowReductionChainReuseFusion.getValue(),
      enableRowReductionSquareFusion.getValue(),
      enableRowReductionMeanScaleFusion.getValue(), smemBytes.getValue(),
      enableAsyncCopy.getValue(), enableSoftwarePipelining.getValue(),
      asyncBypassL1.getValue(),
      enableMatmulSoftmaxSharedReuseFusion.getValue(),
      rowReductionFixupToggles, postbufferizeFixupHooks);
  }
}

#ifndef WELDER_COMPILER_NO_MAIN
int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  return welderCompilerMain(argc, argv);
}
#endif

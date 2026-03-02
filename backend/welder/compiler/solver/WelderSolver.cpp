#include "WelderSolverLib.h"
#include "WelderSolverCandidateGenerator.h"
#include "WelderSolverCostModel.h"
#include "WelderSolverProfilerRunner.h"
#include "WelderSolverReportWriter.h"
#include "WelderSolveOptionDefaults.h"
#include "WelderTrace.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
  using namespace mlir;

  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input.mlir>"), llvm::cl::Required);
  llvm::cl::opt<int64_t> smemBytes(
      "smem-bytes", llvm::cl::desc("Shared memory capacity per block (bytes)"),
      llvm::cl::init(48 * 1024));
  llvm::cl::opt<int64_t> numSM("num-sm",
                               llvm::cl::desc("Number of SMs (approx)"),
                               llvm::cl::init(80));
  llvm::cl::opt<int64_t> maxBlocksPerSM(
      "max-blocks-per-sm",
      llvm::cl::desc("Upper bound for blocks/SM used in wave estimation"),
      llvm::cl::init(4));
  llvm::cl::opt<int64_t> warpSize(
      "warp-size",
      llvm::cl::desc("Paper/Welder parity: warp size (default 32)"),
      llvm::cl::init(32));
  llvm::cl::opt<int64_t> smPartition(
      "sm-partition",
      llvm::cl::desc("Paper/Welder parity: SM partition heuristic (default 4)"),
      llvm::cl::init(4));
  llvm::cl::opt<int64_t> maxSmemUsageBytes(
      "max-smem-usage-bytes",
      llvm::cl::desc("Paper/Welder parity: max shared usage per SM (bytes). "
                     "0 means 2*smem-bytes."),
      llvm::cl::init(0));
  llvm::cl::opt<int64_t> globalTransactionBytes(
      "global-transaction-bytes",
      llvm::cl::desc("Paper-aligned: global memory transaction width in bytes "
                     "(default 128)"),
      llvm::cl::init(128));
  llvm::cl::opt<int64_t> globalReadTransactionBytes(
      "global-read-transaction-bytes",
      llvm::cl::desc("Paper/Welder parity: global read transaction width in bytes "
                     "(default 128)"),
      llvm::cl::init(128));
  llvm::cl::opt<int64_t> globalWriteTransactionBytes(
      "global-write-transaction-bytes",
      llvm::cl::desc("Paper/Welder parity: global write transaction width in bytes "
                     "(default 32)"),
      llvm::cl::init(32));
  llvm::cl::opt<int64_t> maxThreadsPerSM(
      "max-threads-per-sm",
      llvm::cl::desc("Approx max resident threads per SM (for occupancy heuristic)"),
      llvm::cl::init(2048));
  llvm::cl::opt<int64_t> maxRegistersPerSM(
      "max-registers-per-sm",
      llvm::cl::desc("Approx 32-bit registers per SM (for pruning heuristic)"),
      llvm::cl::init(65536));
  llvm::cl::opt<int64_t> maxRegistersPerThread(
      "max-registers-per-thread",
      llvm::cl::desc("Approx max registers per thread (for pruning heuristic)"),
      llvm::cl::init(255));
  llvm::cl::opt<std::string> candidatesMN(
      "candidates-mn",
      llvm::cl::desc("Candidate tile sizes for M/N (csv), e.g. 32,64,128"),
      llvm::cl::init("32,64,128"));
  llvm::cl::opt<std::string> candidatesK(
      "candidates-k",
      llvm::cl::desc("Candidate tile sizes for K (csv), e.g. 8,16,32"),
      llvm::cl::init("8,16,32"));
  llvm::cl::opt<bool> autoCandidates(
      "auto-candidates",
      llvm::cl::desc("Paper-aligned: generate candidate tiles from problem/hardware "
                     "instead of relying on candidates-mn/k"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableRegisterLevelSchedule(
      "enable-register-level-schedule",
      llvm::cl::desc("Paper-aligned: include per-thread tiles (threadTileM/N) in "
                     "the candidate space (affects occupancy/waves estimate)"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> candidatesThreadMN(
      "candidates-thread-mn",
      llvm::cl::desc("Candidate per-thread tiles for M/N (csv), e.g. 1,2,4,8"),
      llvm::cl::init("1,2,4,8"));
  llvm::cl::opt<bool> requirePerfectTiling(
      "require-perfect-tiling",
      llvm::cl::desc("Require M%tm==0 && N%tn==0 && K%tk==0"),
      llvm::cl::init(true));
  llvm::cl::opt<bool> assumeFusedRelu(
      "assume-fused-relu",
      llvm::cl::desc("Assume MatMul->ReLU is fused in one kernel"),
      llvm::cl::init(true));
  llvm::cl::opt<bool> enableFootprintInference(
      "enable-footprint-inference",
      llvm::cl::desc("Use indexing_maps-based footprint inference to compute "
                     "traffic (experimental; keeps old hardcoded model as "
                     "fallback)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableTilePropagation(
      "enable-tile-propagation",
      llvm::cl::desc("Build TileGraph and run Welder-style consumer-driven tile "
                     "propagation (experimental; used to validate fusion "
                     "assumptions)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableGlobalTraffic(
      "enable-global-traffic",
      llvm::cl::desc("Phase A: compute whole-graph traffic assuming fully fused "
                     "(counts only graph-input reads and sink writes; "
                     "requires tile propagation; experimental)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableCutEdges(
      "enable-cut-edges",
      llvm::cl::desc("Phase 13A: allow cut-edges when tile propagation detects "
                     "conflicts; accounts extra global traffic in bytesCut "
                     "(experimental)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableTwoLevelSchedule(
      "enable-two-level-schedule",
      llvm::cl::desc(
          "Phase 14 (paper alignment, skeleton): 2-level (global<->shared) "
          "tile-graph scheduling. Computes whole-graph MemTraffic and filters "
          "candidates by estimated shared MemFootprint (experimental)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enablePaperSchedule(
      "enable-paper-schedule",
      llvm::cl::desc(
          "Welder paper (Figure 7): enable GraphConnecting + SubGraphTiling "
          "(uses traffic-based latency estimate instead of profiling; "
          "experimental)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> paperRecursiveRegisterLevel(
      "paper-recursive-register-level",
      llvm::cl::desc(
          "Paper-aligned: evaluate shared tiles via an inner register-level "
          "search (threadTileM/N + codegen knobs), and use the best inner config "
          "as the score for GraphConnecting and final ranking"),
      llvm::cl::init(true));
  llvm::cl::opt<int> paperRecursiveInnerMinLevelExclusive(
      "paper-recursive-inner-min-level-exclusive",
      llvm::cl::desc(
          "Recursive SubGraphTiling inner-stage boundary (minLevelExclusive). "
          "<=0 means auto(max(1,max-connect-level-1)); >0 forces an explicit "
          "boundary."),
      llvm::cl::init(0));
  llvm::cl::opt<int> paperRecursiveMaxStages(
      "paper-recursive-max-stages",
      llvm::cl::desc(
          "Cap recursive SubGraphTiling stage depth when max-connect-level>2. "
          "<=0 keeps legacy auto behavior; >0 allows at most N recursive "
          "stage windows."),
      llvm::cl::init(0));
  llvm::cl::opt<bool> paperStrict(
      "paper-strict",
      llvm::cl::desc("Paper-aligned strict mode: rank by MemTraffic bytes only "
                     "and prune configs that fail compile/profile"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> paperExpandReductionTile(
      "paper-expand-reduction-tile",
      llvm::cl::desc(
          "Paper-aligned: greedily enlarge reduction tiles (e.g. K) under shared "
          "memory constraints"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> pruneOnProfileFailure(
      "prune-on-profile-failure",
      llvm::cl::desc("When profiling is enabled, drop configs that fail "
                     "compile/profile instead of falling back to heuristics"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableCoalescingPenalty(
      "enable-coalescing-penalty",
      llvm::cl::desc("Paper-aligned: account for uncoalesced global memory access "
                     "by charging extra memory transactions in MemTraffic "
                     "(default: true)"),
      llvm::cl::init(true));
  llvm::cl::opt<int64_t> scheduleTopK(
      "schedule-topk",
      llvm::cl::desc("Top-K configs kept in SubGraphTiling (paper k)"),
      llvm::cl::init(8));
  llvm::cl::opt<int> maxConnectLevel(
      "max-connect-level",
      llvm::cl::desc(
          "Max connect level tried in GraphConnecting (0=cut to global, 1=shared "
          "reuse within one kernel, 2=register-level reuse model; default 1)"),
      llvm::cl::init(1));
  llvm::cl::opt<bool> enableProfiling(
      "enable-profiling",
      llvm::cl::desc("Enable paper-aligned hardware profiling (d.Profile)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableAsyncCopy(
      "profile-enable-async-copy",
      llvm::cl::desc("Enable async copy (cp.async) during profiling compilation"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableSoftwarePipelining(
      "profile-enable-software-pipelining",
      llvm::cl::desc("Enable software pipelining during profiling compilation "
                     "(requires multi-buffering)"),
      llvm::cl::init(false));
  llvm::cl::opt<int64_t> profilePipelineDepth(
      "profile-pipeline-depth",
      llvm::cl::desc("Pipeline depth for software pipelining"),
      llvm::cl::init(2));
  llvm::cl::opt<bool> profilePipelinePeelEpilogue(
      "profile-pipeline-peel-epilogue",
      llvm::cl::desc("Peel epilogue for pipelining"),
      llvm::cl::init(true));
  llvm::cl::opt<bool> profilePipelineSetAsyncWaitGroups(
      "profile-pipeline-set-async-wait-groups",
      llvm::cl::desc("When software pipelining is enabled, set cp.async "
                     "wait_group in-flight counts (emit wait_group N>0) instead "
                     "of the conservative wait_group 0 default"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileAsyncBypassL1(
      "profile-async-bypass-l1",
      llvm::cl::desc("Set bypass_l1 hint on eligible async copies"),
      llvm::cl::init(true));
  llvm::cl::opt<int64_t> profileWorkgroupMultiBufferDepth(
      "profile-workgroup-multibuffer-depth",
      llvm::cl::desc("Workgroup multi-buffer depth used during profiling compilation"),
      llvm::cl::init(1));
  llvm::cl::opt<int64_t> profileWorkgroupPadLastDim(
      "profile-workgroup-pad-last-dim",
      llvm::cl::desc(
          "Workgroup padding on the last dim (in elements) used during profiling "
          "compilation (0=off). For tensorcore schedules this is typically 8."),
      llvm::cl::init(0));
  llvm::cl::opt<bool> profileWorkgroupPadLastDimMatmulOnly(
      "profile-workgroup-pad-last-dim-matmul-only",
      llvm::cl::desc(
          "When workgroup padding is enabled during profiling compilation, apply it "
          "only to matmul operand shared/workgroup buffers (A/B) instead of all "
          "workgroup buffers"),
      llvm::cl::init(false));
  llvm::cl::opt<int64_t> profileWorkgroupSwizzleXor(
      "profile-workgroup-swizzle-xor",
      llvm::cl::desc("Workgroup XOR swizzle factor used during profiling compilation (0=off)"),
      llvm::cl::init(0));
  llvm::cl::opt<int> profileBlockRasterizeMode(
      "profile-block-rasterize-mode",
      llvm::cl::desc("Paper/Welder parity: 2D block rasterization mode used during profiling compilation (0=off, 1=row, 2=column)"),
      llvm::cl::init(0));
  llvm::cl::opt<int64_t> profileBlockRasterizePanelWidth(
      "profile-block-rasterize-panel-width",
      llvm::cl::desc("Panel width used for 2D block rasterization during profiling compilation"),
      llvm::cl::init(0));
  llvm::cl::opt<bool> profileSwapBlockDims(
      "profile-swap-block-dims",
      llvm::cl::desc(
          "Swap (x,y) block/thread mapping order during profiling compilation"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableTensorCoreF16(
      "profile-enable-tensorcore-f16",
      llvm::cl::desc("Enable TensorCore (f16) matmul path during profiling compilation"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionChainReuseFusion(
      "profile-enable-row-reduction-chain-reuse-fusion",
      llvm::cl::desc("Enable row-reduction chain reuse fusion fixups during profiling "
                     "compilation (keep 1D broadcast intermediates inside the fused "
                     "gpu.launch)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionInputPromotion(
      "profile-enable-row-reduction-input-promotion",
      llvm::cl::desc("Enable row-reduction input promotion during profiling compilation "
                     "(stage 2D input tile into workgroup memory once)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionInputPromotionVectorize(
      "profile-enable-row-reduction-input-promotion-vectorize",
      llvm::cl::desc("Enable vectorized cooperative staging for row-reduction input "
                     "promotion during profiling compilation (uses vector.transfer "
                     "when safe)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionWarp(
      "profile-enable-row-reduction-warp",
      llvm::cl::desc("Prefer warp-level reduction for row-reduction kernels during "
                     "profiling compilation (default: false)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionVectorize(
      "profile-enable-row-reduction-vectorize",
      llvm::cl::desc("Enable vectorization for row-reduction elementwise stages "
                     "during profiling compilation (default: false)"),
      llvm::cl::init(false));
  llvm::cl::opt<int64_t> profileRowReductionVectorWidth(
      "profile-row-reduction-vector-width",
      llvm::cl::desc(
          "Row-reduction elementwise vector width during profiling compilation "
          "(0=auto)"),
      llvm::cl::init(0));
  llvm::cl::opt<int64_t> profileRowReductionThreadsX(
      "profile-row-reduction-threads-x",
      llvm::cl::desc(
          "Row-reduction thread count along X during profiling compilation "
          "(0=auto)"),
      llvm::cl::init(0));
  llvm::cl::opt<bool> profileEnableRowReductionRelaxBarriers(
      "profile-enable-row-reduction-relax-barriers",
      llvm::cl::desc("Allow redundant barrier cleanup for row-reduction "
                     "staging during profiling compilation (default: false)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableRowReductionSkipCombineBarrier(
      "profile-enable-row-reduction-skip-combine-barrier",
      llvm::cl::desc("Skip barrier insertion after combining reductions during "
                     "profiling compilation (unsafe, default: false)"),
      llvm::cl::init(false));
  llvm::cl::opt<int64_t> profileRowReductionInputVectorWidth(
      "profile-row-reduction-input-vector-width",
      llvm::cl::desc("Row-reduction input staging vector width during profiling "
                     "compilation (0=auto)"),
      llvm::cl::init(0));
  llvm::cl::opt<bool> profileEnableRowReductionCombineVectorize(
      "profile-enable-row-reduction-combine-vectorize",
      llvm::cl::desc("Enable vectorization on row-reduction combining op "
                     "during profiling compilation (default: false)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> profileEnableMatmulSoftmaxSharedReuseFusion(
      "profile-enable-matmul-softmax-shared-reuse-fusion",
      llvm::cl::desc(
          "Enable Matmul->Softmax shared tile reuse fusion during profiling "
          "compilation (canonicalize matmul output reuse across max/exp/sum/div "
          "stages; reduces global traffic and avoids large local buffers)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableCodegenSearch(
      "codegen-search",
      llvm::cl::desc(
          "Paper-aligned: enumerate codegen knobs per candidate (async-copy, "
          "software pipelining, multi-buffering, padding, tensorcore, etc.)"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> codegenWorkgroupPadLastDim(
      "codegen-workgroup-pad-last-dim",
      llvm::cl::desc("CSV list for CodegenSearchSpace.workgroupPadLastDim (e.g. 0,1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenWorkgroupPadLastDimMatmulOnly(
      "codegen-workgroup-pad-last-dim-matmul-only",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.workgroupPadLastDimMatmulOnly (0/1). "
          "When enabled, padding applies only to matmul operand workgroup buffers."),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenWorkgroupMultiBufferDepth(
      "codegen-workgroup-multibuffer-depth",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.workgroupMultiBufferDepth (e.g. 1,2,3,4)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenWorkgroupSwizzleXor(
      "codegen-workgroup-swizzle-xor",
      llvm::cl::desc("CSV list for CodegenSearchSpace.workgroupSwizzleXor (e.g. 0,8)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenBlockRasterizeXor(
      "codegen-block-rasterize-xor",
      llvm::cl::desc("CSV list for CodegenSearchSpace.blockRasterizeXor (e.g. 0,4,8,16)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenBlockRasterizeMode(
      "codegen-block-rasterize-mode",
      llvm::cl::desc("CSV list for CodegenSearchSpace.blockRasterizeMode (0=off,1=row,2=col)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenBlockRasterizePanelWidth(
      "codegen-block-rasterize-panel-width",
      llvm::cl::desc("CSV list for CodegenSearchSpace.blockRasterizePanelWidth (e.g. 1,2,4,8,16)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenPipelineDepth(
      "codegen-pipeline-depth",
      llvm::cl::desc("CSV list for CodegenSearchSpace.pipelineDepth (e.g. 2,3,4)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableAsyncCopy(
      "codegen-enable-async-copy",
      llvm::cl::desc("CSV list for CodegenSearchSpace.enableAsyncCopy (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenAsyncBypassL1(
      "codegen-async-bypass-l1",
      llvm::cl::desc("CSV list for CodegenSearchSpace.asyncBypassL1 (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableSoftwarePipelining(
      "codegen-enable-software-pipelining",
      llvm::cl::desc("CSV list for CodegenSearchSpace.enableSoftwarePipelining (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenPipelinePeelEpilogue(
      "codegen-pipeline-peel-epilogue",
      llvm::cl::desc("CSV list for CodegenSearchSpace.pipelinePeelEpilogue (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenPipelineSetAsyncWaitGroups(
      "codegen-pipeline-set-async-wait-groups",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.pipelineSetAsyncWaitGroups (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableTensorCoreTf32(
      "codegen-enable-tensorcore-tf32",
      llvm::cl::desc("CSV list for CodegenSearchSpace.enableTensorCoreTf32 (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableTensorCoreF16(
      "codegen-enable-tensorcore-f16",
      llvm::cl::desc("CSV list for CodegenSearchSpace.enableTensorCoreF16 (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenSwapBlockDims(
      "codegen-swap-block-dims",
      llvm::cl::desc("CSV list for CodegenSearchSpace.swapBlockDims (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionChainReuseFusion(
      "codegen-enable-row-reduction-chain-reuse-fusion",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionChainReuseFusion (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionInputPromotion(
      "codegen-enable-row-reduction-input-promotion",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionInputPromotion (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionInputPromotionVectorize(
      "codegen-enable-row-reduction-input-promotion-vectorize",
      llvm::cl::desc("CSV list for CodegenSearchSpace."
                     "enableRowReductionInputPromotionVectorize (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionWarp(
      "codegen-enable-row-reduction-warp",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionWarp (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionVectorize(
      "codegen-enable-row-reduction-vectorize",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionVectorize (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenRowReductionVectorWidth(
      "codegen-row-reduction-vector-width",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.rowReductionVectorWidth (e.g., 0,2,4,8)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenRowReductionThreadsX(
      "codegen-row-reduction-threads-x",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.rowReductionThreadsX (e.g., 0,16,32)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionRelaxBarriers(
      "codegen-enable-row-reduction-relax-barriers",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionRelaxBarriers (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionSkipCombineBarrier(
      "codegen-enable-row-reduction-skip-combine-barrier",
      llvm::cl::desc(
          "CSV list for CodegenSearchSpace.enableRowReductionSkipCombineBarrier (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenRowReductionInputVectorWidth(
      "codegen-row-reduction-input-vector-width",
      llvm::cl::desc("CSV list for CodegenSearchSpace.rowReductionInputVectorWidth (e.g., 0,2,4,8)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableRowReductionCombineVectorize(
      "codegen-enable-row-reduction-combine-vectorize",
      llvm::cl::desc("CSV list for CodegenSearchSpace.enableRowReductionCombineVectorize (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> codegenEnableMatmulSoftmaxSharedReuseFusion(
      "codegen-enable-matmul-softmax-shared-reuse-fusion",
      llvm::cl::desc("CSV list for CodegenSearchSpace."
                     "enableMatmulSoftmaxSharedReuseFusion (0/1)"),
      llvm::cl::init(""));
  llvm::cl::opt<int> profileWarmup(
      "profile-warmup",
      llvm::cl::desc("CUDA event warmup launches (default 10)"),
      llvm::cl::init(10));
  llvm::cl::opt<int> profileIters(
      "profile-iters",
      llvm::cl::desc("CUDA event measured launches (default 100)"),
      llvm::cl::init(100));
  llvm::cl::opt<bool> profileRunAllKernels(
      "profile-run-all-kernels",
      llvm::cl::desc("Profile end-to-end by executing all gpu.launch_func "
                     "kernels in appearance order (welder-profiler "
                     "--run-all-kernels)"),
      llvm::cl::init(false));
  llvm::cl::opt<int> profileMaxParallelJobs(
      "profile-max-parallel-jobs",
      llvm::cl::desc("Max parallel profiling jobs (paper compilation speedup; "
                     "default 1)"),
      llvm::cl::init(1));
  llvm::cl::opt<int> profileTimeoutSec(
      "profile-timeout-sec",
      llvm::cl::desc("Timeout (seconds) for compile/profile per config "
                     "(default 0 = disabled; best-effort)"),
      llvm::cl::init(0));
  llvm::cl::opt<std::string> profileCompilerToNvvm(
      "profile-compiler-to-nvvm",
      llvm::cl::desc("Path to compiler/run_welder_to_nvvm_isa.sh"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> profileProfilerBin(
      "profile-profiler-bin",
      llvm::cl::desc("Path to welder-profiler binary"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> profileCachePath(
      "profile-cache",
      llvm::cl::desc("Append-only cache file for profile results"),
      llvm::cl::init(""));
  llvm::cl::opt<bool> profileVerbose(
      "profile-verbose",
      llvm::cl::desc("Verbose profiling logs (keeps temp artifacts)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> enableGenericProblem(
      "enable-generic-problem",
      llvm::cl::desc(
          "Use generic loop-based problem analysis/enumeration (experimental; "
          "allows non-matmul ops like conv; keeps old matmul path as default)"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> dumpCandidatesTsv(
      "dump-candidates-tsv",
      llvm::cl::desc(
          "Write sorted candidates to a TSV file (paper-style bench output)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpTileGraphJsonPath(
      "dump-tile-graph-json",
      llvm::cl::desc("Dump a raw Linalg TileGraph (JSON) to a file"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpTileGraphCompat(
      "dump-tile-graph",
      llvm::cl::desc("Alias for --dump-tile-graph-json"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpBestSummaryJsonPath(
      "dump-best-summary-json",
      llvm::cl::desc(
          "Dump a JSON summary of the best candidate (tile/cost/codegen knobs)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpPaperExecPlan(
      "dump-paper-exec-plan",
      llvm::cl::desc(
          "Dump a paper-aligned hierarchical execution plan (JSON) to a file "
          "(requires --enable-paper-schedule)"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpScheduleCompat(
      "dump-schedule-json",
      llvm::cl::desc("Alias for --dump-paper-exec-plan"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> dumpScheduleCompat2(
      "dump-schedule",
      llvm::cl::desc("Alias for --dump-paper-exec-plan"),
      llvm::cl::init(""));
  llvm::cl::opt<bool> trace(
      "trace",
      llvm::cl::desc("Real-time trace to stderr (stage timing + progress)"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> traceVerbose(
      "trace-verbose",
      llvm::cl::desc("Verbose trace (per-candidate/per-pass events)"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> traceFile(
      "trace-file",
      llvm::cl::desc("Write JSONL trace events to a file (one JSON object per line)"),
      llvm::cl::init(""));
  llvm::cl::opt<bool> traceFileAppend(
      "trace-file-append",
      llvm::cl::desc("Append to --trace-file instead of truncating"),
      llvm::cl::init(false));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "welder-solver (minimal)\n");

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
    tracerPtr->event("solver.main.start", std::move(f));
  }

  MLIRContext ctx;
  ctx.getOrLoadDialect<affine::AffineDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<bufferization::BufferizationDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<gpu::GPUDialect>();
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<tensor::TensorDialect>();

  OwningOpRef<ModuleOp> module;
  {
    [[maybe_unused]] auto parseSpan =
        tracerPtr ? tracerPtr->span("solver.parse_mlir",
                                    llvm::json::Object{{"path", inputFilename.getValue()}})
                  : welder::Tracer::Span();

    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << "error: cannot open input file: " << inputFilename << "\n";
      llvm::errs() << errorMessage << "\n";
      return 2;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  }
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR: " << inputFilename << "\n";
    return 2;
  }

  welder::SolveOptions opts;
  opts.arch.smemBytes = smemBytes;
  opts.arch.numSM = numSM;
  opts.arch.maxBlocksPerSM = maxBlocksPerSM;
  opts.arch.warpSize = warpSize;
  opts.arch.smPartition = smPartition;
  opts.arch.maxSmemUsageBytes = maxSmemUsageBytes;
  opts.arch.globalTransactionBytes = globalTransactionBytes;
  opts.arch.globalReadTransactionBytes = globalReadTransactionBytes;
  opts.arch.globalWriteTransactionBytes = globalWriteTransactionBytes;
  opts.arch.maxThreadsPerSM = maxThreadsPerSM;
  opts.arch.maxRegistersPerSM = maxRegistersPerSM;
  opts.arch.maxRegistersPerThread = maxRegistersPerThread;
  opts.candidatesMN = welder::parseCsvIntList(candidatesMN);
  opts.candidatesK = welder::parseCsvIntList(candidatesK);
  opts.autoCandidates = autoCandidates;
  opts.enableRegisterLevelSchedule = enableRegisterLevelSchedule;
  opts.candidatesThreadMN = welder::parseCsvIntList(candidatesThreadMN);
  opts.requirePerfectTiling = requirePerfectTiling;
  opts.assumeFusedRelu = assumeFusedRelu;
  opts.enableFootprintInference = enableFootprintInference;
  opts.enableTilePropagation = enableTilePropagation;
  opts.enableGlobalTraffic = enableGlobalTraffic;
  opts.enableCutEdges = enableCutEdges;
  opts.enableTwoLevelSchedule = enableTwoLevelSchedule;
  opts.enablePaperSchedule = enablePaperSchedule;
  opts.paperRecursiveRegisterLevel = paperRecursiveRegisterLevel;
  opts.paperRecursiveInnerMinLevelExclusive =
      paperRecursiveInnerMinLevelExclusive;
  opts.paperRecursiveMaxStages = paperRecursiveMaxStages;
  opts.paperStrict = paperStrict;
  opts.paperExpandReductionTile = paperExpandReductionTile;
  opts.pruneOnProfileFailure = pruneOnProfileFailure;
  opts.enableCoalescingPenalty = enableCoalescingPenalty;
  opts.scheduleTopK = scheduleTopK;
  opts.maxConnectLevel = maxConnectLevel;

  welder::applyPaperModeDefaults(opts,
                                 autoCandidates.getNumOccurrences() != 0,
                                 maxConnectLevel.getNumOccurrences() != 0,
                                 enableProfiling);
  opts.profile.enable = enableProfiling;
  opts.profile.warmup = profileWarmup;
  opts.profile.iters = profileIters;
  opts.profile.runAllKernels = profileRunAllKernels;
  opts.profile.maxParallelJobs = std::max(1, profileMaxParallelJobs.getValue());
  opts.profile.timeoutSec = std::max(0, profileTimeoutSec.getValue());
  opts.profile.cachePath = profileCachePath;
  opts.profile.verbose = profileVerbose;
  opts.profile.enableAsyncCopy = profileEnableAsyncCopy;
  opts.profile.enableSoftwarePipelining = profileEnableSoftwarePipelining;
  opts.profile.pipelineDepth = profilePipelineDepth;
  opts.profile.pipelinePeelEpilogue = profilePipelinePeelEpilogue;
  opts.profile.pipelineSetAsyncWaitGroups = profilePipelineSetAsyncWaitGroups;
  opts.profile.asyncBypassL1 = profileAsyncBypassL1;
  opts.profile.workgroupMultiBufferDepth = profileWorkgroupMultiBufferDepth;
  opts.profile.workgroupPadLastDim = profileWorkgroupPadLastDim;
  opts.profile.workgroupPadLastDimMatmulOnly =
      profileWorkgroupPadLastDimMatmulOnly;
  opts.profile.workgroupSwizzleXor = profileWorkgroupSwizzleXor;
  opts.profile.blockRasterizeMode = profileBlockRasterizeMode;
  opts.profile.blockRasterizePanelWidth =
      static_cast<int>(std::max<int64_t>(0, profileBlockRasterizePanelWidth));
  opts.profile.swapBlockDims = profileSwapBlockDims;
  opts.profile.enableTensorCoreF16 = profileEnableTensorCoreF16;
  opts.profile.enableRowReductionChainReuseFusion =
      profileEnableRowReductionChainReuseFusion;
  opts.profile.enableRowReductionInputPromotion =
      profileEnableRowReductionInputPromotion;
  opts.profile.enableRowReductionInputPromotionVectorize =
      profileEnableRowReductionInputPromotionVectorize;
  opts.profile.enableRowReductionWarp = profileEnableRowReductionWarp;
  opts.profile.enableRowReductionVectorize = profileEnableRowReductionVectorize;
  opts.profile.rowReductionVectorWidth = profileRowReductionVectorWidth;
  opts.profile.rowReductionThreadsX = profileRowReductionThreadsX;
  opts.profile.enableRowReductionRelaxBarriers =
      profileEnableRowReductionRelaxBarriers;
  opts.profile.enableRowReductionSkipCombineBarrier =
      profileEnableRowReductionSkipCombineBarrier;
  opts.profile.rowReductionInputVectorWidth =
      profileRowReductionInputVectorWidth;
  opts.profile.enableRowReductionCombineVectorize =
      profileEnableRowReductionCombineVectorize;
  opts.profile.enableMatmulSoftmaxSharedReuseFusion =
      profileEnableMatmulSoftmaxSharedReuseFusion;
  opts.tracer = tracerPtr;

  opts.codegenSearch.enable = enableCodegenSearch;
  if (opts.codegenSearch.enable) {
    auto parseBoolList = [&](const std::string &csv) -> std::vector<bool> {
      std::vector<bool> out;
      if (csv.empty())
        return out;
      bool sawFalse = false;
      bool sawTrue = false;
      std::string cur;
      auto flush = [&]() {
        if (cur.empty())
          return;
        std::string tok = std::move(cur);
        cur.clear();

    // 去除首尾空白。
        size_t b = tok.find_first_not_of(" \t\n\r");
        if (b == std::string::npos)
          return;
        size_t e = tok.find_last_not_of(" \t\n\r");
        tok = tok.substr(b, e - b + 1);
        if (tok.empty())
          return;

        auto lower = [&](const std::string &s) {
          std::string t = s;
          for (char &ch : t)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
          return t;
        };
        std::string t = lower(tok);

        if (t == "0" || t == "false" || t == "f" || t == "no" || t == "off") {
          sawFalse = true;
          return;
        }
        if (t == "1" || t == "true" || t == "t" || t == "yes" || t == "on") {
          sawTrue = true;
          return;
        }
    // 回退方案：按整数解析（非 0 即 true）。
        char *end = nullptr;
        long long v = std::strtoll(tok.c_str(), &end, 10);
        if (end && end != tok.c_str() && *end == '\0') {
          if (v == 0)
            sawFalse = true;
          else
            sawTrue = true;
        }
      };

      for (char ch : csv) {
        if (ch == ',' || ch == ' ' || ch == '\t' || ch == '\n') {
          flush();
          continue;
        }
        cur.push_back(ch);
      }
      flush();

      if (sawFalse)
        out.push_back(false);
      if (sawTrue)
        out.push_back(true);
      return out;
    };

    if (!codegenWorkgroupPadLastDim.empty())
      opts.codegenSearch.workgroupPadLastDim =
          welder::parseCsvIntList(codegenWorkgroupPadLastDim);
    if (!codegenWorkgroupPadLastDimMatmulOnly.empty())
      opts.codegenSearch.workgroupPadLastDimMatmulOnly =
          parseBoolList(codegenWorkgroupPadLastDimMatmulOnly);
    if (!codegenWorkgroupMultiBufferDepth.empty())
      opts.codegenSearch.workgroupMultiBufferDepth =
          welder::parseCsvIntList(codegenWorkgroupMultiBufferDepth);
    if (!codegenWorkgroupSwizzleXor.empty())
      opts.codegenSearch.workgroupSwizzleXor =
          welder::parseCsvIntList(codegenWorkgroupSwizzleXor);
    if (!codegenBlockRasterizeXor.empty())
      opts.codegenSearch.blockRasterizeXor =
          welder::parseCsvIntList(codegenBlockRasterizeXor);
    if (!codegenBlockRasterizeMode.empty()) {
      std::vector<int> xs;
      for (int64_t v : welder::parseCsvIntList(codegenBlockRasterizeMode))
        xs.push_back(static_cast<int>(v));
      xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
      opts.codegenSearch.blockRasterizeMode = std::move(xs);
    }
    if (!codegenBlockRasterizePanelWidth.empty()) {
      std::vector<int> xs;
      for (int64_t v : welder::parseCsvIntList(codegenBlockRasterizePanelWidth))
        xs.push_back(static_cast<int>(v));
      xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
      opts.codegenSearch.blockRasterizePanelWidth = std::move(xs);
    }
    if (!codegenPipelineDepth.empty())
      opts.codegenSearch.pipelineDepth = welder::parseCsvIntList(codegenPipelineDepth);

    if (!codegenEnableAsyncCopy.empty())
      opts.codegenSearch.enableAsyncCopy = parseBoolList(codegenEnableAsyncCopy);
    if (!codegenAsyncBypassL1.empty())
      opts.codegenSearch.asyncBypassL1 = parseBoolList(codegenAsyncBypassL1);
    if (!codegenEnableSoftwarePipelining.empty())
      opts.codegenSearch.enableSoftwarePipelining =
          parseBoolList(codegenEnableSoftwarePipelining);
    if (!codegenPipelinePeelEpilogue.empty())
      opts.codegenSearch.pipelinePeelEpilogue =
          parseBoolList(codegenPipelinePeelEpilogue);
    if (!codegenPipelineSetAsyncWaitGroups.empty())
      opts.codegenSearch.pipelineSetAsyncWaitGroups =
          parseBoolList(codegenPipelineSetAsyncWaitGroups);
    if (!codegenEnableTensorCoreTf32.empty())
      opts.codegenSearch.enableTensorCoreTf32 =
          parseBoolList(codegenEnableTensorCoreTf32);
    if (!codegenEnableTensorCoreF16.empty())
      opts.codegenSearch.enableTensorCoreF16 =
          parseBoolList(codegenEnableTensorCoreF16);
    if (!codegenSwapBlockDims.empty())
      opts.codegenSearch.swapBlockDims = parseBoolList(codegenSwapBlockDims);
    if (!codegenEnableRowReductionChainReuseFusion.empty())
      opts.codegenSearch.enableRowReductionChainReuseFusion =
          parseBoolList(codegenEnableRowReductionChainReuseFusion);
    if (!codegenEnableRowReductionInputPromotion.empty())
      opts.codegenSearch.enableRowReductionInputPromotion =
          parseBoolList(codegenEnableRowReductionInputPromotion);
    if (!codegenEnableRowReductionInputPromotionVectorize.empty())
      opts.codegenSearch.enableRowReductionInputPromotionVectorize =
          parseBoolList(codegenEnableRowReductionInputPromotionVectorize);
    if (!codegenEnableRowReductionWarp.empty())
      opts.codegenSearch.enableRowReductionWarp =
          parseBoolList(codegenEnableRowReductionWarp);
    if (!codegenEnableRowReductionVectorize.empty())
      opts.codegenSearch.enableRowReductionVectorize =
          parseBoolList(codegenEnableRowReductionVectorize);
    if (!codegenRowReductionVectorWidth.empty())
      opts.codegenSearch.rowReductionVectorWidth =
          welder::parseCsvIntList(codegenRowReductionVectorWidth);
    if (!codegenRowReductionThreadsX.empty())
      opts.codegenSearch.rowReductionThreadsX =
          welder::parseCsvIntList(codegenRowReductionThreadsX);
    if (!codegenEnableRowReductionRelaxBarriers.empty())
      opts.codegenSearch.enableRowReductionRelaxBarriers =
          parseBoolList(codegenEnableRowReductionRelaxBarriers);
    if (!codegenEnableRowReductionSkipCombineBarrier.empty())
      opts.codegenSearch.enableRowReductionSkipCombineBarrier =
          parseBoolList(codegenEnableRowReductionSkipCombineBarrier);
    if (!codegenRowReductionInputVectorWidth.empty())
      opts.codegenSearch.rowReductionInputVectorWidth =
          welder::parseCsvIntList(codegenRowReductionInputVectorWidth);
    if (!codegenEnableRowReductionCombineVectorize.empty())
      opts.codegenSearch.enableRowReductionCombineVectorize =
          parseBoolList(codegenEnableRowReductionCombineVectorize);
    if (!codegenEnableMatmulSoftmaxSharedReuseFusion.empty())
      opts.codegenSearch.enableMatmulSoftmaxSharedReuseFusion =
          parseBoolList(codegenEnableMatmulSoftmaxSharedReuseFusion);
  }

  welder::solver::ProfilerPathOverrides profilerPathOverrides;
  profilerPathOverrides.profilerBin = profileProfilerBin.getValue();
  profilerPathOverrides.compilerToNvvmScript = profileCompilerToNvvm.getValue();
  welder::solver::resolveProfilerToolPaths(opts, argv[0], profilerPathOverrides);

  std::string dumpTileGraphPath = dumpTileGraphJsonPath.getValue();
  if (dumpTileGraphPath.empty())
    dumpTileGraphPath = dumpTileGraphCompat.getValue();
  if (!dumpTileGraphJsonPath.getValue().empty() &&
      !dumpTileGraphCompat.getValue().empty() &&
      dumpTileGraphJsonPath.getValue() != dumpTileGraphCompat.getValue()) {
    llvm::errs() << "error: both --dump-tile-graph-json and --dump-tile-graph "
                    "were provided with different paths\n";
    return 2;
  }

  std::string dumpSchedulePath = dumpPaperExecPlan.getValue();
  if (dumpSchedulePath.empty())
    dumpSchedulePath = dumpScheduleCompat.getValue();
  if (dumpSchedulePath.empty())
    dumpSchedulePath = dumpScheduleCompat2.getValue();
  if (!dumpPaperExecPlan.getValue().empty()) {
    if ((!dumpScheduleCompat.getValue().empty() &&
         dumpPaperExecPlan.getValue() != dumpScheduleCompat.getValue()) ||
        (!dumpScheduleCompat2.getValue().empty() &&
         dumpPaperExecPlan.getValue() != dumpScheduleCompat2.getValue())) {
      llvm::errs() << "error: both --dump-paper-exec-plan and --dump-schedule "
                      "were provided with different paths\n";
      return 2;
    }
  }
  if (!dumpScheduleCompat.getValue().empty() &&
      !dumpScheduleCompat2.getValue().empty() &&
      dumpScheduleCompat.getValue() != dumpScheduleCompat2.getValue()) {
    llvm::errs() << "error: both --dump-schedule-json and --dump-schedule were "
                    "provided with different paths\n";
    return 2;
  }

  if (!dumpTileGraphPath.empty()) {
    if (!welder::dumpTileGraphJson(*module, opts, dumpTileGraphPath)) {
      llvm::errs() << "error: failed to dump tile graph json\n";
      return 2;
    }
  }

  auto dumpCandidates = [&](const welder::SolveResult &sr) {
    if (dumpCandidatesTsv.empty())
      return;
    if (!welder::solver::dumpCandidatesTsv(sr, dumpCandidatesTsv.getValue()))
      llvm::errs() << "error: failed to dump candidates tsv\n";
  };

  welder::SolveResult solveRes;
  if (enableGenericProblem) {
    auto genericOpt = welder::analyzeGenericProblem(*module);
    if (!genericOpt) {
      llvm::errs()
          << "error: cannot find a static-shape linalg op for generic analysis "
             "in "
          << inputFilename << "\n";
      return 2;
    }

    welder::solver::SearchFailure genericFailure;
    if (!welder::solver::runGenericCandidateSearch(*module, opts, solveRes,
                                                   &genericFailure)) {
      llvm::errs() << genericFailure.message << "\n";
      if (!genericFailure.hint.empty())
        llvm::errs() << genericFailure.hint << "\n";
      return 2;
    }

    const welder::Candidate &best = solveRes.sortedCandidates.front();

    llvm::outs() << "GenericProblem:\n";
    llvm::outs() << "  - op=" << genericOpt->getOpName() << "\n";
    llvm::outs() << "  - loops=[";
    for (size_t i = 0; i < genericOpt->loops.size(); ++i) {
      if (i)
        llvm::outs() << ", ";
      const auto &d = genericOpt->loops[i];
      const char *kind = "other";
      if (d.type == mlir::utils::IteratorType::parallel)
        kind = "parallel";
      else if (d.type == mlir::utils::IteratorType::reduction)
        kind = "reduction";
      llvm::outs() << "(" << kind << "," << d.size << ")";
    }
    llvm::outs() << "]\n";
    llvm::outs() << "\n";

    llvm::outs() << "Best candidate:\n";
    welder::solver::printCandidateCostSummary(best, llvm::outs());
    llvm::outs() << "BEST_TILE tile_m=" << best.tileM << " tile_n=" << best.tileN
                 << " tile_k=" << best.tileK;
    if (best.threadTileM > 0 && best.threadTileN > 0) {
      llvm::outs() << " thread_tile_m=" << best.threadTileM
                   << " thread_tile_n=" << best.threadTileN;
    }
    llvm::outs() << "\n";
    if (!best.loopTileExtents.empty()) {
      llvm::outs() << "BEST_LOOP_TILE loop_tile_extents=";
      welder::solver::printIntList(best.loopTileExtents, llvm::outs());
      llvm::outs() << "\n";
    }

    llvm::outs() << "\nTop-5 candidates:\n";
    welder::solver::printTopKCandidateScores(solveRes, /*k=*/5, llvm::outs());

		    dumpCandidates(solveRes);
    if (!dumpBestSummaryJsonPath.getValue().empty()) {
      if (!welder::solver::dumpBestSummaryJson(
              solveRes, opts, dumpBestSummaryJsonPath.getValue()))
        return 2;
    }
		    if (!dumpSchedulePath.empty()) {
		      if (!opts.enablePaperSchedule) {
		        llvm::errs()
		            << "error: --dump-paper-exec-plan/--dump-schedule requires "
		               "--enable-paper-schedule\n";
		        return 2;
		      }
		      if (!welder::dumpPaperExecutionPlan(*module, opts, dumpSchedulePath)) {
		        llvm::errs() << "error: failed to dump paper exec plan\n";
		        return 2;
		      }
		    }
		    return 0;
		  }

  auto probOpt = welder::analyzeMatmulProblem(*module);
  if (!probOpt.has_value()) {
    llvm::errs() << "error: cannot find a static-shape linalg.matmul (2D) in "
                 << inputFilename << "\n";
    llvm::errs() << "hint: try --enable-generic-problem to run the generic "
                    "loop-based solver.\n";
    return 2;
  }

  bool inferredFused = welder::detectMatmulConsumerChain(*module);
  if (assumeFusedRelu && !inferredFused) {
    llvm::errs()
        << "note: --assume-fused-relu=true, but a MatMul->Generic consumer chain "
           "was not detected; continuing with fused=true anyway.\n";
  }
  if (!assumeFusedRelu && inferredFused) {
    llvm::errs() << "note: detected MatMul->Generic consumer chain; "
                    "you may want --assume-fused-relu=true.\n";
  }

  welder::solver::SearchFailure matmulFailure;
  if (!welder::solver::runMatmulCandidateSearch(*module, opts, solveRes,
                                                &matmulFailure)) {
    llvm::errs() << matmulFailure.message << "\n";
    if (!matmulFailure.hint.empty())
      llvm::errs() << matmulFailure.hint << "\n";
    return 2;
  }

  if (!dumpSchedulePath.empty()) {
    if (!opts.enablePaperSchedule) {
      llvm::errs()
          << "error: --dump-paper-exec-plan/--dump-schedule requires "
             "--enable-paper-schedule\n";
      return 2;
    }
    if (!welder::dumpPaperExecutionPlan(*module, opts, dumpSchedulePath)) {
      llvm::errs() << "error: failed to dump paper exec plan\n";
      return 2;
    }
  }

  const welder::ProblemSize prob = solveRes.problem;
  const welder::Candidate &best = solveRes.sortedCandidates.front();

  llvm::outs() << "Problem: M=" << prob.m << " N=" << prob.n << " K=" << prob.k
               << "\n";
  llvm::outs() << "Assumptions:\n";
  llvm::outs() << "  - fused_relu=" << (assumeFusedRelu ? "true" : "false")
               << "\n";
  llvm::outs() << "  - require_perfect_tiling="
               << (requirePerfectTiling ? "true" : "false") << "\n";
	  llvm::outs() << "Arch (approx): smem_bytes=" << opts.arch.smemBytes
	               << " num_sm=" << opts.arch.numSM
	               << " max_blocks_per_sm=" << opts.arch.maxBlocksPerSM
	               << " max_threads_per_sm=" << opts.arch.maxThreadsPerSM
	               << " max_regs_per_sm=" << opts.arch.maxRegistersPerSM << "\n";
  llvm::outs() << "\n";

  llvm::outs() << "Best candidate:\n";
  welder::solver::printCandidateCostSummary(best, llvm::outs());
  llvm::outs() << "BEST_TILE tile_m=" << best.tileM << " tile_n=" << best.tileN
               << " tile_k=" << best.tileK;
  if (best.threadTileM > 0 && best.threadTileN > 0) {
    llvm::outs() << " thread_tile_m=" << best.threadTileM
                 << " thread_tile_n=" << best.threadTileN;
  }
  llvm::outs() << "\n";

  llvm::outs() << "\nTransform suggestion (for your scripts):\n";
  llvm::outs() << "  - L1 block tile: tile_sizes [" << best.tileM << ", "
               << best.tileN << ", 0]\n";
  llvm::outs() << "  - K tile:        tile_sizes [0, 0, " << best.tileK
               << "]\n";

  llvm::outs() << "\nTop-5 candidates:\n";
  welder::solver::printTopKCandidateScores(solveRes, /*k=*/5, llvm::outs());

  dumpCandidates(solveRes);
  if (!dumpBestSummaryJsonPath.getValue().empty()) {
    if (!welder::solver::dumpBestSummaryJson(
            solveRes, opts, dumpBestSummaryJsonPath.getValue()))
      return 2;
  }
  return 0;
}

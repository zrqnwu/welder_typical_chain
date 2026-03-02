#include "WelderSolverReportWriter.h"

#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace {

static const char *feasibilityCodeToString(int32_t code) {
  switch (code) {
  case 0:
    return "ok";
  case 1:
    return "tc_alignment";
  case 2:
    return "tc_layout_stride";
  case 3:
    return "tc_block_threads";
  case 10:
    return "swizzle_inapplicable";
  case 11:
    return "raster_invalid";
  case 12:
    return "raster_conflict";
  case 13:
    return "pipeline_invalid";
  case 20:
    return "reg_overflow";
  case 21:
    return "pipeline_low_occ_reg";
  case 22:
    return "pipeline_tc_occ_pressure";
  case 23:
    return "mm_sm_row_norm";
  case 24:
    return "mm_sm_pipe_spill_guard";
  case 25:
    return "mm_sm_async_stability_guard";
  default:
    return "unknown";
  }
}

} // namespace

namespace welder::solver {

bool dumpBestSummaryJson(const welder::SolveResult &sr,
                         const welder::SolveOptions &opts,
                         const std::string &path) {
  if (path.empty())
    return true;
  if (sr.sortedCandidates.empty()) {
    llvm::errs() << "error: dumpBestSummaryJson: empty candidate list\n";
    return false;
  }
  const welder::Candidate &best = sr.sortedCandidates.front();

  std::ofstream ofs(path);
  if (!ofs) {
    llvm::errs() << "error: dumpBestSummaryJson: cannot open output: " << path
                 << "\n";
    return false;
  }

  double profiled =
      best.cost.profiledMs.has_value() ? *best.cost.profiledMs : -1.0;

  ofs << "{\n";
  ofs << "  \"problem\": {\"m\": " << sr.problem.m << ", \"n\": " << sr.problem.n
      << ", \"k\": " << sr.problem.k << "},\n";
  ofs << "  \"arch\": {\"smemBytes\": " << opts.arch.smemBytes
      << ", \"numSM\": " << opts.arch.numSM
      << ", \"warpSize\": " << opts.arch.warpSize
      << ", \"elementBytes\": " << opts.arch.elementBytes << "},\n";
  ofs << "  \"best\": {";
  ofs << "\"tileM\": " << best.tileM << ", \"tileN\": " << best.tileN
      << ", \"tileK\": " << best.tileK << ", \"threadTileM\": " << best.threadTileM
      << ", \"threadTileN\": " << best.threadTileN << ", ";
  ofs << "\"estRegsPerThread\": " << best.estRegsPerThread << ", ";
  ofs << "\"estFootprintBytes\": " << best.estFootprintBytes << ", ";
  ofs << "\"traffic\": {\"bytesA\": " << best.traffic.bytesA
      << ", \"bytesB\": " << best.traffic.bytesB
      << ", \"bytesC\": " << best.traffic.bytesC
      << ", \"bytesCut\": " << best.traffic.bytesCut << "}, ";
  ofs << "\"cost\": {\"waves\": " << best.cost.waves
      << ", \"blocksTotal\": " << best.cost.blocksTotal
      << ", \"blocksPerSM\": " << best.cost.blocksPerSM
      << ", \"sharedFootprintBytes\": " << best.cost.sharedFootprintBytes
      << ", \"sharedToRegBytes\": " << best.cost.sharedToRegBytes
      << ", \"estimatedLatency\": " << best.cost.estimatedLatency
      << ", \"profiledMs\": " << profiled << "}, ";

  ofs << "\"codegen\": {"
      << "\"enableAsyncCopy\": " << (best.enableAsyncCopy ? "true" : "false")
      << ", \"asyncBypassL1\": " << (best.asyncBypassL1 ? "true" : "false")
      << ", \"enableSoftwarePipelining\": "
      << (best.enableSoftwarePipelining ? "true" : "false")
      << ", \"pipelineDepth\": " << best.pipelineDepth
      << ", \"pipelinePeelEpilogue\": "
      << (best.pipelinePeelEpilogue ? "true" : "false")
      << ", \"pipelineSetAsyncWaitGroups\": "
      << (best.pipelineSetAsyncWaitGroups ? "true" : "false")
      << ", \"workgroupMultiBufferDepth\": " << best.workgroupMultiBufferDepth
      << ", \"workgroupPadLastDim\": " << best.workgroupPadLastDim
      << ", \"workgroupPadLastDimMatmulOnly\": "
      << (best.workgroupPadLastDimMatmulOnly ? "true" : "false")
      << ", \"workgroupSwizzleXor\": " << best.workgroupSwizzleXor
      << ", \"blockRasterizeXor\": " << best.blockRasterizeXor
      << ", \"blockRasterizeMode\": " << best.blockRasterizeMode
      << ", \"blockRasterizePanelWidth\": " << best.blockRasterizePanelWidth
      << ", \"swapBlockDims\": " << (best.swapBlockDims ? "true" : "false")
      << ", \"enableRowReductionChainReuseFusion\": "
      << (best.enableRowReductionChainReuseFusion ? "true" : "false")
      << ", \"enableRowReductionInputPromotion\": "
      << (best.enableRowReductionInputPromotion ? "true" : "false")
      << ", \"enableRowReductionInputPromotionVectorize\": "
      << (best.enableRowReductionInputPromotionVectorize ? "true" : "false")
      << ", \"enableRowReductionWarp\": "
      << (best.enableRowReductionWarp ? "true" : "false")
      << ", \"enableRowReductionVectorize\": "
      << (best.enableRowReductionVectorize ? "true" : "false")
      << ", \"rowReductionVectorWidth\": " << best.rowReductionVectorWidth
      << ", \"rowReductionThreadsX\": " << best.rowReductionThreadsX
      << ", \"enableRowReductionRelaxBarriers\": "
      << (best.enableRowReductionRelaxBarriers ? "true" : "false")
      << ", \"enableRowReductionSkipCombineBarrier\": "
      << (best.enableRowReductionSkipCombineBarrier ? "true" : "false")
      << ", \"rowReductionInputVectorWidth\": "
      << best.rowReductionInputVectorWidth
      << ", \"enableRowReductionCombineVectorize\": "
      << (best.enableRowReductionCombineVectorize ? "true" : "false")
      << ", \"enableMatmulSoftmaxSharedReuseFusion\": "
      << (best.enableMatmulSoftmaxSharedReuseFusion ? "true" : "false")
      << ", \"enableTensorCoreTf32\": "
      << (best.enableTensorCoreTf32 ? "true" : "false")
      << ", \"enableTensorCoreF16\": "
      << (best.enableTensorCoreF16 ? "true" : "false")
      << ", \"useCutlassMma\": " << (best.useCutlassMma ? "true" : "false")
      << ", \"mmaM\": " << best.mmaM
      << ", \"mmaN\": " << best.mmaN
      << ", \"mmaK\": " << best.mmaK
      << "}";

  ofs << "}\n";
  ofs << "}\n";
  return true;
}

bool dumpCandidatesTsv(const welder::SolveResult &sr,
                       const std::string &path) {
  if (path.empty())
    return true;
  std::ofstream out(path);
  if (!out) {
    llvm::errs() << "error: cannot open dump file: " << path << "\n";
    return false;
  }
  out << "rank\t"
      << "tileM\ttileN\ttileK\tthreadTileM\tthreadTileN\t"
      << "estRegsPerThread\t"
      << "estSharedBankConflict\t"
      << "enableAsyncCopy\tasyncBypassL1\t"
      << "enableSoftwarePipelining\tpipelineDepth\tpipelinePeelEpilogue\tpipelineSetAsyncWaitGroups\t"
      << "workgroupMultiBufferDepth\tworkgroupPadLastDim\tworkgroupPadLastDimMatmulOnly\t"
      << "workgroupSwizzleXor\t"
      << "blockRasterizeXor\tblockRasterizeMode\tblockRasterizePanelWidth\t"
      << "swapBlockDims\t"
      << "enableRowReductionChainReuseFusion\t"
      << "enableRowReductionInputPromotion\t"
      << "enableRowReductionInputPromotionVectorize\t"
      << "enableRowReductionWarp\t"
      << "enableRowReductionVectorize\t"
      << "rowReductionVectorWidth\t"
      << "rowReductionThreadsX\t"
      << "enableRowReductionRelaxBarriers\t"
      << "enableRowReductionSkipCombineBarrier\t"
      << "rowReductionInputVectorWidth\t"
      << "enableRowReductionCombineVectorize\t"
      << "enableMatmulSoftmaxSharedReuseFusion\t"
      << "enableTensorCoreTf32\t"
      << "enableTensorCoreF16\t"
      << "useCutlassMma\tmmaM\tmmaN\tmmaK\t"
      << "feasibilityCode\t"
      << "feasibilityReason\t"
      << "estFootprintBytes\tbytesA\tbytesB\tbytesC\tbytesCut\t"
      << "waves\tblocksTotal\tblocksPerSM\t"
      << "sharedToRegBytes\t"
      << "estimatedLatency\tprofiledMs\t"
      << "score\n";

  for (size_t i = 0; i < sr.sortedCandidates.size(); ++i) {
    const welder::Candidate &c = sr.sortedCandidates[i];
    double profiled = c.cost.profiledMs.has_value() ? *c.cost.profiledMs : -1.0;
    out << i << "\t" << c.tileM << "\t" << c.tileN << "\t" << c.tileK << "\t"
        << c.threadTileM << "\t" << c.threadTileN << "\t"
        << c.estRegsPerThread << "\t"
        << c.estSharedBankConflict << "\t"
        << (c.enableAsyncCopy ? 1 : 0) << "\t" << (c.asyncBypassL1 ? 1 : 0)
        << "\t" << (c.enableSoftwarePipelining ? 1 : 0) << "\t"
        << c.pipelineDepth << "\t" << (c.pipelinePeelEpilogue ? 1 : 0) << "\t"
        << (c.pipelineSetAsyncWaitGroups ? 1 : 0) << "\t"
        << c.workgroupMultiBufferDepth << "\t" << c.workgroupPadLastDim << "\t"
        << (c.workgroupPadLastDimMatmulOnly ? 1 : 0) << "\t"
        << c.workgroupSwizzleXor << "\t"
        << c.blockRasterizeXor << "\t" << c.blockRasterizeMode << "\t"
        << c.blockRasterizePanelWidth << "\t"
        << (c.swapBlockDims ? 1 : 0) << "\t"
        << (c.enableRowReductionChainReuseFusion ? 1 : 0) << "\t"
        << (c.enableRowReductionInputPromotion ? 1 : 0) << "\t"
        << (c.enableRowReductionInputPromotionVectorize ? 1 : 0) << "\t"
        << (c.enableRowReductionWarp ? 1 : 0) << "\t"
        << (c.enableRowReductionVectorize ? 1 : 0) << "\t"
        << c.rowReductionVectorWidth << "\t"
        << c.rowReductionThreadsX << "\t"
        << (c.enableRowReductionRelaxBarriers ? 1 : 0) << "\t"
        << (c.enableRowReductionSkipCombineBarrier ? 1 : 0) << "\t"
        << c.rowReductionInputVectorWidth << "\t"
        << (c.enableRowReductionCombineVectorize ? 1 : 0) << "\t"
        << (c.enableMatmulSoftmaxSharedReuseFusion ? 1 : 0) << "\t"
        << (c.enableTensorCoreTf32 ? 1 : 0) << "\t"
        << (c.enableTensorCoreF16 ? 1 : 0) << "\t"
        << (c.useCutlassMma ? 1 : 0) << "\t"
        << c.mmaM << "\t" << c.mmaN << "\t" << c.mmaK << "\t"
        << c.feasibilityCode << "\t"
        << feasibilityCodeToString(c.feasibilityCode) << "\t"
        << c.estFootprintBytes
        << "\t" << c.traffic.bytesA << "\t" << c.traffic.bytesB << "\t"
        << c.traffic.bytesC << "\t" << c.traffic.bytesCut << "\t"
        << c.numWave << "\t" << c.blocksTotal << "\t" << c.blocksPerSM << "\t"
        << c.cost.sharedToRegBytes << "\t"
        << c.cost.estimatedLatency << "\t" << profiled << "\t"
        << c.score << "\n";
  }
  return true;
}

} // namespace welder::solver

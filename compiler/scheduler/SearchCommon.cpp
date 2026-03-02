#include "wtc/scheduler/internal/SearchInternal.h"

#include "wtc/backend/Toolchain.h"

#include <sstream>

namespace wtc::scheduler::internal {

std::string buildCommonSolverFlags(const std::string &inputPath,
                                   const wtc::scheduler::SearchConfig &config,
                                   const std::filesystem::path &bestSummaryJson,
                                   const std::filesystem::path &candidatesTsv) {
  std::ostringstream args;
  args << wtc::backend::shellQuote(inputPath)
       << " --enable-generic-problem"
       << " --enable-tile-propagation"
       << " --enable-cut-edges"
       << " --enable-two-level-schedule"
       << " --require-perfect-tiling=false"
       << " --enable-register-level-schedule"
       << " --max-connect-level " << config.maxConnectLevel
       << " --dump-best-summary-json "
       << wtc::backend::shellQuote(bestSummaryJson.string())
       << " --dump-candidates-tsv "
       << wtc::backend::shellQuote(candidatesTsv.string());
  return args.str();
}

} // namespace wtc::scheduler::internal

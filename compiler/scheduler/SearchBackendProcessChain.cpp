#include "wtc/scheduler/internal/SearchInternal.h"

#include "wtc/backend/Toolchain.h"

#include <filesystem>
#include <sstream>

namespace wtc::scheduler::internal {

bool runSearchByProcessChain(const std::string &inputPath,
                             const wtc::scheduler::SearchConfig &config,
                             const std::filesystem::path &bestSummaryJson,
                             const std::filesystem::path &candidatesTsv,
                             const std::filesystem::path &solverLog,
                             std::string &diagnostic) {
  wtc::backend::ToolchainPaths tc;
  if (!wtc::backend::resolveToolchain(config.backendRoot, tc, diagnostic))
    return false;
  if (!wtc::backend::ensureBackendCompilerTarget(tc, "welder-solver",
                                                 config.verbose, diagnostic)) {
    return false;
  }

  std::ostringstream cmd;
  cmd << wtc::backend::shellQuote(tc.welderSolverBin.string()) << " "
      << buildCommonSolverFlags(inputPath, config, bestSummaryJson, candidatesTsv);

  int rc = wtc::backend::runShellCommand(
      cmd.str(), solverLog.string(), /*appendLog=*/false, config.verbose,
      &diagnostic);
  if (rc != 0) {
    diagnostic =
        "backend solver(process_chain) failed, rc=" + std::to_string(rc) +
                 ", see " + solverLog.string();
    return false;
  }
  return true;
}

} // namespace wtc::scheduler::internal

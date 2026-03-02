#include "wtc/scheduler/internal/SearchInternal.h"

#include "wtc/backend/Toolchain.h"

#include <filesystem>
#include <sstream>

namespace wtc::scheduler::internal {

bool runSearchByShellScript(const std::string &inputPath,
                            const wtc::scheduler::SearchConfig &config,
                            const std::filesystem::path &bestSummaryJson,
                            const std::filesystem::path &candidatesTsv,
                            const std::filesystem::path &solverLog,
                            std::string &diagnostic) {
  const std::filesystem::path solverScript =
      std::filesystem::path(config.backendRoot) / "compiler" /
      "run_welder_solver.sh";
  if (!std::filesystem::exists(solverScript)) {
    diagnostic = "solver script not found: " + solverScript.string();
    return false;
  }

  std::ostringstream cmd;
  wtc::backend::ToolchainPaths tc;
  std::string tcDiag;
  if (wtc::backend::resolveToolchain(config.backendRoot, tc, tcDiag) &&
      !tc.llvmBuildDir.empty()) {
    cmd << "LLVM_BUILD=" << wtc::backend::shellQuote(tc.llvmBuildDir.string())
        << " ";
  }

  cmd << "bash " << wtc::backend::shellQuote(solverScript.string()) << " "
      << buildCommonSolverFlags(inputPath, config, bestSummaryJson, candidatesTsv);

  int rc = wtc::backend::runShellCommand(
      cmd.str(), solverLog.string(), /*appendLog=*/false, config.verbose,
      &diagnostic);
  if (rc != 0) {
    diagnostic = "backend solver(shell) failed, rc=" + std::to_string(rc) +
                 ", see " + solverLog.string();
    return false;
  }
  return true;
}

} // namespace wtc::scheduler::internal

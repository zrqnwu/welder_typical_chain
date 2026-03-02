#include "wtc/transform/internal/BuildCutEdgesTransformInternal.h"

#include "wtc/backend/Toolchain.h"

#include <filesystem>
#include <sstream>

namespace wtc::transform::internal {

bool runCompileByShellScript(const std::string &inputPath,
                             const wtc::scheduler::SearchResult &search,
                             const wtc::transform::BuildConfig &config,
                             const std::filesystem::path &outDir,
                             const std::filesystem::path &compileLog,
                             std::string &diagnostic) {
  const std::filesystem::path compileScript =
      std::filesystem::path(config.backendRoot) / "compiler" /
      "run_welder_to_nvvm_isa.sh";
  if (!std::filesystem::exists(compileScript)) {
    diagnostic = "compile script not found: " + compileScript.string();
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
  cmd << "OUT_DIR=" << wtc::backend::shellQuote(outDir.string())
      << " bash " << wtc::backend::shellQuote(compileScript.string()) << " "
      << buildCommonWelderCompilerFlags(inputPath, search, config);

  int rc = wtc::backend::runShellCommand(
      cmd.str(), compileLog.string(), /*appendLog=*/false, config.verbose,
      &diagnostic);
  if (rc != 0) {
    diagnostic = "backend compile(shell) failed, rc=" + std::to_string(rc) +
                 ", see " + compileLog.string();
    return false;
  }
  return true;
}

} // namespace wtc::transform::internal

#include "wtc/transform/internal/BuildCutEdgesTransformInternal.h"
#include "wtc/transform/internal/CompileApiSession.h"

#include "wtc/backend/Toolchain.h"

#include <filesystem>
#include <fstream>

namespace wtc::transform::internal {

bool runCompileByBackendApi(const std::string &inputPath,
                            const wtc::scheduler::SearchResult &search,
                            const wtc::transform::BuildConfig &config,
                            const std::filesystem::path &outDir,
                            const std::filesystem::path &compileLog,
                            std::string &diagnostic) {
  wtc::backend::ToolchainPaths tc;
  if (!wtc::backend::resolveToolchain(config.backendRoot, tc, diagnostic))
    return false;
  if (!wtc::backend::ensureBackendCompilerTarget(tc, "welder-compile-capi",
                                                 config.verbose, diagnostic)) {
    return false;
  }
  if (!wtc::backend::ensureBackendCompilerTarget(tc, "welder-compiler",
                                                 config.verbose, diagnostic)) {
    return false;
  }
  if (!wtc::backend::ensureWorkgroupPassPlugin(tc, config.verbose, diagnostic))
    return false;

  if (tc.welderCompileCapiLib.empty() ||
      !std::filesystem::exists(tc.welderCompileCapiLib)) {
    diagnostic =
        "compile C API library not found under: " + tc.backendBuildDir.string();
    return false;
  }

  if (config.verbose) {
    std::ofstream ofs(compileLog, std::ios::trunc);
    ofs << "[wtc.compile.api] lib=" << tc.welderCompileCapiLib.string() << "\n";
    ofs << "[wtc.compile.api] input=" << inputPath << "\n";
    ofs << "[wtc.compile.api] plugin=" << tc.passPluginLib.string() << "\n";
    ofs << "[wtc.compile.api] compiler_bin=" << tc.welderCompilerBin.string()
        << "\n";
  }

  const std::string outDirStr = outDir.string();
  if (!CompileApiSession::instance().compile(
          tc.welderCompileCapiLib, inputPath, outDirStr,
          tc.passPluginLib.string(), tc.welderCompilerBin.string(), search.tileM,
          search.tileN, search.tileK, search.threadTileM, search.threadTileN,
          config.maxConnectLevel, config.fused, config.verbose, diagnostic)) {
    diagnostic += ", see " + compileLog.string();
    return false;
  }
  return true;
}

} // namespace wtc::transform::internal

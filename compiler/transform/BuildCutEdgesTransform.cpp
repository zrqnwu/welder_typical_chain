#include "wtc/transform/BuildCutEdgesTransform.h"

#include "wtc/transform/internal/BuildCutEdgesTransformInternal.h"

#include <filesystem>

namespace wtc::transform {

bool buildCutEdgesTransformModule(const std::string &inputPath,
                                  const wtc::scheduler::SearchResult &search,
                                  const BuildConfig &config,
                                  BuildArtifacts &artifacts,
                                  std::string &diagnostic) {
  if (inputPath.empty()) {
    diagnostic = "input path is empty";
    return false;
  }
  if (config.backendRoot.empty()) {
    diagnostic = "backend root is empty";
    return false;
  }
  if (search.tileM <= 0 || search.tileN <= 0 || search.tileK <= 0 ||
      search.threadTileM <= 0 || search.threadTileN <= 0) {
    diagnostic = "invalid tile sizes";
    return false;
  }

  std::filesystem::path outDir = config.outDir;
  if (outDir.empty())
    outDir = std::filesystem::path("/tmp") / "wtc_compile";
  std::filesystem::create_directories(outDir);

  const std::filesystem::path compileLog = outDir / "compile.log";
  const std::filesystem::path runnable = outDir / "05.out.nvvm.runnable.mlir";

  bool ok = false;
  if (config.backendMode == wtc::backend::BackendMode::Shell) {
    ok = internal::runCompileByShellScript(inputPath, search, config, outDir,
                                           compileLog, diagnostic);
  } else if (config.backendMode == wtc::backend::BackendMode::Api) {
    ok = internal::runCompileByBackendApi(inputPath, search, config, outDir,
                                          compileLog, diagnostic);
  } else {
    ok = internal::runCompileByProcessChain(inputPath, search, config, outDir,
                                            compileLog, diagnostic);
  }
  if (!ok)
    return false;

  if (!std::filesystem::exists(runnable)) {
    diagnostic = "compile output missing: " + runnable.string();
    return false;
  }

  artifacts.outDir = outDir.string();
  artifacts.compileLogPath = compileLog.string();
  artifacts.runnableMlirPath = runnable.string();
  diagnostic.clear();
  return true;
}

} // namespace wtc::transform

#ifndef WTC_TRANSFORM_BUILDCUTEDGESTRANSFORM_H
#define WTC_TRANSFORM_BUILDCUTEDGESTRANSFORM_H

#include "wtc/backend/Toolchain.h"
#include "wtc/scheduler/Search.h"

#include <cstdint>
#include <string>

namespace wtc::transform {

struct BuildConfig {
  std::string backendRoot;
  std::string outDir;
  int64_t maxConnectLevel = 1;
  bool fused = true;
  wtc::backend::BackendMode backendMode =
      wtc::backend::BackendMode::ProcessChain;
  bool verbose = false;
};

struct BuildArtifacts {
  std::string outDir;
  std::string compileLogPath;
  std::string runnableMlirPath;
};

bool buildCutEdgesTransformModule(const std::string &inputPath,
                                  const wtc::scheduler::SearchResult &search,
                                  const BuildConfig &config, BuildArtifacts &artifacts,
                                  std::string &diagnostic);

} // namespace wtc::transform

#endif

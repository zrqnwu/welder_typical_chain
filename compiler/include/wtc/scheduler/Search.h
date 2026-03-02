#ifndef WTC_SCHEDULER_SEARCH_H
#define WTC_SCHEDULER_SEARCH_H

#include "wtc/backend/Toolchain.h"

#include <cstdint>
#include <string>

namespace wtc::scheduler {

struct SearchResult {
  int64_t tileM = 32;
  int64_t tileN = 64;
  int64_t tileK = 16;
  int64_t threadTileM = 2;
  int64_t threadTileN = 2;
};

struct SearchConfig {
  std::string backendRoot;
  std::string workDir;
  int64_t maxConnectLevel = 1;
  wtc::backend::BackendMode backendMode =
      wtc::backend::BackendMode::ProcessChain;
  bool verbose = false;
};

struct SearchArtifacts {
  std::string solverLogPath;
  std::string bestSummaryJsonPath;
  std::string normalizedBestJsonPath;
  std::string candidatesTsvPath;
};

bool runTypicalChainSearch(const std::string &inputPath,
                           const SearchConfig &config, SearchResult &result,
                           std::string &diagnostic,
                           SearchArtifacts *artifacts = nullptr);

bool loadBestFromJson(const std::string &jsonPath, SearchResult &result,
                      std::string &diagnostic);
bool writeBestToJson(const SearchResult &result, const std::string &jsonPath,
                     std::string &diagnostic);

} // namespace wtc::scheduler

#endif

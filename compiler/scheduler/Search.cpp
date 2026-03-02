#include "wtc/scheduler/Search.h"

#include "wtc/scheduler/internal/SearchInternal.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace {

bool extractInt64(const std::string &text, const char *key, int64_t &value) {
  std::string pattern = std::string("\\\"") + key + "\\\"\\s*:\\s*(-?[0-9]+)";
  std::regex re(pattern);
  std::smatch m;
  if (!std::regex_search(text, m, re) || m.size() < 2)
    return false;
  value = std::stoll(m[1].str());
  return true;
}

std::string readAll(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs)
    return "";
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

} // namespace

namespace wtc::scheduler {

bool loadBestFromJson(const std::string &jsonPath, SearchResult &result,
                      std::string &diagnostic) {
  if (jsonPath.empty()) {
    diagnostic = "json path is empty";
    return false;
  }
  const std::string json = readAll(jsonPath);
  if (json.empty()) {
    diagnostic = "failed to read json: " + jsonPath;
    return false;
  }

  SearchResult parsed;
  if (!extractInt64(json, "tileM", parsed.tileM) ||
      !extractInt64(json, "tileN", parsed.tileN) ||
      !extractInt64(json, "tileK", parsed.tileK) ||
      !extractInt64(json, "threadTileM", parsed.threadTileM) ||
      !extractInt64(json, "threadTileN", parsed.threadTileN)) {
    diagnostic = "failed to parse tile fields from " + jsonPath;
    return false;
  }

  result = parsed;
  diagnostic.clear();
  return true;
}

bool writeBestToJson(const SearchResult &result, const std::string &jsonPath,
                     std::string &diagnostic) {
  if (jsonPath.empty()) {
    diagnostic = "json path is empty";
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(jsonPath).parent_path());
  std::ofstream ofs(jsonPath);
  if (!ofs) {
    diagnostic = "failed to open json for write: " + jsonPath;
    return false;
  }

  ofs << "{\n"
      << "  \"tileM\": " << result.tileM << ",\n"
      << "  \"tileN\": " << result.tileN << ",\n"
      << "  \"tileK\": " << result.tileK << ",\n"
      << "  \"threadTileM\": " << result.threadTileM << ",\n"
      << "  \"threadTileN\": " << result.threadTileN << "\n"
      << "}\n";

  diagnostic.clear();
  return true;
}

bool runTypicalChainSearch(const std::string &inputPath,
                           const SearchConfig &config, SearchResult &result,
                           std::string &diagnostic,
                           SearchArtifacts *artifacts) {
  if (inputPath.empty()) {
    diagnostic = "input path is empty";
    return false;
  }
  if (config.backendRoot.empty()) {
    diagnostic = "backend root is empty";
    return false;
  }

  std::filesystem::path workDir = config.workDir;
  if (workDir.empty())
    workDir = std::filesystem::path("/tmp") / "wtc_search";
  std::filesystem::create_directories(workDir);

  const std::filesystem::path bestSummaryJson = workDir / "best_summary.json";
  const std::filesystem::path candidatesTsv = workDir / "candidates.tsv";
  const std::filesystem::path bestJson = workDir / "best.json";
  const std::filesystem::path solverLog = workDir / "solver.log";

  bool ok = false;
  if (config.backendMode == wtc::backend::BackendMode::Shell) {
    ok = internal::runSearchByShellScript(inputPath, config, bestSummaryJson,
                                          candidatesTsv, solverLog, diagnostic);
  } else if (config.backendMode == wtc::backend::BackendMode::Api) {
    ok = internal::runSearchByBackendApi(inputPath, config, bestSummaryJson,
                                         candidatesTsv, solverLog, result,
                                         diagnostic);
  } else {
    ok = internal::runSearchByProcessChain(inputPath, config, bestSummaryJson,
                                           candidatesTsv, solverLog,
                                           diagnostic);
  }
  if (!ok)
    return false;

  if (!std::filesystem::exists(bestSummaryJson)) {
    diagnostic = "solver output missing: " + bestSummaryJson.string();
    return false;
  }

  SearchResult parsed = result;
  if (config.backendMode != wtc::backend::BackendMode::Api) {
    if (!loadBestFromJson(bestSummaryJson.string(), parsed, diagnostic))
      return false;
  }

  if (!writeBestToJson(parsed, bestJson.string(), diagnostic))
    return false;

  result = parsed;

  if (artifacts) {
    artifacts->solverLogPath = solverLog.string();
    artifacts->bestSummaryJsonPath = bestSummaryJson.string();
    artifacts->normalizedBestJsonPath = bestJson.string();
    artifacts->candidatesTsvPath = candidatesTsv.string();
  }

  diagnostic.clear();
  return true;
}

} // namespace wtc::scheduler

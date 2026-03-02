#ifndef WTC_SCHEDULER_INTERNAL_SEARCHINTERNAL_H
#define WTC_SCHEDULER_INTERNAL_SEARCHINTERNAL_H

#include "wtc/scheduler/Search.h"

#include <filesystem>
#include <string>

namespace wtc::scheduler::internal {

std::string buildCommonSolverFlags(const std::string &inputPath,
                                   const wtc::scheduler::SearchConfig &config,
                                   const std::filesystem::path &bestSummaryJson,
                                   const std::filesystem::path &candidatesTsv);

bool runSearchByShellScript(const std::string &inputPath,
                            const wtc::scheduler::SearchConfig &config,
                            const std::filesystem::path &bestSummaryJson,
                            const std::filesystem::path &candidatesTsv,
                            const std::filesystem::path &solverLog,
                            std::string &diagnostic);

bool runSearchByProcessChain(const std::string &inputPath,
                             const wtc::scheduler::SearchConfig &config,
                             const std::filesystem::path &bestSummaryJson,
                             const std::filesystem::path &candidatesTsv,
                             const std::filesystem::path &solverLog,
                             std::string &diagnostic);

bool runSearchByBackendApi(const std::string &inputPath,
                           const wtc::scheduler::SearchConfig &config,
                           const std::filesystem::path &bestSummaryJson,
                           const std::filesystem::path &candidatesTsv,
                           const std::filesystem::path &solverLog,
                           wtc::scheduler::SearchResult &result,
                           std::string &diagnostic);

} // namespace wtc::scheduler::internal

#endif

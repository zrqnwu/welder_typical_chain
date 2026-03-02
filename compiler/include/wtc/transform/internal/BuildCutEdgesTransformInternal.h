#ifndef WTC_TRANSFORM_INTERNAL_BUILDCUTEDGESTRANSFORMINTERNAL_H
#define WTC_TRANSFORM_INTERNAL_BUILDCUTEDGESTRANSFORMINTERNAL_H

#include "wtc/transform/BuildCutEdgesTransform.h"

#include <filesystem>
#include <string>

namespace wtc::transform::internal {

std::string buildCommonWelderCompilerFlags(
    const std::string &inputPath, const wtc::scheduler::SearchResult &search,
    const wtc::transform::BuildConfig &config);

bool runCompileByShellScript(const std::string &inputPath,
                             const wtc::scheduler::SearchResult &search,
                             const wtc::transform::BuildConfig &config,
                             const std::filesystem::path &outDir,
                             const std::filesystem::path &compileLog,
                             std::string &diagnostic);

bool runCompileByProcessChain(const std::string &inputPath,
                              const wtc::scheduler::SearchResult &search,
                              const wtc::transform::BuildConfig &config,
                              const std::filesystem::path &outDir,
                              const std::filesystem::path &compileLog,
                              std::string &diagnostic);

bool runCompileByBackendApi(const std::string &inputPath,
                            const wtc::scheduler::SearchResult &search,
                            const wtc::transform::BuildConfig &config,
                            const std::filesystem::path &outDir,
                            const std::filesystem::path &compileLog,
                            std::string &diagnostic);

} // namespace wtc::transform::internal

#endif

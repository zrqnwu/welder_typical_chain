#ifndef WTC_BACKEND_TOOLCHAIN_H
#define WTC_BACKEND_TOOLCHAIN_H

#include <filesystem>
#include <string>

namespace wtc::backend {

enum class BackendMode {
  ProcessChain,
  InProcess = ProcessChain, // 兼容旧名
  Api,
  Shell,
};

const char *toString(BackendMode mode);
bool parseBackendMode(const std::string &text, BackendMode &mode,
                      std::string &diagnostic);

struct ToolchainPaths {
  std::filesystem::path backendRoot;
  std::filesystem::path backendCompilerDir;
  std::filesystem::path backendBuildDir;
  std::filesystem::path mlirPipelineDir;
  std::filesystem::path workgroupPassSrcDir;
  std::filesystem::path workgroupPassBuildDir;

  std::filesystem::path welderSolverBin;
  std::filesystem::path welderSolverCapiLib;
  std::filesystem::path welderCompileCapiLib;
  std::filesystem::path welderCompilerBin;
  std::filesystem::path welderPipelineBin;

  std::filesystem::path passPluginLib;
  std::filesystem::path mlirOptBin;
  std::filesystem::path llvmBuildDir;
  std::filesystem::path llvmDir;
  std::filesystem::path mlirDir;
};

bool resolveToolchain(const std::string &backendRoot, ToolchainPaths &paths,
                      std::string &diagnostic);

bool ensureBackendCompilerTarget(const ToolchainPaths &paths,
                                 const std::string &target, bool verbose,
                                 std::string &diagnostic);

bool ensureWorkgroupPassPlugin(ToolchainPaths &paths, bool verbose,
                               std::string &diagnostic);

std::string shellQuote(const std::string &s);

int runShellCommand(const std::string &cmd, const std::string &logPath,
                    bool appendLog, bool verbose, std::string *diagnostic);

} // namespace wtc::backend

#endif

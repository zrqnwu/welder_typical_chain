#include "wtc/scheduler/internal/SearchInternal.h"

#include "wtc/backend/Toolchain.h"

#include <dlfcn.h>
#if defined(__GLIBC__)
#include <link.h>
#endif
#include <array>
#include <filesystem>
#include <fstream>

namespace wtc::scheduler::internal {

bool runSearchByBackendApi(const std::string &inputPath,
                           const wtc::scheduler::SearchConfig &config,
                           const std::filesystem::path &bestSummaryJson,
                           const std::filesystem::path &candidatesTsv,
                           const std::filesystem::path &solverLog,
                           wtc::scheduler::SearchResult &result,
                           std::string &diagnostic) {
  wtc::backend::ToolchainPaths tc;
  if (!wtc::backend::resolveToolchain(config.backendRoot, tc, diagnostic))
    return false;
  if (!wtc::backend::ensureBackendCompilerTarget(tc, "welder-solver-capi",
                                                 config.verbose, diagnostic)) {
    return false;
  }

  // Re-resolve to refresh the library path after building.
  if (!wtc::backend::resolveToolchain(config.backendRoot, tc, diagnostic))
    return false;
  if (tc.welderSolverCapiLib.empty() ||
      !std::filesystem::exists(tc.welderSolverCapiLib)) {
    diagnostic =
        "solver C API library not found under: " + tc.backendBuildDir.string();
    return false;
  }

  using SolveFn = int (*)(const char *, const char *, const char *, int64_t, int,
                          int64_t *, int64_t *, int64_t *, int64_t *, int64_t *,
                          char *, size_t);

  // 优先用独立 linker namespace 装载 solver C API，尽量避免和同进程后续
  // compile(api) 共享全局静态状态（例如 LLVM/MLIR 注册表）。
  void *handle = nullptr;
#if defined(__GLIBC__)
  handle =
      dlmopen(LM_ID_NEWLM, tc.welderSolverCapiLib.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!handle)
    handle = dlopen(tc.welderSolverCapiLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    diagnostic = "dlopen failed: " + std::string(dlerror());
    return false;
  }

  auto closeHandle = [&]() { dlclose(handle); };
  SolveFn solveFn = reinterpret_cast<SolveFn>(
      dlsym(handle, "welder_solver_solve_typical_chain"));
  if (!solveFn) {
    const char *symErr = dlerror();
    diagnostic = "dlsym(welder_solver_solve_typical_chain) failed: " +
                 std::string(symErr ? symErr : "unknown");
    closeHandle();
    return false;
  }

  std::array<char, 4096> errBuf{};
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t threadTileM = 0;
  int64_t threadTileN = 0;

  // Keep a searchable log path even in API mode.
  if (config.verbose) {
    std::ofstream ofs(solverLog, std::ios::trunc);
    ofs << "[wtc.search.api] lib=" << tc.welderSolverCapiLib.string() << "\n";
    ofs << "[wtc.search.api] input=" << inputPath << "\n";
    ofs << "[wtc.search.api] max_connect_level=" << config.maxConnectLevel << "\n";
  }

  const int rc = solveFn(inputPath.c_str(), bestSummaryJson.c_str(),
                         candidatesTsv.c_str(), config.maxConnectLevel,
                         config.verbose ? 1 : 0, &tileM, &tileN, &tileK,
                         &threadTileM, &threadTileN, errBuf.data(),
                         errBuf.size());
  closeHandle();

  if (rc != 0) {
    diagnostic = "backend solver(api) failed, rc=" + std::to_string(rc) +
                 ", error=" + std::string(errBuf.data()) + ", see " +
                 solverLog.string();
    return false;
  }

  result.tileM = tileM;
  result.tileN = tileN;
  result.tileK = tileK;
  result.threadTileM = threadTileM;
  result.threadTileN = threadTileN;
  return true;
}

} // namespace wtc::scheduler::internal

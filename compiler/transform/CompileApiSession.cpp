#include "wtc/transform/internal/CompileApiSession.h"

#include <array>
#include <dlfcn.h>

namespace wtc::transform::internal {

CompileApiSession &CompileApiSession::instance() {
  static CompileApiSession session;
  return session;
}

bool CompileApiSession::ensureLoadedLocked(
    const std::filesystem::path &capiLibPath, std::string &diagnostic) {
  const std::string libPath = capiLibPath.string();
  if (libPath.empty()) {
    diagnostic = "compile(api) library path is empty";
    return false;
  }

  if (compileFn_ && loadedLibPath_ == libPath)
    return true;

  void *handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    diagnostic = "dlopen failed: " + std::string(dlerror());
    return false;
  }

  auto fn = reinterpret_cast<CompileFn>(
      dlsym(handle, "welder_compile_typical_chain_to_nvvm"));
  if (!fn) {
    const char *symErr = dlerror();
    diagnostic = "dlsym(welder_compile_typical_chain_to_nvvm) failed: " +
                 std::string(symErr ? symErr : "unknown");
    return false;
  }

  handle_ = handle;
  compileFn_ = fn;
  loadedLibPath_ = libPath;
  return true;
}

bool CompileApiSession::compile(const std::filesystem::path &capiLibPath,
                                const std::string &inputPath,
                                const std::string &outDir,
                                const std::string &pluginPath,
                                const std::string &compilerBinPath,
                                int64_t tileM, int64_t tileN, int64_t tileK,
                                int64_t threadTileM, int64_t threadTileN,
                                int64_t maxConnectLevel, bool fused, bool verbose,
                                std::string &diagnostic) {
  std::lock_guard<std::mutex> lock(mu_);

  if (!ensureLoadedLocked(capiLibPath, diagnostic))
    return false;

  std::array<char, 4096> errBuf{};
  const int rc = compileFn_(inputPath.c_str(), outDir.c_str(),
                            pluginPath.c_str(), compilerBinPath.c_str(), tileM,
                            tileN, tileK, threadTileM, threadTileN,
                            maxConnectLevel, fused ? 1 : 0, verbose ? 1 : 0,
                            errBuf.data(), errBuf.size());
  if (rc != 0) {
    diagnostic = "backend compile(api) failed, rc=" + std::to_string(rc) +
                 ", error=" + std::string(errBuf.data());
    return false;
  }
  return true;
}

} // namespace wtc::transform::internal

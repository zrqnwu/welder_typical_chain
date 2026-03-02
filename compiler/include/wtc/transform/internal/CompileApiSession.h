#ifndef WTC_TRANSFORM_INTERNAL_COMPILEAPISESSION_H
#define WTC_TRANSFORM_INTERNAL_COMPILEAPISESSION_H

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>

namespace wtc::transform::internal {

// compile(api) 的轻量会话封装：
// - 复用同一个 dlopen 句柄，避免重复加载导致的全局注册表抖动；
// - 串行化 API 调用，避免后端非线程安全路径并发踩踏。
class CompileApiSession {
public:
  static CompileApiSession &instance();

  bool compile(const std::filesystem::path &capiLibPath,
               const std::string &inputPath, const std::string &outDir,
               const std::string &pluginPath, const std::string &compilerBinPath,
               int64_t tileM, int64_t tileN, int64_t tileK,
               int64_t threadTileM, int64_t threadTileN,
               int64_t maxConnectLevel, bool fused, bool verbose,
               std::string &diagnostic);

private:
  using CompileFn = int (*)(const char *, const char *, const char *,
                            const char *, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int, int, char *, size_t);

  bool ensureLoadedLocked(const std::filesystem::path &capiLibPath,
                          std::string &diagnostic);

  std::mutex mu_;
  void *handle_ = nullptr;
  CompileFn compileFn_ = nullptr;
  std::string loadedLibPath_;
};

} // namespace wtc::transform::internal

#endif

#include "WelderCompileCAPI.h"
#include "WelderCompileSession.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace {

void setError(char *buf, size_t bufSize, const std::string &msg) {
  if (!buf || bufSize == 0)
    return;
  const size_t n = std::min(bufSize - 1, msg.size());
  std::memcpy(buf, msg.data(), n);
  buf[n] = '\0';
}

} // namespace

extern "C" int welder_compile_typical_chain_to_nvvm(
    const char *input_mlir_path, const char *out_dir,
    const char *workgroup_pass_plugin_path,
    const char *welder_compiler_bin_path, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, int64_t thread_tile_m, int64_t thread_tile_n,
    int64_t max_connect_level, int fused, int verbose, char *error_buffer,
    size_t error_buffer_size) {
  setError(error_buffer, error_buffer_size, "");

  if (!input_mlir_path || !*input_mlir_path) {
    setError(error_buffer, error_buffer_size, "input_mlir_path is empty");
    return 1;
  }
  if (!out_dir || !*out_dir) {
    setError(error_buffer, error_buffer_size, "out_dir is empty");
    return 1;
  }
  if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0 || thread_tile_m <= 0 ||
      thread_tile_n <= 0) {
    setError(error_buffer, error_buffer_size, "invalid tile sizes");
    return 1;
  }

  welder::compiler::CompileSessionRequest request;
  request.inputMlirPath = input_mlir_path;
  request.outDir = out_dir;
  request.workgroupPassPluginPath =
      (workgroup_pass_plugin_path && *workgroup_pass_plugin_path)
          ? std::string(workgroup_pass_plugin_path)
          : std::string();
  request.welderCompilerBinPath =
      (welder_compiler_bin_path && *welder_compiler_bin_path)
          ? std::string(welder_compiler_bin_path)
          : std::string();
  request.tileM = tile_m;
  request.tileN = tile_n;
  request.tileK = tile_k;
  request.threadTileM = thread_tile_m;
  request.threadTileN = thread_tile_n;
  request.maxConnectLevel = max_connect_level;
  request.fused = fused != 0;
  request.verbose = verbose != 0;

  std::string error;
  welder::compiler::CompileSession session;
  if (!session.run(request, error)) {
    setError(error_buffer, error_buffer_size, error);
    return 2;
  }

  return 0;
}

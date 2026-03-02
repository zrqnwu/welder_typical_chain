#ifndef WELDER_COMPILE_SESSION_H
#define WELDER_COMPILE_SESSION_H

#include <cstdint>
#include <string>

namespace welder::compiler {

struct CompileSessionRequest {
  std::string inputMlirPath;
  std::string outDir;
  std::string workgroupPassPluginPath;
  std::string welderCompilerBinPath;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t threadTileM = 0;
  int64_t threadTileN = 0;
  int64_t maxConnectLevel = 1;
  bool fused = true;
  bool verbose = false;
};

// 后端 compile(api) 的会话执行器：
// - 封装 welder-compiler + lowering pass 链；
// - 每次 run 使用新的 MLIRContext/PassManager；
// - 统一返回字符串诊断，便于 C API 向上层转译错误。
class CompileSession {
public:
  bool run(const CompileSessionRequest &request, std::string &error) const;
};

} // namespace welder::compiler

#endif

#ifndef WTC_PIPELINE_PIPELINERUNNER_H
#define WTC_PIPELINE_PIPELINERUNNER_H

#include "wtc/backend/Toolchain.h"

#include <cstdint>
#include <string>

namespace wtc::pipeline {

enum class RunMode {
  Full,
  SearchOnly,
  CompileOnly,
};

struct StageArtifacts {
  std::string rootDir;

  std::string irDir;
  std::string canonicalizedMlirPath;
  std::string taggedMlirPath;
  std::string tagsJsonPath;

  std::string searchDir;
  std::string bestJsonPath;

  std::string compileDir;
};

struct RunOptions {
  std::string inputPath;
  std::string outputDir = "/tmp/wtc_out";
  std::string backendRoot = "/home/zhangruiqi/welder_typical_chain/backend/welder";
  std::string bestJsonPath;
  int64_t maxConnectLevel = 1;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t threadTileM = 0;
  int64_t threadTileN = 0;
  bool enableSearch = true;
  bool fused = true;
  wtc::backend::BackendMode backendMode =
      wtc::backend::BackendMode::ProcessChain;
  // 默认保持稳定路径：full 模式下 search(api) 会回退到 process_chain。
  // 显式开启后允许同进程 search(api) -> compile(api) 直连（实验模式）。
  bool pureApiFull = false;
  RunMode mode = RunMode::Full;
  bool verbose = false;
};

int runTypicalChainPipeline(const RunOptions &options);

} // namespace wtc::pipeline

#endif

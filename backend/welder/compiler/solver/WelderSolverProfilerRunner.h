#ifndef WELDER_SOLVER_PROFILER_RUNNER_H
#define WELDER_SOLVER_PROFILER_RUNNER_H

#include "WelderSolverLib.h"

#include <string>

namespace welder::solver {

struct ProfilerPathOverrides {
  std::string profilerBin;
  std::string compilerToNvvmScript;
};

// 解析 profiling 相关工具路径：
// - 用户显式传入时优先使用；
// - 否则按 welder-solver 可执行文件位置推导默认路径。
void resolveProfilerToolPaths(welder::SolveOptions &opts,
                              const std::string &argv0,
                              const ProfilerPathOverrides &overrides);

} // namespace welder::solver

#endif

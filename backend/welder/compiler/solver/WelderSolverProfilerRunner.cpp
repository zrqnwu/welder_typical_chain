#include "WelderSolverProfilerRunner.h"

#include <filesystem>

namespace welder::solver {

void resolveProfilerToolPaths(welder::SolveOptions &opts,
                              const std::string &argv0,
                              const ProfilerPathOverrides &overrides) {
  if (!opts.profile.enable)
    return;

  std::filesystem::path exePath(argv0);
  std::filesystem::path exeDir = exePath.has_parent_path()
                                     ? exePath.parent_path()
                                     : std::filesystem::current_path();

  if (!overrides.profilerBin.empty()) {
    opts.profile.profilerBin = overrides.profilerBin;
  } else {
    opts.profile.profilerBin = (exeDir / "welder-profiler").string();
  }

  if (!overrides.compilerToNvvmScript.empty()) {
    opts.profile.compilerToNvvmScript = overrides.compilerToNvvmScript;
  } else {
    opts.profile.compilerToNvvmScript =
        (exeDir / ".." / "run_welder_to_nvvm_isa.sh").string();
  }
}

} // namespace welder::solver

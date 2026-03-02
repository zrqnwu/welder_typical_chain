#include "wtc/transform/internal/BuildCutEdgesTransformInternal.h"

#include "wtc/backend/Toolchain.h"

#include <filesystem>
#include <sstream>

namespace wtc::transform::internal {

bool runCompileByProcessChain(const std::string &inputPath,
                              const wtc::scheduler::SearchResult &search,
                              const wtc::transform::BuildConfig &config,
                              const std::filesystem::path &outDir,
                              const std::filesystem::path &compileLog,
                              std::string &diagnostic) {
  wtc::backend::ToolchainPaths tc;
  if (!wtc::backend::resolveToolchain(config.backendRoot, tc, diagnostic))
    return false;
  if (!wtc::backend::ensureWorkgroupPassPlugin(tc, config.verbose, diagnostic))
    return false;

  const std::filesystem::path afterT2 = outDir / "03.after_postbufferize.mlir";
  const std::filesystem::path afterWg = outDir / "04.after_workgroup_launch.mlir";
  const std::filesystem::path afterLoops = outDir / "04c.after_linalg_to_loops.mlir";
  const std::filesystem::path loweredNvvm = outDir / "05.out.nvvm.runnable.mlir";

  const std::string commonFlags =
      buildCommonWelderCompilerFlags(inputPath, search, config);

  {
    std::ostringstream cmd;
    cmd << wtc::backend::shellQuote(tc.welderCompilerBin.string()) << " "
        << commonFlags << " --output "
        << wtc::backend::shellQuote(afterT2.string());
    int rc = wtc::backend::runShellCommand(
        cmd.str(), compileLog.string(), /*appendLog=*/false, config.verbose,
        &diagnostic);
    if (rc != 0) {
      diagnostic = "welder-compiler stage failed, rc=" + std::to_string(rc) +
                   ", see " + compileLog.string();
      return false;
    }
  }

  if (!std::filesystem::exists(afterT2)) {
    diagnostic = "missing postbufferize artifact: " + afterT2.string();
    return false;
  }

  {
    const std::string wgPass =
        "builtin.module(workgroup-alloc-to-launch-workgroup)";
    std::ostringstream cmd;
    cmd << wtc::backend::shellQuote(tc.mlirOptBin.string()) << " "
        << wtc::backend::shellQuote(afterT2.string())
        << " --load-pass-plugin="
        << wtc::backend::shellQuote(tc.passPluginLib.string())
        << " --pass-pipeline=" << wtc::backend::shellQuote(wgPass)
        << " -o " << wtc::backend::shellQuote(afterWg.string());
    int rc = wtc::backend::runShellCommand(
        cmd.str(), compileLog.string(), /*appendLog=*/true, config.verbose,
        &diagnostic);
    if (rc != 0) {
      diagnostic = "workgroup launch lowering failed, rc=" + std::to_string(rc) +
                   ", see " + compileLog.string();
      return false;
    }
  }

  {
    const std::string loopsPipeline = "builtin.module(convert-linalg-to-loops)";
    std::ostringstream cmd;
    cmd << wtc::backend::shellQuote(tc.mlirOptBin.string()) << " "
        << wtc::backend::shellQuote(afterWg.string())
        << " --pass-pipeline=" << wtc::backend::shellQuote(loopsPipeline)
        << " -o " << wtc::backend::shellQuote(afterLoops.string());
    int rc = wtc::backend::runShellCommand(
        cmd.str(), compileLog.string(), /*appendLog=*/true, config.verbose,
        &diagnostic);
    if (rc != 0) {
      diagnostic = "linalg-to-loops stage failed, rc=" + std::to_string(rc) +
                   ", see " + compileLog.string();
      return false;
    }
  }

  {
    const std::string nvvmPipeline =
        "builtin.module("
        "convert-nvgpu-to-nvvm,"
        "gpu-kernel-outlining,"
        "convert-vector-to-scf{target-rank=1},"
        "convert-vector-to-llvm,"
        "convert-scf-to-cf,"
        "convert-nvvm-to-llvm,"
        "convert-func-to-llvm,"
        "expand-strided-metadata,"
        "nvvm-attach-target{chip=sm_86},"
        "lower-affine,"
        "convert-arith-to-llvm,"
        "convert-index-to-llvm,"
        "canonicalize,"
        "cse,"
        "gpu.module(convert-vector-to-scf{target-rank=1},"
        "convert-vector-to-llvm,"
        "convert-scf-to-cf,"
        "convert-gpu-to-nvvm,"
        "canonicalize,"
        "cse,"
        "reconcile-unrealized-casts),"
        "gpu-to-llvm,"
        "reconcile-unrealized-casts,"
        "llvm-vector4-align,"
        "gpu-module-to-binary{format=isa},"
        "convert-math-to-llvm,"
        "canonicalize,"
        "cse,"
        "reconcile-unrealized-casts)";
    std::ostringstream cmd;
    cmd << wtc::backend::shellQuote(tc.mlirOptBin.string()) << " "
        << wtc::backend::shellQuote(afterLoops.string())
        << " --load-pass-plugin="
        << wtc::backend::shellQuote(tc.passPluginLib.string())
        << " --pass-pipeline=" << wtc::backend::shellQuote(nvvmPipeline)
        << " -o " << wtc::backend::shellQuote(loweredNvvm.string());
    int rc = wtc::backend::runShellCommand(
        cmd.str(), compileLog.string(), /*appendLog=*/true, config.verbose,
        &diagnostic);
    if (rc != 0) {
      diagnostic = "nvvm lowering stage failed, rc=" + std::to_string(rc) +
                   ", see " + compileLog.string();
      return false;
    }
  }

  return true;
}

} // namespace wtc::transform::internal

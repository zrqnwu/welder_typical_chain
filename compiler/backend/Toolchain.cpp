#include "wtc/backend/Toolchain.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path pickExisting(const std::vector<std::filesystem::path> &cands) {
  for (const auto &p : cands) {
    if (!p.empty() && std::filesystem::exists(p))
      return p;
  }
  return {};
}

std::filesystem::path findPassPluginUnder(const std::filesystem::path &buildDir) {
  return pickExisting({
      buildDir / "WorkgroupAllocToLaunchPass.so",
      buildDir / "libWorkgroupAllocToLaunchPass.so",
      buildDir / "WorkgroupAllocToLaunchPass.dylib",
      buildDir / "libWorkgroupAllocToLaunchPass.dylib",
  });
}

std::filesystem::path findSolverCapiLib(const std::filesystem::path &buildDir) {
  return pickExisting({
      buildDir / "welder-solver-capi.so",
      buildDir / "libwelder-solver-capi.so",
      buildDir / "welder-solver-capi.dylib",
      buildDir / "libwelder-solver-capi.dylib",
  });
}

std::filesystem::path findCompileCapiLib(const std::filesystem::path &buildDir) {
  return pickExisting({
      buildDir / "welder-compile-capi.so",
      buildDir / "libwelder-compile-capi.so",
      buildDir / "welder-compile-capi.dylib",
      buildDir / "libwelder-compile-capi.dylib",
  });
}

std::filesystem::path detectLlvmBuildPath() {
  if (const char *p = std::getenv("WTC_LLVM_BUILD"); p && *p) {
    std::filesystem::path path = p;
    if (std::filesystem::exists(path))
      return path;
  }
  const std::filesystem::path pinned = "/home/zhangruiqi/llvm-project/build";
  if (std::filesystem::exists(pinned))
    return pinned;
  return {};
}

std::filesystem::path detectMlirOptPath(const std::filesystem::path &llvmBuild) {
  if (const char *p = std::getenv("WTC_MLIR_OPT"); p && *p) {
    std::filesystem::path path = p;
    if (std::filesystem::exists(path))
      return path;
  }
  if (!llvmBuild.empty()) {
    const std::filesystem::path pinned = llvmBuild / "bin" / "mlir-opt";
    if (std::filesystem::exists(pinned))
      return pinned;
  }
  return "mlir-opt";
}

std::filesystem::path detectLibDir(const std::filesystem::path &llvmBuild) {
  if (llvmBuild.empty())
    return {};
  const std::filesystem::path lib = llvmBuild / "lib";
  const std::filesystem::path lib64 = llvmBuild / "lib64";
  if (std::filesystem::exists(lib))
    return lib;
  if (std::filesystem::exists(lib64))
    return lib64;
  return lib;
}

bool configureWithCMake(const std::filesystem::path &srcDir,
                        const std::filesystem::path &buildDir,
                        const wtc::backend::ToolchainPaths &paths,
                        bool verbose, std::string &diagnostic) {
  // Always re-run configure to pick up changed targets/options in existing
  // build trees (avoids stale-cache "unknown target" failures).
  std::ostringstream cmd;
  cmd << "cmake -S " << wtc::backend::shellQuote(srcDir.string()) << " -B "
      << wtc::backend::shellQuote(buildDir.string())
      << " -DCMAKE_BUILD_TYPE=Release";
  if (!paths.llvmDir.empty() && !paths.mlirDir.empty()) {
    cmd << " -DLLVM_DIR=" << wtc::backend::shellQuote(paths.llvmDir.string())
        << " -DMLIR_DIR=" << wtc::backend::shellQuote(paths.mlirDir.string());
  }

  const std::filesystem::path logPath = buildDir / "wtc_configure.log";
  int rc = wtc::backend::runShellCommand(cmd.str(), logPath.string(),
                                         /*appendLog=*/false, verbose,
                                         &diagnostic);
  if (rc != 0) {
    diagnostic = "cmake configure failed, rc=" + std::to_string(rc) + ", see " +
                 logPath.string();
    return false;
  }
  return true;
}

} // namespace

namespace wtc::backend {

const char *toString(BackendMode mode) {
  if (mode == BackendMode::Shell)
    return "shell";
  if (mode == BackendMode::Api)
    return "api";
  return "process_chain";
}

bool parseBackendMode(const std::string &text, BackendMode &mode,
                      std::string &diagnostic) {
  if (text == "shell") {
    mode = BackendMode::Shell;
    diagnostic.clear();
    return true;
  }
  if (text == "api") {
    mode = BackendMode::Api;
    diagnostic.clear();
    return true;
  }
  if (text == "process_chain" || text == "subprocess_pipeline" ||
      text == "inprocess") {
    mode = BackendMode::ProcessChain;
    diagnostic.clear();
    return true;
  }
  diagnostic = "unknown backend mode: " + text +
               " (expected shell|process_chain|api)";
  return false;
}

std::string shellQuote(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\'') {
      out.append("'\\''");
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

int runShellCommand(const std::string &cmd, const std::string &logPath,
                    bool appendLog, bool verbose, std::string *diagnostic) {
  std::filesystem::create_directories(std::filesystem::path(logPath).parent_path());
  if (verbose) {
    std::ofstream ofs(logPath, appendLog ? std::ios::app : std::ios::trunc);
    ofs << "[wtc.exec] " << cmd << "\n";
  }

  std::ostringstream full;
  full << cmd << " " << (appendLog ? ">>" : ">") << " " << shellQuote(logPath)
       << " 2>&1";

  const int rc = std::system(full.str().c_str());
  if (rc != 0 && diagnostic)
    *diagnostic = "command failed, rc=" + std::to_string(rc) + ": " + cmd;
  return rc;
}

bool resolveToolchain(const std::string &backendRoot, ToolchainPaths &paths,
                      std::string &diagnostic) {
  if (backendRoot.empty()) {
    diagnostic = "backend root is empty";
    return false;
  }

  paths = {};
  paths.backendRoot = backendRoot;
  if (!std::filesystem::exists(paths.backendRoot)) {
    diagnostic = "backend root not found: " + paths.backendRoot.string();
    return false;
  }

  paths.backendCompilerDir = paths.backendRoot / "compiler";
  paths.backendBuildDir = paths.backendCompilerDir / "build";
  paths.mlirPipelineDir = paths.backendRoot / "mlir_pipeline";
  paths.workgroupPassSrcDir =
      paths.mlirPipelineDir / "workgroup_alloc_to_launch_pass";
  paths.workgroupPassBuildDir = paths.workgroupPassSrcDir / "build";

  if (!std::filesystem::exists(paths.backendCompilerDir)) {
    diagnostic = "backend compiler dir not found: " +
                 paths.backendCompilerDir.string();
    return false;
  }
  if (!std::filesystem::exists(paths.workgroupPassSrcDir)) {
    diagnostic = "workgroup pass dir not found: " +
                 paths.workgroupPassSrcDir.string();
    return false;
  }

  paths.welderSolverBin = paths.backendBuildDir / "welder-solver";
  paths.welderSolverCapiLib = findSolverCapiLib(paths.backendBuildDir);
  paths.welderCompileCapiLib = findCompileCapiLib(paths.backendBuildDir);
  paths.welderCompilerBin = paths.backendBuildDir / "welder-compiler";
  paths.welderPipelineBin = paths.backendBuildDir / "welder-pipeline";

  paths.llvmBuildDir = detectLlvmBuildPath();
  const std::filesystem::path libDir = detectLibDir(paths.llvmBuildDir);
  if (!libDir.empty()) {
    paths.llvmDir = libDir / "cmake" / "llvm";
    paths.mlirDir = libDir / "cmake" / "mlir";
  }
  paths.mlirOptBin = detectMlirOptPath(paths.llvmBuildDir);
  paths.passPluginLib = findPassPluginUnder(paths.workgroupPassBuildDir);

  diagnostic.clear();
  return true;
}

bool ensureBackendCompilerTarget(const ToolchainPaths &paths,
                                 const std::string &target, bool verbose,
                                 std::string &diagnostic) {
  std::filesystem::path targetBin;
  if (target == "welder-solver") {
    targetBin = paths.welderSolverBin;
  } else if (target == "welder-solver-capi") {
    targetBin = findSolverCapiLib(paths.backendBuildDir);
  } else if (target == "welder-compile-capi") {
    targetBin = findCompileCapiLib(paths.backendBuildDir);
  } else if (target == "welder-compiler") {
    targetBin = paths.welderCompilerBin;
  } else if (target == "welder-pipeline") {
    targetBin = paths.welderPipelineBin;
  } else {
    diagnostic = "unsupported backend target: " + target;
    return false;
  }

  if (std::filesystem::exists(targetBin)) {
    diagnostic.clear();
    return true;
  }

  if (!configureWithCMake(paths.backendCompilerDir, paths.backendBuildDir, paths,
                          verbose, diagnostic)) {
    return false;
  }

  std::ostringstream cmd;
  cmd << "cmake --build " << shellQuote(paths.backendBuildDir.string())
      << " --target " << shellQuote(target) << " -j";
  const std::filesystem::path logPath = paths.backendBuildDir / "wtc_build.log";
  int rc = runShellCommand(cmd.str(), logPath.string(),
                           /*appendLog=*/true, verbose, &diagnostic);
  if (rc != 0) {
    diagnostic = "cmake build failed for " + target + ", rc=" +
                 std::to_string(rc) + ", see " + logPath.string();
    return false;
  }
  if (!std::filesystem::exists(targetBin)) {
    if (target == "welder-solver-capi") {
      targetBin = findSolverCapiLib(paths.backendBuildDir);
    } else if (target == "welder-compile-capi") {
      targetBin = findCompileCapiLib(paths.backendBuildDir);
    }
  }
  if (!std::filesystem::exists(targetBin)) {
    diagnostic = "target artifact not found after build: " + targetBin.string();
    return false;
  }

  diagnostic.clear();
  return true;
}

bool ensureWorkgroupPassPlugin(ToolchainPaths &paths, bool verbose,
                               std::string &diagnostic) {
  paths.passPluginLib = findPassPluginUnder(paths.workgroupPassBuildDir);
  if (!paths.passPluginLib.empty()) {
    diagnostic.clear();
    return true;
  }

  if (!configureWithCMake(paths.workgroupPassSrcDir, paths.workgroupPassBuildDir,
                          paths, verbose, diagnostic)) {
    return false;
  }

  std::ostringstream cmd;
  cmd << "cmake --build " << shellQuote(paths.workgroupPassBuildDir.string())
      << " -j";
  const std::filesystem::path logPath =
      paths.workgroupPassBuildDir / "wtc_build.log";
  int rc = runShellCommand(cmd.str(), logPath.string(),
                           /*appendLog=*/true, verbose, &diagnostic);
  if (rc != 0) {
    diagnostic = "workgroup pass build failed, rc=" + std::to_string(rc) +
                 ", see " + logPath.string();
    return false;
  }

  paths.passPluginLib = findPassPluginUnder(paths.workgroupPassBuildDir);
  if (paths.passPluginLib.empty()) {
    diagnostic = "pass plugin not found under: " +
                 paths.workgroupPassBuildDir.string();
    return false;
  }

  diagnostic.clear();
  return true;
}

} // namespace wtc::backend

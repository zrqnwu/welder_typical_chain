#include "WelderCompileSession.h"

#include "WelderCompilerMain.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace {

// 当 workgroup pass 以源码方式链接进本动态库时，由该符号提供注册入口。
extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo();

bool ensureEmbeddedWorkgroupPassesRegistered(std::string &error) {
  static std::once_flag once;
  static bool ok = false;
  static std::string diag;
  std::call_once(once, []() {
    auto info = mlirGetPassPluginInfo();
    if (info.apiVersion != MLIR_PLUGIN_API_VERSION) {
      diag = "embedded workgroup pass API version mismatch";
      ok = false;
      return;
    }
    if (!info.registerPassRegistryCallbacks) {
      diag = "embedded workgroup pass has empty registration callback";
      ok = false;
      return;
    }
    info.registerPassRegistryCallbacks();
    ok = true;
  });
  if (!ok) {
    error = diag.empty() ? "failed to register embedded workgroup pass" : diag;
    return false;
  }
  return true;
}

bool runPassPipeline(const std::string &inputPath, const std::string &outputPath,
                     const std::string &pipeline,
                     const std::string &pluginPath, std::string &error) {
  static bool passesRegistered = []() {
    mlir::registerAllPasses();
    return true;
  }();
  (void)passesRegistered;

  bool embeddedRegistered = ensureEmbeddedWorkgroupPassesRegistered(error);
  if (!embeddedRegistered && pluginPath.empty())
    return false;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::MLIRContext ctx(registry);

  // 已内嵌 pass 时跳过外部插件加载，避免重复注册同名 pass 参数。
  if (!embeddedRegistered && !pluginPath.empty()) {
    std::string loadErr;
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        pluginPath.c_str(), &loadErr);
    if (!lib.isValid()) {
      error = "failed to load pass plugin: " + pluginPath +
              ", detail: " + loadErr;
      return false;
    }
    void *sym = lib.getAddressOfSymbol("mlirGetPassPluginInfo");
    if (!sym) {
      error = "mlirGetPassPluginInfo not found in plugin: " + pluginPath;
      return false;
    }
    using GetInfoFn = ::mlir::PassPluginLibraryInfo (*)();
    auto getInfo = reinterpret_cast<GetInfoFn>(sym);
    auto info = getInfo();
    if (info.apiVersion != MLIR_PLUGIN_API_VERSION) {
      error = "pass plugin API mismatch for: " + pluginPath;
      return false;
    }
    if (!info.registerPassRegistryCallbacks) {
      error = "pass plugin registration callback missing: " + pluginPath;
      return false;
    }
    info.registerPassRegistryCallbacks();
  }

  std::string openErr;
  auto file = mlir::openInputFile(inputPath, &openErr);
  if (!file) {
    error = "cannot open input for pass pipeline: " + inputPath + " (" + openErr +
            ")";
    return false;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    error = "failed to parse module for pass pipeline: " + inputPath;
    return false;
  }

  mlir::PassManager pm(&ctx);
  std::string parseErr;
  llvm::raw_string_ostream parseErrOs(parseErr);
  if (mlir::failed(mlir::parsePassPipeline(pipeline, pm, parseErrOs))) {
    parseErrOs.flush();
    error = "failed to parse pass pipeline: " + pipeline +
            ", detail: " + parseErr;
    return false;
  }

  if (mlir::failed(pm.run(*module))) {
    error = "pass pipeline run failed: " + pipeline;
    return false;
  }

  std::string outErr;
  auto outFile = mlir::openOutputFile(outputPath, &outErr);
  if (!outFile) {
    error = "cannot open output file: " + outputPath + " (" + outErr + ")";
    return false;
  }
  module->print(outFile->os());
  outFile->os() << "\n";
  outFile->keep();
  return true;
}

std::vector<std::string>
buildWelderCompilerArgs(const welder::compiler::CompileSessionRequest &request,
                        const std::string &outAfterPostbufferize) {
  std::vector<std::string> args;
  args.reserve(32);
  args.push_back("welder-compiler");
  args.push_back(request.inputMlirPath);
  args.push_back("--enable-generic-problem");
  args.push_back("--enable-tile-propagation");
  args.push_back("--enable-cut-edges");
  args.push_back("--enable-two-level-schedule");
  args.push_back("--require-perfect-tiling=false");
  args.push_back("--force-tile-m");
  args.push_back(std::to_string(request.tileM));
  args.push_back("--force-tile-n");
  args.push_back(std::to_string(request.tileN));
  args.push_back("--force-tile-k");
  args.push_back(std::to_string(request.tileK));
  args.push_back("--enable-register-level-schedule");
  args.push_back("--thread-tile-m");
  args.push_back(std::to_string(request.threadTileM));
  args.push_back("--thread-tile-n");
  args.push_back(std::to_string(request.threadTileN));
  args.push_back("--max-connect-level");
  args.push_back(std::to_string(request.maxConnectLevel));

  if (request.fused) {
    args.push_back("--reduction-chain-split-broadcast-edges=false");
    args.push_back("--enable-row-reduction-chain-reuse-fusion");
    args.push_back("--enable-row-reduction-input-promotion");
    args.push_back("--enable-matmul-softmax-shared-reuse-fusion");
  } else {
    args.push_back("--reduction-chain-split-broadcast-edges=true");
  }

  args.push_back("--output");
  args.push_back(outAfterPostbufferize);
  return args;
}

bool runWelderCompilerStage(const std::vector<std::string> &args,
                            const std::string &compilerBinPath,
                            std::string &error) {
  if (args.empty()) {
    error = "empty compiler argv";
    return false;
  }

  std::vector<std::string> argvStorage(args.begin(), args.end());
  std::error_code ec;
  if (!compilerBinPath.empty() && std::filesystem::exists(compilerBinPath, ec))
    argvStorage[0] = compilerBinPath;
  else if (argvStorage[0].empty())
    argvStorage[0] = "welder-compiler";

  std::vector<char *> argv;
  argv.reserve(argvStorage.size());
  for (std::string &s : argvStorage)
    argv.push_back(const_cast<char *>(s.c_str()));

  const int rc = welderCompilerMain(static_cast<int>(argv.size()), argv.data());
  if (rc != 0) {
    error = "welder-compiler in-process failed, rc=" + std::to_string(rc);
    return false;
  }
  return true;
}

} // namespace

namespace welder::compiler {

bool CompileSession::run(const CompileSessionRequest &request,
                         std::string &error) const {
  if (request.inputMlirPath.empty()) {
    error = "input_mlir_path is empty";
    return false;
  }
  if (request.outDir.empty()) {
    error = "out_dir is empty";
    return false;
  }
  if (request.tileM <= 0 || request.tileN <= 0 || request.tileK <= 0 ||
      request.threadTileM <= 0 || request.threadTileN <= 0) {
    error = "invalid tile sizes";
    return false;
  }

  const std::filesystem::path outDirPath = request.outDir;
  std::error_code ec;
  std::filesystem::create_directories(outDirPath, ec);
  if (ec) {
    error = "failed to create out_dir: " + outDirPath.string() + ", ec=" +
            std::to_string(ec.value());
    return false;
  }

  const std::string afterT2 =
      (outDirPath / "03.after_postbufferize.mlir").string();
  const std::string afterWg =
      (outDirPath / "04.after_workgroup_launch.mlir").string();
  const std::string afterLoops =
      (outDirPath / "04c.after_linalg_to_loops.mlir").string();
  const std::string loweredNvvm =
      (outDirPath / "05.out.nvvm.runnable.mlir").string();

  const auto welderArgs = buildWelderCompilerArgs(request, afterT2);
  if (!runWelderCompilerStage(welderArgs, request.welderCompilerBinPath, error))
    return false;

  if (!runPassPipeline(afterT2, afterWg, "workgroup-alloc-to-launch-workgroup",
                       request.workgroupPassPluginPath, error)) {
    return false;
  }

  if (!runPassPipeline(afterWg, afterLoops, "convert-linalg-to-loops",
                       /*pluginPath=*/"", error)) {
    return false;
  }

  const std::string nvvmPipeline =
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
      "reconcile-unrealized-casts";

  if (!runPassPipeline(afterLoops, loweredNvvm, nvvmPipeline,
                       request.workgroupPassPluginPath, error)) {
    return false;
  }

  if (!std::filesystem::exists(loweredNvvm)) {
    error = "missing lowered artifact: " + loweredNvvm;
    return false;
  }

  return true;
}

} // namespace welder::compiler

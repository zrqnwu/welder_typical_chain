#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cuda.h>

#include <array>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

struct LaunchArg {
  std::string value;
  std::string type;
};

struct LaunchSpec {
  std::string binaryName;
  std::string entryName;
  std::array<std::string, 3> gridTokens;
  std::array<std::string, 3> blockTokens;
  std::vector<LaunchArg> args;
};

struct MemrefDesc {
  int rank = 0; // 0 表示未知。
  std::string baseSym;
  std::string alignedSym;
  int64_t offset = 0;
  std::array<int64_t, 2> sizes = {0, 0};
  std::array<int64_t, 2> strides = {0, 0};
  int64_t bytes = 0;
};

#include "WelderProfilerParsing.h"
#include "WelderProfilerEval.h"
#include "WelderProfilerRuntime.h"
} // 命名空间

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  llvm::cl::opt<std::string> nvvmMlirPath(
      llvm::cl::Positional, llvm::cl::desc("<nvvm_runnable.mlir>"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> kernelName(
      "kernel",
      llvm::cl::desc("Kernel name (gpu.binary symbol or .entry name). If "
                     "omitted, the first gpu.launch_func is used."),
      llvm::cl::init(""));
  llvm::cl::opt<int64_t> elemBytes(
      "elem-bytes",
      llvm::cl::desc("Element byte size for memref footprint estimation "
                     "(default 4 for f32)"),
      llvm::cl::init(4));
  llvm::cl::opt<int> warmup(
      "warmup",
      llvm::cl::desc("Warmup launches before timing (default 10)"),
      llvm::cl::init(10));
  llvm::cl::opt<int> iters(
      "iters",
      llvm::cl::desc("Measured launches (default 100)"),
      llvm::cl::init(100));
  llvm::cl::list<std::string> i64Overrides(
      "i64",
      llvm::cl::desc(
          "Override i64 SSA token value (repeatable). Example: --i64 %arg16=0"),
      llvm::cl::ZeroOrMore);
  llvm::cl::list<std::string> initPtrs(
      "init-ptr",
      llvm::cl::desc("Initialize a memref buffer by its base/aliased pointer SSA "
                     "token (repeatable). Example: --init-ptr %arg0"),
      llvm::cl::ZeroOrMore);
  llvm::cl::list<std::string> initPtrsF16(
      "init-ptr-f16",
      llvm::cl::desc(
          "Initialize a memref buffer as f16 by its base/aliased pointer SSA "
          "token (repeatable). Example: --init-ptr-f16 %arg0"),
      llvm::cl::ZeroOrMore);
  llvm::cl::list<std::string> fillPtrs(
      "fill",
      llvm::cl::desc("Fill a memref buffer with a constant f32 value by its "
                     "base/aliased pointer SSA token (repeatable). Example: "
                     "--fill %arg0=0 --fill %tmp=-inf"),
      llvm::cl::ZeroOrMore);
  llvm::cl::opt<bool> fillEachIter(
      "fill-each-iter",
      llvm::cl::desc(
          "Re-apply all --fill buffers before each warmup/measurement "
          "iteration. This is important for multi-kernel graphs where some "
          "kernels expect reduction init buffers (e.g., matmul C=0, softmax "
          "max=-inf, sum=0) and would otherwise accumulate across iterations."),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> initMode(
      "init",
      llvm::cl::desc("Initialization pattern for --init-ptr/--init-ptr-f16 "
                     "buffers: "
                     "zero|linear|random (default linear)"),
      llvm::cl::init("linear"));
  llvm::cl::opt<int64_t> initSeed(
      "seed", llvm::cl::desc("Random seed for --init=random (default 1)"),
      llvm::cl::init(1));
  llvm::cl::list<std::string> dumpMemrefs(
      "dump",
      llvm::cl::desc("Dump a memref argument to a binary file (repeatable). "
                     "Example: --dump %arg0=/tmp/in.bin (writes .bin and "
                     ".bin.json)"),
      llvm::cl::ZeroOrMore);
  llvm::cl::opt<std::string> dumpLast2D(
      "dump-last-2d",
      llvm::cl::desc("Dump the last rank-2 memref argument to a binary file "
                     "(writes <path> and <path>.json)"),
      llvm::cl::init(""));
  llvm::cl::opt<bool> listMemrefs(
      "list-memrefs",
      llvm::cl::desc("List detected memref descriptor arguments and exit"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> runAllKernels(
      "run-all-kernels",
      llvm::cl::desc("Execute all gpu.launch_func kernels in the module in "
                     "appearance order (measures total time and dumps after "
                     "the full sequence). Incompatible with --kernel."),
      llvm::cl::init(false));
  llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Verbose logging"),
                              llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv, "welder-profiler\n");

  std::unordered_map<std::string, int64_t> i64OverrideMap;
  i64OverrideMap.reserve(i64Overrides.size());
  for (const std::string &kv : i64Overrides) {
    auto eq = kv.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= kv.size()) {
      llvm::errs() << "error: invalid --i64 override: " << kv
                   << " (expected: %argN=123)\n";
      return 2;
    }
    std::string key = trim(kv.substr(0, eq));
    std::string val = trim(kv.substr(eq + 1));
    if (!key.empty() && key.front() != '%')
      key.insert(key.begin(), '%');
    errno = 0;
    char *endp = nullptr;
    long long parsed = std::strtoll(val.c_str(), &endp, 10);
    if (errno != 0 || endp == val.c_str() || (endp && *endp != '\0')) {
      llvm::errs() << "error: invalid --i64 override value: " << kv << "\n";
      return 2;
    }
    int64_t v = static_cast<int64_t>(parsed);
    i64OverrideMap[key] = v;
  }

  std::string mlirText = readFileOrDie(nvvmMlirPath);
  ConstTables consts = parseConstants(mlirText);

  if (runAllKernels && !kernelName.getValue().empty()) {
    llvm::errs() << "error: --run-all-kernels is incompatible with --kernel\n";
    return 2;
  }

  std::vector<LaunchSpec> launches;
  launches.reserve(4);
  if (runAllKernels) {
    launches = parseAllLaunchSpecs(mlirText);
    if (launches.empty()) {
      llvm::errs() << "error: failed to find any gpu.launch_func in: "
                   << nvvmMlirPath << "\n";
      return 2;
    }
  } else {
    std::optional<LaunchSpec> launchOpt = parseFirstLaunchSpec(
        mlirText, kernelName.getValue().empty()
                      ? std::nullopt
                      : std::make_optional(kernelName.getValue()));
    if (!launchOpt) {
      llvm::errs() << "error: failed to find gpu.launch_func in: "
                   << nvvmMlirPath << "\n";
      return 2;
    }
    launches.push_back(std::move(*launchOpt));
  }

  std::vector<std::vector<MemrefDesc>> memrefsByLaunch;
  memrefsByLaunch.reserve(launches.size());

  std::vector<MemrefDesc> memrefs;
  memrefs.reserve(32);
  std::unordered_map<std::string, size_t> symToMemref;
  symToMemref.reserve(64);
  bool relaxedDescriptorMismatch = false;

  auto addMemref = [&](const MemrefDesc &d) {
    auto it = symToMemref.find(d.baseSym);
    if (it == symToMemref.end()) {
      size_t idx = memrefs.size();
      memrefs.push_back(d);
      symToMemref[d.baseSym] = idx;
      symToMemref[d.alignedSym] = idx;
      return;
    }
    size_t idx = it->second;
    MemrefDesc &cur = memrefs[idx];
    if (cur.rank != d.rank || cur.offset != d.offset || cur.sizes != d.sizes ||
        cur.strides != d.strides || cur.bytes != d.bytes) {
      if (!runAllKernels) {
        llvm::errs() << "error: inconsistent memref descriptor for " << d.baseSym
                     << " across kernels (rank/shape/stride mismatch)\n";
        std::exit(2);
      }
      relaxedDescriptorMismatch = true;
      int64_t maxBytes = std::max<int64_t>(cur.bytes, d.bytes);
      cur = d;
      cur.bytes = maxBytes;
    }
    symToMemref[d.baseSym] = idx;
    symToMemref[d.alignedSym] = idx;
  };

  for (const LaunchSpec &launch : launches) {
    std::vector<MemrefDesc> ms =
        detectMemrefDescriptors(launch, consts, i64OverrideMap, elemBytes);
    for (const MemrefDesc &d : ms)
      addMemref(d);
    memrefsByLaunch.push_back(std::move(ms));
  }

  if (relaxedDescriptorMismatch) {
    llvm::errs()
        << "warning: relaxed inconsistent memref descriptors across kernels "
           "(run-all mode); using merged device buffers\n";
  }

  if (listMemrefs) {
    if (launches.size() == 1) {
      for (size_t mi = 0; mi < memrefs.size(); ++mi) {
        const MemrefDesc &d = memrefs[mi];
        llvm::outs() << "memref[" << mi << "]: sym=" << d.baseSym
                     << " rank=" << d.rank << " bytes=" << d.bytes
                     << " offset=" << d.offset << " sizes=(" << d.sizes[0];
        if (d.rank >= 2)
          llvm::outs() << "," << d.sizes[1];
        llvm::outs() << ") strides=(" << d.strides[0];
        if (d.rank >= 2)
          llvm::outs() << "," << d.strides[1];
        llvm::outs() << ")\n";
      }
      return 0;
    }

    for (size_t ki = 0; ki < launches.size(); ++ki) {
      const LaunchSpec &k = launches[ki];
      llvm::outs() << "kernel[" << ki << "]: binary=" << k.binaryName
                   << " entry=" << k.entryName << "\n";
      const std::vector<MemrefDesc> &ms = memrefsByLaunch[ki];
      for (size_t mi = 0; mi < ms.size(); ++mi) {
        const MemrefDesc &d = ms[mi];
        llvm::outs() << "  memref[" << mi << "]: sym=" << d.baseSym
                     << " rank=" << d.rank << " bytes=" << d.bytes
                     << " offset=" << d.offset << " sizes=(" << d.sizes[0];
        if (d.rank >= 2)
          llvm::outs() << "," << d.sizes[1];
        llvm::outs() << ") strides=(" << d.strides[0];
        if (d.rank >= 2)
          llvm::outs() << "," << d.strides[1];
        llvm::outs() << ")\n";
      }
    }
    return 0;
  }

  if (elemBytes <= 0) {
    llvm::errs() << "error: --elem-bytes must be > 0\n";
    return 2;
  }

  // 初始化 CUDA driver 与 context。
  CU_CHECK(cuInit(0));
  CUdevice dev = 0;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx = nullptr;
  CU_CHECK(cuCtxCreate(&ctx, 0, dev));

  CUstream stream = nullptr;
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // 构建参数存储，并为 memref 分配设备缓冲。
  std::unordered_map<std::string, CUdeviceptr> ptrMap;
  ptrMap.reserve(memrefs.size() * 2 + 16);

  struct ArgStorage {
    // 所有参数值放在自有 vector 中，保证传给 `cuLaunchKernel`
    // 的地址稳定。
    std::vector<CUdeviceptr> ptrs;
    std::vector<uint8_t> i8s;
    std::vector<int64_t> i64s;
    std::vector<float> f32s;
    std::vector<uint16_t> f16s;
    std::vector<std::vector<uint8_t>> vecBytes;
    std::vector<void *> argv;
  };
  // 若符号尚未绑定，则分配并绑定设备指针。
  auto bindPtrIfMissing = [&](const std::string &sym,
                              int64_t bytes) -> CUdeviceptr {
    auto it = ptrMap.find(sym);
    if (it != ptrMap.end())
      return it->second;
    CUdeviceptr dptr = 0;
    CU_CHECK(cuMemAlloc(&dptr, static_cast<size_t>(bytes)));
    CU_CHECK(cuMemsetD8(dptr, 0, static_cast<size_t>(bytes)));
    ptrMap.insert({sym, dptr});
    return dptr;
  };

  // 第一轮：为识别到的 memref 描述符分配缓冲。
  for (const MemrefDesc &d : memrefs) {
    if (d.rank <= 0 || d.bytes <= 0)
      continue;
    CUdeviceptr dptr = bindPtrIfMissing(d.baseSym, d.bytes);
    if (ptrMap.find(d.alignedSym) == ptrMap.end())
      ptrMap.insert({d.alignedSym, dptr});
  }

  struct KernelRuntime {
    std::string binaryName;
    std::string entryName;
    int gridX = 1;
    int gridY = 1;
    int gridZ = 1;
    int blockX = 1;
    int blockY = 1;
    int blockZ = 1;
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    ArgStorage st;
  };

  std::unordered_map<std::string, CUmodule> moduleCache;
  moduleCache.reserve(launches.size() * 2 + 4);

  auto getOrLoadModule = [&](const std::string &binaryName) -> CUmodule {
    auto it = moduleCache.find(binaryName);
    if (it != moduleCache.end())
      return it->second;

    auto escapedAsmOpt =
        extractGpuBinaryAssemblyEscaped(mlirText, binaryName);
    if (!escapedAsmOpt) {
      llvm::errs() << "error: failed to find gpu.binary @" << binaryName
                   << " with assembly in: " << nvvmMlirPath << "\n";
      std::exit(2);
    }
    std::string ptx = decodeMlirStringEscapes(*escapedAsmOpt);
    if (ptx.find(".entry") == std::string::npos) {
      llvm::errs()
          << "error: decoded assembly doesn't look like PTX (.entry not found)\n";
      std::exit(2);
    }

    CUmodule mod = nullptr;
    constexpr size_t kJitLogCap = 1 << 16;
    std::vector<char> errorLog(kJitLogCap, '\0');
    std::vector<char> infoLog(kJitLogCap, '\0');
    CUjit_option jitOpts[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE,
    };
    void *jitVals[] = {
        errorLog.data(),
        reinterpret_cast<void *>(errorLog.size()),
        infoLog.data(),
        reinterpret_cast<void *>(infoLog.size()),
        reinterpret_cast<void *>(1),
    };
    unsigned int numJitOpts =
        static_cast<unsigned int>(sizeof(jitOpts) / sizeof(jitOpts[0]));
    CUresult loadRes =
        cuModuleLoadDataEx(&mod, ptx.c_str(), numJitOpts, jitOpts, jitVals);
    if (loadRes != CUDA_SUCCESS) {
      // 某些环境（尤其 CUDA driver JIT）在编译 PTX 时会强制 48KB shared-memory
      // 限制，尽管硬件/工具链 `ptxas` 支持更大的可选 shared 内存。
      // 此时回退为先用 `ptxas` 编译 cubin，再加载 cubin。
      if (auto cubin = compilePtxToCubinWithPtxas(ptx)) {
        CUmodule mod2 = nullptr;
        CUresult r2 = cuModuleLoadDataEx(&mod2, cubin->data(),
                                         /* numOptions=*/0, /*options=*/nullptr,
                                         /* optionValues=*/nullptr);
        if (r2 == CUDA_SUCCESS) {
          moduleCache.insert({binaryName, mod2});
          return mod2;
        }
      }

      const char *name = nullptr;
      const char *msg = nullptr;
      (void)cuGetErrorName(loadRes, &name);
      (void)cuGetErrorString(loadRes, &msg);
      llvm::errs() << "CUDA error in cuModuleLoadDataEx(binary="
                   << binaryName << "): " << (name ? name : "<unknown>") << " ("
                   << (msg ? msg : "<no message>") << ")\n";
      printCudaJitLog("CUDA JIT error log", errorLog.data(), errorLog.size());
      printCudaJitLog("CUDA JIT info log", infoLog.data(), infoLog.size());
      std::exit(1);
    }
    moduleCache.insert({binaryName, mod});
    return mod;
  };

  std::vector<KernelRuntime> kernels;
  kernels.reserve(launches.size());

  for (const LaunchSpec &launch : launches) {
    KernelRuntime k;
    k.binaryName = launch.binaryName;
    k.entryName = launch.entryName;
    k.gridX = static_cast<int>(
        evalI64Token(launch.gridTokens[0], consts, i64OverrideMap));
    k.gridY = static_cast<int>(
        evalI64Token(launch.gridTokens[1], consts, i64OverrideMap));
    k.gridZ = static_cast<int>(
        evalI64Token(launch.gridTokens[2], consts, i64OverrideMap));
    k.blockX = static_cast<int>(
        evalI64Token(launch.blockTokens[0], consts, i64OverrideMap));
    k.blockY = static_cast<int>(
        evalI64Token(launch.blockTokens[1], consts, i64OverrideMap));
    k.blockZ = static_cast<int>(
        evalI64Token(launch.blockTokens[2], consts, i64OverrideMap));

    if (k.gridX <= 0 || k.gridY <= 0 || k.gridZ <= 0 || k.blockX <= 0 ||
        k.blockY <= 0 || k.blockZ <= 0) {
      llvm::errs() << "error: invalid launch dims for kernel(binary)="
                   << k.binaryName << " entry=" << k.entryName << " grid=("
                   << k.gridX << "," << k.gridY << "," << k.gridZ << ") block=("
                   << k.blockX << "," << k.blockY << "," << k.blockZ << ")\n";
      return 2;
    }

    k.mod = getOrLoadModule(k.binaryName);
    CU_CHECK(cuModuleGetFunction(&k.fn, k.mod, k.entryName.c_str()));

    // 当 kernel 的静态 shared 内存超过 48KB 时，按 block 申请更大的 opt-in shared。
    // 这在 Ampere+ 设备上是必须的：默认每 block 上限是 48KB，但存在更高 opt-in 上限。
    int staticSmemBytes = 0;
    if (cuFuncGetAttribute(&staticSmemBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                           k.fn) == CUDA_SUCCESS) {
      int maxSmemDefault = 0;
      int maxSmemOptin = 0;
      (void)cuDeviceGetAttribute(&maxSmemDefault,
                                 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                 dev);
      (void)cuDeviceGetAttribute(
          &maxSmemOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          dev);
      if (maxSmemDefault > 0 && maxSmemOptin > maxSmemDefault &&
          staticSmemBytes > maxSmemDefault) {
        if (staticSmemBytes > maxSmemOptin) {
          llvm::errs() << "error: kernel static shared memory exceeds opt-in "
                          "limit: static_smem="
                       << staticSmemBytes << " optin_max=" << maxSmemOptin
                       << "\n";
          return 2;
        }
        int maxDyn = maxSmemOptin - staticSmemBytes;
        if (maxDyn < 0)
          maxDyn = 0;
        CU_CHECK(cuFuncSetAttribute(k.fn,
                                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    maxDyn));
        CU_CHECK(cuFuncSetAttribute(
            k.fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
      }
    }

    k.st.ptrs.reserve(launch.args.size());
    k.st.i64s.reserve(launch.args.size());
    k.st.f32s.reserve(launch.args.size());
    k.st.vecBytes.reserve(launch.args.size());
    k.st.argv.reserve(launch.args.size());

    // 按参数顺序依次实体化所有实参。
    for (const LaunchArg &a : launch.args) {
      if (a.type == "!llvm.ptr") {
        auto it = ptrMap.find(a.value);
        if (it == ptrMap.end()) {
          // 未识别的指针参数：先分配一个小缓冲兜底。
          bindPtrIfMissing(a.value, /*bytes=*/256);
          it = ptrMap.find(a.value);
        }
        k.st.ptrs.push_back(it->second);
        k.st.argv.push_back(&k.st.ptrs.back());
        continue;
      }
      if (a.type == "i1") {
        int64_t v = evalI64Token(a.value, consts, i64OverrideMap);
        k.st.i8s.push_back(static_cast<uint8_t>(v ? 1 : 0));
        k.st.argv.push_back(&k.st.i8s.back());
        continue;
      }
      if (a.type == "i64") {
        k.st.i64s.push_back(evalI64Token(a.value, consts, i64OverrideMap));
        k.st.argv.push_back(&k.st.i64s.back());
        continue;
      }
      if (a.type == "f32") {
        k.st.f32s.push_back(evalF32Token(a.value, consts));
        k.st.argv.push_back(&k.st.f32s.back());
        continue;
      }
      if (a.type == "f16") {
        float v = evalF16Token(a.value, consts);
        k.st.f16s.push_back(f32ToF16Bits(v));
        k.st.argv.push_back(&k.st.f16s.back());
        continue;
      }
      if (auto vec = parseVectorType(a.type)) {
        int64_t numElems = vec->first;
        int64_t elemBytes = vec->second;
        __int128 bytes128 =
            static_cast<__int128>(numElems) * static_cast<__int128>(elemBytes);
        if (bytes128 <= 0 || bytes128 > static_cast<__int128>(1ULL << 20)) {
          llvm::errs() << "error: unsupported vector arg size for type: "
                       << a.type << "\n";
          return 2;
        }
        int64_t bytes = static_cast<int64_t>(bytes128);
        k.st.vecBytes.emplace_back(static_cast<size_t>(bytes), 0);
        k.st.argv.push_back(k.st.vecBytes.back().data());
        continue;
      }

      llvm::errs() << "error: unsupported arg type: " << a.type
                   << " (value=" << a.value << ")\n";
      return 2;
    }

    kernels.push_back(std::move(k));
  }

  if (verbose) {
    for (size_t ki = 0; ki < kernels.size(); ++ki) {
      const KernelRuntime &k = kernels[ki];
      llvm::outs() << "kernel[" << ki << "](binary)=" << k.binaryName
                   << " entry=" << k.entryName << "\n";
      llvm::outs() << "  grid=(" << k.gridX << "," << k.gridY << "," << k.gridZ
                   << ") block=(" << k.blockX << "," << k.blockY << ","
                   << k.blockZ << ") args=" << launches[ki].args.size() << "\n";
    }
  }

  // 初始化选中的 memref 缓冲（host->device memcpy）。
  if (!initPtrs.empty() && elemBytes != 4) {
    llvm::errs()
        << "error: --init-ptr requires --elem-bytes=4 (f32) in this prototype\n";
    return 2;
  }
  if (!fillPtrs.empty() && elemBytes != 4) {
    llvm::errs()
        << "error: --fill requires --elem-bytes=4 (f32) in this prototype\n";
    return 2;
  }

  std::unordered_map<std::string, float> fillMap;
  fillMap.reserve(fillPtrs.size());
  for (const std::string &kv : fillPtrs) {
    auto p = parseKeyValueEq(kv);
    if (!p) {
      llvm::errs() << "error: invalid --fill spec: " << kv
                   << " (expected: %argN=<float>)\n";
      return 2;
    }
    auto f = parseF32Value(p->second);
    if (!f) {
      llvm::errs() << "error: invalid --fill value: " << kv
                   << " (expected: float / inf / -inf / 0xXXXXXXXX)\n";
      return 2;
    }
    fillMap[p->first] = *f;
  }

  for (const std::string &symRaw : initPtrs) {
    std::string sym = trim(symRaw);
    if (!sym.empty() && sym.front() != '%')
      sym.insert(sym.begin(), '%');
    auto mit = symToMemref.find(sym);
    if (mit == symToMemref.end()) {
      llvm::errs() << "error: --init-ptr " << sym
                   << " does not match any detected memref descriptor\n";
      return 2;
    }
    const MemrefDesc &d = memrefs[mit->second];
    if (d.bytes <= 0) {
      llvm::errs() << "error: cannot init memref with non-positive bytes: "
                   << sym << "\n";
      return 2;
    }
    if ((d.bytes % elemBytes) != 0) {
      llvm::errs() << "error: memref byte size is not aligned to elem-bytes: "
                   << sym << " bytes=" << d.bytes
                   << " elemBytes=" << elemBytes << "\n";
      return 2;
    }
    auto pit = ptrMap.find(sym);
    if (pit == ptrMap.end()) {
      llvm::errs() << "error: internal: missing device binding for " << sym
                   << "\n";
      return 2;
    }
    std::vector<float> host(static_cast<size_t>(d.bytes / elemBytes), 0.0f);
    initMemrefF32(d, host.data(), initMode.getValue(),
                  static_cast<uint64_t>(initSeed.getValue()));
    CU_CHECK(cuMemcpyHtoD(pit->second, host.data(),
                          static_cast<size_t>(d.bytes)));
  }

  for (const std::string &symRaw : initPtrsF16) {
    std::string sym = trim(symRaw);
    if (!sym.empty() && sym.front() != '%')
      sym.insert(sym.begin(), '%');
    auto mit = symToMemref.find(sym);
    if (mit == symToMemref.end()) {
      llvm::errs() << "error: --init-ptr-f16 " << sym
                   << " does not match any detected memref descriptor\n";
      return 2;
    }
    const MemrefDesc &d = memrefs[mit->second];
    if (d.bytes <= 0) {
      llvm::errs() << "error: cannot init memref with non-positive bytes: "
                   << sym << "\n";
      return 2;
    }
    auto elemCountOr = inferMemrefElementCount(d);
    if (!elemCountOr || *elemCountOr <= 0) {
      llvm::errs() << "error: cannot infer element count for --init-ptr-f16 "
                   << sym << "\n";
      return 2;
    }
    int64_t elemCount = *elemCountOr;
    size_t copyBytes =
        static_cast<size_t>(elemCount) * static_cast<size_t>(sizeof(uint16_t));
    if (static_cast<int64_t>(copyBytes) > d.bytes) {
      llvm::errs() << "error: inferred f16 init size exceeds memref bytes: "
                   << sym << " init_bytes=" << static_cast<long long>(copyBytes)
                   << " memref_bytes=" << d.bytes << "\n";
      return 2;
    }
    auto pit = ptrMap.find(sym);
    if (pit == ptrMap.end()) {
      llvm::errs() << "error: internal: missing device binding for " << sym
                   << "\n";
      return 2;
    }
    std::vector<uint16_t> host(static_cast<size_t>(elemCount), 0);
    initMemrefF16(d, host.data(), initMode.getValue(),
                  static_cast<uint64_t>(initSeed.getValue()));
    CU_CHECK(cuMemcpyHtoD(pit->second, host.data(), copyBytes));
  }

  struct FillAction {
    std::string sym;
    CUdeviceptr dptr = 0;
    size_t countD32 = 0;
    unsigned int pattern = 0;
  };
  std::vector<FillAction> fillActions;
  fillActions.reserve(fillMap.size());

  // 预先构建 fill 动作，便于按需在每轮迭代重复应用。
  for (const auto &kv : fillMap) {
    std::string sym = trim(kv.first);
    if (!sym.empty() && sym.front() != '%')
      sym.insert(sym.begin(), '%');
    float fill = kv.second;
    auto mit = symToMemref.find(sym);
    if (mit == symToMemref.end()) {
      llvm::errs() << "error: --fill " << sym
                   << " does not match any detected memref descriptor\n";
      return 2;
    }
    const MemrefDesc &d = memrefs[mit->second];
    if (d.bytes <= 0) {
      llvm::errs() << "error: cannot fill memref with non-positive bytes: "
                   << sym << "\n";
      return 2;
    }
    if ((d.bytes % elemBytes) != 0) {
      llvm::errs() << "error: memref byte size is not aligned to elem-bytes: "
                   << sym << " bytes=" << d.bytes
                   << " elemBytes=" << elemBytes << "\n";
      return 2;
    }
    if (elemBytes != 4) {
      llvm::errs()
          << "error: internal: fillActions requires elemBytes=4 (f32)\n";
      return 2;
    }
    if ((d.bytes % 4) != 0) {
      llvm::errs() << "error: memref byte size is not 4-byte aligned for "
                      "--fill-each-iter memset: "
                   << sym << " bytes=" << d.bytes << "\n";
      return 2;
    }
    auto pit = ptrMap.find(sym);
    if (pit == ptrMap.end()) {
      llvm::errs() << "error: internal: missing device binding for " << sym
                   << "\n";
      return 2;
    }
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(fill), "f32 must be 32-bit");
    std::memcpy(&bits, &fill, sizeof(bits));
    FillAction a;
    a.sym = sym;
    a.dptr = pit->second;
    a.countD32 = static_cast<size_t>(d.bytes / 4);
    a.pattern = static_cast<unsigned int>(bits);
    fillActions.push_back(std::move(a));
  }

  auto applyFillActions = [&]() {
    for (const FillAction &a : fillActions) {
      CU_CHECK(cuMemsetD32Async(a.dptr, a.pattern, a.countD32, stream));
    }
  };

  // 若用户未请求“每轮填充”，则在 warmup 前只执行一次 fill。
  if (!fillActions.empty() && !fillEachIter.getValue()) {
    applyFillActions();
  }

  // 预热。
  int warm = std::max(0, warmup.getValue());
  for (int i = 0; i < warm; ++i) {
    if (!fillActions.empty() && fillEachIter.getValue()) {
      applyFillActions();
    }
    for (KernelRuntime &k : kernels) {
      CU_CHECK(cuLaunchKernel(k.fn, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY,
                              k.blockZ,
                              /* sharedMemBytes=*/0, stream, k.st.argv.data(),
                              /* extra=*/nullptr));
    }
  }
  CU_CHECK(cuStreamSynchronize(stream));

  // 使用 CUDA event 计时。
  int n = std::max(1, iters.getValue());
  CUevent evStart = nullptr;
  CUevent evStop = nullptr;
  CU_CHECK(cuEventCreate(&evStart, CU_EVENT_DEFAULT));
  CU_CHECK(cuEventCreate(&evStop, CU_EVENT_DEFAULT));

  CU_CHECK(cuEventRecord(evStart, stream));
  for (int i = 0; i < n; ++i) {
    if (!fillActions.empty() && fillEachIter.getValue()) {
      applyFillActions();
    }
    for (KernelRuntime &k : kernels) {
      CU_CHECK(cuLaunchKernel(k.fn, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY,
                              k.blockZ,
                              /* sharedMemBytes=*/0, stream, k.st.argv.data(),
                              /* extra=*/nullptr));
    }
  }
  CU_CHECK(cuEventRecord(evStop, stream));
  CU_CHECK(cuEventSynchronize(evStop));

  float ms = 0.0f;
  CU_CHECK(cuEventElapsedTime(&ms, evStart, evStop));
  float avgMs = ms / static_cast<float>(n);

  llvm::outs() << "avg_ms=" << avgMs << " iters=" << n << " warmup=" << warm
               << "\n";

  // 导出请求的 memref（device->host memcpy + 元数据 JSON）。
  std::unordered_map<std::string, std::string> dumpMap;
  dumpMap.reserve(dumpMemrefs.size() + 2);
  for (const std::string &kv : dumpMemrefs) {
    auto p = parseKeyValueEq(kv);
    if (!p) {
      llvm::errs() << "error: invalid --dump spec: " << kv
                   << " (expected: %argN=/path/file.bin)\n";
      return 2;
    }
    dumpMap[p->first] = p->second;
  }

  if (!dumpLast2D.getValue().empty()) {
    int last = -1;
    const std::vector<MemrefDesc> &lastKernelMemrefs = memrefsByLaunch.back();
    for (int i = static_cast<int>(lastKernelMemrefs.size()) - 1; i >= 0; --i) {
      if (lastKernelMemrefs[static_cast<size_t>(i)].rank == 2) {
        last = i;
        break;
      }
    }
    if (last < 0) {
      llvm::errs() << "error: --dump-last-2d requested but no rank-2 memref "
                      "descriptor was detected\n";
      return 2;
    }
    dumpMap[lastKernelMemrefs[static_cast<size_t>(last)].baseSym] =
        dumpLast2D.getValue();
  }

  for (const auto &kv : dumpMap) {
    std::string sym = trim(kv.first);
    if (!sym.empty() && sym.front() != '%')
      sym.insert(sym.begin(), '%');
    const std::string &path = kv.second;
    auto mit = symToMemref.find(sym);
    if (mit == symToMemref.end()) {
      llvm::errs() << "error: --dump " << sym
                   << " does not match any detected memref descriptor\n";
      return 2;
    }
    const MemrefDesc &d = memrefs[mit->second];
    auto pit = ptrMap.find(sym);
    if (pit == ptrMap.end()) {
      llvm::errs() << "error: internal: missing device binding for dump sym "
                   << sym << "\n";
      return 2;
    }
    std::vector<uint8_t> host(static_cast<size_t>(d.bytes), 0);
    CU_CHECK(cuMemcpyDtoH(host.data(), pit->second,
                          static_cast<size_t>(d.bytes)));
    std::ofstream bin(path, std::ios::out | std::ios::binary);
    if (!bin) {
      llvm::errs() << "error: cannot open dump path: " << path << "\n";
      return 2;
    }
    bin.write(reinterpret_cast<const char *>(host.data()),
              static_cast<std::streamsize>(host.size()));
    bin.close();
    std::string metaPath = path + ".json";
    if (!writeJsonMetaForDump(metaPath, d, elemBytes)) {
      llvm::errs() << "error: cannot write dump metadata: " << metaPath << "\n";
      return 2;
    }
    if (verbose) {
      llvm::outs() << "dumped " << sym << " -> " << path << " (" << d.bytes
                   << " bytes)\n";
    }
  }

  // 清理资源。
  CU_CHECK(cuEventDestroy(evStart));
  CU_CHECK(cuEventDestroy(evStop));
  CU_CHECK(cuStreamDestroy(stream));
  for (const auto &kv : moduleCache)
    CU_CHECK(cuModuleUnload(kv.second));
  std::unordered_set<CUdeviceptr> freed;
  freed.reserve(ptrMap.size());
  for (const auto &kv : ptrMap) {
    if (!freed.insert(kv.second).second)
      continue;
    (void)cuMemFree(kv.second);
  }
  CU_CHECK(cuCtxDestroy(ctx));

  return 0;
}

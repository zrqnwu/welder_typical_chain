static std::string shellEscapeSingleQuotes(const std::string &s) {
  // 用单引号包裹，并转义内部单引号。
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\'')
      out.append("'\\''");
    else
      out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

static std::optional<std::string> makeTempDir(std::string_view prefix) {
  std::string tmpl = "/tmp/";
  tmpl.append(prefix);
  tmpl.append("_XXXXXX");
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  char *res = ::mkdtemp(buf.data());
  if (!res)
    return std::nullopt;
  return std::string(res);
}

struct TempDirGuard {
  std::string path;
  bool keep = false;

  ~TempDirGuard() {
    if (keep || path.empty())
      return;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

static bool writeModuleToFile(ModuleOp module, llvm::StringRef path,
                              std::string *err) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    if (err)
      * err = ec.message();
    return false;
  }
  module.print(os);
  os << "\n";
  return true;
}

static int runShellCommand(const std::string &cmd) {
  int rc = std::system(cmd.c_str());
  if (rc == -1)
    return -1;
  if (WIFEXITED(rc))
    return WEXITSTATUS(rc);
  return -1;
}

static bool
isRetryableVectorMaskCompileFailure(llvm::StringRef compileLog) {
  return compileLog.contains("'vector.mask' op expects only one operation to mask");
}

static bool
isRetryableWorkgroupPackLayoutCompileFailure(llvm::StringRef compileLog) {
  return compileLog.contains("workgroup pack requires consistent layout");
}

static bool
isRetryableKernelRootCompileFailure(llvm::StringRef compileLog) {
  return compileLog.contains(
      "--codegen-from-kernel-attrs found no linalg ops with welder.kernel_root");
}

static bool
isRetryableUnsupportedTargetCompileFailure(llvm::StringRef compileLog) {
  return compileLog.contains("unsupported target op:");
}

static bool isRetryableRowReductionCompileFailure(llvm::StringRef compileLog) {
  // 与 bench 侧 prebufferize 行归约失败的重试策略保持一致，
  // 让性能测量/代码生成搜索可恢复，避免把有价值候选全部剪掉。
  return compileLog.contains("Attempted to vectorize, but failed") ||
         compileLog.contains(
             "scf.forall mapping attribute size must match op rank") ||
         isRetryableVectorMaskCompileFailure(compileLog) ||
         compileLog.contains("prebufferize transform failed");
}

static bool
isRetryableParallelResourceOverflowCompileFailure(llvm::StringRef compileLog) {
  return compileLog.contains("the number of required parallel resources") &&
         compileLog.contains("overflows the number of available resources") &&
         compileLog.contains("postbufferize transform failed");
}

static bool
isRetryableConnectLevelCompileFailure(llvm::StringRef compileLog) {
  const bool retryByConnectHandle =
      compileLog.contains("requires exactly one containing_op handle") &&
      compileLog.contains("postbufferize transform failed");
  return retryByConnectHandle ||
         isRetryableWorkgroupPackLayoutCompileFailure(compileLog) ||
         isRetryableKernelRootCompileFailure(compileLog) ||
         isRetryableUnsupportedTargetCompileFailure(compileLog);
}

static bool
isRetryableThreadFuseCompileFailure(llvm::StringRef compileLog) {
  const bool retryByThreadFusePattern =
      compileLog.contains(
          "could not find next producer to fuse into container") &&
      compileLog.contains("postbufferize transform failed");
  return retryByThreadFusePattern ||
         isRetryableWorkgroupPackLayoutCompileFailure(compileLog) ||
         isRetryableUnsupportedTargetCompileFailure(compileLog) ||
         isRetryableKernelRootCompileFailure(compileLog);
}

static void eraseAllSubstrInPlace(std::string &s, llvm::StringRef needle) {
  if (needle.empty())
    return;
  std::string n = needle.str();
  size_t pos = 0;
  while ((pos = s.find(n, pos)) != std::string::npos) {
    s.erase(pos, n.size());
  }
}

static void eraseArgWithPrefixInPlace(std::string &s, llvm::StringRef prefix) {
  if (prefix.empty())
    return;
  std::string p = prefix.str();
  while (true) {
    size_t pos = s.find(p);
    if (pos == std::string::npos)
      break;
    size_t end = s.find(" --", pos + 1);
    size_t redir = s.find(" >", pos + 1);
    if (redir != std::string::npos &&
        (end == std::string::npos || redir < end)) {
      end = redir;
    }
    if (end == std::string::npos)
      end = s.size();
    s.erase(pos, end - pos);
  }
}

static bool replaceArgWithPrefixInPlace(std::string &s, llvm::StringRef prefix,
                                        llvm::StringRef value) {
  if (prefix.empty())
    return false;
  std::string p = prefix.str();
  size_t pos = s.find(p);
  if (pos == std::string::npos)
    return false;
  size_t valueBeg = pos + p.size();
  size_t valueEnd = s.find(' ', valueBeg);
  if (valueEnd == std::string::npos)
    valueEnd = s.size();
  s.replace(valueBeg, valueEnd - valueBeg, value.str());
  return true;
}

static void injectEnvBeforeBashInPlace(std::string &cmd,
                                       llvm::StringRef envAssignments);

static void appendArgBeforeRedirectInPlace(std::string &cmd,
                                           llvm::StringRef arg) {
  if (arg.empty())
    return;
  const std::string toInsert = arg.str();
  size_t redirPos = cmd.rfind(" > ");
  if (redirPos == std::string::npos) {
    cmd.append(toInsert);
    return;
  }
  cmd.insert(redirPos, toInsert);
}

static std::string buildConnectLevelRetryCompileCmd(std::string compileCmd,
                                                    int64_t connectLevel) {
  const int64_t cl = std::max<int64_t>(1, connectLevel);
  (void)replaceArgWithPrefixInPlace(compileCmd, " --max-connect-level=",
                                    std::to_string(cl));
  return compileCmd;
}

static std::string buildTcAsyncRecoveryRetryCompileCmd(
    std::string compileCmd, int64_t connectLevel, int64_t threadTileM,
    int64_t threadTileN, int64_t rowReductionThreadsX, bool disableRowWarp,
    bool forceTcSafeRowReduction) {
  compileCmd = buildConnectLevelRetryCompileCmd(std::move(compileCmd),
                                                /* connectLevel=*/connectLevel);
  if (threadTileM > 0) {
    (void)replaceArgWithPrefixInPlace(compileCmd, " --thread-tile-m ",
                                      std::to_string(threadTileM));
  }
  if (threadTileN > 0) {
    (void)replaceArgWithPrefixInPlace(compileCmd, " --thread-tile-n ",
                                      std::to_string(threadTileN));
  }
  if (rowReductionThreadsX > 0) {
    eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-threads-x=");
    appendArgBeforeRedirectInPlace(
        compileCmd, (" --row-reduction-threads-x=" +
                     std::to_string(std::max<int64_t>(1, rowReductionThreadsX))));
  }
  if (disableRowWarp) {
    eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-warp");
  }
  std::string env;
  if (disableRowWarp) {
    env.append("WELDER_MM_SM_TC_DISABLE_ROW_WARP=1 ");
  }
  if (forceTcSafeRowReduction) {
    env.append("WELDER_FORCE_MM_SM_FAST_ROW=0 ");
    env.append("WELDER_FORCE_MM_SM_TC_SAFE_ROW=1 ");
  }
  injectEnvBeforeBashInPlace(compileCmd, env);
  return compileCmd;
}

static std::string buildTcAsyncWaitRowSafeRetryCompileCmd(
    std::string compileCmd, int64_t connectLevel, int64_t threadTileM,
    int64_t threadTileN, int64_t rowReductionThreadsX, bool forceWaitGroups) {
  compileCmd = buildTcAsyncRecoveryRetryCompileCmd(
      std::move(compileCmd), connectLevel, threadTileM, threadTileN,
      rowReductionThreadsX, /*disableRowWarp=*/true,
      /* forceTcSafeRowReduction=*/true);
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-vectorize");
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-relax-barriers");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-skip-combine-barrier");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-combine-vectorize");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-input-promotion-vectorize");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-vector-width=");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-input-vector-width=");
  if (rowReductionThreadsX > 0) {
    eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-threads-x=");
    appendArgBeforeRedirectInPlace(
        compileCmd, (" --row-reduction-threads-x=" +
                     std::to_string(std::max<int64_t>(1, rowReductionThreadsX))));
  }
  (void)replaceArgWithPrefixInPlace(compileCmd, " --pipeline-depth ", "2");
  if (compileCmd.find(" --pipeline-depth ") == std::string::npos) {
    appendArgBeforeRedirectInPlace(compileCmd, " --pipeline-depth 2");
  }
  if (forceWaitGroups) {
    if (compileCmd.find(" --pipeline-set-async-wait-groups") ==
        std::string::npos) {
      appendArgBeforeRedirectInPlace(compileCmd,
                                     " --pipeline-set-async-wait-groups");
    }
  } else {
    eraseAllSubstrInPlace(compileCmd, " --pipeline-set-async-wait-groups");
  }
  std::string env =
      "ENABLE_SOFTWARE_PIPELINING=1 PIPELINE_DEPTH=2 "
      "WORKGROUP_MULTIBUFFER_DEPTH=2 "
      "WELDER_MM_SM_TC_DISABLE_ROW_WARP=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_COMBINE_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_INPUT_PROMO_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_RELAX_BARRIERS=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_SKIP_COMBINE_BARRIER=1 ";
  env.append("PIPELINE_SET_ASYNC_WAIT_GROUPS=");
  env.append(forceWaitGroups ? "1 " : "0 ");
  injectEnvBeforeBashInPlace(compileCmd, env);
  return compileCmd;
}

static std::string buildWorkgroupPackLayoutSafeRetryCompileCmd(
    std::string compileCmd, int64_t connectLevel, bool tensorCoreCandidate) {
  compileCmd = buildConnectLevelRetryCompileCmd(std::move(compileCmd),
                                                /* connectLevel=*/connectLevel);
  (void)replaceArgWithPrefixInPlace(compileCmd, "WORKGROUP_SWIZZLE_XOR=", "0");
  (void)replaceArgWithPrefixInPlace(compileCmd, "BLOCK_RASTERIZE_XOR=", "0");
  (void)replaceArgWithPrefixInPlace(compileCmd, "BLOCK_RASTERIZE_MODE=", "0");
  (void)replaceArgWithPrefixInPlace(compileCmd, "BLOCK_RASTERIZE_PANEL_WIDTH=",
                                    "0");
  (void)replaceArgWithPrefixInPlace(compileCmd,
                                    "WORKGROUP_PAD_LAST_DIM_MATMUL_ONLY=", "0");
  if (tensorCoreCandidate) {
    const bool hasPadToken =
        replaceArgWithPrefixInPlace(compileCmd, "WORKGROUP_PAD_LAST_DIM=", "8");
    if (!hasPadToken) {
      injectEnvBeforeBashInPlace(compileCmd, "WORKGROUP_PAD_LAST_DIM=8 ");
    }
  }
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-vectorize");
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-combine-vectorize");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-input-promotion-vectorize");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-vector-width=");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-input-vector-width=");
  injectEnvBeforeBashInPlace(
      compileCmd,
      "WELDER_MM_SM_TC_DISABLE_ROW_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_COMBINE_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_INPUT_PROMO_VECTORIZE=1 ");
  return compileCmd;
}

static std::string
buildSafeRowReductionRetryCompileCmd(std::string compileCmd,
                                     bool dropMatmulSoftmaxSharedReuse,
                                     bool dropRowReductionReuseFusion,
                                     bool forceFastRowReduction,
                                     bool forceTcSafeRowReduction) {
  if (dropRowReductionReuseFusion) {
    eraseAllSubstrInPlace(compileCmd,
                          " --enable-row-reduction-chain-reuse-fusion");
    eraseAllSubstrInPlace(compileCmd,
                          " --reduction-chain-split-broadcast-edges=false");
  }
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-input-promotion");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-input-promotion-vectorize");
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-warp");
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-vectorize");
  eraseAllSubstrInPlace(compileCmd, " --enable-row-reduction-relax-barriers");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-skip-combine-barrier");
  eraseAllSubstrInPlace(compileCmd,
                        " --enable-row-reduction-combine-vectorize");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-vector-width=");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-threads-x=");
  eraseArgWithPrefixInPlace(compileCmd, " --row-reduction-input-vector-width=");
  if (dropMatmulSoftmaxSharedReuse) {
    eraseAllSubstrInPlace(compileCmd,
                          " --enable-matmul-softmax-shared-reuse-fusion");
  }
  std::string retryCmd;
  retryCmd.reserve(compileCmd.size() + 96);
  retryCmd.append("WELDER_FORCE_MM_SM_FAST_ROW=");
  retryCmd.append(forceFastRowReduction ? "1 " : "0 ");
  retryCmd.append("WELDER_FORCE_MM_SM_TC_SAFE_ROW=");
  retryCmd.append(forceTcSafeRowReduction ? "1 " : "0 ");
  retryCmd.append(compileCmd);
  return retryCmd;
}

static void injectEnvBeforeBashInPlace(std::string &cmd,
                                       llvm::StringRef envAssignments) {
  if (envAssignments.empty())
    return;
  std::string assigns = envAssignments.str();
  if (assigns.back() != ' ')
    assigns.push_back(' ');
  size_t bashPos = cmd.find(" bash ");
  if (bashPos == std::string::npos) {
    cmd.insert(0, assigns);
    return;
  }
  cmd.insert(bashPos + 1, assigns);
}

static std::string
buildTensorCoreMissingMmaRetryCompileCmd(std::string compileCmd) {
  compileCmd = buildSafeRowReductionRetryCompileCmd(
      std::move(compileCmd), /*dropMatmulSoftmaxSharedReuse=*/false,
      /* dropRowReductionReuseFusion=*/false,
      /* forceFastRowReduction=*/false, /*forceTcSafeRowReduction=*/true);
  eraseAllSubstrInPlace(compileCmd, " --enable-async-copy");
  eraseAllSubstrInPlace(compileCmd, " --enable-software-pipelining");
  eraseAllSubstrInPlace(compileCmd, " --pipeline-set-async-wait-groups");
  eraseArgWithPrefixInPlace(compileCmd, " --pipeline-depth ");
  eraseAllSubstrInPlace(compileCmd, " --pipeline-peel-epilogue=false");
  injectEnvBeforeBashInPlace(
      compileCmd,
      "ENABLE_SOFTWARE_PIPELINING=0 PIPELINE_SET_ASYNC_WAIT_GROUPS=0 "
      "WORKGROUP_MULTIBUFFER_DEPTH=1 WELDER_MM_SM_TC_DISABLE_ROW_WARP=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_COMBINE_VECTORIZE=1 "
      "WELDER_MM_SM_TC_DISABLE_ROW_INPUT_PROMO_VECTORIZE=1");
  return compileCmd;
}

static std::string wrapWithTimeoutIfRequested(const std::string &cmd,
                                              int timeoutSec) {
  if (timeoutSec <= 0)
    return cmd;
  // 尽力实现：依赖系统可用 `coreutils timeout`。
  //
  // 重要：`cmd` 常以环境变量赋值开头（如 `OUT_DIR=... bash ...`）。
  // 若直接前置 `timeout ...`，`timeout` 会把 `OUT_DIR=...` 误当可执行程序，
  // 导致 rc=127。这里统一用 `bash -lc` 包裹整条命令。
  std::string out;
  out.reserve(cmd.size() + 128);
  out.append("timeout -k 1 ");
  out.append(std::to_string(timeoutSec));
  out.append(" bash -lc ");
  out.append(shellEscapeSingleQuotes(cmd));
  return out;
}

static std::optional<double> parseAvgMsFromProfilerOutput(const std::string &s) {
  // 期望格式："avg_ms=<float> iters=... warmup=..."
  size_t pos = s.find("avg_ms=");
  if (pos == std::string::npos)
    return std::nullopt;
  pos += std::string("avg_ms=").size();
  size_t end = pos;
  while (end < s.size() &&
         (std::isdigit(static_cast<unsigned char>(s[end])) || s[end] == '.' ||
          s[end] == 'e' || s[end] == 'E' || s[end] == '+' || s[end] == '-')) {
    ++end;
  }
  if (end <= pos)
    return std::nullopt;
  double v = std::strtod(s.substr(pos, end - pos).c_str(), nullptr);
  if (!(v > 0.0))
    return std::nullopt;
  return v;
}

static std::optional<std::vector<std::pair<std::string, int64_t>>>
inferProfilerI64OverridesFromMainFunc(ModuleOp module) {
  if (!module)
    return std::nullopt;
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    // 尽力回退：选择模块中的第一个 `func.func`。
    module.walk([&](mlir::func::FuncOp f) {
      if (!main)
        main = f;
    });
  }
  if (!main)
    return std::nullopt;

  std::vector<std::pair<std::string, int64_t>> out;
  // 在 NVVM runnable 模块中，memref 参数会被降成扁平签名：
  // 其展开形式为 `(base_ptr, aligned_ptr, offset, sizes[rank], strides[rank])`，
  // 即每个 memref 对应 `3 + 2*rank` 个 LLVM 层参数。
  int64_t llvmArgBase = 0;
  for (BlockArgument arg : main.getArguments()) {
    Type t = arg.getType();
    SmallVector<int64_t, 4> dims;
    if (auto rt = dyn_cast<RankedTensorType>(t)) {
      if (!rt.hasStaticShape())
        return std::nullopt;
      dims.assign(rt.getShape().begin(), rt.getShape().end());
    } else if (auto mr = dyn_cast<MemRefType>(t)) {
      if (!mr.hasStaticShape())
        return std::nullopt;
      dims.assign(mr.getShape().begin(), mr.getShape().end());
    } else {
      // 当前性能测量 harness 不期望出现非 tensor 参数。
      return std::nullopt;
    }

    int64_t rank = static_cast<int64_t>(dims.size());
    if (rank <= 0)
      return std::nullopt;
    for (int64_t d : dims) {
      if (d <= 0)
        return std::nullopt;
    }

    // offset 固定为 0。
    out.emplace_back("%arg" + std::to_string(llvmArgBase + 2), 0);

    // sizes（按 row-major、identity layout 解释）。
    for (int64_t i = 0; i < rank; ++i) {
      out.emplace_back("%arg" + std::to_string(llvmArgBase + 3 + i), dims[i]);
    }

    // strides：最后一维为 1，`stride[i] = product(dims[i+1:])`。
    SmallVector<int64_t, 4> strides(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) {
      if (dims[i + 1] != 0 &&
          strides[i + 1] > (std::numeric_limits<int64_t>::max() / dims[i + 1]))
        return std::nullopt;
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    for (int64_t i = 0; i < rank; ++i) {
      out.emplace_back(
          "%arg" + std::to_string(llvmArgBase + 3 + rank + i), strides[i]);
    }

    llvmArgBase += (3 + 2 * rank);
  }

  return out;
}

static std::optional<std::vector<std::string>>
inferProfilerInitPtrSymsFromMainFunc(ModuleOp module) {
  if (!module)
    return std::nullopt;
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    // 尽力回退：选择模块中的第一个 `func.func`。
    module.walk([&](mlir::func::FuncOp f) {
      if (!main)
        main = f;
    });
  }
  if (!main)
    return std::nullopt;

  std::vector<std::string> out;
  // 在 NVVM runnable 模块中，memref 参数会被降成扁平签名：
  // 其展开形式为 `(base_ptr, aligned_ptr, offset, sizes[rank], strides[rank])`，
  // 即每个 memref 对应 `3 + 2*rank` 个 LLVM 层参数。
  int64_t llvmArgBase = 0;
  for (BlockArgument arg : main.getArguments()) {
    Type t = arg.getType();
    SmallVector<int64_t, 4> dims;
    if (auto rt = dyn_cast<RankedTensorType>(t)) {
      if (!rt.hasStaticShape())
        return std::nullopt;
      dims.assign(rt.getShape().begin(), rt.getShape().end());
    } else if (auto mr = dyn_cast<MemRefType>(t)) {
      if (!mr.hasStaticShape())
        return std::nullopt;
      dims.assign(mr.getShape().begin(), mr.getShape().end());
    } else {
      // 当前性能测量 harness 不期望出现非 tensor 参数。
      return std::nullopt;
    }

    int64_t rank = static_cast<int64_t>(dims.size());
    if (rank <= 0)
      return std::nullopt;
    for (int64_t d : dims) {
      if (d <= 0)
        return std::nullopt;
    }

    // 该 memref 描述符对应的基地址 token。
    out.push_back("%arg" + std::to_string(llvmArgBase));

    llvmArgBase += (3 + 2 * rank);
  }
  return out;
}

static bool moduleHasMaximumFReduction(ModuleOp module) {
  if (!module)
    return false;
  bool found = false;
  module.walk([&](arith::MaximumFOp) { found = true; });
  return found;
}

struct ListedMemrefInfo {
  int rank = 0;
  int minKernel = std::numeric_limits<int>::max();
};

static std::unordered_map<std::string, ListedMemrefInfo>
parseProfilerListMemrefsOutput(const std::string &text) {
  std::unordered_map<std::string, ListedMemrefInfo> out;
  out.reserve(32);

  std::istringstream iss(text);
  std::string line;
  int curKernel = 0;
  while (std::getline(iss, line)) {
    // 行头格式示例：`kernel[<n>]`。
    if (line.rfind("kernel[", 0) == 0) {
      size_t lb = line.find('[');
      size_t rb = line.find(']');
      if (lb != std::string::npos && rb != std::string::npos && rb > lb + 1) {
        int k = std::atoi(line.substr(lb + 1, rb - (lb + 1)).c_str());
        if (k >= 0)
          curKernel = k;
      }
      continue;
    }

    size_t symPos = line.find("sym=");
    size_t rankPos = line.find("rank=");
    if (symPos == std::string::npos || rankPos == std::string::npos)
      continue;

    symPos += 4;
    if (symPos >= line.size())
      continue;
    size_t symEnd = line.find_first_of(" \t\r\n", symPos);
    if (symEnd == std::string::npos)
      symEnd = line.size();
    std::string sym = line.substr(symPos, symEnd - symPos);
    if (sym.empty())
      continue;

    rankPos += 5;
    if (rankPos >= line.size())
      continue;
    size_t rankEnd = line.find_first_of(" \t\r\n", rankPos);
    if (rankEnd == std::string::npos)
      rankEnd = line.size();
    int rank = std::atoi(line.substr(rankPos, rankEnd - rankPos).c_str());
    if (rank <= 0)
      continue;

    auto it = out.find(sym);
    if (it == out.end()) {
      ListedMemrefInfo info;
      info.rank = rank;
      info.minKernel = curKernel;
      out.emplace(std::move(sym), info);
      continue;
    }
    it->second.minKernel = std::min(it->second.minKernel, curKernel);
  }
  return out;
}

static std::vector<std::pair<std::string, std::string>>
inferFillSpecsFromListedMemrefs(
    const std::unordered_map<std::string, ListedMemrefInfo> &memrefs,
    const std::unordered_set<std::string> &skipSyms, bool hasMaxReduction) {
  std::vector<std::pair<std::string, std::string>> fills;
  fills.reserve(memrefs.size());

  // Rank-2：只填充最早 kernel 中出现的缓冲（通常是 matmul C 的初始化缓冲）。
  // 这样测得的 memset 流量更接近原始 host 侧 `linalg.fill`，
  // 也可避免为后续 kernel 中会被完全覆盖的中间缓冲付出额外代价。
  for (const auto &kv : memrefs) {
    const std::string &sym = kv.first;
    const ListedMemrefInfo &info = kv.second;
    if (skipSyms.count(sym))
      continue;
    if (info.rank == 2 && info.minKernel == 0) {
      fills.emplace_back(sym, "0");
    }
  }

  // Rank-1：默认填 0。若图中存在 maximumf 归约（softmax），
  // 则把最早出现的 rank-1 缓冲初始化为 -inf。
  std::string maxSym;
  if (hasMaxReduction) {
    int bestK = std::numeric_limits<int>::max();
    for (const auto &kv : memrefs) {
      const std::string &sym = kv.first;
      const ListedMemrefInfo &info = kv.second;
      if (skipSyms.count(sym))
        continue;
      if (info.rank != 1)
        continue;
      if (info.minKernel < bestK) {
        bestK = info.minKernel;
        maxSym = sym;
      }
    }
  }

  for (const auto &kv : memrefs) {
    const std::string &sym = kv.first;
    const ListedMemrefInfo &info = kv.second;
    if (skipSyms.count(sym))
      continue;
    if (info.rank != 1)
      continue;
    if (!maxSym.empty() && sym == maxSym) {
      fills.emplace_back(sym, "-inf");
    } else {
      fills.emplace_back(sym, "0");
    }
  }

  // 为缓存/调试保持稳定顺序。
  std::sort(fills.begin(), fills.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  fills.erase(std::unique(fills.begin(), fills.end(),
                          [](const auto &a, const auto &b) {
                            return a.first == b.first && a.second == b.second;
                          }),
              fills.end());
  return fills;
}

static std::string buildProfileKeyForSubgraph(const TileGraph &graph,
                                              const PaperSubgraph &sg,
                                              int sinkNodeIdx,
                                              const Candidate &cand) {
  std::string key;
  key.reserve(256);
  key.append("sink=");
  key.append(std::to_string(sinkNodeIdx));
  key.append("|tm=");
  key.append(std::to_string(cand.tileM));
  key.append("|tn=");
  key.append(std::to_string(cand.tileN));
  key.append("|tk=");
  key.append(std::to_string(cand.tileK));
  key.append("|ttm=");
  key.append(std::to_string(cand.threadTileM));
  key.append("|ttn=");
  key.append(std::to_string(cand.threadTileN));
  key.append("|ac=");
  key.append(cand.enableAsyncCopy ? "1" : "0");
  key.append("|bypass=");
  key.append(cand.asyncBypassL1 ? "1" : "0");
  key.append("|pipe=");
  key.append(cand.enableSoftwarePipelining ? "1" : "0");
  key.append("|pdepth=");
  key.append(std::to_string(cand.pipelineDepth));
  key.append("|peel=");
  key.append(cand.pipelinePeelEpilogue ? "1" : "0");
  key.append("|waitg=");
  key.append(cand.pipelineSetAsyncWaitGroups ? "1" : "0");
  key.append("|wgmb=");
  key.append(std::to_string(cand.workgroupMultiBufferDepth));
  key.append("|wgpad=");
  key.append(std::to_string(cand.workgroupPadLastDim));
  key.append("|wgpadmm=");
  key.append(cand.workgroupPadLastDimMatmulOnly ? "1" : "0");
  key.append("|wgswz=");
  key.append(std::to_string(cand.workgroupSwizzleXor));
  key.append("|rastxor=");
  key.append(std::to_string(cand.blockRasterizeXor));
  key.append("|rastmode=");
  key.append(std::to_string(cand.blockRasterizeMode));
  key.append("|rastpanel=");
  key.append(std::to_string(cand.blockRasterizePanelWidth));
  key.append("|swapxy=");
  key.append(cand.swapBlockDims ? "1" : "0");
  key.append("|rr_reuse=");
  key.append(cand.enableRowReductionChainReuseFusion ? "1" : "0");
  key.append("|rr_promo=");
  key.append(cand.enableRowReductionInputPromotion ? "1" : "0");
  key.append("|rr_promo_vec=");
  key.append(cand.enableRowReductionInputPromotionVectorize ? "1" : "0");
  key.append("|rr_warp=");
  key.append(cand.enableRowReductionWarp ? "1" : "0");
  key.append("|rr_vec=");
  key.append(cand.enableRowReductionVectorize ? "1" : "0");
  key.append("|rr_vec_w=");
  key.append(std::to_string(cand.rowReductionVectorWidth));
  key.append("|rr_tx=");
  key.append(std::to_string(cand.rowReductionThreadsX));
  key.append("|rr_relax=");
  key.append(cand.enableRowReductionRelaxBarriers ? "1" : "0");
  key.append("|rr_skipc=");
  key.append(cand.enableRowReductionSkipCombineBarrier ? "1" : "0");
  key.append("|rr_in_vec=");
  key.append(std::to_string(cand.rowReductionInputVectorWidth));
  key.append("|rr_comb_vec=");
  key.append(cand.enableRowReductionCombineVectorize ? "1" : "0");
  key.append("|mm_sm_reuse=");
  key.append(cand.enableMatmulSoftmaxSharedReuseFusion ? "1" : "0");
  key.append("|tc_tf32=");
  key.append(cand.enableTensorCoreTf32 ? "1" : "0");
  key.append("|tc_f16=");
  key.append(cand.enableTensorCoreF16 ? "1" : "0");
  key.append("|cutlass=");
  key.append(cand.useCutlassMma ? "1" : "0");
  key.append("|mmam=");
  key.append(std::to_string(cand.mmaM));
  key.append("|mman=");
  key.append(std::to_string(cand.mmaN));
  key.append("|mmak=");
  key.append(std::to_string(cand.mmaK));
  key.append("|edges=");
  bool firstEdge = true;
  for (const TileGraphEdge &e : graph.edges) {
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
      continue;
    if (!firstEdge)
      key.push_back(',');
    firstEdge = false;
    key.append(std::to_string(e.src));
    key.push_back('>');
    key.append(std::to_string(e.dst));
    key.push_back('.');
    key.append(std::to_string(e.dstOperand));
    key.push_back(':');
    key.append(std::to_string(e.connectLevel));
  }
  key.append("|nodes=");
  for (size_t i = 0; i < static_cast<size_t>(sg.nodes.size()); ++i) {
    if (i)
      key.push_back(',');
    key.append(std::to_string(sg.nodes[i]));
  }
  // 当 solver 侧启发式发生实质变化时提升 cache key 版本，
  // 避免旧 codegen 假设下的过期测量结果污染选优。
  key.append("|profile_key_rev=20260213_mm_sm_spill_v11_tc_async_rowreuse");
  return key;
}

static void loadDiskProfileCacheIfNeeded(
    const std::string &cachePath,
    std::unordered_map<std::string, double> &cache) {
  if (cachePath.empty())
    return;
  static std::unordered_map<std::string, bool> loaded;
  if (loaded[cachePath])
    return;

  std::ifstream in(cachePath);
  if (!in) {
    loaded[cachePath] = true;
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    size_t tab = line.find('\t');
    if (tab == std::string::npos)
      continue;
    std::string key = line.substr(0, tab);
    double ms = std::strtod(line.substr(tab + 1).c_str(), nullptr);
    if (ms > 0.0)
      cache.emplace(std::move(key), ms);
  }
  loaded[cachePath] = true;
}

static void appendDiskProfileCache(const std::string &cachePath,
                                   const std::string &key, double avgMs) {
  if (cachePath.empty())
    return;
  std::ofstream out(cachePath, std::ios::out | std::ios::app);
  if (!out)
    return;
  out << key << "\t" << avgMs << "\n";
}

static std::string readFileOrEmpty(const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in)
    return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static int64_t getEnvInt64OrDefault(const char *name, int64_t defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  errno = 0;
  char *end = nullptr;
  long long parsed = std::strtoll(raw, &end, 10);
  if (errno != 0 || end == raw)
    return defaultValue;
  return static_cast<int64_t>(parsed);
}

static double getEnvDoubleOrDefault(const char *name, double defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  errno = 0;
  char *end = nullptr;
  double parsed = std::strtod(raw, &end);
  if (errno != 0 || end == raw || !std::isfinite(parsed))
    return defaultValue;
  return parsed;
}

static constexpr int64_t kCompileRetryBudgetUnlimited = -1;
static constexpr int64_t kCompileRetryBudgetUninitialized =
    std::numeric_limits<int64_t>::min();

static std::atomic<int64_t> &getCompileRetryBudgetRemaining() {
  static std::atomic<int64_t> remaining(kCompileRetryBudgetUninitialized);
  return remaining;
}

static int64_t initCompileRetryBudgetIfNeeded() {
  auto &remaining = getCompileRetryBudgetRemaining();
  int64_t cur = remaining.load(std::memory_order_acquire);
  if (cur != kCompileRetryBudgetUninitialized)
    return cur;
  int64_t budget =
      getEnvInt64OrDefault("WELDER_PROFILE_COMPILE_RETRY_BUDGET", 0);
  int64_t normalized =
      budget > 0 ? budget : static_cast<int64_t>(kCompileRetryBudgetUnlimited);
  int64_t expected = kCompileRetryBudgetUninitialized;
  if (!remaining.compare_exchange_strong(expected, normalized,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
    return expected;
  return normalized;
}

static bool tryConsumeCompileRetryBudget(int64_t *remainingAfter = nullptr) {
  auto &remaining = getCompileRetryBudgetRemaining();
  int64_t cur = initCompileRetryBudgetIfNeeded();
  if (cur == kCompileRetryBudgetUnlimited) {
    if (remainingAfter)
      * remainingAfter = kCompileRetryBudgetUnlimited;
    return true;
  }
  while (cur > 0) {
    if (remaining.compare_exchange_weak(cur, cur - 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      if (remainingAfter)
        * remainingAfter = cur - 1;
      return true;
    }
  }
  if (remainingAfter)
    * remainingAfter = std::max<int64_t>(int64_t(0), cur);
  return false;
}

static int64_t countSubstringOccurrences(llvm::StringRef text,
                                         llvm::StringRef needle) {
  if (text.empty() || needle.empty())
    return 0;
  int64_t count = 0;
  size_t pos = 0;
  while (true) {
    pos = text.find(needle, pos);
    if (pos == llvm::StringRef::npos)
      break;
    ++count;
    pos += needle.size();
  }
  return count;
}

static bool parsePositiveInt64Range(llvm::StringRef text, size_t begin, size_t end,
                                    int64_t &out) {
  if (begin >= end || end > text.size())
    return false;
  int64_t value = 0;
  bool anyDigit = false;
  for (size_t i = begin; i < end; ++i) {
    unsigned char ch = static_cast<unsigned char>(text[i]);
    if (!std::isdigit(ch))
      return false;
    anyDigit = true;
    if (value > (std::numeric_limits<int64_t>::max() - 9) / 10)
      return false;
    value = value * 10 + static_cast<int64_t>(ch - '0');
  }
  if (!anyDigit)
    return false;
  out = value;
  return true;
}

static int64_t parseMaxLocalDepotBytesFromNvvm(llvm::StringRef nvvmText) {
  int64_t maxBytes = 0;
  size_t pos = 0;
  while (true) {
    pos = nvvmText.find("__local_depot", pos);
    if (pos == llvm::StringRef::npos)
      break;
    size_t open = nvvmText.find('[', pos);
    if (open == llvm::StringRef::npos) {
      pos += strlen("__local_depot");
      continue;
    }
    size_t close = nvvmText.find(']', open + 1);
    if (close == llvm::StringRef::npos) {
      pos = open + 1;
      continue;
    }
    int64_t bytes = 0;
    if (parsePositiveInt64Range(nvvmText, open + 1, close, bytes))
      maxBytes = std::max(maxBytes, bytes);
    pos = close + 1;
  }
  return maxBytes;
}

static int64_t parseMaxPtxRegisterCount(llvm::StringRef nvvmText,
                                        llvm::StringRef token) {
  int64_t maxRegs = 0;
  size_t pos = 0;
  while (true) {
    pos = nvvmText.find(token, pos);
    if (pos == llvm::StringRef::npos)
      break;
    size_t begin = pos + token.size();
    size_t end = begin;
    while (end < nvvmText.size() &&
           std::isdigit(static_cast<unsigned char>(nvvmText[end])))
      ++end;
    if (end < nvvmText.size() && nvvmText[end] == '>') {
      int64_t regs = 0;
      if (parsePositiveInt64Range(nvvmText, begin, end, regs))
        maxRegs = std::max(maxRegs, regs);
      pos = end + 1;
      continue;
    }
    pos = begin;
  }
  return maxRegs;
}

struct NvvmPtxStats {
  int64_t maxLocalDepotBytes = 0;
  int64_t localLoadOps = 0;
  int64_t localStoreOps = 0;
  int64_t maxRegB32 = 0;
  int64_t maxRegB64 = 0;
  int64_t cpAsyncOps = 0;
  int64_t cpAsyncWaitGroupOps = 0;
  bool hasMmaSync = false;
  bool hasCpAsync = false;
  bool hasCpAsyncWaitGroup = false;

  int64_t localMemOps() const { return localLoadOps + localStoreOps; }
};

static NvvmPtxStats collectNvvmPtxStats(llvm::StringRef nvvmText) {
  NvvmPtxStats out;
  out.maxLocalDepotBytes = parseMaxLocalDepotBytesFromNvvm(nvvmText);
  out.localLoadOps = countSubstringOccurrences(nvvmText, "ld.local");
  out.localStoreOps = countSubstringOccurrences(nvvmText, "st.local");
  out.maxRegB64 = parseMaxPtxRegisterCount(nvvmText, "%rd<");
  out.maxRegB32 = parseMaxPtxRegisterCount(nvvmText, "%r<");
  out.cpAsyncWaitGroupOps =
      countSubstringOccurrences(nvvmText, "cp.async.wait_group");
  out.cpAsyncOps = countSubstringOccurrences(nvvmText, "cp.async");
  out.hasMmaSync = nvvmText.contains("mma.sync");
  out.hasCpAsync = out.cpAsyncOps > 0;
  out.hasCpAsyncWaitGroup = out.cpAsyncWaitGroupOps > 0;
  return out;
}

template <typename Func>
static void runParallelBounded(int maxJobs, size_t n, Func fn) {
  // 确定性调度：按索引顺序迭代；并发仅影响墙钟时间，不影响结果顺序。
  maxJobs = std::max(1, maxJobs);
  if (n == 0) {
    return;
  }
  if (maxJobs == 1 || n == 1) {
    for (size_t i = 0; i < n; ++i)
      fn(i);
    return;
  }

  std::mutex mu;
  size_t next = 0;
  auto worker = [&]() {
    while (true) {
      size_t idx = 0;
      {
        std::lock_guard<std::mutex> lock(mu);
        if (next >= n)
          return;
        idx = next++;
      }
      fn(idx);
    }
  };

  int threads = std::min<int>(maxJobs, static_cast<int>(n));
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t)
    pool.emplace_back(worker);
  for (auto &th : pool)
    th.join();
}

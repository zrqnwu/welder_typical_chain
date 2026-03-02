static bool isHex(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

static int hexVal(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F')
    return 10 + (c - 'A');
  return 0;
}

static std::string trim(std::string s) {
  auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
  return s;
}

static std::string readFileOrDie(const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    llvm::errs() << "error: cannot open file: " << path << "\n";
    std::exit(2);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string decodeMlirStringEscapes(std::string_view in) {
  // MLIR 字符串属性会把字节写成 `\XX`（XX 为十六进制）。
  // 例如：`\0A` 表示换行，`\09` 表示制表符。
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size();) {
    char c = in[i];
    if (c != '\\') {
      out.push_back(c);
      ++i;
      continue;
    }

    if (i + 1 < in.size() && in[i + 1] == '\\') {
      out.push_back('\\');
      i += 2;
      continue;
    }

    if (i + 2 < in.size() && isHex(in[i + 1]) && isHex(in[i + 2])) {
      int v = (hexVal(in[i + 1]) << 4) | hexVal(in[i + 2]);
      out.push_back(static_cast<char>(v));
      i += 3;
      continue;
    }

    // 未识别转义：若后面有字符则原样保留。
    if (i + 1 < in.size()) {
      out.push_back(in[i + 1]);
      i += 2;
      continue;
    }
    ++i;
  }
  return out;
}

static std::vector<std::string> splitTopLevelCommaList(std::string_view in) {
  std::vector<std::string> out;
  std::string cur;
  int parenDepth = 0;
  for (char c : in) {
    if (c == '(')
      ++parenDepth;
    else if (c == ')')
      --parenDepth;

    if (c == ',' && parenDepth == 0) {
      out.push_back(trim(cur));
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  if (!cur.empty())
    out.push_back(trim(cur));
  return out;
}

static std::optional<std::pair<std::string, std::string>>
parseKeyValueEq(std::string_view in) {
  size_t eq = in.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 >= in.size())
    return std::nullopt;
  std::string key = trim(std::string(in.substr(0, eq)));
  std::string val = trim(std::string(in.substr(eq + 1)));
  if (!key.empty() && key.front() != '%')
    key.insert(key.begin(), '%');
  return std::make_optional(std::make_pair(std::move(key), std::move(val)));
}

static std::optional<float> parseF32Value(std::string_view in) {
  std::string tok = trim(std::string(in));
  if (tok.empty())
    return std::nullopt;
  if (tok == "inf" || tok == "+inf" || tok == "infinity" ||
      tok == "+infinity") {
    return std::numeric_limits<float>::infinity();
  }
  if (tok == "-inf" || tok == "-infinity") {
    return -std::numeric_limits<float>::infinity();
  }
  if (tok.size() >= 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X')) {
    // 允许传入 MLIR 常量风格的原始位模式，如 `0xFF800000`（即 -inf）。
    uint32_t bits = static_cast<uint32_t>(std::stoul(tok, nullptr, 0));
    float v = 0.0f;
    static_assert(sizeof(bits) == sizeof(v), "f32 must be 32-bit");
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  }
  errno = 0;
  char *endp = nullptr;
  float v = std::strtof(tok.c_str(), &endp);
  if (errno != 0 || endp == tok.c_str() || (endp && *endp != '\0'))
    return std::nullopt;
  return v;
}

static std::optional<LaunchSpec>
parseFirstLaunchSpec(const std::string &mlirText,
                     const std::optional<std::string> &kernelFilter) {
  std::istringstream iss(mlirText);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("gpu.launch_func") == std::string::npos)
      continue;

    // 示例：
    // 例如：`gpu.launch_func @main_kernel::@main_kernel blocks in (%6, %6, %6)
    // 即 threads in (%8, %8, %6) : i64 args(...)`
    auto atPos = line.find('@');
    auto sepPos = line.find("::@", atPos == std::string::npos ? 0 : atPos);
    if (atPos == std::string::npos || sepPos == std::string::npos)
      continue;

    std::string binaryName = trim(line.substr(atPos + 1, sepPos - (atPos + 1)));
    auto entryEnd = line.find(' ', sepPos + 3);
    std::string entryName =
        trim(line.substr(sepPos + 3, entryEnd - (sepPos + 3)));

    if (kernelFilter && binaryName != *kernelFilter &&
        entryName != *kernelFilter) {
      continue;
    }

    auto blocksPos = line.find("blocks in (");
    auto threadsPos = line.find("threads in (");
    if (blocksPos == std::string::npos || threadsPos == std::string::npos)
      continue;

    auto blocksStart = blocksPos + std::string("blocks in (").size();
    auto blocksEnd = line.find(')', blocksStart);
    auto threadsStart = threadsPos + std::string("threads in (").size();
    auto threadsEnd = line.find(')', threadsStart);
    if (blocksEnd == std::string::npos || threadsEnd == std::string::npos)
      continue;

    auto blocksList = splitTopLevelCommaList(
        std::string_view(line).substr(blocksStart, blocksEnd - blocksStart));
    auto threadsList = splitTopLevelCommaList(
        std::string_view(line).substr(threadsStart, threadsEnd - threadsStart));
    if (blocksList.size() != 3 || threadsList.size() != 3)
      continue;

    auto argsPos = line.find("args(", threadsEnd);
    if (argsPos == std::string::npos)
      continue;
    auto argsStart = argsPos + std::string("args(").size();
    auto argsEnd = line.rfind(')');
    if (argsEnd == std::string::npos || argsEnd <= argsStart)
      continue;

    std::vector<std::string> argTokens = splitTopLevelCommaList(
        std::string_view(line).substr(argsStart, argsEnd - argsStart));

    LaunchSpec spec;
    spec.binaryName = std::move(binaryName);
    spec.entryName = std::move(entryName);
    for (int i = 0; i < 3; ++i) {
      spec.gridTokens[i] = std::move(blocksList[i]);
      spec.blockTokens[i] = std::move(threadsList[i]);
    }
    for (const std::string &tok : argTokens) {
      auto colon = tok.find(':');
      if (colon == std::string::npos)
        continue;
      LaunchArg a;
      a.value = trim(tok.substr(0, colon));
      a.type = trim(tok.substr(colon + 1));
      spec.args.push_back(std::move(a));
    }

    return spec;
  }
  return std::nullopt;
}

static std::vector<LaunchSpec> parseAllLaunchSpecs(const std::string &mlirText) {
  std::vector<LaunchSpec> out;
  std::istringstream iss(mlirText);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("gpu.launch_func") == std::string::npos)
      continue;

    // 示例：
    // 例如：`gpu.launch_func @main_kernel::@main_kernel blocks in (%6, %6, %6)
    // 即 threads in (%8, %8, %6) : i64 args(...)`
    auto atPos = line.find('@');
    auto sepPos = line.find("::@", atPos == std::string::npos ? 0 : atPos);
    if (atPos == std::string::npos || sepPos == std::string::npos)
      continue;

    std::string binaryName = trim(line.substr(atPos + 1, sepPos - (atPos + 1)));
    auto entryEnd = line.find(' ', sepPos + 3);
    std::string entryName =
        trim(line.substr(sepPos + 3, entryEnd - (sepPos + 3)));

    auto blocksPos = line.find("blocks in (");
    auto threadsPos = line.find("threads in (");
    if (blocksPos == std::string::npos || threadsPos == std::string::npos)
      continue;

    auto blocksStart = blocksPos + std::string("blocks in (").size();
    auto blocksEnd = line.find(')', blocksStart);
    auto threadsStart = threadsPos + std::string("threads in (").size();
    auto threadsEnd = line.find(')', threadsStart);
    if (blocksEnd == std::string::npos || threadsEnd == std::string::npos)
      continue;

    auto blocksList = splitTopLevelCommaList(
        std::string_view(line).substr(blocksStart, blocksEnd - blocksStart));
    auto threadsList = splitTopLevelCommaList(
        std::string_view(line).substr(threadsStart, threadsEnd - threadsStart));
    if (blocksList.size() != 3 || threadsList.size() != 3)
      continue;

    auto argsPos = line.find("args(", threadsEnd);
    if (argsPos == std::string::npos)
      continue;
    auto argsStart = argsPos + std::string("args(").size();
    auto argsEnd = line.rfind(')');
    if (argsEnd == std::string::npos || argsEnd <= argsStart)
      continue;

    std::vector<std::string> argTokens = splitTopLevelCommaList(
        std::string_view(line).substr(argsStart, argsEnd - argsStart));

    LaunchSpec spec;
    spec.binaryName = std::move(binaryName);
    spec.entryName = std::move(entryName);
    for (int i = 0; i < 3; ++i) {
      spec.gridTokens[i] = std::move(blocksList[i]);
      spec.blockTokens[i] = std::move(threadsList[i]);
    }
    for (const std::string &tok : argTokens) {
      auto colon = tok.find(':');
      if (colon == std::string::npos)
        continue;
      LaunchArg a;
      a.value = trim(tok.substr(0, colon));
      a.type = trim(tok.substr(colon + 1));
      spec.args.push_back(std::move(a));
    }

    out.push_back(std::move(spec));
  }
  return out;
}

struct ConstTables {
  std::unordered_map<std::string, int64_t> i64;
  std::unordered_map<std::string, float> f32;
  std::unordered_map<std::string, float> f16;
};

static bool startsWith(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() &&
         s.substr(0, prefix.size()) == prefix;
}

static bool endsWith(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() &&
         s.substr(s.size() - suffix.size()) == suffix;
}

static std::optional<std::pair<int64_t, int64_t>>
parseVectorType(std::string_view type) {
  // 示例：`vector<4xf32>`、`vector<2x4xf16>`、`vector<8xi32>` 等。
  if (!startsWith(type, "vector<") || !endsWith(type, ">"))
    return std::nullopt;
  std::string_view inside = type.substr(std::string_view("vector<").size());
  inside = inside.substr(0, inside.size() - 1);
  if (inside.empty())
    return std::nullopt;

  // 按 `x` 切分：最后一个 token 是元素类型，其余是各维大小。
  std::vector<std::string_view> parts;
  size_t pos = 0;
  while (pos < inside.size()) {
    size_t next = inside.find('x', pos);
    if (next == std::string_view::npos) {
      parts.push_back(inside.substr(pos));
      break;
    }
    parts.push_back(inside.substr(pos, next - pos));
    pos = next + 1;
  }
  if (parts.size() < 2)
    return std::nullopt;
  std::string_view elemType = parts.back();
  parts.pop_back();

  int64_t numElems = 1;
  for (std::string_view d : parts) {
    if (d.empty())
      return std::nullopt;
    int64_t v = 0;
    auto fcRes = std::from_chars(d.data(), d.data() + d.size(), v);
    if (fcRes.ec != std::errc() || fcRes.ptr != d.data() + d.size())
      return std::nullopt;
    if (v <= 0)
      return std::nullopt;
    if (numElems > (INT64_MAX / v))
      return std::nullopt;
    numElems *= v;
  }

  int64_t elemBytes = 0;
  if (elemType == "f32" || elemType == "i32")
    elemBytes = 4;
  else if (elemType == "f16" || elemType == "bf16" || elemType == "i16")
    elemBytes = 2;
  else if (elemType == "i64")
    elemBytes = 8;
  else if (elemType == "i8" || elemType == "i1")
    elemBytes = 1;
  else
    return std::nullopt;

  return std::make_pair(numElems, elemBytes);
}

static ConstTables parseConstants(const std::string &mlirText) {
  ConstTables tbl;
  std::istringstream iss(mlirText);
  std::string line;
  while (std::getline(iss, line)) {
    auto pos = line.find("= llvm.mlir.constant(");
    if (pos == std::string::npos)
      continue;
    auto ssaStart = line.find('%');
    if (ssaStart == std::string::npos || ssaStart > pos)
      continue;
    auto ssaEnd = line.find(' ', ssaStart);
    if (ssaEnd == std::string::npos || ssaEnd > pos)
      continue;
    std::string ssa = line.substr(ssaStart, ssaEnd - ssaStart);

    auto lpar = line.find('(', pos);
    auto rpar = line.find(')', lpar == std::string::npos ? 0 : lpar);
    if (lpar == std::string::npos || rpar == std::string::npos)
      continue;
    std::string inside = trim(line.substr(lpar + 1, rpar - (lpar + 1)));

    auto typePos = line.find(':', rpar);
    if (typePos == std::string::npos)
      continue;
    std::string type = trim(line.substr(typePos + 1));

    if (type == "i64") {
      // inside 形如：`128 : index` 或 `0 : index`。
      size_t endNum = 0;
      int64_t v = std::stoll(inside, &endNum);
      (void)endNum;
      tbl.i64[ssa] = v;
      continue;
    }
    if (type == "i1") {
      // inside 形如：`true` / `false`。
      std::string tok = inside;
      size_t cut = tok.find_first_of(" \t:");
      if (cut != std::string::npos)
        tok = trim(tok.substr(0, cut));
      if (tok == "true" || tok == "1") {
        tbl.i64[ssa] = 1;
        continue;
      }
      if (tok == "false" || tok == "0") {
        tbl.i64[ssa] = 0;
        continue;
      }
      continue;
    }
    if (type == "f32") {
      // inside 示例：
      // - 数值形式：`1.000000e+00 : f32`
      // - `0xFF800000 : f32`（位模式，例如 -inf）
      std::string tok = inside;
      size_t cut = tok.find_first_of(" \t:");
      if (cut != std::string::npos)
        tok = trim(tok.substr(0, cut));

      float v = 0.0f;
      if (startsWith(tok, "0x") || startsWith(tok, "0X")) {
        uint32_t bits = static_cast<uint32_t>(std::stoul(tok, nullptr, 0));
        static_assert(sizeof(bits) == sizeof(v), "f32 must be 32-bit");
        std::memcpy(&v, &bits, sizeof(v));
      } else {
        size_t endNum = 0;
        v = std::stof(inside, &endNum);
        (void)endNum;
      }
      tbl.f32[ssa] = v;
      continue;
    }
    if (type == "f16") {
      // inside 示例：
      // - 数值形式：`0.000000e+00 : f16`
      // - `0xH3C00 : f16`（位模式）
      std::string tok = inside;
      size_t cut = tok.find_first_of(" \t:");
      if (cut != std::string::npos)
        tok = trim(tok.substr(0, cut));

      float v = 0.0f;
      if (startsWith(tok, "0xH") || startsWith(tok, "0XH")) {
        // MLIR 会用 `0xH` 前缀打印 half 的十六进制常量。
        uint16_t bits =
            static_cast<uint16_t>(std::stoul(tok.substr(3), nullptr, 16));
        // 将 half 位模式转换为 float（尽力实现）。
        // 注意：逻辑保持本地实现，避免引入 CUDA half 头文件依赖。
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t exp = (bits >> 10) & 0x1f;
        uint32_t mant = bits & 0x3ff;
        uint32_t fbits = 0;
        if (exp == 0) {
          if (mant == 0) {
            fbits = sign << 31;
          } else {
            // 非正规 half -> 正常 float。
            int e = -14;
            while ((mant & 0x400) == 0) {
              mant <<= 1;
              --e;
            }
            mant &= 0x3ff;
            uint32_t fexp = static_cast<uint32_t>(e + 127);
            uint32_t fmant = mant << 13;
            fbits = (sign << 31) | (fexp << 23) | fmant;
          }
        } else if (exp == 0x1f) {
          // 无穷大或 NaN。
          fbits = (sign << 31) | (0xff << 23) | (mant ? 0x7fffff : 0);
        } else {
          uint32_t fexp = (exp - 15 + 127);
          uint32_t fmant = mant << 13;
          fbits = (sign << 31) | (fexp << 23) | fmant;
        }
        static_assert(sizeof(fbits) == sizeof(v), "f32 must be 32-bit");
        std::memcpy(&v, &fbits, sizeof(v));
      } else if (startsWith(tok, "0x") || startsWith(tok, "0X")) {
        // 某些打印器可能省略 `H` 前缀。
        uint16_t bits = static_cast<uint16_t>(std::stoul(tok, nullptr, 0));
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t exp = (bits >> 10) & 0x1f;
        uint32_t mant = bits & 0x3ff;
        uint32_t fbits = 0;
        if (exp == 0) {
          if (mant == 0) {
            fbits = sign << 31;
          } else {
            int e = -14;
            while ((mant & 0x400) == 0) {
              mant <<= 1;
              --e;
            }
            mant &= 0x3ff;
            uint32_t fexp = static_cast<uint32_t>(e + 127);
            uint32_t fmant = mant << 13;
            fbits = (sign << 31) | (fexp << 23) | fmant;
          }
        } else if (exp == 0x1f) {
          fbits = (sign << 31) | (0xff << 23) | (mant ? 0x7fffff : 0);
        } else {
          uint32_t fexp = (exp - 15 + 127);
          uint32_t fmant = mant << 13;
          fbits = (sign << 31) | (fexp << 23) | fmant;
        }
        static_assert(sizeof(fbits) == sizeof(v), "f32 must be 32-bit");
        std::memcpy(&v, &fbits, sizeof(v));
      } else {
        size_t endNum = 0;
        v = std::stof(inside, &endNum);
        (void)endNum;
      }
      tbl.f16[ssa] = v;
      continue;
    }
  }
  return tbl;
}

static std::optional<std::string>
extractGpuBinaryAssemblyEscaped(const std::string &mlirText,
                                const std::string &binaryName) {
  // 目标匹配：`gpu.binary @<binaryName> ... assembly = "<escaped>"`。
  std::string needle = "gpu.binary @" + binaryName;
  size_t start = mlirText.find(needle);
  if (start == std::string::npos)
    return std::nullopt;
  size_t asmKey = mlirText.find("assembly = \"", start);
  if (asmKey == std::string::npos)
    return std::nullopt;
  size_t asmStart = asmKey + std::string("assembly = \"").size();

  // 扫描到未转义的结束引号。
  std::string escaped;
  escaped.reserve(1024);
  for (size_t i = asmStart; i < mlirText.size(); ++i) {
    char c = mlirText[i];
    if (c == '"')
      return escaped;
    escaped.push_back(c);
  }
  return std::nullopt;
}

static void printCudaErrorAndExit(CUresult r, const char *what) {
  const char *name = nullptr;
  const char *msg = nullptr;
  (void)cuGetErrorName(r, &name);
  (void)cuGetErrorString(r, &msg);
  llvm::errs() << "CUDA error in " << what << ": "
               << (name ? name : "<unknown>") << " ("
               << (msg ? msg : "<no message>") << ")\n";
  std::exit(1);
}

static void printCudaJitLog(const char *label, const char *log, size_t cap) {
  if (!log || cap == 0)
    return;
  // `CU_JIT_*_LOG_BUFFER` 是 char 数组；这里尽力打印其
  // 以 NUL 结尾的前缀内容。
  size_t n = 0;
  while (n < cap && log[n] != '\0')
    ++n;
  if (n == 0)
    return;
  llvm::errs() << (label ? label : "CUDA JIT log") << ":\n";
  llvm::errs().write(log, static_cast<std::streamsize>(n));
  llvm::errs() << "\n";
}

static std::string parsePtxTargetArchOrDefault(const std::string &ptx,
                                              const std::string &fallback) {
  // PTX 语法示例：`.target sm_86`。
  size_t pos = ptx.find(".target");
  if (pos == std::string::npos)
    return fallback;
  size_t lineEnd = ptx.find('\n', pos);
  std::string_view line =
      (lineEnd == std::string::npos)
          ? std::string_view(ptx).substr(pos)
          : std::string_view(ptx).substr(pos, lineEnd - pos);
  // 找到 `.target` 后面的架构 token。
  size_t tok = line.find("sm_");
  if (tok == std::string::npos)
    return fallback;
  size_t tokEnd = tok;
  while (tokEnd < line.size()) {
    char c = line[tokEnd];
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
      break;
    ++tokEnd;
  }
  std::string arch(line.substr(tok, tokEnd - tok));
  if (arch.empty())
    return fallback;
  return arch;
}

static std::optional<std::vector<uint8_t>>
compilePtxToCubinWithPtxas(const std::string &ptx) {
  // CUDA JIT/ptxas 拒绝 kernel（如 shared 内存 > 48KB）时的尽力回退：
  // 调用工具链 `ptxas` 生成 cubin 并改为加载 cubin。
  //
  // 注意：要求 `ptxas` 在 PATH 中可用。
  const std::string arch = parsePtxTargetArchOrDefault(ptx, /*fallback=*/"sm_86");

  // 生成唯一的临时文件基础路径。
  std::string tmpl = "/tmp/welder_profiler_ptxas_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  int fd = mkstemp(buf.data());
  if (fd < 0)
    return std::nullopt;
  std::string ptxPath(buf.data());
  std::string cubinPath = ptxPath + ".cubin";

  // 将 PTX 写入临时文件。
  size_t off = 0;
  while (off < ptx.size()) {
    ssize_t n = ::write(fd, ptx.data() + off, ptx.size() - off);
    if (n <= 0)
      break;
    off += static_cast<size_t>(n);
  }
  ::close(fd);
  if (off != ptx.size()) {
    (void)std::remove(ptxPath.c_str());
    return std::nullopt;
  }

  // 编译为 cubin。
  // 默认静默输出；失败时由调用方打印原始 JIT 日志。
  std::string cmd = "ptxas -arch=" + arch + " \"" + ptxPath + "\" -o \"" +
                    cubinPath + "\" >/dev/null 2>&1";
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    (void)std::remove(ptxPath.c_str());
    (void)std::remove(cubinPath.c_str());
    return std::nullopt;
  }

  // 读取 cubin 二进制内容。
  std::ifstream fin(cubinPath, std::ios::binary);
  if (!fin.good()) {
    (void)std::remove(ptxPath.c_str());
    (void)std::remove(cubinPath.c_str());
    return std::nullopt;
  }
  fin.seekg(0, std::ios::end);
  std::streamoff sz = fin.tellg();
  fin.seekg(0, std::ios::beg);
  if (sz <= 0 || sz > (1LL << 30)) {
    (void)std::remove(ptxPath.c_str());
    (void)std::remove(cubinPath.c_str());
    return std::nullopt;
  }
  std::vector<uint8_t> out;
  out.resize(static_cast<size_t>(sz));
  fin.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(out.size()));

  (void)std::remove(ptxPath.c_str());
  (void)std::remove(cubinPath.c_str());
  return out;
}

#define CU_CHECK(expr)                                                       \
  do {                                                                       \
    CUresult _r = (expr);                                                    \
    if (_r != CUDA_SUCCESS)                                                  \
      printCudaErrorAndExit(_r, #expr);                                      \
  } while (0)

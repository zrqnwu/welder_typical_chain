#include "wtc/ir/ChainCanonicalize.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

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

std::string readAll(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs)
    return "";
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

bool writeAll(const std::string &path, const std::string &content) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream ofs(path);
  if (!ofs)
    return false;
  ofs << content;
  return true;
}

std::string findMlirOpt() {
  if (const char *p = std::getenv("WTC_MLIR_OPT"); p && *p) {
    if (std::filesystem::exists(p))
      return p;
  }

  const std::filesystem::path pinned =
      "/home/zhangruiqi/llvm-project/build/bin/mlir-opt";
  if (std::filesystem::exists(pinned))
    return pinned.string();

  return "mlir-opt";
}

std::string buildSoftmaxDecomposeTransformLibrary() {
  return R"(module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %softmax = transform.structured.match ops{["linalg.softmax"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %decomp = transform.structured.decompose_interface %softmax : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)";
}

} // namespace

namespace wtc::ir {

bool canonicalizeMatmulSoftmaxChain(const std::string &inputPath,
                                    const std::string &outputPath,
                                    std::string &diagnostic) {
  if (inputPath.empty()) {
    diagnostic = "input path is empty";
    return false;
  }
  if (outputPath.empty()) {
    diagnostic = "output path is empty";
    return false;
  }
  if (!std::filesystem::exists(inputPath)) {
    diagnostic = "input file not found: " + inputPath;
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());
  const std::filesystem::path logPath =
      std::filesystem::path(outputPath).parent_path() / "01.canonicalize.log";

  const std::string mlirOpt = findMlirOpt();
  std::ostringstream cmd;
  const std::string inputText = readAll(inputPath);
  if (inputText.empty()) {
    diagnostic = "failed to read input MLIR: " + inputPath;
    return false;
  }

  const bool hasNamedSoftmax = contains(inputText, "linalg.softmax");
  if (hasNamedSoftmax) {
    const std::filesystem::path transformLibPath =
        std::filesystem::path(outputPath).parent_path() /
        "01.softmax_decompose.transform.mlir";
    if (!writeAll(transformLibPath.string(),
                  buildSoftmaxDecomposeTransformLibrary())) {
      diagnostic = "failed to write softmax decomposition transform library: " +
                   transformLibPath.string();
      return false;
    }

    // 先对 linalg.softmax 做结构化分解，再 canonicalize。
    cmd << shellQuote(mlirOpt) << " " << shellQuote(inputPath) << " "
        << shellQuote("--transform-preload-library=transform-library-paths=" +
                      transformLibPath.string())
        << " " << shellQuote("--transform-interpreter=entry-point=__transform_main")
        << " --canonicalize"
        << " -o " << shellQuote(outputPath)
        << " > " << shellQuote(logPath.string()) << " 2>&1";
  } else {
    cmd << shellQuote(mlirOpt) << " " << shellQuote(inputPath)
        << " --canonicalize"
        << " -o " << shellQuote(outputPath)
        << " > " << shellQuote(logPath.string()) << " 2>&1";
  }

  int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    diagnostic = "mlir-opt canonicalize failed, rc=" + std::to_string(rc) +
                 ", see " + logPath.string();
    return false;
  }

  const std::string text = readAll(outputPath);
  if (text.empty()) {
    diagnostic = "canonicalized output is empty: " + outputPath;
    return false;
  }

  if (!contains(text, "linalg.matmul")) {
    diagnostic = "canonicalized IR lost linalg.matmul unexpectedly";
    return false;
  }

  if (hasNamedSoftmax && contains(text, "linalg.softmax")) {
    diagnostic = "softmax decomposition did not run: linalg.softmax remains";
    return false;
  }

  if (!contains(text, "math.exp") || !contains(text, "arith.divf")) {
    diagnostic =
        "canonicalized IR does not contain expected softmax decomposition ops";
    return false;
  }

  diagnostic.clear();
  return true;
}

} // namespace wtc::ir

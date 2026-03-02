#include "wtc/ir/Tagging.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string readAll(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs)
    return "";
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

bool writeAll(const std::string &path, const std::string &content) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream ofs(path);
  if (!ofs)
    return false;
  ofs << content;
  return true;
}

int countContains(const std::string &text, const std::string &needle) {
  int count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::string ltrim(const std::string &s) {
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
    ++i;
  return s.substr(i);
}

} // namespace

namespace wtc::ir {

bool tagTypicalChainOps(const std::string &inputPath,
                        const std::string &outputPath,
                        const std::string &tagsJsonPath,
                        std::string &diagnostic) {
  if (inputPath.empty()) {
    diagnostic = "input path is empty";
    return false;
  }
  if (outputPath.empty()) {
    diagnostic = "output path is empty";
    return false;
  }
  if (tagsJsonPath.empty()) {
    diagnostic = "tags json path is empty";
    return false;
  }

  const std::string text = readAll(inputPath);
  if (text.empty()) {
    diagnostic = "failed to read input MLIR for tagging: " + inputPath;
    return false;
  }

  const int matmulCount = countContains(text, "linalg.matmul");
  const int genericCount = countContains(text, "linalg.generic");
  const int reductionHintCount = countContains(text, "\"reduction\"");
  const int expCount = countContains(text, "math.exp");
  const int divCount = countContains(text, "arith.divf");

  if (matmulCount == 0 || genericCount == 0) {
    diagnostic =
        "tagging expects matmul + generic softmax decomposition in input";
    return false;
  }

  std::vector<std::string> lines;
  {
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
      lines.push_back(line);
  }

  std::ostringstream tagged;
  tagged << "// [wtc tagging] tags are emitted as comments for readability\n";

  bool matmulTagged = false;
  bool firstReductionTagged = false;
  bool firstExpTagged = false;
  bool firstDivTagged = false;

  for (const std::string &line : lines) {
    const std::string trimmed = ltrim(line);

    if (!matmulTagged && trimmed.find("linalg.matmul") != std::string::npos) {
      tagged << "// wtc.tag: welder.matmul_anchor\n";
      matmulTagged = true;
    }

    if (!firstReductionTagged &&
        trimmed.find("\"reduction\"") != std::string::npos) {
      tagged << "// wtc.tag: welder.row_reduction\n";
      firstReductionTagged = true;
    }

    if (!firstExpTagged && trimmed.find("math.exp") != std::string::npos) {
      tagged << "// wtc.tag: welder.elementwise_exp\n";
      firstExpTagged = true;
    }

    if (!firstDivTagged && trimmed.find("arith.divf") != std::string::npos) {
      tagged << "// wtc.tag: welder.elementwise_div\n";
      firstDivTagged = true;
    }

    tagged << line << '\n';
  }

  if (!writeAll(outputPath, tagged.str())) {
    diagnostic = "failed to write tagged MLIR: " + outputPath;
    return false;
  }

  std::ostringstream tagsJson;
  tagsJson << "{\n"
           << "  \"pipeline\": \"typical_chain\",\n"
           << "  \"source\": \"" << inputPath << "\",\n"
           << "  \"counts\": {\n"
           << "    \"matmul\": " << matmulCount << ",\n"
           << "    \"generic\": " << genericCount << ",\n"
           << "    \"reduction_hints\": " << reductionHintCount << ",\n"
           << "    \"exp\": " << expCount << ",\n"
           << "    \"div\": " << divCount << "\n"
           << "  },\n"
           << "  \"tags\": [\n"
           << "    \"welder.matmul_anchor\",\n"
           << "    \"welder.row_reduction\",\n"
           << "    \"welder.elementwise_exp\",\n"
           << "    \"welder.elementwise_div\"\n"
           << "  ]\n"
           << "}\n";

  if (!writeAll(tagsJsonPath, tagsJson.str())) {
    diagnostic = "failed to write tags json: " + tagsJsonPath;
    return false;
  }

  diagnostic.clear();
  return true;
}

} // namespace wtc::ir

#include "wtc/runtime/PostbufferizeFixups.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string readAll(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs)
    return "";
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
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

} // namespace

namespace wtc::runtime {

bool validatePostbufferizeArtifacts(const std::string &artifactDir,
                                    std::string &diagnostic) {
  if (artifactDir.empty()) {
    diagnostic = "artifact dir is empty";
    return false;
  }

  const std::filesystem::path outDir = artifactDir;
  if (!std::filesystem::exists(outDir)) {
    diagnostic = "artifact dir not found: " + artifactDir;
    return false;
  }

  const std::filesystem::path postPath = outDir / "03.after_postbufferize.mlir";
  const std::filesystem::path launchPath = outDir / "04.after_workgroup_launch.mlir";
  const std::filesystem::path runnablePath = outDir / "05.out.nvvm.runnable.mlir";

  if (!std::filesystem::exists(postPath) || !std::filesystem::exists(launchPath) ||
      !std::filesystem::exists(runnablePath)) {
    diagnostic = "expected compiler artifacts are missing under " + artifactDir;
    return false;
  }

  const std::string post = readAll(postPath.string());
  const std::string launch = readAll(launchPath.string());
  const std::string runnable = readAll(runnablePath.string());

  if (post.empty() || launch.empty() || runnable.empty()) {
    diagnostic = "failed to read compiler artifacts under " + artifactDir;
    return false;
  }

  // 后缓冲化阶段的基础统计，用于稳定诊断与面试说明。
  const int postMatmulCount = countContains(post, "linalg.matmul");
  const int launchKernelCount = countContains(launch, "gpu.launch");
  const int launchAllocaCount = countContains(launch, "memref.alloca");
  const int nvvmMmaCount = countContains(runnable, "nvvm.mma.sync");
  const int localLoadCount = countContains(runnable, "ld.local");
  const int localStoreCount = countContains(runnable, "st.local");

  const std::filesystem::path reportPath = outDir / "postbufferize_report.txt";
  std::ofstream report(reportPath);
  if (!report) {
    diagnostic = "failed to write postbufferize report: " + reportPath.string();
    return false;
  }

  report << "post_matmul_count=" << postMatmulCount << "\n";
  report << "launch_kernel_count=" << launchKernelCount << "\n";
  report << "launch_alloca_count=" << launchAllocaCount << "\n";
  report << "nvvm_mma_sync_count=" << nvvmMmaCount << "\n";
  report << "ptx_ld_local_count=" << localLoadCount << "\n";
  report << "ptx_st_local_count=" << localStoreCount << "\n";

  diagnostic.clear();
  return true;
}

} // namespace wtc::runtime

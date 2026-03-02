#include "WelderCompilerPassTraceAndEnv.h"

#include "llvm/Support/JSON.h"
#include "mlir/Pass/Pass.h"

#include <cerrno>
#include <cstdlib>

namespace welder::compiler {

int64_t getEnvInt64OrDefault(const char *name, int64_t defaultValue) {
  if (!name || !*name)
    return defaultValue;
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  char *end = nullptr;
  errno = 0;
  long long parsed = std::strtoll(raw, &end, 10);
  if (errno != 0 || end == raw)
    return defaultValue;
  return static_cast<int64_t>(parsed);
}

MlirPassTraceInstrumentation::MlirPassTraceInstrumentation(welder::Tracer *tracer)
    : tracer_(tracer) {}

void MlirPassTraceInstrumentation::runBeforePipeline(
    std::optional<mlir::OperationName> name,
    const mlir::PassInstrumentation::PipelineParentInfo &) {
  if (!tracer_ || !tracer_->enabled())
    return;
  tracer_->event("mlir.pipeline.start",
                 llvm::json::Object{
                     {"op", name ? name->getStringRef().str() : "any"},
                 },
                 /*isVerbose=*/true);
}

void MlirPassTraceInstrumentation::runAfterPipeline(
    std::optional<mlir::OperationName> name,
    const mlir::PassInstrumentation::PipelineParentInfo &) {
  if (!tracer_ || !tracer_->enabled())
    return;
  tracer_->event("mlir.pipeline.end",
                 llvm::json::Object{
                     {"op", name ? name->getStringRef().str() : "any"},
                 },
                 /*isVerbose=*/true);
}

void MlirPassTraceInstrumentation::runBeforePass(mlir::Pass *pass,
                                                 mlir::Operation *op) {
  if (!tracer_ || !tracer_->enabled())
    return;

  {
    std::lock_guard<std::mutex> lock(mu_);
    passStarts_[pass] = std::chrono::steady_clock::now();
  }

  llvm::json::Object f;
  f["pass"] = std::string(pass->getName());
  if (op)
    f["op"] = op->getName().getStringRef().str();
  tracer_->event("mlir.pass.start", std::move(f));
}

void MlirPassTraceInstrumentation::runAfterPass(mlir::Pass *pass,
                                                mlir::Operation *op) {
  if (!tracer_ || !tracer_->enabled())
    return;

  auto endT = std::chrono::steady_clock::now();
  std::optional<double> durMs;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = passStarts_.find(pass);
    if (it != passStarts_.end()) {
      durMs =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
              endT - it->second)
              .count();
      passStarts_.erase(it);
    }
  }

  llvm::json::Object f;
  f["pass"] = std::string(pass->getName());
  if (op)
    f["op"] = op->getName().getStringRef().str();
  if (durMs)
    f["dur_ms"] = *durMs;
  tracer_->event("mlir.pass.end", std::move(f));
}

void MlirPassTraceInstrumentation::runAfterPassFailed(mlir::Pass *pass,
                                                      mlir::Operation *op) {
  if (!tracer_ || !tracer_->enabled())
    return;

  auto endT = std::chrono::steady_clock::now();
  std::optional<double> durMs;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = passStarts_.find(pass);
    if (it != passStarts_.end()) {
      durMs =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
              endT - it->second)
              .count();
      passStarts_.erase(it);
    }
  }

  llvm::json::Object f;
  f["pass"] = std::string(pass->getName());
  if (op)
    f["op"] = op->getName().getStringRef().str();
  if (durMs)
    f["dur_ms"] = *durMs;
  tracer_->event("mlir.pass.failed", std::move(f));
}

} // namespace welder::compiler

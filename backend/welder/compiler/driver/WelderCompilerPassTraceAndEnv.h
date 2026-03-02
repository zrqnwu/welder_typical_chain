#ifndef WELDER_COMPILER_PASS_TRACE_AND_ENV_H
#define WELDER_COMPILER_PASS_TRACE_AND_ENV_H

#include "WelderTrace.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassInstrumentation.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace welder::compiler {

int64_t getEnvInt64OrDefault(const char *name, int64_t defaultValue);

class MlirPassTraceInstrumentation : public mlir::PassInstrumentation {
public:
  explicit MlirPassTraceInstrumentation(welder::Tracer *tracer);

  void runBeforePipeline(
      std::optional<mlir::OperationName> name,
      const mlir::PassInstrumentation::PipelineParentInfo &) override;
  void runAfterPipeline(
      std::optional<mlir::OperationName> name,
      const mlir::PassInstrumentation::PipelineParentInfo &) override;
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override;

private:
  welder::Tracer *tracer_ = nullptr;
  std::mutex mu_;
  std::unordered_map<const mlir::Pass *, std::chrono::steady_clock::time_point>
      passStarts_;
};

} // namespace welder::compiler

#endif

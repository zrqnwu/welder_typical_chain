#ifndef WELDER_SOLVER_CANDIDATE_GENERATOR_H
#define WELDER_SOLVER_CANDIDATE_GENERATOR_H

#include "WelderSolverLib.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace welder::solver {

struct SearchFailure {
  std::string message;
  std::string hint;
};

bool runGenericCandidateSearch(mlir::ModuleOp module,
                               const welder::SolveOptions &opts,
                               welder::SolveResult &out,
                               SearchFailure *failure = nullptr);

bool runMatmulCandidateSearch(mlir::ModuleOp module,
                              const welder::SolveOptions &opts,
                              welder::SolveResult &out,
                              SearchFailure *failure = nullptr);

} // namespace welder::solver

#endif

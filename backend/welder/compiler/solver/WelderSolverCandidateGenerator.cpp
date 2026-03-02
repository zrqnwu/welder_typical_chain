#include "WelderSolverCandidateGenerator.h"

namespace welder::solver {

bool runGenericCandidateSearch(mlir::ModuleOp module,
                               const welder::SolveOptions &opts,
                               welder::SolveResult &out,
                               SearchFailure *failure) {
  out = welder::solveGeneric(module, opts);
  if (!out.sortedCandidates.empty())
    return true;

  if (failure) {
    failure->message = "error: no valid candidates under constraints.";
    failure->hint = "hint: try --require-perfect-tiling=false.";
  }
  return false;
}

bool runMatmulCandidateSearch(mlir::ModuleOp module,
                              const welder::SolveOptions &opts,
                              welder::SolveResult &out,
                              SearchFailure *failure) {
  out = welder::solve(module, opts);
  if (!out.sortedCandidates.empty())
    return true;

  if (failure) {
    failure->message = "error: no valid candidates under constraints.";
    failure->hint = "hint: try --require-perfect-tiling=false or increase --smem-bytes.";
  }
  return false;
}

} // namespace welder::solver

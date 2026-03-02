#ifndef WELDER_SOLVER_COST_MODEL_H
#define WELDER_SOLVER_COST_MODEL_H

#include "WelderSolverLib.h"

#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <vector>

namespace welder::solver {

void printIntList(const std::vector<int64_t> &xs, llvm::raw_ostream &os);

void printCandidateCostSummary(const welder::Candidate &c, llvm::raw_ostream &os);

void printTopKCandidateScores(const welder::SolveResult &sr, std::size_t k,
                              llvm::raw_ostream &os);

} // namespace welder::solver

#endif

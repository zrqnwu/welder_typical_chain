#ifndef WELDER_SOLVER_REPORT_WRITER_H
#define WELDER_SOLVER_REPORT_WRITER_H

#include "WelderSolverLib.h"

#include <string>

namespace welder::solver {

bool dumpBestSummaryJson(const welder::SolveResult &sr,
                         const welder::SolveOptions &opts,
                         const std::string &path);

bool dumpCandidatesTsv(const welder::SolveResult &sr,
                       const std::string &path);

} // namespace welder::solver

#endif

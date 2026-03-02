#pragma once

#include "WelderSolverLib.h"

#include <algorithm>

namespace welder {

inline void applyPaperScheduleAutoCandidateDefault(
    SolveOptions &opts, bool autoCandidatesExplicitlySet) {
  if (opts.enablePaperSchedule && !autoCandidatesExplicitlySet)
    opts.autoCandidates = true;
}

inline void applyPaperScheduleConnectLevelDefault(
    SolveOptions &opts, bool maxConnectLevelExplicitlySet) {
  if (opts.enablePaperSchedule && !maxConnectLevelExplicitlySet)
    opts.maxConnectLevel = std::max(opts.maxConnectLevel, 2);
}

inline void applyPaperStrictDefaults(SolveOptions &opts,
                                     bool profilingEnabled) {
  if (!opts.paperStrict)
    return;
  opts.autoCandidates = true;
  opts.enableRegisterLevelSchedule = true;
  opts.enableCoalescingPenalty = true;
  opts.paperExpandReductionTile = true;
  opts.paperRecursiveRegisterLevel = true;
  opts.maxConnectLevel = std::max(opts.maxConnectLevel, 2);
  if (profilingEnabled)
    opts.pruneOnProfileFailure = true;
}

inline void applyPaperModeDefaults(SolveOptions &opts,
                                   bool autoCandidatesExplicitlySet,
                                   bool maxConnectLevelExplicitlySet,
                                   bool profilingEnabled) {
  applyPaperScheduleAutoCandidateDefault(opts, autoCandidatesExplicitlySet);
  applyPaperScheduleConnectLevelDefault(opts, maxConnectLevelExplicitlySet);
  applyPaperStrictDefaults(opts, profilingEnabled);
}

} // 命名空间 welder

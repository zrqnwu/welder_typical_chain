#pragma once

#include "WelderCompilerModeDispatch.h"

namespace welder::compiler {

int buildTransformLibraryFromKernelAttrsBranch(const ModeDispatchContext &modeCtx);
int buildTransformLibraryFromGenericProblemBranch(const ModeDispatchContext &modeCtx);
int buildTransformLibraryFromMatmulBranch(const ModeDispatchContext &modeCtx);

} // namespace welder::compiler


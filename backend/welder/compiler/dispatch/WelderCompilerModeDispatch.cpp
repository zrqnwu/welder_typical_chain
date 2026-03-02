#include "WelderCompilerModeDispatch.h"

#include "WelderCompilerModeDispatchBranches.h"

#include "llvm/Support/raw_ostream.h"

namespace welder::compiler {

int buildTransformLibraryForMode(const ModeDispatchContext &modeCtx) {
  if (!modeCtx.module || !modeCtx.ctx || !modeCtx.transformLib ||
      !modeCtx.solveOpts) {
    llvm::errs() << "error: invalid mode dispatch context\n";
    return 2;
  }

  if (modeCtx.codegenFromKernelAttrs)
    return buildTransformLibraryFromKernelAttrsBranch(modeCtx);
  if (modeCtx.enableGenericProblem)
    return buildTransformLibraryFromGenericProblemBranch(modeCtx);
  return buildTransformLibraryFromMatmulBranch(modeCtx);
}

} // namespace welder::compiler


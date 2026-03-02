#include "WelderCompilerMemoryReuseTransforms.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>

namespace welder::compiler::memory_reuse_impl {

using namespace mlir;

// Keep this helper local to the memory-reuse TU so the impl header no longer
// depends on symbols from postprocess cleanup internals.
static Value stripToBaseMemref(Value v) {
  while (v) {
    if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
      v = sub.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto rcast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
      v = rcast.getSource();
      continue;
    }
    if (auto view = v.getDefiningOp<memref::ViewOp>()) {
      v = view.getSource();
      continue;
    }
    if (auto assumed = v.getDefiningOp<memref::AssumeAlignmentOp>()) {
      v = assumed.getMemref();
      continue;
    }
    break;
  }
  return v;
}

#include "WelderCompilerMemoryReuseTransformsImpl.h"

} // namespace welder::compiler::memory_reuse_impl

namespace welder::compiler {

void stageRowReduction2DTileToWorkgroup(mlir::ModuleOp module, int64_t smemBytes,
                                        int64_t reserveBytes,
                                        bool enableInPlace2DReuse,
                                        bool enableVectorize,
                                        bool enableAsyncCopy,
                                        bool enableSoftwarePipelining,
                                        bool asyncBypassL1,
                                        bool relaxBarriers,
                                        int64_t vectorWidth) {
  memory_reuse_impl::stageRowReduction2DTileToWorkgroup(
      module, smemBytes, reserveBytes, enableInPlace2DReuse, enableVectorize,
      enableAsyncCopy, enableSoftwarePipelining, asyncBypassL1, relaxBarriers,
      vectorWidth);
}

void promoteRowReductionScratchToWorkgroup(mlir::ModuleOp module) {
  memory_reuse_impl::promoteRowReductionScratchToWorkgroup(module);
}

void promoteSharedRowReductionResultAllocasToWorkgroup(mlir::ModuleOp module) {
  memory_reuse_impl::promoteSharedRowReductionResultAllocasToWorkgroup(module);
}

void promoteLaunchLocal1DBuffersToWorkgroup(mlir::ModuleOp module) {
  memory_reuse_impl::promoteLaunchLocal1DBuffersToWorkgroup(module);
}

} // namespace welder::compiler

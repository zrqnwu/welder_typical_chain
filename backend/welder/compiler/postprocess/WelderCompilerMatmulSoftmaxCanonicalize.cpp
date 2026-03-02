#include "WelderCompilerMatmulSoftmaxCanonicalize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace welder::compiler::matmul_softmax_impl {

using namespace mlir;

static bool isWorkgroupMemoryType(MemRefType memrefType) {
  if (!memrefType)
    return false;
  if (gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memrefType))
    return true;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
    return intAttr.getInt() == 3;
  return false;
}

static int64_t getElementByteWidth(Type elemType) {
  if (auto it = dyn_cast<IntegerType>(elemType))
    return std::max<int64_t>(1, (it.getWidth() + 7) / 8);
  if (auto ft = dyn_cast<FloatType>(elemType))
    return std::max<int64_t>(1, (ft.getWidth() + 7) / 8);
  return -1;
}

static bool isUsedInLaunch(Operation *op, gpu::LaunchOp launch) {
  if (!op || !launch)
    return false;
  return launch->isAncestor(op);
}

// Localized copy to decouple matmul/softmax canonicalization from memory-reuse
// internals after implementation split.
static void fixSubviewResultTypesAfterWorkgroupPromotion(ModuleOp module) {
  if (!module)
    return;

  SmallVector<memref::SubViewOp, 16> subviewsToFix;
  module.walk([&](memref::SubViewOp sv) {
    auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
    auto resTy = dyn_cast<MemRefType>(sv.getResult().getType());
    if (!srcTy || !resTy)
      return;
    if (!isWorkgroupMemoryType(srcTy))
      return;
    if (isWorkgroupMemoryType(resTy))
      return;
    subviewsToFix.push_back(sv);
  });

  for (memref::SubViewOp sv : subviewsToFix) {
    auto srcTy = dyn_cast<MemRefType>(sv.getSource().getType());
    auto oldResTy = dyn_cast<MemRefType>(sv.getResult().getType());
    if (!srcTy || !oldResTy)
      continue;

    MemRefType newResTy = memref::SubViewOp::inferRankReducedResultType(
        oldResTy.getShape(), srcTy, sv.getMixedOffsets(), sv.getMixedSizes(),
        sv.getMixedStrides());
    if (newResTy == oldResTy)
      continue;

    OpBuilder b(sv);
    auto newSv = memref::SubViewOp::create(
        b, sv.getLoc(), newResTy, sv.getSource(), sv.getMixedOffsets(),
        sv.getMixedSizes(), sv.getMixedStrides());
    sv.getResult().replaceAllUsesWith(newSv.getResult());
    sv.erase();
  }
}

#include "WelderCompilerMatmulSoftmaxCanonicalizeImpl.h"

} // namespace welder::compiler::matmul_softmax_impl

namespace welder::compiler {

void canonicalizeMatmulSoftmaxSharedReuse(mlir::ModuleOp module) {
  matmul_softmax_impl::canonicalizeMatmulSoftmaxSharedReuse(module);
}

} // namespace welder::compiler

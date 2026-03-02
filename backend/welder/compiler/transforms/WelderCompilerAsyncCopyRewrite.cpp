#include "WelderCompilerAsyncCopyRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace {

static bool isWorkgroupMemoryTypeLocal(MemRefType memrefType) {
  if (!memrefType)
    return false;
  if (gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(memrefType))
    return true;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
    return intAttr.getInt() == 3;
  return false;
}

static int64_t getElementByteWidthLocal(Type elemType) {
  if (!elemType)
    return -1;
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return (intTy.getWidth() + 7) / 8;
  if (auto floatTy = dyn_cast<FloatType>(elemType))
    return (floatTy.getWidth() + 7) / 8;
  return -1;
}

static Value stripToBaseMemrefLocal(Value v) {
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
    break;
  }
  return v;
}

} // namespace

namespace welder::compiler {

void rewritePromotedLinalgCopiesToAsyncCopy(ModuleOp module, bool enableAsyncCopy,
                                            bool asyncBypassL1) {
  if (!module || !enableAsyncCopy)
    return;
  MLIRContext *ctx = module.getContext();
  if (!ctx)
    return;

  const int64_t kAsyncBytes = 16;
  Builder b(ctx);
  UnitAttr bypassL1Hint = asyncBypassL1 ? UnitAttr::get(ctx) : UnitAttr();

  SmallVector<linalg::CopyOp, 64> copies;
  module.walk([&](linalg::CopyOp copy) {
    if (!copy)
      return;
    if (!copy->getParentOfType<gpu::LaunchOp>())
      return;
    copies.push_back(copy);
  });

  for (linalg::CopyOp copy : copies) {
    Value src = copy.getInputs().empty() ? Value() : copy.getInputs().front();
    Value dst = copy.getOutputs().empty() ? Value() : copy.getOutputs().front();
    auto srcTy = src ? dyn_cast<MemRefType>(src.getType()) : MemRefType();
    auto dstTy = dst ? dyn_cast<MemRefType>(dst.getType()) : MemRefType();
    if (!srcTy || !dstTy)
      continue;

    // 仅重写 global->workgroup 拷贝。
    if (!isWorkgroupMemoryTypeLocal(dstTy))
      continue;
    if (isWorkgroupMemoryTypeLocal(srcTy))
      continue;
    auto isGlobalMem = [](MemRefType ty) -> bool {
      Attribute ms = ty.getMemorySpace();
      if (!ms)
        return true;
      if (auto intAttr = dyn_cast<IntegerAttr>(ms))
        return intAttr.getInt() == 0;
      return false;
    };
    if (!isGlobalMem(srcTy))
      continue;
    Value srcBase = stripToBaseMemrefLocal(src);
    auto ba = dyn_cast_or_null<BlockArgument>(srcBase);
    if (!ba)
      continue;
    Operation *parent = ba.getOwner() ? ba.getOwner()->getParentOp() : nullptr;
    if (!parent || (!isa<gpu::LaunchOp>(parent) && !isa<func::FuncOp>(parent)))
      continue;

    if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
      continue;
    if (!srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      continue;
    if (srcTy.getShape() != dstTy.getShape())
      continue;
    if (srcTy.getElementType() != dstTy.getElementType())
      continue;

    int64_t elemBytes = getElementByteWidthLocal(srcTy.getElementType());
    if (elemBytes <= 0 || (kAsyncBytes % elemBytes) != 0)
      continue;
    int64_t maxVecElems = std::max<int64_t>(1, kAsyncBytes / elemBytes);

    // NVGPU async copy 要求最内层维度 unit stride
    // （dst 必须静态满足；src 可动态，并在运行时加保护）。
    if (!dstTy.isLastDimUnitStride())
      continue;

    ArrayRef<int64_t> shape = srcTy.getShape();
    int64_t tileM = shape[0];
    int64_t tileN = shape[1];
    if (tileM <= 0 || tileN <= 0)
      continue;

    OpBuilder ib(copy);
    Location loc = copy.getLoc();

    Value c0 = ib.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = ib.create<arith::ConstantIndexOp>(loc, 1);
    Value c1i1 = ib.create<arith::ConstantIntOp>(loc, 1, 1);
    Value strideOk;
    Value asyncSrc = src;

    if (!srcTy.isLastDimUnitStride()) {
      auto meta = ib.create<memref::ExtractStridedMetadataOp>(loc, src);
      SmallVector<Value, 4> strides(meta.getStrides().begin(),
                                    meta.getStrides().end());
      if (strides.size() != 2)
        continue;
      Value stride0 = strides[0];
      Value stride1 = strides[1];
      strideOk = ib.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, stride1,
                                          c1);

      // 将类型重解释为“最后一维静态 stride=1”，使 NVGPU verifier
      // 接受 `device_async_copy`（真实正确性由 `strideOk` 保障）。
      auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                           {ShapedType::kDynamic, 1});
      // 注意：当存在动态 size 操作数时，`memref.reinterpret_cast`
      // 期望结果类型是动态 size，即使运行时 size 是常量。
      SmallVector<int64_t, 2> dynShape{ShapedType::kDynamic,
                                       ShapedType::kDynamic};
      auto asyncSrcTy = MemRefType::get(dynShape, srcTy.getElementType(), layout,
                                        srcTy.getMemorySpace());
      Value tileMVal = ib.create<arith::ConstantIndexOp>(loc, tileM);
      Value tileNVal = ib.create<arith::ConstantIndexOp>(loc, tileN);
      SmallVector<OpFoldResult, 2> asyncSizes{tileMVal, tileNVal};
      SmallVector<OpFoldResult, 2> asyncStrides{stride0, ib.getIndexAttr(1)};
      auto asyncSrcCast = memref::ReinterpretCastOp::create(
          ib, loc, asyncSrcTy, src, meta.getOffset(), asyncSizes, asyncStrides);
      asyncSrc = asyncSrcCast.getResult();
    } else {
      strideOk = ib.create<arith::ConstantIntOp>(loc, 1, 1);
    }

    // 发射 async copy（按最内层维度分块）。
    SmallVector<Value, 32> tokens;
    auto tokenTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    for (int64_t r = 0; r < tileM; ++r) {
      Value row = ib.create<arith::ConstantIndexOp>(loc, r);
      for (int64_t c = 0; c < tileN; c += maxVecElems) {
        int64_t segElems = std::min<int64_t>(maxVecElems, tileN - c);
        Value col = ib.create<arith::ConstantIndexOp>(loc, c);
        Value segV = ib.create<arith::ConstantIndexOp>(loc, segElems);
        Value srcElems = segElems == 0
                             ? c0
                             : ib.create<arith::SelectOp>(loc, strideOk, segV,
                                                          c0);
        UnitAttr bypass =
            (bypassL1Hint && (segElems * elemBytes) == kAsyncBytes)
                ? bypassL1Hint
                : UnitAttr();

        auto tok = nvgpu::DeviceAsyncCopyOp::create(
            ib, loc, tokenTy,
            /*dst=*/dst, /*dstIndices=*/ValueRange{row, col},
            /*src=*/asyncSrc, /*srcIndices=*/ValueRange{row, col},
            /*dstElements=*/ib.getIndexAttr(segElems),
            /*srcElements=*/srcElems,
            /*bypassL1=*/bypass);
        tokens.push_back(tok.getAsyncToken());
      }
    }
    if (tokens.empty())
      continue;

    auto group = ib.create<nvgpu::DeviceAsyncCreateGroupOp>(loc, tokenTy,
                                                             ValueRange(tokens));
    (void)ib.create<nvgpu::DeviceAsyncWaitOp>(loc, group.getAsyncToken(),
                                              IntegerAttr());

    // 对非 unit-stride 的源执行运行时回退：仅在 `strideOk == false` 时
    // 执行原始 copy。
    if (!srcTy.isLastDimUnitStride()) {
      Value notOk = ib.create<arith::XOrIOp>(loc, strideOk, c1i1);
      auto ifOp = ib.create<scf::IfOp>(loc, notOk, /*withElse=*/false);
      OpBuilder fb(ifOp.thenBlock(), ifOp.thenBlock()->begin());
      (void)fb.create<linalg::CopyOp>(loc, src, dst);
    }

    copy.erase();
  }
}

} // namespace welder::compiler

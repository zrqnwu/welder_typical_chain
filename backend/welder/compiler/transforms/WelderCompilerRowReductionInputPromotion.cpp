#include "WelderCompilerRowReductionInputPromotion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace welder::compiler {

void promoteRowReductionInputsToWorkgroup(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  auto workgroupAttr = b.getI64IntegerAttr(3);

  module.walk([&](linalg::GenericOp op) {
    if (!op->hasAttr("welder.row_reduction"))
      return;
    if (!op->getParentOfType<gpu::LaunchOp>())
      return;
    if (op.getNumDpsInputs() < 1)
      return;

    Value in = op.getDpsInputOperand(0)->get();
    auto subview = in.getDefiningOp<memref::SubViewOp>();
    if (!subview) {
      op.emitWarning() << "row-reduction input promotion skipped: input is not "
                          "a subview (needs static tile to safely promote)";
      return;
    }
    for (Value off : subview.getOffsets()) {
      if (!off.getDefiningOp<arith::ConstantIndexOp>()) {
        op.emitWarning() << "row-reduction input promotion skipped: dynamic "
                            "subview offsets (needs block-tile promotion)";
        return;
      }
    }
    auto memrefType = dyn_cast<MemRefType>(in.getType());
    if (memrefType && memrefType.getMemorySpace() == workgroupAttr)
      return;

    b.setInsertionPoint(op);
    linalg::LinalgPromotionOptions options;
    options.setOperandsToPromote({0});
    options.setUseFullTileBuffers({false});
    options.setUseFullTileBuffersByDefault(false);
    options.setUseOriginalSubviewSize(false);
    options.setUseAlloca(false);
    options.setMemorySpace(workgroupAttr);
    options.setCopyInOutFns(
        [&](OpBuilder &cb, Value src, Value dst) -> LogicalResult {
          Location loc = src.getLoc();
          Value tidx = cb.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
          Value tidy = cb.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
          Value c0 = cb.create<arith::ConstantIndexOp>(loc, 0);
          Value predX = cb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 tidx, c0);
          Value predY = cb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 tidy, c0);
          Value pred = cb.create<arith::AndIOp>(loc, predX, predY);
          auto ifOp = cb.create<scf::IfOp>(loc, pred, /*withElse=*/false);
          cb.setInsertionPointToStart(ifOp.thenBlock());
          cb.create<linalg::CopyOp>(loc, src, dst);
          cb.setInsertionPointAfter(ifOp);
          cb.create<gpu::BarrierOp>(loc);
          return success();
        },
        [&](OpBuilder &cb, Value src, Value dst) -> LogicalResult {
          cb.create<linalg::CopyOp>(src.getLoc(), src, dst);
          return success();
        });

    if (failed(linalg::promoteSubViews(b, op, options))) {
      op.emitWarning()
          << "row-reduction input promotion failed; keeping global input";
    }
  });
}

} // namespace welder::compiler

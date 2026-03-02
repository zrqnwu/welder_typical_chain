#include "WelderCompilerFusionAnchors.h"

#include "WelderCompilerPassTraceAndEnv.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

Value stripToBaseMemrefLocal(Value v) {
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

bool isRowWiseReductionOpLocal(Operation *op) {
  auto gen = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!gen)
    return false;
  if (gen.getNumLoops() != 2 || gen.getNumReductionLoops() != 1)
    return false;
  auto iters = gen.getIteratorTypesArray();
  if (iters.size() != 2)
    return false;
  return iters[0] == mlir::utils::IteratorType::parallel &&
         iters[1] == mlir::utils::IteratorType::reduction;
}

} // namespace

namespace welder::compiler {

void eraseDeadHostIntermediateAllocs(ModuleOp module) {
  if (!module)
    return;

  auto isViewLike = [](Operation *op) -> bool {
    return isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
               memref::ExpandShapeOp, memref::CollapseShapeOp>(op);
  };
  auto getViewResult = [](Operation *op) -> Value {
    if (auto sv = dyn_cast<memref::SubViewOp>(op))
      return sv.getResult();
    if (auto cast = dyn_cast<memref::CastOp>(op))
      return cast.getResult();
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(op))
      return rc.getResult();
    if (auto ex = dyn_cast<memref::ExpandShapeOp>(op))
      return ex.getResult();
    if (auto co = dyn_cast<memref::CollapseShapeOp>(op))
      return co.getResult();
    return {};
  };

  module.walk([&](func::FuncOp func) {
    struct AllocCandidate {
      Operation *op = nullptr;
      Value memref;
    };

    llvm::SmallVector<AllocCandidate, 8> allocs;
    func.walk([&](memref::AllocOp a) {
      if (a->getParentOfType<gpu::LaunchOp>())
        return;
      allocs.push_back(AllocCandidate{a.getOperation(), a.getResult()});
    });
    func.walk([&](gpu::AllocOp a) {
      if (a->getParentOfType<gpu::LaunchOp>())
        return;
      allocs.push_back(AllocCandidate{a.getOperation(), a.getMemref()});
    });

    llvm::SmallPtrSet<Operation *, 32> eraseSet;
    llvm::SmallVector<Operation *, 128> erasePostOrder;

    auto collectDead =
        [&](auto &&self, Value v, llvm::DenseSet<Value> &visiting,
            llvm::SmallVectorImpl<Operation *> &localPostOrder,
            llvm::SmallPtrSetImpl<Operation *> &localEraseSet) -> bool {
      if (!v)
        return false;
      if (visiting.contains(v))
        return true;
      visiting.insert(v);

      for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
        Operation *user = use.getOwner();
        if (!user)
          return false;

        if (isViewLike(user)) {
          Value res = getViewResult(user);
          if (!res || !self(self, res, visiting, localPostOrder, localEraseSet))
            return false;
          if (localEraseSet.insert(user).second)
            localPostOrder.push_back(user);
          continue;
        }

        if (auto fill = dyn_cast<linalg::FillOp>(user)) {
          if (fill.getNumDpsInits() != 1)
            return false;
          Value out = fill.getDpsInitOperand(0)->get();
          if (stripToBaseMemrefLocal(out) != stripToBaseMemrefLocal(v))
            return false;
          if (localEraseSet.insert(user).second)
            localPostOrder.push_back(user);
          continue;
        }

        if (isa<gpu::DeallocOp, memref::DeallocOp>(user)) {
          if (localEraseSet.insert(user).second)
            localPostOrder.push_back(user);
          continue;
        }

        return false;
      }

      return true;
    };

    for (const AllocCandidate &a : allocs) {
      if (!a.op || !a.memref)
        continue;
      llvm::DenseSet<Value> visiting;
      llvm::SmallVector<Operation *, 16> localPostOrder;
      llvm::SmallPtrSet<Operation *, 16> localEraseSet;
      if (!collectDead(collectDead, a.memref, visiting, localPostOrder,
                       localEraseSet))
        continue;
      if (localEraseSet.insert(a.op).second)
        localPostOrder.push_back(a.op);
      for (Operation *op : localPostOrder) {
        if (eraseSet.insert(op).second)
          erasePostOrder.push_back(op);
      }
    }

    for (Operation *op : erasePostOrder) {
      if (!op)
        continue;
      if (op->getNumResults() > 0) {
        bool allUnused = true;
        for (Value r : op->getResults()) {
          if (!r.use_empty()) {
            allUnused = false;
            break;
          }
        }
        if (!allUnused)
          continue;
      }
      op->erase();
    }
  });
}

void tagLinalgOpsForGenericCodegen(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  auto unit = UnitAttr::get(ctx);

  module.walk([&](linalg::GenericOp op) {
    if (!op->hasAttr("welder.kernel_id"))
      return;

    op->removeAttr("welder.elementwise");
    op->removeAttr("welder.elementwise_1d");
    op->removeAttr("welder.elementwise_nd");
    op->removeAttr("welder.row_reduction");
    op->removeAttr("welder.vectorizable");

    const int64_t nLoops = op.getNumLoops();
    const int64_t nRed = op.getNumReductionLoops();
    if (nRed == 0) {
      op->setAttr("welder.elementwise", unit);
      if (nLoops == 1)
        op->setAttr("welder.elementwise_1d", unit);
      else
        op->setAttr("welder.elementwise_nd", unit);
      bool allParallel = true;
      auto iters = op.getIteratorTypesArray();
      for (auto t : iters) {
        if (t != mlir::utils::IteratorType::parallel) {
          allParallel = false;
          break;
        }
      }
      if (allParallel) {
        bool allId = true;
        auto maps = op.getIndexingMapsArray();
        for (AffineMap m : maps) {
          if (!m.isIdentity()) {
            allId = false;
            break;
          }
        }
        if (allId) {
          bool sameElemType = true;
          Type elemTy;
          for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i) {
            OpOperand *in = op.getDpsInputOperand(i);
            auto mt = dyn_cast<ShapedType>(in->get().getType());
            if (!mt) {
              sameElemType = false;
              break;
            }
            if (!elemTy)
              elemTy = mt.getElementType();
            else if (mt.getElementType() != elemTy)
              sameElemType = false;
          }
          for (unsigned i = 0, e = op.getNumDpsInits(); i < e; ++i) {
            OpOperand *out = op.getDpsInitOperand(i);
            auto mt = dyn_cast<ShapedType>(out->get().getType());
            if (!mt) {
              sameElemType = false;
              break;
            }
            if (!elemTy)
              elemTy = mt.getElementType();
            else if (mt.getElementType() != elemTy)
              sameElemType = false;
          }
          if (sameElemType && elemTy && elemTy.isF32())
            op->setAttr("welder.vectorizable", unit);
        }
      }
      return;
    }

    auto iters = op.getIteratorTypesArray();
    if (nLoops == 2 && nRed == 1 && iters.size() == 2 &&
        iters[0] == mlir::utils::IteratorType::parallel &&
        iters[1] == mlir::utils::IteratorType::reduction) {
      op->setAttr("welder.row_reduction", unit);
    }
  });
}

Operation *findFirstLinalgConsumer(Operation *producer) {
  if (!producer)
    return nullptr;

  llvm::SmallVector<Value, 8> worklist;
  worklist.append(producer->result_begin(), producer->result_end());

  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!user)
        continue;
      if (auto lop = dyn_cast<linalg::LinalgOp>(user))
        return lop.getOperation();

      if (visited.contains(user))
        continue;
      visited.insert(user);

      if (isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
              tensor::ExpandShapeOp>(user)) {
        worklist.append(user->result_begin(), user->result_end());
      }
    }
  }
  return nullptr;
}

void markMatmulFusionAnchors(ModuleOp module) {
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();
  bool debugConsumer =
      (getEnvInt64OrDefault("WELDER_DEBUG_MATMUL_CONSUMER", 0) != 0);
  module.walk([&](linalg::LinalgOp op) {
    op->removeAttr("welder.matmul_consumer");
    op->removeAttr("welder.matmul_producer");
  });

  func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
  if (!main)
    return;

  func::ReturnOp ret;
  main.walk([&](func::ReturnOp r) {
    if (!ret)
      ret = r;
  });
  if (!ret || ret.getNumOperands() != 1)
    return;

  auto stripTensorPassthrough = [](Value v) -> Value {
    while (v) {
      Operation *def = v.getDefiningOp();
      if (!def)
        break;
      if (isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
              tensor::ExpandShapeOp>(def)) {
        if (def->getNumOperands() < 1)
          break;
        v = def->getOperand(0);
        continue;
      }
      break;
    }
    return v;
  };

  auto countParallel = [&](linalg::LinalgOp op) -> int64_t {
    int64_t cnt = 0;
    for (auto it : op.getIteratorTypesArray()) {
      if (it == mlir::utils::IteratorType::parallel)
        ++cnt;
    }
    return cnt;
  };

  Operation *sinkOp = nullptr;
  linalg::LinalgOp sinkLinalg;
  main.walk([&](linalg::LinalgOp op) {
    if (countParallel(op) >= 2) {
      sinkOp = op.getOperation();
      sinkLinalg = op;
    }
  });
  if (!sinkLinalg) {
    Value out = stripTensorPassthrough(ret.getOperand(0));
    sinkOp = out ? out.getDefiningOp() : nullptr;
    sinkLinalg = dyn_cast_or_null<linalg::LinalgOp>(sinkOp);
  }
  if (!sinkLinalg)
    return;

  sinkOp->setAttr("welder.matmul_consumer", UnitAttr::get(ctx));
  if (debugConsumer) {
    llvm::errs() << "[welder] matmul_consumer="
                 << sinkOp->getName().getStringRef() << " loc="
                 << sinkOp->getLoc() << " parallel="
                 << countParallel(sinkLinalg) << "\n";
  }

  llvm::SmallPtrSet<Operation *, 16> visited;
  llvm::SmallVector<Operation *, 8> stack;

  auto tryPushProducer = [&](Value v) {
    v = stripTensorPassthrough(v);
    Operation *def = v ? v.getDefiningOp() : nullptr;
    auto lop = dyn_cast_or_null<linalg::LinalgOp>(def);
    if (!lop)
      return;
    if (isRowWiseReductionOpLocal(def))
      return;
    Operation *op = lop.getOperation();
    if (!visited.contains(op)) {
      visited.insert(op);
      stack.push_back(op);
    }
  };

  for (Value in : sinkLinalg.getDpsInputs())
    tryPushProducer(in);

  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    op->setAttr("welder.matmul_producer", UnitAttr::get(ctx));

    auto lop = dyn_cast<linalg::LinalgOp>(op);
    if (!lop)
      continue;
    for (Value in : lop.getDpsInputs())
      tryPushProducer(in);
  }
}

} // namespace welder::compiler

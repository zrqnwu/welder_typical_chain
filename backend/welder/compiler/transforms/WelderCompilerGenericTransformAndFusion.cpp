#include "WelderCompilerGenericTransformAndFusion.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace {

int64_t countNonZeroI64Local(ArrayRef<int64_t> xs) {
  int64_t n = 0;
  for (int64_t v : xs)
    if (v != 0)
      ++n;
  return n;
}

ArrayAttr makeGpuBlockMappingLocal(MLIRContext *ctx, bool swapXY) {
  if (swapXY) {
    return ArrayAttr::get(ctx, {gpu::GPUBlockMappingAttr::get(
                                   ctx, gpu::MappingId::LinearDim0),
                                gpu::GPUBlockMappingAttr::get(
                                   ctx, gpu::MappingId::LinearDim1)});
  }
  return ArrayAttr::get(ctx, {gpu::GPUBlockMappingAttr::get(
                                 ctx, gpu::MappingId::LinearDim1),
                              gpu::GPUBlockMappingAttr::get(
                                 ctx, gpu::MappingId::LinearDim0)});
}

ArrayAttr makeGpuBlockMappingNLocal(MLIRContext *ctx, int64_t nDims,
                                    bool swapXY) {
  if (nDims <= 0)
    return ArrayAttr::get(ctx, {});
  if (nDims == 1) {
    return ArrayAttr::get(ctx, {gpu::GPUBlockMappingAttr::get(
                                   ctx, gpu::MappingId::LinearDim0)});
  }
  if (nDims == 2)
    return makeGpuBlockMappingLocal(ctx, swapXY);
  return ArrayAttr::get(
      ctx, {gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim2),
            gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim1),
            gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim0)});
}

uint64_t packKernelNodeKey(int64_t kernelId, int64_t nodeId) {
  uint64_t hi = static_cast<uint64_t>(static_cast<uint32_t>(kernelId));
  uint64_t lo = static_cast<uint64_t>(static_cast<uint32_t>(nodeId));
  return (hi << 32) ^ lo;
}

} // namespace

namespace welder::compiler {

OwningOpRef<ModuleOp>
buildGenericTransformLibrary(MLIRContext *ctx, StringRef targetOpName,
                             ArrayRef<int64_t> l1TileSizes,
                             ArrayRef<int64_t> l2TileSizes, bool enableFusion,
                             StringRef consumerOpName,
                             ArrayRef<int64_t> consumerTileSizes,
                             bool swapBlockDims,
                             bool skipMapForallToBlocks) {
  OpBuilder b(ctx);
  Location loc = b.getUnknownLoc();
  auto lib = ModuleOp::create(loc);
  lib->setAttr("transform.with_named_sequence", UnitAttr::get(ctx));

  auto anyOp = transform::AnyOpType::get(ctx);
  DictionaryAttr readonlyArg =
      DictionaryAttr::get(ctx, {b.getNamedAttr("transform.readonly",
                                              UnitAttr::get(ctx))});
  llvm::SmallVector<DictionaryAttr, 1> argAttrs{readonlyArg};

  const int64_t numBlockDims = countNonZeroI64Local(l1TileSizes);
  const bool doL2 = countNonZeroI64Local(l2TileSizes) > 0;
  const int64_t consumerBlockDims = countNonZeroI64Local(consumerTileSizes);

  b.setInsertionPointToEnd(lib.getBody());

  // === Pre-bufferize 阶段（generic）===
  //
  // Phase 10（无 fusion）：保持空操作。
  // Phase 11（generic fusion，第一版）：tile consumer（block 级）+ fuse producer。
  b.create<transform::NamedSequenceOp>(
      loc, "__welder_prebufferize", anyOp, TypeRange{},
      [&](OpBuilder &b, Location loc, BlockArgument root) {
        if (!enableFusion) {
          b.create<transform::YieldOp>(loc);
          return;
        }

        // 匹配 `func @main`
        auto funcMatch =
            b.create<transform::MatchOp>(loc, root,
                                         ArrayRef<StringRef>{"func.func"});
        funcMatch->setAttr(
            "op_attrs",
            DictionaryAttr::get(ctx, {b.getNamedAttr(
                                         "sym_name",
                                         b.getStringAttr("main"))}));

        // 匹配 consumer（由 C++ 侧 `welder.consumer` 锚点保证唯一）
        auto consumerMatch =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{consumerOpName});
        consumerMatch->setAttr(
            "op_attrs",
            DictionaryAttr::get(ctx, {b.getNamedAttr("welder.consumer",
                                                    UnitAttr::get(ctx))}));

        // 匹配 producer（同样通过 `welder.target` 锚点保证唯一）
        auto producerMatch =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{targetOpName});
        producerMatch->setAttr(
            "op_attrs",
            DictionaryAttr::get(ctx, {b.getNamedAttr("welder.target",
                                                    UnitAttr::get(ctx))}));

        if (consumerBlockDims <= 0) {
          b.create<transform::YieldOp>(loc);
          return;
        }

        // 对 consumer 做 tile -> scf.forall（映射到 gpu.block）。
        ArrayAttr blockMapping =
            makeGpuBlockMappingNLocal(ctx, consumerBlockDims, swapBlockDims);
        auto tiledConsumer = transform::TileUsingForallOp::create(
            b, loc, consumerMatch.getResult(), consumerTileSizes,
            transform::TileSizesSpec(), blockMapping);

        // 将 producer 融合进 consumer 的 forall（tile-and-fuse / clone-and-fuse）。
        (void)transform::FuseIntoContainingOp::create(
            b, loc, producerMatch.getResult(), tiledConsumer.getForallOp());

        b.create<transform::YieldOp>(loc);
      },
      /* attrs=*/ArrayRef<NamedAttribute>{},
      /* argAttrs=*/argAttrs);

  // === Post-bufferize（generic）：动态 Match + 动态 Tile（Phase 10 最小实现）===
  b.create<transform::NamedSequenceOp>(
      loc, "__welder_postbufferize", anyOp, TypeRange{},
      [&](OpBuilder &b, Location loc, BlockArgument root) {
        // 匹配 `func @main`
        auto funcMatch =
            b.create<transform::MatchOp>(loc, root,
                                         ArrayRef<StringRef>{"func.func"});
        funcMatch->setAttr(
            "op_attrs",
            DictionaryAttr::get(ctx, {b.getNamedAttr(
                                         "sym_name",
                                         b.getStringAttr("main"))}));

        if (!enableFusion) {
          // 匹配目标算子（通过自定义 attribute 精准定位）
          auto targetMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{targetOpName});
          targetMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(ctx, {b.getNamedAttr("welder.target",
                                                      UnitAttr::get(ctx))}));

          if (numBlockDims <= 0) {
            // 若没有 parallel 维可映射到 block，则该算子当前不做 GPU 分发。
            b.create<transform::YieldOp>(loc);
            return;
          }

          // L1 tiling：仅对空间维（parallel）做切分，并映射到 gpu.block。
          ArrayAttr blockMapping =
              makeGpuBlockMappingNLocal(ctx, numBlockDims, swapBlockDims);
          auto tiledL1 = transform::TileUsingForallOp::create(
              b, loc, targetMatch.getResult(), l1TileSizes,
              transform::TileSizesSpec(), blockMapping);

          Value tiledHandle = tiledL1.getTiledOp();

          // L2 tiling：对归约维做切分（K-split / reduction split）。
          if (doL2) {
            auto tiledL2 = transform::TileUsingForOp::create(
                b, loc, tiledHandle, l2TileSizes);
            (void)tiledL2;
          }
        }

        // 将顶层 forall 映射到 blocks -> gpu.launch。
        if (!skipMapForallToBlocks) {
          (void)transform::MapForallToBlocks::create(
              b, loc, anyOp, funcMatch.getResult(),
              /* grid_dims=*/ArrayRef<int64_t>{},
              /* generate_gpu_launch=*/true);
        }

        b.create<transform::YieldOp>(loc);
      },
      /* attrs=*/ArrayRef<NamedAttribute>{},
      /* argAttrs=*/argAttrs);

  return lib;
}

void normalizeThreadFusionPairs(std::vector<ThreadFusionPair> &pairs) {
  if (pairs.empty())
    return;

  std::sort(pairs.begin(), pairs.end(),
            [](const ThreadFusionPair &a, const ThreadFusionPair &b) {
              if (a.kernelId != b.kernelId)
                return a.kernelId < b.kernelId;
              if (a.consumerNodeId != b.consumerNodeId)
                return a.consumerNodeId < b.consumerNodeId;
              if (a.consumerOperand != b.consumerOperand)
                return a.consumerOperand < b.consumerOperand;
              return a.producerNodeId < b.producerNodeId;
            });
  pairs.erase(std::unique(pairs.begin(), pairs.end(),
                          [](const ThreadFusionPair &a,
                             const ThreadFusionPair &b) {
                            return a.kernelId == b.kernelId &&
                                   a.producerNodeId == b.producerNodeId &&
                                   a.consumerNodeId == b.consumerNodeId &&
                                   a.consumerOperand == b.consumerOperand;
                          }),
              pairs.end());
  if (pairs.size() <= 1)
    return;

  llvm::DenseMap<uint64_t, int64_t> nodeLevel;
  for (const ThreadFusionPair &pair : pairs) {
    nodeLevel.try_emplace(
        packKernelNodeKey(pair.kernelId, pair.producerNodeId), 0);
    nodeLevel.try_emplace(
        packKernelNodeKey(pair.kernelId, pair.consumerNodeId), 0);
  }

  for (size_t iter = 0; iter < pairs.size(); ++iter) {
    bool changed = false;
    for (const ThreadFusionPair &pair : pairs) {
      const uint64_t prodKey =
          packKernelNodeKey(pair.kernelId, pair.producerNodeId);
      const uint64_t consKey =
          packKernelNodeKey(pair.kernelId, pair.consumerNodeId);
      const int64_t srcLevel = nodeLevel.lookup(prodKey);
      int64_t &dstLevel = nodeLevel[consKey];
      if (dstLevel < srcLevel + 1) {
        dstLevel = srcLevel + 1;
        changed = true;
      }
    }
    if (!changed)
      break;
  }

  std::stable_sort(pairs.begin(), pairs.end(),
                   [&](const ThreadFusionPair &a, const ThreadFusionPair &b) {
                     const int64_t aConsLevel = nodeLevel.lookup(
                         packKernelNodeKey(a.kernelId, a.consumerNodeId));
                     const int64_t bConsLevel = nodeLevel.lookup(
                         packKernelNodeKey(b.kernelId, b.consumerNodeId));
                     if (aConsLevel != bConsLevel)
                       return aConsLevel < bConsLevel;
                     const int64_t aProdLevel = nodeLevel.lookup(
                         packKernelNodeKey(a.kernelId, a.producerNodeId));
                     const int64_t bProdLevel = nodeLevel.lookup(
                         packKernelNodeKey(b.kernelId, b.producerNodeId));
                     if (aProdLevel != bProdLevel)
                       return aProdLevel < bProdLevel;
                     if (a.kernelId != b.kernelId)
                       return a.kernelId < b.kernelId;
                     if (a.consumerNodeId != b.consumerNodeId)
                       return a.consumerNodeId < b.consumerNodeId;
                     if (a.consumerOperand != b.consumerOperand)
                       return a.consumerOperand < b.consumerOperand;
                     return a.producerNodeId < b.producerNodeId;
                   });
}

} // namespace welder::compiler

#include "WelderCompilerGenericCutEdgesTransformLibrary.h"

#include "WelderCompilerPassTraceAndEnv.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"

#include <algorithm>

using namespace mlir;

namespace {

ArrayAttr makeGpuBlockMapping(MLIRContext *ctx, bool swapXY) {
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

ArrayAttr makeGpuBlockMappingN(MLIRContext *ctx, int64_t nDims, bool swapXY) {
  if (nDims <= 0)
    return ArrayAttr::get(ctx, {});
  if (nDims == 1) {
    return ArrayAttr::get(ctx, {gpu::GPUBlockMappingAttr::get(
                                   ctx, gpu::MappingId::LinearDim0)});
  }
  if (nDims == 2)
    return makeGpuBlockMapping(ctx, swapXY);
  return ArrayAttr::get(
      ctx, {gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim2),
            gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim1),
            gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::LinearDim0)});
}

ArrayAttr makeGpuThreadMapping(MLIRContext *ctx, bool swapXY) {
  if (swapXY) {
    return ArrayAttr::get(
        ctx,
        {gpu::GPUThreadMappingAttr::get(ctx, gpu::MappingId::LinearDim0),
         gpu::GPUThreadMappingAttr::get(ctx, gpu::MappingId::LinearDim1)});
  }
  return ArrayAttr::get(
      ctx, {gpu::GPUThreadMappingAttr::get(ctx, gpu::MappingId::LinearDim1),
            gpu::GPUThreadMappingAttr::get(ctx, gpu::MappingId::LinearDim0)});
}

ArrayAttr makeGpuWarpMapping(MLIRContext *ctx, bool swapXY) {
  if (swapXY) {
    return ArrayAttr::get(
        ctx,
        {gpu::GPUWarpMappingAttr::get(ctx, gpu::MappingId::LinearDim0),
         gpu::GPUWarpMappingAttr::get(ctx, gpu::MappingId::LinearDim1)});
  }
  return ArrayAttr::get(
      ctx, {gpu::GPUWarpMappingAttr::get(ctx, gpu::MappingId::LinearDim1),
            gpu::GPUWarpMappingAttr::get(ctx, gpu::MappingId::LinearDim0)});
}

int64_t countNonZeroI64(ArrayRef<int64_t> xs) {
  int64_t n = 0;
  for (int64_t v : xs)
    if (v != 0)
      ++n;
  return n;
}

} // namespace

namespace welder::compiler {

// Phase 13B：基于 cut-edge 的多 kernel 生成（最小可用版本）。
// - 每个 KernelSpec 对应一个“kernel root”（sink + cut producers）。
// - 对 root 做 L1 tile -> scf.forall（gpu.block）。
// - 仅 fuse 直接 producer（且必须是未 cut 的边）。
// - 让 MapForallToBlocks 在 postbufferize 阶段把多个 top-level forall 变成多个 gpu.launch。
OwningOpRef<ModuleOp> buildGenericTransformLibraryCutEdges(
    MLIRContext *ctx, ArrayRef<KernelSpec> kernels,
    ArrayRef<RowReductionFusionPair> fuseElementwiseIntoRowReductions,
    ArrayRef<ThreadFusionPair> fuseIntoThreadForall, bool defaultSwapBlockDims,
    int64_t tileK, int64_t blockDimX, int64_t blockDimY, int64_t threadTileM,
    int64_t threadTileN, bool enableAsyncCopy, bool asyncBypassL1,
    bool enableAsyncGroups, bool enableTensorCoreTf32,
    bool enableTensorCoreF16, bool enableRowReductionInputPromotion,
    bool enableRowReductionWarp, bool enableRowReductionVectorize,
    int64_t rowReductionVectorWidth, int64_t rowReductionThreadsX,
    bool enableRowReductionCombineVectorize, bool skipMapForallToBlocks,
    bool skipMapNestedForallToThreads) {
  OpBuilder b(ctx);
  Location loc = b.getUnknownLoc();
  auto lib = ModuleOp::create(loc);
  lib->setAttr("transform.with_named_sequence", UnitAttr::get(ctx));

  auto anyOp = transform::AnyOpType::get(ctx);
  DictionaryAttr readonlyArg =
      DictionaryAttr::get(ctx, {b.getNamedAttr("transform.readonly",
                                              UnitAttr::get(ctx))});
  llvm::SmallVector<DictionaryAttr, 1> argAttrs{readonlyArg};

  b.setInsertionPointToEnd(lib.getBody());

  auto emitPrebufferizeSequence = [&]() {
    // === Pre-bufferize（generic，cut-edge 多 kernel）===
    b.create<transform::NamedSequenceOp>(
        loc, "__welder_prebufferize", anyOp, TypeRange{},
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

        auto emitKernelProducerFusion = [&](const KernelSpec &k,
                                            Value kernelForall) {
          // Phase 13B++：深度融合（multi-hop）。
          // 按 sink->source 顺序融合，减少 FuseIntoContainingOp 的脆弱失败。
          if (!k.orderedProducerNodeIds.empty()) {
            for (int64_t nodeId : k.orderedProducerNodeIds) {
              auto prodMatch = transform::MatchOp::create(
                  b, loc, anyOp, funcMatch.getResult(),
                  /* ops=*/ArrayAttr(),
                  /* interface=*/transform::MatchInterfaceEnumAttr::get(
                      ctx, transform::MatchInterfaceEnum::LinalgOp),
                  /* op_attrs=*/DictionaryAttr::get(
                      ctx, {b.getNamedAttr("welder.node_id",
                                           b.getI64IntegerAttr(nodeId)),
                            b.getNamedAttr("welder.kernel_producer",
                                           b.getI32IntegerAttr(k.kernelId))}),
                  /* filter_result_type=*/TypeAttr(),
                  /* filter_operand_types=*/ArrayAttr());
              (void)transform::FuseIntoContainingOp::create(
                  b, loc, prodMatch.getResult(), kernelForall);
            }
            return;
          }

          auto producersMatch = transform::MatchOp::create(
              b, loc, anyOp, funcMatch.getResult(),
              /* ops=*/ArrayAttr(),
              /* interface=*/transform::MatchInterfaceEnumAttr::get(
                  ctx, transform::MatchInterfaceEnum::LinalgOp),
              /* op_attrs=*/DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.kernel_producer",
                                       b.getI32IntegerAttr(k.kernelId))}),
              /* filter_result_type=*/TypeAttr(),
              /* filter_operand_types=*/ArrayAttr());
          (void)transform::FuseIntoContainingOp::create(
              b, loc, producersMatch.getResult(), kernelForall);
        };

        auto emitKernelRowReductionLowering = [&](const KernelSpec &k,
                                                  bool swapXY,
                                                  Value kernelForall) {
          // Reduction↔归约融合代码生成 v1：
          // - 将行归约切到线程级 scf.forall（映射 threadIdx.x）。
          // - fill/combine 归约分发到线程，避免 gpu.launch 内重复执行与竞争。
          auto redsMatch = b.create<transform::MatchOp>(
              loc, kernelForall, ArrayRef<StringRef>{"linalg.generic"});
          redsMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.kernel_id",
                                       b.getI32IntegerAttr(k.kernelId)),
                        b.getNamedAttr("welder.row_reduction",
                                       UnitAttr::get(ctx))}));

          ArrayAttr redThreadMapping =
              ArrayAttr::get(ctx, {gpu::GPUThreadMappingAttr::get(
                                     ctx, gpu::MappingId::LinearDim0)});
          ArrayAttr fillThreadMapping = makeGpuThreadMapping(ctx, /*swapXY=*/false);
          ArrayAttr combineThreadMapping = ArrayAttr::get(
              ctx, {gpu::GPUThreadMappingAttr::get(
                       ctx, swapXY ? gpu::MappingId::LinearDim0
                                   : gpu::MappingId::LinearDim1)});

          int64_t rowThreadTileM = std::max<int64_t>(1, threadTileM);
          bool rowReduce1D =
              enableRowReductionWarp || enableTensorCoreTf32 || enableTensorCoreF16;
          if (rowReduce1D) {
            rowThreadTileM = 1;
            bool forceCombineOnX =
                getEnvInt64OrDefault("WELDER_MM_SM_ROW_COMBINE_ON_X", 0) != 0;
            if ((enableTensorCoreTf32 || enableTensorCoreF16)) {
              if (forceCombineOnX)
                combineThreadMapping = redThreadMapping;
            } else {
              combineThreadMapping = redThreadMapping;
            }
          }
          if ((enableTensorCoreTf32 || enableTensorCoreF16) &&
              k.rowReductionCount > 0 && blockDimY > 1) {
            rowThreadTileM =
                std::max<int64_t>(rowThreadTileM, std::max<int64_t>(1, blockDimY / 2));
          }

          auto foreachReds =
              transform::ForeachOp::create(b, loc, TypeRange{},
                                           ValueRange{redsMatch.getResult()},
                                           /* with_zip_shortest=*/false);
          Region &foreachRedsBody = foreachReds.getBody();
          Block *redBody = new Block();
          foreachRedsBody.push_back(redBody);
          redBody->addArgument(anyOp, loc);
          {
            OpBuilder rb = OpBuilder::atBlockBegin(redBody);
            Value redHandle = redBody->getArgument(0);
            int64_t redThreadsX = blockDimX;
            if (rowReductionThreadsX > 0)
              redThreadsX = std::max<int64_t>(
                  1, std::min<int64_t>(blockDimX, rowReductionThreadsX));
            if (enableRowReductionWarp) {
              redThreadsX = std::max<int64_t>(1, std::min<int64_t>(blockDimX, 32));
            } else if ((enableTensorCoreTf32 || enableTensorCoreF16) &&
                       blockDimX * blockDimY >= 1024 && blockDimX > 1) {
              redThreadsX = std::max<int64_t>(1, blockDimX / 2);
            } else if (k.rowReductionCount > 1 && blockDimX > 1) {
              redThreadsX = std::max<int64_t>(1, blockDimX / 2);
            }
            int64_t inferredParallelThreads = std::max<int64_t>(1, blockDimY);
            if (!k.tileSizes.empty() && k.tileSizes[0] > 0 && rowThreadTileM > 0) {
              int64_t rowsPerBlock = k.tileSizes[0];
              inferredParallelThreads = std::max<int64_t>(
                  1, (rowsPerBlock + rowThreadTileM - 1) / rowThreadTileM);
            }
            if (rowReduce1D && !k.tileSizes.empty() && k.tileSizes[0] > 0) {
              inferredParallelThreads =
                  std::max<int64_t>(inferredParallelThreads, k.tileSizes[0]);
            }
            int64_t maxThreadBudget =
                std::max<int64_t>(1, std::min<int64_t>(1024, blockDimX * blockDimY));
            int64_t reductionStageFactor =
                std::max<int64_t>(1, k.rowReductionCount);
            if (rowReduce1D) {
              const int64_t rowReduce1DStageFactor = std::max<int64_t>(
                  1, getEnvInt64OrDefault("WELDER_MM_SM_ROW_REDUCTION_STAGE_FACTOR_1D",
                                          2));
              reductionStageFactor *= rowReduce1DStageFactor;
            }
            if ((enableTensorCoreTf32 || enableTensorCoreF16) && enableAsyncCopy) {
              const int64_t tcAsyncStageFactor = std::max<int64_t>(
                  1, getEnvInt64OrDefault(
                         "WELDER_MM_SM_TC_ASYNC_ROW_REDUCTION_STAGE_FACTOR", 4));
              reductionStageFactor =
                  std::max<int64_t>(reductionStageFactor, tcAsyncStageFactor);
            }
            int64_t reductionParallelBudget =
                std::max<int64_t>(1, maxThreadBudget / reductionStageFactor);
            int64_t maxRedThreadsX =
                std::max<int64_t>(1, reductionParallelBudget / inferredParallelThreads);
            redThreadsX = std::min<int64_t>(redThreadsX, maxRedThreadsX);
            if ((enableTensorCoreTf32 || enableTensorCoreF16) && enableAsyncCopy) {
              const int64_t tcAsyncMaxRedThreadsX = std::max<int64_t>(
                  1, getEnvInt64OrDefault("WELDER_MM_SM_TC_ASYNC_MAX_ROW_THREADS_X",
                                          8));
              redThreadsX = std::min<int64_t>(redThreadsX, tcAsyncMaxRedThreadsX);
            }
            auto tiledRed = transform::TileReductionUsingForallOp::create(
                rb, loc, redHandle,
                /* staticNumThreads=*/ArrayRef<int64_t>{0, redThreadsX},
                /* staticTileSizes=*/ArrayRef<int64_t>{},
                /* mapping=*/redThreadMapping);

            for (Value fillHandle : tiledRed.getFillOp()) {
              (void)transform::TileUsingForallOp::create(
                  rb, loc, fillHandle, ArrayRef<int64_t>{rowThreadTileM, 1},
                  transform::TileSizesSpec(), fillThreadMapping);
            }

            auto combineTiled = transform::TileUsingForallOp::create(
                rb, loc, tiledRed.getCombiningOp(),
                ArrayRef<int64_t>{rowThreadTileM},
                transform::TileSizesSpec(), combineThreadMapping);
            if (enableRowReductionCombineVectorize) {
              int64_t vecM =
                  std::max<int64_t>(1, std::min<int64_t>(rowThreadTileM, 4));
              if (rowReductionVectorWidth > 0) {
                vecM = std::max<int64_t>(
                    1, std::min<int64_t>(rowThreadTileM, rowReductionVectorWidth));
              }
              while (vecM > 1 && (rowThreadTileM % vecM) != 0)
                vecM /= 2;
              (void)transform::VectorizeOp::create(
                  rb, loc, combineTiled.getTiledOp(),
                  /* vector_sizes=*/ValueRange{},
                  /* static_vector_sizes=*/ArrayRef<int64_t>{vecM},
                  /*vectorize_nd_extract=*/UnitAttr(),
                  /* assume_dynamic_dims_match_vec_sizes=*/UnitAttr(),
                  /* create_named_contraction=*/UnitAttr(),
                  /* scalable_sizes=*/ArrayRef<bool>{false});
            }
            rb.create<transform::YieldOp>(loc);
          }
        };

        auto emitRowReductionChainFusion = [&](const KernelSpec &k,
                                               Value kernelForall) {
          // Row-归约 chain fusion v2 (minimal, opt-in):
          // 把逐元素 producer 融到行归约消费者内，减少大中间值落地。
          for (const RowReductionFusionPair &pair :
               fuseElementwiseIntoRowReductions) {
            if (pair.kernelId != k.kernelId)
              continue;

            auto consMatch = b.create<transform::MatchOp>(
                loc, kernelForall, ArrayRef<StringRef>{"linalg.generic"});
            consMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx,
                    {b.getNamedAttr("welder.kernel_id",
                                    b.getI32IntegerAttr(k.kernelId)),
                     b.getNamedAttr("welder.node_id",
                                    b.getI64IntegerAttr(pair.consumerNodeId))}));

            auto consForall = transform::GetParentOp::create(
                b, loc, anyOp, consMatch.getResult(),
                /* isolated_from_above=*/false,
                /* allow_empty_results=*/false,
                /* op_name=*/b.getStringAttr("scf.forall"),
                /* deduplicate=*/true,
                /* nth_parent=*/pair.fuseIntoBlockForall ? 2 : 1);

            auto prodMatch = b.create<transform::MatchOp>(
                loc, kernelForall, ArrayRef<StringRef>{"linalg.generic"});
            prodMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx,
                    {b.getNamedAttr("welder.kernel_id",
                                    b.getI32IntegerAttr(k.kernelId)),
                     b.getNamedAttr("welder.node_id",
                                    b.getI64IntegerAttr(pair.producerNodeId))}));

            (void)transform::FuseIntoContainingOp::create(
                b, loc, prodMatch.getResult(), consForall.getResult());
          }
        };

        for (const KernelSpec &k : kernels) {
          if (k.opName.empty())
            continue;
          if (k.tileSizes.empty())
            continue;

          const int64_t numBlockDims = countNonZeroI64(k.tileSizes);
          if (numBlockDims <= 0)
            continue;

          auto rootMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{k.opName});
          rootMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.kernel_root",
                                       b.getI32IntegerAttr(k.kernelId))}));

          bool swapXY = defaultSwapBlockDims;
          if (k.swapXY)
            swapXY = true;
          ArrayAttr blockMapping = makeGpuBlockMappingN(ctx, numBlockDims, swapXY);
          auto tiled = transform::TileUsingForallOp::create(
              b, loc, rootMatch.getResult(), k.tileSizes,
              transform::TileSizesSpec(), blockMapping);
          (void)transform::AnnotateOp::create(
              b, loc, tiled.getForallOp(),
              /* name=*/b.getStringAttr("welder.kernel_block_forall"),
              /* param=*/Value());

          emitKernelProducerFusion(k, tiled.getForallOp());
          emitKernelRowReductionLowering(k, swapXY, tiled.getForallOp());
          emitRowReductionChainFusion(k, tiled.getForallOp());
        }

        b.create<transform::YieldOp>(loc);
        },
        /* attrs=*/ArrayRef<NamedAttribute>{},
        /* argAttrs=*/argAttrs);
  };
  emitPrebufferizeSequence();

  auto emitPostbufferizeSequence = [&]() {
    // === Post-bufferize：K-tiling + shared promotion + thread tiling + 向量化
    // + 映射到 blocks/threads（multi-kernel）===
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

        auto emitBlockForallMappings = [&]() {
          auto blockForallsMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"scf.forall"});
          blockForallsMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(ctx,
                                  {b.getNamedAttr("welder.kernel_block_forall",
                                                  UnitAttr::get(ctx))}));
          if (!skipMapForallToBlocks) {
            (void)transform::MapForallToBlocks::create(
                b, loc, anyOp, blockForallsMatch.getResult(),
                /* grid_dims=*/ArrayRef<int64_t>{},
                /* generate_gpu_launch=*/true);
          }

          auto mapBlockForallsByMapping = [&](ArrayAttr blockMapping) {
            if (!blockMapping || blockMapping.empty())
              return;
            auto forallsMatch =
                b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                             ArrayRef<StringRef>{"scf.forall"});
            forallsMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx, {b.getNamedAttr("mapping", blockMapping)}));
            if (!skipMapForallToBlocks) {
              (void)transform::MapForallToBlocks::create(
                  b, loc, anyOp, forallsMatch.getResult(),
                  /* grid_dims=*/ArrayRef<int64_t>{},
                  /* generate_gpu_launch=*/true);
            }
          };
          mapBlockForallsByMapping(makeGpuBlockMappingN(ctx, /*nDims=*/1,
                                                        /* swapXY=*/false));
          mapBlockForallsByMapping(makeGpuBlockMappingN(ctx, /*nDims=*/2,
                                                        /* swapXY=*/false));
          mapBlockForallsByMapping(makeGpuBlockMappingN(ctx, /*nDims=*/2,
                                                        /* swapXY=*/true));
          mapBlockForallsByMapping(makeGpuBlockMappingN(ctx, /*nDims=*/3,
                                                        /* swapXY=*/false));
        };

        auto emitKernelMatmulLowering = [&](const KernelSpec &k, int64_t kid,
                                            bool swapXY, bool kHasRowReduction) {
          auto mmMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"linalg.matmul"});
          mmMatch->setAttr("op_attrs",
                           DictionaryAttr::get(
                               ctx, {b.getNamedAttr("welder.kernel_id",
                                                    b.getI32IntegerAttr(kid))}));
          if (enableTensorCoreTf32 || enableTensorCoreF16) {
            auto tiledK = transform::TileUsingForOp::create(
                b, loc, mmMatch.getResult(), ArrayRef<int64_t>{0, 0, tileK});

            Value tiledForMma = tiledK.getTiledLinalgOp();
            const bool enableTcSharedPromotionNoAsync =
                getEnvInt64OrDefault("WELDER_TC_PROMOTE_SHARED_NO_ASYNC", 0) !=
                0;
            const bool enableTcSharedPromotion =
                enableAsyncCopy || enableTcSharedPromotionNoAsync;
            if (enableTcSharedPromotion) {
              auto operandsToPromote = ArrayAttr::get(
                  ctx, {b.getI64IntegerAttr(0), b.getI64IntegerAttr(1)});
              auto useFullTileBuffers = ArrayAttr::get(
                  ctx, {b.getBoolAttr(false), b.getBoolAttr(false)});
              auto promoted = transform::PromoteOp::create(
                  b, loc, anyOp, tiledForMma, operandsToPromote, useFullTileBuffers,
                  /*use_full_tiles_by_default=*/false,
                  /* use_original_subview_size=*/false,
                  /* use_alloca=*/false,
                  /* memory_space=*/b.getI64IntegerAttr(3),
                  /* mapping=*/ArrayAttr(),
                  /* alignment=*/IntegerAttr());
              tiledForMma = promoted.getTransformed();
            }

            ArrayAttr warpMapping = makeGpuWarpMapping(ctx, swapXY);
            if ((enableTensorCoreF16 || enableTensorCoreTf32) &&
                kHasRowReduction) {
              bool allowThreadMappedMma = getEnvInt64OrDefault(
                                               "WELDER_MM_SM_TC_ALLOW_THREAD_MAPPED_MMA",
                                               0) != 0;
              if (allowThreadMappedMma)
                warpMapping = makeGpuThreadMapping(ctx, swapXY);
            }
            auto tiledWarp = transform::TileUsingForallOp::create(
                b, loc, tiledForMma, ArrayRef<int64_t>{16, 8, 0},
                transform::TileSizesSpec(), warpMapping);

            (void)transform::ApplyPatternsOp::create(
                b, loc, funcMatch.getResult(), [&](OpBuilder &pb, Location loc) {
                  pb.create<transform::ApplyCanonicalizationPatternsOp>(loc);
                });
            (void)transform::RewriteMatmulAsMmaSyncOp::create(
                b, loc, tiledWarp.getTiledOp());
            return;
          }

          auto tiledK = transform::TileUsingForOp::create(
              b, loc, mmMatch.getResult(), ArrayRef<int64_t>{0, 0, tileK});
          auto operandsToPromote = ArrayAttr::get(
              ctx, {b.getI64IntegerAttr(0), b.getI64IntegerAttr(1)});
          auto useFullTileBuffers = ArrayAttr::get(
              ctx, {b.getBoolAttr(false), b.getBoolAttr(false)});
          auto promoted = transform::PromoteOp::create(
              b, loc, anyOp, tiledK.getTiledLinalgOp(), operandsToPromote,
              useFullTileBuffers,
              /*use_full_tiles_by_default=*/false,
              /* use_original_subview_size=*/false,
              /* use_alloca=*/false,
              /* memory_space=*/b.getI64IntegerAttr(3),
              /* mapping=*/ArrayAttr(),
              /* alignment=*/IntegerAttr());
          (void)transform::TileUsingForallOp::create(
              b, loc, promoted.getTransformed(),
              ArrayRef<int64_t>{std::max<int64_t>(1, threadTileM),
                                std::max<int64_t>(1, threadTileN), 0},
              transform::TileSizesSpec(), makeGpuThreadMapping(ctx, swapXY));
        };

        auto emitKernelElementwiseLowering = [&](const KernelSpec &k, int64_t kid,
                                                 bool swapXY) {
          auto gen1dMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"linalg.generic"});
          gen1dMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.kernel_id",
                                       b.getI32IntegerAttr(kid)),
                        b.getNamedAttr("welder.elementwise_1d",
                                       UnitAttr::get(ctx))}));
          ArrayAttr gen1dMapping = ArrayAttr::get(
              ctx, {gpu::GPUThreadMappingAttr::get(
                       ctx, swapXY ? gpu::MappingId::LinearDim0
                                   : gpu::MappingId::LinearDim1)});
          (void)transform::TileUsingForallOp::create(
              b, loc, gen1dMatch.getResult(),
              ArrayRef<int64_t>{std::max<int64_t>(1, threadTileM)},
              transform::TileSizesSpec(), gen1dMapping);

          auto genNdMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"linalg.generic"});
          genNdMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.kernel_id",
                                       b.getI32IntegerAttr(kid)),
                        b.getNamedAttr("welder.elementwise_nd",
                                       UnitAttr::get(ctx))}));
          int64_t elemTileM = std::max<int64_t>(1, threadTileM);
          int64_t elemTileN = std::max<int64_t>(1, threadTileN);
          if (enableTensorCoreTf32 || enableTensorCoreF16)
            elemTileM = std::max<int64_t>(elemTileM, 2);

          int64_t tileM = 1;
          int64_t tileN = 1;
          if (!k.tileSizes.empty())
            tileM = std::max<int64_t>(1, k.tileSizes[0]);
          if (k.tileSizes.size() > 1)
            tileN = std::max<int64_t>(1, k.tileSizes[1]);
          auto ceilDivI64 = [](int64_t a, int64_t b) -> int64_t {
            if (b <= 0)
              return 0;
            return (a + b - 1) / b;
          };
          int64_t blockThreads = std::max<int64_t>(1, blockDimX * blockDimY);
          int64_t reqThreads = std::max<int64_t>(
              1, ceilDivI64(tileM, elemTileM) * ceilDivI64(tileN, elemTileN));
          while (reqThreads > blockThreads) {
            if (elemTileN < tileN) {
              elemTileN = std::min<int64_t>(tileN, elemTileN * 2);
            } else if (elemTileM < tileM) {
              elemTileM = std::min<int64_t>(tileM, elemTileM * 2);
            } else {
              break;
            }
            reqThreads = std::max<int64_t>(
                1, ceilDivI64(tileM, elemTileM) * ceilDivI64(tileN, elemTileN));
          }
          (void)transform::TileUsingForallOp::create(
              b, loc, genNdMatch.getResult(),
              ArrayRef<int64_t>{elemTileM, elemTileN},
              transform::TileSizesSpec(), makeGpuThreadMapping(ctx, swapXY));

          for (const ThreadFusionPair &pair : fuseIntoThreadForall) {
            if (pair.kernelId != kid)
              continue;

            auto consMatch = b.create<transform::MatchOp>(
                loc, funcMatch.getResult(), ArrayRef<StringRef>{"linalg.generic"});
            consMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx,
                    {b.getNamedAttr("welder.kernel_id",
                                    b.getI32IntegerAttr(kid)),
                     b.getNamedAttr("welder.node_id",
                                    b.getI64IntegerAttr(pair.consumerNodeId))}));
            auto foreachConsumer =
                transform::ForeachOp::create(b, loc, TypeRange{},
                                             ValueRange{consMatch.getResult()},
                                             /* with_zip_shortest=*/false);
            Region &foreachConsumerBody = foreachConsumer.getBody();
            Block *consumerBody = new Block();
            foreachConsumerBody.push_back(consumerBody);
            consumerBody->addArgument(anyOp, loc);
            {
              OpBuilder fb = OpBuilder::atBlockBegin(consumerBody);
              Value consumerHandle = consumerBody->getArgument(0);
              auto consForall = transform::GetParentOp::create(
                  fb, loc, anyOp, consumerHandle,
                  /* isolated_from_above=*/false,
                  /* allow_empty_results=*/false,
                  /* op_name=*/fb.getStringAttr("scf.forall"),
                  /* deduplicate=*/true,
                  /* nth_parent=*/1);

              Value producerHandle;
              if (pair.consumerOperand >= 0) {
                auto producer = transform::GetProducerOfOperand::create(
                    fb, loc, anyOp, consumerHandle,
                    static_cast<uint64_t>(pair.consumerOperand));
                producerHandle = producer.getResult();
              } else {
                auto prodMatch = fb.create<transform::MatchOp>(
                    loc, funcMatch.getResult(),
                    ArrayRef<StringRef>{"linalg.generic"});
                prodMatch->setAttr(
                    "op_attrs",
                    DictionaryAttr::get(
                        ctx,
                        {fb.getNamedAttr("welder.kernel_id",
                                         fb.getI32IntegerAttr(kid)),
                         fb.getNamedAttr("welder.node_id",
                                         fb.getI64IntegerAttr(pair.producerNodeId))}));
                producerHandle = prodMatch.getResult();
              }

              (void)transform::FuseIntoContainingOp::create(
                  fb, loc, producerHandle, consForall.getResult());
              fb.create<transform::YieldOp>(loc);
            }
          }

          if (enableRowReductionVectorize) {
            auto vecMatch =
                b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                             ArrayRef<StringRef>{"linalg.generic"});
            vecMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx, {b.getNamedAttr("welder.kernel_id",
                                         b.getI32IntegerAttr(kid)),
                          b.getNamedAttr("welder.vectorizable",
                                         UnitAttr::get(ctx))}));
            int64_t vecN =
                std::max<int64_t>(1, std::min<int64_t>(elemTileN, 4));
            if (rowReductionVectorWidth > 0) {
              vecN = std::max<int64_t>(
                  1, std::min<int64_t>(elemTileN, rowReductionVectorWidth));
            }
            while (vecN > 1 && (elemTileN % vecN) != 0)
              vecN /= 2;
            auto foreachVec =
                transform::ForeachOp::create(b, loc, TypeRange{},
                                             ValueRange{vecMatch.getResult()},
                                             /* with_zip_shortest=*/false);
            Region &foreachVecBody = foreachVec.getBody();
            Block *vecBody = new Block();
            foreachVecBody.push_back(vecBody);
            vecBody->addArgument(anyOp, loc);
            OpBuilder vb = OpBuilder::atBlockBegin(vecBody);
            Value genHandle = vecBody->getArgument(0);
            (void)transform::VectorizeOp::create(
                vb, loc, genHandle, /*vector_sizes=*/ValueRange{},
                /* static_vector_sizes=*/ArrayRef<int64_t>{1, vecN},
                /*vectorize_nd_extract=*/UnitAttr(),
                /* assume_dynamic_dims_match_vec_sizes=*/UnitAttr(),
                /* create_named_contraction=*/UnitAttr(),
                /* scalable_sizes=*/ArrayRef<bool>{false, false});
            vb.create<transform::YieldOp>(loc);
          }

          if (enableRowReductionInputPromotion) {
            // 实际 promotion 作为 C++ IR 变换在 postbufferize 之后执行
            //（见 `promoteRowReductionInputsToWorkgroup`）。
          }
        };

        auto emitCopyDistribution = [&]() {
          auto copies =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"linalg.copy"});
          auto ceilDivI64 = [](int64_t a, int64_t b) -> int64_t {
            if (b <= 0)
              return 0;
            return (a + b - 1) / b;
          };
          int64_t blockThreads = std::max<int64_t>(1, blockDimX * blockDimY);
          int64_t effTileM = std::max<int64_t>(1, threadTileM) *
                             (defaultSwapBlockDims ? blockDimX : blockDimY);
          int64_t effTileN = std::max<int64_t>(1, threadTileN) *
                             (defaultSwapBlockDims ? blockDimY : blockDimX);

          int64_t prefCopyTileM =
              std::max<int64_t>(1, std::min<int64_t>(4, threadTileM));
          int64_t prefCopyTileN =
              std::max<int64_t>(1, std::min<int64_t>(4, threadTileN));

          int64_t copyTileM = prefCopyTileM;
          int64_t rowTiles = std::max<int64_t>(ceilDivI64(effTileM, copyTileM),
                                               ceilDivI64(tileK, copyTileM));
          rowTiles = std::max<int64_t>(1, rowTiles);
          int64_t copyParallelFactor = 1;
          if ((enableTensorCoreTf32 || enableTensorCoreF16) && enableAsyncCopy) {
            copyParallelFactor = std::max<int64_t>(
                1, getEnvInt64OrDefault("WELDER_MM_SM_TC_ASYNC_COPY_STAGE_FACTOR",
                                        2));
          }
          int64_t maxColTiles =
              std::max<int64_t>(1, blockThreads / (rowTiles * copyParallelFactor));
          int64_t minCopyTileN = std::max<int64_t>(
              ceilDivI64(tileK, maxColTiles), ceilDivI64(effTileN, maxColTiles));
          int64_t copyTileN = std::max<int64_t>(prefCopyTileN, minCopyTileN);
          (void)transform::TileUsingForallOp::create(
              b, loc, copies.getResult(), ArrayRef<int64_t>{copyTileM, copyTileN},
              transform::TileSizesSpec(),
              makeGpuThreadMapping(ctx, defaultSwapBlockDims));
        };

        auto emitLaunchThreadMapping = [&]() {
          auto launches = b.create<transform::MatchOp>(
              loc, funcMatch.getResult(), ArrayRef<StringRef>{"gpu.launch"});
          auto foreachLaunches =
              transform::ForeachOp::create(b, loc, TypeRange{},
                                           ValueRange{launches.getResult()},
                                           /* with_zip_shortest=*/false);
          Region &foreachLaunchesBody = foreachLaunches.getBody();
          Block *launchBody = new Block();
          foreachLaunchesBody.push_back(launchBody);
          launchBody->addArgument(anyOp, loc);

          OpBuilder lb = OpBuilder::atBlockBegin(launchBody);
          Value launchHandle = launchBody->getArgument(0);
          if (!skipMapNestedForallToThreads) {
            auto threadsMapped = transform::MapNestedForallToThreads::create(
                lb, loc, anyOp, launchHandle,
                /* block_dims=*/ArrayRef<int64_t>{blockDimX, blockDimY, 1},
                /* sync_after_distribute=*/true,
                /* warp_size=*/32);
            launchHandle = threadsMapped.getResult();
          }

          if (enableAsyncCopy && enableAsyncGroups) {
            auto asyncGroups = transform::CreateAsyncGroupsOp::create(
                lb, loc, anyOp, launchHandle, /*bypass_l1=*/asyncBypassL1);
            launchHandle = asyncGroups.getResult();
          }
          lb.create<transform::YieldOp>(loc);
        };

        emitBlockForallMappings();
        for (const KernelSpec &k : kernels) {
          int64_t kid = k.kernelId;
          bool swapXY = defaultSwapBlockDims;
          if (k.swapXY)
            swapXY = true;
          bool kHasRowReduction = k.rowReductionCount > 0;
          emitKernelMatmulLowering(k, kid, swapXY, kHasRowReduction);
          emitKernelElementwiseLowering(k, kid, swapXY);
        }
        emitCopyDistribution();
        emitLaunchThreadMapping();

        b.create<transform::YieldOp>(loc);
        },
        /* attrs=*/ArrayRef<NamedAttribute>{},
        /* argAttrs=*/argAttrs);
  };
  emitPostbufferizeSequence();

  return lib;
}

} // namespace welder::compiler

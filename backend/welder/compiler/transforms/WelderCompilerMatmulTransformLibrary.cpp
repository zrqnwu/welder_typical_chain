#include "WelderCompilerMatmulTransformLibrary.h"

#include "WelderCompilerPassTraceAndEnv.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
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

} // namespace

namespace welder::compiler {

OwningOpRef<ModuleOp>
buildTransformLibrary(MLIRContext *ctx, int64_t tileM, int64_t tileN,
                      int64_t tileK, int64_t blockDimX, int64_t blockDimY,
                      int64_t mmaM, int64_t mmaN, int64_t elementBytes,
                      int64_t threadTileM, int64_t threadTileN,
                      bool hasConsumerChain, bool hasRowReduction,
                      bool enableAsyncCopy, bool enableSoftwarePipelining,
                      int64_t pipelineDepth, bool pipelinePeelEpilogue,
                      bool asyncBypassL1, bool enableTensorCoreTf32,
                      bool enableTensorCoreF16, bool swapBlockDims,
                      bool skipMapForallToBlocks,
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
  bool disableFusedRowReductionTransform =
      (getEnvInt64OrDefault("WELDER_DISABLE_FUSED_ROWRED", 1) != 0);

  b.setInsertionPointToEnd(lib.getBody());

  // === Pre-bufferize：以 consumer 驱动的 L1 tile + 融合 ===
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

        // 匹配唯一 consumer 算子（由 C++ 侧 `welder.matmul_consumer` 锚点指定）。
        auto consumerMatch = transform::MatchOp::create(
            b, loc, anyOp, funcMatch.getResult(),
            /* ops=*/ArrayAttr(),
            /* interface=*/transform::MatchInterfaceEnumAttr::get(
                ctx, transform::MatchInterfaceEnum::LinalgOp),
            /* op_attrs=*/DictionaryAttr::get(
                ctx, {b.getNamedAttr("welder.matmul_consumer", UnitAttr::get(ctx))}),
            /* filter_result_type=*/TypeAttr(),
            /* filter_operand_types=*/ArrayAttr());

        // block 级切分：对 (M,N) 做 tile 并映射到 gpu.block。
        // 若 consumer 是纯 matmul（无 epilogue），此处 K 维不切
        //（tileK 在 postbufferize 阶段处理）。
        ArrayAttr blockMapping = makeGpuBlockMapping(ctx, swapBlockDims);
        if (hasConsumerChain) {
          auto tiledConsumer = transform::TileUsingForallOp::create(
              b, loc, consumerMatch.getResult(), ArrayRef<int64_t>{tileM, tileN},
              transform::TileSizesSpec(), blockMapping);

          // 将完整 producer 链（多跳）融合进 consumer 的 forall。
          auto producersMatch = transform::MatchOp::create(
              b, loc, anyOp, funcMatch.getResult(),
              /* ops=*/ArrayAttr(),
              /* interface=*/transform::MatchInterfaceEnumAttr::get(
                  ctx, transform::MatchInterfaceEnum::LinalgOp),
              /* op_attrs=*/DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.matmul_producer",
                                       UnitAttr::get(ctx))}),
              /* filter_result_type=*/TypeAttr(),
              /* filter_operand_types=*/ArrayAttr());

          (void)transform::FuseIntoContainingOp::create(
              b, loc, producersMatch.getResult(), tiledConsumer.getForallOp());
        } else {
          // 纯 matmul 场景：直接将 matmul 切分到 block。
          (void)transform::TileUsingForallOp::create(
              b, loc, consumerMatch.getResult(),
              ArrayRef<int64_t>{tileM, tileN, 0}, transform::TileSizesSpec(),
              blockMapping);
        }

        b.create<transform::YieldOp>(loc);
      },
      /* attrs=*/ArrayRef<NamedAttribute>{},
      /* argAttrs=*/argAttrs);

  // === Post-bufferize：K 切分 + shared 提升 + 线程切分 + 向量化
  // + 映射到 blocks/threads ===
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

        // 匹配 bufferize 后的 matmul
        auto mmMatch =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{"linalg.matmul"});

        if (enableTensorCoreTf32 || enableTensorCoreF16) {
          const int64_t effMmaM = std::max<int64_t>(1, mmaM > 0 ? mmaM : 16);
          const int64_t effMmaN = std::max<int64_t>(1, mmaN > 0 ? mmaN : 8);
          Value mmTarget = mmMatch.getResult();
          if (!hasConsumerChain) {
            ArrayAttr blockMapping = makeGpuBlockMapping(ctx, swapBlockDims);
            auto tiledBlock = transform::TileUsingForallOp::create(
                b, loc, mmTarget, ArrayRef<int64_t>{tileM, tileN, 0},
                transform::TileSizesSpec(), blockMapping);
            mmTarget = tiledBlock.getTiledOp();
          }

          // 在 warp 级切到 MMA 指令形状（如 m16n8k16），
          // 让 rewrite_matmul_as_mma_sync 能与每个内层 matmul 1:1 对齐。
          ArrayAttr warpMapping = makeGpuWarpMapping(ctx, swapBlockDims);
          if ((enableTensorCoreF16 || enableTensorCoreTf32) && hasConsumerChain) {
            // 重要：rewrite_matmul_as_mma_sync 期望 warp 级分发。
            // 线程映射的 MMA 切分虽可编译，但在融合 matmul->行归约链中
            // 可能产生错误 lane/tile 分配（出现非有限输出）。
            // 因此默认保持 warp 映射，仅允许本地调试时显式启用线程映射。
            bool allowThreadMappedMma =
                getEnvInt64OrDefault("WELDER_MM_SM_TC_ALLOW_THREAD_MAPPED_MMA", 0) !=
                0;
            if (allowThreadMappedMma)
              warpMapping = makeGpuThreadMapping(ctx, swapBlockDims);
          }
          auto tiledWarp = transform::TileUsingForallOp::create(
              b, loc, mmTarget,
              ArrayRef<int64_t>{effMmaM, effMmaN, 0},
              transform::TileSizesSpec(), warpMapping);

          auto tiledK = transform::TileUsingForOp::create(
              b, loc, tiledWarp.getTiledOp(), ArrayRef<int64_t>{0, 0, tileK});

          Value mmaTarget = tiledK.getTiledLinalgOp();

          // 论文/Welder 对齐（TCPolicy）：先把 A/B tile 经过 shared 内存暂存，
          // 使 async-copy（cp.async）与流水化能够生效。
          //
          // 保持 promotion 可选，避免依赖旧版 MLIR 对 shared-memory 的支持差异。
          if (enableAsyncCopy || enableSoftwarePipelining) {
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
            mmaTarget = promoted.getTransformed();

            // 对生成的 linalg.copy 做分发 + 向量化，
            // 以便 nvgpu.create_async_groups 将其重写为 device_async_copy。
            auto copies =
                b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                             ArrayRef<StringRef>{"linalg.copy"});

            // 最内层维优先 16-byte 向量：
            // - f16 -> 8 元素
            // - f32 -> 4 元素
            int64_t targetElems = 4;
            if (elementBytes > 0)
              targetElems = std::max<int64_t>(1, 16 / elementBytes);
            (void)transform::TileUsingForallOp::create(
                b, loc, copies.getResult(),
                ArrayRef<int64_t>{1, targetElems},
                transform::TileSizesSpec(),
                makeGpuThreadMapping(ctx, swapBlockDims));

            auto copiesAfter =
                b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                             ArrayRef<StringRef>{"linalg.copy"});
            auto foreach =
                transform::ForeachOp::create(b, loc, TypeRange{},
                                             ValueRange{copiesAfter.getResult()},
                                             /* with_zip_shortest=*/false);
            Region &foreachBody = foreach.getBody();
            Block *body = new Block();
            foreachBody.push_back(body);
            body->addArgument(anyOp, loc);
            OpBuilder nb = OpBuilder::atBlockBegin(body);
            Value copyHandle = body->getArgument(0);

            (void)transform::VectorizeOp::create(
                nb, loc, copyHandle, /*vector_sizes=*/ValueRange{},
                /* static_vector_sizes=*/ArrayRef<int64_t>{1, targetElems},
                /*vectorize_nd_extract=*/UnitAttr(),
                /* assume_dynamic_dims_match_vec_sizes=*/UnitAttr(),
                /* create_named_contraction=*/UnitAttr(),
                /* scalable_sizes=*/ArrayRef<bool>{false, false});
            nb.create<transform::YieldOp>(loc);

            (void)transform::ApplyPatternsOp::create(
                b, loc, funcMatch.getResult(),
                [&](OpBuilder &pb, Location loc) {
                  pb.create<transform::ApplyTransferToScfPatternsOp>(
                      loc, /*max_transfer_rank=*/1, /*full_unroll=*/true);
                  pb.create<transform::ApplyCanonicalizationPatternsOp>(loc);
                });
          }

          (void)transform::RewriteMatmulAsMmaSyncOp::create(b, loc, mmaTarget);

          if (!disableFusedRowReductionTransform) {
            // 对 Matmul->Softmax 融合 kernel 中的行归约，将归约映射到线程
            //（二维映射）以避免映射不匹配。
            auto rowRedMatch =
                b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                             ArrayRef<StringRef>{"linalg.generic"});
            rowRedMatch->setAttr(
                "op_attrs",
                DictionaryAttr::get(
                    ctx, {b.getNamedAttr("welder.row_reduction",
                                         UnitAttr::get(ctx))}));
            ArrayAttr redThreadMapping =
                makeGpuThreadMapping(ctx, /*swapXY=*/false);
            int64_t redThreadsX = std::max<int64_t>(1, blockDimX);
            auto foreachReds =
                transform::ForeachOp::create(
                    b, loc, TypeRange{}, ValueRange{rowRedMatch.getResult()},
                    /* with_zip_shortest=*/false);
            Region &foreachRedsBody = foreachReds.getBody();
            Block *redBody = new Block();
            foreachRedsBody.push_back(redBody);
            redBody->addArgument(anyOp, loc);
            {
              OpBuilder rb = OpBuilder::atBlockBegin(redBody);
              Value redHandle = redBody->getArgument(0);
              auto tiledRed = transform::TileReductionUsingForallOp::create(
                  rb, loc, redHandle,
                  /* staticNumThreads=*/ArrayRef<int64_t>{0, redThreadsX},
                  /* staticTileSizes=*/ArrayRef<int64_t>{},
                  /* mapping=*/redThreadMapping);

              for (Value fillHandle : tiledRed.getFillOp()) {
                (void)transform::TileUsingForallOp::create(
                    rb, loc, fillHandle, ArrayRef<int64_t>{1, 1},
                    transform::TileSizesSpec(), redThreadMapping);
              }

              (void)transform::TileUsingForallOp::create(
                  rb, loc, tiledRed.getCombiningOp(),
                  ArrayRef<int64_t>{1, 1}, transform::TileSizesSpec(),
                  redThreadMapping);

              rb.create<transform::YieldOp>(loc);
            }
          }

          // 将 epilogue（逐元素 linalg.generic）分发到线程，
          // 避免 gpu.launch 内重复执行和写竞争。
          auto genMatch =
              b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                           ArrayRef<StringRef>{"linalg.generic"});
          genMatch->setAttr(
              "op_attrs",
              DictionaryAttr::get(
                  ctx, {b.getNamedAttr("welder.elementwise_nd",
                                       UnitAttr::get(ctx))}));
          int64_t epilogThreadTileM = std::max<int64_t>(1, threadTileM);
          int64_t epilogThreadTileN = std::max<int64_t>(1, threadTileN);
          ArrayAttr epilogMapping = makeGpuThreadMapping(ctx, swapBlockDims);
          if (hasConsumerChain && hasRowReduction) {
            // 当融合行归约时，降低每线程 epilogue 工作量。
            if (blockDimX > 0)
              epilogThreadTileN =
                  std::max<int64_t>(1, tileN / std::max<int64_t>(1, blockDimX));
            epilogThreadTileM = 1;
          } else if (hasConsumerChain) {
            // 保持 epilogue 线程映射与 block 维度一致，避免
            // map_nested_forall_to_threads 申请超过 launch 提供的线程数。
            if (blockDimY > 0)
              epilogThreadTileM =
                  std::max<int64_t>(1, tileM / std::max<int64_t>(1, blockDimY));
            if (blockDimX > 0)
              epilogThreadTileN =
                  std::max<int64_t>(1, tileN / std::max<int64_t>(1, blockDimX));
          }
          (void)transform::TileUsingForallOp::create(
              b, loc, genMatch.getResult(),
              ArrayRef<int64_t>{epilogThreadTileM, epilogThreadTileN},
              transform::TileSizesSpec(), epilogMapping);

          // 将顶层 forall 映射到 blocks -> gpu.launch。
          if (!skipMapForallToBlocks) {
            auto launch = transform::MapForallToBlocks::create(
                b, loc, anyOp, funcMatch.getResult(),
                /* grid_dims=*/ArrayRef<int64_t>{},
                /* generate_gpu_launch=*/true);
            if (!skipMapNestedForallToThreads) {
              auto threadsMapped = transform::MapNestedForallToThreads::create(
                  b, loc, anyOp, launch.getResult(),
                  /* block_dims=*/ArrayRef<int64_t>{blockDimX, blockDimY, 1},
                  /* sync_after_distribute=*/true,
                  /* warp_size=*/32);

              if (enableAsyncCopy) {
                (void)transform::CreateAsyncGroupsOp::create(
                    b, loc, anyOp, threadsMapped.getResult(),
                    /* bypass_l1=*/asyncBypassL1);
              }
            }
          }

	          b.create<transform::YieldOp>(loc);
	          return;
	        }

        // K 维切分
        auto tiledK = transform::TileUsingForOp::create(
            b, loc, mmMatch.getResult(), ArrayRef<int64_t>{0, 0, tileK});

        // 将 A/B tile 提升到 shared（memory_space=3），以兼容
        // 以兼容 convert-vector-to-llvm。
        auto operandsToPromote = ArrayAttr::get(
            ctx, {b.getI64IntegerAttr(0), b.getI64IntegerAttr(1)});
        auto useFullTileBuffers =
            ArrayAttr::get(ctx, {b.getBoolAttr(false), b.getBoolAttr(false)});
        auto promoted = transform::PromoteOp::create(
            b, loc, anyOp, tiledK.getTiledLinalgOp(), operandsToPromote,
            useFullTileBuffers,
            /*use_full_tiles_by_default=*/false,
            /* use_original_subview_size=*/false,
            /* use_alloca=*/false,
            /* memory_space=*/b.getI64IntegerAttr(3),
            /* mapping=*/ArrayAttr(),
            /* alignment=*/IntegerAttr());

        // copy 的线程切分：
        // 分发保持保守（<=4x4），以兼容当前 transform 向量化 +
        // transfer_to_scf 模式。计算阶段 threadTileM/N 可继续更大。
        const int64_t copyTileM =
            std::max<int64_t>(1, std::min<int64_t>(4, threadTileM));
        const int64_t copyTileN =
            std::max<int64_t>(1, std::min<int64_t>(4, threadTileN));
        auto copies =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{"linalg.copy"});
        (void)transform::TileUsingForallOp::create(
            b, loc, copies.getResult(),
            ArrayRef<int64_t>{copyTileM, copyTileN},
            transform::TileSizesSpec(), makeGpuThreadMapping(ctx, swapBlockDims));

        // 对 copy 做向量化（掩码向量化，vector_sizes [4,4]）。
        // 随后用 vector.transfer_to_scf 把 transfer 下沉为 rank-1，
        // 便于 nvgpu.create_async_groups 识别连续 rank-1 传输。
        auto copiesAfter =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{"linalg.copy"});
        auto foreach =
            transform::ForeachOp::create(b, loc, TypeRange{},
                                         ValueRange{copiesAfter.getResult()},
                                         /* with_zip_shortest=*/false);
        Region &foreachBody = foreach.getBody();
        Block *body = new Block();
        foreachBody.push_back(body);
        body->addArgument(anyOp, loc);
        OpBuilder nb = OpBuilder::atBlockBegin(body);
        Value copyHandle = body->getArgument(0);
        // 论文/Welder 对齐（向量化规划，按当前 MLIR transform 能力做简化）：
        // 尽量在最内层维使用 16-byte 向量（f16->8 元素，f32->4 元素），
        // 以提升 NVGPU async copy lowering（cp.async）匹配稳定性。
        int64_t fallbackM = 4;
        int64_t fallbackN = 4;
        int64_t vecM = fallbackM;
        int64_t vecN = fallbackN;

        int64_t targetElems = 4;
        if (elementBytes > 0) {
          targetElems = std::max<int64_t>(1, 16 / elementBytes);
        }

        if (threadTileM > 0)
          vecM = std::max<int64_t>(1, std::min<int64_t>(vecM, threadTileM));
        if (threadTileN > 0) {
          vecN = std::max<int64_t>(1, std::min<int64_t>(targetElems, threadTileN));
          // 为稳健性起见，尽量让 vecN 维持近似 2 的幂。
          while (vecN > 1 && (threadTileN % vecN) != 0)
            vecN /= 2;
        } else {
          vecN = std::max<int64_t>(1, std::min<int64_t>(targetElems, vecN));
        }

        (void)transform::VectorizeOp::create(nb, loc, copyHandle,
                                             /* vector_sizes=*/ValueRange{},
                                             /* static_vector_sizes=*/ArrayRef<int64_t>{vecM, vecN},
                                             /*vectorize_nd_extract=*/UnitAttr(),
                                             /* assume_dynamic_dims_match_vec_sizes=*/UnitAttr(),
                                             /* create_named_contraction=*/UnitAttr(),
                                             /* scalable_sizes=*/ArrayRef<bool>{false, false});
        nb.create<transform::YieldOp>(loc);

        // 将 N-D `vector.transfer_{read,write}` 拆成 rank-1 transfer，
        // 便于 async-copy 改写匹配（当前仅支持 rank-1 向量）。
        (void)transform::ApplyPatternsOp::create(
            b, loc, funcMatch.getResult(),
            [&](OpBuilder &pb, Location loc) {
              pb.create<transform::ApplyTransferToScfPatternsOp>(
                  loc, /*max_transfer_rank=*/1, /*full_unroll=*/true);
              pb.create<transform::ApplyCanonicalizationPatternsOp>(loc);
            });

        // 对 matmul + relu 做线程级切分。
        (void)transform::TileUsingForallOp::create(
            b, loc, promoted.getTransformed(),
            ArrayRef<int64_t>{std::max<int64_t>(1, threadTileM),
                              std::max<int64_t>(1, threadTileN), 0},
            transform::TileSizesSpec(), makeGpuThreadMapping(ctx, swapBlockDims));

        auto relu =
            b.create<transform::MatchOp>(loc, funcMatch.getResult(),
                                         ArrayRef<StringRef>{"linalg.generic"});
        (void)transform::TileUsingForallOp::create(
            b, loc, relu.getResult(),
            ArrayRef<int64_t>{std::max<int64_t>(1, threadTileM),
                              std::max<int64_t>(1, threadTileN)},
            transform::TileSizesSpec(), makeGpuThreadMapping(ctx, swapBlockDims));

        // 将顶层 forall 映射到 blocks -> gpu.launch。
        Value launchHandle;
        if (!skipMapForallToBlocks) {
          auto launch = transform::MapForallToBlocks::create(
              b, loc, anyOp, funcMatch.getResult(),
              /* grid_dims=*/ArrayRef<int64_t>{},
              /* generate_gpu_launch=*/true);
          launchHandle = launch.getResult();
        }

        // 将嵌套 forall 映射到线程并设置 block 维度。
        if (launchHandle && !skipMapNestedForallToThreads) {
          auto threadsMapped = transform::MapNestedForallToThreads::create(
              b, loc, anyOp, launchHandle,
              /* block_dims=*/ArrayRef<int64_t>{blockDimX, blockDimY, 1},
              /* sync_after_distribute=*/true,
              /* warp_size=*/32);
          launchHandle = threadsMapped.getResult();
        }

        // === Phase 14 (论文对齐): async copy + software pipelining ===
        // 将 transform 约束在生成的 gpu.launch 内，避免误处理 host 侧循环
        //（否则 pipeliner 会对不含 shared-memory copy 的循环给出诊断）。
        if (enableAsyncCopy && launchHandle) {
          auto asyncGroups = transform::CreateAsyncGroupsOp::create(
              b, loc, anyOp, launchHandle, /*bypass_l1=*/asyncBypassL1);
          launchHandle = asyncGroups.getResult();
        }

        b.create<transform::YieldOp>(loc);
      },
      /* attrs=*/ArrayRef<NamedAttribute>{},
      /* argAttrs=*/argAttrs);

  return lib;
}

} // namespace welder::compiler

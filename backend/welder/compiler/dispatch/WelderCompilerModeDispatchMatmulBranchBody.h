#pragma once

// 路径三实现主体（由 WelderCompilerModeDispatchMatmulBranch.cpp 包装调用）。
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.solve_matmul")
                  : welder::Tracer::Span();

    welder::SolveResult solveRes;
    {
      [[maybe_unused]] auto span2 =
          tracerPtr ? tracerPtr->span("compiler.solve_matmul.call")
                    : welder::Tracer::Span();
      solveRes = welder::solve(*module, solveOpts);
    }
    if (solveRes.problem.m < 0 || solveRes.sortedCandidates.empty()) {
      llvm::errs()
          << "error: cannot solve: no static-shape linalg.matmul or no candidates\n";
      llvm::errs()
          << "hint: try --enable-generic-problem for non-matmul ops.\n";
      return 2;
    }

    // 为 matmul 流水线的 prebufferize tile-and-fuse 过程打 consumer 锚点
    //（含 producer 链），避免 epilogue 含多个逐元素算子时
    //（如 BiasAdd + ReLU）出现 transform 匹配歧义。
    markMatmulFusionAnchors(*module);

	    const welder::Candidate &best = solveRes.sortedCandidates.front();
	    int64_t tileM = best.tileM;
	    int64_t tileN = best.tileN;
	    int64_t tileK = best.tileK;

	    int64_t chosenThreadTileM = threadTileM;
	    int64_t chosenThreadTileN = threadTileN;
	    if (enableRegisterLevelSchedule && best.threadTileM > 0 &&
	        best.threadTileN > 0) {
	      chosenThreadTileM = best.threadTileM;
	      chosenThreadTileN = best.threadTileN;
	    }

	    int64_t blockDimX = 0;
	    int64_t blockDimY = 0;
		    int64_t chosenMmaM = 0;
		    int64_t chosenMmaN = 0;
		    if (enableTensorCoreTf32 || enableTensorCoreF16) {
		      // TensorCore 路径（论文/Welder 对齐风格）：
		      // - 优先使用 solver 给出的 MMA 形状；
		      // - 否则回退到接近 TCPolicy 的启发式：
		      //   * cutlass 风格：m16n8k16
		      //   * 非 cutlass：k16 下的 (16,16)/(32,8)/(8,32)
		      // - 支持 `--force-tile-m/n/k` 覆盖（用于 e2e/性能测量）；
		      // - 使用二维线程布局保证 warp/thread 映射：
		      //   * warp 按 (M,N)->(warp<y>,warp<x>) 排布
		      //   * thread 按 (M,N)->(thread<y>,thread<x>) 排布
		      int64_t mmaK = enableTensorCoreF16 ? 16 : 4;
		      int64_t mmaM = 16;
		      int64_t mmaN = 8;

		      auto pickMmaShape = [&](int64_t m, int64_t n, int64_t k) -> bool {
		        if (m <= 0 || n <= 0 || k <= 0)
		          return false;
		        if (enableTensorCoreF16) {
		          // 注意：当前 MLIR 版本的 `rewrite_matmul_as_mma_sync` 对
		          // cutlass 风格 m16n8k16 形状支持最稳定；这里固定该形状以保证
		          // 正确性和论文等价功能。
		          mmaM = 16;
		          mmaN = 8;
		          mmaK = 16;
		          if ((m % mmaM) != 0 || (n % mmaN) != 0 || (k % mmaK) != 0)
		            return false;
		        } else {
		          // TF32 路径保持最小配置。
		          mmaM = 16;
		          mmaN = 8;
		          mmaK = 4;
		        }
		        if (m % mmaM != 0 || n % mmaN != 0 || k % mmaK != 0)
		          return false;
		        int64_t warps = (m / mmaM) * (n / mmaN);
		        if (warps <= 0)
		          return false;
		        if (warps * 32 > 1024)
		          return false;
		        chosenMmaM = mmaM;
		        chosenMmaN = mmaN;
		        return true;
		      };

		      if (forceTileM > 0 && forceTileN > 0 && forceTileK > 0 &&
		          pickMmaShape(forceTileM, forceTileN, forceTileK)) {
		        tileM = forceTileM;
		        tileN = forceTileN;
		        tileK = forceTileK;
		      } else if (pickMmaShape(tileM, tileN, tileK)) {
			        // 保持 solver 的选择。
		      } else if (pickMmaShape(16, 8, mmaK)) {
		        tileM = 16;
		        tileN = 8;
		        tileK = mmaK;
		      } else {
		        llvm::errs()
		            << "error: invalid TensorCore tile/MMA shape, got M=" << tileM
		            << " N=" << tileN << " K=" << tileK << "\n";
		        return 2;
		      }

		      int64_t warps = (tileM / std::max<int64_t>(1, chosenMmaM)) *
		                      (tileN / std::max<int64_t>(1, chosenMmaN));
		      // 为逐元素 epilogue/copy 推导逐线程 tile，满足：
		      // 即 `(tileM/threadTileM) * (tileN/threadTileN) == warps * 32`。
		      //
		      // 对默认 TF32/F16 MMA 形状（m16n8），会得到 (2,2)，
		      // 即每线程 4 个输出元素。
		      chosenThreadTileM =
		          std::max<int64_t>(1, std::max<int64_t>(1, chosenMmaM) / 8);
		      chosenThreadTileN =
		          std::max<int64_t>(1, std::max<int64_t>(1, chosenMmaN) / 4);

		      if (tileM % chosenThreadTileM != 0 || tileN % chosenThreadTileN != 0) {
		        llvm::errs() << "error: TensorCore requires TILE_M/TILE_N divisible "
		                        "by derived thread tile, got TILE_M="
		                     << tileM << " TILE_N=" << tileN
		                     << " threadTileM=" << chosenThreadTileM
		                     << " threadTileN=" << chosenThreadTileN << "\n";
		        return 2;
		      }

			      auto blockDims = welder::compiler::computeBlockDimsExact(
			          tileM, tileN, chosenThreadTileM, chosenThreadTileN,
			          swapBlockDims);
			      blockDimX = blockDims.x;
			      blockDimY = blockDims.y;

		      if (blockDimX * blockDimY > 1024) {
		        llvm::errs() << "error: TensorCore block threads exceed 1024: ("
		                     << blockDimX << "x" << blockDimY << ")\n";
		        return 2;
		      }

		      // 一致性检查：二维线程布局必须与 warp 数匹配。
		      if (blockDimX * blockDimY != warps * 32) {
		        llvm::errs() << "error: internal: TensorCore thread layout mismatch "
		                        "(threads="
		                     << (blockDimX * blockDimY) << " expected="
		                     << (warps * 32) << ")\n";
		        return 2;
		      }
		    } else {
	      if (chosenThreadTileM <= 0 || chosenThreadTileN <= 0) {
	        llvm::errs() << "error: thread tile must be > 0\n";
	        return 2;
	      }
	      if (tileM % chosenThreadTileM != 0 || tileN % chosenThreadTileN != 0) {
	        llvm::errs()
	            << "error: TILE_M/TILE_N must be divisible by thread tile "
	               "(need TILE_M%threadTileM==0 and TILE_N%threadTileN==0)\n";
	        return 2;
	      }
			      auto blockDims = welder::compiler::computeBlockDimsExact(
			          tileM, tileN, chosenThreadTileM, chosenThreadTileN,
			          swapBlockDims);
			      blockDimX = blockDims.x;
			      blockDimY = blockDims.y;
		      if (blockDimX * blockDimY > 1024) {
		        llvm::errs() << "error: block threads exceed 1024: (" << blockDimX
		                     << "x" << blockDimY << ")\n";
		        return 2;
		      }
		    }

	    // 从第一个 matmul 输入推断元素字节宽度。
	    int64_t elementBytes = 4;
	    module->walk([&](linalg::MatmulOp mm) {
	      if (elementBytes != 4)
	        return;
	      Value a = mm.getDpsInputOperand(0)->get();
	      if (auto st = dyn_cast<ShapedType>(a.getType())) {
	        Type et = st.getElementType();
	        if (auto ft = dyn_cast<FloatType>(et))
	          elementBytes = std::max<int64_t>(1, (ft.getWidth() + 7) / 8);
	        else if (auto it = dyn_cast<IntegerType>(et))
	          elementBytes = std::max<int64_t>(1, (it.getWidth() + 7) / 8);
	      }
	    });

	    bool hasRowReduction = false;
	    module->walk([&](Operation *op) {
	      if (isRowWiseReductionOp(op)) {
	        hasRowReduction = true;
	        return WalkResult::interrupt();
	      }
	      return WalkResult::advance();
	    });

	    int64_t tilingThreadTileM = chosenThreadTileM;
	    if ((enableTensorCoreTf32 || enableTensorCoreF16) && !hasRowReduction) {
	      tilingThreadTileM = std::max<int64_t>(tilingThreadTileM, 2);
	    }
	    tagLinalgOpsForGenericCodegen(*module);
    transformLib = buildTransformLibrary(
        &ctx, tileM, tileN, tileK, blockDimX, blockDimY, chosenMmaM, chosenMmaN,
        elementBytes, tilingThreadTileM,
        chosenThreadTileN, solveRes.detectedConsumerChain, hasRowReduction,
        enableAsyncCopy,
        enableSoftwarePipelining, pipelineDepth, pipelinePeelEpilogue,
        asyncBypassL1, enableTensorCoreTf32, enableTensorCoreF16,
        swapBlockDims, gSkipMapForallToBlocks,
        gSkipMapNestedForallToThreads);
    return 0;

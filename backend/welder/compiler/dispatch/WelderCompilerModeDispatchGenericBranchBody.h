#pragma once

// 路径二实现主体（由 WelderCompilerModeDispatchGenericBranch.cpp 包装调用）。
    [[maybe_unused]] auto span =
        tracerPtr ? tracerPtr->span("compiler.solve_generic")
                  : welder::Tracer::Span();

    auto genericProbOpt = welder::analyzeGenericProblem(*module);
    if (!genericProbOpt) {
      llvm::errs() << "error: cannot find a static-shape linalg op for generic "
                      "analysis in "
                   << inputFilename << "\n";
      return 2;
    }
    if (!genericProbOpt->targetOp) {
      llvm::errs()
          << "error: generic analysis returned null targetOp (unexpected)\n";
      return 2;
    }

    // 给 target op 打一个“锚点”attribute，后续 transform 用它精准 match，避免多个
    // linalg.generic 混淆。
    genericProbOpt->targetOp->setAttr("welder.target", UnitAttr::get(&ctx));

    welder::SolveResult solveRes;
    {
      [[maybe_unused]] auto span =
          tracerPtr ? tracerPtr->span("compiler.solve_generic.call")
                    : welder::Tracer::Span();
      solveRes = welder::solveGeneric(*module, solveOpts);
    }
    if (solveRes.sortedCandidates.empty()) {
      llvm::errs() << "error: generic solve found no valid candidates.\n";
      llvm::errs() << "hint: try --require-perfect-tiling=false.\n";
      return 2;
    }

    const welder::Candidate &best = solveRes.sortedCandidates.front();
    llvm::SmallVector<int64_t, 8> bestLoopTileExtents;
    if (!best.loopTileExtents.empty() &&
        best.loopTileExtents.size() == genericProbOpt->loops.size()) {
      bestLoopTileExtents.assign(best.loopTileExtents.begin(),
                                 best.loopTileExtents.end());
    } else {
      // 论文调度当前仍按 legacy 三元组（tileM/tileN/tileK）排序候选。
      // 这里补齐每个 loop 的 extent，便于下游 generic 代码生成继续复用同一套
      // transform library 构建逻辑。
      bestLoopTileExtents.assign(genericProbOpt->loops.size(), 0);
      int64_t pSeen = 0;
      int64_t rSeen = 0;
      for (size_t i = 0; i < genericProbOpt->loops.size(); ++i) {
        const auto &loop = genericProbOpt->loops[i];
        int64_t full = loop.size;
        if (full <= 0)
          full = 1;
        if (loop.type == mlir::utils::IteratorType::parallel) {
          int64_t t = full;
          if (pSeen == 0 && best.tileM > 0)
            t = best.tileM;
          else if (pSeen == 1 && best.tileN > 0)
            t = best.tileN;
          bestLoopTileExtents[i] = t;
          ++pSeen;
          continue;
        }
        if (loop.type == mlir::utils::IteratorType::reduction) {
          int64_t t = full;
          if (rSeen == 0 && best.tileK > 0)
            t = best.tileK;
          bestLoopTileExtents[i] = t;
          ++rSeen;
          continue;
        }
        bestLoopTileExtents[i] = full;
      }
    }
    if (bestLoopTileExtents.empty() ||
        bestLoopTileExtents.size() != genericProbOpt->loops.size()) {
      llvm::errs()
          << "error: generic best candidate has unexpected loopTileExtents.\n";
      return 2;
    }

    // 允许通过 legacy `--force-tile-m/n/k` 覆盖已选的每层 loop tile。
    // 这在 TensorCore 典型链实验中尤其有用：可将 shared 层 block tile 固定为
    // 已知可行的 MMA 形状（例如满足 m16n8 的 warps-per-block<=32 约束）。
    if (forceTileM > 0 || forceTileN > 0 || forceTileK > 0) {
      int64_t pSeen = 0;
      int64_t rSeen = 0;
      for (size_t i = 0; i < genericProbOpt->loops.size(); ++i) {
        const auto &loop = genericProbOpt->loops[i];
        int64_t full = loop.size;
        if (full <= 0)
          full = 1;

        if (loop.type == mlir::utils::IteratorType::parallel) {
          int64_t t = bestLoopTileExtents[i];
          if (pSeen == 0 && forceTileM > 0)
            t = forceTileM;
          else if (pSeen == 1 && forceTileN > 0)
            t = forceTileN;
          ++pSeen;

          if (t > 0) {
            if (t > full) {
              llvm::errs() << "error: forced tile exceeds loop range in generic "
                              "mode: tile="
                           << t << " full=" << full << "\n";
              return 2;
            }
            if (solveOpts.requirePerfectTiling && (full % t != 0)) {
              llvm::errs() << "error: forced tile violates perfect-tiling in "
                              "generic mode: tile="
                           << t << " full=" << full << "\n";
              llvm::errs() << "hint: try --require-perfect-tiling=false.\n";
              return 2;
            }
            bestLoopTileExtents[i] = t;
          }
          continue;
        }

        if (loop.type == mlir::utils::IteratorType::reduction) {
          int64_t t = bestLoopTileExtents[i];
          if (rSeen == 0 && forceTileK > 0)
            t = forceTileK;
          ++rSeen;
          if (t > 0) {
            if (t > full) {
              llvm::errs() << "error: forced tile exceeds reduction range in "
                              "generic mode: tile="
                           << t << " full=" << full << "\n";
              return 2;
            }
            if (solveOpts.requirePerfectTiling && (full % t != 0)) {
              llvm::errs() << "error: forced tile violates perfect-tiling for "
                              "reduction in generic mode: tile="
                           << t << " full=" << full << "\n";
              llvm::errs() << "hint: try --require-perfect-tiling=false.\n";
              return 2;
            }
            bestLoopTileExtents[i] = t;
          }
          continue;
        }
      }
    }

    // Phase 13B：当 enableCutEdges 时，我们在 compiler 侧复跑一次 propagation，
    // 读取 cut-edge 标记，并生成“多 kernel”的 transform library。
    if (enableCutEdges) {
      auto graphOpt = welder::buildLinalgTileGraph(*module);
      if (!graphOpt || graphOpt->nodes.empty()) {
        llvm::errs()
            << "error: --enable-cut-edges requires a non-empty TileGraph.\n";
        return 2;
      }

      // 多 sink 支持：从每个 sink（outEdges 为空）传播 required tile，
      // 保证断连图或多输出图也能得到正确的 `{welder.kernel_root}` 标记
      // 与切边处理结果。
      llvm::SmallVector<int, 8> sinks;
      for (int i = 0; i < static_cast<int>(graphOpt->nodes.size()); ++i) {
        if (!graphOpt->nodes[i].outEdges.empty())
          continue;
        auto op0 = dyn_cast_or_null<linalg::LinalgOp>(graphOpt->nodes[i].op);
        if (!op0)
          continue;
        sinks.push_back(i);
      }
      if (sinks.empty()) {
        llvm::errs()
            << "error: --enable-cut-edges requires at least one linalg sink op.\n";
        return 2;
      }

      // 复用 solver 启发式：将 best.tileM/tileN 映射到 sink 的前两条 parallel loop，
      // 其余 parallel 维默认使用 full size。
      auto buildRootParallelExtents =
          [&](linalg::LinalgOp op, const welder::Candidate &c)
              -> std::optional<llvm::SmallVector<int64_t, 8>> {
        if (!op)
          return std::nullopt;
        llvm::SmallVector<int64_t, 8> ranges = op.getStaticLoopRanges();
        if (static_cast<int64_t>(ranges.size()) != op.getNumLoops())
          return std::nullopt;

        llvm::SmallVector<int64_t, 8> parallel;
        parallel.reserve(op.getNumParallelLoops());

        int64_t pSeen = 0;
        auto iters = op.getIteratorTypesArray();
        for (int64_t i = 0; i < op.getNumLoops(); ++i) {
          if (iters[i] != mlir::utils::IteratorType::parallel)
            continue;

          int64_t full = ranges[i];
          if (full == ShapedType::kDynamic || full <= 0)
            return std::nullopt;

          int64_t t = full;
          // 允许通过 `--force-tile-m/n` 固定 sink（block 级）tile。
          // 这对 Matmul->Softmax 等典型链很关键：
          // - 需要 TILE_N 覆盖整行范围（softmax 行归约）；
          // - TensorCore 需要满足 MMA 可行 tile（warps-per-block<=32）；
          // - 否则 generic solver 可能选出更大的 TILE_M，导致归约切分时线程预算溢出。
          if (pSeen == 0)
            t = (forceTileM > 0) ? forceTileM : c.tileM;
          else if (pSeen == 1)
            t = (forceTileN > 0) ? forceTileN : c.tileN;

          if (t <= 0 || t > full)
            return std::nullopt;
          if (solveOpts.requirePerfectTiling && (full % t != 0))
            return std::nullopt;

          parallel.push_back(t);
          ++pSeen;
        }

        if (static_cast<int64_t>(parallel.size()) != op.getNumParallelLoops())
          return std::nullopt;
        return parallel;
      };

      auto buildOpTileFromParallelExtents =
          [&](linalg::LinalgOp op, llvm::ArrayRef<int64_t> parallelExtents,
              int64_t defaultReductionTile) -> std::optional<welder::OpTile> {
        if (!op)
          return std::nullopt;
        if (static_cast<int64_t>(parallelExtents.size()) !=
            op.getNumParallelLoops())
          return std::nullopt;

        llvm::SmallVector<int64_t, 8> staticLoopRanges = op.getStaticLoopRanges();
        if (static_cast<int64_t>(staticLoopRanges.size()) != op.getNumLoops())
          return std::nullopt;

        welder::OpTile tile;
        tile.loopExtents.resize(op.getNumLoops(), 0);

        int64_t pIdx = 0;
        auto iters = op.getIteratorTypesArray();
        for (int64_t i = 0; i < op.getNumLoops(); ++i) {
          if (iters[i] == mlir::utils::IteratorType::parallel) {
            int64_t extent = parallelExtents[pIdx++];
            if (extent <= 0)
              return std::nullopt;
            tile.loopExtents[i] = extent;
            continue;
          }
          if (iters[i] == mlir::utils::IteratorType::reduction) {
            int64_t extent = defaultReductionTile;
            if (extent <= 0) {
              int64_t full = staticLoopRanges[i];
              if (full == ShapedType::kDynamic || full <= 0)
                return std::nullopt;
              extent = full;
            }
            tile.loopExtents[i] = extent;
            continue;
          }
          return std::nullopt;
        }

        return tile;
      };

      welder::LinalgIndexingMapsFootprintInference infer;
      std::vector<std::vector<int64_t>> reductionTilesByNode;
      welder::TilePropagationOptions popts;
      // 允许在 generic/切边代码生成中覆盖默认归约 tile（例如 matmul 的 K 切分）。
      // 这有助于在小型号 GPU 上把 shared 内存占用控制在单 block 预算内，
      // 也便于复现 A/B 性能测量结果。
      int64_t effectiveDefaultReductionTile = best.tileK;
      if (forceTileK > 0)
        effectiveDefaultReductionTile = forceTileK;
      popts.defaultReductionTile = effectiveDefaultReductionTile;
      popts.enableCutEdges = true;
      popts.resetGraphState = true;

      // 行归约链启发式：传播 sink tile 时，将完整归约/广播范围保留在 block 内。
      // generic solver 对逐元素 sink 可能会选更小的 TILE_N，但当前行归约代码生成
      // 依赖 block 内 full-N，才能避免重复归约并支持复用融合。
      int64_t graphMaxRowReductionExtent = 1;
      if (enableRowReductionChainReuseFusion) {
        for (const welder::TileGraphNode &n : graphOpt->nodes) {
          Operation *op0 = n.op;
          if (!op0)
            continue;
          if (!isRowWiseReductionOp(op0))
            continue;
          auto gen0 = dyn_cast_or_null<linalg::GenericOp>(op0);
          if (!gen0)
            continue;
          llvm::SmallVector<int64_t, 4> ranges = gen0.getStaticLoopRanges();
          if (ranges.size() == 2 && ranges[1] != ShapedType::kDynamic &&
              ranges[1] > 0) {
            graphMaxRowReductionExtent =
                std::max<int64_t>(graphMaxRowReductionExtent, ranges[1]);
          }
        }
      }

      // 行归约链（Softmax/LayerNorm 类）通常需要在 block 内保留完整归约范围，
      // 避免共享 2D producer（如 exp_sub 同时喂 sum/out）上的传播冲突。
      //
      // 仅对行归约请求“full 归约”（extent=0 表示 full），其余归约仍沿用
      // generic 默认 tileK。
      if (enableRowReductionChainReuseFusion) {
        reductionTilesByNode.resize(graphOpt->nodes.size());
        for (size_t ni = 0; ni < graphOpt->nodes.size(); ++ni) {
          Operation *op0 = graphOpt->nodes[static_cast<int>(ni)].op;
          if (!op0)
            continue;
          if (!isRowWiseReductionOp(op0))
            continue;
          reductionTilesByNode[ni] = {0};
        }
        popts.reductionTilesByNode = &reductionTilesByNode;
      }

      bool firstSink = true;
      for (int sink : sinks) {
        auto sinkOp =
            dyn_cast_or_null<linalg::LinalgOp>(graphOpt->nodes[sink].op);
        if (!sinkOp)
          continue;

        auto rootParallelExtentsOpt = buildRootParallelExtents(sinkOp, best);
        if (!rootParallelExtentsOpt) {
          llvm::errs()
              << "error: failed to build sink parallel extents under constraints.\n";
          return 2;
        }

        if (enableRowReductionChainReuseFusion && graphMaxRowReductionExtent > 1) {
          // 对典型行归约链，sink 往往是 2D 逐元素算子（M,N）。
          // 将完整 N（广播/归约）范围保留在 block 内，避免 tile 不匹配导致传播切边。
          auto sinkGen = dyn_cast<linalg::GenericOp>(sinkOp.getOperation());
          if (sinkGen && sinkGen.getNumLoops() == 2 &&
              sinkGen.getNumReductionLoops() == 0 &&
              sinkGen.getNumParallelLoops() == 2 &&
              rootParallelExtentsOpt->size() >= 2) {
            llvm::SmallVector<int64_t, 8> ranges = sinkGen.getStaticLoopRanges();
            if (ranges.size() == 2 && ranges[1] != ShapedType::kDynamic &&
                ranges[1] > 0) {
              (*rootParallelExtentsOpt)[1] = ranges[1];
            } else {
              (*rootParallelExtentsOpt)[1] =
                  std::max<int64_t>((*rootParallelExtentsOpt)[1],
                                    graphMaxRowReductionExtent);
            }
          }
        }

        auto rootTileOpt = buildOpTileFromParallelExtents(
            sinkOp, *rootParallelExtentsOpt,
            /*defaultReductionTile=*/effectiveDefaultReductionTile);
        if (!rootTileOpt) {
          llvm::errs() << "error: failed to build sink OpTile.\n";
          return 2;
        }

        welder::TilePropagationResult pr = welder::propagateTilesBackward(
            * graphOpt, sink, *rootTileOpt, infer, popts);
        if (!pr.success) {
          llvm::errs() << "error: cut-edge propagation failed: " << pr.error
                       << "\n";
          return 2;
        }

        // 在多个 sink 之间累计约束（不要覆盖 requiredTile/isCut）。
        if (firstSink) {
          firstSink = false;
          popts.resetGraphState = false;
          popts.resetCutEdges = false;
        }
      }

      // 识别 kernel roots：
      // - sink 节点（无 outEdges）；
      // - cut producer：任意 outEdge 被标记为 isCut。
      // === 确定性、互斥的 kernel 划分 ===
      //
      // 基于 transform 的融合路径会把 producer 克隆到 kernel root 的 `scf.forall` 中。
      // 若同一 producer 同时服务多个 kernel root，直接克隆会导致重复计算，
      // 或删除原 producer 后破坏其他消费者。
      //
      // 为保证语义正确，并对齐 Welder“通过切边显式物化中间值”的原则，这里约束：
      // - 每个非平凡 linalg 算子只能归属一个 kernel；
      // - 若某算子会落入多个 kernel，则自动切断对应跨 kernel 边，
      //   强制走 global-memory 复用。

      auto isNonTrivialLinalgNode = [&](int idx) -> bool {
        return welder::compiler::isNonTrivialLinalgNode(*graphOpt, idx);
      };

      llvm::SmallVector<int, 64> topo =
          welder::compiler::computeTopoOrder(*graphOpt);
      llvm::DenseMap<int, int> topoIndex =
          welder::compiler::buildTopoIndex(topo);

      // 归约链启发式：对含行归约的图，来自“归约派生”1D 值到 2D 逐元素消费者的
      // 广播边在当前 generic 代码生成里常难以稳定融合，可能把 1D 链误留在 host。
      // 优先切断这类边，让两侧都成为显式 GPU kernel（仍可做 A/B 对比）。
	      {
	        bool hasRowReduction = false;
        for (int idx : topo) {
          if (!isNonTrivialLinalgNode(idx))
            continue;
          Operation *op = graphOpt->nodes[idx].op;
          if (isRowWiseReductionOp(op)) {
            hasRowReduction = true;
            break;
          }
        }
        bool warnedUnsafeBroadcast = false;
        auto hasNonBroadcastConsumer = [&](int src) -> bool {
          if (src < 0 || src >= static_cast<int>(graphOpt->nodes.size()))
            return false;
          for (int edgeIdx : graphOpt->nodes[src].outEdges) {
            if (edgeIdx < 0 ||
                edgeIdx >= static_cast<int>(graphOpt->edges.size()))
              continue;
            const welder::TileGraphEdge &oe = graphOpt->edges[edgeIdx];
            if (oe.isCut)
              continue;
            if (oe.dst < 0 ||
                oe.dst >= static_cast<int>(graphOpt->nodes.size()))
              continue;
            if (!isBroadcast1DTo2DEdge(*graphOpt, oe))
              return true;
          }
          return false;
        };
        if (hasRowReduction && reductionChainSplitBroadcastEdges) {
          auto dependsOnRowReduction = [&](int nodeIdx) -> bool {
            if (nodeIdx < 0 ||
                nodeIdx >= static_cast<int>(graphOpt->nodes.size()))
              return false;
            for (int edgeIdx : graphOpt->nodes[nodeIdx].inEdges) {
              if (edgeIdx < 0 ||
                  edgeIdx >= static_cast<int>(graphOpt->edges.size()))
                continue;
              const welder::TileGraphEdge &ie = graphOpt->edges[edgeIdx];
              int src = ie.src;
              if (src < 0 ||
                  src >= static_cast<int>(graphOpt->nodes.size()))
                continue;
              if (isRowWiseReductionOp(graphOpt->nodes[src].op))
                return true;
            }
            return false;
          };
          for (welder::TileGraphEdge &e : graphOpt->edges) {
            if (!isBroadcast1DTo2DEdge(*graphOpt, e))
              continue;
            // 两类情况都切：
            // - 归约派生 1D 值（如 mean*scale）；
            // - 直接行归约结果（如 max/sum 供给 softmax 各阶段）。
            if (!dependsOnRowReduction(e.src) &&
                !(e.src >= 0 &&
                  e.src < static_cast<int>(graphOpt->nodes.size()) &&
                  isRowWiseReductionOp(graphOpt->nodes[e.src].op)))
              continue;
            welder::setEdgeConnectLevel(e, welder::kConnectLevelGlobal);
          }
        } else if (hasRowReduction) {
          // 安全兜底：若行归约结果扇出到多个消费者（如 mean 同时喂 sq/var），
          // generic fusion 目前还不能稳定保证其留在同一个 gpu.launch 内。
          // 因此切断广播边，避免错误的 host 侧计算。
          for (int src = 0; src < static_cast<int>(graphOpt->nodes.size());
               ++src) {
            if (!isRowWiseReductionOp(graphOpt->nodes[src].op))
              continue;
            bool hasBroadcast = false;
            for (int edgeIdx : graphOpt->nodes[src].outEdges) {
              if (edgeIdx < 0 ||
                  edgeIdx >= static_cast<int>(graphOpt->edges.size()))
                continue;
              const welder::TileGraphEdge &oe = graphOpt->edges[edgeIdx];
              if (oe.isCut)
                continue;
              if (isBroadcast1DTo2DEdge(*graphOpt, oe)) {
                hasBroadcast = true;
                break;
              }
            }
            if (!hasBroadcast)
              continue;
            if (!hasNonBroadcastConsumer(src))
              continue;
            // 切断该归约节点所有出边，使其成为 kernel root；
            // 在复用融合尚未实现完整前，可避免错误的 host 执行。
            for (int edgeIdx : graphOpt->nodes[src].outEdges) {
              if (edgeIdx < 0 ||
                  edgeIdx >= static_cast<int>(graphOpt->edges.size()))
                continue;
              welder::TileGraphEdge &oe = graphOpt->edges[edgeIdx];
              if (oe.isCut)
                continue;
              welder::setEdgeConnectLevel(oe, welder::kConnectLevelGlobal);
            }
            if (!warnedUnsafeBroadcast) {
              llvm::errs()
                  << "note: unsafe row-reduction broadcast fusion detected; "
                     "falling back to cut-edge for correctness. "
                     "Enable dedicated reuse fusion to avoid this.\n";
              warnedUnsafeBroadcast = true;
            }
          }
	        }
	      }

	      // 切边基线模式的稳健性处理：
	      //
	      // 当切断行归约广播边（max/sum/mean -> 2D）后，上游 2D producer
	      // （如 matmul 输出）常会变成多 kernel 扇出值。若仍强行融合到单消费者，
	      // `transform.structured.fuse_into_containing_op` 可能克隆该 producer 并遗留
	      // host 侧副本，后续会破坏 shared promotion 与 NVVM lowering。
	      //
	      // 在完整的多消费者复用调度落地前，此模式优先将多消费者 producer
	      // 物化到 global 内存。
	      if (reductionChainSplitBroadcastEdges &&
	          !enableRowReductionChainReuseFusion) {
	        for (int src = 0; src < static_cast<int>(graphOpt->nodes.size());
	             ++src) {
	          if (!isNonTrivialLinalgNode(src))
	            continue;
	          if (!graphOpt->nodes[src].hasRequiredTile)
	            continue;

	          int nonCutOut = 0;
	          for (int edgeIdx : graphOpt->nodes[src].outEdges) {
	            if (edgeIdx < 0 ||
	                edgeIdx >= static_cast<int>(graphOpt->edges.size()))
	              continue;
	            const welder::TileGraphEdge &oe = graphOpt->edges[edgeIdx];
	            if (!oe.isCut)
	              ++nonCutOut;
	          }
	          if (nonCutOut <= 1)
	            continue;

	          for (int edgeIdx : graphOpt->nodes[src].outEdges) {
	            if (edgeIdx < 0 ||
	                edgeIdx >= static_cast<int>(graphOpt->edges.size()))
	              continue;
	            welder::TileGraphEdge &oe = graphOpt->edges[edgeIdx];
	            if (oe.isCut)
	              continue;
	            welder::setEdgeConnectLevel(oe, welder::kConnectLevelGlobal);
	          }
	        }
	      }

      llvm::SmallVector<int, 16> rootNodes =
          welder::compiler::collectInitialKernelRoots(*graphOpt, topoIndex);
      if (rootNodes.empty()) {
        llvm::errs()
            << "error: --enable-cut-edges found no kernel roots (unexpected).\n";
        return 2;
      }

      // node -> 已分配 kernelId，-1 表示未分配。
      std::vector<int64_t> nodeToKernel(graphOpt->nodes.size(), -1);

      auto assignKernelFromRoot = [&](int rootNode, int64_t kernelId,
                                      bool kernelSwapXY) {
        llvm::SmallVector<int, 32> worklist;
        worklist.push_back(rootNode);
        nodeToKernel[rootNode] = kernelId;

        while (!worklist.empty()) {
          int cur = worklist.pop_back_val();
          if (cur < 0 || cur >= static_cast<int>(graphOpt->nodes.size()))
            continue;
          for (int edgeIdx : graphOpt->nodes[cur].inEdges) {
            if (edgeIdx < 0 ||
                edgeIdx >= static_cast<int>(graphOpt->edges.size()))
              continue;
            welder::TileGraphEdge &e = graphOpt->edges[edgeIdx];
            if (e.isCut)
              continue;
            int src = e.src;
            if (!isNonTrivialLinalgNode(src))
              continue;
            if (!graphOpt->nodes[src].hasRequiredTile)
              continue;

            // 论文/Welder 对齐：融合 kernel 内保持一致的 block 顺序。
            // 若 producer 需要不同的 swapXY 映射，则切边并物化到 global 内存。
            if (graphOpt->nodes[src].swapXYHint &&
                * graphOpt->nodes[src].swapXYHint != kernelSwapXY) {
              welder::setEdgeConnectLevel(e, welder::kConnectLevelGlobal);
              continue;
            }

            if (nodeToKernel[src] == -1) {
              nodeToKernel[src] = kernelId;
              worklist.push_back(src);
              continue;
            }

            if (nodeToKernel[src] != kernelId) {
              // 跨 kernel 冲突：通过切边物化到 global 内存。
              welder::setEdgeConnectLevel(e, welder::kConnectLevelGlobal);
            }
          }
        }
      };

      // 不动点式 kernel 发现：任何尚未分配的节点都作为新 root，
      // 持续分配直到全部覆盖。
      for (size_t kernelId = 0; kernelId < rootNodes.size(); ++kernelId) {
        int root = rootNodes[kernelId];
        bool swapXY = false;
        if (root >= 0 && root < static_cast<int>(graphOpt->nodes.size()) &&
            graphOpt->nodes[root].swapXYHint)
          swapXY = *graphOpt->nodes[root].swapXYHint;
        assignKernelFromRoot(root, static_cast<int64_t>(kernelId), swapXY);
      }

      while (true) {
        bool added = false;
        for (int idx : topo) {
          if (!isNonTrivialLinalgNode(idx))
            continue;
          if (!graphOpt->nodes[idx].hasRequiredTile)
            continue;
          if (nodeToKernel[idx] != -1)
            continue;
          // 未覆盖节点：强制提升为 kernel root。
          rootNodes.push_back(idx);
          int64_t kernelId = static_cast<int64_t>(rootNodes.size() - 1);
          bool swapXY = false;
          if (graphOpt->nodes[idx].swapXYHint)
            swapXY = *graphOpt->nodes[idx].swapXYHint;
          assignKernelFromRoot(idx, kernelId, swapXY);
          added = true;
        }
        if (!added)
          break;
      }

      // 以确定性顺序（拓扑序）排列 root，并压缩 kernelId。
      llvm::sort(rootNodes, [&](int a, int b) {
        return topoIndex.lookup(a) < topoIndex.lookup(b);
      });
      rootNodes.erase(std::unique(rootNodes.begin(), rootNodes.end()),
                      rootNodes.end());

      // 排序后按压缩后的 id 重新分配。
      std::fill(nodeToKernel.begin(), nodeToKernel.end(), -1);
      for (size_t kernelId = 0; kernelId < rootNodes.size(); ++kernelId) {
        int root = rootNodes[kernelId];
        bool swapXY = false;
        if (root >= 0 && root < static_cast<int>(graphOpt->nodes.size()) &&
            graphOpt->nodes[root].swapXYHint)
          swapXY = *graphOpt->nodes[root].swapXYHint;
        assignKernelFromRoot(root, static_cast<int64_t>(kernelId), swapXY);
      }

      // 最终覆盖性检查。
      for (int idx : topo) {
        if (!isNonTrivialLinalgNode(idx))
          continue;
        if (!graphOpt->nodes[idx].hasRequiredTile)
          continue;
        if (nodeToKernel[idx] == -1) {
          llvm::errs() << "error: internal: linalg op not assigned to any kernel: "
                       << graphOpt->nodes[idx].op->getName().getStringRef()
                       << "\n";
          return 2;
        }
      }

      // 归约链 v1 启发式：
      // 若 kernel 含行归约（["parallel","归约"]），则避免在 block 级
      // 沿归约/广播维切 kernel root；否则 tile-and-fuse 往往会在每个 block
      // 重复完整归约计算。
      llvm::SmallVector<bool, 8> kernelHasRowReduction(rootNodes.size(), false);
      llvm::SmallVector<bool, 8> kernelHasMatmul(rootNodes.size(), false);
      llvm::SmallVector<int64_t, 8> kernelRowReductionCount(rootNodes.size(), 0);
      int64_t maxRowReductionExtent = 1;
      for (int idx : topo) {
        int64_t kid = nodeToKernel[idx];
        if (kid < 0 || kid >= static_cast<int64_t>(kernelHasRowReduction.size()))
          continue;
        Operation *op = graphOpt->nodes[idx].op;
        if (isa<linalg::MatmulOp>(op))
          kernelHasMatmul[kid] = true;
        auto gen = dyn_cast_or_null<linalg::GenericOp>(op);
        if (!gen)
          continue;
        if (gen.getNumLoops() != 2 || gen.getNumReductionLoops() != 1)
          continue;
        auto iters = gen.getIteratorTypesArray();
        if (iters.size() != 2 ||
            iters[0] != mlir::utils::IteratorType::parallel ||
            iters[1] != mlir::utils::IteratorType::reduction)
          continue;
        kernelHasRowReduction[kid] = true;
        kernelRowReductionCount[kid] += 1;
        llvm::SmallVector<int64_t, 4> ranges = gen.getStaticLoopRanges();
        if (ranges.size() == 2 && ranges[1] != ShapedType::kDynamic &&
            ranges[1] > 0) {
          maxRowReductionExtent = std::max<int64_t>(maxRowReductionExtent, ranges[1]);
        }
      }
      bool tcHasMatmulRowReductionKernel = false;
      if (enableTensorCoreTf32 || enableTensorCoreF16) {
        for (size_t kid = 0; kid < kernelHasRowReduction.size(); ++kid) {
          if (!kernelHasRowReduction[kid] || !kernelHasMatmul[kid])
            continue;
          tcHasMatmulRowReductionKernel = true;
          break;
        }
      }

      // 写入属性并生成 kernel 规格。
      OpBuilder tagBuilder(&ctx);
      llvm::SmallVector<KernelSpec, 8> kernels;
      kernels.reserve(rootNodes.size());

      for (int64_t kernelId = 0; kernelId < static_cast<int64_t>(rootNodes.size());
           ++kernelId) {
        int nodeIdx = rootNodes[kernelId];
        welder::TileGraphNode &node = graphOpt->nodes[nodeIdx];
        auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(node.op);
        if (!linalgOp)
          continue;
        if (!node.hasRequiredTile)
          continue;
        if (static_cast<int64_t>(node.requiredTile.loopExtents.size()) !=
            linalgOp.getNumLoops())
          continue;

        node.op->setAttr("welder.kernel_root",
                         tagBuilder.getI32IntegerAttr(kernelId));
        node.op->setAttr("welder.kernel_id",
                         tagBuilder.getI32IntegerAttr(kernelId));

        KernelSpec spec;
        spec.kernelId = kernelId;
        spec.opName = node.op->getName().getStringRef().str();
        if (kernelId >= 0 &&
            kernelId < static_cast<int64_t>(kernelRowReductionCount.size())) {
          spec.rowReductionCount = kernelRowReductionCount[kernelId];
        }

        // tileSizes：只切前两维 parallel；reduction 维置 0（L1 不切）。
        int64_t maxParallelTileDims = 2;
        if (kernelId >= 0 &&
            kernelId < static_cast<int64_t>(kernelHasRowReduction.size()) &&
            kernelHasRowReduction[kernelId]) {
          maxParallelTileDims = 1;
        }
        spec.tileSizes.assign(linalgOp.getNumLoops(), 0);
        int64_t pSeen = 0;
        auto iters = linalgOp.getIteratorTypesArray();
        for (int64_t i = 0; i < linalgOp.getNumLoops(); ++i) {
          if (iters[i] == mlir::utils::IteratorType::reduction) {
            spec.tileSizes[i] = 0;
            continue;
          }
          if (iters[i] != mlir::utils::IteratorType::parallel) {
            spec.tileSizes[i] = 0;
            continue;
          }
          if (pSeen < maxParallelTileDims)
            spec.tileSizes[i] = node.requiredTile.loopExtents[i];
          else
            spec.tileSizes[i] = 0;
          ++pSeen;
        }

        // 每个 kernel 的可选 block 顺序提示，来源于 tile-graph 构建阶段。
        // 这是对参考实现 block_order 分配的近似。
        if (node.swapXYHint && *node.swapXYHint) {
          spec.swapXY = true;
          node.op->setAttr("welder.swap_xy", tagBuilder.getBoolAttr(true));
        } else {
          node.op->removeAttr("welder.swap_xy");
        }

        kernels.push_back(std::move(spec));
      }

      // 给每个 kernel 标记可融合的 producer（不含 kernel root 本身）。
      for (int idx : topo) {
        if (!isNonTrivialLinalgNode(idx))
          continue;
        if (!graphOpt->nodes[idx].hasRequiredTile)
          continue;
        int64_t kid = nodeToKernel[idx];
        if (kid < 0)
          continue;
        int rootIdx = rootNodes[static_cast<size_t>(kid)];
        if (idx == rootIdx)
          continue;
        Operation *op = graphOpt->nodes[idx].op;
        if (!op)
          continue;
        // 避免标签冲突：正确划分下同一算子只应被标记一次。
        if (!op->hasAttr("welder.kernel_id")) {
          op->setAttr("welder.kernel_id", tagBuilder.getI32IntegerAttr(kid));
        }
        // transform 融合额外标签：排除 kernel root（它也带 `welder.kernel_id`），
        // 防止把 root 误融合进自身。
        if (!op->hasAttr("welder.kernel_producer")) {
          op->setAttr("welder.kernel_producer",
                      tagBuilder.getI32IntegerAttr(kid));
        }
      }

      // 为每个 kernel 计算确定性的 producer 融合顺序（逆拓扑）：
      // 按 sink->source 融合，确保多跳 producer 仅在其消费者链已经进入
      // kernel root 的 forall 后再融合。
      auto orderedProducers = welder::compiler::buildOrderedProducersByKernel(
          *graphOpt, topo, nodeToKernel, rootNodes);
      for (KernelSpec &spec : kernels) {
        if (spec.kernelId < 0 ||
            spec.kernelId >= static_cast<int64_t>(orderedProducers.size()))
          continue;
        spec.orderedProducerNodeIds = std::move(
            orderedProducers[static_cast<size_t>(spec.kernelId)]);
      }

      // 行归约链融合：在 tile graph 中，针对“逐元素->行归约”且 producer
      // 只有单个（未切边）消费者的配对执行融合。
      //
      // 这样可避免 bufferization 后把大 2D 中间结果落到 global 内存
      // （如 LayerNorm 链中 sq 供给 sumsq）。
      std::vector<RowReductionFusionPair> fuseElementwiseIntoRowReductions;
      {
        auto rowPairSpecs = welder::compiler::buildRowReductionFusionPairSpecs(
            *graphOpt, nodeToKernel, enableRowReductionChainReuseFusion,
            enableTensorCoreTf32, enableTensorCoreF16);
        fuseElementwiseIntoRowReductions.reserve(rowPairSpecs.size());
        for (const auto &spec : rowPairSpecs) {
          RowReductionFusionPair p;
          p.kernelId = spec.kernelId;
          p.producerNodeId = spec.producerNodeId;
          p.consumerNodeId = spec.consumerNodeId;
          p.fuseIntoBlockForall = spec.fuseIntoBlockForall;
          fuseElementwiseIntoRowReductions.push_back(std::move(p));
        }
      }

      // 寄存器级融合：把“逐元素->逐元素”配对融合到消费者线程级 forall，
      // 让中间结果停留在寄存器。
      // 对递归调度（max_connect_level>2），遵循 solver 的递归内层边界，
      // 仅融合更深连接层级的边，并且仅融合同一 kernel 内单消费者 producer。
      std::vector<ThreadFusionPair> threadFusionPairs;
      welder::compiler::ThreadFusionDecision threadFusionDecision;
      {
        auto threadPairSpecs =
            welder::compiler::buildThreadFusionPairSpecsFromGraph(
                *graphOpt, nodeToKernel, solveOpts, &threadFusionDecision);
        threadFusionPairs.reserve(threadPairSpecs.size());
        for (const auto &spec : threadPairSpecs) {
          ThreadFusionPair p;
          p.kernelId = spec.kernelId;
          p.producerNodeId = spec.producerNodeId;
          p.consumerNodeId = spec.consumerNodeId;
          p.consumerOperand = spec.consumerOperand;
          threadFusionPairs.push_back(std::move(p));
        }
      }
      normalizeThreadFusionPairs(threadFusionPairs);
      if (tracerPtr) {
        int64_t pairWithOperand = 0;
        for (const ThreadFusionPair &p : threadFusionPairs) {
          if (p.consumerOperand >= 0)
            ++pairWithOperand;
        }
        llvm::json::Object f;
        f["max_connect_level"] = static_cast<int64_t>(solveOpts.maxConnectLevel);
        f["register_fuse_min_connect_level"] =
            static_cast<int64_t>(threadFusionDecision.registerFuseMinConnectLevel);
        f["promote_shared_edges"] =
            threadFusionDecision.promoteSharedEdgesForRegisterFuse;
        f["pair_count"] = static_cast<int64_t>(threadFusionPairs.size());
        f["pair_with_operand"] = pairWithOperand;
        tracerPtr->event("compiler.thread_fusion_pairs", std::move(f),
                         /* isVerbose=*/true);
      }

      // 推导多 kernel 映射使用的 `(tileK, blockDim)`，并与 generic 最佳候选保持一致
      //（前两条 parallel loop + 第一条归约 loop）。
      int64_t effTileM = 1;
      int64_t effTileN = 1;
      int64_t effTileK = 1;
      int64_t fullReductionExtent = 1;
      {
        int64_t pSeen = 0;
        for (size_t i = 0; i < genericProbOpt->loops.size(); ++i) {
          if (genericProbOpt->loops[i].type == mlir::utils::IteratorType::parallel) {
            if (pSeen == 0)
              effTileM = bestLoopTileExtents[i];
            else if (pSeen == 1)
              effTileN = bestLoopTileExtents[i];
            ++pSeen;
          } else if (genericProbOpt->loops[i].type ==
                     mlir::utils::IteratorType::reduction) {
            if (genericProbOpt->loops[i].size > 0)
              fullReductionExtent =
                  std::max<int64_t>(fullReductionExtent, genericProbOpt->loops[i].size);
            if (effTileK <= 1)
              effTileK = bestLoopTileExtents[i];
          }
        }
      }
      if (forceTileM > 0)
        effTileM = forceTileM;
      if (forceTileN > 0)
        effTileN = forceTileN;
      // 若对行归约 kernel 决定将归约/广播维保留在 block 内，
      // 线程映射应基于完整归约范围推导 blockDimX，而不是使用 block 级 tile。
      if (maxRowReductionExtent > 1) {
        if ((enableTensorCoreTf32 || enableTensorCoreF16) &&
            tcHasMatmulRowReductionKernel) {
          constexpr int64_t kTcRowThreadExtentCap = 64;
          effTileN = std::max<int64_t>(
              effTileN,
              std::min<int64_t>(maxRowReductionExtent, kTcRowThreadExtentCap));
        } else {
          effTileN = std::max<int64_t>(effTileN, maxRowReductionExtent);
        }
      }
      effTileM = std::max<int64_t>(1, effTileM);
      effTileN = std::max<int64_t>(1, effTileN);
      effTileK = std::max<int64_t>(1, effTileK);
      if (forceTileK > 0)
        effTileK = std::max<int64_t>(1, forceTileK);

      // TensorCore（切边多 kernel）：推导与 MMA 对齐的 K 及一维线程映射
      //（即 `blockDim=[warps*32,1,1]`）。
      if (enableTensorCoreTf32 || enableTensorCoreF16) {
        const int64_t mmaM = 16;
        const int64_t mmaN = 8;
        const int64_t mmaK = enableTensorCoreF16 ? 16 : 4;
        if (effTileM % mmaM != 0 || effTileN % mmaN != 0) {
          llvm::errs() << "error: cut-edge tensorcore requires TILE_M%16==0 and "
                          "TILE_N%8==0, got M="
                       << effTileM << " N=" << effTileN << "\n";
          return 2;
        }
        int64_t warps = (effTileM / mmaM) * (effTileN / mmaN);
        if (warps <= 0 || warps * 32 > 1024) {
          llvm::errs() << "error: cut-edge tensorcore invalid warps-per-block: "
                       << warps << " (threads=" << (warps * 32) << ")\n";
          return 2;
        }
        effTileK = mmaK;
      }

      int64_t chosenThreadTileM =
          welder::compiler::pickThreadTileDivisible(effTileM, threadTileM);
      int64_t chosenThreadTileN =
          welder::compiler::pickThreadTileDivisible(effTileN, threadTileN);
      // 切边模式下的线程级切分可能覆盖与主 `(tileM,tileN)` 不同的操作数 tile
      //（例如 matmul 的 B 为 [K,N]）。这里采用保守基准，确保
      // map_nested_forall_to_threads 对这些 producer tile 也有足够线程。
      int64_t threadBasisM =
          std::max<int64_t>(effTileM,
                            std::max<int64_t>(effTileK, fullReductionExtent));
      int64_t threadBasisN =
          std::max<int64_t>(effTileN,
                            std::max<int64_t>(effTileK, fullReductionExtent));

      // 对归约占比高的 generic kernel，一旦把完整归约范围放入 block，
      // 朴素 `(threadTileM, threadTileN)` 选择很容易超过 1024 threads/block。
      // 因此逐步增大逐线程 tile（优先 N），直到固定 block 映射合法。
      if (!(enableTensorCoreTf32 || enableTensorCoreF16)) {
        for (int tries = 0; tries < 8; ++tries) {
          int64_t bx = 1;
          int64_t by = 1;
          if (!swapBlockDims) {
            bx = std::max<int64_t>(
                1, welder::compiler::ceilDivI64(threadBasisN, chosenThreadTileN));
            by = std::max<int64_t>(
                1, welder::compiler::ceilDivI64(threadBasisM, chosenThreadTileM));
          } else {
            bx = std::max<int64_t>(
                1, welder::compiler::ceilDivI64(threadBasisM, chosenThreadTileM));
            by = std::max<int64_t>(
                1, welder::compiler::ceilDivI64(threadBasisN, chosenThreadTileN));
          }
          if (bx * by <= 1024)
            break;

          // 优先增大 N 方向 tile（在 swapBlockDims=false 时可先降低 blockDimX）。
          if (chosenThreadTileN > 0 && chosenThreadTileN < effTileN &&
              (effTileN % (chosenThreadTileN * 2) == 0)) {
            chosenThreadTileN *= 2;
            continue;
          }
          if (chosenThreadTileM > 0 && chosenThreadTileM < effTileM &&
              (effTileM % (chosenThreadTileM * 2) == 0)) {
            chosenThreadTileM *= 2;
            continue;
          }
          break;
        }
      }
      int64_t blockDimX = 1;
      int64_t blockDimY = 1;
      if (enableTensorCoreTf32 || enableTensorCoreF16) {
        // TensorCore：保持二维线程布局，使逐元素 epilogue/copy 仍可分发，
        // 同时不超过 1024 线程。其逐线程 tile 推导与纯 matmul TensorCore 路径保持一致。
        const int64_t mmaM = 16;
        const int64_t mmaN = 8;
        int64_t warps = (effTileM / mmaM) * (effTileN / mmaN);
        bool allowThreadMappedMma =
            getEnvInt64OrDefault("WELDER_MM_SM_TC_ALLOW_THREAD_MAPPED_MMA", 0) !=
            0;
        // 推导逐线程 tile：对 m16n8 形状，每线程对应 4 个输出元素。
        chosenThreadTileM = std::max<int64_t>(1, mmaM / 8);
        chosenThreadTileN = std::max<int64_t>(1, mmaN / 4);

        // Matmul->Softmax 风格 kernel 包含行归约。
        // 当前归约切分会把整行映射到线程，`threadTileM > 1` 时容易溢出；
        // 因此 TensorCore 路径优先 `threadTileM==1`，保证归约映射在预算内。
        bool forceTcRowThreadTile =
            (getEnvInt64OrDefault("WELDER_MM_SM_TC_FORCE_ROW_THREAD_TILE", 1) !=
             0);
        if (!swapBlockDims && forceTcRowThreadTile) {
          bool hasRowReduction = false;
          module->walk([&](Operation *op) {
            if (isRowWiseReductionOp(op)) {
              hasRowReduction = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          if (hasRowReduction) {
            const int64_t perThreadElems =
                std::max<int64_t>(1, (mmaM * mmaN) / 32);
            if ((effTileN % perThreadElems) == 0) {
              chosenThreadTileM = 1;
              chosenThreadTileN = perThreadElems;
            }
          }
        }
        if (tcHasMatmulRowReductionKernel && !allowThreadMappedMma) {
          // 融合行归约 kernel 中，warp 映射的 MMA 可能需要比默认逐线程 epilogue
          // 切分更高的线程容量。这里持续收紧逐线程 tile（优先 N），直到 block 线程数
          // 满足保守目标预算。
          const int64_t rowThreadFactor = std::max<int64_t>(
              1, getEnvInt64OrDefault("WELDER_MM_SM_TC_WARP_ROW_THREAD_FACTOR", 2));
          const int64_t targetThreads = std::min<int64_t>(
              1024, std::max<int64_t>(warps * 32, warps * 32 * rowThreadFactor));
          auto computeBlockThreads = [&]() -> int64_t {
            int64_t bx = 1;
            int64_t by = 1;
            if (!swapBlockDims) {
              bx = effTileN / chosenThreadTileN;
              by = effTileM / chosenThreadTileM;
            } else {
              bx = effTileM / chosenThreadTileM;
              by = effTileN / chosenThreadTileN;
            }
            return bx * by;
          };
          for (int tries = 0; tries < 8; ++tries) {
            int64_t threads = computeBlockThreads();
            if (threads >= targetThreads)
              break;
            bool tightened = false;
            if (chosenThreadTileN > 1 &&
                (effTileN % (chosenThreadTileN / 2)) == 0) {
              chosenThreadTileN /= 2;
              tightened = true;
            } else if (chosenThreadTileM > 1 &&
                       (effTileM % (chosenThreadTileM / 2)) == 0) {
              chosenThreadTileM /= 2;
              tightened = true;
            }
            if (!tightened)
              break;
          }
        }

        if (effTileM % chosenThreadTileM != 0 || effTileN % chosenThreadTileN != 0) {
          llvm::errs() << "error: TensorCore requires TILE_M/TILE_N divisible by "
                          "derived thread tile in cut-edge codegen: TILE_M="
                       << effTileM << " TILE_N=" << effTileN
                       << " threadTileM=" << chosenThreadTileM
                       << " threadTileN=" << chosenThreadTileN << "\n";
          return 2;
        }

        if (!swapBlockDims) {
          blockDimX = effTileN / chosenThreadTileN;
          blockDimY = effTileM / chosenThreadTileM;
        } else {
          blockDimX = effTileM / chosenThreadTileM;
          blockDimY = effTileN / chosenThreadTileN;
        }

        if (tcHasMatmulRowReductionKernel) {
          // 允许融合 matmul->行归约 kernel 使用更大的二维线程网格。
          if (blockDimX * blockDimY < warps * 32) {
            llvm::errs()
                << "error: internal: TensorCore thread layout under-provisioned in "
                   "cut-edge codegen (threads="
                << (blockDimX * blockDimY) << " expected>=" << (warps * 32)
                << ")\n";
            return 2;
          }
        } else {
          if (blockDimX * blockDimY != warps * 32) {
            llvm::errs() << "error: internal: TensorCore thread layout mismatch in "
                            "cut-edge codegen (threads="
                         << (blockDimX * blockDimY) << " expected=" << (warps * 32)
                         << ")\n";
            return 2;
          }
        }
      } else {
        auto blockDims = welder::compiler::computeBlockDimsCeil(
            threadBasisM, threadBasisN, chosenThreadTileM, chosenThreadTileN,
            swapBlockDims);
        blockDimX = blockDims.x;
        blockDimY = blockDims.y;
      }
      if (blockDimX * blockDimY > 1024) {
        llvm::errs() << "error: block threads exceed 1024 in cut-edge codegen: ("
                     << blockDimX << "x" << blockDimY << ")\n";
        return 2;
      }
      if (traceVerbose) {
        llvm::errs() << "[welder] cut-edge blockDim=(" << blockDimX << "x"
                     << blockDimY << ") effTile=(" << effTileM << "x"
                     << effTileN << ") threadTile=(" << chosenThreadTileM << "x"
                     << chosenThreadTileN << ") forceTile=(" << forceTileM << "x"
                     << forceTileN << ") optThreadTile=(" << threadTileM << "x"
                     << threadTileN << ") maxRowRed=" << maxRowReductionExtent
                     << " fullRed=" << fullReductionExtent << "\n";
      }

      tagLinalgOpsForGenericCodegen(*module);
      bool enableAsyncGroups = enableAsyncCopy;
      transformLib = buildGenericTransformLibraryCutEdges(
          &ctx, kernels, fuseElementwiseIntoRowReductions, threadFusionPairs,
          swapBlockDims, effTileK,
          blockDimX, blockDimY, chosenThreadTileM, chosenThreadTileN,
          enableAsyncCopy, asyncBypassL1, enableAsyncGroups,
          enableTensorCoreTf32,
          enableTensorCoreF16, effEnableRowReductionInputPromotion,
          effEnableRowReductionWarp, effEnableRowReductionVectorize,
          effRowReductionVectorWidth, effRowReductionThreadsX,
          effEnableRowReductionCombineVectorize, gSkipMapForallToBlocks,
          gSkipMapNestedForallToThreads);
      // Phase 13B：cut-edge 多 kernel 模式下，我们不走旧的 “generic fusion” 分支。
    } else {
      // 构造两级 tiling：
      // - L1：最多取前两个 parallel loop，映射到 gpu.block；
      // - L2：取第一个 reduction loop 做 split（其余维度暂不切分）。
      llvm::SmallVector<int64_t, 8> l1TileSizes;
      llvm::SmallVector<int64_t, 8> l2TileSizes;
      l1TileSizes.assign(bestLoopTileExtents.begin(), bestLoopTileExtents.end());
      l2TileSizes.assign(bestLoopTileExtents.begin(), bestLoopTileExtents.end());

      int64_t pSeen = 0;
      int64_t rSeen = 0;
      for (size_t i = 0; i < genericProbOpt->loops.size(); ++i) {
        auto type = genericProbOpt->loops[i].type;
        if (type == mlir::utils::IteratorType::reduction) {
          // L1 阶段不切 reduction。
          l1TileSizes[i] = 0;
          // L2 阶段只切第一个 reduction。
          if (rSeen > 0)
            l2TileSizes[i] = 0;
          ++rSeen;
          continue;
        }

        // 其它（parallel/window/…）都按“空间维”对待。
        // L1 阶段最多切两维，其余置 0（避免生成 >2D 的 forall 无法映射到 block）。
        if (pSeen >= 2)
          l1TileSizes[i] = 0;
        ++pSeen;
        // L2 阶段不切空间维。
        l2TileSizes[i] = 0;
      }

      // Phase 11（generic fusion，第一版）：
      // - C++ 侧找到 “producer 的第一个 linalg consumer”，并打上 welder.consumer 锚点；
      // - transform 侧对 consumer 做 tile（block 级），再把 producer fuse 进 forall。
      bool doFusion = enableGenericFusion;
      llvm::SmallVector<int64_t, 8> consumerTileSizes;
      StringRef consumerOpName;
      if (doFusion) {
        Operation *consumer = findFirstLinalgConsumer(genericProbOpt->targetOp);
        auto consumerLinalg = dyn_cast_or_null<linalg::LinalgOp>(consumer);
        if (!consumerLinalg || consumerLinalg.getNumReductionLoops() != 0) {
          llvm::errs()
              << "note: --enable-generic-fusion=true, but a suitable elementwise "
                 "linalg consumer was not found; fallback to non-fusion generic "
                 "tiling.\n";
          doFusion = false;
        } else {
          consumerLinalg->setAttr("welder.consumer", UnitAttr::get(&ctx));
          consumerOpName = consumerLinalg->getName().getStringRef();

          // consumer 的 tile sizes：只切前两维 parallel（复用 best.tileM/tileN），其余置 0。
          int64_t nLoops = consumerLinalg.getNumLoops();
          consumerTileSizes.assign(nLoops, 0);
          int64_t pSeen = 0;
          auto iters = consumerLinalg.getIteratorTypesArray();
          auto ranges = consumerLinalg.getStaticLoopRanges();
          for (int64_t i = 0; i < nLoops; ++i) {
            if (iters[i] != mlir::utils::IteratorType::parallel)
              continue;
            int64_t full = ranges[i];
            int64_t t = 0;
            if (pSeen == 0)
              t = best.tileM;
            else if (pSeen == 1)
              t = best.tileN;
            else
              t = 0;
            if (t != 0) {
              if (full == ShapedType::kDynamic || t <= 0 || t > full ||
                  (solveOpts.requirePerfectTiling && (full % t != 0))) {
                doFusion = false;
                break;
              }
            }
            consumerTileSizes[i] = t;
            ++pSeen;
          }

          if (!doFusion) {
            llvm::errs()
                << "note: --enable-generic-fusion=true, but consumer tile sizes "
                   "are invalid under constraints; fallback to non-fusion.\n";
          }
        }
      }

      StringRef targetOpName =
          genericProbOpt->targetOp->getName().getStringRef();
      [[maybe_unused]] auto span =
          tracerPtr ? tracerPtr->span("compiler.build_transform_library.generic")
                    : welder::Tracer::Span();
      transformLib = buildGenericTransformLibrary(
          &ctx, targetOpName, l1TileSizes, l2TileSizes, doFusion, consumerOpName,
          consumerTileSizes, swapBlockDims, gSkipMapForallToBlocks);
    }
    return 0;

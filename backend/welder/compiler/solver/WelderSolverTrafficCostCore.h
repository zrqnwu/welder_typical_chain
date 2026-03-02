static int64_t alignUpI64(int64_t x, int64_t a) {
  if (a <= 0)
    return x;
  int64_t r = x % a;
  if (r == 0)
    return x;
  return x + (a - r);
}

struct BestFitAllocator {
  struct Block {
    int64_t offset = 0;
    int64_t size = 0;
  };

  int64_t highWatermark = 0;
  llvm::SmallVector<Block, 32> freeList;

  int64_t allocate(int64_t size, int64_t alignment) {
    if (size <= 0)
      return 0;
    size = alignUpI64(size, alignment);

    int bestIdx = -1;
    int64_t bestSize = 0;
    for (int i = 0; i < static_cast<int>(freeList.size()); ++i) {
      if (freeList[i].size < size)
        continue;
      if (bestIdx == -1 || freeList[i].size < bestSize) {
        bestIdx = i;
        bestSize = freeList[i].size;
      }
    }

    if (bestIdx != -1) {
      Block b = freeList[bestIdx];
      // 从头分配（不考虑 offset 对齐；这里 size 已对齐，且 free block 由同一对齐产生）。
      int64_t off = b.offset;
      b.offset += size;
      b.size -= size;
      if (b.size == 0) {
        freeList.erase(freeList.begin() + bestIdx);
      } else {
        freeList[bestIdx] = b;
      }
      return off;
    }

    int64_t off = highWatermark;
    highWatermark += size;
    return off;
  }

  void free(int64_t offset, int64_t size, int64_t alignment) {
    if (size <= 0)
      return;
    size = alignUpI64(size, alignment);

    // 插入并尝试按 offset 合并（减少碎片对 best-fit 的影响）。
    Block b{offset, size};
    freeList.push_back(b);
    llvm::sort(freeList, [](const Block &a, const Block &b) {
      return a.offset < b.offset;
    });

    llvm::SmallVector<Block, 32> merged;
    merged.reserve(freeList.size());
    for (const Block &cur : freeList) {
      if (merged.empty()) {
        merged.push_back(cur);
        continue;
      }
      Block &last = merged.back();
      if (last.offset + last.size == cur.offset) {
        last.size += cur.size;
      } else {
        merged.push_back(cur);
      }
    }
    freeList = std::move(merged);
  }
};

static llvm::DenseSet<const void *>
inferMatmulOperandKeysForPadLastDim(const TileGraph &graph,
                                    const PaperSubgraph &sg,
                                    int minLevelExclusive,
                                    int maxLevelInclusive) {
  llvm::DenseSet<const void *> keys;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    Operation *op0 = graph.nodes[nodeIdx].op;
    auto mm = dyn_cast_or_null<linalg::MatmulOp>(op0);
    if (!mm)
      continue;

    // 外部输入：使用 consumer 侧 operand 的 Value 身份作为键。
    auto inputs = mm.getDpsInputs();
    for (int operandIdx = 0; operandIdx < static_cast<int>(inputs.size());
         ++operandIdx) {
      if (operandIdx > 1)
        break;
      Value v = inputs[operandIdx];
      if (v)
        keys.insert(v.getAsOpaquePointer());
    }

    // 参考策略里 Matmul 输出（C）同样携带 stride 偏移信息。
    // Include the dpsInit/output buffer so layout 传播 matches TCPolicy.
    auto inits = mm.getDpsInits();
    for (int initIdx = 0; initIdx < static_cast<int>(inits.size()); ++initIdx) {
      Value v = inits[initIdx];
      if (v)
        keys.insert(v.getAsOpaquePointer());
    }

    // 已连接的 producer 边（位于当前调度子图内）。
    for (int edgeIdx : graph.nodes[nodeIdx].inEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
        continue;
      if (!sg.inSet.contains(e.src))
        continue;
      if (e.dstOperand == 0 || e.dstOperand == 1)
        keys.insert(e.value.getAsOpaquePointer());
    }

    // 已连接的 consumer 边（matmul 输出流向 shared）。
    for (int edgeIdx : graph.nodes[nodeIdx].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
        continue;
      if (!sg.inSet.contains(e.dst))
        continue;
      keys.insert(e.value.getAsOpaquePointer());
    }
  }
  return keys;
}

struct SharedLayoutPolicyV1 {
  int64_t padLastDim = 0;
  bool padMatmulOnly = false;
  int64_t swizzleXor = 0;
  llvm::DenseSet<const void *> matmulOperandKeys;

  SharedLayoutInfo get(const void *key) const {
    SharedLayoutInfo info;
    if (padLastDim > 0 && (!padMatmulOnly || matmulOperandKeys.contains(key)))
      info.padLastDim = padLastDim;
    if (swizzleXor > 1)
      info.swizzleXor = swizzleXor;
    return info;
  }
};

static SharedLayoutPolicyV1
buildSharedLayoutPolicyV1(const TileGraph &graph, const PaperSubgraph &sg,
                          int minLevelExclusive, int maxLevelInclusive,
                          int64_t workgroupPadLastDim,
                          bool workgroupPadLastDimMatmulOnly,
                          int64_t workgroupSwizzleXor) {
  SharedLayoutPolicyV1 policy;
  policy.padLastDim = std::max<int64_t>(0, workgroupPadLastDim);
  policy.padMatmulOnly = workgroupPadLastDimMatmulOnly;
  policy.swizzleXor = std::max<int64_t>(0, workgroupSwizzleXor);
  if (policy.padLastDim > 0 && policy.padMatmulOnly) {
    policy.matmulOperandKeys = inferMatmulOperandKeysForPadLastDim(
        graph, sg, minLevelExclusive, maxLevelInclusive);
  }
  return policy;
}

static int64_t computeSharedFootprintBestFitPaper(const TileGraph &graph,
                                                  const PaperSubgraph &sg,
                                                  const ArchConfig &arch,
                                                  const FootprintInference &inference,
                                                  bool requirePerfectTiling,
                                                  int minLevelExclusive,
                                                  int maxLevelInclusive,
                                                  int64_t workgroupPadLastDim,
                                                  bool workgroupPadLastDimMatmulOnly,
                                                  int64_t workgroupMultiBufferDepth,
                                                  const Candidate *cand) {
  // 论文 §3.1：MemFootprint 用 bestfit 在拓扑序上分配/释放 tiles。
  //
  // 当前实现（global<->shared 对齐版）：
  // - 内部 reuse tile：connectLevel>minLevelExclusive 的边（通常为 connectLevel>0）；
  //   若 maxLevelInclusive>=0，则只统计 connectLevel<=maxLevelInclusive 的边
  //   （用于区分 shared vs register 的复用层级）。
  // - 外部输入 tile：dpsInputs 中“没有来自子图内连接边”的 operand（图外或 cut-edge）；
  // - trivial op（fill/copy）跳过，避免污染 shared footprint。

  double elemBytesD = static_cast<double>(arch.elementBytes);
  SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
      graph, sg, minLevelExclusive, maxLevelInclusive, workgroupPadLastDim,
      workgroupPadLastDimMatmulOnly, /*workgroupSwizzleXor=*/0);
  int64_t multiDepth = std::max<int64_t>(1, workgroupMultiBufferDepth);

  auto getPaddedVolume = [&](const OperandFootprint &fp, const void *key) -> double {
    if (fp.shape.empty())
      return 0.0;
    int64_t pad = layout.get(key).padLastDim;
    double v = 1.0;
    for (size_t i = 0; i < fp.shape.size(); ++i) {
      int64_t dim = fp.shape[i];
      if (dim <= 0)
        return 0.0;
      if (pad > 0 && i + 1 == fp.shape.size())
        dim += pad;
      v *= static_cast<double>(dim);
    }
    return v;
  };

  // 1) 计算子图内部拓扑序（只看 connectLevel>minLevelExclusive 的边作为“同 kernel 内依赖”）。
  llvm::SmallVector<int, 16> topo =
      topoSortSubgraphByConnectedEdges(graph, sg, minLevelExclusive);

  // 2) 统计 buffer size 和 use count。
  llvm::DenseMap<const void *, int64_t> bufferBytes;
  llvm::DenseMap<const void *, int> remainingUses;
  llvm::DenseSet<const void *> externalInputs;

  // 2.1 内部 reuse tiles（producer->consumer 的连接边）。
  for (const TileGraphEdge &e : graph.edges) {
    if (e.connectLevel <= minLevelExclusive)
      continue;
    if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;

    const void *key = e.value.getAsOpaquePointer();
    double vol = getPaddedVolume(e.footprint, key);
    if (vol == 0.0)
      continue;
    // Multi-buffering 只作用于 async global->shared 的 staging buffer
    //（即 external input）。内部复用 tile 在 kernel 内计算，不需要
    // DEPTH 路乒乓存储。
    int64_t bytes = static_cast<int64_t>(vol * elemBytesD);
    if (bytes <= 0)
      continue;

    auto it = bufferBytes.find(key);
    if (it == bufferBytes.end())
      bufferBytes.insert({key, bytes});
    else
      it->second = std::max<int64_t>(it->second, bytes);

    remainingUses[key] += 1;
  }

  // 2.2 外部输入 tiles（按 operand 的 SSA value 去重）。
  for (int n : sg.nodes) {
    Operation *op0 = graph.nodes[n].op;
    if (!op0)
      continue;
    if (isTrivialOpFor2LevelFootprint(op0))
      continue;
    if (!graph.nodes[n].hasRequiredTile)
      continue;

    auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
    if (!op)
      continue;
    auto fpOpt = inference.infer(op0, graph.nodes[n].requiredTile);
    if (!fpOpt)
      continue;

    int numInputs = op.getNumDpsInputs();
    for (int operandIdx = 0; operandIdx < numInputs; ++operandIdx) {
      // 若此 operand 有来自子图内的“连接边”，则它不是外部输入。
      bool hasConnectedInEdge = false;
      for (int edgeIdx : graph.nodes[n].inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (e.dstOperand != operandIdx)
          continue;
        if (e.connectLevel <= minLevelExclusive)
          continue;
        if (sg.inSet.contains(e.src)) {
          hasConnectedInEdge = true;
          break;
        }
      }
      if (hasConnectedInEdge)
        continue;

      if (operandIdx < 0 || operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
        continue;

      Value v = op.getDpsInputs()[operandIdx];
      const void *key = v.getAsOpaquePointer();
      double vol = getPaddedVolume(fpOpt->perOperand[operandIdx], key);
      if (vol == 0.0)
        continue;
      int64_t bytes = static_cast<int64_t>(vol * elemBytesD) * multiDepth;
      if (bytes <= 0)
        continue;

      externalInputs.insert(key);
      auto it = bufferBytes.find(key);
      if (it == bufferBytes.end())
        bufferBytes.insert({key, bytes});
      else
        it->second = std::max<int64_t>(it->second, bytes);

      remainingUses[key] += 1;
    }
  }

  // 3) bestfit 分配（在 topo 序列中 allocate/free）。
  BestFitAllocator alloc;
  llvm::DenseMap<const void *, int64_t> liveOffset;

  constexpr int64_t kAlign = 32; // paper §4.2 提到 32B 对齐；这里统一使用 32B。

  auto isRowWiseReductionOp = [&](Operation *op) -> bool {
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
  };

  auto getBlockDimXFromCandidate = [&]() -> int64_t {
    if (!cand)
      return 0;
    if (cand->tileM <= 0 || cand->tileN <= 0)
      return 0;
    if (cand->threadTileM <= 0 || cand->threadTileN <= 0)
      return 0;
    int64_t xTile = cand->swapBlockDims ? cand->tileM : cand->tileN;
    int64_t xThreadTile = cand->swapBlockDims ? cand->threadTileM
                                              : cand->threadTileN;
    if (xThreadTile <= 0 || xTile <= 0)
      return 0;
    if (xTile % xThreadTile != 0)
      return 0;
    int64_t bx = xTile / xThreadTile;
    if (bx <= 0 || bx > 1024)
      return 0;
    return bx;
  };

  int64_t rowReductionBlockDimX = getBlockDimXFromCandidate();

  auto doAlloc = [&](const void *key) {
    if (liveOffset.count(key))
      return;
    auto it = bufferBytes.find(key);
    if (it == bufferBytes.end())
      return;
    int64_t bytes = it->second;
    int64_t off = alloc.allocate(bytes, kAlign);
    liveOffset.insert({key, off});
  };

  auto doFreeIfDead = [&](const void *key) {
    auto itUse = remainingUses.find(key);
    if (itUse == remainingUses.end())
      return;
    int n = --itUse->second;
    if (n != 0)
      return;
    auto itOff = liveOffset.find(key);
    if (itOff == liveOffset.end())
      return;
    auto itBytes = bufferBytes.find(key);
    if (itBytes == bufferBytes.end())
      return;
    alloc.free(itOff->second, itBytes->second, kAlign);
    liveOffset.erase(itOff);
  };

  // 3.1 LoadTiles：先分配所有外部输入 tiles（符合论文 Figure 8 的执行模型）。
  for (const void *key : externalInputs)
    doAlloc(key);

  // 3.2 逐节点执行：producer allocate + consumer free
  for (int nodeIdx : topo) {
    // 分配该节点产出的内部 tile（connectLevel > minLevelExclusive）。
    for (int edgeIdx : graph.nodes[nodeIdx].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (!sg.inSet.contains(e.dst))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;
      const void *key = e.value.getAsOpaquePointer();
      doAlloc(key);
    }

    // 论文/Welder 对齐: row-wise 归约 lower to a 2D scratch buffer
    // [rows, threads_x] in shared 内存. Model its lifetime as "within this op"
    // 这样 best-fit 复用可以与更早释放的缓冲重叠，
    // 但不会与仍存活的 operand/result 重叠。
    int64_t scratchOff = 0;
    int64_t scratchBytes = 0;
    if (rowReductionBlockDimX > 0 && maxLevelInclusive >= 1 &&
        isRowWiseReductionOp(graph.nodes[nodeIdx].op) &&
        graph.nodes[nodeIdx].hasRequiredTile &&
        !graph.nodes[nodeIdx].requiredTile.loopExtents.empty()) {
      int64_t rows = graph.nodes[nodeIdx].requiredTile.loopExtents[0];
      if (rows > 0) {
        double vol = static_cast<double>(rows) *
                     static_cast<double>(rowReductionBlockDimX);
        scratchBytes = static_cast<int64_t>(vol * elemBytesD);
        if (scratchBytes > 0) {
          scratchOff = alloc.allocate(scratchBytes, kAlign);
        }
      }
    }

    // 输入 tile 在使用后释放。
    Operation *op0 = graph.nodes[nodeIdx].op;
    if (op0 && !isTrivialOpFor2LevelFootprint(op0)) {
      auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
      if (op && graph.nodes[nodeIdx].hasRequiredTile) {
        auto fpOpt = inference.infer(op0, graph.nodes[nodeIdx].requiredTile);
        if (fpOpt) {
          int numInputs = op.getNumDpsInputs();
          for (int operandIdx = 0; operandIdx < numInputs; ++operandIdx) {
            bool hasConnectedInEdge = false;
            for (int eidx : graph.nodes[nodeIdx].inEdges) {
              if (eidx < 0 || eidx >= static_cast<int>(graph.edges.size()))
                continue;
              const TileGraphEdge &e = graph.edges[eidx];
              if (e.dstOperand != operandIdx)
                continue;
              if (e.connectLevel <= minLevelExclusive)
                continue;
              if (sg.inSet.contains(e.src)) {
                hasConnectedInEdge = true;
                break;
              }
            }
            if (hasConnectedInEdge)
              continue;
            Value v = op.getDpsInputs()[operandIdx];
            doFreeIfDead(v.getAsOpaquePointer());
          }
        }
      }
    }

    for (int edgeIdx : graph.nodes[nodeIdx].inEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (!sg.inSet.contains(e.src))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;
      const void *key = e.value.getAsOpaquePointer();
      doFreeIfDead(key);
    }

    if (scratchBytes > 0) {
      alloc.free(scratchOff, scratchBytes, kAlign);
    }
  }

  // 3.3 末尾释放所有剩余（不影响 peak）。
  // 注意： bestfit 的 highWatermark 即为 footprint。
  return alloc.highWatermark;
}

static Traffic computeGlobalTrafficForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, bool applyCoalescingPenalty,
    const SharedLayoutPolicyV1 *layout);

struct MemTrafficBreakdown {
  Traffic raw;
  Traffic mem;
};

static MemTrafficBreakdown computeMemTrafficForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, bool applyCoalescingPenalty,
    const SharedLayoutPolicyV1 *layout) {
  MemTrafficBreakdown out;
  out.raw = computeGlobalTrafficForSubgraph(graph, sg, arch, inference,
                                           requirePerfectTiling, minLevelExclusive,
                                           /* applyCoalescingPenalty=*/false,
                                           layout);
  out.mem = computeGlobalTrafficForSubgraph(graph, sg, arch, inference,
                                           requirePerfectTiling, minLevelExclusive,
                                           /* applyCoalescingPenalty=*/applyCoalescingPenalty,
                                           layout);
  return out;
}

// （定义已移到 GridInfo 辅助函数附近，这里仅保留前置声明。）

static void fillCostAndScoreFromPaperModel(Candidate &cand,
                                           int64_t sharedFootprintBytes,
                                           const MemTrafficBreakdown &mt,
                                           double sharedToRegBytes,
                                           const SolveOptions &opts) {
  CostBreakdown cb;
  cb.rawTraffic = mt.raw;
  cb.memTraffic = mt.mem;
  cb.sharedToRegBytes = std::max(0.0, sharedToRegBytes);
  cb.sharedFootprintBytes = std::max<int64_t>(0, sharedFootprintBytes);
  cb.waves = std::max<int64_t>(1, cand.numWave);
  cb.blocksTotal = std::max<int64_t>(1, cand.blocksTotal);
  cb.blocksPerSM = std::max<int64_t>(1, cand.blocksPerSM);

  int64_t concurrentBlocks =
      std::max<int64_t>(1, cb.blocksPerSM * std::max<int64_t>(1, opts.arch.numSM));
  if (cb.blocksTotal < concurrentBlocks) {
    cb.underutilPenalty =
        static_cast<double>(concurrentBlocks) /
        static_cast<double>(std::max<int64_t>(1, cb.blocksTotal));
  }

  cb.bankConflictFactor = std::max(1.0, cand.estSharedBankConflict);

  // Penalize high register pressure and low 占用率 to reduce selection of
  // fragile fused kernels (especially f16 TensorCore + 归约-chain cases).
  cb.regPenalty = 1.0;
  if (opts.arch.maxRegistersPerThread > 0 && cand.estRegsPerThread > 0) {
    double regRatio = static_cast<double>(cand.estRegsPerThread) /
                      static_cast<double>(opts.arch.maxRegistersPerThread);
    regRatio = std::max(0.0, regRatio);
    cb.regPenalty *= 1.0 + std::max(0.0, regRatio - 0.33) * 1.35;
    if (cand.enableSoftwarePipelining)
      cb.regPenalty *= 1.0 + std::max(0.0, regRatio - 0.28) * 0.80;
    if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
        regRatio > 0.45) {
      cb.regPenalty *= 1.0 + (regRatio - 0.45) * 0.60;
    }
  }

  if (opts.arch.maxBlocksPerSM > 0) {
    double occRatio =
        static_cast<double>(std::max<int64_t>(1, cb.blocksPerSM)) /
        static_cast<double>(opts.arch.maxBlocksPerSM);
    occRatio = std::max(0.0, std::min(1.0, occRatio));
    if (occRatio < 0.5)
      cb.underutilPenalty *= 1.0 + (0.5 - occRatio) * 1.60;
    if (cand.enableSoftwarePipelining && occRatio < 0.5)
      cb.regPenalty *= 1.15;
    if ((cand.enableTensorCoreF16 || cand.enableTensorCoreTf32) &&
        occRatio < 0.35) {
      cb.regPenalty *= 1.10;
    }
  }

  // Matmul->Softmax 行链的 anti-spill 先验：
  // 在估计阶段直接近似 spill 风险，让未测量或部分测量场景
  // 更偏向稳定的 kernel 形状。
  const bool mmSmRowChain =
      cand.enableMatmulSoftmaxSharedReuseFusion &&
      cand.enableRowReductionChainReuseFusion;
  if (mmSmRowChain &&
      getEnvInt64OrDefault("WELDER_EST_MM_SM_SPILL_PENALTY_ENABLE", 1) != 0) {
    const bool tcCand = cand.enableTensorCoreF16 || cand.enableTensorCoreTf32;
    double spillPenalty = 1.0;
    if (tcCand)
      spillPenalty *= 1.12;
    if (cand.enableRowReductionInputPromotion)
      spillPenalty *= 1.04;
    if (cand.enableRowReductionInputPromotionVectorize)
      spillPenalty *= 1.08;
    if (cand.enableRowReductionVectorize)
      spillPenalty *= 1.08;
    if (cand.enableRowReductionCombineVectorize)
      spillPenalty *= 1.05;
    if (cand.rowReductionVectorWidth > 4)
      spillPenalty *= 1.06;
    if (cand.rowReductionInputVectorWidth > 4)
      spillPenalty *= 1.06;
    if (cand.enableSoftwarePipelining)
      spillPenalty *= tcCand ? 1.22 : 1.12;
    if (cand.enableSoftwarePipelining && cand.pipelineSetAsyncWaitGroups)
      spillPenalty *= 1.04;
    if (cand.blocksPerSM <= 1)
      spillPenalty *= tcCand ? 1.28 : 1.16;
    else if (cand.blocksPerSM == 2)
      spillPenalty *= tcCand ? 1.14 : 1.08;
    if (cand.estRegsPerThread > 0) {
      int64_t softRegs = tcCand ? 96 : 112;
      if (cand.estRegsPerThread > softRegs) {
        double over =
            static_cast<double>(cand.estRegsPerThread - softRegs) /
            static_cast<double>(std::max<int64_t>(1, softRegs));
        spillPenalty *= 1.0 + std::max(0.0, over) * 0.70;
      }
    }
    double maxPenalty = std::max(
        1.0, getEnvDoubleOrDefault("WELDER_EST_MM_SM_SPILL_PENALTY_MAX",
                                   /*default=*/2.4));
    cb.regPenalty *= std::min(maxPenalty, std::max(1.0, spillPenalty));
  }

  double trafficBytes = opts.enableCoalescingPenalty ? cb.memTraffic.totalBytes()
                                                     : cb.rawTraffic.totalBytes();
  // 多层级论文语义：当考虑寄存器层复用（connectLevel>1）时，
  // shared<->register 流量会显著影响结果。这里按默认带宽比
  //（global store vs 片上 load）给出保守权重，并并入主估计。
  double shWeight = 0.0625;
  if (opts.arch.bandwidthLoad > 0.0 && opts.arch.bandwidthStore > 0.0)
    shWeight = std::max(0.0, opts.arch.bandwidthStore / opts.arch.bandwidthLoad);
  trafficBytes += cb.sharedToRegBytes * shWeight;
  cb.estimatedLatency = trafficBytes * cb.underutilPenalty *
                        static_cast<double>(cb.waves) *
                        cb.bankConflictFactor * cb.regPenalty;

  cand.cost = cb;
  cand.traffic = opts.enableCoalescingPenalty ? cb.memTraffic : cb.rawTraffic;
  cand.score = cb.estimatedLatency;
}

static Traffic computeGlobalTrafficForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, bool applyCoalescingPenalty,
    const SharedLayoutPolicyV1 *layout) {
  // 基于 Phase A 的全图记账逻辑做子图版本：
  // - “连接边” = connectLevel>minLevelExclusive；
  // - cut-edge（connectLevel<=minLevelExclusive）在子图内部视为 global 落地，需要计入读写。

  double totalReadBytes = 0.0;
  double totalWriteBytes = 0.0;
  double totalCutBytes = 0.0;
  double elemBytes = static_cast<double>(arch.elementBytes);
  int64_t txnReadElems = getTxnElemsForRead(arch);
  int64_t txnWriteElems = getTxnElemsForWrite(arch);

  llvm::DenseSet<const void *> seenCutWrites;

  for (int nodeIdx : sg.nodes) {
    const TileGraphNode &node = graph.nodes[nodeIdx];
    if (!node.op || !node.hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(node.op))
      continue;

    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(node.op);
    if (!linalgOp)
      continue;

    // Phase A 风格的 grid/hits：blocksTotal（parallel tiles）+ reductionTiles（K tiles）。
    int64_t blocksTotal = 1;
    int64_t reductionTiles = 1;
    auto iters = linalgOp.getIteratorTypesArray();
    llvm::SmallVector<int64_t, 8> ranges = linalgOp.getStaticLoopRanges();
    if (static_cast<int64_t>(ranges.size()) != linalgOp.getNumLoops())
      continue;
    if (static_cast<int64_t>(iters.size()) != linalgOp.getNumLoops())
      continue;

    for (int64_t i = 0; i < linalgOp.getNumLoops(); ++i) {
      int64_t full = ranges[i];
      if (full == ShapedType::kDynamic || full <= 0)
        continue;
      if (i < 0 || i >= static_cast<int64_t>(node.requiredTile.loopExtents.size()))
        continue;
      int64_t t = node.requiredTile.loopExtents[i];
      if (t <= 0)
        continue;
      int64_t tiles = requirePerfectTiling ? (full / t) : ceilDiv(full, t);
      if (iters[i] == utils::IteratorType::parallel)
        blocksTotal *= tiles;
      else if (iters[i] == utils::IteratorType::reduction)
        reductionTiles *= tiles;
    }

    auto dependsOnReduction = [&](int operandIdx) -> bool {
      auto maps = linalgOp.getIndexingMapsArray();
      if (operandIdx < 0 || operandIdx >= static_cast<int>(maps.size()))
        return false;
      AffineMap m = maps[operandIdx];
      for (AffineExpr e : m.getResults()) {
        for (int dim = 0; dim < static_cast<int>(iters.size()); ++dim) {
          if (iters[dim] == utils::IteratorType::reduction) {
            if (e.isFunctionOfDim(dim))
              return true;
          }
        }
      }
      return false;
    };

    auto fpOpt = inference.infer(node.op, node.requiredTile);
    if (!fpOpt)
      continue;

    int numInputs = linalgOp.getNumDpsInputs();
    int numInits = linalgOp.getNumDpsInits();

    auto footprintBytesRead = [&](const OperandFootprint &fp,
                                  Value fullVal) -> double {
      if (fp.shape.empty())
        return 0.0;
      double elems = getVolume(fp);
      if (elems == 0.0)
        return 0.0;
      if (!applyCoalescingPenalty)
        return elems * elemBytes;
      llvm::SmallVector<int64_t, 4> fullShape = getStaticShapeOrUnknown(fullVal);
      llvm::SmallVector<int64_t, 4> fullStrides = getStaticStridesOrEmpty(fullVal);
      double coalescedElems = (!fullStrides.empty())
                                  ? coalescedTensorElements(fp.shape, fullShape,
                                                            fullStrides,
                                                            txnReadElems)
                                  : coalescedTensorElements(fp.shape, fullShape,
                                                            txnReadElems);
      return coalescedElems * elemBytes;
    };

    auto footprintBytesWrite = [&](const OperandFootprint &fp,
                                   Value fullVal) -> double {
      if (fp.shape.empty())
        return 0.0;
      double elems = getVolume(fp);
      if (elems == 0.0)
        return 0.0;
      if (!applyCoalescingPenalty)
        return elems * elemBytes;
      llvm::SmallVector<int64_t, 4> fullShape = getStaticShapeOrUnknown(fullVal);
      llvm::SmallVector<int64_t, 4> fullStrides = getStaticStridesOrEmpty(fullVal);
      double coalescedElems = (!fullStrides.empty())
                                  ? coalescedTensorElements(fp.shape, fullShape,
                                                            fullStrides,
                                                            txnWriteElems)
                                  : coalescedTensorElements(fp.shape, fullShape,
                                                            txnWriteElems);
      return coalescedElems * elemBytes;
    };

    // 1) global read：图外输入 + cut-edge 输入。
    for (int i = 0; i < numInputs; ++i) {
      bool hasConnectedProducer = false;
      bool hasAnyProducer = false;
      int cutEdgeIdx = -1;
      for (int edgeIdx : node.inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (e.dstOperand != i)
          continue;
        if (!sg.inSet.contains(e.src))
          continue;
        hasAnyProducer = true;
        if (e.connectLevel > minLevelExclusive) {
          hasConnectedProducer = true;
        } else {
          cutEdgeIdx = edgeIdx;
        }
      }

      bool dependsOnK = dependsOnReduction(i);
      int64_t hits =
          blocksTotal * (dependsOnK ? reductionTiles : static_cast<int64_t>(1));

      if (!hasAnyProducer) {
        // 图外输入。
        if (i < 0 || i >= static_cast<int>(fpOpt->perOperand.size()))
          continue;
        Value fullVal = linalgOp.getDpsInputs()[i];
        double bytesPerTile = footprintBytesRead(fpOpt->perOperand[i], fullVal);
        if (bytesPerTile == 0.0)
          continue;
        totalReadBytes += bytesPerTile * static_cast<double>(hits);
        continue;
      }

      if (!hasConnectedProducer && cutEdgeIdx != -1) {
        // 子图内的 cut-edge：consumer 侧需要从 global 读。
        const TileGraphEdge &e = graph.edges[cutEdgeIdx];
        Value fullVal = linalgOp.getDpsInputs()[i];
        double bytesPerTile = footprintBytesRead(e.footprint, fullVal);
        if (bytesPerTile == 0.0)
          continue;
        totalCutBytes += bytesPerTile * static_cast<double>(hits);
      }
    }

    // 2) global read：DPS init 的读取（例如 matmul 里的 C）。
    //
    // Linalg matmul 语义为 `out = init + sum(A*B)`，
    // 因此每个输出 tile（并行 tile）至少读取一次 init；
    // 若 indexing map 依赖归约循环，则读取次数还会相应放大。
    for (int outIdx = 0; outIdx < numInits; ++outIdx) {
      int operandIdx = numInputs + outIdx;
      if (operandIdx < 0 ||
          operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
        continue;

      bool hasConnectedProducer = false;
      bool hasAnyProducer = false;
      int cutEdgeIdx = -1;
      for (int edgeIdx : node.inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (e.dstOperand != operandIdx)
          continue;
        if (!sg.inSet.contains(e.src))
          continue;
        hasAnyProducer = true;
        if (e.connectLevel > minLevelExclusive) {
          hasConnectedProducer = true;
        } else {
          cutEdgeIdx = edgeIdx;
        }
      }

      bool dependsOnK = dependsOnReduction(operandIdx);
      int64_t hits =
          blocksTotal * (dependsOnK ? reductionTiles : static_cast<int64_t>(1));

      Value fullVal = linalgOp.getDpsInits()[outIdx];

      if (!hasAnyProducer) {
        double bytesPerTile =
            footprintBytesRead(fpOpt->perOperand[operandIdx], fullVal);
        if (bytesPerTile == 0.0)
          continue;
        totalReadBytes += bytesPerTile * static_cast<double>(hits);
        continue;
      }

      if (!hasConnectedProducer && cutEdgeIdx != -1) {
        const TileGraphEdge &e = graph.edges[cutEdgeIdx];
        double bytesPerTile = footprintBytesRead(e.footprint, fullVal);
        if (bytesPerTile == 0.0)
          continue;
        totalCutBytes += bytesPerTile * static_cast<double>(hits);
      }
    }

    // 3) global write：子图的“对外输出”（sink 或 cut-edge producer 输出）。
    //
    // - 若该 node 的结果被子图外使用，算一次 write；
    // - 若被子图内 cut-edge consumer 使用，也算一次 write（去重）。
    for (int edgeIdx : node.outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      bool toOutside = !sg.inSet.contains(e.dst);
      bool isCutEdge = (e.connectLevel <= minLevelExclusive);
      if (!toOutside && !isCutEdge)
        continue;

      const void *key = e.value.getAsOpaquePointer();
      if (!seenCutWrites.insert(key).second)
        continue;

      int outIdx = e.srcResult;
      if (outIdx < 0 || outIdx >= numInits)
        continue;
      int operandIdx = numInputs + outIdx;
      if (operandIdx < 0 ||
          operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
        continue;

      Value fullVal = linalgOp.getDpsInits()[outIdx];
      double bytesPerTile =
          footprintBytesWrite(fpOpt->perOperand[operandIdx], fullVal);
      if (bytesPerTile == 0.0)
        continue;
      double bytes = bytesPerTile * static_cast<double>(blocksTotal);
      if (toOutside) {
        totalWriteBytes += bytes;
      } else {
        totalCutBytes += bytes;
      }
    }

    // 4) Graph sink：写回最终结果（与 Phase A baseline 对齐）。
    if (node.outEdges.empty()) {
      for (int outIdx = 0; outIdx < numInits; ++outIdx) {
        int operandIdx = numInputs + outIdx;
        if (operandIdx < 0 ||
            operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
          continue;
        Value fullVal = linalgOp.getDpsInits()[outIdx];
        double bytesPerTile =
            footprintBytesWrite(fpOpt->perOperand[operandIdx], fullVal);
        if (bytesPerTile == 0.0)
          continue;
        totalWriteBytes += bytesPerTile * static_cast<double>(blocksTotal);
      }
    }
  }

  return Traffic{totalReadBytes, 0.0, totalWriteBytes, totalCutBytes};
}

struct PaperScheduleCandidate {
  Candidate cand;
  int64_t sharedFootprintBytes = 0;
  Traffic traffic;
  double estimatedLatency = 0.0;
};

static std::optional<double> getFiniteProfiledMs(const Candidate &cand) {
  if (!cand.cost.profiledMs.has_value())
    return std::nullopt;
  double ms = *cand.cost.profiledMs;
  if (!std::isfinite(ms) || ms < 0.0)
    return std::nullopt;
  return ms;
}

static double normalizeLatencyOrInf(double latency) {
  if (!std::isfinite(latency) || latency < 0.0)
    return std::numeric_limits<double>::infinity();
  return latency;
}

static int compareProfilePriorityLatency(const Candidate &lhsCand,
                                         double lhsFallbackLatency,
                                         const Candidate &rhsCand,
                                         double rhsFallbackLatency) {
  std::optional<double> lhsProfile = getFiniteProfiledMs(lhsCand);
  std::optional<double> rhsProfile = getFiniteProfiledMs(rhsCand);
  if (lhsProfile.has_value() != rhsProfile.has_value())
    return lhsProfile.has_value() ? -1 : 1;

  double lhs = normalizeLatencyOrInf(
      lhsProfile.has_value() ? *lhsProfile : lhsFallbackLatency);
  double rhs = normalizeLatencyOrInf(
      rhsProfile.has_value() ? *rhsProfile : rhsFallbackLatency);

  if (std::isfinite(lhs) && std::isfinite(rhs)) {
    double scale = std::max(1.0, std::max(std::fabs(lhs), std::fabs(rhs)));
    if (std::fabs(lhs - rhs) > 1e-9 * scale)
      return lhs < rhs ? -1 : 1;
    return 0;
  }
  if (std::isfinite(lhs) != std::isfinite(rhs))
    return std::isfinite(lhs) ? -1 : 1;
  return 0;
}

static double getCandidateFallbackLatency(const Candidate &cand) {
  if (std::isfinite(cand.score) && cand.score > 0.0)
    return cand.score;
  if (std::isfinite(cand.cost.estimatedLatency) && cand.cost.estimatedLatency > 0.0)
    return cand.cost.estimatedLatency;
  return std::numeric_limits<double>::infinity();
}

static double
getPaperCandidateSortLatencyProfileFirst(const PaperScheduleCandidate &pc) {
  if (std::optional<double> prof = getFiniteProfiledMs(pc.cand))
    return *prof;
  if (std::isfinite(pc.estimatedLatency) && pc.estimatedLatency > 0.0)
    return pc.estimatedLatency;
  return getCandidateFallbackLatency(pc.cand);
}

static bool betterCandidateByProfilePriority(const Candidate &a,
                                             const Candidate &b) {
  int cmp = compareProfilePriorityLatency(
      a, getCandidateFallbackLatency(a), b, getCandidateFallbackLatency(b));
  if (cmp != 0)
    return cmp < 0;
  if (a.smemBytes != b.smemBytes)
    return a.smemBytes < b.smemBytes;
  if (a.tileM != b.tileM)
    return a.tileM > b.tileM;
  return a.tileN > b.tileN;
}

static bool betterPaperCandidateByProfilePriority(const PaperScheduleCandidate &a,
                                                  const PaperScheduleCandidate &b) {
  int cmp = compareProfilePriorityLatency(
      a.cand, getPaperCandidateSortLatencyProfileFirst(a), b.cand,
      getPaperCandidateSortLatencyProfileFirst(b));
  if (cmp != 0)
    return cmp < 0;
  if (a.cand.blocksPerSM != b.cand.blocksPerSM)
    return a.cand.blocksPerSM > b.cand.blocksPerSM;
  if (a.cand.estRegsPerThread != b.cand.estRegsPerThread)
    return a.cand.estRegsPerThread < b.cand.estRegsPerThread;
  if (a.sharedFootprintBytes != b.sharedFootprintBytes)
    return a.sharedFootprintBytes < b.sharedFootprintBytes;
  return a.cand.score < b.cand.score;
}

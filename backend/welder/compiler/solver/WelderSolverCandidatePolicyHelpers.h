static int64_t getMaxSmemUsageBytes(const ArchConfig &arch) {
  if (arch.maxSmemUsageBytes > 0)
    return arch.maxSmemUsageBytes;
  if (arch.smemBytes > 0 && arch.smemBytes <= (std::numeric_limits<int64_t>::max() / 2))
    return 2 * arch.smemBytes;
  return std::max<int64_t>(1, arch.smemBytes);
}

static int64_t getTxnElemsForRead(const ArchConfig &arch) {
  int64_t bytes = arch.globalReadTransactionBytes > 0
                      ? arch.globalReadTransactionBytes
                      : arch.globalTransactionBytes;
  if (arch.elementBytes <= 0 || bytes <= 0)
    return 1;
  return std::max<int64_t>(1, bytes / arch.elementBytes);
}

static int64_t getTxnElemsForWrite(const ArchConfig &arch) {
  int64_t bytes = arch.globalWriteTransactionBytes > 0
                      ? arch.globalWriteTransactionBytes
                      : arch.globalTransactionBytes;
  if (arch.elementBytes <= 0 || bytes <= 0)
    return 1;
  return std::max<int64_t>(1, bytes / arch.elementBytes);
}

static void chooseMmaShapeForCandidate(Candidate &cand) {
  cand.mmaM = 0;
  cand.mmaN = 0;
  cand.mmaK = 0;

  if (cand.enableTensorCoreF16) {
    const int64_t k = 16;
    // 注意：本仓库当前使用的 MLIR
    // `transform.nvgpu.rewrite_matmul_as_mma_sync` 路径，
    // 稳定支持的是 cutlass 风格的 `m16n8k16`。
    // 参考策略中的其它 warp 形状（m16n16/m32n8/m8n32）在该 MLIR 版本
    // 下尚不稳定，因此这里保持 codegen 对齐 `m16n8k16`。
    if (cand.tileM > 0 && cand.tileN > 0 && cand.tileK > 0 &&
        (cand.tileM % 16) == 0 && (cand.tileN % 8) == 0 &&
        (cand.tileK % k) == 0) {
      cand.useCutlassMma = true;
      cand.mmaM = 16;
      cand.mmaN = 8;
      cand.mmaK = k;
      return;
    }
    return;
  }

  if (cand.enableTensorCoreTf32) {
    // 本仓库 transform.nvgpu 流水线使用的最小 TF32 MMA 形状。
    const int64_t k = 4;
    if (cand.tileM > 0 && cand.tileN > 0 && cand.tileK > 0 &&
        (cand.tileM % 16) == 0 && (cand.tileN % 8) == 0 &&
        (cand.tileK % k) == 0) {
      cand.mmaM = 16;
      cand.mmaN = 8;
      cand.mmaK = k;
    }
  }
}

struct PaperSubgraph {
  llvm::SmallVector<int, 16> nodes;
  llvm::DenseSet<int> inSet;
};

static bool isRowWiseReduction2DOp(Operation *op, int64_t *outReductionExtent = nullptr) {
  auto gen = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!gen)
    return false;
  if (gen.getNumLoops() != 2 || gen.getNumReductionLoops() != 1)
    return false;
  auto iters = gen.getIteratorTypesArray();
  if (iters.size() != 2 ||
      iters[0] != mlir::utils::IteratorType::parallel ||
      iters[1] != mlir::utils::IteratorType::reduction)
    return false;
  if (outReductionExtent) {
    int64_t extent = 1;
    llvm::SmallVector<int64_t, 4> ranges = gen.getStaticLoopRanges();
    if (ranges.size() == 2 && ranges[1] != ShapedType::kDynamic && ranges[1] > 0)
      extent = ranges[1];
    * outReductionExtent = extent;
  }
  return true;
}

[[maybe_unused]] static int64_t
computeMaxRowReductionExtentForGraph(const TileGraph &graph) {
  int64_t maxExtent = 1;
  for (const TileGraphNode &node : graph.nodes) {
    int64_t extent = 1;
    if (!isRowWiseReduction2DOp(node.op, &extent))
      continue;
    maxExtent = std::max<int64_t>(maxExtent, extent);
  }
  return std::max<int64_t>(1, maxExtent);
}

static int64_t computeMaxRowReductionExtentForSubgraph(const TileGraph &graph,
                                                       const PaperSubgraph &sg) {
  int64_t maxExtent = 1;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    int64_t extent = 1;
    if (!isRowWiseReduction2DOp(graph.nodes[nodeIdx].op, &extent))
      continue;
    maxExtent = std::max<int64_t>(maxExtent, extent);
  }
  return std::max<int64_t>(1, maxExtent);
}

static bool subgraphHasMatmulOp(const TileGraph &graph,
                                const PaperSubgraph &sg) {
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    if (isa<linalg::MatmulOp>(graph.nodes[nodeIdx].op))
      return true;
  }
  return false;
}

static bool subgraphHasRowReduction2DOp(const TileGraph &graph,
                                        const PaperSubgraph &sg,
                                        int64_t *outMaxExtent = nullptr) {
  int64_t maxExtent = 1;
  bool has = false;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    int64_t extent = 1;
    if (!isRowWiseReduction2DOp(graph.nodes[nodeIdx].op, &extent))
      continue;
    has = true;
    maxExtent = std::max<int64_t>(maxExtent, extent);
  }
  if (outMaxExtent)
    * outMaxExtent = std::max<int64_t>(1, maxExtent);
  return has;
}

static bool isMatmulSoftmaxLikeSubgraph(const TileGraph &graph,
                                        const PaperSubgraph &sg) {
  return subgraphHasMatmulOp(graph, sg) &&
         subgraphHasRowReduction2DOp(graph, sg, nullptr);
}

static bool isF16ScalarOrElementType(Type type) {
  if (!type)
    return false;
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  auto floatType = dyn_cast<FloatType>(type);
  return floatType && floatType.isF16();
}

static bool subgraphHasF16Type(const TileGraph &graph, const PaperSubgraph &sg) {
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    auto op = dyn_cast_or_null<linalg::LinalgOp>(graph.nodes[nodeIdx].op);
    if (!op)
      continue;
    for (Value operand : op->getOperands()) {
      if (isF16ScalarOrElementType(operand.getType()))
        return true;
    }
    for (Type resultType : op->getResultTypes()) {
      if (isF16ScalarOrElementType(resultType))
        return true;
    }
  }
  return false;
}

static int64_t computeEffectivePaperSolveTopK(const TileGraph &graph,
                                              const PaperSubgraph &sg,
                                              const SolveOptions &opts) {
  int64_t topK = opts.scheduleTopK;
  if (topK <= 0)
    return topK;
  if (!isMatmulSoftmaxLikeSubgraph(graph, sg))
    return topK;
  if (!subgraphHasF16Type(graph, sg))
    return topK;
  const bool wantTensorCoreF16 =
      (opts.profile.enable && opts.profile.enableTensorCoreF16) ||
      (opts.codegenSearch.enable &&
       llvm::is_contained(opts.codegenSearch.enableTensorCoreF16, true));
  if (!wantTensorCoreF16)
    return topK;
  const int64_t minTopKForF16MatmulSoftmax = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_MM_SM_F16_MIN_SOLVER_TOPK",
                              opts.profile.enable ? 8 : 6));
  return std::max<int64_t>(topK, minTopKForF16MatmulSoftmax);
}

static bool graphHasMatmulSoftmaxLikePattern(const TileGraph &graph) {
  bool hasMatmul = false;
  bool hasRowReduction = false;
  for (const TileGraphNode &node : graph.nodes) {
    if (!hasMatmul && isa<linalg::MatmulOp>(node.op))
      hasMatmul = true;
    if (!hasRowReduction && isRowWiseReduction2DOp(node.op, nullptr))
      hasRowReduction = true;
    if (hasMatmul && hasRowReduction)
      return true;
  }
  return false;
}

static bool
isTensorCoreStrideLayoutFeasibleForSubgraph(const TileGraph &graph,
                                            const PaperSubgraph &sg) {
  const bool enablePrecheck =
      getEnvInt64OrDefault("WELDER_TC_LAYOUT_PRECHECK", 1) != 0;
  if (!enablePrecheck)
    return true;
  const bool requireStatic =
      getEnvInt64OrDefault("WELDER_TC_LAYOUT_REQUIRE_STATIC", 1) != 0;
  const bool requireInnerContiguous =
      getEnvInt64OrDefault("WELDER_TC_LAYOUT_REQUIRE_INNER_CONTIGUOUS", 1) != 0;
  const bool requireOuterStrideBound = getEnvInt64OrDefault(
                                           "WELDER_TC_LAYOUT_REQUIRE_OUTER_STRIDE_BOUND",
                                           1) != 0;
  const bool requireKDimAlign =
      getEnvInt64OrDefault("WELDER_TC_LAYOUT_REQUIRE_K_DIM_ALIGN", 1) != 0;
  const bool requireNDimAlign =
      getEnvInt64OrDefault("WELDER_TC_LAYOUT_REQUIRE_N_DIM_ALIGN", 1) != 0;
  const int64_t f16DimAlign = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_TC_LAYOUT_F16_DIM_ALIGN", 8));
  const int64_t tf32DimAlign = std::max<int64_t>(
      1, getEnvInt64OrDefault("WELDER_TC_LAYOUT_TF32_DIM_ALIGN", 4));
  const int64_t f16OuterStrideAlign =
      std::max<int64_t>(1, getEnvInt64OrDefault(
                               "WELDER_TC_LAYOUT_F16_OUTER_STRIDE_ALIGN", 8));
  const int64_t tf32OuterStrideAlign =
      std::max<int64_t>(1, getEnvInt64OrDefault(
                               "WELDER_TC_LAYOUT_TF32_OUTER_STRIDE_ALIGN", 4));

  auto isOperandTcFriendly = [&](Value v, int64_t innerDimAlign,
                                 int64_t outerStrideAlign) -> bool {
    auto st = dyn_cast<ShapedType>(v.getType());
    if (!st || !st.hasRank() || st.getRank() != 2)
      return !requireStatic;

    const int64_t dimM = st.getDimSize(0);
    const int64_t dimN = st.getDimSize(1);
    if (requireStatic &&
        (dimM == ShapedType::kDynamic || dimN == ShapedType::kDynamic))
      return false;

    llvm::SmallVector<int64_t, 4> strides = getStaticStridesOrEmpty(v);
    if (strides.size() != 2)
      return !requireStatic;

    if (requireInnerContiguous && strides[1] != 1)
      return false;
    if (innerDimAlign > 1 && dimN != ShapedType::kDynamic && dimN > 0 &&
        (dimN % innerDimAlign) != 0)
      return false;
    if (outerStrideAlign > 1 && strides[0] > 0 &&
        (strides[0] % outerStrideAlign) != 0)
      return false;
    if (requireOuterStrideBound && dimN != ShapedType::kDynamic && dimN > 0 &&
        strides[0] > 0 && strides[1] > 0 &&
        strides[0] < dimN * strides[1])
      return false;
    return true;
  };

  bool sawMatmul = false;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    auto matmul = dyn_cast_or_null<linalg::MatmulOp>(graph.nodes[nodeIdx].op);
    if (!matmul)
      continue;
    sawMatmul = true;
    Value a = matmul.getDpsInputOperand(0)->get();
    Value b = matmul.getDpsInputOperand(1)->get();
    Value c = matmul.getDpsInitOperand(0)->get();

    auto aTy = dyn_cast<ShapedType>(a.getType());
    auto bTy = dyn_cast<ShapedType>(b.getType());
    auto cTy = dyn_cast<ShapedType>(c.getType());
    const bool f16Matmul =
        isF16ScalarOrElementType(a.getType()) || isF16ScalarOrElementType(b.getType()) ||
        isF16ScalarOrElementType(c.getType());
    const int64_t dimAlign = f16Matmul ? f16DimAlign : tf32DimAlign;
    const int64_t outerStrideAlign =
        f16Matmul ? f16OuterStrideAlign : tf32OuterStrideAlign;

    if (!isOperandTcFriendly(a, dimAlign, outerStrideAlign) ||
        !isOperandTcFriendly(b, dimAlign, outerStrideAlign) ||
        !isOperandTcFriendly(c, dimAlign, outerStrideAlign))
      return false;
    if (requireStatic && aTy && bTy && aTy.hasRank() && bTy.hasRank() &&
        aTy.getRank() == 2 && bTy.getRank() == 2) {
      const int64_t aK = aTy.getDimSize(1);
      const int64_t bK = bTy.getDimSize(0);
      if (aK != ShapedType::kDynamic && bK != ShapedType::kDynamic && aK > 0 &&
          bK > 0 && aK != bK)
        return false;
      if (requireKDimAlign && dimAlign > 1 && aK != ShapedType::kDynamic &&
          aK > 0 && (aK % dimAlign) != 0)
        return false;
      const int64_t bN = bTy.getDimSize(1);
      if (requireNDimAlign && dimAlign > 1 && bN != ShapedType::kDynamic &&
          bN > 0 && (bN % dimAlign) != 0)
        return false;
      if (cTy && cTy.hasRank() && cTy.getRank() == 2) {
        const int64_t cN = cTy.getDimSize(1);
        if (requireNDimAlign && dimAlign > 1 && cN != ShapedType::kDynamic &&
            cN > 0 && (cN % dimAlign) != 0)
          return false;
      }
    }
  }
  (void)sawMatmul;
  return true;
}

static int64_t computeTcRowReductionExtentForThreadMapping(
    const TileGraph &graph, const PaperSubgraph &sg) {
  int64_t rrExtent = 1;
  const bool hasRowReduction =
      subgraphHasRowReduction2DOp(graph, sg, &rrExtent);
  if (!hasRowReduction)
    return 1;
  // Matmul->Softmax 风格融合子图不应直接使用完整行归约 extent
  // （常见可达 128/256），否则会过度剪枝 TensorCore 候选。
  // 但归约链 codegen 仍需足够的 X 方向线程并行度，因此这里对 TC 的
  // thread mapping 使用有上限的 extent。
  if (subgraphHasMatmulOp(graph, sg)) {
    constexpr int64_t kTcRowThreadExtentCap = 64;
    return std::max<int64_t>(
        1, std::min<int64_t>(rrExtent, kTcRowThreadExtentCap));
  }
  return std::max<int64_t>(1, rrExtent);
}

static bool isTwoDimElementwiseOrRowReductionGeneric(Operation *op,
                                                     bool *outIsRowReduction =
                                                         nullptr) {
  auto gen = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!gen)
    return false;
  if (gen.getNumLoops() != 2)
    return false;
  auto iters = gen.getIteratorTypesArray();
  if (iters.size() != 2)
    return false;
  const bool isElemwise =
      (iters[0] == mlir::utils::IteratorType::parallel &&
       iters[1] == mlir::utils::IteratorType::parallel);
  const bool isRowReduction =
      (iters[0] == mlir::utils::IteratorType::parallel &&
       iters[1] == mlir::utils::IteratorType::reduction);
  if (outIsRowReduction)
    * outIsRowReduction = isRowReduction;
  return isElemwise || isRowReduction;
}

static llvm::DenseMap<int, int64_t>
collectMatmulSoftmaxChainMinConnectLevels(const TileGraph &graph,
                                          int64_t minLevel,
                                          int64_t minLevelElemwiseChain) {
  llvm::DenseMap<int, int64_t> forcedMinLevels;
  if (minLevel <= 0)
    return forcedMinLevels;
  minLevelElemwiseChain = std::max<int64_t>(minLevel, minLevelElemwiseChain);
  for (int m = 0; m < static_cast<int>(graph.nodes.size()); ++m) {
    Operation *rootOp = graph.nodes[m].op;
    if (!isa<linalg::MatmulOp>(rootOp))
      continue;

    llvm::SmallVector<int, 32> queue;
    llvm::SmallDenseSet<int, 64> visitedNodes;
    llvm::SmallDenseSet<int, 64> candidateEdges;
    bool sawRowReduction = false;
    queue.push_back(m);
    visitedNodes.insert(m);

    while (!queue.empty()) {
      int cur = queue.pop_back_val();
      if (cur < 0 || cur >= static_cast<int>(graph.nodes.size()))
        continue;
      for (int edgeIdx : graph.nodes[cur].outEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (e.dst < 0 || e.dst >= static_cast<int>(graph.nodes.size()))
          continue;
        bool isRowReduction = false;
        if (!isTwoDimElementwiseOrRowReductionGeneric(
                graph.nodes[e.dst].op, &isRowReduction))
          continue;
        candidateEdges.insert(edgeIdx);
        if (isRowReduction)
          sawRowReduction = true;
        if (visitedNodes.insert(e.dst).second)
          queue.push_back(e.dst);
      }
    }

    if (!sawRowReduction)
      continue;

    for (int edgeIdx : candidateEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      int64_t targetMinLevel = minLevel;
      auto srcGen = dyn_cast_or_null<linalg::GenericOp>(graph.nodes[e.src].op);
      auto dstGen = dyn_cast_or_null<linalg::GenericOp>(graph.nodes[e.dst].op);
      if (srcGen && dstGen && srcGen.getNumReductionLoops() == 0 &&
          dstGen.getNumReductionLoops() == 0 &&
          srcGen.getNumLoops() == dstGen.getNumLoops()) {
        targetMinLevel = std::max<int64_t>(targetMinLevel,
                                           minLevelElemwiseChain);
      }
      auto it = forcedMinLevels.find(edgeIdx);
      if (it == forcedMinLevels.end())
        forcedMinLevels[edgeIdx] = targetMinLevel;
      else
        it->second = std::max<int64_t>(it->second, targetMinLevel);
    }
  }
  return forcedMinLevels;
}

static int64_t
enforceMatmulSoftmaxChainConnectLevels(TileGraph &graph, int64_t minLevel,
                                       int64_t minLevelElemwiseChain) {
  auto forcedMinLevels = collectMatmulSoftmaxChainMinConnectLevels(
      graph, minLevel, minLevelElemwiseChain);
  int64_t forced = 0;
  for (const auto &kv : forcedMinLevels) {
    int edgeIdx = kv.first;
    int64_t targetMinLevel = kv.second;
    if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
      continue;
    TileGraphEdge &e = graph.edges[edgeIdx];
    if (e.connectLevel >= targetMinLevel)
      continue;
    e.connectLevel = targetMinLevel;
    ++forced;
  }
  return forced;
}

static std::optional<int64_t>
computeTensorCoreBlockThreadsForCodegen(const Candidate &cand,
                                        int64_t maxRowReductionExtent) {
  if (!(cand.enableTensorCoreF16 || cand.enableTensorCoreTf32))
    return std::nullopt;

  int64_t mmaM = cand.mmaM > 0 ? cand.mmaM : 16;
  int64_t mmaN = cand.mmaN > 0 ? cand.mmaN : 8;
  if (cand.tileM <= 0 || cand.tileN <= 0 || mmaM <= 0 || mmaN <= 0)
    return std::nullopt;

  int64_t effTileN = cand.tileN;
  if (maxRowReductionExtent > 1)
    effTileN = std::max<int64_t>(effTileN, maxRowReductionExtent);
  if ((cand.tileM % mmaM) != 0 || (effTileN % mmaN) != 0)
    return std::nullopt;

  int64_t warpsM = cand.tileM / mmaM;
  int64_t warpsN = effTileN / mmaN;
  if (warpsM <= 0 || warpsN <= 0)
    return std::nullopt;
  if (warpsM > (std::numeric_limits<int64_t>::max() / warpsN))
    return std::nullopt;
  int64_t warps = warpsM * warpsN;
  if (warps > (std::numeric_limits<int64_t>::max() / 32))
    return std::nullopt;
  int64_t threads = warps * 32;
  if (threads <= 0)
    return std::nullopt;
  return threads;
}

static int64_t estimateBlockThreadsForCandidate(
    const Candidate &cand, int64_t maxRowReductionExtentForTc) {
  if (cand.enableTensorCoreTf32 || cand.enableTensorCoreF16) {
    if (auto tcThreads = computeTensorCoreBlockThreadsForCodegen(
            cand, maxRowReductionExtentForTc);
        tcThreads && *tcThreads > 0 && *tcThreads <= 1024) {
      return *tcThreads;
    }
  }

  auto pickThreadTile = [](int64_t tile, int64_t prefer) -> int64_t {
    if (prefer > 0 && tile > 0 && tile % prefer == 0)
      return prefer;
    const int64_t fallbacks[] = {4, 2, 1};
    for (int64_t v : fallbacks) {
      if (v > 0 && tile > 0 && tile % v == 0)
        return v;
    }
    return 1;
  };

  int64_t effTTM =
      cand.threadTileM > 0 ? cand.threadTileM : pickThreadTile(cand.tileM, 4);
  int64_t effTTN =
      cand.threadTileN > 0 ? cand.threadTileN : pickThreadTile(cand.tileN, 4);
  if (cand.tileM <= 0 || cand.tileN <= 0 || effTTM <= 0 || effTTN <= 0)
    return 1;
  if (cand.tileM % effTTM != 0 || cand.tileN % effTTN != 0)
    return 1;

  int64_t blockDimX =
      cand.swapBlockDims ? (cand.tileM / effTTM) : (cand.tileN / effTTN);
  int64_t blockDimY =
      cand.swapBlockDims ? (cand.tileN / effTTN) : (cand.tileM / effTTM);
  if (blockDimX <= 0 || blockDimY <= 0)
    return 1;
  if (blockDimX > (std::numeric_limits<int64_t>::max() / blockDimY))
    return 1;
  int64_t blockThreads = blockDimX * blockDimY;
  if (blockThreads <= 0 || blockThreads > 1024)
    return 1;
  return blockThreads;
}

[[maybe_unused]] static bool isBlockOrderConsistentInSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg) {
  // 论文/Welder 对齐：融合 kernel 需要一致的 block_order 分配
  // （DefaultPolicy._assign_block_order 在冲突时返回 False）。
  //
  // 当前 MLIR 实现只建模了基于 indexing map 推导出的 2D swap/no-swap 提示，
  // 因此这里保守处理：
  // - `std::nullopt` 视为“未知/不关心”；
  // - 子图内显式 `{true,false}` 不一致视为冲突。
  bool seenTrue = false;
  bool seenFalse = false;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    const TileGraphNode &n = graph.nodes[nodeIdx];
    if (!n.swapXYHint)
      continue;
    if (*n.swapXYHint)
      seenTrue = true;
    else
      seenFalse = true;
    if (seenTrue && seenFalse)
      return false;
  }
  return true;
}

static std::optional<bool> inferSwapXYHintForSubgraph(const TileGraph &graph,
                                                      const PaperSubgraph &sg) {
  bool seenTrue = false;
  bool seenFalse = false;
  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    const TileGraphNode &n = graph.nodes[nodeIdx];
    if (!n.swapXYHint)
      continue;
    if (*n.swapXYHint)
      seenTrue = true;
    else
      seenFalse = true;
  }
  if (seenTrue && seenFalse)
    return std::nullopt;
  if (seenTrue)
    return true;
  if (seenFalse)
    return false;
  return std::nullopt;
}

static int countNonCutConsumersInSubgraphAtLevel(const TileGraph &graph,
                                                 const PaperSubgraph &sg,
                                                 int src, int minLevelExclusive,
                                                 int maxLevelInclusive) {
  if (src < 0 || src >= static_cast<int>(graph.nodes.size()))
    return 0;
  int count = 0;
  for (int eidx : graph.nodes[src].outEdges) {
    if (eidx < 0 || eidx >= static_cast<int>(graph.edges.size()))
      continue;
    const TileGraphEdge &e = graph.edges[eidx];
    if (e.isCut || e.connectLevel <= minLevelExclusive)
      continue;
    if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
      continue;
    if (!sg.inSet.contains(e.dst))
      continue;
    ++count;
  }
  return count;
}

static bool isCheapDuplicateFusionProducer(linalg::GenericOp gen) {
  if (!gen)
    return false;
  if (gen.getNumReductionLoops() != 0)
    return false;
  if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
    return false;
  if (gen->getNumRegions() != 1 || gen.getRegion().empty())
    return false;

  Block &body = gen.getRegion().front();
  Operation *payloadOp = nullptr;
  for (Operation &op : body.without_terminator()) {
    if (payloadOp)
      return false;
    payloadOp = &op;
  }
  if (!payloadOp)
    return false;
  return isa<arith::ExtFOp, arith::TruncFOp, arith::BitcastOp,
             arith::IndexCastOp, arith::SIToFPOp, arith::UIToFPOp,
             arith::FPToSIOp, arith::FPToUIOp>(payloadOp);
}

static bool isRegisterFuseEligibleEdge(const TileGraph &graph,
                                       const PaperSubgraph &sg,
                                       const TileGraphEdge &e,
                                       int minLevelExclusive,
                                       int maxLevelInclusive = -1) {
  if (e.isCut || e.connectLevel <= minLevelExclusive)
    return false;
  if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
    return false;
  if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
    return false;
  Operation *srcOp = graph.nodes[e.src].op;
  Operation *dstOp = graph.nodes[e.dst].op;
  auto srcGen = dyn_cast_or_null<linalg::GenericOp>(srcOp);
  auto dstGen = dyn_cast_or_null<linalg::GenericOp>(dstOp);
  if (!srcGen || !dstGen)
    return false;
  if (srcGen.getNumReductionLoops() != 0 || dstGen.getNumReductionLoops() != 0)
    return false;
  if (srcGen.getNumLoops() != dstGen.getNumLoops())
    return false;
  const int nonCutConsumers = countNonCutConsumersInSubgraphAtLevel(
      graph, sg, e.src, minLevelExclusive, maxLevelInclusive);
  const bool cheapDupProducer = isCheapDuplicateFusionProducer(srcGen);
  if (nonCutConsumers != 1 &&
      !(cheapDupProducer && nonCutConsumers > 0 && nonCutConsumers <= 2)) {
    return false;
  }
  return true;
}

static void cutEdgesOnSwapXYConflict(TileGraph &graph, const PaperSubgraph &sg,
                                     int minLevelExclusive) {
  // 论文/Welder 对齐（鲁棒变体）：
  // 遇到 block_order 冲突时，不直接让候选失败，而是切断冲突边，
  // 让调度回退为 kernel 间通过 global 内存衔接。
  // 这与“断开连接 + 计入代价”的思想一致。
  for (int edgeIdx = 0; edgeIdx < static_cast<int>(graph.edges.size()); ++edgeIdx) {
    TileGraphEdge &e = graph.edges[edgeIdx];
    if (e.connectLevel <= minLevelExclusive)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!sg.inSet.contains(e.src) || !sg.inSet.contains(e.dst))
      continue;
    const TileGraphNode &a = graph.nodes[e.src];
    const TileGraphNode &b = graph.nodes[e.dst];
    if (!a.swapXYHint || !b.swapXYHint)
      continue;
    if (*a.swapXYHint == *b.swapXYHint)
      continue;
    setEdgeConnectLevel(e, kConnectLevelGlobal);
  }
}

static PaperSubgraph extractSubgraphByConnectLevel(const TileGraph &graph,
                                                   int startNode,
                                                   int minLevelExclusive) {
  PaperSubgraph sg;
  if (startNode < 0 || startNode >= static_cast<int>(graph.nodes.size()))
    return sg;

  llvm::SmallVector<int, 16> stack;
  stack.push_back(startNode);
  sg.inSet.insert(startNode);

  while (!stack.empty()) {
    int cur = stack.pop_back_val();
    sg.nodes.push_back(cur);

    const TileGraphNode &n = graph.nodes[cur];
    for (int edgeIdx : n.inEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      int nei = e.src;
      if (nei < 0 || nei >= static_cast<int>(graph.nodes.size()))
        continue;
      if (sg.inSet.insert(nei).second)
        stack.push_back(nei);
    }
    for (int edgeIdx : n.outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      int nei = e.dst;
      if (nei < 0 || nei >= static_cast<int>(graph.nodes.size()))
        continue;
      if (sg.inSet.insert(nei).second)
        stack.push_back(nei);
    }
  }

  // 去重/稳定化（不依赖 DFS 顺序）。
  llvm::sort(sg.nodes);
  sg.nodes.erase(std::unique(sg.nodes.begin(), sg.nodes.end()), sg.nodes.end());
  return sg;
}

static std::optional<int> pickConnectedSinkInSubgraph(const TileGraph &graph,
                                                      const PaperSubgraph &sg,
                                                      int minLevelExclusive) {
  // “sink” 的定义：在连接子图里，没有任何 outEdge 满足 connectLevel>minLevelExclusive 且
  // 指向同一子图内的节点。
  llvm::SmallVector<int, 8> sinks;
  for (int n : sg.nodes) {
    bool hasConnectedOut = false;
    for (int edgeIdx : graph.nodes[n].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (sg.inSet.contains(e.dst)) {
        hasConnectedOut = true;
        break;
      }
    }
    if (!hasConnectedOut)
      sinks.push_back(n);
  }
  if (sinks.empty())
    return std::nullopt;

  // 调度 sink 优先选择非 trivial op。trivial op（fill/copy）在低 connect-level
  // 被切边后容易“意外成为 sink”，从而扭曲论文风格的子图抽取与分块逻辑。
  for (int n : sinks) {
    if (!isTrivialOpFor2LevelFootprint(graph.nodes[n].op))
      return n;
  }
  return sinks.front();
}

static llvm::SmallVector<int, 16>
topoSortSubgraphByConnectedEdges(const TileGraph &graph, const PaperSubgraph &sg,
                                 int minLevelExclusive) {
  llvm::DenseMap<int, int> idxOf;
  idxOf.reserve(sg.nodes.size());
  for (int i = 0; i < static_cast<int>(sg.nodes.size()); ++i)
    idxOf.insert({sg.nodes[i], i});

  llvm::SmallVector<int, 16> indeg(sg.nodes.size(), 0);

  for (int n : sg.nodes) {
    for (int edgeIdx : graph.nodes[n].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (!sg.inSet.contains(e.dst))
        continue;
      auto it = idxOf.find(e.dst);
      if (it != idxOf.end())
        indeg[it->second] += 1;
    }
  }

  llvm::SmallVector<int, 16> q;
  q.reserve(sg.nodes.size());
  for (int n : sg.nodes) {
    auto it = idxOf.find(n);
    if (it == idxOf.end())
      continue;
    if (indeg[it->second] == 0)
      q.push_back(n);
  }

  llvm::SmallVector<int, 16> topo;
  topo.reserve(sg.nodes.size());
  while (!q.empty()) {
    int n = q.pop_back_val();
    topo.push_back(n);
    for (int edgeIdx : graph.nodes[n].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.connectLevel <= minLevelExclusive)
        continue;
      if (!sg.inSet.contains(e.dst))
        continue;
      auto it = idxOf.find(e.dst);
      if (it == idxOf.end())
        continue;
      int &d = indeg[it->second];
      if (--d == 0)
        q.push_back(e.dst);
    }
  }

  if (topo.size() != sg.nodes.size()) {
    // 出现环/拓扑失败时，退化成稳定顺序（依旧可用于 footprint 估算）。
    topo.assign(sg.nodes.begin(), sg.nodes.end());
  }
  return topo;
}

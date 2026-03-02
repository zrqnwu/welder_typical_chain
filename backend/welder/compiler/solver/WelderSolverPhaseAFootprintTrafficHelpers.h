static bool dependsOnAnyReductionDim(linalg::LinalgOp op, int operandIdx,
                                    const GridInfo &grid) {
  auto maps = op.getIndexingMapsArray();
  if (operandIdx < 0 || operandIdx >= static_cast<int>(maps.size()))
    return false;
  AffineMap m = maps[operandIdx];
  for (AffineExpr e : m.getResults()) {
    for (int dim = 0; dim < static_cast<int>(grid.iterators.size()); ++dim) {
      if (grid.iterators[dim] == utils::IteratorType::reduction) {
        if (e.isFunctionOfDim(dim))
          return true;
      }
    }
  }
  return false;
}

static double getVolume(const OperandFootprint &fp) {
  // 标量（rank=0 / shape 为空）不计入 global memory traffic（避免噪声破坏 baseline）。
  if (fp.shape.empty())
    return 0.0;
  double v = 1.0;
  for (int64_t s : fp.shape) {
    if (s <= 0)
      return 0.0;
    v *= static_cast<double>(s);
  }
  return v;
}

static double computeSharedToRegTrafficBytesForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, int maxLevelInclusive, int64_t workgroupPadLastDim,
    bool workgroupPadLastDimMatmulOnly, int64_t workgroupSwizzleXor) {
  // 论文语义（level>0 递归）：估计当前内存层（如 shared）到下一层
  // （如 registers）之间的流量。
  //
  // 对寄存器层复用（connectLevel > minLevelExclusive）：
  // - 若某个 operand 由满足条件的入边提供，则认为它可在寄存器复用，
  //   无需再从 shared 加载；
  // - 否则，该 operand 的 footprint 仍从当前层（shared）加载。
  double bytes = 0.0;
  double elemBytes = static_cast<double>(arch.elementBytes);
  // 布局旋钮（padding/重排）会影响 shared footprint 与 bank 冲突，
  // 但不会改变 shared->regs 的逻辑加载元素数。
  // padding 通过 strided view + subview 下沉，因此访问切片体积保持不变。
  (void)workgroupPadLastDim;
  (void)workgroupPadLastDimMatmulOnly;
  (void)workgroupSwizzleXor;

  for (int nodeIdx : sg.nodes) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(graph.nodes.size()))
      continue;
    const TileGraphNode &node = graph.nodes[nodeIdx];
    if (!node.op || !node.hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(node.op))
      continue;

    auto op = dyn_cast_or_null<linalg::LinalgOp>(node.op);
    if (!op)
      continue;

    auto fpOpt = inference.infer(node.op, node.requiredTile);
    if (!fpOpt)
      continue;

    auto gridOpt = computeGridInfo(op, node.requiredTile, requirePerfectTiling);
    if (!gridOpt)
      continue;
    const GridInfo &grid = *gridOpt;

    int numInputs = op.getNumDpsInputs();
    for (int operandIdx = 0; operandIdx < numInputs; ++operandIdx) {
      if (operandIdx < 0 ||
          operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
        continue;
      const OperandFootprint &fp = fpOpt->perOperand[operandIdx];
      double vol = getVolume(fp);
      if (!(vol > 0.0))
        continue;

      bool hasHigherLevelProducer = false;
      for (int edgeIdx : graph.nodes[nodeIdx].inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (e.dstOperand != operandIdx)
          continue;
        if (!isRegisterFuseEligibleEdge(graph, sg, e, minLevelExclusive,
                                        maxLevelInclusive))
          continue;
        hasHigherLevelProducer = true;
        break;
      }
      if (hasHigherLevelProducer)
        continue;

      bool dependsOnK = dependsOnAnyReductionDim(op, operandIdx, grid);
      int64_t hits =
          grid.blocksTotal *
          (dependsOnK ? grid.reductionTiles : static_cast<int64_t>(1));
      hits = std::max<int64_t>(1, hits);
      bytes += vol * elemBytes * static_cast<double>(hits);
    }
  }
  return bytes;
}

static llvm::SmallVector<int64_t, 4> getStaticShapeOrUnknown(Value v) {
  llvm::SmallVector<int64_t, 4> shape;
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank())
    return shape;
  shape.reserve(st.getRank());
  for (int64_t d : st.getShape())
    shape.push_back(d);
  return shape;
}

static llvm::SmallVector<int64_t, 4> getStaticStridesOrEmpty(Value v) {
  llvm::SmallVector<int64_t, 4> strides;
  auto mt = dyn_cast<MemRefType>(v.getType());
  if (mt && mt.hasRank()) {
    llvm::SmallVector<int64_t, 4> s;
    int64_t offset = 0;
    if (failed(mt.getStridesAndOffset(s, offset)))
      return strides;
    if (offset == ShapedType::kDynamic)
      return strides;
    for (int64_t st : s) {
      if (st == ShapedType::kDynamic || st <= 0)
        return llvm::SmallVector<int64_t, 4>();
    }
    strides = s;
    return strides;
  }

  // 对 tensor：若 shape 静态，默认按 row-major 连续布局处理。
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank())
    return strides;
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(st.getRank());
  for (int64_t d : st.getShape()) {
    if (d == ShapedType::kDynamic || d <= 0)
      return strides;
    shape.push_back(d);
  }
  if (shape.empty())
    return strides;
  strides.resize(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    if (shape[i] > 0 && running <= (std::numeric_limits<int64_t>::max() / shape[i]))
      running *= shape[i];
    else
      running = std::numeric_limits<int64_t>::max();
  }
  return strides;
}

static int64_t coalescedFactorWithLayout(ArrayRef<int64_t> subtensor,
                                         ArrayRef<int64_t> full,
                                         ArrayRef<int64_t> fullStrides) {
  if (subtensor.empty() || full.empty() || subtensor.size() != full.size())
    return subtensor.empty() ? 0 : std::max<int64_t>(1, subtensor.back());
  if (!fullStrides.empty() && fullStrides.size() == full.size()) {
    // 计算 row-major 遍历下最大的连续访问长度（按元素计），
    // 并考虑非连续布局（如 transpose）。
    int n = static_cast<int>(subtensor.size());
    if (n <= 0)
      return 0;
    if (fullStrides[n - 1] != 1)
      return 1;
    int64_t factor = 1;
    for (int i = n - 1; i >= 0; --i) {
      int64_t sub = subtensor[i];
      int64_t fullDim = full[i];
      if (sub <= 0)
        return 0;
      factor *= sub;
      if (i == 0)
        break;
      // 仅当满足以下条件时，连续性才能向外一维扩展：
      // - subtensor 完整覆盖当前内层维；
      // - stride 符合 row-major 连续关系：
      //   即满足 `stride[i-1] == full[i] * stride[i]`。
      if (fullDim <= 0 || sub != fullDim)
        break;
      int64_t expected = 0;
      if (fullStrides[i] > 0 && fullDim > 0 &&
          fullStrides[i] <= (std::numeric_limits<int64_t>::max() / fullDim))
        expected = fullDim * fullStrides[i];
      if (expected == 0 || fullStrides[i - 1] != expected)
        break;
    }
    return std::max<int64_t>(1, factor);
  }

  // 回退：使用旧的“仅 shape”启发式。
  return coalescedFactor(subtensor, full);
}

static int64_t coalescedFactor(ArrayRef<int64_t> subtensor,
                               ArrayRef<int64_t> full) {
  if (subtensor.empty() || full.empty() || subtensor.size() != full.size())
    return subtensor.empty() ? 0 : std::max<int64_t>(1, subtensor.back());
  if (subtensor.size() == 1)
    return std::max<int64_t>(1, subtensor.back());
  int64_t lastSub = subtensor.back();
  int64_t lastFull = full.back();
  if (lastSub <= 0)
    return 0;
  // 若完整维度是动态/未知，则保守地只认为最后一维可合并访问
  // （论文对齐：无法证明连续切片可继续向外扩展）。
  if (lastFull <= 0 || lastSub != lastFull)
    return lastSub;
  return lastSub *
         coalescedFactor(subtensor.drop_back(), full.drop_back());
}

static double coalescedTensorElements(ArrayRef<int64_t> subtensor,
                                      ArrayRef<int64_t> full,
                                      ArrayRef<int64_t> fullStrides,
                                      int64_t transactionElements) {
  if (subtensor.empty())
    return 0.0;
  long double elems = 1.0;
  for (int64_t s : subtensor) {
    if (s <= 0)
      return 0.0;
    elems *= static_cast<long double>(s);
  }
  if (elems == 0.0)
    return 0.0;
  int64_t factor = coalescedFactorWithLayout(subtensor, full, fullStrides);
  if (factor <= 0)
    return 0.0;

  int64_t txn = std::max<int64_t>(1, transactionElements);
  int64_t denom = std::max<int64_t>(1, std::min<int64_t>(txn, factor));
  long double charged = elems * static_cast<long double>(txn) /
                        static_cast<long double>(denom);
  return static_cast<double>(charged);
}

static double coalescedTensorElements(ArrayRef<int64_t> subtensor,
                                      ArrayRef<int64_t> full,
                                      int64_t transactionElements) {
  if (subtensor.empty())
    return 0.0;
  long double elems = 1.0;
  for (int64_t s : subtensor) {
    if (s <= 0)
      return 0.0;
    elems *= static_cast<long double>(s);
  }
  if (elems == 0.0)
    return 0.0;
  int64_t factor = coalescedFactor(subtensor, full);
  if (factor <= 0)
    return 0.0;

  int64_t txn = std::max<int64_t>(1, transactionElements);
  int64_t denom = std::max<int64_t>(1, std::min<int64_t>(txn, factor));
  long double charged = elems * static_cast<long double>(txn) /
                        static_cast<long double>(denom);
  return static_cast<double>(charged);
}

static std::optional<int64_t> inferElementBytesFromScalarType(Type t) {
  if (!t)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(t)) {
    int64_t bits = ft.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, (bits + 7) / 8);
  }
  if (auto it = dyn_cast<IntegerType>(t)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, (bits + 7) / 8);
  }
  return std::nullopt;
}

static void inferArchElementBytesFromModule(ModuleOp module, ArchConfig &arch) {
  // 优先用第一个“真实” linalg op 作为 dtype 锚点。
  std::optional<int64_t> bytes;
  module.walk([&](linalg::LinalgOp op) {
    if (bytes)
      return;
    // 优先取第一个 DPS input 的元素类型。
    if (op.getNumDpsInputs() > 0) {
      Value v = op.getDpsInputOperand(0)->get();
      if (auto st = dyn_cast<ShapedType>(v.getType())) {
        if (auto b = inferElementBytesFromScalarType(st.getElementType()))
          bytes = *b;
      }
    }
    if (bytes)
      return;
    // 回退：使用任意 operand 的元素类型。
    for (Value v : op->getOperands()) {
      if (auto st = dyn_cast<ShapedType>(v.getType())) {
        if (auto b = inferElementBytesFromScalarType(st.getElementType())) {
          bytes = *b;
          return;
        }
      }
    }
  });
  if (bytes && *bytes > 0)
    arch.elementBytes = *bytes;
}

static bool isPowerOfTwoI64(int64_t v) {
  if (v <= 0)
    return false;
  return (v & (v - 1)) == 0;
}

static bool canApplyXorSwizzleLastDim(int64_t lastDim, int64_t swizzle) {
  if (swizzle <= 1)
    return false;
  if (!isPowerOfTwoI64(swizzle))
    return false;
  if (lastDim <= 1)
    return false;
  // 需与 pass 里的可用性判定保持一致。
  if (!isPowerOfTwoI64(lastDim))
    return false;
  return swizzle <= lastDim;
}

static int64_t bankIndexForOffsetElems(int64_t offsetElems,
                                       int64_t elementBytes) {
  // CUDA shared memory 通常是 32 个 bank、每个 bank 4 字节（32-bit banking）。
  // 这里用 4B word 索引近似映射访问 bank。
  int64_t bytes = offsetElems * std::max<int64_t>(1, elementBytes);
  int64_t word = bytes / 4;
  int64_t bank = word % 32;
  if (bank < 0)
    bank += 32;
  return bank;
}

static int64_t maxBankMultiplicityForWarp(ArrayRef<int64_t> banks) {
  int counts[32] = {0};
  for (int64_t b : banks) {
    if (b < 0)
      continue;
    counts[b % 32]++;
  }
  int mx = 0;
  for (int c : counts)
    mx = std::max(mx, c);
  return std::max<int64_t>(1, mx);
}

static double estimateMatmulSharedBankConflictFactor(const Candidate &cand,
                                                     const ArchConfig &arch) {
  // 这是仅在关闭性能测量时使用的启发式模型。
  // 它近似估计首个 warp 对 A/B shared load 的最大 bank 冲突倍数，
  // 并返回 >= 1.0 的因子。
  if (cand.enableTensorCoreTf32 || cand.enableTensorCoreF16)
    return 1.0; // MMA paths have different access patterns; keep neutral.

  if (cand.tileM <= 0 || cand.tileN <= 0 || cand.tileK <= 0)
    return 1.0;

  int64_t ttm = cand.threadTileM;
  int64_t ttn = cand.threadTileN;
  if (ttm <= 0 || ttn <= 0)
    return 1.0;
  if (cand.tileM % ttm != 0 || cand.tileN % ttn != 0)
    return 1.0;

  int64_t blockDimX = cand.swapBlockDims ? (cand.tileM / ttm) : (cand.tileN / ttn);
  int64_t blockDimY = cand.swapBlockDims ? (cand.tileN / ttn) : (cand.tileM / ttm);
  if (blockDimX <= 0 || blockDimY <= 0)
    return 1.0;

  // 建模可选 padding 后的 shared tile stride。
  int64_t pad = std::max<int64_t>(0, cand.workgroupPadLastDim);
  // padding 后的物理 stride（按元素计）；shape 的最后一维保持不变，
  // 因为 padding 是通过 strided view + subview 下沉实现的。
  int64_t aStride = cand.tileK + pad; // A is [M, K]
  int64_t bStride = cand.tileN + pad; // B is [K, N]

  // 与 workgroup 重排 pass 的可用性保持一致：要求逻辑最后一维（shape）
  // 为 2 的幂，不依赖 padding 后的 stride。
  bool swizzleA =
      canApplyXorSwizzleLastDim(cand.tileK, cand.workgroupSwizzleXor);
  bool swizzleB =
      canApplyXorSwizzleLastDim(cand.tileN, cand.workgroupSwizzleXor);

  auto swizzledCol = [&](int64_t row, int64_t col, bool enabled) -> int64_t {
    if (!enabled)
      return col;
    int64_t swz = cand.workgroupSwizzleXor;
    if (swz <= 1)
      return col;
    // 变换：`col' = col ^ (row & (swz-1))`
    return col ^ (row & (swz - 1));
  };

  // 模拟第一个 warp（lane 0..31）。
  llvm::SmallVector<int64_t, 32> aBanks;
  llvm::SmallVector<int64_t, 32> bBanks;
  aBanks.reserve(32);
  bBanks.reserve(32);

  auto laneToXY = [&](int lane) -> std::pair<int64_t, int64_t> {
    int64_t x = lane % blockDimX;
    int64_t y = lane / blockDimX;
    return {x, y};
  };

  // 每个 thread tile 取一个代表性内部元素（rm=0, cn=0），
  // 并取一个代表性 k 切片（k=0）。
  const int64_t rm = 0;
  const int64_t cn = 0;
  const int64_t k = 0;

  for (int lane = 0; lane < 32; ++lane) {
    auto [tx, ty] = laneToXY(lane);
    // 根据 swapBlockDims，将 (tx, ty) 映射为逻辑 (mIdx, nIdx)。
    int64_t mThread = cand.swapBlockDims ? tx : ty;
    int64_t nThread = cand.swapBlockDims ? ty : tx;
    int64_t mIdx = mThread * ttm + rm;
    int64_t nIdx = nThread * ttn + cn;

    // A 访问坐标：[mIdx, k]
    int64_t aCol = swizzledCol(mIdx, k, swizzleA);
    int64_t aOff = mIdx * aStride + aCol;
    aBanks.push_back(bankIndexForOffsetElems(aOff, arch.elementBytes));

    // B 访问坐标：[k, nIdx]
    int64_t bCol = swizzledCol(k, nIdx, swizzleB);
    int64_t bOff = k * bStride + bCol;
    bBanks.push_back(bankIndexForOffsetElems(bOff, arch.elementBytes));
  }

  int64_t aConf = maxBankMultiplicityForWarp(aBanks);
  int64_t bConf = maxBankMultiplicityForWarp(bBanks);
  int64_t worst = std::max<int64_t>(aConf, bConf);
  return std::max<double>(1.0, static_cast<double>(worst));
}

static bool isTrivialOpFor2LevelFootprint(Operation *op);

static int64_t estimateRegisterReuseRegsPerThreadForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, int minLevelExclusive,
    int64_t blockThreads, const ArchConfig &arch, int maxLevelInclusive) {
  if (blockThreads <= 0)
    return 0;
  if (arch.elementBytes <= 0)
    return 0;

  long double totalBytes = 0.0;
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
    double vol = getVolume(e.footprint);
    if (!(vol > 0.0))
      continue;
    totalBytes += static_cast<long double>(vol) *
                  static_cast<long double>(arch.elementBytes);
  }

  if (!(totalBytes > 0.0))
    return 0;

  long double bytesPerThread =
      totalBytes / static_cast<long double>(std::max<int64_t>(1, blockThreads));
  // 一个 32-bit 寄存器按 4 字节计。
  long double regs = std::ceil(bytesPerThread / 4.0L);
  if (!(regs > 0.0L))
    return 0;
  if (regs > static_cast<long double>(std::numeric_limits<int64_t>::max()))
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(regs);
}

// 论文/Welder 对齐：TCPolicy.plan_光栅化。
//
// 参考行为（python/welder/policy/tc.py）：
// - 仅考虑单节点调度；
// - 仅在 num_wave >= 4 时启用；
// - panel_width 计算规则：`clamp(round(L2_size / traffic), 1..16)`；
// - 若“行”方向 tile >= “列”方向 tile，则选 Row，否则选 Column。
static void maybeApplyRasterizationTcPolicyPaper(const PaperSubgraph &sg,
                                                 const Traffic &traffic,
                                                 Candidate &cand) {
  if (sg.nodes.size() != 1)
    return;
  if (!(cand.enableTensorCoreF16 || cand.enableTensorCoreTf32))
    return;
  if (cand.numWave < 4)
    return;
  // XOR 光栅化与 2D 光栅化保持互斥。
  if (cand.blockRasterizeXor != 0)
    return;
  // 尊重显式或已测量得到的设置。
  if (cand.blockRasterizeMode != 0 || cand.blockRasterizePanelWidth != 0)
    return;

  double bytes = traffic.totalBytes();
  if (!(bytes > 0.0))
    return;

  static constexpr double kL2Bytes = 25.0 * 1024.0 * 1024.0;
  int panel = static_cast<int>(std::llround(kL2Bytes / bytes));
  panel = std::max(1, std::min(16, panel));

  int64_t rowTile = cand.swapBlockDims ? cand.tileN : cand.tileM;
  int64_t colTile = cand.swapBlockDims ? cand.tileM : cand.tileN;
  int mode = (rowTile >= colTile) ? 1 : 2; // 1=row, 2=col

  cand.blockRasterizeMode = mode;
  cand.blockRasterizePanelWidth = panel;
}

static bool isTrivialOpFor2LevelFootprint(Operation *op) {
  return op && isa<linalg::FillOp, linalg::CopyOp>(op);
}

static int64_t estimateComponentFootprintBytes2Level(
    const TileGraph &graph, const ArchConfig &arch,
    const llvm::SmallVectorImpl<int> &componentNodes,
    const llvm::DenseSet<int> &inComponent) {
  // 2-level footprint 的最小可用估算：
  // - 只把“未 cut 的 producer->consumer tile”（reuse-tile）当作 shared 中需要缓存的对象；
  // - 用简单的 liveness（producer allocate + last-consumer free）估算峰值；
  // - 为了避免把 fill/copy 的 init buffer 计进来，这里只对“非 trivial op <-> 非 trivial op”
  //   的边计费。

  // 1) 先做 component 内的拓扑序（只看未 cut 的边）。
  llvm::SmallVector<int, 16> topo;
  topo.reserve(componentNodes.size());
  llvm::SmallVector<int, 16> queue;

  llvm::SmallVector<int, 64> indeg(graph.nodes.size(), 0);
  for (int n : componentNodes)
    indeg[n] = 0;

  for (const TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!inComponent.contains(e.src) || !inComponent.contains(e.dst))
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;
    ++indeg[e.dst];
  }

  for (int n : componentNodes) {
    if (indeg[n] == 0)
      queue.push_back(n);
  }
  while (!queue.empty()) {
    int n = queue.pop_back_val();
    topo.push_back(n);
    for (int edgeIdx : graph.nodes[n].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.isCut)
        continue;
      if (!inComponent.contains(e.dst))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;
      if (--indeg[e.dst] == 0)
        queue.push_back(e.dst);
    }
  }
  if (topo.size() != componentNodes.size()) {
    // 出现环/或拓扑序失败时，退化成“操作顺序”估算（依旧保守）。
    topo.assign(componentNodes.begin(), componentNodes.end());
  }

  // 2) 统计每个 SSA tile 的 bytes（取 max），以及在 component 内的 use 次数。
  llvm::DenseMap<const void *, int64_t> valueBytes;
  llvm::DenseMap<const void *, int> remainingUses;

  for (const TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!inComponent.contains(e.src) || !inComponent.contains(e.dst))
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;

    double vol = getVolume(e.footprint);
    if (vol == 0.0)
      continue;
    int64_t bytes =
        static_cast<int64_t>(vol * static_cast<double>(arch.elementBytes));
    if (bytes <= 0)
      continue;

    const void *key = e.value.getAsOpaquePointer();
    auto it = valueBytes.find(key);
    if (it == valueBytes.end())
      valueBytes.insert({key, bytes});
    else
      it->second = std::max<int64_t>(it->second, bytes);

    remainingUses[key] += 1;
  }

  // 3) Liveness 估算峰值。
  llvm::DenseSet<const void *> allocated;
  int64_t cur = 0;
  int64_t peak = 0;

  for (int nodeIdx : topo) {
    // allocate：producer 输出（未 cut）进入 shared cache
    for (int edgeIdx : graph.nodes[nodeIdx].outEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.isCut)
        continue;
      if (!inComponent.contains(e.dst))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;

      const void *key = e.value.getAsOpaquePointer();
      auto it = valueBytes.find(key);
      if (it == valueBytes.end())
        continue;
      if (!allocated.insert(key).second)
        continue;

      cur += it->second;
      peak = std::max<int64_t>(peak, cur);
    }

    // free：consumer 使用完最后一次后释放
    for (int edgeIdx : graph.nodes[nodeIdx].inEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
        continue;
      const TileGraphEdge &e = graph.edges[edgeIdx];
      if (e.isCut)
        continue;
      if (!inComponent.contains(e.src))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;

      const void *key = e.value.getAsOpaquePointer();
      auto useIt = remainingUses.find(key);
      if (useIt == remainingUses.end())
        continue;
      int n = --useIt->second;
      if (n != 0)
        continue;

      auto bytesIt = valueBytes.find(key);
      if (bytesIt == valueBytes.end())
        continue;

      cur -= bytesIt->second;
      allocated.erase(key);
    }
  }

  return peak;
}

static int64_t estimateSharedFootprintBytes2Level(const TileGraph &graph,
                                                  const ArchConfig &arch) {
  // 对每个“未 cut 的连通分量”（视为 1 个 kernel），分别估算 shared footprint，
  // 返回 max（因为 kernel 是串行执行的，容量约束是 per-kernel）。
  int64_t maxBytes = 0;

  llvm::SmallVector<char, 64> visited(graph.nodes.size(), 0);
  llvm::SmallVector<llvm::SmallVector<int, 4>, 64> adj(graph.nodes.size());

  // 基于“未切边”构建无向连通关系（跳过 trivial op）。
  for (const TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (e.src >= static_cast<int>(graph.nodes.size()) ||
        e.dst >= static_cast<int>(graph.nodes.size()))
      continue;
    // 只统计 propagation 覆盖到的 sub-graph（有 requiredTile 的节点）。
    if (!graph.nodes[e.src].hasRequiredTile || !graph.nodes[e.dst].hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;
    adj[e.src].push_back(e.dst);
    adj[e.dst].push_back(e.src);
  }

  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    if (visited[i])
      continue;
    if (!graph.nodes[i].hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[i].op))
      continue;

    llvm::SmallVector<int, 16> component;
    llvm::DenseSet<int> inComponent;

    llvm::SmallVector<int, 16> stack;
    stack.push_back(i);
    visited[i] = 1;
    while (!stack.empty()) {
      int n = stack.pop_back_val();
      component.push_back(n);
      inComponent.insert(n);
      for (int nei : adj[n]) {
        if (nei < 0 || nei >= static_cast<int>(graph.nodes.size()))
          continue;
        if (visited[nei])
          continue;
        visited[nei] = 1;
        stack.push_back(nei);
      }
    }

    int64_t bytes =
        estimateComponentFootprintBytes2Level(graph, arch, component, inComponent);
    maxBytes = std::max<int64_t>(maxBytes, bytes);
  }

  return maxBytes;
}

static std::optional<int>
pickCutEdgeForSharedFootprint2Level(const TileGraph &graph,
                                    const ArchConfig &arch) {
  // 选择一条“最值得 cut 的边”，把 shared footprint 压到容量以内。
  // 当前最小策略：
  // 1) 找到 footprint 最大的未切连通分量；
  // 2) 在该分量内切掉 footprint 最大的边（按 footprint.shape 体积估算）。

  if (graph.nodes.empty() || graph.edges.empty())
    return std::nullopt;

  llvm::SmallVector<char, 64> visited(graph.nodes.size(), 0);
  llvm::SmallVector<llvm::SmallVector<int, 4>, 64> adj(graph.nodes.size());

  // 基于“未切边”构建无向连通关系（跳过 trivial op）。
  for (const TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (e.src >= static_cast<int>(graph.nodes.size()) ||
        e.dst >= static_cast<int>(graph.nodes.size()))
      continue;
    if (!graph.nodes[e.src].hasRequiredTile || !graph.nodes[e.dst].hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;
    adj[e.src].push_back(e.dst);
    adj[e.dst].push_back(e.src);
  }

  int64_t bestCompBytes = -1;
  llvm::SmallVector<int, 16> bestComponent;
  llvm::DenseSet<int> bestInComponent;

  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    if (visited[i])
      continue;
    if (!graph.nodes[i].hasRequiredTile)
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[i].op))
      continue;

    llvm::SmallVector<int, 16> component;
    llvm::DenseSet<int> inComponent;

    llvm::SmallVector<int, 16> stack;
    stack.push_back(i);
    visited[i] = 1;
    while (!stack.empty()) {
      int n = stack.pop_back_val();
      component.push_back(n);
      inComponent.insert(n);
      for (int nei : adj[n]) {
        if (nei < 0 || nei >= static_cast<int>(graph.nodes.size()))
          continue;
        if (visited[nei])
          continue;
        visited[nei] = 1;
        stack.push_back(nei);
      }
    }

    int64_t bytes =
        estimateComponentFootprintBytes2Level(graph, arch, component, inComponent);
    if (bytes > bestCompBytes) {
      bestCompBytes = bytes;
      bestComponent = std::move(component);
      bestInComponent = std::move(inComponent);
    }
  }

  if (bestCompBytes <= 0)
    return std::nullopt;

  int bestEdge = -1;
  int64_t bestEdgeBytes = -1;
  for (int edgeIdx = 0; edgeIdx < static_cast<int>(graph.edges.size()); ++edgeIdx) {
    const TileGraphEdge &e = graph.edges[edgeIdx];
    if (e.isCut)
      continue;
    if (e.src < 0 || e.dst < 0)
      continue;
    if (!graph.nodes[e.src].hasRequiredTile || !graph.nodes[e.dst].hasRequiredTile)
      continue;
    if (!bestInComponent.contains(e.src) || !bestInComponent.contains(e.dst))
      continue;
    if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
        isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
      continue;

    double vol = getVolume(e.footprint);
    if (vol == 0.0)
      continue;
    int64_t bytes =
        static_cast<int64_t>(vol * static_cast<double>(arch.elementBytes));
    if (bytes > bestEdgeBytes) {
      bestEdgeBytes = bytes;
      bestEdge = edgeIdx;
    }
  }

  if (bestEdge < 0)
    return std::nullopt;
  return bestEdge;
}

TilePropagationResult propagateTilesBackward(TileGraph &graph, int rootNode,
                                            const OpTile &rootTile,
                                            const FootprintInference &inference,
                                            const TilePropagationOptions &opts) {
  TilePropagationResult result;

  if (rootNode < 0 || rootNode >= static_cast<int>(graph.nodes.size())) {
    result.error = "invalid root node index";
    return result;
  }

  if (opts.resetGraphState)
    resetTileGraphState(graph, opts.resetCutEdges);

  struct Pending {
    int node = -1;
    OpTile tile;
    // Phase 13A：记录约束来源（从哪条 edge 传过来）。root 节点为 -1。
    int incomingEdgeIdx = -1;
  };
  llvm::SmallVector<Pending, 16> stack;
  stack.push_back(Pending{rootNode, rootTile, /*incomingEdgeIdx=*/-1});

  while (!stack.empty()) {
    Pending cur = std::move(stack.back());
    stack.pop_back();

    if (cur.node < 0 || cur.node >= static_cast<int>(graph.nodes.size())) {
      result.error = "internal error: invalid node index on stack";
      return result;
    }

    TileGraphNode &node = graph.nodes[cur.node];
    if (!node.op) {
      result.error = "internal error: null op in TileGraphNode";
      return result;
    }

    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(node.op);
    if (!linalgOp) {
      result.error = "TileGraphNode is not a linalg op: " + opDebugName(node.op);
      return result;
    }

    if (node.hasRequiredTile) {
      if (!tilesEqual(node.requiredTile, cur.tile)) {
        // Phase 13A：冲突时不直接失败，而是把冲突边切断（走 global memory）。
        if (opts.enableCutEdges && cur.incomingEdgeIdx != -1) {
          if (cur.incomingEdgeIdx < 0 ||
              cur.incomingEdgeIdx >= static_cast<int>(graph.edges.size())) {
            result.error = "internal error: incomingEdgeIdx out of range";
            return result;
          }
          TileGraphEdge &e = graph.edges[cur.incomingEdgeIdx];
          // 保持 GraphConnecting/connectLevel 语义一致：被切边后，
          // 后续不能再按“shared 已连接”处理。
          setEdgeConnectLevel(e, kConnectLevelGlobal);
          // 边切断后，当前路径的约束不再向上游传播。
          continue;
        }

        result.error =
            "tile conflict on op " + opDebugName(node.op) + " (inconsistent requirements)";
        return result;
      }
      continue;
    }

    node.hasRequiredTile = true;
    node.requiredTile = cur.tile;

    auto fpOpt = inference.infer(node.op, cur.tile);
    if (!fpOpt) {
      result.error = "footprint inference failed on op " + opDebugName(node.op);
      return result;
    }

    // 反向传播：遍历所有输入边（producer -> this consumer）。
    for (int edgeIdx : node.inEdges) {
      if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size())) {
        result.error = "internal error: invalid edge index";
        return result;
      }
      TileGraphEdge &edge = graph.edges[edgeIdx];
      if (edge.dstOperand < 0 ||
          edge.dstOperand >= static_cast<int>(fpOpt->perOperand.size())) {
        result.error = "internal error: dstOperand out of range";
        return result;
      }

      // 保存 footprint（便于后续做 traffic 计算 / 调试 dump）。
      edge.footprint = fpOpt->perOperand[edge.dstOperand];

      // GraphConnecting/Cut-edge：如果此边已被切断，则不向上游传播 tile 约束。
      // 但我们仍保留 consumer 侧 footprint，供 traffic/footprint 估算使用。
      if (edge.isCut)
        continue;

      // Propagate v2（indexing_map 驱动）：
      // 用 producer 的输出 indexing_map，把所需输出 tensor footprint
      // 反推为 producer 的并行循环 tile extent。
      if (edge.src < 0 || edge.src >= static_cast<int>(graph.nodes.size())) {
        result.error = "internal error: invalid edge.src";
        return result;
      }
      TileGraphNode &producerNode = graph.nodes[edge.src];
      auto producerOp = dyn_cast_or_null<linalg::LinalgOp>(producerNode.op);
      if (!producerOp) {
        // 论文对齐的鲁棒性策略：若该边无法继续传播，
        // 则将其视为 cut 并中断该约束链。
        if (opts.enableCutEdges) {
          setEdgeConnectLevel(edge, kConnectLevelGlobal);
          continue;
        }
        result.error = "producer is not a linalg op: " + opDebugName(producerNode.op);
        return result;
      }

      const std::vector<int64_t> &outShapeRaw = edge.footprint.shape;
      if (outShapeRaw.empty())
        continue;

      // Reduction-consumer 传播修正：
      //
      // 对于 consumer 带归约循环的边，footprint inference 生成的 operand
      // footprint shape 会使用 consumer 的归约 tile extent（rstep）。
      // 这些 extent 不能反向约束 producer 的输出 tile 形状，否则在如下多消费者模式中：
      // 例如：
      // 示例代码：C = matmul(...)
      // 示例代码：max = reduce_max(C)          // rstep=32
      // exp = exp(C - max)          // 期望 tileN=128
      // 会把 producer 的 N tile 强行限制为 32，与 exp 的 128 冲突。
      //
      // Paper/Welder 语义：rstep 控制的是 tile 内“如何计算归约”，
      // 而非兄弟节点复用的缓存 tile 的空间范围。
      //
      // 最小 v1 规则：对于由归约循环维索引到的 operand 维，
      // 将需求 extent 视为完整循环范围，而不是 rstep extent。
      llvm::SmallVector<int64_t, 8> outShapeStorage;
      llvm::ArrayRef<int64_t> outShapeRef(outShapeRaw);
      if (linalgOp.getNumReductionLoops() > 0) {
        auto maps = linalgOp.getIndexingMapsArray();
        if (edge.dstOperand >= 0 &&
            edge.dstOperand < static_cast<int>(maps.size())) {
          AffineMap inMap = maps[static_cast<size_t>(edge.dstOperand)];
          if (inMap && inMap.getNumSymbols() == 0) {
            auto iters = linalgOp.getIteratorTypesArray();
            llvm::SmallVector<int64_t, 8> ranges = linalgOp.getStaticLoopRanges();
            if (static_cast<int64_t>(ranges.size()) == linalgOp.getNumLoops() &&
                static_cast<int64_t>(iters.size()) == linalgOp.getNumLoops() &&
                static_cast<int64_t>(outShapeRef.size()) ==
                    static_cast<int64_t>(inMap.getNumResults())) {
              outShapeStorage.assign(outShapeRef.begin(), outShapeRef.end());
              for (int64_t dim = 0; dim < inMap.getNumResults(); ++dim) {
                AffineExpr expr = inMap.getResult(dim);
                auto d = dyn_cast<AffineDimExpr>(expr);
                if (!d)
                  continue;
                int64_t loopPos = d.getPosition();
                if (loopPos < 0 || loopPos >= linalgOp.getNumLoops())
                  continue;
                if (iters[loopPos] != utils::IteratorType::reduction)
                  continue;
                int64_t full = ranges[loopPos];
                if (full == ShapedType::kDynamic || full <= 0)
                  continue;
                outShapeStorage[static_cast<size_t>(dim)] = full;
              }
              outShapeRef = outShapeStorage;
            }
          }
        }
      }

      llvm::SmallVector<int64_t, 8> mappedShapeStorage;
      llvm::ArrayRef<int64_t> mappedShapeRef(outShapeRef);
      if (!edge.viewOps.empty()) {
        auto mappedOpt =
            mapFootprintShapeThroughViewOps(mappedShapeRef, edge.viewOps);
        if (!mappedOpt) {
          if (opts.enableCutEdges) {
            setEdgeConnectLevel(edge, kConnectLevelGlobal);
            continue;
          }
          result.error =
              "failed to map footprint through view ops for edge into op " +
              opDebugName(node.op);
          return result;
        }
        mappedShapeStorage = std::move(*mappedOpt);
        mappedShapeRef = mappedShapeStorage;
      }

      std::optional<llvm::SmallVector<int64_t, 8>> parExtOpt =
          inferParallelExtentsFromOutputFootprintShape(producerOp, edge.srcResult,
                                                       mappedShapeRef);
      llvm::SmallVector<int64_t, 8> fallbackParExtents;
      llvm::ArrayRef<int64_t> parExtentsRef;
      if (parExtOpt) {
        parExtentsRef = *parExtOpt;
      } else {
        // 回退：假设输出维与并行循环顺序一一对应。
        if (static_cast<int64_t>(mappedShapeRef.size()) !=
            producerOp.getNumParallelLoops()) {
          if (opts.enableCutEdges) {
            setEdgeConnectLevel(edge, kConnectLevelGlobal);
            continue;
          }
          result.error =
              "cannot infer producer parallel tile from output footprint for op " +
              opDebugName(producerNode.op);
          return result;
        }
        fallbackParExtents.assign(mappedShapeRef.begin(), mappedShapeRef.end());
        parExtentsRef = fallbackParExtents;
      }

      llvm::ArrayRef<int64_t> redExtents;
      if (opts.reductionTilesByNode &&
          edge.src >= 0 &&
          static_cast<size_t>(edge.src) < opts.reductionTilesByNode->size()) {
        redExtents = (*opts.reductionTilesByNode)[edge.src];
      }

      auto producerTileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
          producerOp, parExtentsRef, redExtents, opts.defaultReductionTile);
      if (!producerTileOpt) {
        if (opts.enableCutEdges) {
          setEdgeConnectLevel(edge, kConnectLevelGlobal);
          continue;
        }
        result.error =
            "failed to build producer tile for op " + opDebugName(producerNode.op);
        return result;
      }
      stack.push_back(Pending{edge.src, std::move(*producerTileOpt), edgeIdx});
    }
  }

  result.success = true;
  return result;
}

//===----------------------------------------------------------------------===//
// Phase A：全图流量记账（假设完全融合）
//===----------------------------------------------------------------------===//

static Traffic computeGlobalTrafficAssumingFullyFused(
    const TileGraph &graph, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling) {
  // 约定（Phase A baseline 对齐版）：
  // - 只统计“图外输入的 global read”；
  // - 只统计“最终 sink 节点的 global write”；
  // - 图内 producer->consumer 边假设完全融合，traffic 计为 0。
  //
  // 注意：为了对齐旧的 MatMul 专用公式，这里不统计 DpsInit 的 global read，
  // 也不统计中间 tensor 落地。
  double totalReadBytes = 0.0;
  double totalWriteBytes = 0.0;
  double totalCutBytes = 0.0;
  double elemBytes = static_cast<double>(arch.elementBytes);

  // Phase 13A：切边写回去重（同一个 SSA value 只写一次 global）。
  llvm::DenseSet<const void *> seenCutWrites;

  for (const TileGraphNode &node : graph.nodes) {
    if (!node.op || !node.hasRequiredTile)
      continue;

    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(node.op);
    if (!linalgOp)
      continue;

    auto gridOpt =
        computeGridInfo(linalgOp, node.requiredTile, requirePerfectTiling);
    if (!gridOpt)
      continue;
    GridInfo grid = *gridOpt;

    auto fpOpt = inference.infer(node.op, node.requiredTile);
    if (!fpOpt)
      continue;

    // 1) Global Read：只看 dpsInputs。若该 input 没有来自图内的 producer（in-edge），则视为
    // 图外输入（global read）。
    int numInputs = linalgOp.getNumDpsInputs();
    for (int i = 0; i < numInputs; ++i) {
      bool hasAnyInEdge = false;
      int cutEdgeIdx = -1;
      for (int edgeIdx : node.inEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        if (graph.edges[edgeIdx].dstOperand != i)
          continue;
        hasAnyInEdge = true;
        if (graph.edges[edgeIdx].isCut) {
          cutEdgeIdx = edgeIdx;
          break;
        }
      }

      bool dependsOnK = dependsOnAnyReductionDim(linalgOp, i, grid);
      int64_t hits = grid.blocksTotal * (dependsOnK ? grid.reductionTiles : 1);

      if (!hasAnyInEdge) {
        // 图外输入：global read。
        if (i < 0 || i >= static_cast<int>(fpOpt->perOperand.size()))
          continue;
        double vol = getVolume(fpOpt->perOperand[i]);
        if (vol == 0.0)
          continue;
        totalReadBytes += vol * static_cast<double>(hits) * elemBytes;
        continue;
      }

      if (cutEdgeIdx != -1) {
        // Phase 13A：cut-edge 的 consumer 侧需要从 global 读。
        const TileGraphEdge &e = graph.edges[cutEdgeIdx];
        double vol = getVolume(e.footprint);
        if (vol == 0.0)
          continue;
        totalCutBytes += vol * static_cast<double>(hits) * elemBytes;
      }
    }

    // 2) Global Write：只看 sink 节点（无 outEdges）的 dpsInits（写回最终结果）。
    if (node.outEdges.empty()) {
      int numInits = linalgOp.getNumDpsInits();
      for (int i = 0; i < numInits; ++i) {
        int operandIdx = numInputs + i;
        if (operandIdx < 0 ||
            operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
          continue;
        double vol = getVolume(fpOpt->perOperand[operandIdx]);
        if (vol == 0.0)
          continue;
        totalWriteBytes +=
            vol * static_cast<double>(grid.blocksTotal) * elemBytes;
      }
    }

    // 3) Phase 13A：cut-edge 的 producer 侧需要写回 global（去重）。
    if (!node.outEdges.empty()) {
      int numInits = linalgOp.getNumDpsInits();
      for (int edgeIdx : node.outEdges) {
        if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
          continue;
        const TileGraphEdge &e = graph.edges[edgeIdx];
        if (!e.isCut)
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

        double vol = getVolume(fpOpt->perOperand[operandIdx]);
        if (vol == 0.0)
          continue;
        totalCutBytes +=
            vol * static_cast<double>(grid.blocksTotal) * elemBytes;
      }
    }
  }

  // bytesA=global read，bytesB=0，bytesC=global write，bytesCut=切边流量。
  return Traffic{totalReadBytes, 0.0, totalWriteBytes, totalCutBytes};
}

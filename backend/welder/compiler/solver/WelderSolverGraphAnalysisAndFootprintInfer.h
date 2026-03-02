std::vector<int64_t> parseCsvIntList(const std::string &s) {
  std::vector<int64_t> out;
  std::string cur;
  for (char ch : s) {
    if (ch == ',' || ch == ' ' || ch == '\t' || ch == '\n') {
      if (!cur.empty()) {
        out.push_back(std::stoll(cur));
        cur.clear();
      }
      continue;
    }
    cur.push_back(ch);
  }
  if (!cur.empty())
    out.push_back(std::stoll(cur));
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](int64_t v) { return v <= 0; }),
            out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

static std::optional<llvm::SmallVector<int64_t, 4>> getRankedShape(Type t) {
  if (auto mem = dyn_cast<MemRefType>(t)) {
    if (!mem.hasRank())
      return std::nullopt;
    return llvm::SmallVector<int64_t, 4>(mem.getShape().begin(),
                                         mem.getShape().end());
  }
  if (auto ten = dyn_cast<RankedTensorType>(t)) {
    return llvm::SmallVector<int64_t, 4>(ten.getShape().begin(),
                                         ten.getShape().end());
  }
  return std::nullopt;
}

static bool isStatic2DShaped(Value v, int64_t &d0, int64_t &d1) {
  auto shapeOpt = getRankedShape(v.getType());
  if (!shapeOpt || shapeOpt->size() != 2)
    return false;
  auto shape = *shapeOpt;
  if (shape[0] == ShapedType::kDynamic || shape[1] == ShapedType::kDynamic)
    return false;
  d0 = shape[0];
  d1 = shape[1];
  return true;
}

static bool isTrivialLinalgForSolver(linalg::LinalgOp op) {
  // Phase 9：挑选“主要算子”时，过滤掉一些对性能模型没意义的 op（copy/fill 等）。
  if (!op)
    return true;
  return llvm::TypeSwitch<Operation *, bool>(op.getOperation())
      .Case<linalg::FillOp, linalg::CopyOp>([](auto) { return true; })
      .Default([](Operation *) { return false; });
}

//===----------------------------------------------------------------------===//
// Linalg indexing_maps -> Footprint 推导（最小实现）
//===----------------------------------------------------------------------===//

static std::optional<int64_t> getStaticDimSize(Type type, int64_t dim) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return std::nullopt;
  if (dim < 0 || dim >= shaped.getRank())
    return std::nullopt;
  int64_t sz = shaped.getShape()[dim];
  if (sz == ShapedType::kDynamic)
    return std::nullopt;
  return sz;
}

std::optional<FootprintResult>
LinalgIndexingMapsFootprintInference::infer(Operation *op,
                                            const OpTile &tile) const {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp)
    return std::nullopt;

  int64_t numLoops = linalgOp.getNumLoops();
  if (numLoops <= 0)
    return std::nullopt;
  if (static_cast<int64_t>(tile.loopExtents.size()) != numLoops)
    return std::nullopt;

  // 先做最小实现：要求 indexing_maps 没有 symbols（否则需要额外的符号区间约束）。
  // 常见的 linalg.matmul / elementwise generic 都满足这个要求。
  for (AffineMap m : linalgOp.getIndexingMapsArray()) {
    if (m.getNumSymbols() != 0)
      return std::nullopt;
  }

  llvm::SmallVector<Interval, 8> dimRanges;
  dimRanges.reserve(numLoops);
  for (int64_t i = 0; i < numLoops; ++i) {
    int64_t extent = tile.loopExtents[i];
    if (extent <= 0)
      return std::nullopt;
    dimRanges.push_back(Interval{0, extent - 1});
  }
  llvm::SmallVector<Interval, 0> symbolRanges;

  FootprintResult result;
  result.perOperand.reserve(linalgOp->getNumOperands());

  auto maps = linalgOp.getIndexingMapsArray();
  if (maps.size() != linalgOp->getNumOperands())
    return std::nullopt;

  for (auto [idx, map] : llvm::enumerate(maps)) {
    Value operand = linalgOp->getOperand(idx);
    Type operandType = operand.getType();

    // 标量/向量（非 shaped）operand：先忽略，留空 footprint。
    auto shaped = dyn_cast<ShapedType>(operandType);
    if (!shaped || !shaped.hasRank() || shaped.getRank() == 0) {
      result.perOperand.push_back(OperandFootprint{});
      continue;
    }

    auto intervalsOpt =
        evalAffineMapIntervals(map, dimRanges, symbolRanges);
    if (!intervalsOpt)
      return std::nullopt;

    OperandFootprint fp;
    fp.indexBounds = std::move(*intervalsOpt);
    fp.shape.reserve(fp.indexBounds.size());
    for (size_t d = 0; d < fp.indexBounds.size(); ++d) {
      int64_t sz = fp.indexBounds[d].size();
      if (auto staticDim = getStaticDimSize(operandType, d))
        sz = std::min<int64_t>(sz, *staticDim);
      fp.shape.push_back(sz);
    }
    result.perOperand.push_back(std::move(fp));
  }

  return result;
}

std::optional<ProblemSize> analyzeMatmulProblem(ModuleOp module) {
  std::optional<ProblemSize> result;

  module.walk([&](linalg::MatmulOp mm) {
    if (result.has_value())
      return;

    if (mm.getInputs().size() < 2 || mm.getOutputs().size() < 1)
      return;

    int64_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0;
    if (!isStatic2DShaped(mm.getInputs()[0], a0, a1))
      return;
    if (!isStatic2DShaped(mm.getInputs()[1], b0, b1))
      return;
    if (!isStatic2DShaped(mm.getOutputs()[0], c0, c1))
      return;

    // Matmul 形状约定：A=(M x K)、B=(K x N)、C=(M x N)。
    int64_t m = c0;
    int64_t n = c1;
    int64_t k = a1;

    if (a0 != m || b1 != n || b0 != k)
      return;

    result = ProblemSize{m, n, k};
  });

  return result;
}

std::optional<GenericProblem> analyzeGenericProblem(ModuleOp module) {
  linalg::LinalgOp best = nullptr;
  int64_t bestReductionLoops = -1;
  int64_t bestTotalLoops = -1;

  module.walk([&](linalg::LinalgOp op) {
    if (!op)
      return;
    if (isTrivialLinalgForSolver(op))
      return;

    llvm::SmallVector<int64_t, 8> ranges = op.getStaticLoopRanges();
    if (ranges.empty())
      return;
    for (int64_t r : ranges) {
      if (r == ShapedType::kDynamic)
        return; // Phase 9 第一版：只支持静态 loop range。
    }

    int64_t nRed = op.getNumReductionLoops();
    int64_t nLoops = op.getNumLoops();

    bool better = false;
    if (nRed > bestReductionLoops)
      better = true;
    else if (nRed == bestReductionLoops && nLoops > bestTotalLoops)
      better = true;

    if (better) {
      best = op;
      bestReductionLoops = nRed;
      bestTotalLoops = nLoops;
    }
  });

  if (!best)
    return std::nullopt;

  GenericProblem prob;
  prob.targetOp = best.getOperation();

  llvm::SmallVector<int64_t, 8> ranges = best.getStaticLoopRanges();
  auto iters = best.getIteratorTypesArray();
  if (static_cast<int64_t>(ranges.size()) != best.getNumLoops())
    return std::nullopt;
  if (static_cast<int64_t>(iters.size()) != best.getNumLoops())
    return std::nullopt;

  prob.loops.reserve(best.getNumLoops());
  for (int64_t i = 0; i < best.getNumLoops(); ++i) {
    prob.loops.push_back(LoopDim{ranges[i], iters[i]});
  }
  return prob;
}

bool detectMatmulConsumerChain(ModuleOp module) {
  bool found = false;
  module.walk([&](linalg::MatmulOp mm) {
    if (found)
      return;

    llvm::SmallVector<Value, 4> produced;
    if (mm->getNumResults() > 0)
      produced.push_back(mm->getResult(0));
    for (Value o : mm.getOutputs())
      produced.push_back(o);

    for (Value v : produced) {
      if (!v)
        continue;
      for (Operation *user : v.getUsers()) {
        auto gen = dyn_cast<linalg::GenericOp>(user);
        if (!gen)
          continue;
        for (Value in : gen.getInputs()) {
          if (in == v) {
            found = true;
            return;
          }
        }
      }
    }
  });
  return found;
}

static Traffic computeTrafficBytes(const ProblemSize &p, int64_t tileM,
                                  int64_t tileN, bool assumeFusedRelu,
                                  const ArchConfig &arch) {
  // 约定：这里的 traffic 是“估算 global memory 读写字节数”，用于比较候选 tile。
  // - A/B：每个 (Mtile, Ntile) block 都会把 A/B 从 global 重新搬到 shared（跨 block 不共享）。
  // - C：如果 MatMul->ReLU 真的 fuse 进同一个 kernel，那么中间结果不需要单独回写/再读。
  //      我们保守估计最终只写回一次 C。
  int64_t blocksM = p.m / tileM;
  int64_t blocksN = p.n / tileN;

  double elem = static_cast<double>(arch.elementBytes);
  double bytesA = static_cast<double>(blocksM) * blocksN *
                  static_cast<double>(tileM) * p.k * elem;
  double bytesB = static_cast<double>(blocksM) * blocksN *
                  static_cast<double>(p.k) * tileN * elem;

  double bytesC = static_cast<double>(p.m) * p.n * elem;
  if (!assumeFusedRelu) {
    // 未融合：MatMul 写一次，Consumer 再读一次再写一次（非常粗略）。
    bytesC *= 3.0;
  }

  return Traffic{bytesA, bytesB, bytesC};
}

static std::optional<Traffic>
computeTrafficBytesViaFootprint(linalg::LinalgOp op, int64_t tileM,
                                int64_t tileN, int64_t tileK,
                                bool assumeFusedRelu, const ArchConfig &arch,
                                bool requirePerfectTiling) {
  // 当前先只覆盖 MatMul 这种最常见的模式：
  // - 2 个 parallel loops (M, N)
  // - 1 个 reduction loop (K)
  if (op.getNumParallelLoops() != 2 || op.getNumReductionLoops() != 1)
    return std::nullopt;

  // 构造一个 “单个 op-tile” 的 loop extents：parallel 用 (tileM,tileN)，
  // reduction 用 tileK。
  OpTile tile;
  tile.loopExtents.resize(op.getNumLoops(), 0);

  int64_t pIdx = 0;
  int64_t rIdx = 0;
  auto iters = op.getIteratorTypesArray();
  for (int64_t i = 0; i < op.getNumLoops(); ++i) {
    if (iters[i] == utils::IteratorType::parallel) {
      tile.loopExtents[i] = (pIdx == 0) ? tileM : tileN;
      ++pIdx;
    } else if (iters[i] == utils::IteratorType::reduction) {
      tile.loopExtents[i] = tileK;
      ++rIdx;
    } else {
      return std::nullopt;
    }
  }
  if (pIdx != 2 || rIdx != 1)
    return std::nullopt;

  auto gridOpt = computeGridInfo(op, tile, requirePerfectTiling);
  if (!gridOpt)
    return std::nullopt;
  GridInfo grid = *gridOpt;

  LinalgIndexingMapsFootprintInference infer;
  auto fpOpt = infer.infer(op, tile);
  if (!fpOpt)
    return std::nullopt;

  // 对 linalg.matmul：operand 顺序固定为 [A, B, C]（C 是 output）。
  // 这里先只取前三个 operand。
  if (fpOpt->perOperand.size() < 3)
    return std::nullopt;

  double elem = static_cast<double>(arch.elementBytes);

  // A/B：如果依赖 reduction 维，则每个 reduction tile 都要加载不同切片。
  double bytesA = getVolume(fpOpt->perOperand[0]) * elem;
  double bytesB = getVolume(fpOpt->perOperand[1]) * elem;
  double bytesC = getVolume(fpOpt->perOperand[2]) * elem;

  double multA = dependsOnAnyReductionDim(op, /*operandIdx=*/0, grid)
                     ? static_cast<double>(grid.reductionTiles)
                     : 1.0;
  double multB = dependsOnAnyReductionDim(op, /*operandIdx=*/1, grid)
                     ? static_cast<double>(grid.reductionTiles)
                     : 1.0;
  double multC = 1.0;

  // 未融合：粗略估计 output tile 的 read + write + write（3x）。
  if (!assumeFusedRelu)
    multC = 3.0;

  bytesA *= static_cast<double>(grid.blocksTotal) * multA;
  bytesB *= static_cast<double>(grid.blocksTotal) * multB;
  bytesC *= static_cast<double>(grid.blocksTotal) * multC;

  return Traffic{bytesA, bytesB, bytesC};
}

static std::vector<Candidate>
enumerateCandidates(const ProblemSize &p, linalg::LinalgOp linalgOpOrNull,
                    bool enableFootprintInference, const SolveOptions &opts,
                    const std::vector<int64_t> &mnTiles,
                    const std::vector<int64_t> &kTiles) {
  [[maybe_unused]] auto span = [&]() -> Tracer::Span {
    if (!opts.tracer)
      return Tracer::Span();
    llvm::json::Object f;
    f["M"] = p.m;
    f["N"] = p.n;
    f["K"] = p.k;
    f["mn_tiles"] = static_cast<int64_t>(mnTiles.size());
    f["k_tiles"] = static_cast<int64_t>(kTiles.size());
    f["register_level"] = opts.enableRegisterLevelSchedule;
    f["perfect_tiling"] = opts.requirePerfectTiling;
    return opts.tracer->span("solver.enumerate_candidates", std::move(f));
  }();
  std::vector<Candidate> out;
  const ArchConfig &arch = opts.arch;
  bool assumeFusedRelu = opts.assumeFusedRelu;
  bool requirePerfectTiling = opts.requirePerfectTiling;

  std::vector<int64_t> threadList = opts.candidatesThreadMN;
  if (threadList.empty())
    threadList.push_back(1);
  llvm::sort(threadList);
  threadList.erase(std::unique(threadList.begin(), threadList.end()),
                   threadList.end());

  auto emitCandidate = [&](int64_t tm, int64_t tn, int64_t tk, Traffic traffic,
                           int64_t blocksTotal, int64_t smemPerBlock,
                           int64_t threadM, int64_t threadN) {
    Candidate c;
    c.tileM = tm;
    c.tileN = tn;
    c.tileK = tk;
    c.loopTileExtents = {tm, tn, tk};
    c.threadTileM = threadM;
    c.threadTileN = threadN;
    c.traffic = traffic;
    c.smemBytes = smemPerBlock;
    c.blocksM =
        requirePerfectTiling ? (p.m / tm) : ceilDiv(p.m, std::max<int64_t>(1, tm));
    c.blocksN =
        requirePerfectTiling ? (p.n / tn) : ceilDiv(p.n, std::max<int64_t>(1, tn));
    c.blocksTotal = blocksTotal;

    // Occupancy 启发式（论文风格）：min(smem, threads, regs, maxBlocksPerSM)。
    int64_t blocksPerSM = std::max<int64_t>(1, arch.maxBlocksPerSM);
    if (smemPerBlock > 0)
      blocksPerSM =
          std::min<int64_t>(blocksPerSM,
                            std::max<int64_t>(1, arch.smemBytes / smemPerBlock));

    if (threadM > 0 && threadN > 0) {
      if (tm % threadM != 0 || tn % threadN != 0)
        return;
      int64_t blockDimX = tn / threadN;
      int64_t blockDimY = tm / threadM;
      int64_t threads = blockDimX * blockDimY;
      if (threads <= 0 || threads > 1024)
        return;
      int64_t byThreads = std::max<int64_t>(
          1, arch.maxThreadsPerSM / std::max<int64_t>(1, threads));
      blocksPerSM = std::min<int64_t>(blocksPerSM, byThreads);

      int64_t regsAcc = threadM * threadN;
      int64_t regsOverhead = 32;
      int64_t regsPerThread = std::max<int64_t>(1, regsAcc + regsOverhead);
      regsPerThread = std::min<int64_t>(regsPerThread, arch.maxRegistersPerThread);
      c.estRegsPerThread = regsPerThread;
      int64_t regsPerBlock = regsPerThread * threads;
      if (regsPerBlock > 0) {
        int64_t byRegs =
            std::max<int64_t>(1, arch.maxRegistersPerSM / regsPerBlock);
        blocksPerSM = std::min<int64_t>(blocksPerSM, byRegs);
      }
    }
    c.blocksPerSM = std::max<int64_t>(1, blocksPerSM);

    int64_t concurrentBlocks =
        std::max<int64_t>(1, c.blocksPerSM * arch.numSM);
    c.numWave = ceilDiv(std::max<int64_t>(1, blocksTotal), concurrentBlocks);
    c.score = traffic.totalBytes() * static_cast<double>(c.numWave);
    out.push_back(std::move(c));
  };

  for (int64_t tm : mnTiles) {
    for (int64_t tn : mnTiles) {
      for (int64_t tk : kTiles) {
        if (tm <= 0 || tn <= 0 || tk <= 0)
          continue;
        if (tm > p.m || tn > p.n || tk > p.k)
          continue;

        if (requirePerfectTiling) {
          if (p.m % tm != 0 || p.n % tn != 0 || p.k % tk != 0)
            continue;
        }

        int64_t blocksM = requirePerfectTiling ? (p.m / tm) : ceilDiv(p.m, tm);
        int64_t blocksN = requirePerfectTiling ? (p.n / tn) : ceilDiv(p.n, tn);
        int64_t blocksTotal = blocksM * blocksN;

        // shared memory 占用：仅对 A/B tile promotion 计入 padding。
        // 这样可保持 matmul 专用路径与 codegen 旋钮一致。
        int64_t pad = std::max<int64_t>(0, opts.profile.workgroupPadLastDim);
        if (opts.codegenSearch.enable)
          pad = 0; // actual pad is handled per expanded 候选 later
        int64_t smem = (tm * (tk + pad) + tk * (tn + pad)) * arch.elementBytes;
        if (smem <= 0 || smem > arch.smemBytes)
          continue;

        Traffic traffic = computeTrafficBytes(p, tm, tn, assumeFusedRelu, arch);
        if (enableFootprintInference && linalgOpOrNull) {
          if (auto via = computeTrafficBytesViaFootprint(
                  linalgOpOrNull, tm, tn, tk, assumeFusedRelu, arch,
                  requirePerfectTiling)) {
            traffic = *via;
          }
        }

        if (opts.enableRegisterLevelSchedule) {
          for (int64_t ttm : threadList)
            for (int64_t ttn : threadList)
              emitCandidate(tm, tn, tk, traffic, blocksTotal, smem, ttm, ttn);
        } else {
          emitCandidate(tm, tn, tk, traffic, blocksTotal, smem, 0, 0);
        }
      }
    }
  }
  if (opts.tracer) {
    llvm::json::Object f;
    f["count"] = static_cast<int64_t>(out.size());
    opts.tracer->event("solver.enumerate_candidates.result", std::move(f));
  }
  return out;
}

static std::optional<Traffic>
computeTrafficBytesViaFootprintGenericSingleOp(linalg::LinalgOp op,
                                               const OpTile &tile,
                                               const ArchConfig &arch,
                                               bool requirePerfectTiling,
                                               bool applyCoalescingPenalty) {
  auto gridOpt = computeGridInfo(op, tile, requirePerfectTiling);
  if (!gridOpt)
    return std::nullopt;
  GridInfo grid = *gridOpt;

  LinalgIndexingMapsFootprintInference infer;
  auto fpOpt = infer.infer(op, tile);
  if (!fpOpt)
    return std::nullopt;

  double elemBytes = static_cast<double>(arch.elementBytes);
  int64_t txnReadElems = getTxnElemsForRead(arch);
  int64_t txnWriteElems = getTxnElemsForWrite(arch);

  double readBytes = 0.0;
  double writeBytes = 0.0;

  int numInputs = op.getNumDpsInputs();
  int numInits = op.getNumDpsInits();

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

  // 只统计 dpsInputs 的 global read（Phase 9/Phase A 风格，对齐 baseline）。
  for (int i = 0; i < numInputs; ++i) {
    if (i < 0 || i >= static_cast<int>(fpOpt->perOperand.size()))
      continue;
    Value fullVal = op.getDpsInputs()[i];
    double bytesPerTile = footprintBytesRead(fpOpt->perOperand[i], fullVal);
    if (bytesPerTile == 0.0)
      continue;
    bool dependsOnReduction = dependsOnAnyReductionDim(op, i, grid);
    int64_t hits = grid.blocksTotal *
                   (dependsOnReduction ? grid.reductionTiles : 1);
    readBytes += bytesPerTile * static_cast<double>(hits);
  }

  // 只统计 dpsInits 的最终写回（不计 init 的 read）。
  for (int i = 0; i < numInits; ++i) {
    int operandIdx = numInputs + i;
    if (operandIdx < 0 ||
        operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
      continue;
    Value fullVal = op.getDpsInits()[i];
    double bytesPerTile =
        footprintBytesWrite(fpOpt->perOperand[operandIdx], fullVal);
    if (bytesPerTile == 0.0)
      continue;
    writeBytes += bytesPerTile * static_cast<double>(grid.blocksTotal);
  }

  // bytesA=总读，bytesB=0，bytesC=总写。
  return Traffic{readBytes, 0.0, writeBytes};
}

static std::vector<Candidate>
enumerateCandidatesGeneric(const GenericProblem &prob, const SolveOptions &opts) {
  [[maybe_unused]] auto span = [&]() -> Tracer::Span {
    if (!opts.tracer)
      return Tracer::Span();
    llvm::json::Object f;
    f["op"] = prob.getOpName();
    f["loops"] = static_cast<int64_t>(prob.loops.size());
    f["auto_candidates"] = opts.autoCandidates;
    f["register_level"] = opts.enableRegisterLevelSchedule;
    f["perfect_tiling"] = opts.requirePerfectTiling;
    return opts.tracer->span("solver.enumerate_candidates_generic", std::move(f));
  }();
  std::vector<Candidate> out;
  if (!prob.targetOp)
    return out;

  auto op = dyn_cast_or_null<linalg::LinalgOp>(prob.targetOp);
  if (!op)
    return out;
  if (static_cast<int64_t>(prob.loops.size()) != op.getNumLoops())
    return out;

  // 1) 找到 parallel/reduction loop 的索引（按 iterator_types 顺序）。
  std::vector<int> pDims;
  std::vector<int> rDims;
  pDims.reserve(prob.loops.size());
  rDims.reserve(prob.loops.size());

  for (int i = 0; i < static_cast<int>(prob.loops.size()); ++i) {
    auto type = prob.loops[i].type;
    if (type == utils::IteratorType::parallel)
      pDims.push_back(i);
    else if (type == utils::IteratorType::reduction)
      rDims.push_back(i);
    else
      return out; // Phase 9 第一版：只支持 parallel/reduction。
  }

  auto makeAutoList = [&](int64_t extent, bool isReduction)
      -> std::vector<int64_t> {
    (void)isReduction;
    std::vector<int64_t> out;
    if (extent <= 0) {
      out.push_back(1);
      return out;
    }
    auto add = [&](int64_t v) {
      if (v <= 0 || v > extent)
        return;
      if (opts.requirePerfectTiling && (extent % v != 0))
        return;
      out.push_back(v);
    };

    add(1);
    for (int64_t v = 2; v > 0 && v <= extent; v *= 2)
      add(v);
    const int64_t cand[] = {8, 12, 16, 20, 24, 28, 32, 40, 48, 56,
                            64, 80, 96, 112, 128, 160, 192, 224, 256};
    for (int64_t v : cand)
      add(v);
    add(extent);

    llvm::sort(out);
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  };

  std::vector<int64_t> mn0List = opts.candidatesMN;
  std::vector<int64_t> mn1List = opts.candidatesMN;
  // 每个归约维的候选列表（论文对齐：rDims 可能包含多轴）。
  std::vector<std::vector<int64_t>> rLists;
  rLists.reserve(rDims.size());
  std::vector<std::vector<int64_t>> pLists;
  pLists.reserve(pDims.size());

  if (opts.autoCandidates) {
    if (!pDims.empty())
      mn0List = makeAutoList(prob.loops[pDims[0]].size, /*isReduction=*/false);
    if (pDims.size() >= 2)
      mn1List = makeAutoList(prob.loops[pDims[1]].size, /*isReduction=*/false);
    for (int pd : pDims)
      pLists.push_back(makeAutoList(prob.loops[pd].size, /*isReduction=*/false));
    for (int rd : rDims)
      rLists.push_back(makeAutoList(prob.loops[rd].size, /*isReduction=*/true));
  } else {
    for (size_t i = 0; i < rDims.size(); ++i)
      rLists.push_back(opts.candidatesK);
    if (!pDims.empty())
      pLists.push_back(mn0List);
    if (pDims.size() >= 2)
      pLists.push_back(mn1List);
    for (size_t i = 2; i < pDims.size(); ++i)
      pLists.push_back({prob.loops[pDims[i]].size});
  }

  auto normalizeList = [&](std::vector<int64_t> &xs, int64_t extent) {
    std::vector<int64_t> tmp;
    tmp.reserve(xs.size() + 2);
    for (int64_t v : xs) {
      if (v <= 0 || v > extent)
        continue;
      if (opts.requirePerfectTiling && extent > 0 && (extent % v != 0))
        continue;
      tmp.push_back(v);
    }
    if (extent > 0)
      tmp.push_back(extent);
    tmp.push_back(1);
    llvm::sort(tmp);
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    // 控制笛卡尔积爆炸：仅保留少量小值和最大值。
    if (tmp.size() > 8) {
      std::vector<int64_t> keep;
      keep.reserve(8);
      for (size_t i = 0; i < 7; ++i)
        keep.push_back(tmp[i]);
      keep.push_back(tmp.back());
      tmp.swap(keep);
    }
    xs.swap(tmp);
  };

  for (size_t i = 0; i < rLists.size(); ++i) {
    int rd = rDims[i];
    normalizeList(rLists[i], prob.loops[rd].size);
  }
  for (size_t i = 0; i < pLists.size(); ++i) {
    int pd = pDims[i];
    normalizeList(pLists[i], prob.loops[pd].size);
  }
  if (mn0List.empty())
    mn0List.push_back(1);
  if (mn1List.empty())
    mn1List = mn0List;
  if (!pDims.empty() && pLists.empty())
    pLists.push_back(mn0List);
  if (pDims.size() >= 2 && pLists.size() < 2)
    pLists.push_back(mn1List);

  std::vector<int64_t> threadList = opts.candidatesThreadMN;
  if (threadList.empty())
    threadList.push_back(1);
  llvm::sort(threadList);
  threadList.erase(std::unique(threadList.begin(), threadList.end()),
                   threadList.end());

  auto tilesAlong = [&](int dim, int64_t tile) -> int64_t {
    int64_t full = prob.loops[dim].size;
    if (tile <= 0)
      return 1;
    return opts.requirePerfectTiling ? (full / tile) : ceilDiv(full, tile);
  };

  auto emitForChoices = [&](ArrayRef<int64_t> pChoice,
                            ArrayRef<int64_t> rChoice) {
    // 2) 构造每个 loop 的 tile extent：默认 full size（不切分）。
    std::vector<int64_t> loopTile(prob.loops.size(), 1);
    for (int i = 0; i < static_cast<int>(prob.loops.size()); ++i) {
      loopTile[i] = prob.loops[i].size;
    }

    for (size_t j = 0; j < pChoice.size(); ++j)
      loopTile[pDims[j]] = pChoice[j];
    for (size_t j = 0; j < rChoice.size(); ++j)
      loopTile[rDims[j]] = rChoice[j];

    // 3) 边界/整除剪枝。
    bool skip = false;
    for (int i = 0; i < static_cast<int>(prob.loops.size()); ++i) {
      int64_t full = prob.loops[i].size;
      int64_t t = loopTile[i];
      if (t <= 0 || full <= 0 || t > full) {
        skip = true;
        break;
      }
      if (opts.requirePerfectTiling && (full % t != 0)) {
        skip = true;
        break;
      }
    }
    if (skip)
      return;

    OpTile tile;
    tile.loopExtents = loopTile;

    auto gridOpt = computeGridInfo(op, tile, opts.requirePerfectTiling);
    if (!gridOpt)
      return;
    GridInfo grid = *gridOpt;

    auto trafficOpt = computeTrafficBytesViaFootprintGenericSingleOp(
        op, tile, opts.arch, opts.requirePerfectTiling,
        /* applyCoalescingPenalty=*/opts.enableCoalescingPenalty);
    if (!trafficOpt)
      return;
    Traffic traffic = *trafficOpt;

    const int64_t tileM = !pChoice.empty() ? pChoice[0] : 1;
    const int64_t tileN = (pChoice.size() >= 2) ? pChoice[1] : 1;

    auto emitCandidate = [&](int64_t threadM, int64_t threadN) {
      int64_t blocksPerSM = std::max<int64_t>(1, opts.arch.maxBlocksPerSM);
      if (threadM > 0 && threadN > 0) {
        if (tileM % threadM != 0 || tileN % threadN != 0)
          return;
        int64_t blockDimX = tileN / threadN;
        int64_t blockDimY = tileM / threadM;
        int64_t threads = blockDimX * blockDimY;
        if (threads <= 0 || threads > 1024)
          return;
        int64_t byThreads =
            std::max<int64_t>(1, opts.arch.maxThreadsPerSM /
                                     std::max<int64_t>(1, threads));
        blocksPerSM = std::max<int64_t>(
            1, std::min<int64_t>(blocksPerSM, byThreads));
      }

      int64_t concurrentBlocks =
          std::max<int64_t>(1, blocksPerSM * opts.arch.numSM);
      int64_t waves = ceilDiv(grid.blocksTotal, concurrentBlocks);
      double score = traffic.totalBytes() * static_cast<double>(waves);

      Candidate c;
      c.tileM = tileM;
      c.tileN = tileN;
      c.tileK = !rChoice.empty() ? rChoice[0] : 1;
      c.threadTileM = threadM;
      c.threadTileN = threadN;
      c.loopTileExtents = loopTile;
      c.smemBytes = 0;
      c.blocksM = !pChoice.empty() ? tilesAlong(pDims[0], c.tileM) : 1;
      c.blocksN = (pChoice.size() >= 2) ? tilesAlong(pDims[1], c.tileN) : 1;
      c.blocksTotal = grid.blocksTotal;
      c.blocksPerSM = blocksPerSM;
      c.numWave = waves;
      c.traffic = traffic;
      c.score = score;
      out.push_back(std::move(c));
    };

    if (opts.enableRegisterLevelSchedule) {
      for (int64_t ttm : threadList)
        for (int64_t ttn : threadList)
          emitCandidate(ttm, ttn);
    } else {
      emitCandidate(/*threadM=*/0, /*threadN=*/0);
    }
  };

  if (pDims.size() <= 2) {
    for (int64_t tx : mn0List) {
      for (int64_t ty : mn1List) {
      llvm::SmallVector<int64_t, 4> rChoice(rDims.size(), 1);
      auto enumR = [&](auto &&self, size_t idx) -> void {
        if (idx == rDims.size()) {
          llvm::SmallVector<int64_t, 4> pChoice;
          if (!pDims.empty())
            pChoice.push_back(tx);
          if (pDims.size() >= 2)
            pChoice.push_back(ty);
          emitForChoices(pChoice, rChoice);
        return;
        }

        for (int64_t v : rLists[idx]) {
          rChoice[idx] = v;
          self(self, idx + 1);
        }
      };
      enumR(enumR, 0);
      }
    }
  } else {
    // ND 并行维枚举（论文 EnumerateSubtiles 变体）。
    std::vector<int64_t> pChoice(pDims.size(), 1);
    const size_t maxComb = 512;
    size_t combCount = 0;

    auto enumP = [&](auto &&self, size_t idx) -> void {
      if (combCount >= maxComb)
        return;
      if (idx == pDims.size()) {
        llvm::SmallVector<int64_t, 4> rChoice(rDims.size(), 1);
        auto enumR = [&](auto &&selfR, size_t ridx) -> void {
          if (combCount >= maxComb)
            return;
          if (ridx == rDims.size()) {
            emitForChoices(pChoice, rChoice);
            ++combCount;
            return;
          }
          for (int64_t v : rLists[ridx]) {
            rChoice[ridx] = v;
            selfR(selfR, ridx + 1);
            if (combCount >= maxComb)
              return;
          }
        };
        enumR(enumR, 0);
        return;
      }

      if (idx >= pLists.size() || pLists[idx].empty())
        return;
      for (int64_t v : pLists[idx]) {
        pChoice[idx] = v;
        self(self, idx + 1);
        if (combCount >= maxComb)
          return;
      }
    };

    enumP(enumP, 0);
  }
  if (opts.tracer) {
    llvm::json::Object f;
    f["count"] = static_cast<int64_t>(out.size());
    opts.tracer->event("solver.enumerate_candidates_generic.result", std::move(f));
  }
  return out;
}

#include "WelderSolverPaperSolvePath.h"
#include "WelderSolverSolveEntrypoints.h"

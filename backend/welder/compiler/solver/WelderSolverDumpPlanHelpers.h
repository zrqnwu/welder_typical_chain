static std::string jsonEscape(llvm::StringRef s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        // 其余控制字符统一丢弃，简化处理。
        out += ' ';
      } else {
        out += c;
      }
      break;
    }
  }
  return out;
}

static std::string valueToString(Value v) {
  if (!v)
    return "<null>";
  std::string s;
  llvm::raw_string_ostream os(s);
  v.print(os);
  return os.str();
}

static std::optional<int> findGraphSinkNode(const TileGraph &g) {
  int sink = -1;
  for (int i = 0; i < static_cast<int>(g.nodes.size()); ++i) {
    if (g.nodes[i].outEdges.empty() &&
        !isTrivialOpFor2LevelFootprint(g.nodes[i].op)) {
      sink = i;
    }
  }
  if (sink < 0)
    return std::nullopt;
  return sink;
}

struct SharedAllocEvent {
  std::string action; // "alloc" 或 "free"
  std::string kind;   // "external_input" 或 "internal_reuse"
  std::string value;
  int64_t bytes = 0;
  int64_t offset = 0;
  int64_t padLastDim = 0;
  int64_t swizzleXor = 0;
  int nodeIdx = -1; // -1 表示 prologue 的 LoadTiles。
};

struct SharedAllocPlan {
  int64_t peakBytes = 0;
  llvm::SmallVector<SharedAllocEvent, 128> events;
};

static SharedAllocPlan computeSharedAllocPlanBestFitPaper(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, int maxLevelInclusive, int64_t workgroupPadLastDim,
    bool workgroupPadLastDimMatmulOnly, int64_t workgroupMultiBufferDepth,
    int64_t workgroupSwizzleXor, const Candidate *cand) {
  SharedAllocPlan plan;
  // 这里需要与 `computeSharedFootprintBestFitPaper` 保持一致。
  double elemBytesD = static_cast<double>(arch.elementBytes);
  SharedLayoutPolicyV1 layout = buildSharedLayoutPolicyV1(
      graph, sg, minLevelExclusive, maxLevelInclusive, workgroupPadLastDim,
      workgroupPadLastDimMatmulOnly, workgroupSwizzleXor);
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

  llvm::SmallVector<int, 16> topo =
      topoSortSubgraphByConnectedEdges(graph, sg, minLevelExclusive);

  llvm::DenseMap<const void *, int64_t> bufferBytes;
  llvm::DenseMap<const void *, int> remainingUses;
  llvm::DenseSet<const void *> externalInputs;
  llvm::DenseMap<const void *, Value> keyToValue;
  llvm::DenseMap<const void *, std::string> keyToKind;

  // 内部复用 tile。
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
    // Multi-buffering 只作用于 async global->shared staging buffer
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
    if (!keyToValue.count(key))
      keyToValue.insert({key, e.value});
    if (!keyToKind.count(key))
      keyToKind.insert({key, "internal_reuse"});
  }

  // 外部输入。
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
      if (operandIdx < 0 ||
          operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
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
      if (!keyToValue.count(key))
        keyToValue.insert({key, v});
      if (!keyToKind.count(key))
        keyToKind.insert({key, "external_input"});
    }
  }

  BestFitAllocator alloc;
  llvm::DenseMap<const void *, int64_t> liveOffset;
  constexpr int64_t kAlign = 32;

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

  auto doAlloc = [&](const void *key, int nodeIdx) {
    if (liveOffset.count(key))
      return;
    auto it = bufferBytes.find(key);
    if (it == bufferBytes.end())
      return;
    int64_t bytes = it->second;
    int64_t off = alloc.allocate(bytes, kAlign);
    liveOffset.insert({key, off});
    SharedAllocEvent ev;
    ev.action = "alloc";
    auto itKind = keyToKind.find(key);
    ev.kind = itKind == keyToKind.end() ? "unknown" : itKind->second;
    auto itVal = keyToValue.find(key);
    ev.value = itVal == keyToValue.end() ? "<unknown>" : valueToString(itVal->second);
    ev.bytes = bytes;
    ev.offset = off;
    SharedLayoutInfo li = layout.get(key);
    ev.padLastDim = li.padLastDim;
    ev.swizzleXor = li.swizzleXor;
    ev.nodeIdx = nodeIdx;
    plan.events.push_back(std::move(ev));
  };

  auto doFreeIfDead = [&](const void *key, int nodeIdx) {
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
    SharedAllocEvent ev;
    ev.action = "free";
    auto itKind = keyToKind.find(key);
    ev.kind = itKind == keyToKind.end() ? "unknown" : itKind->second;
    auto itVal = keyToValue.find(key);
    ev.value = itVal == keyToValue.end() ? "<unknown>" : valueToString(itVal->second);
    ev.bytes = itBytes->second;
    ev.offset = itOff->second;
    SharedLayoutInfo li = layout.get(key);
    ev.padLastDim = li.padLastDim;
    ev.swizzleXor = li.swizzleXor;
    ev.nodeIdx = nodeIdx;
    plan.events.push_back(std::move(ev));
    liveOffset.erase(itOff);
  };

  for (const void *key : externalInputs)
    doAlloc(key, /*nodeIdx=*/-1);

  for (int nodeIdx : topo) {
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
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;
      const void *key = e.value.getAsOpaquePointer();
      doAlloc(key, nodeIdx);
    }

    // 记录行归约 scratch 的 alloc/free 事件（见 `computeSharedFootprintBestFitPaper`）。
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
          SharedAllocEvent ev;
          ev.action = "alloc";
          ev.kind = "row_reduction_scratch";
          ev.value = "<row_reduction_scratch>";
          ev.bytes = scratchBytes;
          ev.offset = scratchOff;
          ev.padLastDim = 0;
          ev.swizzleXor = 0;
          ev.nodeIdx = nodeIdx;
          plan.events.push_back(std::move(ev));
        }
      }
    }

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
              if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
                continue;
              if (sg.inSet.contains(e.src)) {
                hasConnectedInEdge = true;
                break;
              }
            }
            if (hasConnectedInEdge)
              continue;
            Value v = op.getDpsInputs()[operandIdx];
            doFreeIfDead(v.getAsOpaquePointer(), nodeIdx);
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
      if (maxLevelInclusive >= 0 && e.connectLevel > maxLevelInclusive)
        continue;
      if (!sg.inSet.contains(e.src))
        continue;
      if (isTrivialOpFor2LevelFootprint(graph.nodes[e.src].op) ||
          isTrivialOpFor2LevelFootprint(graph.nodes[e.dst].op))
        continue;
      const void *key = e.value.getAsOpaquePointer();
      doFreeIfDead(key, nodeIdx);
    }

    if (scratchBytes > 0) {
      alloc.free(scratchOff, scratchBytes, kAlign);
      SharedAllocEvent ev;
      ev.action = "free";
      ev.kind = "row_reduction_scratch";
      ev.value = "<row_reduction_scratch>";
      ev.bytes = scratchBytes;
      ev.offset = scratchOff;
      ev.padLastDim = 0;
      ev.swizzleXor = 0;
      ev.nodeIdx = nodeIdx;
      plan.events.push_back(std::move(ev));
    }
  }

  plan.peakBytes = alloc.highWatermark;
  return plan;
}

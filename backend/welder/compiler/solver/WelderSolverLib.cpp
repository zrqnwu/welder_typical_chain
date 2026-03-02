#include "WelderSolverLib.h"

#include "AffineIntervalEvaluator.h"
#include "WelderTrace.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;

namespace welder {

static constexpr int64_t kUnknown = -1;

static std::string formatTraffic(const Traffic &t) {
  std::string s;
  s.reserve(128);
  s.append("A=");
  s.append(std::to_string(t.bytesA));
  s.append(" B=");
  s.append(std::to_string(t.bytesB));
  s.append(" C=");
  s.append(std::to_string(t.bytesC));
  s.append(" Cut=");
  s.append(std::to_string(t.bytesCut));
  s.append(" Total=");
  s.append(std::to_string(t.totalBytes()));
  return s;
}

std::string CostBreakdown::toString() const {
  std::string s;
  s.reserve(512);
  s.append("raw{");
  s.append(formatTraffic(rawTraffic));
  s.append("} mem{");
  s.append(formatTraffic(memTraffic));
  s.append("} sh2reg=");
  s.append(std::to_string(sharedToRegBytes));
  s.append("} waves=");
  s.append(std::to_string(std::max<int64_t>(1, waves)));
  s.append(" blocks=");
  s.append(std::to_string(std::max<int64_t>(1, blocksTotal)));
  s.append(" blocksPerSM=");
  s.append(std::to_string(std::max<int64_t>(1, blocksPerSM)));
  s.append(" smemFootprint=");
  s.append(std::to_string(std::max<int64_t>(0, sharedFootprintBytes)));
  s.append(" underutil=");
  s.append(std::to_string(underutilPenalty));
  s.append(" bank=");
  s.append(std::to_string(bankConflictFactor));
  s.append(" reg=");
  s.append(std::to_string(regPenalty));
  s.append(" est=");
  s.append(std::to_string(estimatedLatency));
  if (profiledMs) {
    s.append(" profiledMs=");
    s.append(std::to_string(*profiledMs));
  }
  return s;
}

static int64_t ceilDiv(int64_t a, int64_t b) {
  if (b == 0)
    return 0;
  return (a + b - 1) / b;
}

// 前置声明（供论文对齐辅助函数在定义前使用）。
static llvm::SmallVector<int64_t, 4> getStaticShapeOrUnknown(Value v);
static int64_t coalescedFactor(ArrayRef<int64_t> subtensor,
                               ArrayRef<int64_t> full);
struct PaperSubgraph;
static std::vector<Candidate> enumerateSharedTilesPaperDfs2D(
    const TileGraph &graph, const PaperSubgraph &sg, linalg::LinalgOp sinkOp,
    int sinkNodeIdx, const SolveOptions &opts, const FootprintInference &inference,
    const std::vector<std::vector<int64_t>> &reduceTilesByNode);
static double computeSharedToRegTrafficBytesForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling,
    int minLevelExclusive, int maxLevelInclusive, int64_t workgroupPadLastDim,
    bool workgroupPadLastDimMatmulOnly, int64_t workgroupSwizzleXor);
static int64_t getEnvInt64OrDefault(const char *name, int64_t defaultValue);
static double getEnvDoubleOrDefault(const char *name, double defaultValue);

//===----------------------------------------------------------------------===//
// TileGraph 构建（最小实现）
//===----------------------------------------------------------------------===//

struct StrippedValueChain {
  Value base;
  llvm::SmallVector<Operation *, 2> viewOps; // consumer->producer order
};

static StrippedValueChain stripViewLikeTensorOps(Value v) {
  StrippedValueChain out;
  out.base = v;
  while (true) {
    if (auto cast = out.base.getDefiningOp<tensor::CastOp>()) {
      out.base = cast.getSource();
      continue;
    }
    if (auto expand = out.base.getDefiningOp<tensor::ExpandShapeOp>()) {
      out.viewOps.push_back(expand.getOperation());
      out.base = expand.getSrc();
      continue;
    }
    if (auto collapse = out.base.getDefiningOp<tensor::CollapseShapeOp>()) {
      out.viewOps.push_back(collapse.getOperation());
      out.base = collapse.getSrc();
      continue;
    }
    if (auto slice = out.base.getDefiningOp<tensor::ExtractSliceOp>()) {
      out.viewOps.push_back(slice.getOperation());
      out.base = slice.getSource();
      continue;
    }
    break;
  }
  return out;
}

static std::optional<bool> inferSwapXYHintForLinalgOpImpl(linalg::LinalgOp op) {
  if (!op)
    return std::nullopt;

  // 仅建模常见的 2D block 映射：前两个并行循环作为空间维，
  // 映射到 `gpu.block_id.{x,y}`。
  if (op.getNumParallelLoops() < 2)
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> parallelLoopIdx;
  parallelLoopIdx.reserve(2);
  auto iters = op.getIteratorTypesArray();
  for (int64_t i = 0; i < op.getNumLoops(); ++i) {
    if (iters[i] == mlir::utils::IteratorType::parallel) {
      parallelLoopIdx.push_back(i);
      if (parallelLoopIdx.size() == 2)
        break;
    }
  }
  if (parallelLoopIdx.size() < 2)
    return std::nullopt;
  const int64_t p0 = parallelLoopIdx[0];
  const int64_t p1 = parallelLoopIdx[1];

  // 优先使用第一个输出结果对应的 indexing map。
  if (op.getNumDpsInits() < 1)
    return std::nullopt;

  AffineMap outMap;
  // LinalgOp 是接口封装，indexing map 的常规顺序为
  // `[inputs..., outputs...]`，这里选第一个输出 map。
  auto maps = op.getIndexingMapsArray();
  unsigned firstOut = op.getNumDpsInputs();
  if (firstOut >= maps.size())
    return std::nullopt;
  outMap = maps[firstOut];
  if (!outMap)
    return std::nullopt;

  // 将 block_id.x 映射到索引“最内层（最后一维）输出维”的循环维，
  // 因为它在 row-major 下通常是连续维。对 transpose 类 op，
  // 该规则常会触发 p0/p1 交换。
  int64_t outRank = outMap.getNumResults();
  if (outRank < 2)
    return std::nullopt;
  AffineExpr last = outMap.getResult(outRank - 1);

  llvm::SmallDenseSet<int64_t, 8> usedDims;
  last.walk([&](AffineExpr e) {
    if (auto d = dyn_cast<AffineDimExpr>(e))
      usedDims.insert(d.getPosition());
  });

  // 仅当最内层输出维恰好由前两个并行循环中的一个索引时，
  // 才推断 swap/no-swap；否则保守返回 unknown
  //（论文对齐：不猜测无法自证的 remap）。
  bool usesP0 = usedDims.contains(p0);
  bool usesP1 = usedDims.contains(p1);
  if (usesP0 && !usesP1)
    return true; // need swap so p0 maps to block<x>.
  if (usesP1 && !usesP0)
    return false; // 默认 mapping already maps p1 -> block<x>.
  return std::nullopt;
}

std::optional<bool> inferSwapXYHintForLinalgOp(mlir::Operation *op) {
  auto lop = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!lop)
    return std::nullopt;
  return inferSwapXYHintForLinalgOpImpl(lop);
}

std::optional<TileGraph> buildLinalgTileGraph(ModuleOp module) {
  TileGraph graph;

  llvm::DenseMap<Operation *, int> opToNode;
  Builder b(module.getContext());
  module.walk([&](linalg::LinalgOp op) {
    TileGraphNode node;
    node.op = op.getOperation();
    node.swapXYHint = inferSwapXYHintForLinalgOpImpl(op);
    int idx = static_cast<int>(graph.nodes.size());
    graph.nodes.push_back(std::move(node));
    opToNode[op.getOperation()] = idx;

    // 分配模块内稳定 node id，便于外部工具（性能测量、缓存）使用，
    // 无需依赖内存中的 `Operation*` 身份。
    // 该属性可跨变换保留；若不查询则不影响 codegen。
    if (!op->hasAttr("welder.node_id"))
      op->setAttr("welder.node_id", b.getI64IntegerAttr(idx));
  });

  // 第二遍：通过 SSA use-def 连接 edges（producer result -> consumer operand）。
  for (int dst = 0; dst < static_cast<int>(graph.nodes.size()); ++dst) {
    auto *consumerOp = graph.nodes[dst].op;
    auto consumer = dyn_cast_or_null<linalg::LinalgOp>(consumerOp);
    if (!consumer)
      continue;

  int numOperands = static_cast<int>(consumerOp->getNumOperands());
  for (int operandIdx = 0; operandIdx < numOperands; ++operandIdx) {
      Value operand = consumerOp->getOperand(operandIdx);
      StrippedValueChain stripped = stripViewLikeTensorOps(operand);

      auto res = dyn_cast<OpResult>(stripped.base);
      if (!res)
        continue;

      Operation *producerOp = res.getOwner();
      auto it = opToNode.find(producerOp);
      if (it == opToNode.end())
        continue;

      int src = it->second;
      if (src < 0 || src >= static_cast<int>(graph.nodes.size()))
        continue;

      TileGraphEdge edge;
      edge.src = src;
      edge.dst = dst;
      edge.srcResult = static_cast<int>(res.getResultNumber());
      edge.dstOperand = operandIdx;
      edge.value = stripped.base;
      edge.viewOps.assign(stripped.viewOps.begin(), stripped.viewOps.end());

      int edgeIdx = static_cast<int>(graph.edges.size());
      graph.edges.push_back(std::move(edge));
      graph.nodes[src].outEdges.push_back(edgeIdx);
      graph.nodes[dst].inEdges.push_back(edgeIdx);
    }
  }

  return graph;
}

static bool tilesEqual(const OpTile &a, const OpTile &b) {
  // 当前最小实现：只比较 loopExtents。reductionSteps 暂未参与 propagation。
  return a.loopExtents == b.loopExtents;
}

static std::string opDebugName(Operation *op) {
  if (!op)
    return "<null>";
  return op->getName().getStringRef().str();
}

static void resetTileGraphState(TileGraph &graph, bool resetCutEdges) {
  for (auto &n : graph.nodes) {
    n.hasRequiredTile = false;
    n.requiredTile.loopExtents.clear();
    n.requiredTile.reductionSteps.clear();
  }
  for (auto &e : graph.edges) {
    e.footprint.indexBounds.clear();
    e.footprint.shape.clear();
    if (resetCutEdges && e.connectLevel <= kConnectLevelGlobal)
      e.connectLevel = kConnectLevelShared;
    syncCutFlagFromConnectLevel(e);
  }
}

static void syncCutFlagFromConnectLevel(TileGraph &graph) {
  // 论文语义：connectLevel==0 表示不连接（落地到 global），对当前 codegen 等价于 cut-edge。
  for (auto &e : graph.edges)
    syncCutFlagFromConnectLevel(e);
}

static std::optional<OpTile>
buildOpTileFromParallelExtents(linalg::LinalgOp op,
                               llvm::ArrayRef<int64_t> parallelExtents,
                               int64_t defaultReductionTile) {
  if (!op)
    return std::nullopt;
  if (static_cast<int64_t>(parallelExtents.size()) != op.getNumParallelLoops())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> staticLoopRanges = op.getStaticLoopRanges();
  if (static_cast<int64_t>(staticLoopRanges.size()) != op.getNumLoops())
    return std::nullopt;

  OpTile tile;
  tile.loopExtents.resize(op.getNumLoops(), 0);

  int64_t pIdx = 0;
  auto iters = op.getIteratorTypesArray();
  for (int64_t i = 0; i < op.getNumLoops(); ++i) {
    if (iters[i] == utils::IteratorType::parallel) {
      int64_t extent = parallelExtents[pIdx++];
      if (extent <= 0)
        return std::nullopt;
      tile.loopExtents[i] = extent;
      continue;
    }

    if (iters[i] == utils::IteratorType::reduction) {
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

    // 其它 iterator type（例如 window）暂不支持。
    return std::nullopt;
  }

  if (pIdx != static_cast<int64_t>(parallelExtents.size()))
    return std::nullopt;
  return tile;
}

static std::optional<OpTile> buildOpTileFromParallelExtentsWithReductionTiles(
    linalg::LinalgOp op, llvm::ArrayRef<int64_t> parallelExtents,
    llvm::ArrayRef<int64_t> reductionExtents, int64_t defaultReductionTile) {
  if (!op)
    return std::nullopt;
  if (static_cast<int64_t>(parallelExtents.size()) != op.getNumParallelLoops())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> staticLoopRanges = op.getStaticLoopRanges();
  if (static_cast<int64_t>(staticLoopRanges.size()) != op.getNumLoops())
    return std::nullopt;

  // 若提供归约列表，其长度必须与 `numReductionLoops` 一致。
  if (!reductionExtents.empty() &&
      static_cast<int64_t>(reductionExtents.size()) != op.getNumReductionLoops())
    return std::nullopt;

  OpTile tile;
  tile.loopExtents.resize(op.getNumLoops(), 0);

  int64_t pIdx = 0;
  int64_t rIdx = 0;
  auto iters = op.getIteratorTypesArray();
  for (int64_t i = 0; i < op.getNumLoops(); ++i) {
    if (iters[i] == utils::IteratorType::parallel) {
      int64_t full = staticLoopRanges[i];
      if (full == ShapedType::kDynamic || full <= 0)
        return std::nullopt;
      int64_t extent = parallelExtents[pIdx++];
      if (extent <= 0 || extent > full)
        return std::nullopt;
      tile.loopExtents[i] = extent;
      continue;
    }

    if (iters[i] == utils::IteratorType::reduction) {
      int64_t full = staticLoopRanges[i];
      if (full == ShapedType::kDynamic || full <= 0)
        return std::nullopt;

      int64_t extent = 0;
      if (!reductionExtents.empty()) {
        extent = reductionExtents[rIdx++];
      } else {
        extent = defaultReductionTile;
      }

      if (extent <= 0) {
        extent = full;
      }
      if (extent > full)
        extent = full;

      tile.loopExtents[i] = extent;
      continue;
    }

    // 其它 iterator type（例如 window）暂不支持。
    return std::nullopt;
  }

  if (pIdx != static_cast<int64_t>(parallelExtents.size()))
    return std::nullopt;
  if (!reductionExtents.empty() &&
      rIdx != static_cast<int64_t>(reductionExtents.size()))
    return std::nullopt;
  return tile;
}

// 根据输出 tensor footprint 反推所需的并行循环 tile extent。
//
// 已知：
// - `outShape`：输出 tensor 索引空间中的需求 extent（按输出维）
// - producer 的输出 indexing_map（loops -> 输出索引）
//
// 返回值按 producer 的并行循环顺序排列（供
// `buildOpTileFromParallelExtentsWithReductionTiles()` 使用）。
//
// 这是“Propagate v2”的核心：不要假设输出维与并行循环顺序一致
//（Welder 调度里 transpose/置换输出 map 很常见）。
static std::optional<llvm::SmallVector<int64_t, 8>>
inferParallelExtentsFromOutputFootprintShape(linalg::LinalgOp op, int outResultIdx,
                                             llvm::ArrayRef<int64_t> outShape) {
  if (!op)
    return std::nullopt;
  if (outResultIdx < 0)
    return std::nullopt;
  if (outResultIdx >= op.getNumDpsInits())
    return std::nullopt;

  auto maps = op.getIndexingMapsArray();
  const int outOperandIdx = op.getNumDpsInputs() + outResultIdx;
  if (outOperandIdx < 0 || outOperandIdx >= static_cast<int>(maps.size()))
    return std::nullopt;
  AffineMap outMap = maps[static_cast<size_t>(outOperandIdx)];
  if (!outMap)
    return std::nullopt;
  // 与最小 footprint 推导保持一致：symbol 需要额外区间约束，
  // 当前尚未建模。
  if (outMap.getNumSymbols() != 0)
    return std::nullopt;
  if (static_cast<int64_t>(outMap.getNumDims()) != op.getNumLoops())
    return std::nullopt;
  if (static_cast<int64_t>(outMap.getNumResults()) !=
      static_cast<int64_t>(outShape.size()))
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> staticLoopRanges = op.getStaticLoopRanges();
  if (static_cast<int64_t>(staticLoopRanges.size()) != op.getNumLoops())
    return std::nullopt;

  // loop-dim -> parallel-loop 索引（按 iterator 顺序）。
  llvm::SmallVector<int, 8> loopToParallel(op.getNumLoops(), -1);
  {
    int pIdx = 0;
    auto iters = op.getIteratorTypesArray();
    for (int i = 0; i < static_cast<int>(op.getNumLoops()); ++i) {
      if (iters[static_cast<size_t>(i)] == utils::IteratorType::parallel)
        loopToParallel[static_cast<size_t>(i)] = pIdx++;
    }
    if (pIdx != static_cast<int>(op.getNumParallelLoops()))
      return std::nullopt;
  }

  llvm::SmallVector<int64_t, 8> parallelExtents(op.getNumParallelLoops(), -1);

  auto extractLoopDim = [&](AffineExpr e) -> std::optional<int> {
    if (auto d = dyn_cast<AffineDimExpr>(e))
      return d.getPosition();
    if (auto c = dyn_cast<AffineConstantExpr>(e)) {
      (void)c;
      return std::nullopt;
    }
    auto bin = dyn_cast<AffineBinaryOpExpr>(e);
    if (!bin)
      return std::nullopt;

    // 处理常见的 `dim + constant` / `constant + dim` 模式。
    if (bin.getKind() == AffineExprKind::Add) {
      auto lhsD = dyn_cast<AffineDimExpr>(bin.getLHS());
      auto rhsD = dyn_cast<AffineDimExpr>(bin.getRHS());
      auto lhsC = dyn_cast<AffineConstantExpr>(bin.getLHS());
      auto rhsC = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (lhsD && rhsC)
        return lhsD.getPosition();
      if (rhsD && lhsC)
        return rhsD.getPosition();
    }
    return std::nullopt;
  };

  for (int outDim = 0; outDim < static_cast<int>(outShape.size()); ++outDim) {
    int64_t extent = outShape[static_cast<size_t>(outDim)];
    if (extent <= 0)
      return std::nullopt;

    AffineExpr e = outMap.getResult(outDim);
    if (auto c = dyn_cast<AffineConstantExpr>(e)) {
      // 输出维是常量时，footprint extent 必须为 1。
      if (extent != 1)
        return std::nullopt;
      continue;
    }
    std::optional<int> loopDimOpt = extractLoopDim(e);
    if (!loopDimOpt)
      return std::nullopt;
    int loopDim = *loopDimOpt;
    if (loopDim < 0 || loopDim >= static_cast<int>(loopToParallel.size()))
      return std::nullopt;
    int pIdx = loopToParallel[static_cast<size_t>(loopDim)];
    if (pIdx < 0)
      return std::nullopt; // output depends on a non-parallel loop (unsupported)

    int64_t full = staticLoopRanges[static_cast<size_t>(loopDim)];
    if (full == ShapedType::kDynamic || full <= 0)
      return std::nullopt;
    if (extent > full)
      return std::nullopt;

    int64_t &slot = parallelExtents[static_cast<size_t>(pIdx)];
    if (slot == -1) {
      slot = extent;
      continue;
    }
    if (slot != extent)
      return std::nullopt;
  }

  for (int64_t v : parallelExtents) {
    if (v <= 0)
      return std::nullopt;
  }
  return parallelExtents;
}

static std::optional<int64_t> getRankIfShaped(Value v) {
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank())
    return std::nullopt;
  return st.getRank();
}

static std::optional<llvm::SmallVector<int64_t, 8>>
getStaticShapeOrNullopt(Value v) {
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank())
    return std::nullopt;
  llvm::SmallVector<int64_t, 8> shape;
  shape.reserve(st.getRank());
  for (int64_t d : st.getShape()) {
    if (d == ShapedType::kDynamic || d <= 0)
      return std::nullopt;
    shape.push_back(d);
  }
  return shape;
}

static std::optional<llvm::SmallVector<int64_t, 8>>
mapFootprintShapeResultToSrc(tensor::ExpandShapeOp op,
                             llvm::ArrayRef<int64_t> resultTileShape) {
  if (!op)
    return std::nullopt;
  auto fullResOpt = getStaticShapeOrNullopt(op.getResult());
  auto fullSrcOpt = getStaticShapeOrNullopt(op.getSrc());
  if (!fullResOpt || !fullSrcOpt)
    return std::nullopt;
  auto &fullRes = *fullResOpt;
  auto &fullSrc = *fullSrcOpt;
  if (resultTileShape.size() != fullRes.size())
    return std::nullopt;

  auto reassociation = op.getReassociationIndices();
  if (reassociation.size() != fullSrc.size())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> srcTile;
  srcTile.resize(fullSrc.size(), 1);

  for (size_t srcDim = 0; srcDim < reassociation.size(); ++srcDim) {
    auto group = reassociation[srcDim];
    if (group.empty())
      return std::nullopt;

    int64_t stride = 1;
    int64_t maxLinear = 0;
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      int64_t resDim = *it;
      if (resDim < 0 || static_cast<size_t>(resDim) >= fullRes.size())
        return std::nullopt;
      int64_t full = fullRes[static_cast<size_t>(resDim)];
      if (full <= 0)
        return std::nullopt;
      int64_t t = resultTileShape[static_cast<size_t>(resDim)];
      if (t <= 0)
        return std::nullopt;
      if (t > full)
        t = full;

      int64_t term = 0;
      if (t > 0 && t - 1 > 0) {
        if (stride > 0 &&
            (t - 1) > (std::numeric_limits<int64_t>::max() / stride))
          return std::nullopt;
        term = (t - 1) * stride;
      }
      if (term > 0 &&
          maxLinear > (std::numeric_limits<int64_t>::max() - term))
        return std::nullopt;
      maxLinear += term;

      if (stride > 0 && full > (std::numeric_limits<int64_t>::max() / stride))
        return std::nullopt;
      stride *= full;
    }

    int64_t extent = maxLinear + 1;
    if (extent <= 0)
      extent = 1;
    extent = std::min<int64_t>(extent, fullSrc[srcDim]);
    srcTile[srcDim] = extent;
  }

  return srcTile;
}

static std::optional<llvm::SmallVector<int64_t, 8>>
mapFootprintShapeResultToSrc(tensor::CollapseShapeOp op,
                             llvm::ArrayRef<int64_t> resultTileShape) {
  if (!op)
    return std::nullopt;
  auto fullResOpt = getStaticShapeOrNullopt(op.getResult());
  auto fullSrcOpt = getStaticShapeOrNullopt(op.getSrc());
  if (!fullResOpt || !fullSrcOpt)
    return std::nullopt;
  auto &fullRes = *fullResOpt;
  auto &fullSrc = *fullSrcOpt;
  if (resultTileShape.size() != fullRes.size())
    return std::nullopt;

  auto reassociation = op.getReassociationIndices();
  if (reassociation.size() != fullRes.size())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> srcTile;
  srcTile.resize(fullSrc.size(), 1);

  for (size_t resDim = 0; resDim < reassociation.size(); ++resDim) {
    auto group = reassociation[resDim];
    if (group.empty())
      return std::nullopt;

    int64_t fullOut = fullRes[resDim];
    if (fullOut <= 0)
      return std::nullopt;
    int64_t t = resultTileShape[resDim];
    if (t <= 0)
      return std::nullopt;
    if (t > fullOut)
      t = fullOut;

    int64_t stride = 1;
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      int64_t srcDim = *it;
      if (srcDim < 0 || static_cast<size_t>(srcDim) >= fullSrc.size())
        return std::nullopt;
      int64_t full = fullSrc[static_cast<size_t>(srcDim)];
      if (full <= 0)
        return std::nullopt;

      int64_t extent = ceilDiv(t, stride);
      extent = std::min<int64_t>(extent, full);
      extent = std::max<int64_t>(1, extent);
      srcTile[static_cast<size_t>(srcDim)] = extent;

      if (stride > 0 && full > (std::numeric_limits<int64_t>::max() / stride))
        return std::nullopt;
      stride *= full;
    }
  }

  return srcTile;
}

static std::optional<llvm::SmallVector<int64_t, 8>>
mapFootprintShapeResultToSrc(tensor::ExtractSliceOp op,
                             llvm::ArrayRef<int64_t> resultTileShape) {
  if (!op)
    return std::nullopt;
  auto srcRankOpt = getRankIfShaped(op.getSource());
  auto resRankOpt = getRankIfShaped(op.getResult());
  if (!srcRankOpt || !resRankOpt)
    return std::nullopt;
  if (*srcRankOpt != *resRankOpt)
    return std::nullopt;
  if (resultTileShape.size() != static_cast<size_t>(*resRankOpt))
    return std::nullopt;

  // 若能拿到静态结果 shape，则对结果做上限裁剪
  //（extract_slice 的结果类型常携带 tile 形状，即使 op 尺寸是动态的）。
  auto fullResOpt = getStaticShapeOrNullopt(op.getResult());

  // 优先使用静态 slice size；若 slice size 为动态，
  // 则回退到需求结果 footprint extent（保守但对传播/代价评估安全）。
  auto staticSizes = op.getStaticSizes();
  if (static_cast<size_t>(staticSizes.size()) != resultTileShape.size())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> srcTile(resultTileShape.begin(),
                                        resultTileShape.end());
  for (size_t i = 0; i < srcTile.size(); ++i) {
    int64_t t = srcTile[i];
    if (t <= 0)
      return std::nullopt;
    if (fullResOpt) {
      if (fullResOpt->size() != srcTile.size())
        return std::nullopt;
      t = std::min<int64_t>(t, (*fullResOpt)[i]);
    }
    int64_t s = staticSizes[static_cast<unsigned>(i)];
    if (s != ShapedType::kDynamic && s > 0)
      t = std::min<int64_t>(t, s);
    srcTile[i] = t;
  }

  // 若可获得静态源 shape，则继续做上限裁剪。
  if (auto fullSrcOpt = getStaticShapeOrNullopt(op.getSource())) {
    if (fullSrcOpt->size() != srcTile.size())
      return std::nullopt;
    for (size_t i = 0; i < srcTile.size(); ++i)
      srcTile[i] = std::min<int64_t>(srcTile[i], (*fullSrcOpt)[i]);
  }

  return srcTile;
}

static std::optional<llvm::SmallVector<int64_t, 8>>
mapFootprintShapeThroughViewOps(llvm::ArrayRef<int64_t> shape,
                                llvm::ArrayRef<Operation *> viewOps) {
  llvm::SmallVector<int64_t, 8> cur(shape.begin(), shape.end());
  for (Operation *op : viewOps) {
    if (!op)
      return std::nullopt;
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      auto nextOpt = mapFootprintShapeResultToSrc(expand, cur);
      if (!nextOpt)
        return std::nullopt;
      cur = std::move(*nextOpt);
      continue;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      auto nextOpt = mapFootprintShapeResultToSrc(collapse, cur);
      if (!nextOpt)
        return std::nullopt;
      cur = std::move(*nextOpt);
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      auto srcRankOpt = getRankIfShaped(cast.getSource());
      auto resRankOpt = getRankIfShaped(cast.getResult());
      if (!srcRankOpt || !resRankOpt)
        return std::nullopt;
      if (static_cast<size_t>(*resRankOpt) != cur.size())
        return std::nullopt;
      if (*srcRankOpt != *resRankOpt)
        return std::nullopt;
      // 仅形状 cast 视为恒等映射；若源维静态则做上限裁剪。
      if (auto fullSrcOpt = getStaticShapeOrNullopt(cast.getSource())) {
        if (fullSrcOpt->size() != cur.size())
          return std::nullopt;
        for (size_t i = 0; i < cur.size(); ++i)
          cur[i] = std::min<int64_t>(cur[i], (*fullSrcOpt)[i]);
      }
      continue;
    }
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
      auto nextOpt = mapFootprintShapeResultToSrc(slice, cur);
      if (!nextOpt)
        return std::nullopt;
      cur = std::move(*nextOpt);
      continue;
    }
    return std::nullopt;
  }
  return cur;
}

//===----------------------------------------------------------------------===//
// Phase 14（论文对齐）：
// GraphConnecting + SubGraphTiling 辅助逻辑。
//
// 
// - Welder 论文中的调度流程是“先连边（Graph Connecting），再做子图切分
//   （即 Scheduling / SubGraphTiling）”。
// - 当前实现已经覆盖：
//   1) shared 级别（global<->shared）的 Propagate/cut-edge/MemTraffic/MemFootprint；
//   2) register 级别的有限递归窗口（maxConnectLevel>2 时按 stage window 逐层估计，
//      并可通过 paperRecursiveMaxStages 做深度上限）。
// - 这仍是“有限递归”版本：默认保持兼容行为，按需放开更多递归层。
//===----------------------------------------------------------------------===//

// 前置声明：Phase A 的全图流量记账（定义在后面）。
static Traffic computeGlobalTrafficAssumingFullyFused(
    const TileGraph &graph, const ArchConfig &arch,
    const FootprintInference &inference, bool requirePerfectTiling);

// 前置声明：2-level shared footprint 估算（定义在后面）。
static int64_t estimateSharedFootprintBytes2Level(const TileGraph &graph,
                                                  const ArchConfig &arch);
static std::optional<int>
pickCutEdgeForSharedFootprint2Level(const TileGraph &graph,
                                    const ArchConfig &arch);

// 把 (tileM,tileN) 映射到 root op 的“前两维 parallel loop”，其它 parallel loop 默认 full。
// 这与 solveGeneric / compiler 里的启发式保持一致。
#include "WelderSolverTwoLevelCore.h"

//===----------------------------------------------------------------------===//
// 论文对齐：Figure 7 的 Two-step tile-graph 调度
//（global<->shared 的 GraphConnecting + SubGraphTiling）。
//
// 
// - 当前 repo 的 codegen 仍以 global<->shared 为主线（并在 solver 侧建模更高层 connectLevel）。
// - 论文里的 d.Profile(configs) 在本 repo 里由可选的 CUDA event profiling 闭环实现：
//   通过 opts.profile.enable 触发 compile->PTX/NVVM + welder-profiler 计时。
//   未开启 profiling 时，仍用 traffic-based latency 估算作为 ranking key。
//===----------------------------------------------------------------------===//

// 本文件后续辅助函数的前置声明。
static double getVolume(const OperandFootprint &fp);
static llvm::SmallVector<int64_t, 4> getStaticShapeOrUnknown(Value v);
static llvm::SmallVector<int64_t, 4> getStaticStridesOrEmpty(Value v);
static double coalescedTensorElements(ArrayRef<int64_t> subtensor,
                                      ArrayRef<int64_t> full,
                                      ArrayRef<int64_t> fullStrides,
                                      int64_t transactionElements);
static double coalescedTensorElements(ArrayRef<int64_t> subtensor,
                                      ArrayRef<int64_t> full,
                                      int64_t transactionElements);
static bool isTrivialOpFor2LevelFootprint(Operation *op);
static std::vector<Candidate> enumerateCandidatesGeneric(const GenericProblem &prob,
                                                         const SolveOptions &opts);

#include "WelderSolverCandidatePolicyHelpers.h"
#include "WelderSolverTrafficCostCore.h"
#include "WelderSolverProfileRetryHelpers.h"
#include "WelderSolverProfileCompileAndRun.h"
static void applyFixedCodegenKnobsFromProfile(Candidate &c,
                                              const SolveOptions &opts) {
  // 当 `CodegenSearchSpace` 关闭时，性能测量旋钮直接来自 `opts.profile`。
  c.enableAsyncCopy = opts.profile.enableAsyncCopy;
  c.asyncBypassL1 = opts.profile.asyncBypassL1;
  c.enableSoftwarePipelining = opts.profile.enableSoftwarePipelining;
  c.pipelineDepth = opts.profile.pipelineDepth;
  c.pipelinePeelEpilogue = opts.profile.pipelinePeelEpilogue;
  c.pipelineSetAsyncWaitGroups = opts.profile.pipelineSetAsyncWaitGroups;
  c.workgroupMultiBufferDepth = opts.profile.workgroupMultiBufferDepth;
  c.workgroupPadLastDim = opts.profile.workgroupPadLastDim;
  c.workgroupPadLastDimMatmulOnly = opts.profile.workgroupPadLastDimMatmulOnly;
  c.swapBlockDims = opts.profile.swapBlockDims;
  c.workgroupSwizzleXor = opts.profile.workgroupSwizzleXor;
  c.enableTensorCoreF16 = opts.profile.enableTensorCoreF16;
  c.enableTensorCoreTf32 = false;
  c.blockRasterizeMode = opts.profile.blockRasterizeMode;
  c.blockRasterizePanelWidth = opts.profile.blockRasterizePanelWidth;

  c.enableRowReductionChainReuseFusion =
      opts.profile.enableRowReductionChainReuseFusion;
  c.enableRowReductionInputPromotion = opts.profile.enableRowReductionInputPromotion;
  c.enableRowReductionInputPromotionVectorize =
      opts.profile.enableRowReductionInputPromotionVectorize;
  c.enableRowReductionWarp = opts.profile.enableRowReductionWarp;
  c.enableRowReductionVectorize = opts.profile.enableRowReductionVectorize;
  c.rowReductionVectorWidth = opts.profile.rowReductionVectorWidth;
  c.rowReductionThreadsX = opts.profile.rowReductionThreadsX;
  c.enableRowReductionRelaxBarriers = opts.profile.enableRowReductionRelaxBarriers;
  c.enableRowReductionSkipCombineBarrier =
      opts.profile.enableRowReductionSkipCombineBarrier;
  c.rowReductionInputVectorWidth = opts.profile.rowReductionInputVectorWidth;
  c.enableRowReductionCombineVectorize =
      opts.profile.enableRowReductionCombineVectorize;
  c.enableMatmulSoftmaxSharedReuseFusion =
      opts.profile.enableMatmulSoftmaxSharedReuseFusion;

  c.useCutlassMma = false;
  c.mmaM = 0;
  c.mmaN = 0;
  c.mmaK = 0;
  if (c.enableTensorCoreF16 || c.enableTensorCoreTf32) {
    if (c.enableTensorCoreF16 && c.tileK >= 32)
      c.useCutlassMma = true;
    chooseMmaShapeForCandidate(c);
    if (c.mmaM <= 0 || c.mmaN <= 0 || c.mmaK <= 0) {
      c.enableTensorCoreF16 = false;
      c.enableTensorCoreTf32 = false;
      c.useCutlassMma = false;
    } else {
      int64_t warps = (c.tileM / c.mmaM) * (c.tileN / c.mmaN);
      if (warps <= 0 || warps * 32 > 1024) {
        c.enableTensorCoreF16 = false;
        c.enableTensorCoreTf32 = false;
        c.useCutlassMma = false;
        c.mmaM = 0;
        c.mmaN = 0;
        c.mmaK = 0;
      }
    }
  }
}

static double estimateMatmulSharedBankConflictFactor(const Candidate &cand,
                                                     const ArchConfig &arch);

static double simScore(double a, double b) {
  // 2ab/(a^2+b^2) in [0,1], used by the reference policy for 合并访问 match.
  double denom = a * a + b * b;
  if (denom == 0.0)
    return 0.0;
  return (2.0 * a * b) / denom;
}

static llvm::SmallVector<int64_t, 4>
getReductionLoopFullRanges(linalg::LinalgOp op) {
  llvm::SmallVector<int64_t, 4> red;
  if (!op)
    return red;
  llvm::SmallVector<int64_t, 8> ranges = op.getStaticLoopRanges();
  if (static_cast<int64_t>(ranges.size()) != op.getNumLoops())
    return red;
  auto iters = op.getIteratorTypesArray();
  for (int i = 0; i < static_cast<int>(iters.size()); ++i) {
    if (iters[i] != utils::IteratorType::reduction)
      continue;
    int64_t full = ranges[i];
    if (full == ShapedType::kDynamic || full <= 0)
      continue;
    red.push_back(full);
  }
  return red;
}

static llvm::SmallVector<int64_t, 64> getAllFactorsSorted(int64_t n) {
  llvm::SmallVector<int64_t, 64> fs;
  if (n <= 0)
    return fs;
  for (int64_t i = 1; i * i <= n; ++i) {
    if (n % i != 0)
      continue;
    fs.push_back(i);
    if (i != n / i)
      fs.push_back(n / i);
  }
  llvm::sort(fs);
  fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
  return fs;
}

static llvm::SmallVector<int64_t, 64> factorsOrPowersForReduceStep(int64_t full) {
  // 对齐 python `DefaultPolicy.get_node_reduce_step` 的候选构造：
  // - 优先使用精确因子；
  // - 对较大的“近似质数”维度，回退到不超过 full 的 2 的幂序列。
  llvm::SmallVector<int64_t, 64> fs = getAllFactorsSorted(full);
  if (fs.size() == 2 && full > 64) {
    fs.clear();
    for (int64_t v = 1; v > 0 && v * 2 < full; v *= 2)
      fs.push_back(v);
    fs.push_back(std::max<int64_t>(1, full));
  }
  if (fs.empty())
    fs.push_back(1);
  if (fs.front() != 1)
    fs.insert(fs.begin(), 1);
  llvm::sort(fs);
  fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
  return fs;
}

static std::vector<std::vector<int64_t>> assignReduceTilesByCoalescingPaper(
    const TileGraph &graph, const ArchConfig &arch,
    const FootprintInference &inference) {
  // 论文/Welder 对齐: DefaultPolicy._assign_reduce_step.
  // 为每个 op 选择归约步长，以提升 placeholder 输入上的 global read 合并访问。
  std::vector<std::vector<int64_t>> out;
  out.resize(graph.nodes.size());

  int64_t txnElems = getTxnElemsForRead(arch);

  for (int nodeIdx = 0; nodeIdx < static_cast<int>(graph.nodes.size()); ++nodeIdx) {
    Operation *op0 = graph.nodes[nodeIdx].op;
    auto op = dyn_cast_or_null<linalg::LinalgOp>(op0);
    if (!op)
      continue;
    if (op.getNumReductionLoops() <= 0)
      continue;

    llvm::SmallVector<int64_t, 4> redFull = getReductionLoopFullRanges(op);
    if (redFull.empty())
      continue;

    llvm::SmallVector<llvm::SmallVector<int64_t, 64>, 4> allSteps;
    allSteps.reserve(redFull.size());
    for (int64_t full : redFull)
      allSteps.push_back(factorsOrPowersForReduceStep(full));

    // 并行维 extent 固定为 1（与参考策略一致）。
    llvm::SmallVector<int64_t, 8> par(op.getNumParallelLoops(), 1);

    auto scoreForStepIds = [&](ArrayRef<int> stepIds) -> double {
      llvm::SmallVector<int64_t, 4> red;
      red.reserve(stepIds.size());
      for (size_t i = 0; i < stepIds.size(); ++i) {
        int id = stepIds[i];
        if (id < 0 || id >= static_cast<int>(allSteps[i].size()))
          return 0.0;
        red.push_back(allSteps[i][id]);
      }

      auto tileOpt = buildOpTileFromParallelExtentsWithReductionTiles(
          op, par, red, /*defaultReductionTile=*/0);
      if (!tileOpt)
        return 0.0;
      auto fpOpt = inference.infer(op0, *tileOpt);
      if (!fpOpt)
        return 0.0;

      double score = 0.0;
      int numInputs = op.getNumDpsInputs();
      for (int operandIdx = 0; operandIdx < numInputs; ++operandIdx) {
        // 若该 operand 没有 producer 边，则视为 placeholder 输入。
        bool hasProducer = false;
        for (int edgeIdx : graph.nodes[nodeIdx].inEdges) {
          if (edgeIdx < 0 || edgeIdx >= static_cast<int>(graph.edges.size()))
            continue;
          if (graph.edges[edgeIdx].dstOperand == operandIdx) {
            hasProducer = true;
            break;
          }
        }
        if (hasProducer)
          continue;

        if (operandIdx < 0 ||
            operandIdx >= static_cast<int>(fpOpt->perOperand.size()))
          continue;
        const OperandFootprint &ofp = fpOpt->perOperand[operandIdx];
        llvm::SmallVector<int64_t, 4> fullShape =
            getStaticShapeOrUnknown(op.getDpsInputs()[operandIdx]);
        if (fullShape.empty())
          continue;
        int64_t cf = coalescedFactor(ArrayRef<int64_t>(ofp.shape), fullShape);
        if (cf <= 0)
          continue;
        score += simScore(static_cast<double>(cf), static_cast<double>(txnElems));
      }
      return score;
    };

    llvm::SmallVector<int, 4> curIds;
    curIds.resize(allSteps.size(), 0);
    double curScore = scoreForStepIds(curIds);

    // If we can't score 合并访问 (e.g. footprint inference unsupported for
    // this op), fall back to a simple 事务-aligned 归约 step.
    // 这样可避免 `tileK=1` 这类“能编译但极慢”的病态选择，
    // 也符合参考策略“挑一个合理 rstep”的意图。
    if (!(curScore > 0.0)) {
      std::vector<int64_t> picked;
      picked.reserve(redFull.size());
      for (size_t ax = 0; ax < redFull.size(); ++ax) {
        int64_t full = redFull[ax];
        int64_t best = 1;
        // 优先选择 `<= txnElems` 的最大因子，在保持归约轴完美切分的同时，
        // 尽量提升每个事务的连续读取长度。
        for (int64_t s : allSteps[ax]) {
          if (s <= 0 || s > full)
            continue;
          if (full % s != 0)
            continue;
          if (s <= txnElems && s > best)
            best = s;
        }
        picked.push_back(std::max<int64_t>(1, best));
      }
      out[nodeIdx] = std::move(picked);
      continue;
    }

    while (curScore > 0.0) {
      llvm::SmallVector<int, 4> bestIds = curIds;
      double bestScore = curScore;

      for (size_t ax = 0; ax < curIds.size(); ++ax) {
        if (curIds[ax] + 1 >= static_cast<int>(allSteps[ax].size()))
          continue;
        llvm::SmallVector<int, 4> trial = curIds;
        trial[ax] += 1;
        double s = scoreForStepIds(trial);
        if (s > bestScore) {
          bestScore = s;
          bestIds = std::move(trial);
        }
      }

      if (bestScore <= curScore)
        break;
      curIds = bestIds;
      curScore = bestScore;
    }

    std::vector<int64_t> picked;
    picked.reserve(curIds.size());
    for (size_t i = 0; i < curIds.size(); ++i)
      picked.push_back(allSteps[i][curIds[i]]);
    out[nodeIdx] = std::move(picked);
  }

  return out;
}

static std::vector<Candidate>
expandCandidatesWithCodegenSearch(const Candidate &base,
                                 const SolveOptions &opts,
                                 int64_t maxRowReductionExtentForTc,
                                 bool tensorCoreLayoutFeasible,
                                 bool matmulSoftmaxLikeSubgraph,
                                 bool tensorCoreCapableSubgraph) {
  std::vector<Candidate> out;
  struct PrefilterStats {
    int64_t generated = 0;
    int64_t kept = 0;
    int64_t droppedTcDowngraded = 0;
    int64_t droppedRefreshFail = 0;
    int64_t droppedDuplicate = 0;
    int64_t droppedIllegalWaitGroup = 0;
    int64_t droppedIllegalPipeline = 0;
    int64_t droppedIllegalSwizzle = 0;
    int64_t droppedIllegalRasterConflict = 0;
    int64_t droppedIllegalRasterParam = 0;
  } prefilterStats;
  const bool strictLegalityPrefilter =
      opts.codegenSearch.enable &&
      (getEnvInt64OrDefault("WELDER_PREFILTER_STRICT_LEGALITY", 1) != 0);
  std::unordered_set<std::string> emittedVariantKeys;
  emittedVariantKeys.reserve(256);
  auto buildVariantDedupKey = [](const Candidate &c) -> std::string {
    std::string key;
    key.reserve(256);
    auto appendI64 = [&](int64_t v) {
      key.append(std::to_string(v));
      key.push_back('|');
    };
    auto appendBool = [&](bool v) { key.push_back(v ? '1' : '0'); };
    appendI64(c.tileM);
    appendI64(c.tileN);
    appendI64(c.tileK);
    appendI64(c.threadTileM);
    appendI64(c.threadTileN);
    appendBool(c.enableAsyncCopy);
    appendBool(c.asyncBypassL1);
    appendBool(c.enableSoftwarePipelining);
    appendI64(c.pipelineDepth);
    appendBool(c.pipelinePeelEpilogue);
    appendBool(c.pipelineSetAsyncWaitGroups);
    appendI64(c.workgroupMultiBufferDepth);
    appendI64(c.workgroupPadLastDim);
    appendBool(c.workgroupPadLastDimMatmulOnly);
    appendI64(c.workgroupSwizzleXor);
    appendI64(c.blockRasterizeXor);
    appendI64(c.blockRasterizeMode);
    appendI64(c.blockRasterizePanelWidth);
    appendBool(c.swapBlockDims);
    appendBool(c.enableRowReductionChainReuseFusion);
    appendBool(c.enableRowReductionInputPromotion);
    appendBool(c.enableRowReductionInputPromotionVectorize);
    appendBool(c.enableRowReductionWarp);
    appendBool(c.enableRowReductionVectorize);
    appendI64(c.rowReductionVectorWidth);
    appendI64(c.rowReductionThreadsX);
    appendBool(c.enableRowReductionRelaxBarriers);
    appendBool(c.enableRowReductionSkipCombineBarrier);
    appendI64(c.rowReductionInputVectorWidth);
    appendBool(c.enableRowReductionCombineVectorize);
    appendBool(c.enableMatmulSoftmaxSharedReuseFusion);
    appendBool(c.enableTensorCoreTf32);
    appendBool(c.enableTensorCoreF16);
    appendBool(c.useCutlassMma);
    appendI64(c.mmaM);
    appendI64(c.mmaN);
    appendI64(c.mmaK);
    return key;
  };
  auto emitPrefilterTrace = [&]() {
    if (!opts.tracer || !opts.codegenSearch.enable)
      return;
    const int64_t totalDropped =
        prefilterStats.droppedTcDowngraded + prefilterStats.droppedRefreshFail +
        prefilterStats.droppedDuplicate + prefilterStats.droppedIllegalWaitGroup +
        prefilterStats.droppedIllegalPipeline +
        prefilterStats.droppedIllegalSwizzle +
        prefilterStats.droppedIllegalRasterConflict +
        prefilterStats.droppedIllegalRasterParam;
    const bool traceAll =
        getEnvInt64OrDefault("WELDER_PREFILTER_TRACE_ALL", 0) != 0;
    if (!traceAll && totalDropped <= 0)
      return;
    llvm::json::Object f;
    f["strict_legality_prefilter"] = strictLegalityPrefilter;
    f["generated"] = prefilterStats.generated;
    f["kept"] = prefilterStats.kept;
    f["dropped"] = totalDropped;
    f["dropped_tc_downgraded"] = prefilterStats.droppedTcDowngraded;
    f["dropped_refresh_fail"] = prefilterStats.droppedRefreshFail;
    f["dropped_duplicate"] = prefilterStats.droppedDuplicate;
    f["dropped_illegal_wait_group"] = prefilterStats.droppedIllegalWaitGroup;
    f["dropped_illegal_pipeline"] = prefilterStats.droppedIllegalPipeline;
    f["dropped_illegal_swizzle"] = prefilterStats.droppedIllegalSwizzle;
    f["dropped_illegal_raster_conflict"] =
        prefilterStats.droppedIllegalRasterConflict;
    f["dropped_illegal_raster_param"] = prefilterStats.droppedIllegalRasterParam;
    opts.tracer->event("paper.codegen_prefilter", std::move(f),
                       /* isVerbose=*/true);
  };
  // 论文/Welder 对齐：TensorCore 的 TCPolicy 风格可行性检查。
  //
  // 参考实现使用更完整的 layout/stride_map 体系。本 MLIR 版本仅近似建模
  // 对 TensorCore 调度可实现性影响最大的约束；若不满足，则清除
  // TensorCore 标志并回退到 SIMT 路径。
  //
  // feasibilityCode 含义（候选级）：
  // 0：合法（无问题）
  // 1：TensorCore MMA 对齐违规
  // 2：TensorCore layout/stride-map 约束违规（近似）
  // 3：block 线程数溢出（warps*32 > 1024）
  // 27：非 matmul 子图请求了 TensorCore
  // 28：mm_sm 链路中 TensorCore f16 的 tileK 超出保护上限
  auto applyTensorCoreFeasibilityOrDowngrade = [&](Candidate &c) {
    if (!(c.enableTensorCoreF16 || c.enableTensorCoreTf32))
      return;
    int64_t origThreadTileM = c.threadTileM;
    int64_t origThreadTileN = c.threadTileN;
    auto downgradeToSimt = [&](int64_t code) {
      if (c.feasibilityCode == 0)
        c.feasibilityCode = code;
      c.enableTensorCoreF16 = false;
      c.enableTensorCoreTf32 = false;
      c.useCutlassMma = false;
      c.mmaM = 0;
      c.mmaN = 0;
      c.mmaK = 0;
      if (origThreadTileM > 0 && origThreadTileN > 0) {
        c.threadTileM = origThreadTileM;
        c.threadTileN = origThreadTileN;
      }
    };

    // 论文/Welder 对齐：TCPolicy 在 shared 布局上使用 offset=8。
    if (c.workgroupPadLastDim == 0)
      c.workgroupPadLastDim = 8;
    // TensorCore 调度仅对 matmul 输入应用 padding。
    c.workgroupPadLastDimMatmulOnly = true;

    // TensorCore lowering 需要子图中存在真实的 matmul 负载。
    if (!tensorCoreCapableSubgraph) {
      downgradeToSimt(/*code=*/27);
      return;
    }

    if (matmulSoftmaxLikeSubgraph && c.enableTensorCoreF16) {
      const int64_t maxTcTileKF16 = std::max<int64_t>(
          16, getEnvInt64OrDefault("WELDER_MM_SM_TC_F16_MAX_TILE_K", 16));
      if (c.tileK > maxTcTileKF16) {
        downgradeToSimt(/*code=*/28);
        return;
      }
    }

    if (!tensorCoreLayoutFeasible) {
      downgradeToSimt(/*code=*/2);
      return;
    }

    chooseMmaShapeForCandidate(c);
    int64_t mmaM = c.mmaM > 0 ? c.mmaM : 16;
    int64_t mmaN = c.mmaN > 0 ? c.mmaN : 8;
    int64_t mmaK =
        c.mmaK > 0 ? c.mmaK : (c.enableTensorCoreF16 ? int64_t(16) : int64_t(4));
    int64_t effTileN = c.tileN;
    if (maxRowReductionExtentForTc > 1)
      effTileN = std::max<int64_t>(effTileN, maxRowReductionExtentForTc);

    // 基础 MMA 对齐检查。
    if (c.tileM <= 0 || c.tileN <= 0 || c.tileK <= 0 || mmaM <= 0 || mmaN <= 0 ||
        mmaK <= 0 || (c.tileM % mmaM) != 0 || (effTileN % mmaN) != 0 ||
        (c.tileK % mmaK) != 0) {
      downgradeToSimt(/*code=*/1);
      return;
    }

    // 论文/Welder 对齐说明：
    // 参考 TCPolicy 通过 stride_map 判断 TC 布局是否可实现。
    // 本 MLIR 原型未在此完整建模这些约束，后续
    // `transform.nvgpu.rewrite_matmul_as_mma_sync` + 编译步骤会拒绝非法布局；
    // 因此这里保持候选级过滤最小化，避免误剪掉全部 TC-F16 变体。

    // 保持 thread-tile 语义与编译器 TensorCore launch 映射一致：
    // block 使用 `(tileM/mmaM)*(tileN/mmaN)` 个 warp。
    //
    // 对融合了行归约阶段的 kernel，当前 codegen 在非交换场景下
    // `threadTile=(1,4)` 效果更稳；这里同步该默认值，
    // 让 solver 的占用率估计与实际编译结果一致。
    auto tcThreads =
        computeTensorCoreBlockThreadsForCodegen(c, maxRowReductionExtentForTc);
    if (!tcThreads || *tcThreads <= 0 || *tcThreads > 1024) {
      downgradeToSimt(/*code=*/3);
      return;
    }

    if (!c.swapBlockDims) {
      c.threadTileM = 1;
      c.threadTileN = 4;
    } else {
      c.threadTileM = 4;
      c.threadTileN = 1;
    }
  };

  // 非 TensorCore 路径的 codegen 可行性检查与归一化。
  //
  // 某些旋钮仅在特定条件下有效（如 XOR 重排要求最后一维为 2 的幂；
  // pipelining 需要足够的 multi-buffer 深度）。这里会把非法组合
  // 归一化到安全配置，并记录 feasibilityCode，便于 dump 解释禁用原因。
  //
  // feasibilityCode 含义（旋钮级）：
  // 10：workgroupSwizzleXor 不适用（已禁用）
  // 11：blockRasterizeMode/panel 非法（已禁用）
  // 12：blockRasterizeXor 与 2D 光栅化冲突（已禁用）
  // 13：software pipelining 深度非法（已禁用）
  // 30：wait_group 组合非法（strict prefilter 丢弃）
  // 31：software-pipeline 组合非法（strict prefilter 丢弃）
  // 32：XOR 重排组合非法（strict prefilter 丢弃）
  // 33：光栅化冲突非法（strict prefilter 丢弃）
  // 34：光栅化参数非法（strict prefilter 丢弃）
  // 29：matmul-softmax tensorcore 的 pad-last-dim 被归一化到统一布局
  // 23：matmul-softmax 行向量化组合被归一化（anti-spill）
  // 24：matmul-softmax pipelining 被 anti-spill 启发式拒绝
  // 25：matmul-softmax async-copy 被稳定性保护禁用
  // 21：software pipelining 被占用率/寄存器启发式拒绝
  // 22：software pipelining 被 TensorCore 占用率启发式拒绝
  auto normalizeCodegenKnobsOrAnnotate = [&](Candidate &c) {
    auto isPow2 = [&](int64_t v) -> bool { return v > 0 && (v & (v - 1)) == 0; };
    auto canSwizzleLastDim = [&](int64_t lastDim, int64_t swizzle) -> bool {
      if (swizzle <= 1)
        return false;
      if (lastDim <= 1)
        return false;
      if (!isPow2(swizzle))
        return false;
      if (!isPow2(lastDim))
        return false;
      return swizzle <= lastDim;
    };
    // XOR 光栅化与 2D 光栅化保持互斥。
    if (c.blockRasterizeXor != 0 &&
        (c.blockRasterizeMode != 0 || c.blockRasterizePanelWidth != 0)) {
      c.blockRasterizeXor = 0;
      if (c.feasibilityCode == 0)
        c.feasibilityCode = 12;
    }

    // 校验 2D 光栅化参数。
    if (c.blockRasterizeMode != 0 || c.blockRasterizePanelWidth != 0) {
      bool ok = (c.blockRasterizeMode == 1 || c.blockRasterizeMode == 2) &&
                (c.blockRasterizePanelWidth > 0 &&
                 c.blockRasterizePanelWidth <= 16);
      if (!ok) {
        c.blockRasterizeMode = 0;
        c.blockRasterizePanelWidth = 0;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 11;
      }
    }

    // 校验 software pipelining 前提条件。
    if (c.enableSoftwarePipelining) {
      if (!c.enableAsyncCopy) {
        c.enableSoftwarePipelining = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 13;
      }
      if (c.pipelineDepth < 2)
        c.pipelineDepth = 2;
      if (c.workgroupMultiBufferDepth < 2 ||
          c.workgroupMultiBufferDepth < c.pipelineDepth) {
        c.enableSoftwarePipelining = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 13;
      }
    }
    if (!c.enableSoftwarePipelining)
      c.pipelineSetAsyncWaitGroups = false;

    // 校验 XOR 重排是否适用（近似按 A/B shared 布局判断）。
    if (c.workgroupSwizzleXor != 0) {
      // padding 通过 strided view 实现，不改变重排 pass 看到的逻辑最后一维。
      // 此检查需与 `WorkgroupAllocToLaunch.cpp` 的可用性规则一致：
      // 最后一维必须为 2 的幂，且 >= 重排因子。
      int64_t aLast = c.tileK; // A: [M,K]
      int64_t bLast = c.tileN; // B: [K,N]
      bool okA = canSwizzleLastDim(aLast, c.workgroupSwizzleXor);
      bool okB = canSwizzleLastDim(bLast, c.workgroupSwizzleXor);
      if (!(okA || okB)) {
        c.workgroupSwizzleXor = 0;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 10;
      }
    }

    // Matmul->Softmax 行链 anti-spill 归一化：
    // 将最脆弱的向量化组合移出默认搜索路径
    //（它们常能编译/融合，但会因 local spill 明显退化）。
    if (matmulSoftmaxLikeSubgraph &&
        (c.enableTensorCoreF16 || c.enableTensorCoreTf32) &&
        getEnvInt64OrDefault("WELDER_MM_SM_TC_FORCE_SHARED_REUSE", 1) != 0) {
      c.enableMatmulSoftmaxSharedReuseFusion = true;
    }
    const bool mmSmSharedReuse = c.enableMatmulSoftmaxSharedReuseFusion;
    bool mmSmRowChain =
        mmSmSharedReuse && c.enableRowReductionChainReuseFusion;
    if (mmSmRowChain) {
      // F16 matmul->softmax 原生链安全保护：
      // 当前单 kernel 行链 lowering 可能复制大中间量，
      // 导致严重 local-memory spill 与性能回退。
      // 这里保留可关闭路径，在完善 f16 行链 lowering 前先保证基线稳定。
      const bool disableF16RowChainReuse =
          (opts.arch.elementBytes <= 2) &&
          (getEnvInt64OrDefault("WELDER_MM_SM_F16_DISABLE_ROW_CHAIN_REUSE", 0) !=
           0);
      if (disableF16RowChainReuse) {
        c.enableRowReductionChainReuseFusion = false;
        c.enableRowReductionInputPromotion = false;
        c.enableRowReductionInputPromotionVectorize = false;
        c.enableRowReductionWarp = false;
        c.enableRowReductionVectorize = false;
        c.rowReductionVectorWidth = 0;
        c.rowReductionThreadsX = 0;
        c.enableRowReductionRelaxBarriers = false;
        c.enableRowReductionSkipCombineBarrier = false;
        c.rowReductionInputVectorWidth = 0;
        c.enableRowReductionCombineVectorize = false;
        mmSmRowChain = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 26;
      }
    }
    if (mmSmSharedReuse) {
      const bool tcCand = c.enableTensorCoreF16 || c.enableTensorCoreTf32;
      const bool forceUniformTcPadLayout =
          tcCand && c.enableRowReductionChainReuseFusion &&
          c.workgroupPadLastDim > 0 && c.workgroupPadLastDimMatmulOnly &&
          (getEnvInt64OrDefault("WELDER_MM_SM_TC_FORCE_UNIFORM_PAD_LAYOUT", 1) !=
           0);
      if (forceUniformTcPadLayout) {
        c.workgroupPadLastDimMatmulOnly = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 29;
      }
      const bool disableDualRowVectorize =
          getEnvInt64OrDefault("WELDER_MM_SM_DISABLE_DUAL_ROW_VECTORIZE", 1) != 0;
      if (disableDualRowVectorize &&
          c.enableRowReductionInputPromotionVectorize &&
          (c.enableRowReductionVectorize ||
           c.enableRowReductionCombineVectorize)) {
        c.enableRowReductionVectorize = false;
        c.enableRowReductionCombineVectorize = false;
        c.rowReductionVectorWidth = 0;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }

      const int64_t maxRowVecWidth = std::max<int64_t>(
          1, getEnvInt64OrDefault("WELDER_MM_SM_MAX_ROW_VEC_WIDTH", 4));
      const int64_t maxRowInputVecWidth = std::max<int64_t>(
          1, getEnvInt64OrDefault("WELDER_MM_SM_MAX_ROW_INPUT_VEC_WIDTH", 4));
      const int64_t maxRowThreadsX = std::max<int64_t>(
          1, getEnvInt64OrDefault(tcCand ? "WELDER_MM_SM_TC_MAX_ROW_THREADS_X"
                                         : "WELDER_MM_SM_MAX_ROW_THREADS_X",
                                  tcCand ? 32 : 32));
      const bool disableTcRowWarp = tcCand &&
                                    (getEnvInt64OrDefault(
                                         "WELDER_MM_SM_TC_DISABLE_ROW_WARP", 0) != 0);
      const bool disableTcRowVectorize =
          tcCand &&
          (getEnvInt64OrDefault("WELDER_MM_SM_TC_DISABLE_ROW_VECTORIZE", 0) !=
           0);
      const bool disableTcRowCombineVectorize =
          tcCand &&
          (getEnvInt64OrDefault(
               "WELDER_MM_SM_TC_DISABLE_ROW_COMBINE_VECTORIZE", 0) != 0);
      const bool disableTcRowInputPromoVectorize =
          tcCand &&
          (getEnvInt64OrDefault(
               "WELDER_MM_SM_TC_DISABLE_ROW_INPUT_PROMO_VECTORIZE", 0) != 0);
      const bool disableTcRowRelaxBarriers =
          tcCand &&
          (getEnvInt64OrDefault("WELDER_MM_SM_TC_DISABLE_ROW_RELAX_BARRIERS", 0) !=
           0);
      const bool disableTcRowSkipCombineBarrier =
          tcCand &&
          (getEnvInt64OrDefault(
               "WELDER_MM_SM_TC_DISABLE_ROW_SKIP_COMBINE_BARRIER", 0) != 0);
      const int64_t minTcRowThreadsX = std::max<int64_t>(
          0, getEnvInt64OrDefault("WELDER_MM_SM_TC_MIN_ROW_THREADS_X", 0));
      if (c.rowReductionVectorWidth > maxRowVecWidth) {
        c.rowReductionVectorWidth = maxRowVecWidth;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (c.rowReductionInputVectorWidth > maxRowInputVecWidth) {
        c.rowReductionInputVectorWidth = maxRowInputVecWidth;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (c.rowReductionThreadsX > maxRowThreadsX) {
        c.rowReductionThreadsX = maxRowThreadsX;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (tcCand && minTcRowThreadsX > 0 && c.rowReductionThreadsX > 0 &&
          c.rowReductionThreadsX < minTcRowThreadsX) {
        c.rowReductionThreadsX = minTcRowThreadsX;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowWarp && c.enableRowReductionWarp) {
        c.enableRowReductionWarp = false;
        if (c.rowReductionThreadsX <= 0)
          c.rowReductionThreadsX = maxRowThreadsX;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowVectorize && c.enableRowReductionVectorize) {
        c.enableRowReductionVectorize = false;
        c.rowReductionVectorWidth = 0;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowCombineVectorize && c.enableRowReductionCombineVectorize) {
        c.enableRowReductionCombineVectorize = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowInputPromoVectorize &&
          c.enableRowReductionInputPromotionVectorize) {
        c.enableRowReductionInputPromotionVectorize = false;
        c.rowReductionInputVectorWidth = 0;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowRelaxBarriers && c.enableRowReductionRelaxBarriers) {
        c.enableRowReductionRelaxBarriers = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }
      if (disableTcRowSkipCombineBarrier &&
          c.enableRowReductionSkipCombineBarrier) {
        c.enableRowReductionSkipCombineBarrier = false;
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 23;
      }

      if (!c.enableRowReductionInputPromotion)
        c.enableRowReductionInputPromotionVectorize = false;
      if (!c.enableRowReductionInputPromotionVectorize)
        c.rowReductionInputVectorWidth = 0;
      if (!c.enableRowReductionVectorize)
        c.rowReductionVectorWidth = 0;

      // Matmul->Softmax 的 async-copy 稳定性保护（SIMT 路径）：
      // 部分较大 threadTileN 变体虽可编译，但在该融合链上可能触发
      // cp.async 运行时地址未对齐错误。
      if (!tcCand && c.enableAsyncCopy) {
        const int64_t maxSimtAsyncThreadTileN = std::max<int64_t>(
            1, getEnvInt64OrDefault("WELDER_MM_SM_ASYNC_MAX_THREAD_TILE_N", 2));
        if (c.threadTileN > maxSimtAsyncThreadTileN) {
          c.enableAsyncCopy = false;
          c.enableSoftwarePipelining = false;
          c.pipelineSetAsyncWaitGroups = false;
          c.pipelineDepth = 2;
          c.workgroupMultiBufferDepth = 1;
          if (c.feasibilityCode == 0)
            c.feasibilityCode = 25;
        }
      }

      // Matmul->Softmax 的 async-copy 策略保护（TensorCore 路径）：
      // 仅当每个 tile 的 MMA-K 工作量足以摊销 async/pipeline 开销时，
      // 才保留 cp.async/pipeline；否则强制回退到稳定的非 async 路径。
      if (tcCand && c.enableAsyncCopy) {
        const bool allowTcAsyncWithoutPipe =
            getEnvInt64OrDefault("WELDER_MM_SM_TC_ALLOW_ASYNC_NO_PIPE", 0) != 0;
        const int64_t minTcPipeMmaIters = std::max<int64_t>(
            1, getEnvInt64OrDefault("WELDER_MM_SM_TC_PIPE_MIN_MMA_ITERS", 1));
        const bool allowTcPipeDepthDowngradeOnSmallK =
            getEnvInt64OrDefault("WELDER_MM_SM_TC_PIPE_DOWNGRADE_DEPTH_ON_SMALL_K",
                                 1) != 0;
        const int64_t mmaK =
            c.mmaK > 0 ? c.mmaK : (c.enableTensorCoreF16 ? int64_t(16)
                                                          : int64_t(4));
        const int64_t mmaIters =
            (mmaK > 0) ? std::max<int64_t>(0, c.tileK / mmaK) : int64_t(0);
        bool disableTcAsync = false;
        if (!c.enableSoftwarePipelining && !allowTcAsyncWithoutPipe)
          disableTcAsync = true;
        if (c.enableSoftwarePipelining) {
          int64_t minItersByDepth = std::max<int64_t>(1, c.pipelineDepth - 1);
          int64_t requiredMmaIters =
              std::max<int64_t>(minTcPipeMmaIters, minItersByDepth);
          if (mmaIters < requiredMmaIters && allowTcPipeDepthDowngradeOnSmallK &&
              mmaIters >= minTcPipeMmaIters) {
            const int64_t downgradedDepth = std::max<int64_t>(2, mmaIters + 1);
            if (downgradedDepth < c.pipelineDepth)
              c.pipelineDepth = downgradedDepth;
            if (c.workgroupMultiBufferDepth < c.pipelineDepth)
              c.workgroupMultiBufferDepth = c.pipelineDepth;
            minItersByDepth = std::max<int64_t>(1, c.pipelineDepth - 1);
            requiredMmaIters =
                std::max<int64_t>(minTcPipeMmaIters, minItersByDepth);
          }
          if (mmaIters < requiredMmaIters)
            disableTcAsync = true;
        }
        if (disableTcAsync) {
          c.enableAsyncCopy = false;
          c.asyncBypassL1 = true;
          c.enableSoftwarePipelining = false;
          c.pipelineSetAsyncWaitGroups = false;
          c.pipelineDepth = 2;
          c.workgroupMultiBufferDepth = 1;
          if (c.feasibilityCode == 0)
            c.feasibilityCode = 25;
        }
      }

      // Matmul->Softmax 的 pipeline 策略保护：
      // K 很小或 multi-buffer 不足时启用 pipelining 往往会明显退化
      //（尤其当行归约阶段与主计算同 kernel 融合时）。
      // 在性能测量前先归一化这些组合，让搜索空间聚焦于可落地候选。
      if (c.enableSoftwarePipelining) {
        const int64_t defaultTcPipeMinTileK =
            mmSmRowChain ? int64_t(16) : int64_t(16);
        const int64_t minPipeTileK = std::max<int64_t>(
            1, getEnvInt64OrDefault(
                   tcCand ? "WELDER_MM_SM_TC_PIPE_MIN_TILE_K"
                          : "WELDER_MM_SM_PIPE_MIN_TILE_K",
                   tcCand ? defaultTcPipeMinTileK : int64_t(32)));
        const int64_t minPipeMb = std::max<int64_t>(
            1, getEnvInt64OrDefault(
                   tcCand ? "WELDER_MM_SM_TC_PIPE_MIN_MULTIBUFFER"
                          : "WELDER_MM_SM_PIPE_MIN_MULTIBUFFER",
                   2));
        const bool riskyWarpPipeCombo =
            (!tcCand && c.enableRowReductionWarp &&
             c.rowReductionThreadsX >=
                 std::max<int64_t>(1, getEnvInt64OrDefault(
                                          "WELDER_MM_SM_PIPE_WARP_THREADS_X", 32)));
        const int64_t minSimtPipeThreadsX =
            std::max<int64_t>(1, getEnvInt64OrDefault(
                                     "WELDER_MM_SM_PIPE_MIN_THREADS_X", 32));
        const bool riskySimtPipeThreads =
            (!tcCand && c.rowReductionThreadsX > 0 &&
             c.rowReductionThreadsX < minSimtPipeThreadsX);
        if (c.tileK < minPipeTileK || c.workgroupMultiBufferDepth < minPipeMb ||
            riskyWarpPipeCombo || riskySimtPipeThreads) {
          c.enableSoftwarePipelining = false;
          c.pipelineSetAsyncWaitGroups = false;
          c.pipelineDepth = 2;
          c.workgroupMultiBufferDepth = 1;
          if (c.feasibilityCode == 0)
            c.feasibilityCode = 24;
        }
      }
    }
  };
  auto refreshMatmulSgemmModelIfNeeded = [&](Candidate &c) -> bool {
    // 仅适用于旧的 matmul 专用模型（smemBytes>0）。
    if (c.smemBytes <= 0)
      return true;
    if (c.tileM <= 0 || c.tileN <= 0 || c.tileK <= 0)
      return false;

    int64_t pad = std::max<int64_t>(0, c.workgroupPadLastDim);
    int64_t aBytes = c.tileM * (c.tileK + pad) * opts.arch.elementBytes; // A: [M,K]
    int64_t bBytes = c.tileK * (c.tileN + pad) * opts.arch.elementBytes; // B: [K,N]
    int64_t smem = aBytes + bBytes;
    if (smem <= 0)
      return false;
    int64_t depth = std::max<int64_t>(1, c.workgroupMultiBufferDepth);
    if (depth > 1) {
      if (smem > (std::numeric_limits<int64_t>::max() / depth))
        return false;
      smem *= depth;
    }
    c.smemBytes = smem;
    if (c.smemBytes > opts.arch.smemBytes)
      return false;

    // 更新占用率/waves 与 score，确保关闭性能测量时
    // solver 排名逻辑仍保持一致。
    int64_t blocksPerSM = std::max<int64_t>(1, opts.arch.smemBytes / c.smemBytes);
    blocksPerSM = std::min<int64_t>(blocksPerSM, opts.arch.maxBlocksPerSM);
    c.blocksPerSM = blocksPerSM;
    int64_t concurrentBlocks =
        std::max<int64_t>(1, blocksPerSM * opts.arch.numSM);
    c.numWave = ceilDiv(std::max<int64_t>(1, c.blocksTotal), concurrentBlocks);
    c.score = c.traffic.totalBytes() * static_cast<double>(c.numWave);
    return true;
  };

  if (!opts.codegenSearch.enable) {
    Candidate c = base;
    applyFixedCodegenKnobsFromProfile(c, opts);
    if (c.enableTensorCoreF16 || c.enableTensorCoreTf32) {
      applyTensorCoreFeasibilityOrDowngrade(c);
    }
    normalizeCodegenKnobsOrAnnotate(c);
    // 旋钮归一化仍可能改变 TensorCore 相关 launch/layout 属性
    //（如行链 padding 策略）。这里再次执行 TC 可行性检查，
    // 让性能测量阶段的大部分合法性问题在编译前就被处理。
    if (c.enableTensorCoreF16 || c.enableTensorCoreTf32) {
      applyTensorCoreFeasibilityOrDowngrade(c);
    }
    if (!refreshMatmulSgemmModelIfNeeded(c))
      return out;
    out.push_back(std::move(c));
    return out;
  }

  const SolveOptions::CodegenSearchSpace &s = opts.codegenSearch;

  auto pads = s.workgroupPadLastDim;
  if (pads.empty())
    pads.push_back(0);
  auto padMatmulOnlys = s.workgroupPadLastDimMatmulOnly;
  if (padMatmulOnlys.empty())
    padMatmulOnlys.push_back(false);
  auto mbDepths = s.workgroupMultiBufferDepth;
  if (mbDepths.empty())
    mbDepths.push_back(1);
  auto swizzles = s.workgroupSwizzleXor;
  if (swizzles.empty())
    swizzles.push_back(0);
  auto rastersXor = s.blockRasterizeXor;
  if (rastersXor.empty())
    rastersXor.push_back(0);
  auto rasterModes = s.blockRasterizeMode;
  if (rasterModes.empty())
    rasterModes.push_back(0);
  auto rasterPanels = s.blockRasterizePanelWidth;
  // 始终包含 0，确保“光栅化关闭”（mode=0,panel=0）始终在搜索空间中，
  // 即便默认 panel 列表里全是 >0 的值。
  if (rasterPanels.empty())
    rasterPanels.push_back(0);
  else if (!llvm::is_contained(rasterPanels, 0))
    rasterPanels.push_back(0);
  llvm::sort(rasterPanels);
  rasterPanels.erase(std::unique(rasterPanels.begin(), rasterPanels.end()),
                     rasterPanels.end());
  auto swaps = s.swapBlockDims;
  if (swaps.empty())
    swaps.push_back(false);
  auto pipeDepths = s.pipelineDepth;
  if (pipeDepths.empty())
    pipeDepths.push_back(2);
  auto waitGroups = s.pipelineSetAsyncWaitGroups;
  if (waitGroups.empty())
    waitGroups.push_back(false);

  auto rrReuseFusions = s.enableRowReductionChainReuseFusion;
  if (rrReuseFusions.empty())
    rrReuseFusions.push_back(false);
  auto rrInputPromotions = s.enableRowReductionInputPromotion;
  if (rrInputPromotions.empty())
    rrInputPromotions.push_back(false);
  auto rrInputPromotionVectorize = s.enableRowReductionInputPromotionVectorize;
  if (rrInputPromotionVectorize.empty())
    rrInputPromotionVectorize.push_back(false);
  auto rrWarp = s.enableRowReductionWarp;
  if (rrWarp.empty())
    rrWarp.push_back(false);
  auto rrVectorize = s.enableRowReductionVectorize;
  if (rrVectorize.empty())
    rrVectorize.push_back(false);
  auto rrVecWidths = s.rowReductionVectorWidth;
  if (rrVecWidths.empty())
    rrVecWidths.push_back(0);
  auto rrThreadsX = s.rowReductionThreadsX;
  if (rrThreadsX.empty())
    rrThreadsX.push_back(0);
  auto rrRelax = s.enableRowReductionRelaxBarriers;
  if (rrRelax.empty())
    rrRelax.push_back(false);
  auto rrSkipCombine = s.enableRowReductionSkipCombineBarrier;
  if (rrSkipCombine.empty())
    rrSkipCombine.push_back(false);
  auto rrInputVecWidths = s.rowReductionInputVectorWidth;
  if (rrInputVecWidths.empty())
    rrInputVecWidths.push_back(0);
  auto rrCombineVec = s.enableRowReductionCombineVectorize;
  if (rrCombineVec.empty())
    rrCombineVec.push_back(false);
  auto mmSoftmaxReuseFusions = s.enableMatmulSoftmaxSharedReuseFusion;
  if (mmSoftmaxReuseFusions.empty())
    mmSoftmaxReuseFusions.push_back(false);

  auto emit = [&](Candidate c) {
    ++prefilterStats.generated;
    if (strictLegalityPrefilter) {
      if (c.pipelineSetAsyncWaitGroups &&
          !(c.enableAsyncCopy && c.enableSoftwarePipelining)) {
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 30;
        ++prefilterStats.droppedIllegalWaitGroup;
        return;
      }
      if (c.enableSoftwarePipelining &&
          (!c.enableAsyncCopy || c.pipelineDepth < 2 ||
           c.workgroupMultiBufferDepth < 2 ||
           c.workgroupMultiBufferDepth < c.pipelineDepth)) {
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 31;
        ++prefilterStats.droppedIllegalPipeline;
        return;
      }
      if (c.workgroupSwizzleXor != 0) {
        auto isPow2 = [](int64_t v) -> bool {
          return v > 0 && (v & (v - 1)) == 0;
        };
        auto canSwizzleLastDimStrict = [&](int64_t lastDim) -> bool {
          if (c.workgroupSwizzleXor <= 1)
            return false;
          if (lastDim <= 1)
            return false;
          if (!isPow2(c.workgroupSwizzleXor))
            return false;
          if (!isPow2(lastDim))
            return false;
          return c.workgroupSwizzleXor <= lastDim;
        };
        bool okA = canSwizzleLastDimStrict(c.tileK);
        bool okB = canSwizzleLastDimStrict(c.tileN);
        if (!(okA || okB)) {
          if (c.feasibilityCode == 0)
            c.feasibilityCode = 32;
          ++prefilterStats.droppedIllegalSwizzle;
          return;
        }
      }
      if (c.blockRasterizeXor != 0 &&
          (c.blockRasterizeMode != 0 || c.blockRasterizePanelWidth != 0)) {
        if (c.feasibilityCode == 0)
          c.feasibilityCode = 33;
        ++prefilterStats.droppedIllegalRasterConflict;
        return;
      }
      if (c.blockRasterizeMode != 0 || c.blockRasterizePanelWidth != 0) {
        bool ok = (c.blockRasterizeMode == 1 || c.blockRasterizeMode == 2) &&
                  (c.blockRasterizePanelWidth > 0 &&
                   c.blockRasterizePanelWidth <= 16);
        if (!ok) {
          if (c.feasibilityCode == 0)
            c.feasibilityCode = 34;
          ++prefilterStats.droppedIllegalRasterParam;
          return;
        }
      }
    }
    c.workgroupMultiBufferDepth = std::max<int64_t>(1, c.workgroupMultiBufferDepth);
    c.pipelineDepth = std::max<int64_t>(2, c.pipelineDepth);
    const bool tcRequestedBySearch = c.enableTensorCoreF16 || c.enableTensorCoreTf32;
    auto isTensorCoreDowngradeCode = [](int64_t code) {
      switch (code) {
      case 1:  // TensorCore MMA alignment violation.
      case 2:  // TensorCore layout/stride-map feasibility violation.
      case 3:  // TensorCore block-warp mapping overflow.
      case 27: // TensorCore requested on non-matmul subgraph.
      case 28: // TensorCore f16 tileK exceeds guarded max for mm_sm chain.
        return true;
      default:
        return false;
      }
    };
    if (c.enableTensorCoreF16 || c.enableTensorCoreTf32) {
      // 论文/Welder 对齐（TCPolicy stride_map）：
      // TensorCore 调度通常会在 matmul operand 的 shared 布局上使用 offset=8 padding。
      if (c.workgroupPadLastDim == 0)
        c.workgroupPadLastDim = 8;
      c.workgroupPadLastDimMatmulOnly = true;
      if (c.enableTensorCoreF16 && !c.useCutlassMma) {
        // 参考启发式：cutlass warp-mma 主要用于较大 K。
        // 这里用已选 shared 层 K tile 做近似判定。
        if (c.tileK >= 32)
          c.useCutlassMma = true;
      }
      chooseMmaShapeForCandidate(c);
      // 若无法选出合法 MMA 形状，则该变体关闭 TensorCore。
      if (c.mmaM <= 0 || c.mmaN <= 0 || c.mmaK <= 0) {
        c.enableTensorCoreF16 = false;
        c.enableTensorCoreTf32 = false;
        c.useCutlassMma = false;
      } else {
        applyTensorCoreFeasibilityOrDowngrade(c);
      }
    }
    const bool searchIncludesNonTensorCore =
        llvm::is_contained(s.enableTensorCoreF16, false) ||
        llvm::is_contained(s.enableTensorCoreTf32, false);
    const bool dropTcDowngraded = getEnvInt64OrDefault(
                                      "WELDER_PREFILTER_DROP_TC_DOWNGRADED", 1) !=
                                  0;
    const bool dropTcDowngradedRequireNonTcSpace =
        getEnvInt64OrDefault(
            "WELDER_PREFILTER_DROP_TC_DOWNGRADED_REQUIRE_NON_TC_SPACE", 1) != 0;
    if (dropTcDowngraded && tcRequestedBySearch &&
        !(c.enableTensorCoreF16 || c.enableTensorCoreTf32) &&
        isTensorCoreDowngradeCode(c.feasibilityCode) &&
        (!dropTcDowngradedRequireNonTcSpace || searchIncludesNonTensorCore)) {
      ++prefilterStats.droppedTcDowngraded;
      return;
    }
    normalizeCodegenKnobsOrAnnotate(c);
    // 旋钮归一化可能影响 TensorCore 合法性假设，
    // 因此这里再次检查，提前过滤性能测量前的非法 TC 变体。
    if (c.enableTensorCoreF16 || c.enableTensorCoreTf32) {
      applyTensorCoreFeasibilityOrDowngrade(c);
    }
    if (dropTcDowngraded && tcRequestedBySearch &&
        !(c.enableTensorCoreF16 || c.enableTensorCoreTf32) &&
        isTensorCoreDowngradeCode(c.feasibilityCode) &&
        (!dropTcDowngradedRequireNonTcSpace || searchIncludesNonTensorCore)) {
      ++prefilterStats.droppedTcDowngraded;
      return;
    }
    if (!refreshMatmulSgemmModelIfNeeded(c)) {
      ++prefilterStats.droppedRefreshFail;
      return;
    }
    const std::string dedupKey = buildVariantDedupKey(c);
    if (!dedupKey.empty() && !emittedVariantKeys.insert(dedupKey).second) {
      ++prefilterStats.droppedDuplicate;
      return;
    }
    ++prefilterStats.kept;
    out.push_back(std::move(c));
  };

  auto emitWithRowReductionKnobs = [&](const Candidate &c0) {
    for (bool mmSmReuse : mmSoftmaxReuseFusions) {
      for (bool reuse : rrReuseFusions) {
        for (bool promo : rrInputPromotions) {
          for (bool vec : rrInputPromotionVectorize) {
            for (bool warp : rrWarp) {
              for (bool rvec : rrVectorize) {
                for (int64_t rvecWidth : rrVecWidths) {
                  for (int64_t rtx : rrThreadsX) {
                    for (bool relax : rrRelax) {
                      for (bool skipCombine : rrSkipCombine) {
                        for (int64_t inVec : rrInputVecWidths) {
                          for (bool combVec : rrCombineVec) {
                Candidate c = c0;
                c.enableMatmulSoftmaxSharedReuseFusion = mmSmReuse;
                c.enableRowReductionChainReuseFusion = reuse;
                c.enableRowReductionInputPromotion = promo;
                // 只有启用 promotion 时，vectorize 才有意义。
                c.enableRowReductionInputPromotionVectorize = promo && vec;
                c.enableRowReductionWarp = warp;
                c.enableRowReductionVectorize = rvec;
                c.rowReductionVectorWidth = rvec ? rvecWidth : 0;
                c.rowReductionThreadsX = rtx;
                c.enableRowReductionRelaxBarriers = relax;
                c.enableRowReductionSkipCombineBarrier = skipCombine;
                c.rowReductionInputVectorWidth = promo ? inVec : 0;
                c.enableRowReductionCombineVectorize = combVec;
                emit(std::move(c));
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  };
  const int64_t tcTileKTf32 = [&]() -> int64_t {
    int64_t tk = std::max<int64_t>(
        4, getEnvInt64OrDefault("WELDER_TC_TF32_TILE_K", 4));
    tk = (tk / 4) * 4;
    return std::max<int64_t>(4, tk);
  }();
  const llvm::SmallVector<int64_t, 4> tcTileKF16Candidates = [&]() {
    llvm::SmallVector<int64_t, 4> ks;
    auto normalizeTileK = [](int64_t tk) -> int64_t {
      tk = std::max<int64_t>(16, tk);
      tk = (tk / 16) * 16;
      return std::max<int64_t>(16, tk);
    };
    auto addTileK = [&](int64_t tk) {
      ks.push_back(normalizeTileK(tk));
    };
    const bool preferTileK32ForMmSm =
        matmulSoftmaxLikeSubgraph &&
        (getEnvInt64OrDefault("WELDER_MM_SM_TC_F16_DEFAULT_TILE_K32", 0) != 0);
    const int64_t defaultTileK = preferTileK32ForMmSm ? 32 : 16;
    const char *tileKCsvEnv = std::getenv("WELDER_TC_F16_TILE_K_CSV");
    if (tileKCsvEnv && tileKCsvEnv[0] != '\0') {
      llvm::StringRef csv(tileKCsvEnv);
      llvm::SmallVector<llvm::StringRef, 8> parts;
      csv.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (llvm::StringRef part : parts) {
        auto trimmed = part.trim();
        if (trimmed.empty())
          continue;
        int64_t v = 0;
        if (trimmed.getAsInteger(10, v))
          continue;
        addTileK(v);
      }
    } else {
      addTileK(getEnvInt64OrDefault("WELDER_TC_F16_TILE_K", defaultTileK));
    }
    if (matmulSoftmaxLikeSubgraph &&
        getEnvInt64OrDefault("WELDER_MM_SM_TC_F16_INCLUDE_TILE_K16", 1) != 0)
      addTileK(16);
    if (preferTileK32ForMmSm &&
        getEnvInt64OrDefault("WELDER_MM_SM_TC_F16_INCLUDE_TILE_K32", 0) != 0)
      addTileK(32);
    if (matmulSoftmaxLikeSubgraph &&
        getEnvInt64OrDefault("WELDER_MM_SM_TC_F16_INCLUDE_FALLBACK_TILE_K16", 0) !=
            0)
      addTileK(16);
    llvm::sort(ks);
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    if (ks.empty())
      ks.push_back(defaultTileK);
    return ks;
  }();
  auto buildTensorCoreTilePairs =
      [&](int64_t mmaM, int64_t mmaN)
      -> std::vector<std::pair<int64_t, int64_t>> {
    std::vector<std::pair<int64_t, int64_t>> pairs;
    if (mmaM <= 0 || mmaN <= 0)
      return pairs;
    auto alignUp = [](int64_t v, int64_t a) -> int64_t {
      if (a <= 0)
        return v;
      if (v <= 0)
        return a;
      return ((v + a - 1) / a) * a;
    };
    auto addAxisValue = [&](std::vector<int64_t> &axis, int64_t raw,
                            int64_t align) {
      if (raw <= 0 || align <= 0)
        return;
      int64_t up = alignUp(raw, align);
      axis.push_back(std::max<int64_t>(align, up));
      if ((raw % align) != 0) {
        int64_t down = (raw / align) * align;
        if (down >= align)
          axis.push_back(down);
      }
    };

    std::vector<int64_t> axisM;
    std::vector<int64_t> axisN;
    addAxisValue(axisM, base.tileM, mmaM);
    addAxisValue(axisN, base.tileN, mmaN);
    addAxisValue(axisM, mmaM, mmaM);
    addAxisValue(axisM, mmaM * 2, mmaM);
    addAxisValue(axisN, mmaN, mmaN);
    addAxisValue(axisN, mmaN * 8, mmaN);
    addAxisValue(axisN, std::max<int64_t>(mmaN, maxRowReductionExtentForTc),
                 mmaN);
    for (int64_t v : opts.candidatesMN) {
      addAxisValue(axisM, v, mmaM);
      addAxisValue(axisN, v, mmaN);
    }

    llvm::sort(axisM);
    axisM.erase(std::unique(axisM.begin(), axisM.end()), axisM.end());
    llvm::sort(axisN);
    axisN.erase(std::unique(axisN.begin(), axisN.end()), axisN.end());

    for (int64_t tm : axisM) {
      if (tm < mmaM || (tm % mmaM) != 0)
        continue;
      for (int64_t tn0 : axisN) {
        int64_t tn = std::max<int64_t>(tn0, maxRowReductionExtentForTc);
        tn = alignUp(tn, mmaN);
        if (tn < mmaN || (tn % mmaN) != 0)
          continue;
        int64_t warpsM = tm / mmaM;
        int64_t warpsN = tn / mmaN;
        if (warpsM <= 0 || warpsN <= 0)
          continue;
        if (warpsM > (std::numeric_limits<int64_t>::max() / warpsN))
          continue;
        int64_t warps = warpsM * warpsN;
        if (warps <= 0 || warps * 32 > 1024)
          continue;
        pairs.emplace_back(tm, tn);
      }
    }

    llvm::sort(pairs, [&](const auto &a, const auto &b) {
      int64_t da = std::llabs(a.first - base.tileM) +
                   std::llabs(a.second - base.tileN);
      int64_t db = std::llabs(b.first - base.tileM) +
                   std::llabs(b.second - base.tileN);
      if (da != db)
        return da < db;
      if (a.first != b.first)
        return a.first < b.first;
      return a.second < b.second;
    });
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    const bool expandPairs =
        getEnvInt64OrDefault("WELDER_TC_TILE_PAIR_EXPAND", 1) != 0;
    const int64_t pairLimit = std::max<int64_t>(
        1, getEnvInt64OrDefault("WELDER_TC_TILE_PAIR_LIMIT",
                                expandPairs ? 8 : 1));
    if (static_cast<int64_t>(pairs.size()) > pairLimit)
      pairs.resize(static_cast<size_t>(pairLimit));
    return pairs;
  };
  const std::vector<std::pair<int64_t, int64_t>> tcTf32TilePairs =
      buildTensorCoreTilePairs(/*mmaM=*/16, /*mmaN=*/8);
  const std::vector<std::pair<int64_t, int64_t>> tcF16TilePairs =
      buildTensorCoreTilePairs(/*mmaM=*/16, /*mmaN=*/8);

  for (int64_t pad : pads) {
    for (bool padMatmulOnly : padMatmulOnlys) {
      for (int64_t swz : swizzles) {
        for (int rm : rasterModes) {
          for (int rp : rasterPanels) {
            if (rm == 0 && rp != 0)
              continue;
            if (rm != 0 && rp <= 0)
              continue;
            for (int64_t rastXor : rastersXor) {
              if (rm != 0 && rastXor != 0)
                continue; // keep XOR and 2D 光栅化 mutually exclusive
              for (bool swap : swaps) {
                for (bool enableTCtf32 : s.enableTensorCoreTf32) {
                for (bool enableTCf16 : s.enableTensorCoreF16) {
                  if (enableTCtf32 && enableTCf16)
                    continue;
                  if ((enableTCtf32 || enableTCf16) &&
                      !tensorCoreCapableSubgraph)
                    continue;

                    if (enableTCtf32) {
                      // 避免在 padding/重排搜索循环中重复生成 TensorCore 配置
                      //（TensorCore 在下方使用固定旋钮）。
                      if (pad != 0 || padMatmulOnly || swz != 0)
                        continue;
                      if (tcTf32TilePairs.empty())
                        continue;

                      for (const auto &tcTile : tcTf32TilePairs) {
                        for (bool enableAsync : s.enableAsyncCopy) {
                          if (!enableAsync) {
                            // 无 async-copy 的基线配置；该情况下 bypass/pipelining 无效。
                            Candidate c = base;
                            c.tileM = tcTile.first;
                            c.tileN = tcTile.second;
                            c.tileK = tcTileKTf32;
                            c.enableTensorCoreTf32 = true;
                            c.enableTensorCoreF16 = false;
                            c.enableAsyncCopy = false;
                            c.asyncBypassL1 = true;
                            c.enableSoftwarePipelining = false;
                            c.pipelineDepth = 2;
                            c.pipelinePeelEpilogue = true;
                            c.pipelineSetAsyncWaitGroups = false;
                            c.workgroupMultiBufferDepth = 1;
                            c.workgroupPadLastDim = 8;
                            c.workgroupPadLastDimMatmulOnly = true;
                            c.workgroupSwizzleXor = 0;
                            c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                            c.blockRasterizeMode = std::max(0, rm);
                            c.blockRasterizePanelWidth = std::max(0, rp);
                            c.swapBlockDims = swap;
                            emitWithRowReductionKnobs(c);
                            continue;
                          }
                          for (bool bypass : s.asyncBypassL1) {
                            for (bool enablePipe : s.enableSoftwarePipelining) {
                              if (!enablePipe) {
                                Candidate c = base;
                                c.tileM = tcTile.first;
                                c.tileN = tcTile.second;
                                c.tileK = tcTileKTf32;
                                c.enableTensorCoreTf32 = true;
                                c.enableTensorCoreF16 = false;
                                c.enableAsyncCopy = true;
                                c.asyncBypassL1 = bypass;
                                c.enableSoftwarePipelining = false;
                                c.pipelineDepth = 2;
                                c.pipelinePeelEpilogue = true;
                                c.pipelineSetAsyncWaitGroups = false;
                                c.workgroupMultiBufferDepth = 1;
                                c.workgroupPadLastDim = 8;
                                c.workgroupPadLastDimMatmulOnly = true;
                                c.workgroupSwizzleXor = 0;
                                c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                                c.blockRasterizeMode = std::max(0, rm);
                                c.blockRasterizePanelWidth = std::max(0, rp);
                                c.swapBlockDims = swap;
                                emitWithRowReductionKnobs(c);
                                continue;
                              }

                              for (int64_t depth : pipeDepths) {
                                for (bool peel : s.pipelinePeelEpilogue) {
                                  for (bool setWait : waitGroups) {
                                    for (int64_t mb : mbDepths) {
                                      if (mb < 2)
                                        continue;
                                      if (mb < depth)
                                        continue;
                                      Candidate c = base;
                                      c.tileM = tcTile.first;
                                      c.tileN = tcTile.second;
                                      c.tileK = tcTileKTf32;
                                      c.enableTensorCoreTf32 = true;
                                      c.enableTensorCoreF16 = false;
                                      c.enableAsyncCopy = true;
                                      c.asyncBypassL1 = bypass;
                                      c.enableSoftwarePipelining = true;
                                      c.pipelineDepth = depth;
                                      c.pipelinePeelEpilogue = peel;
                                      c.pipelineSetAsyncWaitGroups = setWait;
                                      c.workgroupMultiBufferDepth = mb;
                                      c.workgroupPadLastDim = 8;
                                      c.workgroupPadLastDimMatmulOnly = true;
                                      c.workgroupSwizzleXor = 0;
                                      c.blockRasterizeXor =
                                          std::max<int64_t>(0, rastXor);
                                      c.blockRasterizeMode = std::max(0, rm);
                                      c.blockRasterizePanelWidth =
                                          std::max(0, rp);
                                      c.swapBlockDims = swap;
                                      emitWithRowReductionKnobs(c);
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      continue;
                    }

                    if (enableTCf16) {
                      if (pad != 0 || padMatmulOnly || swz != 0)
                        continue;
                      if (tcF16TilePairs.empty())
                        continue;

                      for (int64_t tcTileK : tcTileKF16Candidates) {
                        for (const auto &tcTile : tcF16TilePairs) {
                          for (bool enableAsync : s.enableAsyncCopy) {
                            if (!enableAsync) {
                              Candidate c = base;
                              c.tileM = tcTile.first;
                              c.tileN = tcTile.second;
                              c.tileK = tcTileK;
                              c.enableTensorCoreTf32 = false;
                              c.enableTensorCoreF16 = true;
                              c.enableAsyncCopy = false;
                              c.asyncBypassL1 = true;
                              c.enableSoftwarePipelining = false;
                              c.pipelineDepth = 2;
                              c.pipelinePeelEpilogue = true;
                              c.pipelineSetAsyncWaitGroups = false;
                              c.workgroupMultiBufferDepth = 1;
                              c.workgroupPadLastDim = 8;
                              c.workgroupPadLastDimMatmulOnly = true;
                              c.workgroupSwizzleXor = 0;
                              c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                              c.blockRasterizeMode = std::max(0, rm);
                              c.blockRasterizePanelWidth = std::max(0, rp);
                              c.swapBlockDims = swap;
                              emitWithRowReductionKnobs(c);
                              continue;
                            }
                            for (bool bypass : s.asyncBypassL1) {
                              for (bool enablePipe : s.enableSoftwarePipelining) {
                                if (!enablePipe) {
                                  Candidate c = base;
                                  c.tileM = tcTile.first;
                                  c.tileN = tcTile.second;
                                  c.tileK = tcTileK;
                                  c.enableTensorCoreTf32 = false;
                                  c.enableTensorCoreF16 = true;
                                  c.enableAsyncCopy = true;
                                  c.asyncBypassL1 = bypass;
                                  c.enableSoftwarePipelining = false;
                                  c.pipelineDepth = 2;
                                  c.pipelinePeelEpilogue = true;
                                  c.pipelineSetAsyncWaitGroups = false;
                                  c.workgroupMultiBufferDepth = 1;
                                  c.workgroupPadLastDim = 8;
                                  c.workgroupPadLastDimMatmulOnly = true;
                                  c.workgroupSwizzleXor = 0;
                                  c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                                  c.blockRasterizeMode = std::max(0, rm);
                                  c.blockRasterizePanelWidth = std::max(0, rp);
                                  c.swapBlockDims = swap;
                                  emitWithRowReductionKnobs(c);
                                  continue;
                                }

                                for (int64_t depth : pipeDepths) {
                                  for (bool peel : s.pipelinePeelEpilogue) {
                                    for (bool setWait : waitGroups) {
                                      for (int64_t mb : mbDepths) {
                                        if (mb < 2)
                                          continue;
                                        if (mb < depth)
                                          continue;
                                        Candidate c = base;
                                        c.tileM = tcTile.first;
                                        c.tileN = tcTile.second;
                                        c.tileK = tcTileK;
                                        c.enableTensorCoreTf32 = false;
                                        c.enableTensorCoreF16 = true;
                                        c.enableAsyncCopy = true;
                                        c.asyncBypassL1 = bypass;
                                        c.enableSoftwarePipelining = true;
                                        c.pipelineDepth = depth;
                                        c.pipelinePeelEpilogue = peel;
                                        c.pipelineSetAsyncWaitGroups = setWait;
                                        c.workgroupMultiBufferDepth = mb;
                                        c.workgroupPadLastDim = 8;
                                        c.workgroupPadLastDimMatmulOnly = true;
                                        c.workgroupSwizzleXor = 0;
                                        c.blockRasterizeXor =
                                            std::max<int64_t>(0, rastXor);
                                        c.blockRasterizeMode = std::max(0, rm);
                                        c.blockRasterizePanelWidth =
                                            std::max(0, rp);
                                        c.swapBlockDims = swap;
                                        emitWithRowReductionKnobs(c);
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      continue;
                    }

                    for (bool enableAsync : s.enableAsyncCopy) {
                      if (!enableAsync) {
                        Candidate c = base;
                        c.enableTensorCoreTf32 = false;
                        c.enableTensorCoreF16 = false;
                        c.workgroupPadLastDim = std::max<int64_t>(0, pad);
                        c.workgroupPadLastDimMatmulOnly = padMatmulOnly;
                        c.workgroupSwizzleXor = std::max<int64_t>(0, swz);
                        c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                        c.blockRasterizeMode = std::max(0, rm);
                        c.blockRasterizePanelWidth = std::max(0, rp);
                        c.swapBlockDims = swap;
                        c.enableAsyncCopy = false;
                        c.asyncBypassL1 = true;
                        c.enableSoftwarePipelining = false;
                        c.pipelineDepth = 2;
                        c.pipelinePeelEpilogue = true;
                        c.pipelineSetAsyncWaitGroups = false;
                        c.workgroupMultiBufferDepth = 1;
                        emitWithRowReductionKnobs(c);
                        continue;
                      }
                      for (bool bypass : s.asyncBypassL1) {
                        for (bool enablePipe : s.enableSoftwarePipelining) {
	                          if (!enablePipe) {
	                            Candidate c = base;
	                            c.enableTensorCoreTf32 = false;
	                            c.enableTensorCoreF16 = false;
                            c.workgroupPadLastDim = std::max<int64_t>(0, pad);
                            c.workgroupPadLastDimMatmulOnly = padMatmulOnly;
                            c.workgroupSwizzleXor = std::max<int64_t>(0, swz);
                            c.blockRasterizeXor = std::max<int64_t>(0, rastXor);
                            c.blockRasterizeMode = std::max(0, rm);
                            c.blockRasterizePanelWidth = std::max(0, rp);
                            c.swapBlockDims = swap;
                            c.enableAsyncCopy = true;
                            c.asyncBypassL1 = bypass;
	                            c.enableSoftwarePipelining = false;
	                            c.pipelineDepth = 2;
		                            c.pipelinePeelEpilogue = true;
		                            c.pipelineSetAsyncWaitGroups = false;
		                            c.workgroupMultiBufferDepth = 1;
		                            emitWithRowReductionKnobs(c);
		                            continue;
		                          }

	                          for (int64_t depth : pipeDepths) {
	                            for (bool peel : s.pipelinePeelEpilogue) {
	                              for (bool setWait : waitGroups) {
	                                for (int64_t mb : mbDepths) {
	                                if (mb < 2)
	                                  continue;
	                                if (mb < depth)
	                                  continue;
	                                Candidate c = base;
                                c.enableTensorCoreTf32 = false;
                                c.enableTensorCoreF16 = false;
                                c.workgroupPadLastDim = std::max<int64_t>(0, pad);
                                c.workgroupPadLastDimMatmulOnly = padMatmulOnly;
                                c.workgroupSwizzleXor = std::max<int64_t>(0, swz);
                                c.blockRasterizeXor =
                                    std::max<int64_t>(0, rastXor);
                                c.blockRasterizeMode = std::max(0, rm);
                                c.blockRasterizePanelWidth = std::max(0, rp);
                                c.swapBlockDims = swap;
                                c.enableAsyncCopy = true;
                                c.asyncBypassL1 = bypass;
	                                c.enableSoftwarePipelining = true;
	                                c.pipelineDepth = depth;
	                                c.pipelinePeelEpilogue = peel;
	                                c.pipelineSetAsyncWaitGroups = setWait;
		                                c.workgroupMultiBufferDepth = mb;
		                                emitWithRowReductionKnobs(c);
		                                }
	                              }
	                            }
	                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (out.empty()) {
    Candidate c = base;
    // 尽力回退：若用户提供的旋钮组合把所有变体都剪掉，
    // 则从搜索空间挑一个代表性配置，保证调用方仍能拿到一致候选，
    // 而不是静默回退到“全零旋钮”。
    if (!pads.empty())
      c.workgroupPadLastDim = std::max<int64_t>(0, pads.front());
    if (!padMatmulOnlys.empty())
      c.workgroupPadLastDimMatmulOnly = padMatmulOnlys.front();
    if (!swizzles.empty())
      c.workgroupSwizzleXor = std::max<int64_t>(0, swizzles.front());
    if (!rastersXor.empty())
      c.blockRasterizeXor = std::max<int64_t>(0, rastersXor.front());
    if (!rasterModes.empty())
      c.blockRasterizeMode = std::max(0, rasterModes.front());
    if (c.blockRasterizeMode == 0) {
      c.blockRasterizePanelWidth = 0;
    } else {
      int panel = 1;
      for (int p : rasterPanels) {
        if (p > 0) {
          panel = p;
          break;
        }
      }
      c.blockRasterizePanelWidth = panel;
    }
    if (!swaps.empty())
      c.swapBlockDims = swaps.front();

    if (!s.enableAsyncCopy.empty())
      c.enableAsyncCopy = s.enableAsyncCopy.front();
    if (c.enableAsyncCopy && !s.asyncBypassL1.empty())
      c.asyncBypassL1 = s.asyncBypassL1.front();
    if (!s.enableSoftwarePipelining.empty())
      c.enableSoftwarePipelining = s.enableSoftwarePipelining.front();
    if (!pipeDepths.empty())
      c.pipelineDepth = pipeDepths.front();
    if (!s.pipelinePeelEpilogue.empty())
      c.pipelinePeelEpilogue = s.pipelinePeelEpilogue.front();
    if (!waitGroups.empty())
      c.pipelineSetAsyncWaitGroups = waitGroups.front();
    if (!mbDepths.empty())
      c.workgroupMultiBufferDepth = mbDepths.front();

    // TensorCore 开关（若搜索空间请求）。当 TF32/F16 同时请求时优先 TF32，
    // 因为两者互斥。
    if (!s.enableTensorCoreTf32.empty())
      c.enableTensorCoreTf32 = s.enableTensorCoreTf32.front();
    if (!s.enableTensorCoreF16.empty())
      c.enableTensorCoreF16 = s.enableTensorCoreF16.front();
    if (!tensorCoreCapableSubgraph) {
      c.enableTensorCoreTf32 = false;
      c.enableTensorCoreF16 = false;
    }
    if (c.enableTensorCoreTf32 && c.enableTensorCoreF16)
      c.enableTensorCoreF16 = false;
    if (c.enableTensorCoreTf32)
      c.tileK = tcTileKTf32;
    if (c.enableTensorCoreF16)
      c.tileK = tcTileKF16Candidates.front();

    // 论文/Welder 对齐：TCPolicy 在 shared 布局上使用 stride offset=8。
    // 在本 MLIR 实现中对应“最后一维 padding”。
    if ((c.enableTensorCoreF16 || c.enableTensorCoreTf32) &&
        c.workgroupPadLastDim == 0) {
      c.workgroupPadLastDim = 8;
    }
	    emitWithRowReductionKnobs(c);
	  }

  emitPrefilterTrace();
  return out;
}

static void maybeApplyRasterizationTcPolicyPaper(const PaperSubgraph &sg,
                                                 const Traffic &traffic,
                                                 Candidate &cand);
static int64_t estimateRegisterReuseRegsPerThreadForSubgraph(
    const TileGraph &graph, const PaperSubgraph &sg, int minLevelExclusive,
    int64_t blockThreads, const ArchConfig &arch,
    int maxLevelInclusive = -1);
#include "WelderSolverRecursiveStagesCore.h"

#include "WelderSolverRecursiveStageTopKCandidates.h"
#include "WelderSolverRecursiveStageTopKRegisterTiles.h"
#include "WelderSolverRecursiveStageTopKSharedCandidates.h"
#include "WelderSolverSubGraphTilingPaperGlobalShared.h"
#include "WelderSolverGraphConnectingPaperGlobalShared.h"
#include "WelderSolverPhaseAGridReuseHelpers.h"

#include "WelderSolverPhaseAFootprintTrafficHelpers.h"
#include "WelderSolverGraphAnalysisAndFootprintInfer.h"
#include "WelderSolverDumpPlanHelpers.h"
#include "WelderSolverDumpReports.h"
} // 命名空间 welder

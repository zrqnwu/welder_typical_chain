#include "WelderCompilerFusionPairBuild.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

using namespace mlir;

namespace {

static int64_t getEnvInt64OrDefault(const char *name, int64_t defaultValue) {
  if (!name || !*name)
    return defaultValue;
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;

  char *end = nullptr;
  long long v = std::strtoll(raw, &end, 10);
  if (!end || *end != '\0')
    return defaultValue;
  return static_cast<int64_t>(v);
}

static bool isRowWiseReductionOp(Operation *op) {
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

static int countNonCutConsumers(const welder::TileGraph &graph, int src) {
  if (src < 0 || src >= static_cast<int>(graph.nodes.size()))
    return 0;
  int count = 0;
  for (int eidx : graph.nodes[static_cast<size_t>(src)].outEdges) {
    if (eidx < 0 || eidx >= static_cast<int>(graph.edges.size()))
      continue;
    if (!graph.edges[static_cast<size_t>(eidx)].isCut)
      ++count;
  }
  return count;
}

static int64_t getNodeIdOrIdx(const welder::TileGraph &graph, int idx) {
  if (idx < 0 || idx >= static_cast<int>(graph.nodes.size()))
    return static_cast<int64_t>(idx);
  Operation *op0 = graph.nodes[static_cast<size_t>(idx)].op;
  if (!op0)
    return static_cast<int64_t>(idx);
  if (auto idAttr = op0->getAttrOfType<IntegerAttr>("welder.node_id"))
    return idAttr.getInt();
  return static_cast<int64_t>(idx);
}

} // namespace

namespace welder::compiler {

ThreadFusionFromAttrsResult buildThreadFusionPairSpecsFromKernelAttrs(
    mlir::ModuleOp module, int64_t maxConnectLevel, bool enableInferFallback) {
  ThreadFusionFromAttrsResult out;

  llvm::DenseMap<int64_t, Operation *> idToOp;
  module.walk([&](linalg::LinalgOp op) {
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("welder.node_id"))
      idToOp[idAttr.getInt()] = op.getOperation();
  });

  module.walk([&](linalg::LinalgOp op) {
    Operation *op0 = op.getOperation();
    if (!op0)
      return;

    auto consAttr = op0->getAttrOfType<IntegerAttr>("welder.thread_fuse_into");
    auto consOperandAttr =
        op0->getAttrOfType<IntegerAttr>("welder.thread_fuse_into_operand");
    if (!consAttr)
      return;

    auto prodIdAttr = op0->getAttrOfType<IntegerAttr>("welder.node_id");
    auto kidAttr = op0->getAttrOfType<IntegerAttr>("welder.kernel_id");
    if (prodIdAttr && kidAttr) {
      ++out.threadFusionAttrPairs;
      if (consOperandAttr)
        ++out.threadFusionAttrPairsWithOperand;
    }
    if (!prodIdAttr || !kidAttr)
      return;

    int64_t consId = consAttr.getInt();
    Operation *consOp = idToOp.lookup(consId);
    if (!consOp)
      return;
    auto consKidAttr = consOp->getAttrOfType<IntegerAttr>("welder.kernel_id");
    if (!consKidAttr || consKidAttr.getInt() != kidAttr.getInt())
      return;

    ThreadFusionPairSpec p;
    p.kernelId = kidAttr.getInt();
    p.producerNodeId = prodIdAttr.getInt();
    p.consumerNodeId = consId;
    if (consOperandAttr)
      p.consumerOperand = consOperandAttr.getInt();
    out.pairs.push_back(std::move(p));
  });

  out.explicitThreadFusionPairs = static_cast<int64_t>(out.pairs.size());

  if (!(enableInferFallback && out.pairs.empty() && maxConnectLevel >= 2))
    return out;

  llvm::DenseMap<Operation *, int> sameKernelConsumerCount;
  module.walk([&](linalg::LinalgOp consumer) {
    Operation *consumerOp = consumer.getOperation();
    if (!consumerOp)
      return;
    auto consKidAttr = consumerOp->getAttrOfType<IntegerAttr>("welder.kernel_id");
    if (!consKidAttr)
      return;
    int64_t consKid = consKidAttr.getInt();
    for (Value operand : consumerOp->getOperands()) {
      Operation *def = operand.getDefiningOp();
      auto producer = dyn_cast_or_null<linalg::LinalgOp>(def);
      if (!producer)
        continue;
      Operation *producerOp = producer.getOperation();
      auto prodKidAttr = producerOp->getAttrOfType<IntegerAttr>("welder.kernel_id");
      if (!prodKidAttr || prodKidAttr.getInt() != consKid)
        continue;
      sameKernelConsumerCount[producerOp] += 1;
    }
  });

  module.walk([&](linalg::LinalgOp producer) {
    Operation *producerOp = producer.getOperation();
    if (!producerOp)
      return;
    auto prodIdAttr = producerOp->getAttrOfType<IntegerAttr>("welder.node_id");
    auto prodKidAttr = producerOp->getAttrOfType<IntegerAttr>("welder.kernel_id");
    if (!prodIdAttr || !prodKidAttr)
      return;
    if (producer.getNumReductionLoops() != 0)
      return;

    int64_t prodKid = prodKidAttr.getInt();
    int nonCutConsumers = sameKernelConsumerCount.lookup(producerOp);
    if (nonCutConsumers <= 0)
      return;

    bool cheapDupProducer = false;
    if (auto producerGen = dyn_cast<linalg::GenericOp>(producerOp))
      cheapDupProducer = isCheapDuplicateFusionProducer(producerGen);
    if (nonCutConsumers != 1 &&
        !(cheapDupProducer && nonCutConsumers > 0 && nonCutConsumers <= 2))
      return;

    for (Value outValue : producerOp->getResults()) {
      for (OpOperand &use : outValue.getUses()) {
        auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (!consumer || !consumer.getOperation())
          continue;
        if (consumer.getNumReductionLoops() != 0)
          continue;
        if (consumer.getNumLoops() != producer.getNumLoops())
          continue;

        auto consIdAttr =
            consumer.getOperation()->getAttrOfType<IntegerAttr>("welder.node_id");
        auto consKidAttr = consumer.getOperation()->getAttrOfType<IntegerAttr>(
            "welder.kernel_id");
        if (!consIdAttr || !consKidAttr || consKidAttr.getInt() != prodKid)
          continue;

        ThreadFusionPairSpec p;
        p.kernelId = prodKid;
        p.producerNodeId = prodIdAttr.getInt();
        p.consumerNodeId = consIdAttr.getInt();
        p.consumerOperand = static_cast<int64_t>(use.getOperandNumber());
        out.pairs.push_back(std::move(p));
      }
    }
  });

  out.inferredThreadFusionPairs = !out.pairs.empty();
  return out;
}

std::vector<RowReductionFusionPairSpec> buildRowReductionFusionPairSpecs(
    const welder::TileGraph &graph, const std::vector<int64_t> &nodeToKernel,
    bool enableRowReductionChainReuseFusion, bool enableTensorCoreTf32,
    bool enableTensorCoreF16) {
  std::vector<RowReductionFusionPairSpec> out;
  if (!enableRowReductionChainReuseFusion)
    return out;

  llvm::SmallDenseSet<uint64_t, 32> seenPairs;
  for (const welder::TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    int src = e.src;
    int dst = e.dst;
    if (src < 0 || dst < 0)
      continue;
    if (src >= static_cast<int>(graph.nodes.size()) ||
        dst >= static_cast<int>(graph.nodes.size()))
      continue;
    if (src >= static_cast<int>(nodeToKernel.size()) ||
        dst >= static_cast<int>(nodeToKernel.size()))
      continue;
    if (nodeToKernel[static_cast<size_t>(src)] < 0 ||
        nodeToKernel[static_cast<size_t>(dst)] < 0)
      continue;
    if (nodeToKernel[static_cast<size_t>(src)] !=
        nodeToKernel[static_cast<size_t>(dst)])
      continue;
    if (!isRowWiseReductionOp(graph.nodes[static_cast<size_t>(dst)].op))
      continue;

    auto prodGen =
        dyn_cast_or_null<linalg::GenericOp>(graph.nodes[static_cast<size_t>(src)].op);
    if (!prodGen)
      continue;
    if (prodGen.getNumReductionLoops() != 0)
      continue;
    const bool cheapDupProducer = isCheapDuplicateFusionProducer(prodGen);
    const bool allow2DCheapProducer =
        cheapDupProducer && prodGen.getNumLoops() == 2 &&
        !(enableTensorCoreTf32 || enableTensorCoreF16);
    if (prodGen.getNumLoops() != 1 && !allow2DCheapProducer)
      continue;

    int nonCutConsumers = countNonCutConsumers(graph, src);
    if (nonCutConsumers != 1 &&
        !(cheapDupProducer && nonCutConsumers > 0 && nonCutConsumers <= 2))
      continue;

    int64_t prodId = getNodeIdOrIdx(graph, src);
    int64_t consId = getNodeIdOrIdx(graph, dst);
    if (prodId < 0 || consId < 0)
      continue;
    uint64_t key = (static_cast<uint64_t>(prodId) << 32) |
                   (static_cast<uint64_t>(consId) & 0xffffffffULL);
    if (!seenPairs.insert(key).second)
      continue;

    RowReductionFusionPairSpec p;
    p.kernelId = nodeToKernel[static_cast<size_t>(src)];
    p.producerNodeId = prodId;
    p.consumerNodeId = consId;
    p.fuseIntoBlockForall = allow2DCheapProducer;
    out.push_back(std::move(p));
  }
  return out;
}

std::vector<ThreadFusionPairSpec> buildThreadFusionPairSpecsFromGraph(
    const welder::TileGraph &graph, const std::vector<int64_t> &nodeToKernel,
    const welder::SolveOptions &solveOpts, ThreadFusionDecision *decision) {
  std::vector<ThreadFusionPairSpec> out;

  int registerFuseMinConnectLevel = welder::kConnectLevelRegister;
  bool promoteSharedEdgesForRegisterFuse = false;
  if (solveOpts.maxConnectLevel >= 2) {
    registerFuseMinConnectLevel = welder::kConnectLevelRegister;
    if (solveOpts.maxConnectLevel > welder::kConnectLevelRegister) {
      int recursiveInnerMinLevelExclusive =
          solveOpts.paperRecursiveInnerMinLevelExclusive;
      const int recursiveMaxStages =
          solveOpts.paperRecursiveMaxStages > 0
              ? std::max(1, solveOpts.paperRecursiveMaxStages)
              : 0;
      if (recursiveInnerMinLevelExclusive <= welder::kConnectLevelGlobal) {
        if (recursiveMaxStages > 0) {
          recursiveInnerMinLevelExclusive =
              std::max(1, solveOpts.maxConnectLevel - recursiveMaxStages);
        } else {
          recursiveInnerMinLevelExclusive = std::max<int>(
              welder::kConnectLevelShared, solveOpts.maxConnectLevel - 1);
        }
      }
      if (recursiveMaxStages > 0) {
        const int minBoundaryForStageCap =
            std::max(1, solveOpts.maxConnectLevel - recursiveMaxStages);
        recursiveInnerMinLevelExclusive =
            std::max(recursiveInnerMinLevelExclusive, minBoundaryForStageCap);
      }
      recursiveInnerMinLevelExclusive =
          std::max<int>(welder::kConnectLevelShared,
                        std::min(recursiveInnerMinLevelExclusive,
                                 solveOpts.maxConnectLevel - 1));
      registerFuseMinConnectLevel =
          std::max<int>(welder::kConnectLevelRegister,
                        recursiveInnerMinLevelExclusive + 1);
    }
    promoteSharedEdgesForRegisterFuse =
        getEnvInt64OrDefault("WELDER_CONNECT2_PROMOTE_SHARED_EDGES", 0) != 0;
  }

  if (decision) {
    decision->registerFuseMinConnectLevel = registerFuseMinConnectLevel;
    decision->promoteSharedEdgesForRegisterFuse =
        promoteSharedEdgesForRegisterFuse;
  }

  if (solveOpts.maxConnectLevel < 2)
    return out;

  for (const welder::TileGraphEdge &e : graph.edges) {
    if (e.isCut)
      continue;
    const bool registerConnected = e.connectLevel >= registerFuseMinConnectLevel;
    const bool sharedConnected = e.connectLevel >= welder::kConnectLevelShared;
    if (!registerConnected &&
        !(promoteSharedEdgesForRegisterFuse && sharedConnected))
      continue;

    int src = e.src;
    int dst = e.dst;
    if (src < 0 || dst < 0)
      continue;
    if (src >= static_cast<int>(graph.nodes.size()) ||
        dst >= static_cast<int>(graph.nodes.size()))
      continue;
    if (src >= static_cast<int>(nodeToKernel.size()) ||
        dst >= static_cast<int>(nodeToKernel.size()))
      continue;
    if (nodeToKernel[static_cast<size_t>(src)] < 0 ||
        nodeToKernel[static_cast<size_t>(dst)] < 0)
      continue;
    if (nodeToKernel[static_cast<size_t>(src)] !=
        nodeToKernel[static_cast<size_t>(dst)])
      continue;

    auto srcGen =
        dyn_cast_or_null<linalg::GenericOp>(graph.nodes[static_cast<size_t>(src)].op);
    auto dstGen =
        dyn_cast_or_null<linalg::GenericOp>(graph.nodes[static_cast<size_t>(dst)].op);
    if (!srcGen || !dstGen)
      continue;
    if (srcGen.getNumReductionLoops() != 0 || dstGen.getNumReductionLoops() != 0)
      continue;
    if (srcGen.getNumLoops() != dstGen.getNumLoops())
      continue;

    const bool cheapDupProducer = isCheapDuplicateFusionProducer(srcGen);
    int nonCutConsumers = countNonCutConsumers(graph, src);
    if (nonCutConsumers != 1 &&
        !(cheapDupProducer && nonCutConsumers > 0 && nonCutConsumers <= 2))
      continue;

    int64_t prodId = getNodeIdOrIdx(graph, src);
    int64_t consId = getNodeIdOrIdx(graph, dst);
    if (prodId < 0 || consId < 0)
      continue;

    ThreadFusionPairSpec p;
    p.kernelId = nodeToKernel[static_cast<size_t>(src)];
    p.producerNodeId = prodId;
    p.consumerNodeId = consId;
    p.consumerOperand = e.dstOperand;
    out.push_back(std::move(p));
  }

  return out;
}

} // namespace welder::compiler

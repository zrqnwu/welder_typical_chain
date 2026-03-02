#ifndef WELDER_COMPILER_FUSION_PAIR_BUILD_H
#define WELDER_COMPILER_FUSION_PAIR_BUILD_H

#include "WelderSolverLib.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <vector>

namespace welder::compiler {

// 行归约链融合配对（供 compiler 主流程转换为 transform 层配对结构）。
struct RowReductionFusionPairSpec {
  int64_t kernelId = 0;
  int64_t producerNodeId = -1;
  int64_t consumerNodeId = -1;
  bool fuseIntoBlockForall = false;
};

// 寄存器级融合配对（供 compiler 主流程转换为 transform 层配对结构）。
struct ThreadFusionPairSpec {
  int64_t kernelId = 0;
  int64_t producerNodeId = -1;
  int64_t consumerNodeId = -1;
  int64_t consumerOperand = -1;
};

// ThreadFusion 额外决策信息（用于 trace 输出）。
struct ThreadFusionDecision {
  int64_t registerFuseMinConnectLevel = welder::kConnectLevelRegister;
  bool promoteSharedEdgesForRegisterFuse = false;
};

// kernel attrs 路径（--codegen-from-kernel-attrs）下的 thread-fusion 构建统计。
struct ThreadFusionFromAttrsResult {
  std::vector<ThreadFusionPairSpec> pairs;
  int64_t threadFusionAttrPairs = 0;
  int64_t threadFusionAttrPairsWithOperand = 0;
  int64_t explicitThreadFusionPairs = 0;
  bool inferredThreadFusionPairs = false;
};

// 从现有节点属性与 SSA use 边推导 thread-fusion 配对。
ThreadFusionFromAttrsResult buildThreadFusionPairSpecsFromKernelAttrs(
    mlir::ModuleOp module, int64_t maxConnectLevel, bool enableInferFallback);

// 基于 tile-graph 构建“逐元素 -> 行归约”融合配对。
std::vector<RowReductionFusionPairSpec> buildRowReductionFusionPairSpecs(
    const welder::TileGraph &graph, const std::vector<int64_t> &nodeToKernel,
    bool enableRowReductionChainReuseFusion, bool enableTensorCoreTf32,
    bool enableTensorCoreF16);

// 基于 tile-graph 构建“逐元素 -> 逐元素”寄存器融合配对。
std::vector<ThreadFusionPairSpec> buildThreadFusionPairSpecsFromGraph(
    const welder::TileGraph &graph, const std::vector<int64_t> &nodeToKernel,
    const welder::SolveOptions &solveOpts, ThreadFusionDecision *decision);

} // namespace welder::compiler

#endif

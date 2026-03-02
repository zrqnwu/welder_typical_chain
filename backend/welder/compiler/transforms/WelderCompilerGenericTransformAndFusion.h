#ifndef WELDER_COMPILER_GENERIC_TRANSFORM_AND_FUSION_H
#define WELDER_COMPILER_GENERIC_TRANSFORM_AND_FUSION_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace welder::compiler {

struct KernelSpec {
  int64_t kernelId = 0;
  std::string opName;
  llvm::SmallVector<int64_t, 8> tileSizes;
  bool swapXY = false;
  int64_t rowReductionCount = 0;
  // 切边代码生成：按确定性顺序（逆拓扑、sink->source）将 producer
  // 融合到 kernel root 的 block 级 forall。
  // 这样可避免多跳 producer 在其消费者链尚未融合前无法融合的脆弱行为。
  llvm::SmallVector<int64_t, 32> orderedProducerNodeIds;
};

struct RowReductionFusionPair {
  int64_t kernelId = 0;
  int64_t producerNodeId = -1;
  int64_t consumerNodeId = -1;
  bool fuseIntoBlockForall = false;
};

struct ThreadFusionPair {
  int64_t kernelId = 0;
  int64_t producerNodeId = -1;
  int64_t consumerNodeId = -1;
  int64_t consumerOperand = -1;
};

mlir::OwningOpRef<mlir::ModuleOp>
buildGenericTransformLibrary(mlir::MLIRContext *ctx,
                             llvm::StringRef targetOpName,
                             llvm::ArrayRef<int64_t> l1TileSizes,
                             llvm::ArrayRef<int64_t> l2TileSizes,
                             bool enableFusion,
                             llvm::StringRef consumerOpName,
                             llvm::ArrayRef<int64_t> consumerTileSizes,
                             bool swapBlockDims,
                             bool skipMapForallToBlocks);

// 保持 connect_level=2 融合的确定性与可组合性：
// - 去重重复的 producer->consumer 配对；
// - 按上游->下游排序，使多跳链（A->B->C）按 (A->B) 再 (B->C) 融合，
//   便于第一次融合结果传递到第二次融合。
void normalizeThreadFusionPairs(std::vector<ThreadFusionPair> &pairs);

} // namespace welder::compiler

#endif

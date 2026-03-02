#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>

namespace welder::compiler {

// 行归约链协作式 staging（2D tile -> workgroup）。
void stageRowReduction2DTileToWorkgroup(
    mlir::ModuleOp module, int64_t smemBytes, int64_t reserveBytes = 16 * 1024,
    bool enableInPlace2DReuse = true, bool enableVectorize = false,
    bool enableAsyncCopy = false, bool enableSoftwarePipelining = false,
    bool asyncBypassL1 = true, bool relaxBarriers = false,
    int64_t vectorWidth = 0);

// 行归约 scratch/allocal intermediates 提升到 workgroup。
void promoteRowReductionScratchToWorkgroup(mlir::ModuleOp module);
void promoteSharedRowReductionResultAllocasToWorkgroup(mlir::ModuleOp module);

// 典型链 1D 中间缓冲提升到 workgroup，避免 host/global 往返。
void promoteLaunchLocal1DBuffersToWorkgroup(mlir::ModuleOp module);

} // namespace welder::compiler


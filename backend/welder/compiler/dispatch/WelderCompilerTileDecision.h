#ifndef WELDER_COMPILER_TILE_DECISION_H
#define WELDER_COMPILER_TILE_DECISION_H

#include <cstdint>

namespace welder::compiler {

struct BlockDims {
  int64_t x = 1;
  int64_t y = 1;
};

// 在线程 tile 选择上做保守回退：优先使用给定值，其次尝试 {4,2,1} 中可整除的值。
int64_t pickThreadTileDivisible(int64_t tile, int64_t preferred);

// 向上取整除法（对非正输入返回 0）。
int64_t ceilDivI64(int64_t a, int64_t b);

// 适用于“tile 能被 thread-tile 整除”的 block 维度推导。
BlockDims computeBlockDimsExact(int64_t tileM, int64_t tileN,
                                int64_t threadTileM, int64_t threadTileN,
                                bool swapBlockDims);

// 适用于“tile/thread-basis 需要向上取整”的 block 维度推导。
BlockDims computeBlockDimsCeil(int64_t basisM, int64_t basisN,
                               int64_t threadTileM, int64_t threadTileN,
                               bool swapBlockDims);

} // namespace welder::compiler

#endif

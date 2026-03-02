#include "WelderCompilerTileDecision.h"

#include <algorithm>

namespace welder::compiler {

int64_t pickThreadTileDivisible(int64_t tile, int64_t preferred) {
  if (preferred > 0 && tile > 0 && tile % preferred == 0)
    return preferred;
  const int64_t fallbacks[] = {4, 2, 1};
  for (int64_t v : fallbacks) {
    if (v > 0 && tile > 0 && tile % v == 0)
      return v;
  }
  return 1;
}

int64_t ceilDivI64(int64_t a, int64_t b) {
  if (a <= 0 || b <= 0)
    return 0;
  return (a + b - 1) / b;
}

BlockDims computeBlockDimsExact(int64_t tileM, int64_t tileN,
                                int64_t threadTileM, int64_t threadTileN,
                                bool swapBlockDims) {
  BlockDims out;
  if (!swapBlockDims) {
    out.x = std::max<int64_t>(1, tileN / std::max<int64_t>(1, threadTileN));
    out.y = std::max<int64_t>(1, tileM / std::max<int64_t>(1, threadTileM));
  } else {
    out.x = std::max<int64_t>(1, tileM / std::max<int64_t>(1, threadTileM));
    out.y = std::max<int64_t>(1, tileN / std::max<int64_t>(1, threadTileN));
  }
  return out;
}

BlockDims computeBlockDimsCeil(int64_t basisM, int64_t basisN,
                               int64_t threadTileM, int64_t threadTileN,
                               bool swapBlockDims) {
  BlockDims out;
  if (!swapBlockDims) {
    out.x = std::max<int64_t>(1, ceilDivI64(basisN, threadTileN));
    out.y = std::max<int64_t>(1, ceilDivI64(basisM, threadTileM));
  } else {
    out.x = std::max<int64_t>(1, ceilDivI64(basisM, threadTileM));
    out.y = std::max<int64_t>(1, ceilDivI64(basisN, threadTileN));
  }
  return out;
}

} // namespace welder::compiler

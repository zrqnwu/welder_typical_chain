#include "wtc/transform/internal/BuildCutEdgesTransformInternal.h"

#include "wtc/backend/Toolchain.h"

#include <sstream>

namespace wtc::transform::internal {

std::string buildCommonWelderCompilerFlags(
    const std::string &inputPath, const wtc::scheduler::SearchResult &search,
    const wtc::transform::BuildConfig &config) {
  std::ostringstream args;
  args << wtc::backend::shellQuote(inputPath)
       << " --enable-generic-problem"
       << " --enable-tile-propagation"
       << " --enable-cut-edges"
       << " --enable-two-level-schedule"
       << " --require-perfect-tiling=false"
       << " --force-tile-m " << search.tileM
       << " --force-tile-n " << search.tileN
       << " --force-tile-k " << search.tileK
       << " --enable-register-level-schedule"
       << " --thread-tile-m " << search.threadTileM
       << " --thread-tile-n " << search.threadTileN
       << " --max-connect-level=" << config.maxConnectLevel;

  if (config.fused) {
    args << " --reduction-chain-split-broadcast-edges=false"
         << " --enable-row-reduction-chain-reuse-fusion"
         << " --enable-row-reduction-input-promotion"
         << " --enable-matmul-softmax-shared-reuse-fusion";
  } else {
    args << " --reduction-chain-split-broadcast-edges=true";
  }
  return args.str();
}

} // namespace wtc::transform::internal

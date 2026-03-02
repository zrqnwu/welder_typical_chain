#include "WelderSolverCostModel.h"

#include <algorithm>

namespace welder::solver {

void printIntList(const std::vector<int64_t> &xs, llvm::raw_ostream &os) {
  os << "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i)
      os << ", ";
    os << xs[i];
  }
  os << "]";
}

void printCandidateCostSummary(const welder::Candidate &c, llvm::raw_ostream &os) {
  os << "tile_m=" << c.tileM << " tile_n=" << c.tileN << " tile_k=" << c.tileK
     << "\n";
  if (c.threadTileM > 0 && c.threadTileN > 0) {
    int64_t blockDimX =
        (c.tileN > 0 && c.threadTileN > 0) ? (c.tileN / c.threadTileN) : 0;
    int64_t blockDimY =
        (c.tileM > 0 && c.threadTileM > 0) ? (c.tileM / c.threadTileM) : 0;
    os << "  thread_tile_m=" << c.threadTileM
       << " thread_tile_n=" << c.threadTileN
       << " block_dim=(" << blockDimX << "x" << blockDimY << ")"
       << " threads=" << (blockDimX * blockDimY) << "\n";
  }
  if (!c.loopTileExtents.empty()) {
    os << "  loop_tile_extents=";
    printIntList(c.loopTileExtents, os);
    os << "\n";
  }
  os << "  smem_bytes=" << c.smemBytes
     << " est_footprint_bytes=" << c.estFootprintBytes
     << " blocks=(" << c.blocksM << "x" << c.blocksN << ")"
     << " total=" << c.blocksTotal
     << " blocks_per_sm=" << c.blocksPerSM
     << " waves=" << c.numWave
     << " est_regs_per_thread=" << c.estRegsPerThread << "\n";
  os << "  traffic_bytes(A,B,C,Cut,total)=("
     << static_cast<uint64_t>(c.traffic.bytesA) << ","
     << static_cast<uint64_t>(c.traffic.bytesB) << ","
     << static_cast<uint64_t>(c.traffic.bytesC) << ","
     << static_cast<uint64_t>(c.traffic.bytesCut) << ","
     << static_cast<uint64_t>(c.traffic.totalBytes()) << ")\n";
  if (c.cost.sharedToRegBytes > 0.0)
    os << "  shared_to_reg_bytes=" << c.cost.sharedToRegBytes << "\n";
  os << "  score=" << c.score << "\n";
}

void printTopKCandidateScores(const welder::SolveResult &sr, std::size_t k,
                              llvm::raw_ostream &os) {
  const size_t limit = std::min<std::size_t>(k, sr.sortedCandidates.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &c = sr.sortedCandidates[i];
    os << "[" << i << "] "
       << "tm=" << c.tileM << " tn=" << c.tileN << " tk=" << c.tileK
       << " ttm=" << c.threadTileM << " ttn=" << c.threadTileN
       << " score=" << c.score << "\n";
  }
}

} // namespace welder::solver

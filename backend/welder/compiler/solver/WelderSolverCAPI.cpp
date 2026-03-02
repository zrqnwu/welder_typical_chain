#include "WelderSolverCAPI.h"

#include "WelderSolveOptionDefaults.h"
#include "WelderSolverLib.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace {

void setError(char *buf, size_t bufSize, const std::string &msg) {
  if (!buf || bufSize == 0)
    return;
  const size_t n = std::min(bufSize - 1, msg.size());
  std::memcpy(buf, msg.data(), n);
  buf[n] = '\0';
}

bool writeBestSummaryJson(const welder::SolveResult &sr,
                          const welder::SolveOptions &opts,
                          const std::string &path, std::string &error) {
  if (sr.sortedCandidates.empty()) {
    error = "empty candidate list";
    return false;
  }
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream ofs(path);
  if (!ofs) {
    error = "failed to open best summary json: " + path;
    return false;
  }

  const welder::Candidate &best = sr.sortedCandidates.front();
  const double profiled =
      best.cost.profiledMs.has_value() ? *best.cost.profiledMs : -1.0;

  ofs << "{\n";
  ofs << "  \"problem\": {\"m\": " << sr.problem.m << ", \"n\": " << sr.problem.n
      << ", \"k\": " << sr.problem.k << "},\n";
  ofs << "  \"arch\": {\"smemBytes\": " << opts.arch.smemBytes
      << ", \"numSM\": " << opts.arch.numSM
      << ", \"warpSize\": " << opts.arch.warpSize
      << ", \"elementBytes\": " << opts.arch.elementBytes << "},\n";
  ofs << "  \"best\": {";
  ofs << "\"tileM\": " << best.tileM << ", \"tileN\": " << best.tileN
      << ", \"tileK\": " << best.tileK
      << ", \"threadTileM\": " << best.threadTileM
      << ", \"threadTileN\": " << best.threadTileN
      << ", \"score\": " << best.score
      << ", \"estimatedLatency\": " << best.cost.estimatedLatency
      << ", \"profiledMs\": " << profiled
      << ", \"bytesCut\": " << best.traffic.bytesCut
      << "}\n";
  ofs << "}\n";
  return true;
}

bool writeCandidatesTsv(const welder::SolveResult &sr, const std::string &path,
                        std::string &error) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out) {
    error = "failed to open candidates tsv: " + path;
    return false;
  }
  out << "rank\ttileM\ttileN\ttileK\tthreadTileM\tthreadTileN\tscore\t"
         "estimatedLatency\tprofiledMs\tbytesA\tbytesB\tbytesC\tbytesCut\n";

  for (size_t i = 0; i < sr.sortedCandidates.size(); ++i) {
    const welder::Candidate &c = sr.sortedCandidates[i];
    const double profiled = c.cost.profiledMs.has_value() ? *c.cost.profiledMs : -1.0;
    out << i << "\t" << c.tileM << "\t" << c.tileN << "\t" << c.tileK << "\t"
        << c.threadTileM << "\t" << c.threadTileN << "\t" << c.score << "\t"
        << c.cost.estimatedLatency << "\t" << profiled << "\t"
        << c.traffic.bytesA << "\t" << c.traffic.bytesB << "\t"
        << c.traffic.bytesC << "\t" << c.traffic.bytesCut << "\n";
  }
  return true;
}

} // namespace

extern "C" int welder_solver_solve_typical_chain(
    const char *input_mlir_path, const char *best_summary_json_path,
    const char *candidates_tsv_path, int64_t max_connect_level, int /*verbose*/,
    int64_t *out_tile_m, int64_t *out_tile_n, int64_t *out_tile_k,
    int64_t *out_thread_tile_m, int64_t *out_thread_tile_n,
    char *error_buffer, size_t error_buffer_size) {
  setError(error_buffer, error_buffer_size, "");

  if (!input_mlir_path || !*input_mlir_path) {
    setError(error_buffer, error_buffer_size, "input_mlir_path is empty");
    return 1;
  }
  if (!best_summary_json_path || !*best_summary_json_path) {
    setError(error_buffer, error_buffer_size, "best_summary_json_path is empty");
    return 1;
  }
  if (!candidates_tsv_path || !*candidates_tsv_path) {
    setError(error_buffer, error_buffer_size, "candidates_tsv_path is empty");
    return 1;
  }
  if (!out_tile_m || !out_tile_n || !out_tile_k || !out_thread_tile_m ||
      !out_thread_tile_n) {
    setError(error_buffer, error_buffer_size, "output tile pointers are null");
    return 1;
  }

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::affine::AffineDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();

  std::string openErr;
  auto file = mlir::openInputFile(input_mlir_path, &openErr);
  if (!file) {
    setError(error_buffer, error_buffer_size,
             "cannot open input MLIR: " + std::string(input_mlir_path) + " (" +
                 openErr + ")");
    return 2;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    setError(error_buffer, error_buffer_size,
             "failed to parse MLIR: " + std::string(input_mlir_path));
    return 2;
  }

  welder::SolveOptions opts;
  opts.enableTilePropagation = true;
  opts.enableCutEdges = true;
  opts.enableTwoLevelSchedule = true;
  opts.requirePerfectTiling = false;
  opts.enableRegisterLevelSchedule = true;
  opts.maxConnectLevel = std::max<int64_t>(0, max_connect_level);
  welder::applyPaperModeDefaults(opts,
                                 /*autoCandidatesExplicitlySet=*/false,
                                 /*maxConnectLevelExplicitlySet=*/true,
                                 /*profilingEnabled=*/false);

  welder::SolveResult sr = welder::solveGeneric(*module, opts);
  if (sr.sortedCandidates.empty()) {
    setError(error_buffer, error_buffer_size,
             "solver returned no valid candidates");
    return 3;
  }

  std::string ioErr;
  if (!writeBestSummaryJson(sr, opts, best_summary_json_path, ioErr)) {
    setError(error_buffer, error_buffer_size, ioErr);
    return 4;
  }
  if (!writeCandidatesTsv(sr, candidates_tsv_path, ioErr)) {
    setError(error_buffer, error_buffer_size, ioErr);
    return 4;
  }

  const welder::Candidate &best = sr.sortedCandidates.front();
  *out_tile_m = best.tileM;
  *out_tile_n = best.tileN;
  *out_tile_k = best.tileK;
  *out_thread_tile_m = best.threadTileM;
  *out_thread_tile_n = best.threadTileN;
  return 0;
}

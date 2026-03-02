#include "wtc/pipeline/PipelineRunner.h"

#include "wtc/ir/ChainCanonicalize.h"
#include "wtc/ir/Tagging.h"
#include "wtc/runtime/PostbufferizeFixups.h"
#include "wtc/scheduler/Search.h"
#include "wtc/transform/BuildCutEdgesTransform.h"

#include <filesystem>
#include <iostream>

namespace {

wtc::pipeline::StageArtifacts
buildStageArtifacts(const wtc::pipeline::RunOptions &options) {
  wtc::pipeline::StageArtifacts a;
  a.rootDir = options.outputDir;

  const std::filesystem::path root = options.outputDir;
  const std::filesystem::path ir = root / "ir";
  a.irDir = ir.string();
  a.canonicalizedMlirPath = (ir / "01.canonicalized.mlir").string();
  a.taggedMlirPath = (ir / "02.tagged.mlir").string();
  a.tagsJsonPath = (ir / "02.tags.json").string();

  a.searchDir = (root / "search").string();
  a.bestJsonPath = (root / "search" / "best.json").string();

  a.compileDir =
      (root / (options.fused ? "fused" : "baseline")).string();
  return a;
}

bool hasAnyTileOverride(const wtc::pipeline::RunOptions &options) {
  return options.tileM > 0 || options.tileN > 0 || options.tileK > 0 ||
         options.threadTileM > 0 || options.threadTileN > 0;
}

void applyTileOverrides(const wtc::pipeline::RunOptions &options,
                        wtc::scheduler::SearchResult &best) {
  if (options.tileM > 0)
    best.tileM = options.tileM;
  if (options.tileN > 0)
    best.tileN = options.tileN;
  if (options.tileK > 0)
    best.tileK = options.tileK;
  if (options.threadTileM > 0)
    best.threadTileM = options.threadTileM;
  if (options.threadTileN > 0)
    best.threadTileN = options.threadTileN;
}

} // namespace

namespace wtc::pipeline {

int runTypicalChainPipeline(const RunOptions &options) {
  std::string diag;

  if (options.outputDir.empty()) {
    std::cerr << "error: output dir is empty\n";
    return 1;
  }
  const StageArtifacts artifacts = buildStageArtifacts(options);
  std::filesystem::create_directories(artifacts.rootDir);
  std::filesystem::create_directories(artifacts.irDir);

  if (options.verbose)
    std::cout << "[wtc] phase: canonicalize\n";
  if (!wtc::ir::canonicalizeMatmulSoftmaxChain(options.inputPath,
                                               artifacts.canonicalizedMlirPath,
                                               diag)) {
    std::cerr << "error: canonicalize failed: " << diag << "\n";
    return 1;
  }

  if (options.verbose)
    std::cout << "[wtc] phase: tagging\n";
  if (!wtc::ir::tagTypicalChainOps(artifacts.canonicalizedMlirPath,
                                   artifacts.taggedMlirPath,
                                   artifacts.tagsJsonPath,
                                   diag)) {
    std::cerr << "error: tagging failed: " << diag << "\n";
    return 1;
  }

  const std::filesystem::path searchDir = artifacts.searchDir;
  std::filesystem::create_directories(searchDir);
  // 后端统一消费 canonicalize+tagging 后的输入，保证阶段产物真实参与执行。
  const std::string backendInputMlir = artifacts.taggedMlirPath;

  wtc::scheduler::SearchResult best;
  bool haveBest = false;

  if (!options.bestJsonPath.empty()) {
    if (options.verbose)
      std::cout << "[wtc] load best json: " << options.bestJsonPath << "\n";
    if (!wtc::scheduler::loadBestFromJson(options.bestJsonPath, best, diag)) {
      std::cerr << "error: failed to load --best-json: " << diag << "\n";
      return 1;
    }
    haveBest = true;
  }

  const bool needSearch =
      !haveBest && options.enableSearch &&
      (options.mode == RunMode::Full || options.mode == RunMode::SearchOnly ||
       options.mode == RunMode::CompileOnly);
  if (needSearch) {
    if (options.verbose)
      std::cout << "[wtc] phase: search (" << wtc::backend::toString(options.backendMode)
                << ")\n";

    wtc::scheduler::SearchConfig searchCfg;
    searchCfg.backendRoot = options.backendRoot;
    searchCfg.workDir = searchDir.string();
    searchCfg.maxConnectLevel = options.maxConnectLevel;
    // Stability guard: running solver API then compile API in the same process
    // can trigger backend-side assertions in current vendored code.
    // 默认 full 模式保持隔离；显式 pureApiFull 才允许同进程直连。
    searchCfg.backendMode = options.backendMode;
    if (options.backendMode == wtc::backend::BackendMode::Api &&
        !options.pureApiFull &&
        options.mode != RunMode::SearchOnly) {
      searchCfg.backendMode = wtc::backend::BackendMode::ProcessChain;
      if (options.verbose) {
        std::cout << "[wtc] note: search falls back to process_chain mode for "
                     "stability before api compile\n";
      }
    } else if (options.backendMode == wtc::backend::BackendMode::Api &&
               options.pureApiFull && options.verbose &&
               options.mode != RunMode::SearchOnly) {
      std::cout << "[wtc] note: pure-api-full enabled, run search(api) + "
                   "compile(api) in one process\n";
    }
    searchCfg.verbose = options.verbose;

    wtc::scheduler::SearchArtifacts searchArtifacts;
    if (!wtc::scheduler::runTypicalChainSearch(backendInputMlir,
                                               searchCfg, best,
                                               diag, &searchArtifacts)) {
      std::cerr << "error: search failed: " << diag << "\n";
      return 1;
    }
    haveBest = true;

    if (options.verbose) {
      std::cout << "[wtc] search best: " << searchArtifacts.normalizedBestJsonPath
                << "\n";
      std::cout << "[wtc] search candidates: " << searchArtifacts.candidatesTsvPath
                << "\n";
    }
  }

  if (hasAnyTileOverride(options)) {
    if (options.verbose)
      std::cout << "[wtc] apply tile overrides\n";
    applyTileOverrides(options, best);
    haveBest = true;
  }

  if (!haveBest && options.mode == RunMode::SearchOnly) {
    std::cerr << "error: search-only mode has no result. Provide --best-json or enable search.\n";
    return 1;
  }

  if (!haveBest && !options.enableSearch && options.verbose) {
    std::cout << "[wtc] search disabled, using built-in default tiles\n";
  }

  if (!wtc::scheduler::writeBestToJson(best, artifacts.bestJsonPath, diag)) {
    std::cerr << "error: failed to persist best.json: " << diag << "\n";
    return 1;
  }

  if (options.verbose) {
    std::cout << "[wtc] selected tile=(" << best.tileM << "x" << best.tileN
              << "x" << best.tileK << ") thread_tile=(" << best.threadTileM
              << "x" << best.threadTileN << ")\n";
  }

  if (options.mode == RunMode::SearchOnly) {
    std::cout << "[wtc] search completed. best: " << artifacts.bestJsonPath
              << "\n";
    return 0;
  }

  if (options.verbose)
    std::cout << "[wtc] phase: build transform + compile ("
              << wtc::backend::toString(options.backendMode) << ")\n";

  wtc::transform::BuildConfig buildCfg;
  buildCfg.backendRoot = options.backendRoot;
  buildCfg.outDir = artifacts.compileDir;
  buildCfg.maxConnectLevel = options.maxConnectLevel;
  buildCfg.fused = options.fused;
  buildCfg.backendMode = options.backendMode;
  buildCfg.verbose = options.verbose;

  wtc::transform::BuildArtifacts compileArtifacts;
  if (!wtc::transform::buildCutEdgesTransformModule(backendInputMlir, best,
                                                     buildCfg, compileArtifacts,
                                                     diag)) {
    std::cerr << "error: build/compile failed: " << diag << "\n";
    return 1;
  }

  if (options.verbose)
    std::cout << "[wtc] phase: validate postbufferize artifacts\n";
  if (!wtc::runtime::validatePostbufferizeArtifacts(compileArtifacts.outDir,
                                                    diag)) {
    std::cerr << "error: postbufferize artifact validation failed: " << diag
              << "\n";
    return 1;
  }

  std::cout << "[wtc] pipeline completed. artifact: "
            << compileArtifacts.runnableMlirPath << "\n";
  return 0;
}

} // namespace wtc::pipeline

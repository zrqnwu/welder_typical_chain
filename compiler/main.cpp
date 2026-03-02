#include "wtc/pipeline/PipelineRunner.h"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  wtc::pipeline::RunOptions options;
  int64_t repeat = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      options.inputPath = argv[++i];
    } else if ((arg == "--output-dir" || arg == "--out-dir") &&
               i + 1 < argc) {
      options.outputDir = argv[++i];
    } else if ((arg == "--backend-root" || arg == "--legacy-root") &&
               i + 1 < argc) {
      options.backendRoot = argv[++i];
    } else if (arg == "--best-json" && i + 1 < argc) {
      options.bestJsonPath = argv[++i];
    } else if (arg == "--backend-mode" && i + 1 < argc) {
      std::string diag;
      if (!wtc::backend::parseBackendMode(argv[++i], options.backendMode, diag)) {
        std::cerr << "error: " << diag << "\n";
        return 2;
      }
    } else if (arg == "--max-connect-level" && i + 1 < argc) {
      options.maxConnectLevel = std::stoll(argv[++i]);
    } else if (arg == "--tile-m" && i + 1 < argc) {
      options.tileM = std::stoll(argv[++i]);
    } else if (arg == "--tile-n" && i + 1 < argc) {
      options.tileN = std::stoll(argv[++i]);
    } else if (arg == "--tile-k" && i + 1 < argc) {
      options.tileK = std::stoll(argv[++i]);
    } else if (arg == "--thread-tile-m" && i + 1 < argc) {
      options.threadTileM = std::stoll(argv[++i]);
    } else if (arg == "--thread-tile-n" && i + 1 < argc) {
      options.threadTileN = std::stoll(argv[++i]);
    } else if (arg == "--no-search") {
      options.enableSearch = false;
    } else if (arg == "--pure-api-full") {
      options.pureApiFull = true;
    } else if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::stoll(argv[++i]);
    } else if (arg == "--search-only") {
      options.mode = wtc::pipeline::RunMode::SearchOnly;
    } else if (arg == "--compile-only") {
      options.mode = wtc::pipeline::RunMode::CompileOnly;
    } else if (arg == "--baseline") {
      options.fused = false;
      if (options.maxConnectLevel > 0)
        options.maxConnectLevel = 0;
    } else if (arg == "--fused") {
      options.fused = true;
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: wtc-compiler --input <mlir-file> [options]\n"
          << "Options:\n"
          << "  --output-dir <dir>       Output root dir (default /tmp/wtc_out)\n"
          << "  --backend-root <dir>     Path to vendored backend root\n"
          << "  --backend-mode <mode>    shell|process_chain|api (default process_chain)\n"
          << "  --pure-api-full          Disable API search fallback in full mode\n"
          << "  --legacy-root <dir>      Alias of --backend-root for compatibility\n"
          << "  --search-only            Run solver only and dump search artifacts\n"
          << "  --compile-only           Skip solver (unless no tiles provided) and compile\n"
          << "  --best-json <path>       Read tile result from json (tileM/N/K/threadTileM/N)\n"
          << "  --tile-m <n>             Override tileM\n"
          << "  --tile-n <n>             Override tileN\n"
          << "  --tile-k <n>             Override tileK\n"
          << "  --thread-tile-m <n>      Override threadTileM\n"
          << "  --thread-tile-n <n>      Override threadTileN\n"
          << "  --max-connect-level <n>  Connect level for cut-edges path\n"
          << "  --no-search              Do not run solver search\n"
          << "  --repeat <n>             Repeat in one process (stress reentrancy)\n"
          << "  --baseline               Compile baseline variant\n"
          << "  --fused                  Compile fused variant (default)\n"
          << "  --verbose                Verbose logs\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (options.inputPath.empty()) {
    std::cerr << "error: --input is required\n";
    return 2;
  }

  if (repeat <= 0) {
    std::cerr << "error: --repeat must be >= 1\n";
    return 2;
  }

  if (repeat == 1)
    return wtc::pipeline::runTypicalChainPipeline(options);

  for (int64_t i = 0; i < repeat; ++i) {
    wtc::pipeline::RunOptions iterOptions = options;
    iterOptions.outputDir =
        (std::filesystem::path(options.outputDir) / ("iter_" + std::to_string(i)))
            .string();
    if (options.verbose) {
      std::cout << "[wtc] repeat " << (i + 1) << "/" << repeat
                << ", output-dir=" << iterOptions.outputDir << "\n";
    }
    int rc = wtc::pipeline::runTypicalChainPipeline(iterOptions);
    if (rc != 0) {
      std::cerr << "error: repeat iteration failed at i=" << i << ", rc=" << rc
                << "\n";
      return rc;
    }
  }
  return 0;
}

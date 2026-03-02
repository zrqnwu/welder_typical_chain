#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compile typical-chain input to runnable NVVM MLIR in-process.
//
// Output artifacts under out_dir:
// - 03.after_postbufferize.mlir
// - 04.after_workgroup_launch.mlir
// - 04c.after_linalg_to_loops.mlir
// - 05.out.nvvm.runnable.mlir
//
// Returns 0 on success, non-zero on failure.
int welder_compile_typical_chain_to_nvvm(
    const char *input_mlir_path, const char *out_dir,
    const char *workgroup_pass_plugin_path,
    const char *welder_compiler_bin_path, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, int64_t thread_tile_m, int64_t thread_tile_n,
    int64_t max_connect_level, int fused, int verbose, char *error_buffer,
    size_t error_buffer_size);

#ifdef __cplusplus
}
#endif

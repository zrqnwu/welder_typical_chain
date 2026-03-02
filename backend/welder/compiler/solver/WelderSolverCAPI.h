#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, non-zero on failure.
//
// On success:
// - writes best_summary_json_path (best candidate summary)
// - writes candidates_tsv_path (sorted candidate table)
// - fills out best tile fields
//
// On failure:
// - writes diagnostic text into error_buffer (if provided)
int welder_solver_solve_typical_chain(
    const char *input_mlir_path, const char *best_summary_json_path,
    const char *candidates_tsv_path, int64_t max_connect_level, int verbose,
    int64_t *out_tile_m, int64_t *out_tile_n, int64_t *out_tile_k,
    int64_t *out_thread_tile_m, int64_t *out_thread_tile_n,
    char *error_buffer, size_t error_buffer_size);

#ifdef __cplusplus
}
#endif


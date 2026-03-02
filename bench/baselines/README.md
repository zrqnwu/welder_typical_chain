# Baseline Artifacts

This directory stores pinned performance/IR baselines for guard checks.

## Pin baseline

```bash
bash bench/pin_baseline.sh
```

Default output path:

- `bench/baselines/default/`

Pinned files include:

- `ab_summary.tsv`
- `speedup.tsv`
- `search/best.json`
- `baseline/03.after_postbufferize.mlir`
- `baseline/04.after_workgroup_launch.mlir`
- `baseline/04c.after_linalg_to_loops.mlir`
- `baseline/05.out.nvvm.runnable.mlir`
- `fused/03.after_postbufferize.mlir`
- `fused/04.after_workgroup_launch.mlir`
- `fused/04c.after_linalg_to_loops.mlir`
- `fused/05.out.nvvm.runnable.mlir`

## Performance guard

```bash
BASELINE_DIR=bench/baselines/default \
CURRENT_OUT_BASE=/tmp/wtc_ab \
MAX_REGRESSION_PCT=3 \
bash bench/check_perf_guard.sh
```

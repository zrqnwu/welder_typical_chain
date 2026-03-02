# welder_typical_chain

一个只保留单条路径的精简工程：`Matmul -> Softmax` 典型链路。

## 项目目标
- 仅保留典型链路（typical chain）流水线。
- 代码结构尽量紧凑、便于讲解和面试沟通。
- 工程可独立运行，不依赖外部 `welder_try` 路径。

## 当前状态
- `wtc-compiler` 支持显式模式：
  - `--search-only`
  - `--compile-only`
  - 完整流水线（默认）
- 后端执行模式：
  - `--backend-mode inprocess`（默认）：直接编排工具链（`welder-solver`、`welder-compiler`、`mlir-opt`），不经过旧脚本封装
  - `--backend-mode api`：通过 `dlopen` 调用后端 C API（`welder-solver-capi` + `welder-compile-capi`）
    - search(api)：同进程 C API 搜索
    - compile(api)：同进程 C API 编译（`welderCompilerMain`）+ CAPI 内部 lowering
  - `--backend-mode shell`：兼容模式，走历史后端脚本
  - 稳定性默认策略：在 full 模式下若使用 `--backend-mode api`，search 会在 compile(api) 前回退到 `inprocess`
  - 纯 API 全链路实验模式：增加 `--pure-api-full`
- 搜索产物位于 `output_dir/search/`：
  - `best.json`
  - `best_summary.json`
  - `candidates.tsv`
  - `solver.log`
- IR 分阶段产物位于 `output_dir/ir/`：
  - `01.canonicalized.mlir`
  - `01.canonicalize.log`
  - `02.tagged.mlir`
  - `02.tags.json`
- 编译诊断包含 `postbufferize_report.txt`。
- 后端源码已内置到本仓库：
  - `backend/welder/compiler`
  - `backend/welder/mlir_pipeline`

## 构建
```bash
cd /home/zhangruiqi/welder_typical_chain
cmake -S . -B build
cmake --build build -j
```

## 仅搜索
```bash
./build/compiler/wtc-compiler \
  --input mlir/matmul_softmax_chain_f16_native.mlir \
  --output-dir /tmp/wtc_out \
  --backend-root /home/zhangruiqi/welder_typical_chain/backend/welder \
  --backend-mode inprocess \
  --search-only \
  --max-connect-level 1 \
  --verbose
```

## 基于搜索结果编译
```bash
./build/compiler/wtc-compiler \
  --input mlir/matmul_softmax_chain_f16_native.mlir \
  --output-dir /tmp/wtc_out \
  --backend-root /home/zhangruiqi/welder_typical_chain/backend/welder \
  --backend-mode inprocess \
  --compile-only \
  --best-json /tmp/wtc_out/search/best.json \
  --fused \
  --max-connect-level 1 \
  --verbose
```

## 基准与回归
```bash
# 生成搜索产物
bash bench/run_search.sh

# 编译 baseline+fused，做 profile，并输出 ab_summary.tsv + speedup.tsv
bash bench/run_ab.sh

# 回归检查：search/api、named-softmax 分解、full/api、pure-api-full 重复运行
bash bench/run_regression.sh

# 纯 API 全链路压测（同进程循环）
REPEAT=100 bash bench/run_pure_api_stress.sh

# 一键阶段验收（regression + api pair + pure-api stress + AB）
bash bench/run_all_stages.sh

# 固化固定 shape 的 baseline 产物（03/04/04c/05 + ab summary）
bash bench/pin_baseline.sh

# 开启性能回退护栏执行验收（默认 <=3%）
CHECK_PERF_GUARD=1 bash bench/run_all_stages.sh
```

`run_ab.sh` 默认 `VERIFY=0`。如需开启正确性校验：
```bash
VERIFY=1 bash bench/run_ab.sh
```

## 说明
- `--legacy-root` 仍可用，作为 `--backend-root` 的兼容别名。
- 在 bench 脚本中可用 `BACKEND_MODE=shell` 做回退调试。
- `--repeat <n>` 会在同一进程重复执行流水线（用于 API 可重入压测）。
- `--backend-mode api --pure-api-full --repeat>1` 支持同进程循环执行（不强制每轮子进程隔离）。
- full 模式下使用 `--backend-mode api` 时，默认仍是稳定优先：
  除非显式设置 `--pure-api-full`，否则 search 会回退到 `inprocess`。
- Canonicalize 阶段支持两种输入风格：
  - 已分解 softmax 链（`max/exp/sum/div`）
  - 命名 softmax（`linalg.softmax`，通过 transform-interpreter 分解）

# 架构（仅 Typical Chain）

## 范围
当前仅支持：
- `linalg.matmul`
- 按行 softmax 分解（`max / exp / sum / div`）
- generic + cut-edges 流水线

## 流水线
1. `canonicalize`（MLIR pass 阶段）
   - 执行 `mlir-opt --canonicalize`
   - 若输入包含 `linalg.softmax`，先执行 transform-interpreter 分解
     （`transform.structured.decompose_interface`）
   - 产出 `ir/01.canonicalized.mlir`
2. `tagging`（typical-chain 标注 + sidecar 元数据）
   - 产出 `ir/02.tagged.mlir` 和 `ir/02.tags.json`
3. `search`（可选）
   - 产出 `search/best.json`、`search/candidates.tsv`
4. `build transform + compile`
   - 生成 baseline 或 fused 变体
5. `postbufferize` 诊断
   - 产出 `postbufferize_report.txt`
6. 可运行 NVVM 产物（`05.out.nvvm.runnable.mlir`）

## 编译器模式
- `--search-only`：在搜索产物阶段结束
- `--compile-only`：使用 `--best-json` 或 tile 覆盖参数直接编译
- 默认 full 模式：search + compile
- `--repeat <n>`：
  - 所有模式在同一进程中循环运行（用于可重入压测）
- `--backend-mode process_chain|api|shell`：
  - `process_chain`（默认，兼容历史别名 `inprocess`）：
    直接用后端二进制 + `mlir-opt` 编排 solver/compile/lowering
  - `api`：
    - search：同进程调用后端 solver C API
    - compile：同进程调用后端 compile C API，并显式传入工具路径
      （`welderCompilerMain` + CAPI 内部 `workgroup`、`linalg-to-loops`、`nvvm` pass/lowering）
    - full 模式下默认会把 search 回退到 `process_chain`；
      加 `--pure-api-full` 可强制同进程 search(api)+compile(api)
  - `shell`：兼容路径，调用后端包装脚本

## 代码职责
- `compiler/ir/`：canonicalize 检查 + tagging sidecar 生成
- `compiler/scheduler/`：search 编排 + best/candidate 产物
  - `Search.cpp`：search 分发 + 产物归一化/落盘
  - `Search{BackendShell,BackendProcessChain,BackendApi}.cpp`：
    按 backend-mode 分离的 search 实现
  - `SearchCommon.cpp`：共享 solver 参数拼装
- `compiler/transform/`：compile 编排与变体参数
  - `BuildCutEdgesTransform.cpp`：入口分发 + 产物检查
  - `BuildCutEdgesTransform{Shell,ProcessChain,Api}.cpp`：
    按 backend-mode 分离的 compile 实现
  - `BuildCutEdgesTransformCommon.cpp`：共享编译参数拼装
- `compiler/pipeline/`：端到端阶段编排
- `bench/`：search / A-B profile / verify 脚本
  - `run_all_stages.sh`：一键验收（regression + api pair + pure-api stress + AB）

## 后端边界
本仓库已内置后端源码，位置在 `backend/welder/`：
- solver 二进制目标：`backend/welder/compiler/build/welder-solver`
- solver API 库目标：`backend/welder/compiler/build/libwelder-solver-capi.so`
- compiler 二进制目标：`backend/welder/compiler/build/welder-compiler`
- lowering pass 插件：`backend/welder/mlir_pipeline/workgroup_alloc_to_launch_pass/build/WorkgroupAllocToLaunchPass.so`
- shell 兼容脚本：
  - `backend/welder/compiler/run_welder_solver.sh`
  - `backend/welder/compiler/run_welder_to_nvvm_isa.sh`
- profiler 二进制目标：`backend/welder/compiler/build/welder-profiler`

# 迁移计划（源自 `welder_try`）

## 阶段 0（已完成）
- 新建独立的骨架工程。
- 搭建带明确阶段边界的 `wtc-compiler` 原型。

## 阶段 1（已完成）
- 收敛到单一模式族：generic problem + cut edges。
- 保持仅 typical-chain 的范围。

## 阶段 2（仓内打包完成）
- 将搜索后端源码内置到本仓库 `backend/welder/`。
- `scheduler/` 暴露稳定输出：
  - `best.json`
  - `candidates.tsv`
- `scheduler/` 默认改为 `process_chain` 编排（保留 `shell` 回退，兼容旧名 `inprocess`）。
- 新增 solver `api` 模式：同进程调用后端 `welder-solver-capi` 共享库（无 solver 子进程）。

## 阶段 3（仓内打包完成）
- 将 compile/lowering 后端源码内置到本仓库。
- `transform/` 暴露稳定编译产物和变体控制。
- `transform/` 默认改为 `process_chain` 编排（保留 `shell` 回退，兼容旧名 `inprocess`）。
- 新增 compile/lowering 的 `api` 模式（后端 `welder-compile-capi`）。

## 阶段 4（编排完成）
- `pipeline/` 支持显式模式：
  - `--search-only`
  - `--compile-only`
  - full 默认模式
- tile 来源优先级显式化：
  1. `--best-json`
  2. search 结果
  3. 显式 tile 覆盖
  4. 内置默认值

## 阶段 5（bench 入口完成）
- `bench/run_search.sh` 调用 `wtc-compiler --search-only`。
- `bench/run_ab.sh` 调用 `wtc-compiler` + 内置 profiler。
- 不再转发到历史 bench 脚本。

## 阶段 6（已完成）
- canonicalize 阶段支持仓内 named softmax 分解：
  - 输入含 `linalg.softmax` 时，先分解为 `max/exp/sum/div` generic 链
  - 再进入与已分解输入一致的 tagging/search/compile 流程

## 阶段 7（API compile 稳定性加固完成）
- compile API 改为同进程执行编译阶段（`welderCompilerMain`），不再 shell 调 `welder-compiler`。
- 加固 pure API full 模式的重复运行行为：
  - `api + pure-api-full + repeat>1` 走同进程循环
  - 后端编译器选项解析支持 repeat-safe 重置语义
- full 模式保持“稳定优先”默认策略：
  - 当 `--backend-mode api` 且未设置 `--pure-api-full` 时，search 会在 compile(api) 前回退到 `process_chain`

## 阶段 8（验收自动化完成）
- 新增一键验收脚本：
  - `bench/run_all_stages.sh`
- 覆盖内容：
  - regression 检查
  - API compile pair 流程
  - pure API full 重复压测
  - baseline/fused AB profiling

## 阶段 9（transform 模块化完成）
- 按 backend mode 拆分 `transform` 的 compile 实现：
  - shell 路径
  - process_chain 路径
  - api 路径
- 保留小型分发入口（`BuildCutEdgesTransform.cpp`）用于校验和路由。
- 在不改变运行时行为的前提下，降低单体 transform 编译文件复杂度。

## 阶段 10（scheduler 模块化完成）
- 按 backend mode 拆分 `scheduler` 的 search 实现：
  - shell 路径
  - process_chain 路径
  - api 路径
- 保留小型分发入口（`Search.cpp`）用于产物归一化和路由。
- 在不改变运行时行为的前提下，降低单体 search 文件复杂度。

## 阶段 11（基线护栏完成）
- 新增 baseline 固化脚本：
  - `bench/pin_baseline.sh`
- 新增性能回退护栏脚本（阈值机制）：
  - `bench/check_perf_guard.sh`
- 在验收脚本中新增可选 perf guard 阶段：
  - `CHECK_PERF_GUARD=1 bash bench/run_all_stages.sh`

## 阶段 12（编译器入口瘦身 step 1 完成）
- 为 cut-edges 路径引入 `AnchorDiscovery` 模块：
  - `WelderCompilerAnchorDiscovery.h/.cpp`
- 将拓扑与初始 root 发现逻辑从 `WelderCompilerModeDispatchBody.inc` 拆出：
  - 非平凡 linalg 节点过滤
  - topo 顺序构建
  - topo 索引构建
  - 初始 kernel root 收集
- `WelderCompiler.cpp` / 构建目标接入新模块，行为保持不变。

## 阶段 13（编译器入口瘦身 step 2 完成）
- 引入 `TileDecision` 模块：
  - `WelderCompilerTileDecision.h/.cpp`
- 引入 `FusionPairBuild` 模块：
  - `WelderCompilerFusionPairBuild.h/.cpp`
- 将 thread/block tile 辅助逻辑从 `WelderCompilerModeDispatchBody.inc` 拆出：
  - 可整除线程 tile 回退
  - ceil/exact block-dim 推导
- 将 fusion-pair 构建从 `WelderCompilerModeDispatchBody.inc` 拆出：
  - `codegen-from-kernel-attrs` 线程融合 pair 构建
  - cut-edges 行归约融合 pair 构建
  - cut-edges 寄存器级线程融合 pair 构建
- 后端构建目标（`welder-compiler`、`welder-compile-capi`）接入新模块，运行行为与验收结果不变。

## 阶段 14（编译器入口瘦身 step 3 部分完成）
- 引入 `DispatchPlan` 模块：
  - `WelderCompilerDispatchPlan.h/.cpp`
- 将按 kernel 的 producer 排序逻辑从 `WelderCompilerModeDispatchBody.inc` 拆出：
  - 确定性 sink->source producer 排序（reverse-topo）
  - 回写到 `KernelSpec::orderedProducerNodeIds`
- 保持既有 kernel 分配语义不变。

## 阶段 15（solver 分层 step 1 部分完成）
- 引入 `CandidateGenerator` 模块用于 solver CLI 编排：
  - `WelderSolverCandidateGenerator.h/.cpp`
  - 为 `solveGeneric` / `solve` 提供统一包装和失败提示
- 引入 `ReportWriter` 模块用于 solver 输出：
  - `WelderSolverReportWriter.h/.cpp`
  - best summary JSON 写出
  - candidates TSV 排序写出
- `WelderSolver.cpp` 改为委托新模块执行候选运行与报告输出，降低单体 CLI 复杂度，同时保留 `welder-solver-lib` 中的排序/代价逻辑。

## 阶段 16（solver 分层 step 2 部分完成）
- 引入 `ProfilerRunner` 辅助模块（solver profiling 工具路径设置）：
  - `WelderSolverProfilerRunner.h/.cpp`
  - 集中管理 profiler 二进制/脚本路径解析
- 引入 `CostModel` 输出辅助模块：
  - `WelderSolverCostModel.h/.cpp`
  - 集中管理候选摘要与 top-k 分数打印
- `WelderSolver.cpp` 改为委托 profiler 路径解析与 cost 摘要输出，进一步降低 CLI 入口耦合。
- 核心排名公式和递归调度逻辑保持在 `WelderSolverLib.cpp` 与相关 `.inc` 文件中，不改算法行为。

## 阶段 17（性能护栏稳定性加固完成）
- 加固 `bench/check_perf_guard.sh` 的 speedup 判定策略：
  - 新增 `SPEEDUP_CHECK_MODE=auto|strict|off`（默认 `auto`）
  - 在 `auto` 模式下，当 baseline 运行时间改善超过阈值时跳过 speedup 下限检查，避免分母漂移导致的误报
- 绝对时延检查（`baseline.avg_ms`、`fused.avg_ms`）仍为必选，保持对真实变慢的敏感性。

## 阶段 18（前端 runtime 的 compile(api) 会话化完成）
- 引入 compile API 会话封装：
  - `compiler/include/wtc/transform/internal/CompileApiSession.h`
  - `compiler/transform/CompileApiSession.cpp`
- 重构 `BuildCutEdgesTransformApi.cpp`，将 `dlopen/dlsym` 与调用生命周期委托给 `CompileApiSession`：
  - C API 句柄持久复用
  - 内部 mutex 串行化 compile(api) 调用
  - 减少 transform 入口中的临时静态状态
- 更新构建接线：
  - `compiler/CMakeLists.txt` 新增 `transform/CompileApiSession.cpp`
- 已通过完整验收（含 pure-api 压测与 perf guard）。

## 阶段 19（后端 compile(api) 会话化完成）
- 引入后端 compile session 模块：
  - `backend/welder/compiler/WelderCompileSession.h`
  - `backend/welder/compiler/WelderCompileSession.cpp`
- 将 `WelderCompileCAPI.cpp` 重构为薄适配层：构建 `CompileSessionRequest` 并委托 `CompileSession` 执行。
- session 责任包括：
  - 调用 welder-compiler 阶段
  - 执行 pass 流程（`workgroup`、`linalg-to-loops`、`nvvm`）
  - 每轮构建 MLIRContext/PassManager
  - 统一错误传播
- 更新后端构建接线：
  - `backend/welder/compiler/CMakeLists.txt` 将 `WelderCompileSession.cpp` 纳入 `welder-compile-capi`

## 阶段 20（compiler driver 的 `.inc` 债务削减完成）
- 将 pass-trace/env helper 从 `.inc` 迁移到类型化模块：
  - 新增 `backend/welder/compiler/WelderCompilerPassTraceAndEnv.h`
  - 新增 `backend/welder/compiler/WelderCompilerPassTraceAndEnv.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerPassTraceAndEnv.inc`
- `WelderCompiler.cpp` 接入新模块，保留现有全局开关与调用点行为。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`

## 阶段 21（`.inc` 债务削减：postbufferize hook setup 完成）
- 删除 `backend/welder/compiler/WelderCompilerPostbufferizeHookSetup.inc`。
- 将其两个小型 hook 构造函数内联回 `WelderCompiler.cpp`：
  - `makeRowReductionFixupTogglesFromGlobals`
  - `makePostbufferizeFixupHooks`
- 行为不变：hook 绑定和开关接线保持一致。

## 阶段 22（`.inc` 债务削减：mapping/io helpers 完成）
- 删除 `backend/welder/compiler/WelderCompilerMappingAndIoHelpers.inc`。
- 将 helper 集合内联到 `WelderCompiler.cpp`：
  - 输出写文件（`writeModuleToFile`）
  - GPU block/thread/warp 映射构建
  - 行归约/广播边分类辅助谓词
- 行为不变；仅做代码归位，降低 driver 流程中的 include 碎片化。

## 阶段 23（`.inc` 债务削减：fusion anchors 完成）
- 将 fusion-anchor helper 从 include 文件迁移为模块：
  - 新增 `backend/welder/compiler/WelderCompilerFusionAnchors.h`
  - 新增 `backend/welder/compiler/WelderCompilerFusionAnchors.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerFusionAnchors.inc`
- `WelderCompiler.cpp` 改为使用 `welder::compiler` 命名空间中的类型化符号，替代 include 注入的静态 helper。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`
- 预期无行为变化；本阶段目标是降低 `.inc` 耦合并明确 anchor/tagging 模块边界。

## 阶段 24（`.inc` 债务削减：行归约输入提升完成）
- 将 row-reduction input promotion helper 从 include 文件迁移：
  - 新增 `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.h`
  - 新增 `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerRowReductionInputPromotion.inc`
- 更新 `WelderCompilerMemoryReuseTransforms.inc`，停止 include 注入该 helper，改为显式模块声明。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`
- 预期无行为变化；仅减少 `.inc` 依赖扇出。

## 阶段 25（`.inc` 债务削减：async copy rewrite 完成）
- 将 async copy rewrite helper 从 include 文件迁移：
  - 新增 `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.h`
  - 新增 `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerAsyncCopyRewrite.inc`
- 更新 `WelderCompilerMemoryReuseTransforms.inc`，停止 include 注入 async copy rewrite 逻辑。
- `WelderCompiler.cpp` 改为使用 `welder::compiler` 命名空间的类型化符号。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`
- 预期无行为变化；属于模块边界清理。

## 阶段 26（`.inc` 债务削减：async pipeline helpers 完成）
- 将 async pipeline helper 从 include 文件迁移：
  - 新增 `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.h`
  - 新增 `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerAsyncPipelineHelpers.inc`
- `WelderCompiler.cpp` 改为使用 `welder::compiler` 命名空间符号（`padWorkgroupAllocs`）。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`

## 阶段 27（`.inc` 债务削减：generic transform and fusion 完成）
- 将 generic transform/fusion helper 从 include 文件迁移：
  - 新增 `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.h`
  - 新增 `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.cpp`
  - 删除 `backend/welder/compiler/WelderCompilerGenericTransformAndFusion.inc`
- `WelderCompiler.cpp` 改为使用 `welder::compiler` 命名空间中的类型化符号：
  - `buildGenericTransformLibrary`
  - `KernelSpec` / `RowReductionFusionPair` / `ThreadFusionPair`
  - `normalizeThreadFusionPairs`
- generic 路径调用点改为显式传递 `skipMapForallToBlocks` 开关，而不是读取 include 注入的全局状态。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`

## 阶段 28（compiler 侧 `.inc` 债务清理完成）
- 完成 compiler transform library 模块化迁移：
  - 新增 `backend/welder/compiler/WelderCompilerMatmulTransformLibrary.h/.cpp`
  - 新增 `backend/welder/compiler/WelderCompilerGenericCutEdgesTransformLibrary.h/.cpp`
  - 删除历史 include 文件：
    - `WelderCompilerMatmulTransformLibrary.inc`
    - `WelderCompilerGenericCutEdgesTransformLibrary.inc`
- 将剩余 compiler include 片段从 `.inc` 统一改为 `.h`，保持 include 边界清晰并确保行为一致：
  - `WelderCompilerDriverHelpers.h`
  - `WelderCompilerModeDispatchBody.h`
  - `WelderCompilerMemoryReuseTransforms.h`
  - `WelderCompilerMatmulSoftmaxCanonicalize.h`
  - `WelderCompilerPostprocessCleanup.h`
- 验证后端构建目标通过：
  - `welder-compiler`
  - `welder-compile-capi`
- 验证带 perf guard 的全量验收通过：
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash bench/run_all_stages.sh`

## 阶段 29（后端 include 后缀统一完成）
- 将 `backend/welder/compiler` 下 solver/profiler/compiler 相关剩余 include 片段统一从 `.inc` 改为 `.h`。
- 同步更新源码中的 include 指令。
- 结果：backend compiler 目录不再包含 `*.inc` 文件。
- 重建并验证核心目标：
  - `welder-solver`
  - `welder-compiler`
  - `welder-compile-capi`
  - `welder-pipeline`
- 重新执行带 perf guard 的全量验收，全部 PASS。
- 预期无行为变化；本阶段只降低 include 注入耦合。

## 阶段 30（compiler 阶段边界清理完成）
- 引入显式 mode-dispatch 阶段模块：
  - `backend/welder/compiler/WelderCompilerModeDispatch.h`
  - `backend/welder/compiler/WelderCompilerModeDispatch.cpp`
- 将模式分支编排从 `WelderCompiler.cpp` 移至 `buildTransformLibraryForMode(...)`，保持三条分支行为稳定：
  - `codegen-from-kernel-attrs`
  - `generic problem`
  - `matmul path`
- 将 driver helper 提升为类型化模块对：
  - `backend/welder/compiler/WelderCompilerDriverHelpers.h`
  - `backend/welder/compiler/WelderCompilerDriverHelpers.cpp`
- 将大段 CLI 选项声明从主编排文件拆出：
  - `backend/welder/compiler/WelderCompilerCliOptions.h`
- `WelderCompiler.cpp` 聚焦阶段编排：
  1) CLI 解析 + trace 初始化
  2) dialect 注册 + 输入解析
  3) solve 选项构建 + 行归约开关推导
  4) mode dispatch
  5) transform library 装载
  6) lowering pipeline 执行
- 后端构建接线更新，两个目标均通过：
  - `welder-compiler`
  - `welder-compile-capi`
- 带 perf guard 的全量验收通过：
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash bench/run_all_stages.sh`
  - 所有阶段 PASS（`ab_speedup` 约 `1.1751x`）。

## 阶段 31（mode-dispatch 分支拆分 + pipeline 命名收敛完成）
- ModeDispatch 从单体 body 拆为“薄路由 + 三分支实现”：
  - `WelderCompilerModeDispatch.cpp` 仅保留路由
  - `WelderCompilerModeDispatchKernelAttrsBranch.cpp`
  - `WelderCompilerModeDispatchGenericBranch.cpp`
  - `WelderCompilerModeDispatchMatmulBranch.cpp`
  - `WelderCompilerModeDispatchBody.h` 已删除
- 新增 `CompilerRunConfig`（parse+validate 专用模块）：
  - `WelderCompilerRunConfig.h/.cpp`
  - 主流程改为“配置对象校验 + `buildModeDispatchContext(...)`”
- 前端命名与阶段契约收敛：
  - backend mode 默认名升级为 `process_chain`（兼容旧输入 `inprocess`）
  - runtime 后处理函数命名改为 `validatePostbufferizeArtifacts(...)`
  - `PipelineRunner` 引入 `StageArtifacts` 统一阶段 I/O 路径
- 验证结果：
  - `cmake --build /home/zhangruiqi/welder_typical_chain/build -j4`
  - `cmake --build /home/zhangruiqi/welder_typical_chain/backend/welder/compiler/build -j4 --target welder-compiler welder-compile-capi`
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash /home/zhangruiqi/welder_typical_chain/bench/run_all_stages.sh` 全 PASS

## 阶段 32（postprocess 巨型头文件拆分加固完成）
- 完成三大 postprocess 块拆分为公开 API 头 + 实现 `.cpp` 包装：
  - `WelderCompilerMemoryReuseTransforms.h/.cpp`
  - `WelderCompilerPostprocessCleanup.h/.cpp`
  - `WelderCompilerMatmulSoftmaxCanonicalize.h/.cpp`
- 为保证行为一致，重逻辑保留在内部 impl 头：
  - `WelderCompilerMemoryReuseTransformsImpl.h`
  - `WelderCompilerPostprocessCleanupImpl.h`
  - `WelderCompilerMatmulSoftmaxCanonicalizeImpl.h`
- 按子阶段继续拆分内部实现（可读性优先）：
  - `WelderCompilerMemoryReuseTransformsStage{1,2,3}.h`
  - `WelderCompilerPostprocessCleanupStage{1,2,3,4}.h`
  - `WelderCompilerMatmulSoftmaxCanonicalizeStage{1,2,3}.h`
- 通过在所属模块本地化 helper（如 `stripToBaseMemref`、`fixSubviewResultTypes...`、`isUsedInLaunch`）并显式声明 dialect include，移除跨模块隐式符号耦合。
- 后端构建接线更新，两个目标均编译新模块：
  - `welder-compiler`
  - `welder-compile-capi`
- 验证：
  - `cmake --build /home/zhangruiqi/welder_typical_chain/backend/welder/compiler/build -j4 --target welder-compiler welder-compile-capi` PASS
  - `cmake --build /home/zhangruiqi/welder_typical_chain/build -j4` PASS
  - `CHECK_PERF_GUARD=1 MAX_REGRESSION_PCT=3 bash /home/zhangruiqi/welder_typical_chain/bench/run_all_stages.sh` PASS
  - 最新测得 `ab_speedup=1.1753x`

## 验收清单
1. `wtc-compiler` 可在 typical-chain MLIR 上运行 search-only 与 compile-only。
2. `bench/run_search.sh` 产出 `best.json` 与 `candidates.tsv`。
3. `bench/run_ab.sh` 产出 `ab_summary.tsv` 与 `speedup.tsv`。
4. 代码路径保持单一路径、可清晰讲解。

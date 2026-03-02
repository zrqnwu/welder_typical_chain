# LLVM/MLIR Bug 报告材料（可直接提交）

这个目录整理了“`memref.dealloc` 作用在非 0 地址空间（例如 `#gpu.address_space<workgroup>`）时，NVVM/LLVM lowering 报 `free(ptr)` 指针地址空间不匹配”的最小复现与提 bug 所需信息。

## 文件说明

- `repro.mlir`：最小复现 IR（已确保 alloc 不会被 DCE 掉）。
- `repro.sh`：一键复现脚本（打印版本信息 + 运行命令 + 展示报错）。
- `issue.md`：可直接复制粘贴到 `llvm/llvm-project` GitHub issue 的英文描述（含环境信息、复现步骤、期望/实际、根因定位、建议修复）。
- `fix.md`：上游修复的推荐流程（改哪里 + 回归测试建议 + 本地验证命令）。

## 快速复现

在仓库根目录执行：

```bash
bash mlir_pipeline/llvm_bug_report/repro.sh
```

## 标准提 bug 流程（建议按这个顺序）

1. **确认是 tip-of-tree 问题**：尽量用最新 `llvm/llvm-project`（或给出你当前的 commit，如 `issue.md` 里那样）。
2. **做最小复现**：从大 IR 逐步删减，直到只剩“触发错误的最小模式”（本目录的 `repro.mlir` 就是最终最小模式）。
3. **固定复现命令**：给出“一条命令就能跑出同样报错”的指令（`repro.sh` / `issue.md` 已给出）。
4. **收集环境信息**：`mlir-opt --version`、`llvm-project` commit、操作系统版本（必要时再加 GPU 架构参数）。
5. **写清楚期望 vs 实际**：这类问题最关键的是“为什么这是不对的”（例如：lowering 生成了不合法的 LLVM dialect IR）。
6. **提交到 GitHub**：到 `https://github.com/llvm/llvm-project/issues/new/choose` 选择 MLIR 相关模板；把 `issue.md` 内容贴进去即可（必要时把 `repro.mlir` 作为附件或代码块）。
7. **（可选）附上根因定位/修复建议**：能显著提高被快速修复的概率（`fix.md` 有建议）。

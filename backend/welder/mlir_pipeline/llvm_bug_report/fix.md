# 上游修复流程（建议）

## 1) 定位与修改点

- 代码位置：`llvm-project/mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp`
- 目标函数：`DeallocOpLowering::matchAndRewrite`

建议最小修复：在调用 `@free` 前，把 `allocatedPtr` `addrspacecast` 回 `!llvm.ptr`（addrspace(0)）。

逻辑上与 `AllocOpLowering` 里的 `castAllocFuncResult` 对称：alloc 时从 `malloc(!llvm.ptr)` cast 到 memref addrspace，dealloc 时从 memref addrspace cast 回 `!llvm.ptr` 再 free。

## 2) 回归测试建议

建议新增一个 `MemRefToLLVM` 的单测，覆盖非 0 memory space 的 `memref.dealloc`：

- 位置建议：`llvm-project/mlir/test/Conversion/MemRefToLLVM/dealloc-nonzero-addrspace.mlir`
- 思路：
  - 输入里构造 `memref<... , 3>` 或 `memref<..., #gpu.address_space<workgroup>>`
  - 跑 `-finalize-memref-to-llvm`
  - `FileCheck` 验证生成了 `llvm.addrspacecast`，并且 `llvm.call @free` 的实参类型是 `!llvm.ptr`

（用 `memref<..., 3>` 可以避免依赖 `gpu` dialect 的属性解析；用 `#gpu.address_space<workgroup>` 更贴近实际场景，二选一即可。）

## 3) 本地验证命令（示例）

```bash
# 复现（应当失败 -> 修复后应当通过）
bash mlir_pipeline/llvm_bug_report/repro.sh

# 跑单测（示例）
llvm-project/build/bin/llvm-lit -sv \
  llvm-project/mlir/test/Conversion/MemRefToLLVM/dealloc-nonzero-addrspace.mlir

# 或者更全量（较慢）
ninja -C llvm-project/build check-mlir
```

## 4) 提 PR 流程（简版）

1. fork `llvm/llvm-project`，拉分支。
2. 提交修复 + 新增回归测试。
3. 本地跑对应 lit 测试（至少跑新增文件；最好再跑相关 suite）。
4. GitHub 发 PR，component 选 MLIR，描述里附上 issue 链接与复现文件。


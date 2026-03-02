# workgroup `memref.dealloc` → `free(ptr)` addrspace mismatch（打包）

这个目录把“报错的 IR 文件”和“复现步骤/脚本”集中放在一起，方便直接打包发给别人或贴到 LLVM issue。

## 目录结构

- `inputs/`
  - `matmul.mlir`：最初输入（linalg/tensor）。
  - `transform.mlir`：transform 脚本（tiling + promote 到 workgroup + 映射到 gpu.launch）。
- `artifacts/`
  - `matmul.bufferized.tiled.promoted.gpu.loops.mlir`：复现链路生成的、会触发报错的 IR（包含 workgroup `memref.dealloc`）。
  - `nvvm_lower_err.log`：`--gpu-lower-to-nvvm-pipeline` 的报错输出（快照）。
- `minimal/`
  - `repro.mlir`：最小复现（直接构造 workgroup alloc + dealloc，并确保不会被 DCE 掉）。
- `scripts/`
  - `repro_full.sh`：从 `inputs/` 生成 `artifacts/`，并跑 NVVM pipeline 复现报错。
  - `repro_minimal.sh`：跑 `minimal/repro.mlir` 复现报错。

## 如何复现

需要你本机有 `mlir-opt`（带 NVVM/GPU 支持）。默认路径假设是 `llvm-project/build/bin`。

```bash
# 全链路复现（会生成 artifacts 并报错）
bash mlir_pipeline/workgroup_dealloc_bug_pack/scripts/repro_full.sh

# 最小复现（直接报错）
bash mlir_pipeline/workgroup_dealloc_bug_pack/scripts/repro_minimal.sh

# workaround：删除 workgroup 上的 memref.dealloc 后再跑 NVVM（应当能成功）
bash mlir_pipeline/workgroup_dealloc_bug_pack/scripts/repro_workaround_drop_workgroup_dealloc.sh
```

可选参数：

- `BIN=...`：指定 `mlir-opt` 所在目录（例如 `BIN=/path/to/llvm-project/build/bin`）
- `CHIP=sm_86`：指定 NVVM 目标架构

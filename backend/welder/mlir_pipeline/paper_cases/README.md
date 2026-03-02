# Paper / Model-like cases (for tuning & perf validation)

这些 `.mlir` 用例是为了更贴近模型里的常见算子组合（GEMM epilogue、MLP block、Conv block），
方便用 **Welder paper-style** 的 “Enumerate → Prune → Profile → Pick best” 闭环做调优验证。

## 建议怎么跑（调优闭环）

对任意用例：

```bash
cd /home/zhangruiqi/welder_try

# paper-style profile search：输出 candidates.tsv + profile_cache.tsv
OUT_DIR=/tmp/welder_case_out \
  bash bench/run_paper_profile_search.sh mlir_pipeline/paper_cases/<case>.mlir \
    --schedule-topk 2 \
    --profile-warmup 50 \
    --profile-iters 200 \
    --prune-on-profile-failure
```

提示：
- MatMul 系列用例默认走 `matmul` solver 路径，不需要额外参数。
- Conv / bottleneck 系列用例需要加 `--enable-generic-problem`（走 generic solver 路径）。

## 用例列表（常见组合）

### GEMM / MatMul（Transformer/MLP 主力）
- `matmul_f32_m1024_n1024_k1024.mlir`：MatMul
- `matmul_relu_f32_m1024_n1024_k1024.mlir`：MatMul + ReLU（典型 epilogue）
- `matmul_bias_relu_f32_m1024_n1024_k1024.mlir`：MatMul + Bias(Add) + ReLU（典型 MLP / FC 层）
- `matmul_residual_relu_f32_m1024_n1024_k1024.mlir`：MatMul + Residual(Add) + ReLU（残差块常见）
- `matmul_f32_m256_n4096_k1024.mlir`：MatMul（FC1-like shape）
- `matmul_relu_f32_m256_n4096_k1024.mlir`：MatMul + ReLU（FC1 epilogue）
- `matmul_bias_relu_f32_m256_n4096_k1024.mlir`：MatMul + Bias(Add) + ReLU（FC1 epilogue）
- `matmul_f32_m256_n1024_k4096.mlir`：MatMul（FC2-like shape）
- `matmul_relu_f32_m256_n1024_k4096.mlir`：MatMul + ReLU（FC2 epilogue）
- `matmul_bias_relu_f32_m256_n1024_k4096.mlir`：MatMul + Bias(Add) + ReLU（FC2 epilogue）

### MLP block（FFN 的骨架）
- `mlp_2layer_f32_b256_h1024_ff4096.mlir`：`[B,H]x[H,4H] -> ReLU -> [B,4H]x[4H,H]`

### Conv block（VGG/ResNet 常见）
- `conv3x3_relu_f32_n1_h34_w34_c64_f64.mlir`：Conv3x3 + ReLU（generic conv）
- `conv3x3_bias_relu_f32_n1_h34_w34_c64_f64.mlir`：Conv3x3 + Bias(Add) + ReLU
- `bottleneck_1x1_relu_3x3_f32_n1_h34_w34_c64_f64_f64.mlir`：1x1 Conv + ReLU + 3x3 Conv（ResNet bottleneck 常见片段）

// Matmul(f16->f16) -> Softmax(f32) 链路（论文 Figure 2 风格），用于验证：
// - 算子间 shared tile 复用（Matmul -> Softmax）
// - Matmul 部分的 TensorCore（mma.sync）f16 路径
//
// 注意：本仓库的 TensorCore matmul lowering 依赖
// `transform.nvgpu.rewrite_matmul_as_mma_sync`。在当前 MLIR 版本里，
// f16 累加/输出变体比 f32 累加变体更稳定，所以 Matmul 先产出 f16，
// 再在 Softmax 前转成 f32，保证数值稳定性。
//
// 为了兼容性能测试链路（welder-profiler --init-ptr），输入 A/B 仍是 f32，
// 但在 Matmul 前会截断成 f16，以启用 --enable-tensorcore-f16。
module {
  func.func @main(%A: tensor<8192x64xf32>, %B: tensor<64x128xf32>) -> tensor<8192x128xf32> {
    %f0 = arith.constant 0.0 : f32
    %f0_f16 = arith.constant 0.0 : f16
    %neg_inf = arith.constant -3.4028235e+38 : f32

    // A/B 从 f32 转成 f16，用于 TensorCore matmul。
    %a16_0 = tensor.empty() : tensor<8192x64xf16>
    %A16 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : tensor<8192x64xf32>) outs(%a16_0 : tensor<8192x64xf16>) {
    ^bb0(%x: f32, %out: f16):
      %y = arith.truncf %x : f32 to f16
      linalg.yield %y : f16
    } -> tensor<8192x64xf16>

    %b16_0 = tensor.empty() : tensor<64x128xf16>
    %B16 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%B : tensor<64x128xf32>) outs(%b16_0 : tensor<64x128xf16>) {
    ^bb0(%x: f32, %out: f16):
      %y = arith.truncf %x : f32 to f16
      linalg.yield %y : f16
    } -> tensor<64x128xf16>

    // Matmul 初始化：C16[M,N] = 0（f16 累加，对应 TensorCore mma.sync）
    %c0 = tensor.empty() : tensor<8192x128xf16>
    %c_init = linalg.fill ins(%f0_f16 : f16) outs(%c0 : tensor<8192x128xf16>) -> tensor<8192x128xf16>
    %C16 = linalg.matmul
      ins(%A16, %B16 : tensor<8192x64xf16>, tensor<64x128xf16>)
      outs(%c_init : tensor<8192x128xf16>) -> tensor<8192x128xf16>

    // Matmul 结果从 f16 转成 f32，便于 Softmax 稳定计算。
    %c32_0 = tensor.empty() : tensor<8192x128xf32>
    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C16 : tensor<8192x128xf16>) outs(%c32_0 : tensor<8192x128xf32>) {
    ^bb0(%x: f16, %out: f32):
      %y = arith.extf %x : f16 to f32
      linalg.yield %y : f32
    } -> tensor<8192x128xf32>

    // max 初始化：max[M]
    %max0 = tensor.empty() : tensor<8192xf32>
    %max_init = linalg.fill ins(%neg_inf : f32) outs(%max0 : tensor<8192xf32>) -> tensor<8192xf32>
    %max = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%C : tensor<8192x128xf32>) outs(%max_init : tensor<8192xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<8192xf32>

    // 公式：exp_sub[M,N] = exp(C - max)
    %exp0 = tensor.empty() : tensor<8192x128xf32>
    %exp_sub = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // C
        affine_map<(d0, d1) -> (d0)>,     // max (broadcast)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C, %max : tensor<8192x128xf32>, tensor<8192xf32>) outs(%exp0 : tensor<8192x128xf32>) {
    ^bb0(%x: f32, %m: f32, %out: f32):
      %d = arith.subf %x, %m : f32
      %e = math.exp %d : f32
      linalg.yield %e : f32
    } -> tensor<8192x128xf32>

    // sum 初始化：sum[M]
    %sum0 = tensor.empty() : tensor<8192xf32>
    %sum_init = linalg.fill ins(%f0 : f32) outs(%sum0 : tensor<8192xf32>) -> tensor<8192xf32>
    %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%exp_sub : tensor<8192x128xf32>) outs(%sum_init : tensor<8192xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<8192xf32>

    // 公式：out[M,N] = exp_sub / sum（broadcast）
    %out0 = tensor.empty() : tensor<8192x128xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // exp_sub
        affine_map<(d0, d1) -> (d0)>,     // sum (broadcast)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%exp_sub, %sum : tensor<8192x128xf32>, tensor<8192xf32>) outs(%out0 : tensor<8192x128xf32>) {
    ^bb0(%e: f32, %s: f32, %out: f32):
      %y = arith.divf %e, %s : f32
      linalg.yield %y : f32
    } -> tensor<8192x128xf32>

    return %out : tensor<8192x128xf32>
  }
}

// 原生 f16 输入的 Matmul(f16->f16) -> Softmax(f32) 链路。
//
// 相比 `matmul_softmax_chain_f16_matmul.mlir`，该版本去掉了
// kernel 内 A/B 的 f32->f16 截断，从而在更接近 attention 的场景中评估 TensorCore
// 路径（训练/推理输入通常本来就是 f16）。
module {
  func.func @main(%A: tensor<8192x64xf16>, %B: tensor<64x128xf16>) -> tensor<8192x128xf32> {
    %f0 = arith.constant 0.0 : f32
    %f0_f16 = arith.constant 0.0 : f16
    %neg_inf = arith.constant -3.4028235e+38 : f32

    // Matmul 初始化：C16[M,N] = 0（mma.sync 的 f16 累加/输出路径）。
    %c0 = tensor.empty() : tensor<8192x128xf16>
    %c_init = linalg.fill ins(%f0_f16 : f16) outs(%c0 : tensor<8192x128xf16>) -> tensor<8192x128xf16>
    %C16 = linalg.matmul
      ins(%A, %B : tensor<8192x64xf16>, tensor<64x128xf16>)
      outs(%c_init : tensor<8192x128xf16>) -> tensor<8192x128xf16>

    // Matmul 结果 f16 -> f32，提升 Softmax 数值稳定性。
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
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
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
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
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

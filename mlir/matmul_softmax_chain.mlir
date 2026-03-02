// Matmul->Softmax 链路（论文 Figure 2 风格），用于验证算子间 shared 复用。
//
// 结构（全静态形状）：
// 链路：A[M,K] * B[K,N] -> C[M,N] -> softmax_row(C) -> D[M,N]
//
// 这是一个最小化的“Matmul + Softmax”相邻算子模式：
// 当 tile 配置对齐时，中间结果可在 shared memory 中复用，
// 而不是把 C 物化到 global 内存。
module {
  func.func @main(%A: tensor<8192x64xf32>, %B: tensor<64x128xf32>) -> tensor<8192x128xf32> {
    %f0 = arith.constant 0.0 : f32
    %neg_inf = arith.constant -3.4028235e+38 : f32

    // Matmul 初始化：C[M,N] = 0
    %c0 = tensor.empty() : tensor<8192x128xf32>
    %c_init = linalg.fill ins(%f0 : f32) outs(%c0 : tensor<8192x128xf32>) -> tensor<8192x128xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<8192x64xf32>, tensor<64x128xf32>)
      outs(%c_init : tensor<8192x128xf32>) -> tensor<8192x128xf32>

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

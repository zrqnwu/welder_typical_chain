// Softmax 行归约链路，用于验证“归约 -> 归约”融合与复用。
//
// 结构（全静态形状）：
// 链路：X[B,N] -> max[B] -> exp_sub[B,N] -> sum[B] -> out[B,N]
//
// 该用例是“论文风格”的按行归约链路，类似 LayerNorm：
// 输入 tile 可以一次搬入 shared 内存，并在多个归约/逐元素阶段复用。
module {
  func.func @main(%X: tensor<32x256xf32>) -> tensor<32x256xf32> {
    %f0 = arith.constant 0.0 : f32
    %neg_inf = arith.constant -3.4028235e+38 : f32

    // max 初始化：max[B]
    %max0 = tensor.empty() : tensor<32xf32>
    %max_init = linalg.fill ins(%neg_inf : f32) outs(%max0 : tensor<32xf32>) -> tensor<32xf32>
    %max = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%X : tensor<32x256xf32>) outs(%max_init : tensor<32xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<32xf32>

    // 公式：exp_sub[B,N] = exp(X - max)
    %exp0 = tensor.empty() : tensor<32x256xf32>
    %exp_sub = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // X
        affine_map<(d0, d1) -> (d0)>,     // max (broadcast)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%X, %max : tensor<32x256xf32>, tensor<32xf32>) outs(%exp0 : tensor<32x256xf32>) {
    ^bb0(%x: f32, %m: f32, %out: f32):
      %d = arith.subf %x, %m : f32
      %e = math.exp %d : f32
      linalg.yield %e : f32
    } -> tensor<32x256xf32>

    // sum 初始化：sum[B]
    %sum0 = tensor.empty() : tensor<32xf32>
    %sum_init = linalg.fill ins(%f0 : f32) outs(%sum0 : tensor<32xf32>) -> tensor<32xf32>
    %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%exp_sub : tensor<32x256xf32>) outs(%sum_init : tensor<32xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<32xf32>

    // 公式：out[B,N] = exp_sub / sum（broadcast）
    %out0 = tensor.empty() : tensor<32x256xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // exp_sub
        affine_map<(d0, d1) -> (d0)>,     // sum (broadcast)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%exp_sub, %sum : tensor<32x256xf32>, tensor<32xf32>) outs(%out0 : tensor<32x256xf32>) {
    ^bb0(%e: f32, %s: f32, %out: f32):
      %y = arith.divf %e, %s : f32
      linalg.yield %y : f32
    } -> tensor<32x256xf32>

    return %out : tensor<32x256xf32>
  }
}

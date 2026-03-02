// Transpose + 逐元素 + 归约链路，用于压测 Propagate v2
// 在生产者 output map 置换场景下（indexing_map 驱动）的行为。
//
// 结构（静态形状）：
// 链路：A[64,128] -> T[128,64]（通过 output indexing_map 转置）-> relu[128,64]
//            -> sum[128]   (row-wise 归约 over the last dim)
module {
  func.func @main(%A: tensor<64x128xf32>) -> tensor<128xf32> {
    %f0 = arith.constant 0.0 : f32

    // T[128,64] = transpose(A)（output indexing_map 为 (d1,d0)）。
    %t0 = tensor.empty() : tensor<128x64xf32>
    %T = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // A
        affine_map<(d0, d1) -> (d1, d0)>  // out (transpose)
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : tensor<64x128xf32>) outs(%t0 : tensor<128x64xf32>) {
    ^bb0(%a: f32, %out: f32):
      linalg.yield %a : f32
    } -> tensor<128x64xf32>

    // 公式：relu[128,64] = max(T, 0)
    %relu0 = tensor.empty() : tensor<128x64xf32>
    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // in
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%T : tensor<128x64xf32>) outs(%relu0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.maximumf %in, %f0 : f32
      linalg.yield %r : f32
    } -> tensor<128x64xf32>

    // 公式：sum[128] = reduce(relu, dim=1)
    %sum0 = tensor.empty() : tensor<128xf32>
    %sum_init = linalg.fill ins(%f0 : f32) outs(%sum0 : tensor<128xf32>) -> tensor<128xf32>
    %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // relu
        affine_map<(d0, d1) -> (d0)>      // out
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%relu : tensor<128x64xf32>) outs(%sum_init : tensor<128xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>

    return %sum : tensor<128xf32>
  }
}

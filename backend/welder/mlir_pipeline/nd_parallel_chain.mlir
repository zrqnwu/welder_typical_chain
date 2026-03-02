// ND 并行逐元素链路，用于验证 ND EnumerateSubtiles。
module {
  func.func @main(%A: tensor<8x16x32xf32>, %B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
    %out0 = tensor.empty() : tensor<8x16x32xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%A, %B : tensor<8x16x32xf32>, tensor<8x16x32xf32>) outs(%out0 : tensor<8x16x32xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %r = arith.addf %a, %b : f32
      linalg.yield %r : f32
    } -> tensor<8x16x32xf32>
    return %out : tensor<8x16x32xf32>
  }
}

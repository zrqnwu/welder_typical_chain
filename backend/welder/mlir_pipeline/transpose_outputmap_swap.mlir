// 一个最小化 linalg.generic：用置换后的 output indexing_map 写结果。
// 用于验证 Welder 的 block 顺序（x/y 交换）推断：
// output map `(d0, d1) -> (d1, d0)` 会让最后一维输出依赖 d0。

module {
  func.func @main(%arg0: tensor<128x64xf32>) -> tensor<64x128xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x128xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>

    %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1, d0)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<128x64xf32>) outs(%filled : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x128xf32>

    return %0 : tensor<64x128xf32>
  }
}

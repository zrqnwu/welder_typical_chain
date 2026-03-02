// 用 linalg.generic 表达的 Conv3x3 + Bias(Add) + ReLU（f32）。
//
// 说明：
// - Bias 这里建模为完整输出形状 [N,OH,OW,F]（不做 broadcast）。
//
// 形状：
// 输入 A：1x34x34x64
// 卷积核 B：3x3x64x64
// 偏置 Bias：1x32x32x64
// 输出 C：1x32x32x64
module {
  func.func @main(
      %A: tensor<1x34x34x64xf32>,
      %B: tensor<3x3x64x64xf32>,
      %Bias: tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32> {
    %c0f = arith.constant 0.0 : f32

    // 卷积累加。
    %C0 = tensor.empty() : tensor<1x32x32x64xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>

    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0 + d5, d1 + d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d4, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]
    } ins(%A, %B : tensor<1x34x34x64xf32>, tensor<3x3x64x64xf32>)
      outs(%Cinit : tensor<1x32x32x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<1x32x32x64xf32>

    // 偏置相加：C + Bias
    %D0 = tensor.empty() : tensor<1x32x32x64xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>, // conv
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>, // bias
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>  // output
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%C, %Bias : tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>)
      outs(%Dinit : tensor<1x32x32x64xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %y = arith.addf %x, %b : f32
      linalg.yield %y : f32
    } -> tensor<1x32x32x64xf32>

    // 激活：ReLU
    %E0 = tensor.empty() : tensor<1x32x32x64xf32>
    %Einit = linalg.fill ins(%c0f : f32) outs(%E0 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>
    %E = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%D : tensor<1x32x32x64xf32>) outs(%Einit : tensor<1x32x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<1x32x32x64xf32>

    return %E : tensor<1x32x32x64xf32>
  }
}

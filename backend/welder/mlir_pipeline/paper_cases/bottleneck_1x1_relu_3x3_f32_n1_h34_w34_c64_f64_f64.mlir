// 类 ResNet 的 bottleneck 片段（f32）：1x1 Conv + ReLU + 3x3 Conv
//
// 选择这个用例的原因：
// - 在 CNN 主干网络中非常常见（通道混合 + 空间卷积）。
// - First conv is 1x1 (only 1 归约 loop), second is 3x3 (3 归约),
//   因此 generic solver 的“优先选择归约环最多的算子”启发式会自然落到 3x3 conv，
//   更适合验证调度效果。
//
// 形状：
// 输入 A：1x34x34x64
// 卷积核 B1：64x64（1x1 conv 视作 [C,F]）
// conv1 输出：1x34x34x64
// 卷积核 B2：3x3x64x64
// conv2 输出：1x32x32x64
module {
  func.func @main(
      %A: tensor<1x34x34x64xf32>,
      %B1: tensor<64x64xf32>,
      %B2: tensor<3x3x64x64xf32>) -> tensor<1x32x32x64xf32> {
    %c0f = arith.constant 0.0 : f32

    // 1x1 卷积：34->34
    // loops: (oh, ow, f, n) parallel; (c) 归约
    %C10 = tensor.empty() : tensor<1x34x34x64xf32>
    %C1init = linalg.fill ins(%c0f : f32) outs(%C10 : tensor<1x34x34x64xf32>)
        -> tensor<1x34x34x64xf32>
    %C1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d4)>, // A[n,oh,ow,c]
        affine_map<(d0, d1, d2, d3, d4) -> (d4, d2)>,         // B1[c,f]
        affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d2)>  // out[n,oh,ow,f]
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%A, %B1 : tensor<1x34x34x64xf32>, tensor<64x64xf32>)
      outs(%C1init : tensor<1x34x34x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<1x34x34x64xf32>

    // 激活：ReLU
    %R0 = tensor.empty() : tensor<1x34x34x64xf32>
    %Rinit = linalg.fill ins(%c0f : f32) outs(%R0 : tensor<1x34x34x64xf32>)
        -> tensor<1x34x34x64xf32>
    %R = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%C1 : tensor<1x34x34x64xf32>) outs(%Rinit : tensor<1x34x34x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<1x34x34x64xf32>

    // 3x3 卷积：34->32
    %C20 = tensor.empty() : tensor<1x32x32x64xf32>
    %C2init = linalg.fill ins(%c0f : f32) outs(%C20 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>
    %C2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0 + d5, d1 + d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d4, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]
    } ins(%R, %B2 : tensor<1x34x34x64xf32>, tensor<3x3x64x64xf32>)
      outs(%C2init : tensor<1x32x32x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<1x32x32x64xf32>

    return %C2 : tensor<1x32x32x64xf32>
  }
}

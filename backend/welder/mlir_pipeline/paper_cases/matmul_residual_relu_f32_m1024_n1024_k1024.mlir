// 用例：MatMul + Residual(Add) + ReLU（f32）
//
// 说明：
// - Residual 建模为完整 [M,N] 张量（与 matmul 输出同形状）。
// - 用于近似 Transformer/ResNet 一类模型中的残差块。
//
// 形状：(M,N,K) = (1024,1024,1024)
module {
  func.func @main(
      %A: tensor<1024x1024xf32>,
      %B: tensor<1024x1024xf32>,
      %R: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<1024x1024xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<1024x1024xf32>)
        -> tensor<1024x1024xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%Cinit : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %D0 = tensor.empty() : tensor<1024x1024xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<1024x1024xf32>)
        -> tensor<1024x1024xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // matmul result
        affine_map<(d0, d1) -> (d0, d1)>, // residual
        affine_map<(d0, d1) -> (d0, d1)>  // output
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C, %R : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%Dinit : tensor<1024x1024xf32>) {
    ^bb0(%x: f32, %r: f32, %out: f32):
      %y = arith.addf %x, %r : f32
      linalg.yield %y : f32
    } -> tensor<1024x1024xf32>

    %E0 = tensor.empty() : tensor<1024x1024xf32>
    %Einit = linalg.fill ins(%c0f : f32) outs(%E0 : tensor<1024x1024xf32>)
        -> tensor<1024x1024xf32>
    %E = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<1024x1024xf32>) outs(%Einit : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<1024x1024xf32>

    return %E : tensor<1024x1024xf32>
  }
}

// 用例：MatMul + Bias(Add) + ReLU（f32）- Transformer/MLP FC1 epilogue
//
// 说明：
// - 为简化起见，Bias 建模为完整 [M,N] 张量（不做 broadcast）。
//
// 形状：(M,N,K) = (256,4096,1024)
module {
  func.func @main(
      %A: tensor<256x1024xf32>,
      %B: tensor<1024x4096xf32>,
      %Bias: tensor<256x4096xf32>) -> tensor<256x4096xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<256x4096xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<256x1024xf32>, tensor<1024x4096xf32>)
      outs(%Cinit : tensor<256x4096xf32>) -> tensor<256x4096xf32>

    %D0 = tensor.empty() : tensor<256x4096xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // matmul result
        affine_map<(d0, d1) -> (d0, d1)>, // bias
        affine_map<(d0, d1) -> (d0, d1)>  // output
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C, %Bias : tensor<256x4096xf32>, tensor<256x4096xf32>)
      outs(%Dinit : tensor<256x4096xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %y = arith.addf %x, %b : f32
      linalg.yield %y : f32
    } -> tensor<256x4096xf32>

    %E0 = tensor.empty() : tensor<256x4096xf32>
    %Einit = linalg.fill ins(%c0f : f32) outs(%E0 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %E = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<256x4096xf32>) outs(%Einit : tensor<256x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<256x4096xf32>

    return %E : tensor<256x4096xf32>
  }
}

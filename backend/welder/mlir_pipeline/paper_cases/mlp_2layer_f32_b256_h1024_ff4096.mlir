// 两层 MLP 模块（f32）
//
// 结构：
// 公式：Y1 = ReLU(X @ W1)
// 公式：Y2 = Y1 @ W2
//
// 形状（类似 Transformer FFN）：
// 张量：X:  [B, H]  = [256, 1024]
// 张量：W1: [H, 4H] = [1024, 4096]
// 张量：W2: [4H, H] = [4096, 1024]
module {
  func.func @main(
      %X: tensor<256x1024xf32>,
      %W1: tensor<1024x4096xf32>,
      %W2: tensor<4096x1024xf32>) -> tensor<256x1024xf32> {
    %c0f = arith.constant 0.0 : f32

    // 阶段 1：MatMul [256x1024] x [1024x4096] -> [256x4096]
    %Y10 = tensor.empty() : tensor<256x4096xf32>
    %Y1init = linalg.fill ins(%c0f : f32) outs(%Y10 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %Y1 = linalg.matmul
      ins(%X, %W1 : tensor<256x1024xf32>, tensor<1024x4096xf32>)
      outs(%Y1init : tensor<256x4096xf32>) -> tensor<256x4096xf32>

    // 激活：ReLU
    %A0 = tensor.empty() : tensor<256x4096xf32>
    %Ainit = linalg.fill ins(%c0f : f32) outs(%A0 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %A = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%Y1 : tensor<256x4096xf32>) outs(%Ainit : tensor<256x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<256x4096xf32>

    // 阶段 2：MatMul [256x4096] x [4096x1024] -> [256x1024]
    %Y20 = tensor.empty() : tensor<256x1024xf32>
    %Y2init = linalg.fill ins(%c0f : f32) outs(%Y20 : tensor<256x1024xf32>)
        -> tensor<256x1024xf32>
    %Y2 = linalg.matmul
      ins(%A, %W2 : tensor<256x4096xf32>, tensor<4096x1024xf32>)
      outs(%Y2init : tensor<256x1024xf32>) -> tensor<256x1024xf32>

    return %Y2 : tensor<256x1024xf32>
  }
}

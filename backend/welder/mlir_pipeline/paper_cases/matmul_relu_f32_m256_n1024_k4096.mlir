// 用例：MatMul + ReLU（f32）- Transformer/MLP FC2 epilogue
//
// 形状：(M,N,K) = (256,1024,4096)
module {
  func.func @main(%A: tensor<256x4096xf32>, %B: tensor<4096x1024xf32>)
      -> tensor<256x1024xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<256x1024xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<256x1024xf32>)
        -> tensor<256x1024xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<256x4096xf32>, tensor<4096x1024xf32>)
      outs(%Cinit : tensor<256x1024xf32>) -> tensor<256x1024xf32>

    %D0 = tensor.empty() : tensor<256x1024xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<256x1024xf32>)
        -> tensor<256x1024xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C : tensor<256x1024xf32>) outs(%Dinit : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<256x1024xf32>

    return %D : tensor<256x1024xf32>
  }
}

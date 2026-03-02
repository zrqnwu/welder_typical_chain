// MatMul + ReLU（f32）- 典型 GEMM epilogue
//
// 形状：(M,N,K) = (1024,1024,1024)
module {
  func.func @main(%A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32> {
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
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C : tensor<1024x1024xf32>) outs(%Dinit : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<1024x1024xf32>

    return %D : tensor<1024x1024xf32>
  }
}

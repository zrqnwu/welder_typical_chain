// MatMul（f32）- Transformer/MLP FC1 类形状
//
// 形状：(M,N,K) = (256,4096,1024)
module {
  func.func @main(%A: tensor<256x1024xf32>, %B: tensor<1024x4096xf32>)
      -> tensor<256x4096xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<256x4096xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<256x4096xf32>)
        -> tensor<256x4096xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<256x1024xf32>, tensor<1024x4096xf32>)
      outs(%Cinit : tensor<256x4096xf32>) -> tensor<256x4096xf32>

    return %C : tensor<256x4096xf32>
  }
}

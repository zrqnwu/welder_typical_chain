module {
  func.func @main() -> tensor<1024x1024xf32> {
    %cst_1 = arith.constant 1.0 : f32
    %cst_0 = arith.constant 0.0 : f32

    %A_init = tensor.empty() : tensor<1024x1024xf32>
    %B_init = tensor.empty() : tensor<1024x1024xf32>
    %C_init = tensor.empty() : tensor<1024x1024xf32>

    %A = linalg.fill ins(%cst_1 : f32) outs(%A_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %B = linalg.fill ins(%cst_1 : f32) outs(%B_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %C = linalg.fill ins(%cst_0 : f32) outs(%C_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %D = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%C : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %D : tensor<1024x1024xf32>
  }
}

// MatMul（f32）- 类模型 GEMM 基线
//
// 形状：(M,N,K) = (1024,1024,1024)
//
// 典型用法：
// 命令：OUT_DIR=/tmp/welder_case_out \
//   执行：bash bench/run_paper_profile_search.sh mlir_pipeline/paper_cases/matmul_f32_m1024_n1024_k1024.mlir ...
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

    return %C : tensor<1024x1024xf32>
  }
}

// Matmul -> linalg.softmax(named op) 链路。
//
// 用于验证本仓 canonicalize 阶段是否能自动把 linalg.softmax
// 分解成 max/exp/sum/div 的 generic 形式。
module {
  func.func @main(%A: tensor<8192x64xf32>, %B: tensor<64x128xf32>) -> tensor<8192x128xf32> {
    %f0 = arith.constant 0.0 : f32

    %c0 = tensor.empty() : tensor<8192x128xf32>
    %c_init = linalg.fill ins(%f0 : f32) outs(%c0 : tensor<8192x128xf32>) -> tensor<8192x128xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<8192x64xf32>, tensor<64x128xf32>)
      outs(%c_init : tensor<8192x128xf32>) -> tensor<8192x128xf32>

    %out0 = tensor.empty() : tensor<8192x128xf32>
    %out = linalg.softmax dimension(1)
      ins(%C : tensor<8192x128xf32>)
      outs(%out0 : tensor<8192x128xf32>) -> tensor<8192x128xf32>
    return %out : tensor<8192x128xf32>
  }
}


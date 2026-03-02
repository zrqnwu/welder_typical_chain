// TensorCore F16 多 kernel 健康性测试：两个独立 matmul（双汇点）。
//
// 目标：
// - 在启用 TensorCore 时走通切边 multi-kernel codegen 路径。
// - 预期生成 >=2 个 PTX `.entry`，且包含 `mma.sync`。
//
// 运行：
// 命令：bash mlir_pipeline/run_cut_edge_tensorcore_f16_two_sinks_e2e.sh

module {
  func.func @main(%A: tensor<16x16xf16>, %B0: tensor<16x16xf16>,
                  %B1: tensor<16x16xf16>) -> (tensor<16x16xf16>, tensor<16x16xf16>) {
    %c0 = arith.constant 0.0 : f16

    %e0 = tensor.empty() : tensor<16x16xf16>
    %c0init = linalg.fill ins(%c0 : f16) outs(%e0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %C0 = linalg.matmul ins(%A, %B0 : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%c0init : tensor<16x16xf16>) -> tensor<16x16xf16>

    %e1 = tensor.empty() : tensor<16x16xf16>
    %c1init = linalg.fill ins(%c0 : f16) outs(%e1 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %C1 = linalg.matmul ins(%A, %B1 : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%c1init : tensor<16x16xf16>) -> tensor<16x16xf16>

    return %C0, %C1 : tensor<16x16xf16>, tensor<16x16xf16>
  }
}

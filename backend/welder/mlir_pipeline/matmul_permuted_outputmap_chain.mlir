// Propagate v2（indexing_map 驱动）回归用例：
//
// MatMul -> Elementwise（置换后的 output map）-> ReLU
//
// 中间逐元素算子把 output indexing_map 设为 (d0,d1)->(d1,d0)，
// 同时保持输入 map 不变。这样语义上仍是逐元素（不做 transpose），
// 但循环顺序不再与输出张量维度顺序一致。
//
// Tile 传播必须使用生产者的 output indexing_map，把需求输出 footprint
// 反推回生产者并行循环的范围；若假设“输出维度顺序 == 并行循环顺序”，
// 在非方形形状上会把范围映射错，甚至越过静态循环边界。
//
// 形状：A(128x128) * B(128x64) -> C(128x64)
module {
  func.func @main(%A: tensor<128x128xf32>, %B: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<128x64xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x64xf32>)
      outs(%Cinit : tensor<128x64xf32>) -> tensor<128x64xf32>

    // 置换 output-map 的逐元素算子（输入/输出 map 对齐，语义仍正确）。
    %D0 = tensor.empty() : tensor<128x64xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%C : tensor<128x64xf32>) outs(%Dinit : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %relu = arith.maximumf %in, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x64xf32>

    // 常规逐元素汇点。
    %E0 = tensor.empty() : tensor<128x64xf32>
    %Einit = linalg.fill ins(%c0f : f32) outs(%E0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %E = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<128x64xf32>) outs(%Einit : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %relu = arith.maximumf %in, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x64xf32>

    return %E : tensor<128x64xf32>
  }
}

// 一个“类 Conv”的最小通用用例，用于验证 Phase 9.5：
// - generic loop 级分析/枚举（--enable-generic-problem）
// - TileGraph + tile 传播（--enable-tile-propagation）
// - Phase A 全局流量估计（--enable-global-traffic）
//
// 结构：Producer(带 reduction) -> Consumer(ReLU elementwise)
//
// 注意：这里不用 linalg.conv_2d* 的 named op，是为了保持 IR 简单、纯粹验证 solver 的
// loop/indexing_maps/propagation 逻辑。

module {
  func.func @main(%A: tensor<128x32x32xf32>) -> tensor<128x64x32xf32> {
    %f0 = arith.constant 0.0 : f32

    // 输出初始化：B[d0, d1, d2]
    %B0 = tensor.empty() : tensor<128x64x32xf32>
    %Binit = linalg.fill ins(%f0 : f32) outs(%B0 : tensor<128x64x32xf32>) -> tensor<128x64x32xf32>

    // 生产者：4 层循环，其中 (d0,d1,d2) 并行，(d3) 归约
    %B = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%A : tensor<128x32x32xf32>) outs(%Binit : tensor<128x64x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<128x64x32xf32>

    // 消费者：ReLU(B)
    %C0 = tensor.empty() : tensor<128x64x32xf32>
    %Cinit = linalg.fill ins(%f0 : f32) outs(%C0 : tensor<128x64x32xf32>) -> tensor<128x64x32xf32>
    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%B : tensor<128x64x32xf32>) outs(%Cinit : tensor<128x64x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %relu = arith.maximumf %in, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x64x32xf32>

    return %C : tensor<128x64x32xf32>
  }
}

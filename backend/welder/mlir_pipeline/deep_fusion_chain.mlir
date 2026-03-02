// 第 13B++ 阶段（Compiler）：深度融合（多跳）健康性测试
//
// 目标：
// - 构造一个长度 >= 3 的链式 TileGraph：P -> C1 -> Sink
// - 在 --enable-cut-edges 下，如果 compiler 只做 1-hop fuse，
//   则最上游的 producer(P) 往往会残留在 gpu.launch 之外（未进入 kernel）；
// - 在 “深度融合” 实现后，P/C1/Sink 都应被 fuse 进同一个 kernel（同一个 gpu.launch）。
//
// 用法：
// 命令：bash mlir_pipeline/run_deep_fusion_chain_e2e.sh
//
module {
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0f = arith.constant 0.0 : f32
    %c1f = arith.constant 1.0 : f32

    // 节点 P：p = arg0 + 1
    %p0 = tensor.empty() : tensor<8x8xf32>
    %pInit = linalg.fill ins(%c0f : f32) outs(%p0 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %p = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<8x8xf32>) outs(%pInit : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %c1f : f32
      linalg.yield %add : f32
    } -> tensor<8x8xf32>

    // 节点 C1：relu(p)
    %c10 = tensor.empty() : tensor<8x8xf32>
    %c1Init = linalg.fill ins(%c0f : f32) outs(%c10 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %c1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%p : tensor<8x8xf32>) outs(%c1Init : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<8x8xf32>

    // 汇点：out = c1 + 1
    %s0 = tensor.empty() : tensor<8x8xf32>
    %sInit = linalg.fill ins(%c0f : f32) outs(%s0 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%c1 : tensor<8x8xf32>) outs(%sInit : tensor<8x8xf32>) {
    ^bb0(%in: f32, %outv: f32):
      %add = arith.addf %in, %c1f : f32
      linalg.yield %add : f32
    } -> tensor<8x8xf32>

    return %out : tensor<8x8xf32>
  }
}

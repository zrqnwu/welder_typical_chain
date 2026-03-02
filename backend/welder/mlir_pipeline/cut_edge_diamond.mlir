// 第 13A 阶段（Solver）：切边菱形冲突测试
//
// 目标：
// - 构造一个“必然产生 tile 冲突”的 TileGraph，用于验证：
//   1) 旧逻辑：propagateTilesBackward 遇到冲突 -> 失败 -> 丢弃候选
//   2) 新逻辑（--enable-cut-edges）：遇到冲突 -> 标记 edge.isCut -> 继续求解
// - 并且在 --enable-cut-edges 下，Phase A global traffic 会出现 bytesCut > 0。
//
// 图结构（diamond）：
// 生产者 P（8x4）
//        /          \
// C1：identity；C2：读取首列（affine_map (d0,d1)->(d0,0)）
//        \          /
// 汇合节点 S：add(C1, C2)
//
// 冲突点：
// - 当 tileM != tileN（例如 root tile = [8,4]）时：
//   - C1 要求 P 的 tile = [8,4]
//   - C2 要求 P 的 tile = [8,1]
//   -> P 被两个 consumer 约束到不同的 requiredTile，触发冲突。
//
// 用法：
//   # 旧行为：不允许 cut，冲突候选会被过滤，可能无解
// 命令：bash compiler/run_welder_solver.sh mlir_pipeline/cut_edge_diamond.mlir \
//     参数：--enable-generic-problem --enable-tile-propagation --enable-global-traffic \
//     参数：--candidate-mn=8,4,2 --candidate-k=1
//
//   # 新行为：允许 cut，solver 会保留更大的 tile，并出现 bytesCut > 0
// 命令：bash compiler/run_welder_solver.sh mlir_pipeline/cut_edge_diamond.mlir \
//     参数：--enable-generic-problem --enable-cut-edges \
//     参数：--candidate-mn=8,4,2 --candidate-k=1

module {
  func.func @main(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %c0f = arith.constant 0.0 : f32
    %c1f = arith.constant 1.0 : f32

    // 生产者 P：p = arg0 + 1
    %p0 = tensor.empty() : tensor<8x4xf32>
    %pInit = linalg.fill ins(%c0f : f32) outs(%p0 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %p = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<8x4xf32>) outs(%pInit : tensor<8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %c1f : f32
      linalg.yield %add : f32
    } -> tensor<8x4xf32>

    // 消费者 C1：relu(p)
    %c10 = tensor.empty() : tensor<8x4xf32>
    %c1Init = linalg.fill ins(%c0f : f32) outs(%c10 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %c1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%p : tensor<8x4xf32>) outs(%c1Init : tensor<8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<8x4xf32>

    // 消费者 C2：将 p[d0,0] 广播到所有列
    %c20 = tensor.empty() : tensor<8x4xf32>
    %c2Init = linalg.fill ins(%c0f : f32) outs(%c20 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %c2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, 0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%p : tensor<8x4xf32>) outs(%c2Init : tensor<8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x4xf32>

    // 汇合节点 S：out = c1 + c2
    %s0 = tensor.empty() : tensor<8x4xf32>
    %sInit = linalg.fill ins(%c0f : f32) outs(%s0 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%c1, %c2 : tensor<8x4xf32>, tensor<8x4xf32>)
      outs(%sInit : tensor<8x4xf32>) {
    ^bb0(%a: f32, %b: f32, %outv: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<8x4xf32>

    return %out : tensor<8x4xf32>
  }
}

// TileGraph view-op 回归用例：
//
// 链路：linalg.generic -> tensor.collapse_shape(view) -> linalg.generic
//
// 该 reshape 是纯视图类 tensor op（无计算）。TileGraph 构建时
// 应把它剥离为 TileGraphEdge.viewOps，以保证：
// - 生产者/消费者 linalg op 之间仍由一条 edge 连接
// - tile 传播可跨越 reshape 正确映射所需 footprint
//
module {
  func.func @main(%arg0: tensor<128x64xf32>) -> tensor<8192xf32> {
    %c0 = arith.constant 0.0 : f32

    %tmp0 = tensor.empty() : tensor<128x64xf32>
    %init0 = linalg.fill ins(%c0 : f32) outs(%tmp0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %A = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<128x64xf32>) outs(%init0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x64xf32>

    %flat = tensor.collapse_shape %A [[0, 1]]
      : tensor<128x64xf32> into tensor<8192xf32>

    %tmp1 = tensor.empty() : tensor<8192xf32>
    %init1 = linalg.fill ins(%c0 : f32) outs(%tmp1 : tensor<8192xf32>) -> tensor<8192xf32>
    %B = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%flat : tensor<8192xf32>) outs(%init1 : tensor<8192xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8192xf32>

    return %B : tensor<8192xf32>
  }
}

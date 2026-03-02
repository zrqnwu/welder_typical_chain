// Broadcast + view-op 链路，用于压测 Propagate v2：
// - `tensor.expand_shape` 会被剥离到 TileGraphEdge.viewOps
// - rank-2 消费者通过常量 indexing_map 结果 `(d0,d1)->(d0,0)`，
//   将 [M,1] 张量在最后一维上做 broadcast
//
// 结构（静态形状）：
// 链路：v[M] -> W[M] -> expand_shape -> W2[M,1] -> Y[M,N]=A[M,N]+broadcast(W2)
//        -> sum[M] (row-wise 归约 over N)
module {
  func.func @main(%A: tensor<128x64xf32>, %v: tensor<128xf32>) -> tensor<128xf32> {
    %f0 = arith.constant 0.0 : f32

    // 定义：W[M] = identity(v)
    %w0 = tensor.empty() : tensor<128xf32>
    %W = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%v : tensor<128xf32>) outs(%w0 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128xf32>

    // W2[M,1] 是 W[M] 的纯视图 reshape（TileGraph 应把它当作 edge.viewOps）。
    %W2 = tensor.expand_shape %W [[0, 1]] output_shape [128, 1]
      : tensor<128xf32> into tensor<128x1xf32>

    // Y[M,N] = A[M,N] + broadcast(W2[M,1] 到 N 维，映射为 (d0,0))。
    %y0 = tensor.empty() : tensor<128x64xf32>
    %Y = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // A
        affine_map<(d0, d1) -> (d0, 0)>,  // W2 (broadcast along d1)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %W2 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%y0 : tensor<128x64xf32>) {
    ^bb0(%a: f32, %w: f32, %out: f32):
      %r = arith.addf %a, %w : f32
      linalg.yield %r : f32
    } -> tensor<128x64xf32>

    // 归约：sum[M] = reduce_add(Y, dim=1)
    %sum0 = tensor.empty() : tensor<128xf32>
    %sum_init = linalg.fill ins(%f0 : f32) outs(%sum0 : tensor<128xf32>) -> tensor<128xf32>
    %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // Y
        affine_map<(d0, d1) -> (d0)>      // out
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%Y : tensor<128x64xf32>) outs(%sum_init : tensor<128xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>

    return %sum : tensor<128xf32>
  }
}

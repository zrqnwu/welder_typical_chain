// 类 LayerNorm 链路，用于验证“归约 -> 归约”融合。
//
// 结构（全静态形状）：
// 链路：A[B,N] -> sum[B] -> mean[B] -> sq[B,N] -> sumsq[B] -> var[B] -> out[B,N]
//
// 关键模式是：先做归约 sum，再经由 mean 喂给后续归约 sumsq，
// 因此融合调度可以在多个归约之间复用同一份输入 tile/cache。
//
// 该文件用于 welder_try 的论文对齐融合实验。
module {
  func.func @main(%A: tensor<32x256xf32>) -> tensor<32x256xf32> {
    %f0 = arith.constant 0.0 : f32

    // sum 初始化：sum[B]
    %sum0 = tensor.empty() : tensor<32xf32>
    %sum_init = linalg.fill ins(%f0 : f32) outs(%sum0 : tensor<32xf32>) -> tensor<32xf32>
    %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A : tensor<32x256xf32>) outs(%sum_init : tensor<32xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<32xf32>

    // 公式：mean[B] = sum[B] * (1/N)
    %mean0 = tensor.empty() : tensor<32xf32>
    %mean = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%sum : tensor<32xf32>) outs(%mean0 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %invN = arith.constant 0.00390625 : f32 // 1/256
      %m = arith.mulf %in, %invN : f32
      linalg.yield %m : f32
    } -> tensor<32xf32>

    // 公式：sq[B,N] = (A - mean)^2
    %sq0 = tensor.empty() : tensor<32x256xf32>
    %sq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // A
        affine_map<(d0, d1) -> (d0)>,     // mean (broadcast)
        affine_map<(d0, d1) -> (d0, d1)>  // out
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %mean : tensor<32x256xf32>, tensor<32xf32>) outs(%sq0 : tensor<32x256xf32>) {
    ^bb0(%a: f32, %m: f32, %out: f32):
      %d = arith.subf %a, %m : f32
      %s = arith.mulf %d, %d : f32
      linalg.yield %s : f32
    } -> tensor<32x256xf32>

    // sumsq 初始化：sumsq[B]
    %sumsq0 = tensor.empty() : tensor<32xf32>
    %sumsq_init = linalg.fill ins(%f0 : f32) outs(%sumsq0 : tensor<32xf32>) -> tensor<32xf32>
    %sumsq = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%sq : tensor<32x256xf32>) outs(%sumsq_init : tensor<32xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<32xf32>

    // 公式：var[B] = sumsq[B] * (1/N)（variance = E[(x-mean)^2]）
    %var0 = tensor.empty() : tensor<32xf32>
    %var = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>, // sumsq
        affine_map<(d0) -> (d0)>  // out
      ],
      iterator_types = ["parallel"]
    } ins(%sumsq : tensor<32xf32>) outs(%var0 : tensor<32xf32>) {
    ^bb0(%s: f32, %out: f32):
      %invN = arith.constant 0.00390625 : f32 // 1/256
      %v = arith.mulf %s, %invN : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    // 公式：out[B,N] = broadcast(var[B])
    %out0 = tensor.empty() : tensor<32x256xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%var : tensor<32xf32>) outs(%out0 : tensor<32x256xf32>) {
    ^bb0(%v: f32, %out: f32):
      linalg.yield %v : f32
    } -> tensor<32x256xf32>

    return %out : tensor<32x256xf32>
  }
}

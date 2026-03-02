// 一个“可被 FootprintInference 推导”的 Conv2D+ReLU（用 linalg.generic 表达）。
//
// 目的：
// - 作为里程碑用例：Conv2D 能通过 welder-compiler(--enable-generic-problem)
//   走到 NVVM/PTX（gpu-module-to-binary{format=isa}）。
// - 这里故意不用 linalg.conv_2d_* named op，避免 stride/pad/symbol/floordiv 等
//   复杂 affine 表达式，先让骨架跑通。
//
// Loop order（关键）：把需要 block tile 的空间维放在最前面，且把“可分块的 reduction”
// 维（C=32）放在第一个 reduction 位置，方便 Phase 9 的枚举器使用 tk。
//
// 循环结构：
// 并行维：(oh, ow, f, n)
//   (c, kh, kw) 归约
//
// 形状：
// 输入 A：n=1, h=34, w=34, c=32
// 卷积核 B：kh=3, kw=3, c=32, f=64
// 输出 C：n=1, oh=32, ow=32, f=64
//
// 卷积公式：
// 公式：C[n,oh,ow,f] += A[n, oh+kh, ow+kw, c] * B[kh,kw,c,f]

module {
  func.func @main(%A: tensor<1x34x34x32xf32>, %B: tensor<3x3x32x64xf32>)
      -> tensor<1x32x32x64xf32> {
    %c0f = arith.constant 0.0 : f32

    %C0 = tensor.empty() : tensor<1x32x32x64xf32>
    %Cinit = linalg.fill ins(%c0f : f32) outs(%C0 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>

    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0 + d5, d1 + d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d4, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]
    } ins(%A, %B : tensor<1x34x34x32xf32>, tensor<3x3x32x64xf32>)
      outs(%Cinit : tensor<1x32x32x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<1x32x32x64xf32>

    // 对 C 做 ReLU
    %D0 = tensor.empty() : tensor<1x32x32x64xf32>
    %Dinit = linalg.fill ins(%c0f : f32) outs(%D0 : tensor<1x32x32x64xf32>)
        -> tensor<1x32x32x64xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%C : tensor<1x32x32x64xf32>) outs(%Dinit : tensor<1x32x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %c0f : f32
      linalg.yield %relu : f32
    } -> tensor<1x32x32x64xf32>

    return %D : tensor<1x32x32x64xf32>
  }
}

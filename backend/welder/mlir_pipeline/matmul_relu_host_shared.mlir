// MatMul -> ReLU 融合实验（Host Shared + GPU 运行）。
//
// 目标：
// - 让 transform 能“以 ReLU 为 consumer”驱动 matmul 的切分与融合（Phase 3: Fusion 的雏形）。
// - 最终仍能 lower 到 NVVM 并用 mlir-runner 在 GPU 上跑出结果。
//
// 
// - A/B/C 仍然用 gpu.alloc host_shared 分配，保证 host 初始化的数据能被 device 直接访问。
// - MatMul/ReLU 这里用 tensor 语义（有 SSA 结果），这是为了让 transform 的 fusion（tile+fuse）
//   能工作；之后会在 pipeline 里 bufferize 成 memref 再做 shared promotion / gpu.launch。
// - 通过 bufferization.materialize_in_destination 明确把最终 tensor 结果写回到 C(memref)。
//
// 期望输出：128

module attributes {gpu.container_module} {
  func.func private @printF32(f32)
  func.func private @printNewline()

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32

    // host_shared = managed memory（host+device 都可访问）
    %A = gpu.alloc host_shared () : memref<128x128xf32>
    %B = gpu.alloc host_shared () : memref<128x128xf32>
    %C = gpu.alloc host_shared () : memref<128x128xf32>
    // Phase 4（向量化）小技巧：给编译器一个“对齐假设”，更容易生成 ld.global.v4 / st.shared.v4。
    // 如果对齐假设不成立，行为是 poison（所以这里选 16 字节：对 float4 的自然对齐）。
    %A_aligned = memref.assume_alignment %A, 16 : memref<128x128xf32>
    %B_aligned = memref.assume_alignment %B, 16 : memref<128x128xf32>
    %C_aligned = memref.assume_alignment %C, 16 : memref<128x128xf32>

    // 初始化：A/B = 1，C = 0
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %f1, %A_aligned[%i, %j] : memref<128x128xf32>
        memref.store %f1, %B_aligned[%i, %j] : memref<128x128xf32>
        memref.store %f0, %C_aligned[%i, %j] : memref<128x128xf32>
      }
    }

    // 把 memref 视为 tensor（仅用于让上层 transform/fusion 在 tensor SSA 上工作）
    // 注意：One-Shot Bufferize 只支持带 `restrict` 的 to_tensor。
    %At = bufferization.to_tensor %A_aligned restrict
      : memref<128x128xf32> to tensor<128x128xf32>
    %Bt = bufferization.to_tensor %B_aligned restrict
      : memref<128x128xf32> to tensor<128x128xf32>

    // MatMul 的初始输出 tensor（全 0）
    %Cinit0 = tensor.empty() : tensor<128x128xf32>
    %Cinit = linalg.fill ins(%f0 : f32) outs(%Cinit0 : tensor<128x128xf32>) -> tensor<128x128xf32>

    // 生产者：D = A*B
    %D = linalg.matmul
      ins(%At, %Bt : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%Cinit : tensor<128x128xf32>) -> tensor<128x128xf32>

    // 消费者：ReLU(D)
    %E = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // input
        affine_map<(d0, d1) -> (d0, d1)>  // output
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<128x128xf32>) outs(%Cinit : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %relu = arith.maximumf %in, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x128xf32>

    // 明确把最终 tensor 结果写回到 C(memref)，避免依赖 bufferization 的“自动就地复用”决策。
    bufferization.materialize_in_destination %E in writable %C_aligned
      : (tensor<128x128xf32>, memref<128x128xf32>) -> ()

    gpu.wait

    %v = memref.load %C_aligned[%c0, %c0] : memref<128x128xf32>
    func.call @printF32(%v) : (f32) -> ()
    func.call @printNewline() : () -> ()
    return
  }
}

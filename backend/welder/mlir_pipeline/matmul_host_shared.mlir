// 一个可在 GPU 上运行的 matmul 输入（给 transform 用）。
//
// 关键点：
// - A/B/C 用 gpu.alloc host_shared 分配（host+device 都可访问），避免把纯 CPU malloc 的指针
//   直接传给 GPU kernel 导致的 illegal address。
// - matmul 仍然用 linalg.matmul（纯 memref 语义），后续由 transform 做 tiling/mapping/shared promotion。
//
// 期望：如果 A/B 全 1，C[0,0] = 128。
//
// 运行：见 mlir_pipeline/run_transform_k_tiled_runnable.sh

module attributes {gpu.container_module} {
  func.func private @printF32(f32)
  func.func private @printNewline()

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32

    // host_shared = managed memory（示例里最省事，能直接在 host 初始化，也能被 device kernel 访问）。
    %A = gpu.alloc host_shared () : memref<128x128xf32>
    %B = gpu.alloc host_shared () : memref<128x128xf32>
    %C = gpu.alloc host_shared () : memref<128x128xf32>

    // 在 host 初始化。
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %f1, %A[%i, %j] : memref<128x128xf32>
        memref.store %f1, %B[%i, %j] : memref<128x128xf32>
        memref.store %f0, %C[%i, %j] : memref<128x128xf32>
      }
    }

    // 这条 linalg.matmul 会被 transform 改写成 gpu.launch +（可选）shared/workgroup。
    linalg.matmul
      ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
      outs(%C : memref<128x128xf32>)

    // 等 GPU 完成，再读回打印。
    gpu.wait

    %v = memref.load %C[%c0, %c0] : memref<128x128xf32>
    func.call @printF32(%v) : (f32) -> ()
    func.call @printNewline() : () -> ()
    return
  }
}


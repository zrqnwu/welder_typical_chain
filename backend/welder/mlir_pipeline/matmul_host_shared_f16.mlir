// 最小可运行的 f16 matmul（host_shared 缓冲区）。
// 用于验证 TensorCore（mma.sync）的 f16 路径。
//
// 预期：若 A/B 全为 1，则 C[0,0] == 128。
module attributes {gpu.container_module} {
  func.func private @printF32(f32)
  func.func private @printNewline()

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %f0 = arith.constant 0.0 : f16
    %f1 = arith.constant 1.0 : f16

    %A = gpu.alloc host_shared () : memref<128x128xf16>
    %B = gpu.alloc host_shared () : memref<128x128xf16>
    %C = gpu.alloc host_shared () : memref<128x128xf16>

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %f1, %A[%i, %j] : memref<128x128xf16>
        memref.store %f1, %B[%i, %j] : memref<128x128xf16>
        memref.store %f0, %C[%i, %j] : memref<128x128xf16>
      }
    }

    linalg.matmul
      ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
      outs(%C : memref<128x128xf16>)

    gpu.wait

    %v = memref.load %C[%c0, %c0] : memref<128x128xf16>
    %vf32 = arith.extf %v : f16 to f32
    func.call @printF32(%vf32) : (f32) -> ()
    func.call @printNewline() : () -> ()
    return
  }
}

// workgroup/shared `memref.dealloc` lowering 问题的最小复现。
//
// 期望行为：GPU/NVVM lowering 不应把 workgroup dealloc 降成 `free()`。
// 当前行为（缺陷）：`memref.dealloc` 被无条件降成 `free`，
// 导致地址空间不匹配（`!llvm.ptr<3>` 被传给 `free(!llvm.ptr)`）。
//
// 复现命令：
// 环境：BIN=llvm-project/build/bin
// 执行：$BIN/mlir-opt mlir_pipeline/repro_workgroup_dealloc.mlir \
// 参数：--gpu-lower-to-nvvm-pipeline="cubin-chip=sm_86 cubin-format=isa" \
// 输出：-o /tmp/out.nvvm.mlir
//
// 典型报错：
// 报错：'llvm.call' op operand type mismatch for operand 0: '!llvm.ptr<3>' != '!llvm.ptr'
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = memref.alloc() : memref<1xi8>
    gpu.launch blocks(%bx, %by, %bz) in (%gridX = %c1, %gridY = %c1, %gridZ = %c1)
               threads(%tx, %ty, %tz) in (%blockX = %c1, %blockY = %c1, %blockZ = %c1) {
      %wg = memref.alloc() : memref<1xi8, #gpu.address_space<workgroup>>
      %tid = gpu.thread_id x
      %tid_i8 = arith.index_cast %tid : index to i8
      memref.store %tid_i8, %wg[%c0] : memref<1xi8, #gpu.address_space<workgroup>>
      %v = memref.load %wg[%c0] : memref<1xi8, #gpu.address_space<workgroup>>
      memref.store %v, %out[%c0] : memref<1xi8>
      memref.dealloc %wg : memref<1xi8, #gpu.address_space<workgroup>>
      gpu.terminator
    }
    memref.dealloc %out : memref<1xi8>
    return
  }
}

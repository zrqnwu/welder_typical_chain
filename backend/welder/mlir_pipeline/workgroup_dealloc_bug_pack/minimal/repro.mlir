// workgroup/shared `memref.dealloc` lowering 问题的最小复现。
//
// 现象：GPU/NVVM lowering 失败，因为 memref.dealloc 被降成了 `free`，
// 但其操作数是非默认地址空间指针（例如 `!llvm.ptr<3>`）。
//
// 运行命令：
// 环境：BIN=llvm-project/build/bin
// 执行：$BIN/mlir-opt mlir_pipeline/workgroup_dealloc_bug_pack/minimal/repro.mlir \
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

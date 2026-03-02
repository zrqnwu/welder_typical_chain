module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} attributes {sym_name = "main"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %mm = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.any_op) -> !transform.any_op

    // 1) 用 forall 对 M/N 做分块（后续映射到 GPU blocks）。
    //    注意：这里先不切 K，因为 K 是 reduction 维度。
    %tiled_mn, %forall = transform.structured.tile_using_forall %mm
      tile_sizes [64, 64, 0] {mapping = [#gpu.block<y>, #gpu.block<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2) 在每个 block 内，用串行的 scf.for 对 K 再做分块。
    %tiled_mnk, %k_loop = transform.structured.tile_using_for %tiled_mn
      tile_sizes [0, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3) 共享内存（workgroup/shared）promotion：把 A/B 的 tile 提升到 workgroup 内存。
    //
    // 注意：structured.promote(memory_space=workgroup) 默认会生成
    //   memref.alloc + memref.dealloc（类型带 #gpu.address_space<workgroup>）。
    // 在当前 LLVM/NVVM lowering 里，memref.dealloc(workgroup) 会错误地尝试降成 free(ptr<3>)，
    // 触发 addrspace 类型不匹配的编译期报错（这是我们要提交的上游 bug）。
    //
    // 解决思路（不改上游的前提下）：
    // - 旧 workaround：在进 NVVM pipeline 前，用 sed 直接删掉 memref.dealloc(workgroup)；
    // - 更正确的 workaround：把这些 workgroup 的 memref.alloc/dealloc 改写成
    //   gpu.launch workgroup(...) attribution（真正的 shared），并删除 dealloc。
    //   对应脚本：mlir_pipeline/run_transform_k_tiled_runnable.sh
    // 小技巧：我们这里 tile size 都是整除的（128 可被 64/16 整除），不需要用 fill 去处理边界块，
    // 所以把 use_full_tile_buffers 关掉，避免生成额外的 linalg.fill（也避免“所有线程一起 fill 同一块 shared”这种浪费）。
    %promoted = transform.structured.promote %tiled_mnk {
      operands_to_promote = [0, 1],
      use_full_tile_buffers = [false, false],
      memory_space = #gpu.address_space<workgroup>
    } : (!transform.any_op) -> !transform.any_op

    // 4) [新增] L2 Thread Tiling：把 block 内的“搬运(shared copy) + 计算(matmul)”分配给线程。
    //
    // 核心动机：如果不做 thread tiling，gpu.launch 的 threads 维度会是 1x1x1，
    // 也就是每个 block 只有 1 个线程在跑 64x64 的计算（极慢，也不符合论文里的“物理层”）。
    //
    // 这里选一个简单的配置：
    // - block 里用 16x16 = 256 个线程；
    // - 每个线程算 4x4 个输出元素（覆盖 64x64）；
    // - linalg.copy 也按 4x4 切块，让线程并行把 A/B tile 搬到 shared；
    // - 后续用 transform.gpu.map_nested_forall_to_threads 把这些 forall 真正变成 threadIdx.x/y。
    %copies = transform.structured.match ops{["linalg.copy"]} in %func
      : (!transform.any_op) -> !transform.any_op
    %tiled_copy, %forall_copy = transform.structured.tile_using_forall %copies
      tile_sizes [4, 4] {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_thread, %forall_thread = transform.structured.tile_using_forall %promoted
      tile_sizes [4, 4, 0] {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 5) 把 block-level forall 映射到 gpu.launch（生成 launch op）。
    %gpu = transform.gpu.map_forall_to_blocks %forall { generate_gpu_launch }
      : (!transform.any_op) -> !transform.any_op

    // 6) 把 launch 里的 thread-level forall 映射到 gpu.thread_id，并设置 blockDim。
    // 这样 NVVM/PTX 里就会出现 tid.x/tid.y，并且 kernel 的 maxntid 也不再是 1,1,1。
    %gpu2 = transform.gpu.map_nested_forall_to_threads %gpu block_dims = [16, 16, 1]
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

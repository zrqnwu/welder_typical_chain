module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} attributes {sym_name = "main"} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // 1) 匹配（已 bufferize 的）MatMul。
    %mm = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.any_op) -> !transform.any_op

    // 2) K 维（reduction）再切一刀：K=128 -> step 16（和之前 matmul-only 的做法一致）。
    %tiled_mnk, %k_loop = transform.structured.tile_using_for %mm
      tile_sizes [0, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3) shared/workgroup promotion：把 A/B tile 提升到 workgroup(shared)。
    //    注意：这里会生成 memref.alloc(workgroup)；我们用插件 pass 把它改成 gpu.launch workgroup(...)
    //    并删掉 workgroup dealloc，以绕开上游 dealloc lowering bug。
    //
    // 重要：这里的 memory_space 我们用“整数 3”，而不是 #gpu.address_space<workgroup>。
    // 原因是：一旦你对 linalg.copy 做了 vectorize，后续会走 convert-vector-to-llvm；
    // 该转换当前不接受 #gpu.address_space<workgroup> 这种“非整数”的 memref memory space，
    // 会直接报：conversion of memref memory space ... to integer address space failed。
    // 用整数 3（NVVM 的 shared addrspace）可以让整条 pipeline 顺利走完。
    %promoted = transform.structured.promote %tiled_mnk {
      operands_to_promote = [0, 1],
      use_full_tile_buffers = [false, false],
      memory_space = 3
    } : (!transform.any_op) -> !transform.any_op

    // 4) L2 thread tiling：把 copy / matmul / relu 都映射到同一套线程网格上。
    %copies = transform.structured.match ops{["linalg.copy"]} in %func
      : (!transform.any_op) -> !transform.any_op
    %tiled_copy, %forall_copy = transform.structured.tile_using_forall %copies
      tile_sizes [4, 4] {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // L3 向量化（先从“搬运”开始）：优先把最内层连续维打包成 vector<4xf32>。
    //
    // 注意：tile_using_forall 的返回 handle 在“多目标”场景下可能包含额外 op。
    // 这里我们直接在全函数里重新 match linalg.copy（此时只剩搬运用的 copy），再逐个 vectorize，最稳。
    %copies_after_tiling = transform.structured.match ops{["linalg.copy"]} in %func
      : (!transform.any_op) -> !transform.any_op
    transform.foreach %copies_after_tiling : !transform.any_op {
    ^bb1(%copy: !transform.any_op):
      // 这里用 “masked vectorization”：显式给出 vector_sizes，并要求 >= iteration space。
      // 因为我们已经把 copy tile 成 4x4，所以这里用 [4, 4] 是最稳的。
      //
      // 之后在 NVVM pipeline 里我们会用 convert-vector-to-scf{target-rank=1}
      // 把 2D transfer flatten 成 1D 的 vector<4xf32>，再接 convert-vector-to-llvm，
      // 以便让后端有机会生成 ld.global.v4 / st.shared.v4。
      transform.structured.vectorize %copy vector_sizes [4, 4]
        : !transform.any_op
    }

    %tiled_thread_mm, %forall_mm = transform.structured.tile_using_forall %promoted
      tile_sizes [4, 4, 0] {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %relu = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.any_op) -> !transform.any_op
    %tiled_thread_relu, %forall_relu = transform.structured.tile_using_forall %relu
      tile_sizes [4, 4] {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 5) 把“最外层 block-level scf.forall”映射到 gpu.launch。
    //
    // 注意：这一步不能把 handle 指到某个“嵌套的 thread-level forall”上，否则
    // map_forall_to_blocks 会报 “could not find a unique topLevel scf.forall”。
    // 这里直接把 %func 作为 target，让 transform 自己去找唯一的 top-level forall（它会忽略嵌套 forall）。
    %gpu = transform.gpu.map_forall_to_blocks %func { generate_gpu_launch }
      : (!transform.any_op) -> !transform.any_op

    // 6) 把 launch 里的 thread-level forall 映射到 threadIdx，并设置 blockDim。
    %gpu2 = transform.gpu.map_nested_forall_to_threads %gpu block_dims = [16, 16, 1]
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

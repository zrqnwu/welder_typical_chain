module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} attributes {sym_name = "main"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %mm = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.any_op) -> !transform.any_op

    // 1) 切分：在 (M, N) 维使用 64x64；K 维不切（0）。
    // 使用 forall，便于后续映射到 gpu.launch。
    %tiled, %forall = transform.structured.tile_using_forall %mm
      tile_sizes [64, 64, 0] {mapping = [#gpu.block<y>, #gpu.block<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2) Shared memory 提升：将 A/B tile 提升到 workgroup（shared）内存。
    // 注意：该操作作用在 buffer（memref）操作数上；若输入是 tensor，需先做 bufferization。
    %promoted = transform.structured.promote %tiled {
      operands_to_promote = [0, 1],
      use_full_tile_buffers = [true, true],
      memory_space = #gpu.address_space<workgroup>
    } : (!transform.any_op) -> !transform.any_op

    // 3) 将 forall 映射到 gpu.launch（生成 launch op）。
    %gpu = transform.gpu.map_forall_to_blocks %forall { generate_gpu_launch }
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

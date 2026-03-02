// TensorCore（TF32）matmul 路径（最小示例）：
// - 将 matmul 切分为兼容 MMA 的形状：M=16, N=8, K=4。
// - 通过 transform.nvgpu 把 linalg.matmul 改写成 nvgpu.mma.sync。
// - 将 tile 映射到 gpu.blocks，并把 blockDim 设为单个 warp（32 线程）。
//
// 该文件刻意保持最小化，只用于验证论文中“支持 TensorCore”这一路径
// （Welder §4.1），不涉及对整套 Welder solver/codegen 栈的重构。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} attributes {sym_name = "main"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %mm = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.any_op) -> !transform.any_op

    // 1) 块级切分：(M,N) -> (block<y>, block<x>)。
    %tiled_mn, %forall = transform.structured.tile_using_forall %mm
      tile_sizes [16, 8, 0] {mapping = [#gpu.block<y>, #gpu.block<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2) 对 K 维切分，使其匹配 TF32 MMA 的 K=4。
    %tiled_mnk, %k_loop = transform.structured.tile_using_for %tiled_mn
      tile_sizes [0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3) 将切分后的 matmul 改写为 MMA sync 指令及所需的向量搬运。
    transform.nvgpu.rewrite_matmul_as_mma_sync %tiled_mnk
      : (!transform.any_op) -> ()

    // 4) 将块级 forall 映射到 gpu.launch。
    %gpu = transform.gpu.map_forall_to_blocks %forall { generate_gpu_launch }
      : (!transform.any_op) -> !transform.any_op

    // 5) 每个 block 使用单个 warp 执行 MMA sync。
    %gpu2 = transform.gpu.map_nested_forall_to_threads %gpu block_dims = [32, 1, 1]
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// 这个文件是 “模板版” Transform（tensor stage）。
// 目的：让脚本中的 tile 参数可以由外部（WelderSolver / 脚本）注入。
//
// 占位符（会被脚本替换成整数）：
// - 参数 ${TILE_M}
// - 参数 ${TILE_N}
//
// 约束建议：
// - TILE_M / TILE_N 最好能被 thread tile（当前固定 4）整除，
//   这样后续 block_dims = [TILE_N/4, TILE_M/4, 1] 才能“刚好覆盖”整个 tile。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} attributes {sym_name = "main"} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // 1) 匹配 Consumer：ReLU（linalg.generic）
    %relu = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.any_op) -> !transform.any_op

    // 2) 先对 Consumer 做 L1（Block-level）切分，生成一个带 block mapping 的 scf.forall。
    //    这一步相当于 WELDER 里“由 consumer 反推需要的 tile”。
    %tiled_relu, %forall = transform.structured.tile_using_forall %relu
      tile_sizes [${TILE_M}, ${TILE_N}] {mapping = [#gpu.block<y>, #gpu.block<x>]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3) 通过 tiled consumer 的 operand 反向找到 producer（MatMul）。
    //    注意：tile 后 operand 往往先变成 tensor.extract_slice，所以需要再往上追一次。
    %slice = transform.get_producer_of_operand %tiled_relu[0]
      : (!transform.any_op) -> !transform.any_op
    %mm = transform.get_producer_of_operand %slice[0]
      : (!transform.any_op) -> !transform.any_op

    // 4) 把 Producer（MatMul）tile+fuse 进这个 block-level forall 里。
    //    这是真正的“算子融合”（Producer-Consumer 锁死在同一个 block 的同一个 loop body）。
    %fused, %new_forall = transform.structured.fuse_into_containing_op %mm into %forall
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

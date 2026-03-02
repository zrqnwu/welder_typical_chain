// 第 12 阶段：正确性校验（Conv2D+ReLU，可在 GPU 上直接运行）
//
// 目标：
// - 让通用 solver/codegen（--enable-generic-problem + --enable-generic-fusion）
//   产物能够真正跑在 GPU 上，并输出可用于对比的数值。
// - 这里用非常小的静态 shape，方便你用 Python 写 reference（不依赖 torch/numpy 也能算）。
//
// 形状：
// 输入 A：n=1, h=6, w=6, c=2
// 卷积核 B：kh=3, kw=3, c=2, f=3
// 输出 D：n=1, oh=4, ow=4, f=3（OH=H-KH+1, OW=W-KW+1）
//
// Conv2D (用 linalg.generic 表达，避免 named conv 的复杂 affine/symbol):
// 公式：C[n,oh,ow,f] += A[n, oh+kh, ow+kw, c] * B[kh,kw,c,f]
// 激活：ReLU
// 公式：D = max(C, 0)
//
// 初始化数据（确定性、能区分索引错误）：
//   规则：A[n,h,w,c] = (h+1)*100 + (w+1)*10 + (c+1)
// 规则：B[kh,kw,c,f] = sign(f) * ((kh+1)*100 + (kw+1)*10 + (c+1) + f)
// - 当 f 为偶数时 sign(f)=+1，f 为奇数时 sign(f)=-1
//
// 输出（方便 Python 解析）：
//   1) sum(D)  (所有输出元素之和)
//   2) 示例值 D[0,0,0,0]
//   3) D[0,0,0,1]   // f=1 是负号 filter，理论上应被 ReLU 截断为 0
//
// 你可以用：
// 命令：python3 welder_try/bench/verify_conv2d_correctness.py
// 来自动完成 “编译到 NVVM + mlir-runner 执行 + 与 Python reference 对比”。

module attributes {gpu.container_module} {
  func.func private @printF32(f32)
  func.func private @printNewline()

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %cN = arith.constant 1 : index
    %cH = arith.constant 6 : index
    %cW = arith.constant 6 : index
    %cC = arith.constant 2 : index
    %cKH = arith.constant 3 : index
    %cKW = arith.constant 3 : index
    %cF = arith.constant 3 : index

    %cOH = arith.constant 4 : index
    %cOW = arith.constant 4 : index

    %zero_f32 = arith.constant 0.0 : f32

    %zero_i32 = arith.constant 0 : i32
    %one_i32 = arith.constant 1 : i32
    %ten_i32 = arith.constant 10 : i32
    %hundred_i32 = arith.constant 100 : i32

    // host_shared = managed memory（host+device 都可访问）
    %A = gpu.alloc host_shared () : memref<1x6x6x2xf32>
    %B = gpu.alloc host_shared () : memref<3x3x2x3xf32>
    %Out = gpu.alloc host_shared () : memref<1x4x4x3xf32>
    %A_aligned = memref.assume_alignment %A, 16 : memref<1x6x6x2xf32>
    %B_aligned = memref.assume_alignment %B, 16 : memref<3x3x2x3xf32>
    %Out_aligned = memref.assume_alignment %Out, 16 : memref<1x4x4x3xf32>

    // 初始化 A
    scf.for %n = %c0 to %cN step %c1 {
      scf.for %h = %c0 to %cH step %c1 {
        scf.for %w = %c0 to %cW step %c1 {
          scf.for %c = %c0 to %cC step %c1 {
            // 规则：A[n,h,w,c] = (h+1)*100 + (w+1)*10 + (c+1)
            %h_i32 = arith.index_cast %h : index to i32
            %w_i32 = arith.index_cast %w : index to i32
            %c_i32 = arith.index_cast %c : index to i32
            %h1 = arith.addi %h_i32, %one_i32 : i32
            %w1 = arith.addi %w_i32, %one_i32 : i32
            %c1_i32 = arith.addi %c_i32, %one_i32 : i32
            %t0 = arith.muli %h1, %hundred_i32 : i32
            %t1 = arith.muli %w1, %ten_i32 : i32
            %t2 = arith.addi %t0, %t1 : i32
            %t3 = arith.addi %t2, %c1_i32 : i32
            %val = arith.sitofp %t3 : i32 to f32
            memref.store %val, %A_aligned[%n, %h, %w, %c] : memref<1x6x6x2xf32>
          }
        }
      }
    }

    // 初始化 B
    scf.for %kh = %c0 to %cKH step %c1 {
      scf.for %kw = %c0 to %cKW step %c1 {
        scf.for %c = %c0 to %cC step %c1 {
          scf.for %f = %c0 to %cF step %c1 {
            // 规则：base = (kh+1)*100 + (kw+1)*10 + (c+1) + f
            // 规则：f 为偶数时取 +base，f 为奇数时取 -base
            %kh_i32 = arith.index_cast %kh : index to i32
            %kw_i32 = arith.index_cast %kw : index to i32
            %c_i32 = arith.index_cast %c : index to i32
            %f_i32 = arith.index_cast %f : index to i32
            %kh1 = arith.addi %kh_i32, %one_i32 : i32
            %kw1 = arith.addi %kw_i32, %one_i32 : i32
            %c1_i32 = arith.addi %c_i32, %one_i32 : i32
            %t0 = arith.muli %kh1, %hundred_i32 : i32
            %t1 = arith.muli %kw1, %ten_i32 : i32
            %t2 = arith.addi %t0, %t1 : i32
            %t3 = arith.addi %t2, %c1_i32 : i32
            %base = arith.addi %t3, %f_i32 : i32

            %lsb = arith.andi %f_i32, %one_i32 : i32
            %is_odd = arith.cmpi eq, %lsb, %one_i32 : i32
            %signed = scf.if %is_odd -> (i32) {
              %neg = arith.subi %zero_i32, %base : i32
              scf.yield %neg : i32
            } else {
              scf.yield %base : i32
            }
            %val = arith.sitofp %signed : i32 to f32
            memref.store %val, %B_aligned[%kh, %kw, %c, %f] : memref<3x3x2x3xf32>
          }
        }
      }
    }

    // 把 memref 视为 tensor（仅用于让上层 transform/fusion 在 tensor SSA 上工作）
    %At = bufferization.to_tensor %A_aligned restrict
      : memref<1x6x6x2xf32> to tensor<1x6x6x2xf32>
    %Bt = bufferization.to_tensor %B_aligned restrict
      : memref<3x3x2x3xf32> to tensor<3x3x2x3xf32>

    %C0 = tensor.empty() : tensor<1x4x4x3xf32>
    %Cinit = linalg.fill ins(%zero_f32 : f32) outs(%C0 : tensor<1x4x4x3xf32>)
        -> tensor<1x4x4x3xf32>

    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0 + d5, d1 + d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d4, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]
    } ins(%At, %Bt : tensor<1x6x6x2xf32>, tensor<3x3x2x3xf32>)
      outs(%Cinit : tensor<1x4x4x3xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<1x4x4x3xf32>

    // 对 C 做 ReLU
    %D0 = tensor.empty() : tensor<1x4x4x3xf32>
    %Dinit = linalg.fill ins(%zero_f32 : f32) outs(%D0 : tensor<1x4x4x3xf32>)
        -> tensor<1x4x4x3xf32>
    %D = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%C : tensor<1x4x4x3xf32>) outs(%Dinit : tensor<1x4x4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %relu = arith.maximumf %in, %zero_f32 : f32
      linalg.yield %relu : f32
    } -> tensor<1x4x4x3xf32>

    // 明确把最终 tensor 结果写回到 Out(memref)，避免依赖 bufferization 的“自动就地复用”决策。
    bufferization.materialize_in_destination %D in writable %Out_aligned
      : (tensor<1x4x4x3xf32>, memref<1x4x4x3xf32>) -> ()

    gpu.wait

    // 校验和 = sum(Out)
    %sum0 = arith.constant 0.0 : f32
    %sum_n = scf.for %n = %c0 to %cN step %c1 iter_args(%acc0 = %sum0) -> (f32) {
      %sum_h = scf.for %oh = %c0 to %cOH step %c1 iter_args(%acc1 = %acc0) -> (f32) {
        %sum_w = scf.for %ow = %c0 to %cOW step %c1 iter_args(%acc2 = %acc1) -> (f32) {
          %sum_f = scf.for %f = %c0 to %cF step %c1 iter_args(%acc3 = %acc2) -> (f32) {
            %v = memref.load %Out_aligned[%n, %oh, %ow, %f] : memref<1x4x4x3xf32>
            %acc4 = arith.addf %acc3, %v : f32
            scf.yield %acc4 : f32
          }
          scf.yield %sum_f : f32
        }
        scf.yield %sum_w : f32
      }
      scf.yield %sum_h : f32
    }

    func.call @printF32(%sum_n) : (f32) -> ()
    func.call @printNewline() : () -> ()

    %v0 = memref.load %Out_aligned[%c0, %c0, %c0, %c0] : memref<1x4x4x3xf32>
    func.call @printF32(%v0) : (f32) -> ()
    func.call @printNewline() : () -> ()

    %v1 = memref.load %Out_aligned[%c0, %c0, %c0, %c1] : memref<1x4x4x3xf32>
    func.call @printF32(%v1) : (f32) -> ()
    func.call @printNewline() : () -> ()

    return
  }
}

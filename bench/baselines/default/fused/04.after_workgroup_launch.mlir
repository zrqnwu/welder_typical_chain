#map = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 8192 + s2 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0 mod 32)>
#map3 = affine_map<()[s0, s1] -> (s1 * 16 + (s0 floordiv 32) * 8)>
#map4 = affine_map<()[s0, s1] -> (s0 * 4 + s1 * 256)>
#map5 = affine_map<()[s0, s1] -> (s1 * 256 + (s0 floordiv 2) * 8)>
#map6 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 8)>
#map9 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 64) * 128)>
#map10 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 4)>
#map11 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
#map12 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 32) * 4)>
#map13 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 32) * 128)>
#map14 = affine_map<(d0, d1) -> (d0)>
#map15 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 512)>
module {
  func.func @main(%arg0: memref<8192x64xf16, strided<[?, ?], offset: ?>>, %arg1: memref<64x128xf16, strided<[?, ?], offset: ?>>) -> memref<8192x128xf32> {
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %cst_2 = arith.constant -3.40282347E+38 : f32
    %memref = gpu.alloc  host_shared () : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) workgroup(%arg14 : memref<72448xi8, 3>) {
      %c0_3 = arith.constant 0 : index
      %view = memref.view %arg14[%c0_3][] : memref<72448xi8, 3> to memref<1024xi8, 3>
      %c1024 = arith.constant 1024 : index
      %view_4 = memref.view %arg14[%c1024][] : memref<72448xi8, 3> to memref<4096xi8, 3>
      %c5120 = arith.constant 5120 : index
      %view_5 = memref.view %arg14[%c5120][] : memref<72448xi8, 3> to memref<8192xi8, 3>
      %c13312 = arith.constant 13312 : index
      %view_6 = memref.view %arg14[%c13312][] : memref<72448xi8, 3> to memref<1024xi8, 3>
      %c14336 = arith.constant 14336 : index
      %view_7 = memref.view %arg14[%c14336][] : memref<72448xi8, 3> to memref<256xi8, 3>
      %c14336_8 = arith.constant 14336 : index
      %0 = arith.addi %c0, %c14336_8 : index
      %view_9 = memref.view %arg14[%0][] : memref<72448xi8, 3> to memref<64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %1 = arith.muli %thread_id_y, %block_dim_x : index
      %2 = arith.addi %thread_id_x, %1 : index
      %3 = arith.muli %block_dim_x, %block_dim_y : index
      scf.for %arg15 = %2 to %c64 step %3 {
        memref.store %cst_0, %view_9[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %c14592 = arith.constant 14592 : index
      %view_10 = memref.view %arg14[%c14592][] : memref<72448xi8, 3> to memref<256xi8, 3>
      %c14592_11 = arith.constant 14592 : index
      %4 = arith.addi %c0, %c14592_11 : index
      %view_12 = memref.view %arg14[%4][] : memref<72448xi8, 3> to memref<64xf32, 3>
      scf.for %arg15 = %2 to %c64 step %3 {
        memref.store %cst, %view_12[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %c14848 = arith.constant 14848 : index
      %view_13 = memref.view %arg14[%c14848][] : memref<72448xi8, 3> to memref<256xi8, 3>
      %c14848_14 = arith.constant 14848 : index
      %5 = arith.addi %c0, %c14848_14 : index
      %view_15 = memref.view %arg14[%5][] : memref<72448xi8, 3> to memref<64xf32, 3>
      scf.for %arg15 = %2 to %c64 step %3 {
        memref.store %cst, %view_15[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %6 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %arg0[%6, 0] [64, 64] [1, 1] : memref<8192x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
      %alloca = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %alloca_16 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %c15104 = arith.constant 15104 : index
      %view_17 = memref.view %arg14[%c15104][] : memref<72448xi8, 3> to memref<8192xi8, 3>
      %c15104_18 = arith.constant 15104 : index
      %7 = arith.addi %c0, %c15104_18 : index
      %view_19 = memref.view %arg14[%7][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      %8 = affine.apply #map1()[%thread_id_x, %thread_id_y]
      %9 = affine.apply #map2()[%thread_id_x]
      %10 = arith.cmpi ult, %8, %c256 : index
      scf.if %10 {
        %23 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_19[%23, %9] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst : f32) outs(%subview_49 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      %11 = arith.cmpi ult, %8, %c32 : index
      %view_20 = memref.view %arg14[%c0][] : memref<72448xi8, 3> to memref<64x4xf32, 3>
      scf.if %11 {
        %23 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %alloca_16[0, %23] [64, 4] [1, 1] : memref<64x128xf16> to memref<64x4xf16, strided<[128, 1], offset: ?>>
        %24 = arith.cmpi ult, %8, %c16 : index
        scf.if %24 {
          %25 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %26 = affine.apply #map6()[%thread_id_x]
          %subview_50 = memref.subview %subview_49[%25, %26] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>> to memref<8x2xf16, strided<[128, 1], offset: ?>>
          %subview_51 = memref.subview %view_20[%25, %26] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_50 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_51 : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
          ^bb0(%in: f16, %out: f32):
            %27 = arith.extf %in : f16 to f32
            linalg.yield %27 : f32
          }
        }
      }
      %12 = arith.cmpi ult, %8, %c8 : index
      gpu.barrier {welder.keep_barrier}
      %alloca_21 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf32>
      %13 = affine.apply #map8()[%thread_id_x, %thread_id_y]
      %14 = affine.apply #map9()[%thread_id_x]
      %subview_22 = memref.subview %alloca[%13, %14] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_23 = memref.subview %alloca_21[%13, %14] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_22 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_23 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
      ^bb0(%in: f16, %out: f32):
        %23 = arith.extf %in : f16 to f32
        linalg.yield %23 : f32
      }
      %alloca_24 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %c23296 = arith.constant 23296 : index
      %view_25 = memref.view %arg14[%c23296][] : memref<72448xi8, 3> to memref<16384xi8, 3>
      %c23296_26 = arith.constant 23296 : index
      %15 = arith.addi %c0, %c23296_26 : index
      %view_27 = memref.view %arg14[%15][] : memref<72448xi8, 3> to memref<64x128xf16, 3>
      %thread_id_z = gpu.thread_id  z
      %block_dim_z = gpu.block_dim  z
      %16 = arith.muli %thread_id_z, %3 : index
      %17 = arith.addi %2, %16 : index
      %18 = arith.muli %3, %block_dim_z : index
      scf.for %arg15 = %17 to %c8192 step %18 {
        %23 = arith.remsi %arg15, %c128 : index
        %24 = arith.divsi %arg15, %c128 : index
        memref.store %cst_1, %view_27[%24, %23] : memref<64x128xf16, 3>
      }
      gpu.barrier {welder.keep_barrier}
      scf.for %arg15 = %c0 to %c64 step %c32 {
        %subview_49 = memref.subview %subview[0, %arg15] [64, 32] [1, 1] : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        %subview_50 = memref.subview %arg1[%arg15, 0] [32, 128] [1, 1] : memref<64x128xf16, strided<[?, ?], offset: ?>> to memref<32x128xf16, strided<[?, ?], offset: ?>>
        %c1024_51 = arith.constant 1024 : index
        %23 = arith.addi %c0, %c1024_51 : index
        %view_52 = memref.view %arg14[%23][] : memref<72448xi8, 3> to memref<64x32xf16, 3>
        %c5120_53 = arith.constant 5120 : index
        %24 = arith.addi %c0, %c5120_53 : index
        %view_54 = memref.view %arg14[%24][] : memref<72448xi8, 3> to memref<32x128xf16, 3>
        %25 = arith.cmpi ult, %8, %c128 : index
        scf.if %25 {
          %26 = affine.apply #map10()[%thread_id_x, %thread_id_y]
          %27 = affine.apply #map11()[%thread_id_x]
          %subview_58 = memref.subview %subview_49[%26, %27] [4, 4] [1, 1] : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_59 = memref.subview %view_52[%26, %27] [4, 4] [1, 1] : memref<64x32xf16, 3> to memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
          linalg.copy ins(%subview_58 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_59 : memref<4x4xf16, strided<[32, 1], offset: ?>, 3>)
        }
        gpu.barrier
        scf.if %10 {
          %26 = affine.apply #map12()[%thread_id_x, %thread_id_y]
          %27 = affine.apply #map13()[%thread_id_x]
          %subview_58 = memref.subview %subview_50[%26, %27] [4, 4] [1, 1] : memref<32x128xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_59 = memref.subview %view_54[%26, %27] [4, 4] [1, 1] : memref<32x128xf16, 3> to memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
          linalg.copy ins(%subview_58 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_59 : memref<4x4xf16, strided<[128, 1], offset: ?>, 3>)
        }
        gpu.barrier
        %subview_55 = memref.subview %view_52[%13, 0] [8, 32] [1, 1] : memref<64x32xf16, 3> to memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
        %subview_56 = memref.subview %view_54[0, %14] [32, 2] [1, 1] : memref<32x128xf16, 3> to memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
        %subview_57 = memref.subview %view_27[%13, %14] [8, 2] [1, 1] : memref<64x128xf16, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
        linalg.matmul {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 1 : i64, welder.target} ins(%subview_55, %subview_56 : memref<8x32xf16, strided<[32, 1], offset: ?>, 3>, memref<32x2xf16, strided<[128, 1], offset: ?>, 3>) outs(%subview_57 : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>)
        gpu.barrier
      }
      %c15104_28 = arith.constant 15104 : index
      %view_29 = memref.view %arg14[%c15104_28][] : memref<72448xi8, 3> to memref<8192xi8, 3>
      %c15104_30 = arith.constant 15104 : index
      %19 = arith.addi %c0, %c15104_30 : index
      %view_31 = memref.view %arg14[%19][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      scf.if %10 {
        %23 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_31[%23, %9] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst : f32) outs(%subview_49 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      %c13312_32 = arith.constant 13312 : index
      %20 = arith.addi %c0, %c13312_32 : index
      %view_33 = memref.view %arg14[%20][] : memref<72448xi8, 3> to memref<64x4xf32, 3>
      scf.if %11 {
        %23 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_27[0, %23] [64, 4] [1, 1] : memref<64x128xf16, 3> to memref<64x4xf16, strided<[128, 1], offset: ?>, 3>
        %24 = arith.cmpi ult, %8, %c16 : index
        scf.if %24 {
          %25 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %26 = affine.apply #map6()[%thread_id_x]
          %subview_50 = memref.subview %subview_49[%25, %26] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
          %subview_51 = memref.subview %view_33[%25, %26] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_50 : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>) outs(%subview_51 : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
          ^bb0(%in: f16, %out: f32):
            %27 = arith.extf %in : f16 to f32
            linalg.yield %27 : f32
          }
        }
      }
      gpu.barrier
      scf.if %11 {
        %subview_49 = memref.subview %view_31[0, %8] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "reduction"]} ins(%view_33 : memref<64x4xf32, 3>) outs(%subview_49 : memref<64xf32, strided<[32], offset: ?>, 3>) attrs =  {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 4 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %23 = arith.maximumf %in, %out : f32
          linalg.yield %23 : f32
        }
      }
      gpu.barrier
      scf.if %12 {
        %23 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_31[%23, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_50 = memref.subview %view_12[%23] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        linalg.reduce ins(%subview_49 : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>) outs(%subview_50 : memref<8xf32, strided<[1], offset: ?>, 3>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %24 = arith.maximumf %in, %init : f32
            linalg.yield %24 : f32
          }
      }
      gpu.barrier {welder.keep_barrier}
      %alloca_34 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf32>
      %subview_35 = memref.subview %alloca_24[%13, %14] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_36 = memref.subview %alloca_34[%13, %14] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_35 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_36 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
      ^bb0(%in: f16, %out: f32):
        %23 = arith.extf %in : f16 to f32
        linalg.yield %23 : f32
      }
      %c39680 = arith.constant 39680 : index
      %view_37 = memref.view %arg14[%c39680][] : memref<72448xi8, 3> to memref<32768xi8, 3>
      %c39680_38 = arith.constant 39680 : index
      %21 = arith.addi %c0, %c39680_38 : index
      %view_39 = memref.view %arg14[%21][] : memref<72448xi8, 3> to memref<64x128xf32, 3>
      %subview_40 = memref.subview %view_12[%13] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_41 = memref.subview %view_39[%13, %14] [8, 2] [1, 1] : memref<64x128xf32, 3> to memref<8x2xf32, strided<[128, 1], offset: ?>, 3>
      linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_36, %subview_40 : memref<8x2xf32, strided<[128, 1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>, 3>) outs(%subview_41 : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 5 : i64} {
      ^bb0(%in: f32, %in_49: f32, %out: f32):
        %23 = arith.subf %in, %in_49 : f32
        %24 = math.exp %23 : f32
        linalg.yield %24 : f32
      }
      gpu.barrier
      %c5120_42 = arith.constant 5120 : index
      %view_43 = memref.view %arg14[%c5120_42][] : memref<72448xi8, 3> to memref<8192xi8, 3>
      %c5120_44 = arith.constant 5120 : index
      %22 = arith.addi %c0, %c5120_44 : index
      %view_45 = memref.view %arg14[%22][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      scf.if %10 {
        %23 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_45[%23, %9] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst_0 : f32) outs(%subview_49 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      scf.if %11 {
        %23 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_39[0, %23] [64, 4] [1, 1] : memref<64x128xf32, 3> to memref<64x4xf32, strided<[128, 1], offset: ?>, 3>
        %subview_50 = memref.subview %view_45[0, %8] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "reduction"]} ins(%subview_49 : memref<64x4xf32, strided<[128, 1], offset: ?>, 3>) outs(%subview_50 : memref<64xf32, strided<[32], offset: ?>, 3>) attrs =  {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 7 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %24 = arith.addf %in, %out : f32
          linalg.yield %24 : f32
        }
      }
      gpu.barrier
      scf.if %12 {
        %23 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %subview_49 = memref.subview %view_45[%23, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_50 = memref.subview %view_9[%23] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        linalg.reduce ins(%subview_49 : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>) outs(%subview_50 : memref<8xf32, strided<[1], offset: ?>, 3>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %24 = arith.addf %in, %init : f32
            linalg.yield %24 : f32
          }
      }
      gpu.barrier {welder.keep_barrier}
      %subview_46 = memref.subview %memref[%6, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_47 = memref.subview %view_9[%13] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_48 = memref.subview %subview_46[%13, %14] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_41, %subview_47 : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>, memref<8xf32, strided<[1], offset: ?>, 3>) outs(%subview_48 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_root = 0 : i32, welder.node_id = 8 : i64} {
      ^bb0(%in: f32, %in_49: f32, %out: f32):
        %23 = arith.divf %in, %in_49 : f32
        linalg.yield %23 : f32
      }
      gpu.barrier
      gpu.terminator
    }
    return %memref : memref<8192x128xf32>
  }
}


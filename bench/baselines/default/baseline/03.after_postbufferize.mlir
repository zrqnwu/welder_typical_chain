#map = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 64 + s2 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 4)>
#map3 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
#map4 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 32) * 4)>
#map5 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 32) * 128)>
#map6 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 8)>
#map7 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 64) * 128)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 8192 + s2 * 8192)>
#map10 = affine_map<()[s0] -> (s0 mod 64)>
#map11 = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128)>
#map12 = affine_map<(d0, d1) -> (d0)>
#map13 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 512)>
module {
  func.func @main(%arg0: memref<8192x64xf16, strided<[?, ?], offset: ?>>, %arg1: memref<64x128xf16, strided<[?, ?], offset: ?>>) -> memref<8192x128xf32> {
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
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8192x128xf16>
    linalg.fill {welder.node_id = 0 : i64} ins(%cst_1 : f16) outs(%alloc : memref<8192x128xf16>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %alloc_8 = memref.alloc() : memref<4096xi8, 3>
      %alloc_9 = memref.alloc() : memref<8192xi8, 3>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %1 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %2 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %arg0[%1, 0] [64, 64] [1, 1] : memref<8192x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
      %subview_10 = memref.subview %alloc[%2, 0] [64, 128] [1, 1] : memref<8192x128xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
      memref.copy %subview_10, %alloc_11 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
      scf.for %arg14 = %c0 to %c64 step %c32 {
        %subview_16 = memref.subview %subview[0, %arg14] [64, 32] [1, 1] : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        %subview_17 = memref.subview %arg1[%arg14, 0] [32, 128] [1, 1] : memref<64x128xf16, strided<[?, ?], offset: ?>> to memref<32x128xf16, strided<[?, ?], offset: ?>>
        %view = memref.view %alloc_8[%c0][] : memref<4096xi8, 3> to memref<64x32xf16, 3>
        %view_18 = memref.view %alloc_9[%c0][] : memref<8192xi8, 3> to memref<32x128xf16, 3>
        %thread_id_x_19 = gpu.thread_id  x
        %thread_id_y_20 = gpu.thread_id  y
        %5 = affine.apply #map1()[%thread_id_x_19, %thread_id_y_20]
        %6 = arith.cmpi ult, %5, %c128 : index
        scf.if %6 {
          %11 = affine.apply #map2()[%thread_id_x_19, %thread_id_y_20]
          %12 = affine.apply #map3()[%thread_id_x_19]
          %subview_28 = memref.subview %subview_16[%11, %12] [4, 4] [1, 1] : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_29 = memref.subview %view[%11, %12] [4, 4] [1, 1] : memref<64x32xf16, 3> to memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
          linalg.copy ins(%subview_28 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_29 : memref<4x4xf16, strided<[32, 1], offset: ?>, 3>)
        }
        gpu.barrier
        %thread_id_x_21 = gpu.thread_id  x
        %thread_id_y_22 = gpu.thread_id  y
        %7 = affine.apply #map1()[%thread_id_x_21, %thread_id_y_22]
        %8 = arith.cmpi ult, %7, %c256 : index
        scf.if %8 {
          %11 = affine.apply #map4()[%thread_id_x_21, %thread_id_y_22]
          %12 = affine.apply #map5()[%thread_id_x_21]
          %subview_28 = memref.subview %subview_17[%11, %12] [4, 4] [1, 1] : memref<32x128xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_29 = memref.subview %view_18[%11, %12] [4, 4] [1, 1] : memref<32x128xf16, 3> to memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
          linalg.copy ins(%subview_28 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_29 : memref<4x4xf16, strided<[128, 1], offset: ?>, 3>)
        }
        gpu.barrier
        %thread_id_x_23 = gpu.thread_id  x
        %thread_id_y_24 = gpu.thread_id  y
        %9 = affine.apply #map6()[%thread_id_x_23, %thread_id_y_24]
        %10 = affine.apply #map7()[%thread_id_x_23]
        %subview_25 = memref.subview %view[%9, 0] [8, 32] [1, 1] : memref<64x32xf16, 3> to memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
        %subview_26 = memref.subview %view_18[0, %10] [32, 2] [1, 1] : memref<32x128xf16, 3> to memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
        %subview_27 = memref.subview %alloc_11[%9, %10] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
        linalg.matmul {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 1 : i64, welder.target} ins(%subview_25, %subview_26 : memref<8x32xf16, strided<[32, 1], offset: ?>, 3>, memref<32x2xf16, strided<[128, 1], offset: ?>, 3>) outs(%subview_27 : memref<8x2xf16, strided<[128, 1], offset: ?>>)
        gpu.barrier
      }
      %subview_12 = memref.subview %alloc_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %3 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %4 = affine.apply #map7()[%thread_id_x]
      %subview_13 = memref.subview %alloc_11[%3, %4] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_14 = memref.subview %subview_12[%3, %4] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%subview_13 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_14 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_root = 0 : i32, welder.node_id = 2 : i64} {
      ^bb0(%in: f16, %out: f32):
        %5 = arith.extf %in : f16 to f32
        linalg.yield %5 : f32
      }
      gpu.barrier
      %subview_15 = memref.subview %alloc_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview_12, %subview_15 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      gpu.terminator
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8192xf32>
    linalg.fill {welder.node_id = 3 : i64} ins(%cst_2 : f32) outs(%alloc_4 : memref<8192xf32>)
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map9()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %alloc_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_4[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<16384xi8, 3>
      %c0_10 = arith.constant 0 : index
      %view = memref.view %alloc_9[%c0_10][] : memref<16384xi8, 3> to memref<64x64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map10()[%thread_id_x]
      %2 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %subview_11 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      linalg.fill ins(%cst : f32) outs(%subview_11 : memref<8x1xf32, strided<[64, 1], offset: ?>, 3>)
      %subview_12 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      memref.copy %subview_11, %subview_12 : memref<8x1xf32, strided<[64, 1], offset: ?>, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      gpu.barrier
      %thread_id_x_13 = gpu.thread_id  x
      %thread_id_y_14 = gpu.thread_id  y
      %3 = affine.apply #map1()[%thread_id_x_13, %thread_id_y_14]
      %4 = affine.apply #map1()[%thread_id_x_13, %thread_id_y_14]
      %5 = arith.cmpi ult, %3, %c64 : index
      scf.if %5 {
        %8 = affine.apply #map11()[%thread_id_x_13, %thread_id_y_14]
        %subview_18 = memref.subview %subview[0, %8] [64, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x2xf32, strided<[128, 1], offset: ?>>
        %subview_19 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map8, #map12], iterator_types = ["parallel", "reduction"]} ins(%subview_18 : memref<64x2xf32, strided<[128, 1], offset: ?>>) outs(%subview_19 : memref<64xf32, strided<[64], offset: ?>, 3>) attrs =  {welder.kernel_id = 1 : i32, welder.kernel_root = 1 : i32, welder.node_id = 4 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.maximumf %in, %out : f32
          linalg.yield %9 : f32
        }
        %subview_20 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        memref.copy %subview_19, %subview_20 : memref<64xf32, strided<[64], offset: ?>, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
      }
      gpu.barrier
      %thread_id_x_15 = gpu.thread_id  x
      %thread_id_y_16 = gpu.thread_id  y
      %6 = affine.apply #map1()[%thread_id_x_15, %thread_id_y_16]
      %7 = arith.cmpi ult, %6, %c8 : index
      scf.if %7 {
        %8 = affine.apply #map13()[%thread_id_x_15, %thread_id_y_16]
        %subview_18 = memref.subview %view[%8, 0] [8, 64] [1, 1] : memref<64x64xf32, 3> to memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
        %subview_19 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        linalg.reduce ins(%subview_18 : memref<8x64xf32, strided<[64, 1], offset: ?>, 3>) outs(%subview_19 : memref<8xf32, strided<[1], offset: ?>>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %9 = arith.maximumf %in, %init : f32
            linalg.yield %9 : f32
          }
        %subview_20 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        memref.copy %subview_19, %subview_20 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      }
      gpu.barrier {welder.keep_barrier}
      %subview_17 = memref.subview %alloc_4[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      memref.copy %subview_8, %subview_17 : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
      gpu.terminator
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %alloc_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_4[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %2 = affine.apply #map7()[%thread_id_x]
      %subview_10 = memref.subview %subview[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      %subview_11 = memref.subview %subview_8[%1] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %subview_9[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel"]} ins(%subview_10, %subview_11 : memref<8x2xf32, strided<[128, 1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>>) outs(%subview_12 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 2 : i32, welder.kernel_root = 2 : i32, welder.node_id = 5 : i64} {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %3 = arith.subf %in, %in_14 : f32
        %4 = math.exp %3 : f32
        linalg.yield %4 : f32
      }
      gpu.barrier
      %subview_13 = memref.subview %alloc_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview_9, %subview_13 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      gpu.terminator
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8192xf32>
    linalg.fill {welder.node_id = 6 : i64} ins(%cst_0 : f32) outs(%alloc_6 : memref<8192xf32>)
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map9()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %alloc_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_6[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<16384xi8, 3>
      %c0_10 = arith.constant 0 : index
      %view = memref.view %alloc_9[%c0_10][] : memref<16384xi8, 3> to memref<64x64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map10()[%thread_id_x]
      %2 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %subview_11 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      linalg.fill ins(%cst_0 : f32) outs(%subview_11 : memref<8x1xf32, strided<[64, 1], offset: ?>, 3>)
      %subview_12 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      memref.copy %subview_11, %subview_12 : memref<8x1xf32, strided<[64, 1], offset: ?>, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      gpu.barrier
      %thread_id_x_13 = gpu.thread_id  x
      %thread_id_y_14 = gpu.thread_id  y
      %3 = affine.apply #map1()[%thread_id_x_13, %thread_id_y_14]
      %4 = affine.apply #map1()[%thread_id_x_13, %thread_id_y_14]
      %5 = arith.cmpi ult, %3, %c64 : index
      scf.if %5 {
        %8 = affine.apply #map11()[%thread_id_x_13, %thread_id_y_14]
        %subview_18 = memref.subview %subview[0, %8] [64, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x2xf32, strided<[128, 1], offset: ?>>
        %subview_19 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map8, #map12], iterator_types = ["parallel", "reduction"]} ins(%subview_18 : memref<64x2xf32, strided<[128, 1], offset: ?>>) outs(%subview_19 : memref<64xf32, strided<[64], offset: ?>, 3>) attrs =  {welder.kernel_id = 3 : i32, welder.kernel_root = 3 : i32, welder.node_id = 7 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.addf %in, %out : f32
          linalg.yield %9 : f32
        }
        %subview_20 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        memref.copy %subview_19, %subview_20 : memref<64xf32, strided<[64], offset: ?>, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
      }
      gpu.barrier
      %thread_id_x_15 = gpu.thread_id  x
      %thread_id_y_16 = gpu.thread_id  y
      %6 = affine.apply #map1()[%thread_id_x_15, %thread_id_y_16]
      %7 = arith.cmpi ult, %6, %c8 : index
      scf.if %7 {
        %8 = affine.apply #map13()[%thread_id_x_15, %thread_id_y_16]
        %subview_18 = memref.subview %view[%8, 0] [8, 64] [1, 1] : memref<64x64xf32, 3> to memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
        %subview_19 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        linalg.reduce ins(%subview_18 : memref<8x64xf32, strided<[64, 1], offset: ?>, 3>) outs(%subview_19 : memref<8xf32, strided<[1], offset: ?>>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %9 = arith.addf %in, %init : f32
            linalg.yield %9 : f32
          }
        %subview_20 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        memref.copy %subview_19, %subview_20 : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      }
      gpu.barrier {welder.keep_barrier}
      %subview_17 = memref.subview %alloc_6[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      memref.copy %subview_8, %subview_17 : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
      gpu.terminator
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %alloc_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_6[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_7[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %2 = affine.apply #map7()[%thread_id_x]
      %subview_10 = memref.subview %subview[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      %subview_11 = memref.subview %subview_8[%1] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %subview_9[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel"]} ins(%subview_10, %subview_11 : memref<8x2xf32, strided<[128, 1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>>) outs(%subview_12 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 4 : i32, welder.kernel_root = 4 : i32, welder.node_id = 8 : i64} {
      ^bb0(%in: f32, %in_14: f32, %out: f32):
        %3 = arith.divf %in, %in_14 : f32
        linalg.yield %3 : f32
      }
      gpu.barrier
      %subview_13 = memref.subview %alloc_7[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview_9, %subview_13 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      gpu.terminator
    }
    return %alloc_7 : memref<8192x128xf32>
  }
}


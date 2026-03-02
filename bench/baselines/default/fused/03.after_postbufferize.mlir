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
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1024xi8, 3>
      %alloc_4 = memref.alloc() : memref<4096xi8, 3>
      %alloc_5 = memref.alloc() : memref<8192xi8, 3>
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1024xi8, 3>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256xi8, 3>
      %view = memref.view %alloc_7[%c0][] : memref<256xi8, 3> to memref<64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %0 = arith.muli %thread_id_y, %block_dim_x : index
      %1 = arith.addi %thread_id_x, %0 : index
      %2 = arith.muli %block_dim_x, %block_dim_y : index
      scf.for %arg14 = %1 to %c64 step %2 {
        memref.store %cst_0, %view[%arg14] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<256xi8, 3>
      %view_9 = memref.view %alloc_8[%c0][] : memref<256xi8, 3> to memref<64xf32, 3>
      scf.for %arg14 = %1 to %c64 step %2 {
        memref.store %cst, %view_9[%arg14] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<256xi8, 3>
      %view_11 = memref.view %alloc_10[%c0][] : memref<256xi8, 3> to memref<64xf32, 3>
      scf.for %arg14 = %1 to %c64 step %2 {
        memref.store %cst, %view_11[%arg14] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %3 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %arg0[%3, 0] [64, 64] [1, 1] : memref<8192x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
      %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
      %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<8192xi8, 3>
      %view_15 = memref.view %alloc_14[%c0][] : memref<8192xi8, 3> to memref<64x32xf32, 3>
      %4 = affine.apply #map1()[%thread_id_x, %thread_id_y]
      %5 = affine.apply #map2()[%thread_id_x]
      %6 = arith.cmpi ult, %4, %c256 : index
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_15[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst : f32) outs(%subview_38 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      %7 = arith.cmpi ult, %4, %c32 : index
      %view_16 = memref.view %alloc_3[%c0][] : memref<1024xi8, 3> to memref<64x4xf32, 3>
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %alloc_13[0, %14] [64, 4] [1, 1] : memref<64x128xf16> to memref<64x4xf16, strided<[128, 1], offset: ?>>
        %15 = arith.cmpi ult, %4, %c16 : index
        scf.if %15 {
          %16 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %17 = affine.apply #map6()[%thread_id_x]
          %subview_39 = memref.subview %subview_38[%16, %17] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>> to memref<8x2xf16, strided<[128, 1], offset: ?>>
          %subview_40 = memref.subview %view_16[%16, %17] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_39 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_40 : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
          ^bb0(%in: f16, %out: f32):
            %18 = arith.extf %in : f16 to f32
            linalg.yield %18 : f32
          }
        }
      }
      gpu.barrier
      %8 = arith.cmpi ult, %4, %c8 : index
      gpu.barrier {welder.keep_barrier}
      %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
      %9 = affine.apply #map8()[%thread_id_x, %thread_id_y]
      %10 = affine.apply #map9()[%thread_id_x]
      %subview_18 = memref.subview %alloc_12[%9, %10] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_19 = memref.subview %alloc_17[%9, %10] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_18 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_19 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
      ^bb0(%in: f16, %out: f32):
        %14 = arith.extf %in : f16 to f32
        linalg.yield %14 : f32
      }
      gpu.barrier
      %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
      %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<16384xi8, 3>
      %view_22 = memref.view %alloc_21[%c0][] : memref<16384xi8, 3> to memref<64x128xf16, 3>
      %thread_id_z = gpu.thread_id  z
      %block_dim_z = gpu.block_dim  z
      %11 = arith.muli %thread_id_z, %2 : index
      %12 = arith.addi %1, %11 : index
      %13 = arith.muli %2, %block_dim_z : index
      scf.for %arg14 = %12 to %c8192 step %13 {
        %14 = arith.remsi %arg14, %c128 : index
        %15 = arith.divsi %arg14, %c128 : index
        memref.store %cst_1, %view_22[%15, %14] : memref<64x128xf16, 3>
      }
      gpu.barrier {welder.keep_barrier}
      scf.for %arg14 = %c0 to %c64 step %c32 {
        %subview_38 = memref.subview %subview[0, %arg14] [64, 32] [1, 1] : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        %subview_39 = memref.subview %arg1[%arg14, 0] [32, 128] [1, 1] : memref<64x128xf16, strided<[?, ?], offset: ?>> to memref<32x128xf16, strided<[?, ?], offset: ?>>
        %view_40 = memref.view %alloc_4[%c0][] : memref<4096xi8, 3> to memref<64x32xf16, 3>
        %view_41 = memref.view %alloc_5[%c0][] : memref<8192xi8, 3> to memref<32x128xf16, 3>
        %14 = arith.cmpi ult, %4, %c128 : index
        scf.if %14 {
          %15 = affine.apply #map10()[%thread_id_x, %thread_id_y]
          %16 = affine.apply #map11()[%thread_id_x]
          %subview_45 = memref.subview %subview_38[%15, %16] [4, 4] [1, 1] : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_46 = memref.subview %view_40[%15, %16] [4, 4] [1, 1] : memref<64x32xf16, 3> to memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
          linalg.copy ins(%subview_45 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_46 : memref<4x4xf16, strided<[32, 1], offset: ?>, 3>)
        }
        gpu.barrier
        scf.if %6 {
          %15 = affine.apply #map12()[%thread_id_x, %thread_id_y]
          %16 = affine.apply #map13()[%thread_id_x]
          %subview_45 = memref.subview %subview_39[%15, %16] [4, 4] [1, 1] : memref<32x128xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_46 = memref.subview %view_41[%15, %16] [4, 4] [1, 1] : memref<32x128xf16, 3> to memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
          linalg.copy ins(%subview_45 : memref<4x4xf16, strided<[?, ?], offset: ?>>) outs(%subview_46 : memref<4x4xf16, strided<[128, 1], offset: ?>, 3>)
        }
        gpu.barrier
        %subview_42 = memref.subview %view_40[%9, 0] [8, 32] [1, 1] : memref<64x32xf16, 3> to memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
        %subview_43 = memref.subview %view_41[0, %10] [32, 2] [1, 1] : memref<32x128xf16, 3> to memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
        %subview_44 = memref.subview %view_22[%9, %10] [8, 2] [1, 1] : memref<64x128xf16, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
        linalg.matmul {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 1 : i64, welder.target} ins(%subview_42, %subview_43 : memref<8x32xf16, strided<[32, 1], offset: ?>, 3>, memref<32x2xf16, strided<[128, 1], offset: ?>, 3>) outs(%subview_44 : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>)
        gpu.barrier
      }
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<8192xi8, 3>
      %view_24 = memref.view %alloc_23[%c0][] : memref<8192xi8, 3> to memref<64x32xf32, 3>
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_24[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst : f32) outs(%subview_38 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      %view_25 = memref.view %alloc_6[%c0][] : memref<1024xi8, 3> to memref<64x4xf32, 3>
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_22[0, %14] [64, 4] [1, 1] : memref<64x128xf16, 3> to memref<64x4xf16, strided<[128, 1], offset: ?>, 3>
        %15 = arith.cmpi ult, %4, %c16 : index
        scf.if %15 {
          %16 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %17 = affine.apply #map6()[%thread_id_x]
          %subview_39 = memref.subview %subview_38[%16, %17] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
          %subview_40 = memref.subview %view_25[%16, %17] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_39 : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>) outs(%subview_40 : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
          ^bb0(%in: f16, %out: f32):
            %18 = arith.extf %in : f16 to f32
            linalg.yield %18 : f32
          }
        }
      }
      gpu.barrier
      scf.if %7 {
        %subview_38 = memref.subview %view_24[0, %4] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "reduction"]} ins(%view_25 : memref<64x4xf32, 3>) outs(%subview_38 : memref<64xf32, strided<[32], offset: ?>, 3>) attrs =  {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 4 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %14 = arith.maximumf %in, %out : f32
          linalg.yield %14 : f32
        }
      }
      gpu.barrier
      scf.if %8 {
        %14 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_24[%14, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_39 = memref.subview %view_9[%14] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        linalg.reduce ins(%subview_38 : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>) outs(%subview_39 : memref<8xf32, strided<[1], offset: ?>, 3>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %15 = arith.maximumf %in, %init : f32
            linalg.yield %15 : f32
          }
      }
      gpu.barrier {welder.keep_barrier}
      %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
      %subview_27 = memref.subview %alloc_20[%9, %10] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_28 = memref.subview %alloc_26[%9, %10] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_27 : memref<8x2xf16, strided<[128, 1], offset: ?>>) outs(%subview_28 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 2 : i64} {
      ^bb0(%in: f16, %out: f32):
        %14 = arith.extf %in : f16 to f32
        linalg.yield %14 : f32
      }
      gpu.barrier
      %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<32768xi8, 3>
      %view_30 = memref.view %alloc_29[%c0][] : memref<32768xi8, 3> to memref<64x128xf32, 3>
      %subview_31 = memref.subview %view_9[%9] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_32 = memref.subview %view_30[%9, %10] [8, 2] [1, 1] : memref<64x128xf32, 3> to memref<8x2xf32, strided<[128, 1], offset: ?>, 3>
      linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_28, %subview_31 : memref<8x2xf32, strided<[128, 1], offset: ?>>, memref<8xf32, strided<[1], offset: ?>, 3>) outs(%subview_32 : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 5 : i64} {
      ^bb0(%in: f32, %in_38: f32, %out: f32):
        %14 = arith.subf %in, %in_38 : f32
        %15 = math.exp %14 : f32
        linalg.yield %15 : f32
      }
      gpu.barrier
      %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<8192xi8, 3>
      %view_34 = memref.view %alloc_33[%c0][] : memref<8192xi8, 3> to memref<64x32xf32, 3>
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_34[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        linalg.fill ins(%cst_0 : f32) outs(%subview_38 : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>)
      }
      gpu.barrier
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_30[0, %14] [64, 4] [1, 1] : memref<64x128xf32, 3> to memref<64x4xf32, strided<[128, 1], offset: ?>, 3>
        %subview_39 = memref.subview %view_34[0, %4] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        linalg.generic {indexing_maps = [#map7, #map14], iterator_types = ["parallel", "reduction"]} ins(%subview_38 : memref<64x4xf32, strided<[128, 1], offset: ?>, 3>) outs(%subview_39 : memref<64xf32, strided<[32], offset: ?>, 3>) attrs =  {welder.kernel_id = 0 : i32, welder.kernel_producer = 0 : i32, welder.node_id = 7 : i64, welder.row_reduction} {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.addf %in, %out : f32
          linalg.yield %15 : f32
        }
      }
      gpu.barrier
      scf.if %8 {
        %14 = affine.apply #map15()[%thread_id_x, %thread_id_y]
        %subview_38 = memref.subview %view_34[%14, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_39 = memref.subview %view[%14] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        linalg.reduce ins(%subview_38 : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>) outs(%subview_39 : memref<8xf32, strided<[1], offset: ?>, 3>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %15 = arith.addf %in, %init : f32
            linalg.yield %15 : f32
          }
      }
      gpu.barrier {welder.keep_barrier}
      %subview_35 = memref.subview %alloc[%3, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_36 = memref.subview %view[%9] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_37 = memref.subview %subview_35[%9, %10] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel"]} ins(%subview_32, %subview_36 : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>, memref<8xf32, strided<[1], offset: ?>, 3>) outs(%subview_37 : memref<8x2xf32, strided<[128, 1], offset: ?>>) attrs =  {welder.elementwise, welder.elementwise_nd, welder.kernel_id = 0 : i32, welder.kernel_root = 0 : i32, welder.node_id = 8 : i64} {
      ^bb0(%in: f32, %in_38: f32, %out: f32):
        %14 = arith.divf %in, %in_38 : f32
        linalg.yield %14 : f32
      }
      gpu.barrier
      gpu.terminator
    }
    return %alloc : memref<8192x128xf32>
  }
}


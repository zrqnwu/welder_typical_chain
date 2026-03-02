#map = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 64 + s2 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 4)>
#map3 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
#map4 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 32) * 4)>
#map5 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 32) * 128)>
#map6 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 8)>
#map7 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 64) * 128)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 8192 + s2 * 8192)>
#map9 = affine_map<()[s0] -> (s0 mod 64)>
#map10 = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 128)>
#map11 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 512)>
module {
  func.func @main(%arg0: memref<8192x64xf16, strided<[?, ?], offset: ?>>, %arg1: memref<64x128xf16, strided<[?, ?], offset: ?>>) -> memref<8192x128xf32> {
    %c4 = arith.constant 4 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %c2 = arith.constant 2 : index
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
    %memref = gpu.alloc  host_shared () : memref<8192x128xf16>
    scf.for %arg2 = %c0 to %c8192 step %c1 {
      scf.for %arg3 = %c0 to %c128 step %c1 {
        memref.store %cst_1, %memref[%arg2, %arg3] : memref<8192x128xf16>
      }
    }
    %memref_3 = gpu.alloc  host_shared () : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) workgroup(%arg14 : memref<12288xi8, 3>) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %1 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %arg0[%1, 0] [64, 64] [1, 1] : memref<8192x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
      %alloca = memref.alloca() {alignment = 64 : i64} : memref<8x2xf16>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          memref.store %cst_1, %alloca[%arg15, %arg16] : memref<8x2xf16>
        }
      }
      scf.for %arg15 = %c0 to %c64 step %c32 {
        %subview_10 = memref.subview %subview[0, %arg15] [64, 32] [1, 1] : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        %subview_11 = memref.subview %arg1[%arg15, 0] [32, 128] [1, 1] : memref<64x128xf16, strided<[?, ?], offset: ?>> to memref<32x128xf16, strided<[?, ?], offset: ?>>
        %view = memref.view %arg14[%c0][] : memref<12288xi8, 3> to memref<64x32xf16, 3>
        %view_12 = memref.view %arg14[%c4096][] : memref<12288xi8, 3> to memref<32x128xf16, 3>
        %thread_id_x_13 = gpu.thread_id  x
        %thread_id_y_14 = gpu.thread_id  y
        %4 = affine.apply #map1()[%thread_id_x_13, %thread_id_y_14]
        %5 = arith.cmpi ult, %4, %c128 : index
        scf.if %5 {
          %10 = affine.apply #map2()[%thread_id_x_13, %thread_id_y_14]
          %11 = affine.apply #map3()[%thread_id_x_13]
          %subview_21 = memref.subview %subview_10[%10, %11] [4, 4] [1, 1] : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_22 = memref.subview %view[%10, %11] [4, 4] [1, 1] : memref<64x32xf16, 3> to memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
          scf.for %arg16 = %c0 to %c4 step %c1 {
            scf.for %arg17 = %c0 to %c4 step %c1 {
              %12 = memref.load %subview_21[%arg16, %arg17] : memref<4x4xf16, strided<[?, ?], offset: ?>>
              memref.store %12, %subview_22[%arg16, %arg17] : memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
            }
          }
        }
        gpu.barrier
        %thread_id_x_15 = gpu.thread_id  x
        %thread_id_y_16 = gpu.thread_id  y
        %6 = affine.apply #map1()[%thread_id_x_15, %thread_id_y_16]
        %7 = arith.cmpi ult, %6, %c256 : index
        scf.if %7 {
          %10 = affine.apply #map4()[%thread_id_x_15, %thread_id_y_16]
          %11 = affine.apply #map5()[%thread_id_x_15]
          %subview_21 = memref.subview %subview_11[%10, %11] [4, 4] [1, 1] : memref<32x128xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_22 = memref.subview %view_12[%10, %11] [4, 4] [1, 1] : memref<32x128xf16, 3> to memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
          scf.for %arg16 = %c0 to %c4 step %c1 {
            scf.for %arg17 = %c0 to %c4 step %c1 {
              %12 = memref.load %subview_21[%arg16, %arg17] : memref<4x4xf16, strided<[?, ?], offset: ?>>
              memref.store %12, %subview_22[%arg16, %arg17] : memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
            }
          }
        }
        gpu.barrier
        %thread_id_x_17 = gpu.thread_id  x
        %thread_id_y_18 = gpu.thread_id  y
        %8 = affine.apply #map6()[%thread_id_x_17, %thread_id_y_18]
        %9 = affine.apply #map7()[%thread_id_x_17]
        %subview_19 = memref.subview %view[%8, 0] [8, 32] [1, 1] : memref<64x32xf16, 3> to memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
        %subview_20 = memref.subview %view_12[0, %9] [32, 2] [1, 1] : memref<32x128xf16, 3> to memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
        scf.for %arg16 = %c0 to %c8 step %c1 {
          scf.for %arg17 = %c0 to %c2 step %c1 {
            scf.for %arg18 = %c0 to %c32 step %c1 {
              %10 = memref.load %subview_19[%arg16, %arg18] : memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
              %11 = memref.load %subview_20[%arg18, %arg17] : memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
              %12 = memref.load %alloca[%arg16, %arg17] : memref<8x2xf16>
              %13 = arith.mulf %10, %11 : f16
              %14 = arith.addf %12, %13 : f16
              memref.store %14, %alloca[%arg16, %arg17] : memref<8x2xf16>
            }
          }
        }
        gpu.barrier
      }
      %subview_8 = memref.subview %memref_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %2 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %3 = affine.apply #map7()[%thread_id_x]
      %subview_9 = memref.subview %subview_8[%2, %3] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %4 = memref.load %alloca[%arg15, %arg16] : memref<8x2xf16>
          %5 = arith.extf %4 : f16 to f32
          memref.store %5, %subview_9[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      gpu.barrier
      gpu.terminator
    }
    %memref_4 = gpu.alloc  host_shared () : memref<8192xf32>
    scf.for %arg2 = %c0 to %c8192 step %c1 {
      memref.store %cst_2, %memref_4[%arg2] : memref<8192xf32>
    }
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) workgroup(%arg14 : memref<16384xi8, 3>) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map8()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %memref_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %memref_4[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %view = memref.view %arg14[%c0][] : memref<16384xi8, 3> to memref<64x64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map9()[%thread_id_x]
      %2 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %subview_9 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c1 step %c1 {
          memref.store %cst, %subview_9[%arg15, %arg16] : memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
        }
      }
      gpu.barrier
      %thread_id_x_10 = gpu.thread_id  x
      %thread_id_y_11 = gpu.thread_id  y
      %3 = affine.apply #map1()[%thread_id_x_10, %thread_id_y_11]
      %4 = affine.apply #map1()[%thread_id_x_10, %thread_id_y_11]
      %5 = arith.cmpi ult, %3, %c64 : index
      scf.if %5 {
        %8 = affine.apply #map10()[%thread_id_x_10, %thread_id_y_11]
        %subview_14 = memref.subview %subview[0, %8] [64, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x2xf32, strided<[128, 1], offset: ?>>
        %subview_15 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c64 step %c1 {
          scf.for %arg16 = %c0 to %c2 step %c1 {
            %9 = memref.load %subview_14[%arg15, %arg16] : memref<64x2xf32, strided<[128, 1], offset: ?>>
            %10 = memref.load %subview_15[%arg15] : memref<64xf32, strided<[64], offset: ?>, 3>
            %11 = arith.maximumf %9, %10 : f32
            memref.store %11, %subview_15[%arg15] : memref<64xf32, strided<[64], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      %thread_id_x_12 = gpu.thread_id  x
      %thread_id_y_13 = gpu.thread_id  y
      %6 = affine.apply #map1()[%thread_id_x_12, %thread_id_y_13]
      %7 = arith.cmpi ult, %6, %c8 : index
      scf.if %7 {
        %8 = affine.apply #map11()[%thread_id_x_12, %thread_id_y_13]
        %subview_14 = memref.subview %view[%8, 0] [8, 64] [1, 1] : memref<64x64xf32, 3> to memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
        %subview_15 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c64 step %c1 {
            %9 = memref.load %subview_14[%arg15, %arg16] : memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
            %10 = memref.load %subview_15[%arg15] : memref<8xf32, strided<[1], offset: ?>>
            %11 = arith.maximumf %9, %10 : f32
            memref.store %11, %subview_15[%arg15] : memref<8xf32, strided<[1], offset: ?>>
          }
        }
      }
      gpu.barrier {welder.keep_barrier}
      gpu.terminator
    }
    %memref_5 = gpu.alloc  host_shared () : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %memref_3[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %memref_4[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview_9 = memref.subview %memref_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %2 = affine.apply #map7()[%thread_id_x]
      %subview_10 = memref.subview %subview[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      %subview_11 = memref.subview %subview_8[%1] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %subview_9[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg14 = %c0 to %c8 step %c1 {
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %3 = memref.load %subview_10[%arg14, %arg15] : memref<8x2xf32, strided<[128, 1], offset: ?>>
          %4 = memref.load %subview_11[%arg14] : memref<8xf32, strided<[1], offset: ?>>
          %5 = arith.subf %3, %4 : f32
          %6 = math.exp %5 : f32
          memref.store %6, %subview_12[%arg14, %arg15] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      gpu.barrier
      gpu.terminator
    }
    %memref_6 = gpu.alloc  host_shared () : memref<8192xf32>
    scf.for %arg2 = %c0 to %c8192 step %c1 {
      memref.store %cst_0, %memref_6[%arg2] : memref<8192xf32>
    }
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) workgroup(%arg14 : memref<16384xi8, 3>) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map8()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %memref_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %memref_6[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %view = memref.view %arg14[%c0][] : memref<16384xi8, 3> to memref<64x64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map9()[%thread_id_x]
      %2 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %subview_9 = memref.subview %view[%2, %1] [8, 1] [1, 1] : memref<64x64xf32, 3> to memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c1 step %c1 {
          memref.store %cst_0, %subview_9[%arg15, %arg16] : memref<8x1xf32, strided<[64, 1], offset: ?>, 3>
        }
      }
      gpu.barrier
      %thread_id_x_10 = gpu.thread_id  x
      %thread_id_y_11 = gpu.thread_id  y
      %3 = affine.apply #map1()[%thread_id_x_10, %thread_id_y_11]
      %4 = affine.apply #map1()[%thread_id_x_10, %thread_id_y_11]
      %5 = arith.cmpi ult, %3, %c64 : index
      scf.if %5 {
        %8 = affine.apply #map10()[%thread_id_x_10, %thread_id_y_11]
        %subview_14 = memref.subview %subview[0, %8] [64, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x2xf32, strided<[128, 1], offset: ?>>
        %subview_15 = memref.subview %view[0, %4] [64, 1] [1, 1] : memref<64x64xf32, 3> to memref<64xf32, strided<[64], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c64 step %c1 {
          scf.for %arg16 = %c0 to %c2 step %c1 {
            %9 = memref.load %subview_14[%arg15, %arg16] : memref<64x2xf32, strided<[128, 1], offset: ?>>
            %10 = memref.load %subview_15[%arg15] : memref<64xf32, strided<[64], offset: ?>, 3>
            %11 = arith.addf %9, %10 : f32
            memref.store %11, %subview_15[%arg15] : memref<64xf32, strided<[64], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      %thread_id_x_12 = gpu.thread_id  x
      %thread_id_y_13 = gpu.thread_id  y
      %6 = affine.apply #map1()[%thread_id_x_12, %thread_id_y_13]
      %7 = arith.cmpi ult, %6, %c8 : index
      scf.if %7 {
        %8 = affine.apply #map11()[%thread_id_x_12, %thread_id_y_13]
        %subview_14 = memref.subview %view[%8, 0] [8, 64] [1, 1] : memref<64x64xf32, 3> to memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
        %subview_15 = memref.subview %subview_8[%8] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c64 step %c1 {
            %9 = memref.load %subview_14[%arg15, %arg16] : memref<8x64xf32, strided<[64, 1], offset: ?>, 3>
            %10 = memref.load %subview_15[%arg15] : memref<8xf32, strided<[1], offset: ?>>
            %11 = arith.addf %9, %10 : f32
            memref.store %11, %subview_15[%arg15] : memref<8xf32, strided<[1], offset: ?>>
          }
        }
      }
      gpu.barrier {welder.keep_barrier}
      gpu.terminator
    }
    %memref_7 = gpu.alloc  host_shared () : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c128, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %memref_5[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_8 = memref.subview %memref_6[%0] [64] [1] : memref<8192xf32> to memref<64xf32, strided<[1], offset: ?>>
      %subview_9 = memref.subview %memref_7[%0, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %1 = affine.apply #map6()[%thread_id_x, %thread_id_y]
      %2 = affine.apply #map7()[%thread_id_x]
      %subview_10 = memref.subview %subview[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      %subview_11 = memref.subview %subview_8[%1] [8] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
      %subview_12 = memref.subview %subview_9[%1, %2] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg14 = %c0 to %c8 step %c1 {
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %3 = memref.load %subview_10[%arg14, %arg15] : memref<8x2xf32, strided<[128, 1], offset: ?>>
          %4 = memref.load %subview_11[%arg14] : memref<8xf32, strided<[1], offset: ?>>
          %5 = arith.divf %3, %4 : f32
          memref.store %5, %subview_12[%arg14, %arg15] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      gpu.barrier
      gpu.terminator
    }
    return %memref_7 : memref<8192x128xf32>
  }
}


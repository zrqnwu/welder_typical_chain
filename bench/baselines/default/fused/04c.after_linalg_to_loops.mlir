#map = affine_map<()[s0, s1, s2] -> (s0 * 64 + s1 * 8192 + s2 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0 mod 32)>
#map3 = affine_map<()[s0, s1] -> (s1 * 16 + (s0 floordiv 32) * 8)>
#map4 = affine_map<()[s0, s1] -> (s0 * 4 + s1 * 256)>
#map5 = affine_map<()[s0, s1] -> (s1 * 256 + (s0 floordiv 2) * 8)>
#map6 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 4)>
#map7 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 8)>
#map8 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 64) * 128)>
#map9 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 4)>
#map10 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
#map11 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 32) * 4)>
#map12 = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 32) * 128)>
#map13 = affine_map<()[s0, s1] -> (s0 * 8 + s1 * 512)>
module {
  func.func @main(%arg0: memref<8192x64xf16, strided<[?, ?], offset: ?>>, %arg1: memref<64x128xf16, strided<[?, ?], offset: ?>>) -> memref<8192x128xf32> {
    %c39680 = arith.constant 39680 : index
    %c13312 = arith.constant 13312 : index
    %c4 = arith.constant 4 : index
    %c5120 = arith.constant 5120 : index
    %c1024 = arith.constant 1024 : index
    %c23296 = arith.constant 23296 : index
    %c2 = arith.constant 2 : index
    %c15104 = arith.constant 15104 : index
    %c14848 = arith.constant 14848 : index
    %c14592 = arith.constant 14592 : index
    %c14336 = arith.constant 14336 : index
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
    %memref = gpu.alloc  host_shared () : memref<8192x128xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c128, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c8, %arg13 = %c1) workgroup(%arg14 : memref<72448xi8, 3>) {
      %view = memref.view %arg14[%c14336][] : memref<72448xi8, 3> to memref<64xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %0 = arith.muli %thread_id_y, %block_dim_x : index
      %1 = arith.addi %thread_id_x, %0 : index
      %2 = arith.muli %block_dim_x, %block_dim_y : index
      scf.for %arg15 = %1 to %c64 step %2 {
        memref.store %cst_0, %view[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %view_2 = memref.view %arg14[%c14592][] : memref<72448xi8, 3> to memref<64xf32, 3>
      scf.for %arg15 = %1 to %c64 step %2 {
        memref.store %cst, %view_2[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %view_3 = memref.view %arg14[%c14848][] : memref<72448xi8, 3> to memref<64xf32, 3>
      scf.for %arg15 = %1 to %c64 step %2 {
        memref.store %cst, %view_3[%arg15] : memref<64xf32, 3>
      }
      gpu.barrier {welder.keep_barrier}
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %3 = affine.apply #map()[%block_id_x, %block_id_y, %block_id_z]
      %subview = memref.subview %arg0[%3, 0] [64, 64] [1, 1] : memref<8192x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
      %alloca = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %alloca_4 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %view_5 = memref.view %arg14[%c15104][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      %4 = affine.apply #map1()[%thread_id_x, %thread_id_y]
      %5 = affine.apply #map2()[%thread_id_x]
      %6 = arith.cmpi ult, %4, %c256 : index
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_5[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c1 step %c1 {
            memref.store %cst, %subview_24[%arg15, %arg16] : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      %7 = arith.cmpi ult, %4, %c32 : index
      %view_6 = memref.view %arg14[%c0][] : memref<72448xi8, 3> to memref<64x4xf32, 3>
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %alloca_4[0, %14] [64, 4] [1, 1] : memref<64x128xf16> to memref<64x4xf16, strided<[128, 1], offset: ?>>
        %15 = arith.cmpi ult, %4, %c16 : index
        scf.if %15 {
          %16 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %17 = affine.apply #map6()[%thread_id_x]
          %subview_25 = memref.subview %subview_24[%16, %17] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>> to memref<8x2xf16, strided<[128, 1], offset: ?>>
          %subview_26 = memref.subview %view_6[%16, %17] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          scf.for %arg15 = %c0 to %c8 step %c1 {
            scf.for %arg16 = %c0 to %c2 step %c1 {
              %18 = memref.load %subview_25[%arg15, %arg16] : memref<8x2xf16, strided<[128, 1], offset: ?>>
              %19 = arith.extf %18 : f16 to f32
              memref.store %19, %subview_26[%arg15, %arg16] : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
            }
          }
        }
      }
      %8 = arith.cmpi ult, %4, %c8 : index
      gpu.barrier {welder.keep_barrier}
      %alloca_7 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf32>
      %9 = affine.apply #map7()[%thread_id_x, %thread_id_y]
      %10 = affine.apply #map8()[%thread_id_x]
      %subview_8 = memref.subview %alloca[%9, %10] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_9 = memref.subview %alloca_7[%9, %10] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %14 = memref.load %subview_8[%arg15, %arg16] : memref<8x2xf16, strided<[128, 1], offset: ?>>
          %15 = arith.extf %14 : f16 to f32
          memref.store %15, %subview_9[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      %alloca_10 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf16>
      %view_11 = memref.view %arg14[%c23296][] : memref<72448xi8, 3> to memref<64x128xf16, 3>
      %thread_id_z = gpu.thread_id  z
      %block_dim_z = gpu.block_dim  z
      %11 = arith.muli %thread_id_z, %2 : index
      %12 = arith.addi %1, %11 : index
      %13 = arith.muli %2, %block_dim_z : index
      scf.for %arg15 = %12 to %c8192 step %13 {
        %14 = arith.remsi %arg15, %c128 : index
        %15 = arith.divsi %arg15, %c128 : index
        memref.store %cst_1, %view_11[%15, %14] : memref<64x128xf16, 3>
      }
      gpu.barrier {welder.keep_barrier}
      scf.for %arg15 = %c0 to %c64 step %c32 {
        %subview_24 = memref.subview %subview[0, %arg15] [64, 32] [1, 1] : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        %subview_25 = memref.subview %arg1[%arg15, 0] [32, 128] [1, 1] : memref<64x128xf16, strided<[?, ?], offset: ?>> to memref<32x128xf16, strided<[?, ?], offset: ?>>
        %view_26 = memref.view %arg14[%c1024][] : memref<72448xi8, 3> to memref<64x32xf16, 3>
        %view_27 = memref.view %arg14[%c5120][] : memref<72448xi8, 3> to memref<32x128xf16, 3>
        %14 = arith.cmpi ult, %4, %c128 : index
        scf.if %14 {
          %15 = affine.apply #map9()[%thread_id_x, %thread_id_y]
          %16 = affine.apply #map10()[%thread_id_x]
          %subview_31 = memref.subview %subview_24[%15, %16] [4, 4] [1, 1] : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_32 = memref.subview %view_26[%15, %16] [4, 4] [1, 1] : memref<64x32xf16, 3> to memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
          scf.for %arg16 = %c0 to %c4 step %c1 {
            scf.for %arg17 = %c0 to %c4 step %c1 {
              %17 = memref.load %subview_31[%arg16, %arg17] : memref<4x4xf16, strided<[?, ?], offset: ?>>
              memref.store %17, %subview_32[%arg16, %arg17] : memref<4x4xf16, strided<[32, 1], offset: ?>, 3>
            }
          }
        }
        gpu.barrier
        scf.if %6 {
          %15 = affine.apply #map11()[%thread_id_x, %thread_id_y]
          %16 = affine.apply #map12()[%thread_id_x]
          %subview_31 = memref.subview %subview_25[%15, %16] [4, 4] [1, 1] : memref<32x128xf16, strided<[?, ?], offset: ?>> to memref<4x4xf16, strided<[?, ?], offset: ?>>
          %subview_32 = memref.subview %view_27[%15, %16] [4, 4] [1, 1] : memref<32x128xf16, 3> to memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
          scf.for %arg16 = %c0 to %c4 step %c1 {
            scf.for %arg17 = %c0 to %c4 step %c1 {
              %17 = memref.load %subview_31[%arg16, %arg17] : memref<4x4xf16, strided<[?, ?], offset: ?>>
              memref.store %17, %subview_32[%arg16, %arg17] : memref<4x4xf16, strided<[128, 1], offset: ?>, 3>
            }
          }
        }
        gpu.barrier
        %subview_28 = memref.subview %view_26[%9, 0] [8, 32] [1, 1] : memref<64x32xf16, 3> to memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
        %subview_29 = memref.subview %view_27[0, %10] [32, 2] [1, 1] : memref<32x128xf16, 3> to memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
        %subview_30 = memref.subview %view_11[%9, %10] [8, 2] [1, 1] : memref<64x128xf16, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
        scf.for %arg16 = %c0 to %c8 step %c1 {
          scf.for %arg17 = %c0 to %c2 step %c1 {
            scf.for %arg18 = %c0 to %c32 step %c1 {
              %15 = memref.load %subview_28[%arg16, %arg18] : memref<8x32xf16, strided<[32, 1], offset: ?>, 3>
              %16 = memref.load %subview_29[%arg18, %arg17] : memref<32x2xf16, strided<[128, 1], offset: ?>, 3>
              %17 = memref.load %subview_30[%arg16, %arg17] : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
              %18 = arith.mulf %15, %16 : f16
              %19 = arith.addf %17, %18 : f16
              memref.store %19, %subview_30[%arg16, %arg17] : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
            }
          }
        }
        gpu.barrier
      }
      %view_12 = memref.view %arg14[%c15104][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_12[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c1 step %c1 {
            memref.store %cst, %subview_24[%arg15, %arg16] : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      %view_13 = memref.view %arg14[%c13312][] : memref<72448xi8, 3> to memref<64x4xf32, 3>
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_11[0, %14] [64, 4] [1, 1] : memref<64x128xf16, 3> to memref<64x4xf16, strided<[128, 1], offset: ?>, 3>
        %15 = arith.cmpi ult, %4, %c16 : index
        scf.if %15 {
          %16 = affine.apply #map5()[%thread_id_x, %thread_id_y]
          %17 = affine.apply #map6()[%thread_id_x]
          %subview_25 = memref.subview %subview_24[%16, %17] [8, 2] [1, 1] : memref<64x4xf16, strided<[128, 1], offset: ?>, 3> to memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
          %subview_26 = memref.subview %view_13[%16, %17] [8, 2] [1, 1] : memref<64x4xf32, 3> to memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
          scf.for %arg15 = %c0 to %c8 step %c1 {
            scf.for %arg16 = %c0 to %c2 step %c1 {
              %18 = memref.load %subview_25[%arg15, %arg16] : memref<8x2xf16, strided<[128, 1], offset: ?>, 3>
              %19 = arith.extf %18 : f16 to f32
              memref.store %19, %subview_26[%arg15, %arg16] : memref<8x2xf32, strided<[4, 1], offset: ?>, 3>
            }
          }
        }
      }
      gpu.barrier
      scf.if %7 {
        %subview_24 = memref.subview %view_12[0, %4] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c64 step %c1 {
          scf.for %arg16 = %c0 to %c4 step %c1 {
            %14 = memref.load %view_13[%arg15, %arg16] : memref<64x4xf32, 3>
            %15 = memref.load %subview_24[%arg15] : memref<64xf32, strided<[32], offset: ?>, 3>
            %16 = arith.maximumf %14, %15 : f32
            memref.store %16, %subview_24[%arg15] : memref<64xf32, strided<[32], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      scf.if %8 {
        %14 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_12[%14, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_25 = memref.subview %view_2[%14] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c32 step %c1 {
            %15 = memref.load %subview_24[%arg15, %arg16] : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
            %16 = memref.load %subview_25[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
            %17 = arith.maximumf %15, %16 : f32
            memref.store %17, %subview_25[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
          }
        }
      }
      gpu.barrier {welder.keep_barrier}
      %alloca_14 = memref.alloca() {alignment = 64 : i64} : memref<64x128xf32>
      %subview_15 = memref.subview %alloca_10[%9, %10] [8, 2] [1, 1] : memref<64x128xf16> to memref<8x2xf16, strided<[128, 1], offset: ?>>
      %subview_16 = memref.subview %alloca_14[%9, %10] [8, 2] [1, 1] : memref<64x128xf32> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %14 = memref.load %subview_15[%arg15, %arg16] : memref<8x2xf16, strided<[128, 1], offset: ?>>
          %15 = arith.extf %14 : f16 to f32
          memref.store %15, %subview_16[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      %view_17 = memref.view %arg14[%c39680][] : memref<72448xi8, 3> to memref<64x128xf32, 3>
      %subview_18 = memref.subview %view_2[%9] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_19 = memref.subview %view_17[%9, %10] [8, 2] [1, 1] : memref<64x128xf32, 3> to memref<8x2xf32, strided<[128, 1], offset: ?>, 3>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %14 = memref.load %subview_16[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>>
          %15 = memref.load %subview_18[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
          %16 = arith.subf %14, %15 : f32
          %17 = math.exp %16 : f32
          memref.store %17, %subview_19[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>
        }
      }
      gpu.barrier
      %view_20 = memref.view %arg14[%c5120][] : memref<72448xi8, 3> to memref<64x32xf32, 3>
      scf.if %6 {
        %14 = affine.apply #map3()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_20[%14, %5] [8, 1] [1, 1] : memref<64x32xf32, 3> to memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c1 step %c1 {
            memref.store %cst_0, %subview_24[%arg15, %arg16] : memref<8x1xf32, strided<[32, 1], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      scf.if %7 {
        %14 = affine.apply #map4()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_17[0, %14] [64, 4] [1, 1] : memref<64x128xf32, 3> to memref<64x4xf32, strided<[128, 1], offset: ?>, 3>
        %subview_25 = memref.subview %view_20[0, %4] [64, 1] [1, 1] : memref<64x32xf32, 3> to memref<64xf32, strided<[32], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c64 step %c1 {
          scf.for %arg16 = %c0 to %c4 step %c1 {
            %15 = memref.load %subview_24[%arg15, %arg16] : memref<64x4xf32, strided<[128, 1], offset: ?>, 3>
            %16 = memref.load %subview_25[%arg15] : memref<64xf32, strided<[32], offset: ?>, 3>
            %17 = arith.addf %15, %16 : f32
            memref.store %17, %subview_25[%arg15] : memref<64xf32, strided<[32], offset: ?>, 3>
          }
        }
      }
      gpu.barrier
      scf.if %8 {
        %14 = affine.apply #map13()[%thread_id_x, %thread_id_y]
        %subview_24 = memref.subview %view_20[%14, 0] [8, 32] [1, 1] : memref<64x32xf32, 3> to memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
        %subview_25 = memref.subview %view[%14] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
        scf.for %arg15 = %c0 to %c8 step %c1 {
          scf.for %arg16 = %c0 to %c32 step %c1 {
            %15 = memref.load %subview_24[%arg15, %arg16] : memref<8x32xf32, strided<[32, 1], offset: ?>, 3>
            %16 = memref.load %subview_25[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
            %17 = arith.addf %15, %16 : f32
            memref.store %17, %subview_25[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
          }
        }
      }
      gpu.barrier {welder.keep_barrier}
      %subview_21 = memref.subview %memref[%3, 0] [64, 128] [1, 1] : memref<8192x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
      %subview_22 = memref.subview %view[%9] [8] [1] : memref<64xf32, 3> to memref<8xf32, strided<[1], offset: ?>, 3>
      %subview_23 = memref.subview %subview_21[%9, %10] [8, 2] [1, 1] : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<8x2xf32, strided<[128, 1], offset: ?>>
      scf.for %arg15 = %c0 to %c8 step %c1 {
        scf.for %arg16 = %c0 to %c2 step %c1 {
          %14 = memref.load %subview_19[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>, 3>
          %15 = memref.load %subview_22[%arg15] : memref<8xf32, strided<[1], offset: ?>, 3>
          %16 = arith.divf %14, %15 : f32
          memref.store %16, %subview_23[%arg15, %arg16] : memref<8x2xf32, strided<[128, 1], offset: ?>>
        }
      }
      gpu.barrier
      gpu.terminator
    }
    return %memref : memref<8192x128xf32>
  }
}


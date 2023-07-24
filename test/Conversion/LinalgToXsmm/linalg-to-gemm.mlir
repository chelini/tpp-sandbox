// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @simple_batch(%arg0: memref<64x8x32x64xf32>, %arg1: memref<64x8x64x32xf32>, 
                     %arg2: memref<64x8x32x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%arg2 : memref<64x8x32x32xf32>)
  linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}   
    ins(%arg0, %arg1 : memref<64x8x32x64xf32>, memref<64x8x64x32xf32>) 
    outs(%arg2 : memref<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
  } 
  return
}

// CHECK-LABEL: simple_batch
// CHECK-SAME:  %[[ARG0:.+]]: memref<64x8x32x64xf32>, %[[ARG1:.+]]: memref<64x8x64x32xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<64x8x32x32xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: scf.forall (%[[ARG4:.+]], %[[ARG5:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG4]], %[[ARG5]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x32x64xf32> to memref<1x1x32x64xf32, strided<[16384, 2048, 64, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG4]], %[[ARG5]], 0, 0] [1, 1, 64, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x64x32xf32> to memref<1x1x64x32xf32, strided<[16384, 2048, 32, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG4]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x32x32xf32> to memref<1x1x32x32xf32, strided<[8192, 1024, 32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 64, 64, 32, 32, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C1]])

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>

func.func @simple_batch_gemm(%arg0: memref<2x8x2x2xf32>, %arg1: memref<2x8x2x4xf32>, 
                             %arg2: memref<2x8x2x4xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} 
    ins(%arg0, %arg1 : memref<2x8x2x2xf32>, memref<2x8x2x4xf32>) 
    outs(%arg2 : memref<2x8x2x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %1 = arith.mulf %in, %in_8 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: simple_batch_gemm
// CHECK-SAME: %[[ARG0:.+]]: memref<2x8x2x2xf32>, %[[ARG1:.+]]: memref<2x8x2x4xf32>, %[[ARG2:.+]]: memref<2x8x2x4xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (2, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x8x2x2xf32> to memref<1x1x2x2xf32, strided<[32, 4, 2, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x8x2x4xf32> to memref<1x1x2x4xf32, strided<[64, 8, 4, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x8x2x4xf32> to memref<1x1x2x4xf32, strided<[64, 8, 4, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [2, 4, 2, 2, 4, 4, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C1]])

// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -cleanup \
// RUN: -bufferize -convert-linalg-to-xsmm -split-input-file | FileCheck %s 

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @simple_batch(%arg0: tensor<64x8x32x64xf32>, %arg1: tensor<64x8x64x32xf32>,
                        %arg2: tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<64x8x32x64xf32>, tensor<64x8x64x32xf32>)
    outs(%0 : tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %m = arith.mulf %in, %in_0 : f32
      %a = arith.addf %out, %m : f32
      linalg.yield %a : f32
  } -> tensor<64x8x32x32xf32>
  return %1 : tensor<64x8x32x32xf32>
}

// CHECK-LABEL: simple_batch
// CHECK-SAME:  %[[ARG0:.+]]: memref<64x8x32x64xf32>, %[[ARG1:.+]]: memref<64x8x64x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<64x8x32x32xf32>
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG0]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG1]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 64, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x64x32xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[BRGEMM:.+]] = xsmm.gemm.dispatch [32, 32, 64, 64, 32, 32] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[BRGEMM]], %[[SUB_0]], %[[SUB_1]], %[[SUB]])

// -----

#map = affine_map<(i, k, kk, j, jj) -> (i, k, kk)>
#map1 = affine_map<(i, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, k, kk, j, jj) -> (i, j, jj)>

func.func @brgemm_1(%arg0: tensor<32x8x64xf32>, %arg1: tensor<8x64x8x64xf32>,
                    %arg2: tensor<32x8x64xf32>) -> tensor<32x8x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<32x8x64xf32>) -> tensor<32x8x64xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<32x8x64xf32>, tensor<8x64x8x64xf32>)
    outs(%0 : tensor<32x8x64xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  } -> tensor<32x8x64xf32>
  return %1 : tensor<32x8x64xf32>
}

// CHECK-LABEL: brgemm_1
// CHECK-SAME: %[[ARG0:.+]]: memref<32x8x64xf32>, %[[ARG1:.+]]: memref<8x64x8x64xf32>, %[[ARG2:.+]]: memref<32x8x64xf32>
// CHECK: %[[C8:.+]] = arith.constant 8 : i64
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]]) in (8) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][0, %[[ARG3]], 0] [32, 1, 64] [1, 1, 1] 
// CHECK-SAME:  : memref<32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [32, 64, 1, 512] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG3]], 0] [8, 64, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<8x64x8x64xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
// CHECK: %[[BRGEMM:.+]] = xsmm.brgemm.dispatch [32, 64, 64, 512, 512, 512, 64, 32768] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %1, %[[ARG0]], %[[SUB_0]], %[[SUB]], %[[C8]])

// -----

#map = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, k, kk)>
#map1 = affine_map<(i, ii, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, j, jj)>

func.func @brgemm_2(%arg0: tensor<4x8x8x64xf32>, %arg1: tensor<8x64x6x7xf32>, 
                    %arg2: tensor<4x8x6x7xf32>) -> tensor<4x8x6x7xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2: tensor<4x8x6x7xf32>) -> tensor<4x8x6x7xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<4x8x8x64xf32>, tensor<8x64x6x7xf32>)
    outs(%0: tensor<4x8x6x7xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  } -> tensor<4x8x6x7xf32>
  return %1 : tensor<4x8x6x7xf32>
}

// CHECK-LABEL: brgemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<4x8x8x64xf32>, %[[ARG1:.+]]: memref<8x64x6x7xf32>, %[[ARG2:.+]]: memref<4x8x6x7xf32>
// CHECK: %[[C8:.+]] = arith.constant 8 : i64
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (4, 6) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 8, 1, 7] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<4x8x6x7xf32> to memref<8x7xf32, strided<[42, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [8, 7, 1, 42] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 8, 8, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<4x8x8x64xf32> to memref<8x8x64xf32, strided<[512, 64, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG4]], 0] [8, 64, 1, 7] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<8x64x6x7xf32> to memref<8x64x7xf32, strided<[2688, 42, 1], offset: ?>>
// CHECK: %[[BRGEMM:.+]] = xsmm.brgemm.dispatch [8, 7, 64, 512, 42, 42, 64, 2688] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[BRGEMM]], %[[SUB_0]], %[[SUB_1]], %[[SUB]], %[[C8]])

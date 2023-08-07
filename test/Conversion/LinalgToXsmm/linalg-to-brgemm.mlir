// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

#map = affine_map<(i, k, kk, j) -> (i, k, kk)>
#map1 = affine_map<(i, k, kk, j) -> (k, kk, j)>
#map2 = affine_map<(i, k, kk, j) -> (i, j)>

func.func @brgemm(%arg0: memref<2x2x2x4xf32>, %arg1: memref<2x4x8x2xf32>, 
                  %arg2: memref<2x2x8x2xf32>) {
  scf.forall (%arg3, %arg4) in (2, 8) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1] 
      : memref<2x2x2x4xf32> to memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>
    %subview_2 = memref.subview %arg1[0, 0, %arg4, 0] [2, 4, 1, 2] [1, 1, 1, 1] 
      : memref<2x4x8x2xf32> to memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>
    %subview_3 = memref.subview %arg2[%arg3, 0, %arg4, 0] [1, 2, 1, 2] [1, 1, 1, 1] 
      : memref<2x2x8x2xf32> to memref<2x2xf32, strided<[16, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2], 
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]} 
      ins(%subview, %subview_2 : memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>, memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>) 
      outs(%subview_3 : memref<2x2xf32, strided<[16, 1], offset: ?>>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
  }
  return
}

// CHECK-LABEL: brgemm
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2x2x4xf32>, %[[ARG1:.+]]: memref<2x4x8x2xf32>, %[[ARG2:.+]]: memref<2x2x8x2xf32>
// CHECK: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (2, 8) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x2x2x4xf32> to memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG4]], 0] [2, 4, 1, 2] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x4x8x2xf32> to memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 2, 1, 2] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<2x2x8x2xf32> to memref<2x2xf32, strided<[16, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [2, 2, 4, 8, 16, 16, 4, 64] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C2]])

// m = 2
// n = 2
// k = 4
// lda = 8
// ldb = 16
// ldc = 16
// stride_a = 4
// stride_b = 64

// -----

#map = affine_map<(i, k, kk, j, jj) -> (i, k, kk)>
#map1 = affine_map<(i, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, k, kk, j, jj) -> (i, j, jj)>

func.func @brgemm_1(%arg0: memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>>, %arg1: memref<8x64x8x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x8x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<32x8x64xf32>)
  linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]} 
    ins(%arg0, %arg1 : memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>>, memref<8x64x8x64xf32>) 
    outs(%alloc_0 : memref<32x8x64xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return 
}

// CHECK-LABEL: brgemm_1
// CHECK-SAME: %[[ARG0:.+]]: memref<32x8x64xf32, strided<[512, 64, 1], offset: ?>>, %[[ARG1:.+]]: memref<8x64x8x64xf32>
// CHECK: %[[C8:.+]] = arith.constant 8 : i64
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x8x64xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<32x8x64xf32>)
// CHECK: scf.forall (%[[ARG2:.+]]) in (8) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG2]], 0] [8, 64, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<8x64x8x64xf32> to memref<8x64x1x64xf32, strided<[32768, 512, 64, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ALLOC]][0, %[[ARG2]], 0] [32, 1, 64] [1, 1, 1] 
// CHECK-SAME:  : memref<32x8x64xf32> to memref<32x1x64xf32, strided<[512, 64, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 64, 64, 512, 512, 512, 64, 32768] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[SUB]], %[[SUB_0]], %[[C8]])

// -----
                 
#map = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, k, kk)>
#map1 = affine_map<(i, ii, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, j, jj)>

func.func @brgemm_2(%arg0: memref<4x8x8x64xf32>, %arg1: memref<8x64x6x7xf32>, %arg2: memref<4x8x6x7xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<4x8x8x64xf32>, memref<8x64x6x7xf32>)
    outs(%arg2: memref<4x8x6x7xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<4x8x8x64xf32>, %[[ARG1:.+]]: memref<8x64x6x7xf32>, %[[ARG2:.+]]: memref<4x8x6x7xf32>
// CHECK: %[[C8:.+]] = arith.constant 8 : i64
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (4, 6) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 8, 8, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<4x8x8x64xf32> to memref<1x8x8x64xf32, strided<[4096, 512, 64, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG4]], 0] [8, 64, 1, 7] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<8x64x6x7xf32> to memref<8x64x1x7xf32, strided<[2688, 42, 7, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 8, 1, 7] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<4x8x6x7xf32> to memref<1x8x1x7xf32, strided<[336, 42, 7, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [8, 7, 64, 512, 42, 42, 64, 2688] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C8]])

// -----

#map = affine_map<(i, j, kk, k) -> (kk, i, k)>
#map1 = affine_map<(i, j, kk, k) -> (kk, k, j)>
#map2 = affine_map<(i, j, kk, k) -> (i, j)>

func.func @brgemm_3(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x5x8xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_3
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x5x8xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %0, %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C9]])

// -----

#map = affine_map<(kk, k, i, j) -> (kk, i, k)>
#map1 = affine_map<(kk, k, i, j) -> (kk, k, j)>
#map2 = affine_map<(kk, k, i, j) -> (i, j)>

func.func @brgemm_4(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x5x8xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_4
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x5x8xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %0, %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C9]])

// -----

#map = affine_map<(i, j, kk, k) -> (kk, i, k)>
#map1 = affine_map<(i, j, kk, k) -> (kk, j, k)>
#map2 = affine_map<(i, j, kk, k) -> (i, j)>

func.func @brgemm_5(%arg0: memref<9x4x5xf32>, %arg1: memref<9x8x5xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x8x5xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_5
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x8x5xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<9x5x8xf32>
// CHECK: linalg.transpose ins(%[[ARG1]] : memref<9x8x5xf32>) 
// CHECK-SAME:  outs(%[[ALLOC]] : memref<9x5x8xf32>) permutation = [0, 2, 1]
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ALLOC]], %[[ARG2]], %[[C9]])

// -----

#map = affine_map<(i, k, kk, j, jj) -> (i, k, kk)>
#map1 = affine_map<(i, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, k, kk, j, jj) -> (i, j, jj)>

// The minor k dimension has a stride of 2.
// CHECK-LABEL: invalid_brgemm_6
func.func @invalid_brgemm_6(%arg0: memref<32x8x64xf32, strided<[512, 64, 2], offset: ?>>, %arg1: memref<8x64x8x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x8x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<32x8x64xf32>)
  // CHECK-NOT: scf.forall
  // CHECK-NOT: xsmm.brgemm
  // CHECK: linalg.generic
  linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]} 
    ins(%arg0, %arg1 : memref<32x8x64xf32, strided<[512, 64, 2], offset: ?>>, memref<8x64x8x64xf32>) 
    outs(%alloc_0 : memref<32x8x64xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return 
}

// -----

#map = affine_map<(i, kk, k, j, jj) -> (i, kk, k)>
#map1 = affine_map<(i, kk, k, j, jj) -> (kk, k, j, jj)>
#map2 = affine_map<(i, kk, k, j, jj) -> (i, j, jj)>

// Mixed semantics.
// CHECK-LABEL: invalid_brgemm_7
func.func @invalid_brgemm_7(%arg0: memref<32x8x64xf32, strided<[512, 64, 2], offset: ?>>, %arg1: tensor<8x64x8x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x8x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<32x8x64xf32>)
  // CHECK-NOT: scf.forall
  // CHECK-NOT: xsmm.brgemm
  // CHECK: linalg.generic
  linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]} 
    ins(%arg0, %arg1 : memref<32x8x64xf32, strided<[512, 64, 2], offset: ?>>, tensor<8x64x8x64xf32>) 
    outs(%alloc_0 : memref<32x8x64xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return 
}

// -----

#map = affine_map<(i, k, j) -> (i, k)>
#map1 = affine_map<(i, k, j) -> (j, k)>
#map2 = affine_map<(i, k, j) -> (j, i)>

// CHECK-LABEL: brgemm_8
func.func @brgemm_8(%arg0: memref<64x32x8x64xf32>, %arg1: memref<64x32x8x64xf32>) -> memref<64x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x8x32x32xf32>
  scf.forall (%arg2, %arg3) in (64, 8) {
    %subview = memref.subview %alloc[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %arg1[%arg2, 0, %arg3, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    %subview_1 = memref.subview %arg0[%arg2, 0, %arg3, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    // CHECK: %{{.+}} = memref.alloc() : memref<64x32xf32>
    // CHECK: %{{.+}} = xsmm.unary.dispatch transpose [32, 64, 512, 32] flags = (none) data_type = f32
    // CHECK: %{{.+}} = xsmm.brgemm.dispatch [32, 32, 64, 512, 32, 32, 1, 1] flags = (none) data_type = f32
    // CHECK-NOT: linalg.generic
    linalg.generic {
      indexing_maps = [#map, #map1, #map2], 
      iterator_types = ["parallel", "reduction", "parallel"]} 
      ins(%subview_0, %subview_1 : memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32, strided<[512, 1], offset: ?>>) 
      outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.mulf %in, %in_2 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
    }
  }
  return %alloc : memref<64x8x32x32xf32>
}

// -----

#map = affine_map<(i, k, j) -> (i, k)>
#map1 = affine_map<(i, k, j) -> (k, j)>
#map2 = affine_map<(i, k, j) -> (i, j)>

func.func @brgemm_9(%arg0: memref<64x64x8x32xf32>, %arg1: memref<64x32x8x64xf32>) -> memref<64x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x8x32x32xf32>
  scf.forall (%arg2, %arg3) in (64, 8) {
    %subview = memref.subview %alloc[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %arg1[%arg2, 0, %arg3, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    %subview_1 = memref.subview %arg0[%arg2, 0, %arg3, 0] [1, 64, 1, 32] [1, 1, 1, 1] : memref<64x64x8x32xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2], 
      iterator_types = ["parallel", "reduction", "parallel"]} 
      ins(%subview_0, %subview_1 : memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<64x32xf32, strided<[256, 1], offset: ?>>) 
      outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.mulf %in, %in_2 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
    }
  }
  return %alloc : memref<64x8x32x32xf32>
}

// CHECK-LABEL: brgemm_9
// CHECK-SAME: %[[ARG0:.+]]: memref<64x64x8x32xf32>, %[[ARG1:.+]]: memref<64x32x8x64xf32>
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<64x8x32x32xf32>
// CHECK: scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ALLOC]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[DIS_FILL:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[DIS_FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 64, 1, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<64x64x8x32xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 64, 512, 256, 32, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB_0]], %[[SUB_1]], %[[SUB]], %[[C1]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (i, j)>

func.func @gemm_1(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_1
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C1]])

// -----

#map = affine_map<(k, i, j) -> (i, k)>
#map1 = affine_map<(k, i, j) -> (k, j)>
#map2 = affine_map<(k, i, j) -> (i, j)>

func.func @gemm_2(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel"]} 
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C1]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_3(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_3
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: %[[DIS_TRANS:.+]] = xsmm.unary.dispatch transpose [64, 32, 32, 64] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRANS]], %[[ARG0]], %[[ALLOC]])
// CHECK: %[[ALLOC_0:.+]] = memref.alloc() : memref<64x32xf32>
// CHECK: %[[DIS_TRANS_1:.+]] = xsmm.unary.dispatch transpose [32, 64, 64, 32] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRANS_1]], %[[ARG1]], %[[ALLOC_0]])
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ALLOC_0]], %[[ALLOC]], %[[ARG2]], %[[C1]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_4(%arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1: memref<64x32xf32>, memref<64x32xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_4
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<64x32xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: %[[DIS_TRAN:.+]] = xsmm.unary.dispatch transpose [64, 32, 32, 64] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRAN]], %[[ARG0]], %[[ALLOC]])
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG1]], %[[ALLOC]], %[[ARG2]], %[[C1]])

// -----

#map = affine_map<(i, j, k) -> (k, i)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_5(%arg0: memref<32x64xf32>, %arg1: memref<64x32xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1: memref<32x64xf32>, memref<64x32xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_5
// CHECK-SAME: %[[ARG0:.+]]: memref<32x64xf32>, %[[ARG1:.+]]: memref<64x32xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]], %[[C1]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (i, j)>

// Sizes not statically known.
// CHECK-LABEL: gemm_6
func.func @gemm_6(%arg0: memref<64x32xf32>, %arg1: memref<?x?xf32>, %arg2: memref<64x64xf32>) {
  // CHECK: linalg.generic
  // CHECK-NOT: xsmm.brgemm
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1: memref<64x32xf32>, memref<?x?xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// -----

#map = affine_map<(b, i, j, k) -> (b, i, k)>
#map1 = affine_map<(b, i, j, k) -> (b, k, j)>
#map2 = affine_map<(b, i, j, k) -> (b, i, j)>

func.func @simple_batch(%arg0: memref<5x64x32xf32>, %arg1: memref<5x32x64xf32>, %arg2: memref<5x64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: memref<5x64x32xf32>, memref<5x32x64xf32>)
    outs(%arg2: memref<5x64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: simple_batch
// CHECK-SAME: %[[ARG0:.+]]: memref<5x64x32xf32>, %[[ARG1:.+]]: memref<5x32x64xf32>, %[[ARG2:.+]]: memref<5x64x64xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: scf.forall (%[[ARG3:.+]]) in (5)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0] [1, 64, 32] [1, 1, 1] 
// CHECK-SAME:  : memref<5x64x32xf32> to memref<1x64x32xf32, strided<[2048, 32, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0, 0] [1, 32, 64] [1, 1, 1] 
// CHECK-SAME:  : memref<5x32x64xf32> to memref<1x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, 0] [1, 64, 64] [1, 1, 1] 
// CHECK-SAME:  : memref<5x64x64xf32> to memref<1x64x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [64, 64, 32, 32, 64, 64, 1, 1] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C1]])

// RUN: tpp-opt %s -canonicalize-contractions -split-input-file | FileCheck %s

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

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_5
// CHECK-SAME: %[[ARG0:.+]]: memref<32x64xf32>, %[[ARG1:.+]]: memref<64x32xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG1]], %[[ARG0]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

#map = affine_map<(i, j, k) -> (k, i)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_5_t(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2: tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_5_t
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x64xf32>, %[[ARG1:.+]]: tensor<64x32xf32>, %[[ARG2:.+]]: tensor<64x64xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG1]], %[[ARG0]]
// CHECK-SAME: outs(%[[ARG2]]

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

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_4
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<64x32xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: linalg.transpose ins(%[[ARG0]] : memref<64x32xf32>) 
// CHECK-SAME:  outs(%[[ALLOC]] : memref<32x64xf32>) permutation = [1, 0] 
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG1]], %[[ALLOC]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_4_t(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<64x32xf32>, tensor<64x32xf32>)
    outs(%arg2: tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_4_t
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x32xf32>, %[[ARG1:.+]]: tensor<64x32xf32>, %[[ARG2:.+]]: tensor<64x64xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[TRANS:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<64x32xf32>) 
// CHECK-SAME:  outs(%[[EMPTY]] : tensor<32x64xf32>) permutation = [1, 0]
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG1]], %[[TRANS]]
// CHECK-SAME: outs(%[[ARG2]]


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

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_3
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: linalg.transpose ins(%[[ARG0]] : memref<64x32xf32>) 
// CHECK-SAME:  outs(%[[ALLOC]] : memref<32x64xf32>) permutation = [1, 0]
// CHECK: %[[ALLOC_0:.+]] = memref.alloc() : memref<64x32xf32>
// CHECK: linalg.transpose ins(%[[ARG1]] : memref<32x64xf32>) 
// CHECK-SAME:  outs(%[[ALLOC_0]] : memref<64x32xf32>) permutation = [1, 0]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ALLOC_0]], %[[ALLOC]]
// CHECK-SAME: outs(%[[ARG2]]


// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_3_t(%arg0: tensor<64x32xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<64x32xf32>, tensor<32x64xf32>)
    outs(%arg2: tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>

// CHECK-LABEL: gemm_3_t
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x32xf32>, %[[ARG1:.+]]: tensor<32x64xf32>, %[[ARG2:.+]]: tensor<64x64xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[TRANS:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<64x32xf32>) 
// CHECK-SAME:  outs(%[[EMPTY]] : tensor<32x64xf32>) permutation = [1, 0] 
// CHECK: %[[EMPTY_1:.+]] = tensor.empty() : tensor<64x32xf32>
// CHECK: %[[TRANS_0:.+]] = linalg.transpose ins(%[[ARG1]] : tensor<32x64xf32>) 
// CHECK-SAME:  outs(%[[EMPTY_1]] : tensor<64x32xf32>) permutation = [1, 0]
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[TRANS_0]], %[[TRANS]]
// CHECK-SAME: outs(%[[ARG2]]

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

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-LABEL: gemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

#map = affine_map<(k, i, j) -> (i, k)>
#map1 = affine_map<(k, i, j) -> (k, j)>
#map2 = affine_map<(k, i, j) -> (i, j)>

func.func @gemm_2_t(%arg0: tensor<64x32xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel"]} 
    ins(%arg0, %arg1: tensor<64x32xf32>, tensor<32x64xf32>)
    outs(%arg2: tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK-LABEL: gemm_2_t
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x32xf32>, %[[ARG1:.+]]: tensor<32x64xf32>, %[[ARG2:.+]]: tensor<64x64xf32>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME: outs(%[[ARG2]]

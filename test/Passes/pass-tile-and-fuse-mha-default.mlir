// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -split-input-file | FileCheck %s

#map3 = affine_map<(b, i, h, k, j) -> (b, i, h, k)>
#map4 = affine_map<(b, i, h, k, j) -> (b, j, h, k)>
#map5 = affine_map<(b, i, h, k, j) -> (b, h, j, i)>

func.func @batch_matmul(%4: tensor<64x32x8x64xf32>, %6: tensor<64x32x8x64xf32>) -> tensor<64x8x32x32xf32> {
  %cst_3 = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<64x8x32x32xf32>
  %8 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %9 = linalg.generic {
    indexing_maps = [#map3, #map4, #map5], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} 
    ins(%6, %4 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) 
    outs(%8 : tensor<64x8x32x32xf32>) attrs =  {__Q_times_K__} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %15 = arith.mulf %in, %in_4 : f32
      %16 = arith.addf %out, %15 : f32
      linalg.yield %16 : f32
  } -> tensor<64x8x32x32xf32>
  return %9 : tensor<64x8x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>

// CHECK-LABEL: batch_matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x32x8x64xf32>, %[[ARG1:.+]]: tensor<64x32x8x64xf32>
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[E:.+]] = tensor.empty() : tensor<64x8x32x32xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (%[[C64]], %[[C8]]) shared_outs(%[[ARG4:.+]] = %[[E]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:  ins(%[[SLICE_0]], %[[SLICE_1]] : tensor<32x64xf32>, tensor<32x64xf32>
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>

// -----

#map3 = affine_map<(b, i, h, k, j) -> (b, i, h, k)>
#map4 = affine_map<(b, i, h, k, j) -> (b, k, h, j)>
#map5 = affine_map<(b, i, h, k, j) -> (b, h, i, j)>

func.func @batch_matmul_1(%4: tensor<64x64x8x32xf32>, %6: tensor<64x32x8x64xf32>) -> tensor<64x8x32x32xf32> {
  %cst_3 = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<64x8x32x32xf32> 
  %8 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %9 = linalg.generic {
    indexing_maps = [#map3, #map4, #map5], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} 
    ins(%6, %4 : tensor<64x32x8x64xf32>, tensor<64x64x8x32xf32>) 
    outs(%8 : tensor<64x8x32x32xf32>) attrs =  {__Q_times_K__} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %15 = arith.mulf %in, %in_4 : f32
      %16 = arith.addf %out, %15 : f32
      linalg.yield %16 : f32
  } -> tensor<64x8x32x32xf32>
  return %9 : tensor<64x8x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: @batch_matmul_1
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x64x8x32xf32>, %[[ARG1:.+]]: tensor<64x32x8x64xf32>
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[E:.+]] = tensor.empty() : tensor<64x8x32x32xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (%[[C64]], %[[C8]]) shared_outs(%[[ARG4:.+]] = %[[E]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, %[[ARG3]], 0] [1, 64, 1, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x64x8x32xf32> to tensor<64x32xf32>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[SLICE_0]], %[[SLICE_1]] : tensor<32x64xf32>, tensor<64x32xf32>
// CHECK-SAME: outs(%[[FILL]] : tensor<32x32xf32>

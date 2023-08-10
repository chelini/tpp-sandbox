// RUN: tpp-opt %s -tile-consumer-and-fuse-producers | FileCheck %s

#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>

func.func @softmax(%arg0: tensor<64x32x8x64xf32>, %arg1: tensor<64x32x8x64xf32>, 
                   %arg2: tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32> {

  %cst_1 = arith.constant 0.0 : f32
  %6 = tensor.empty() : tensor<64x8x32x32xf32>
  %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %8 = linalg.generic {
    "__Q_times_K__",
    indexing_maps = [#map4, #map5, #map6], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) outs(%7 : tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x8x32x32xf32>

  %softmax_empty = tensor.empty() : tensor<64x8x32x32xf32>
  %softmax = linalg.softmax dimension(3) ins(%8 : tensor<64x8x32x32xf32>) 
                                         outs(%softmax_empty : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>

  %0 = tensor.empty() : tensor<64x32x8x64xf32>
  %fill_4 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %11 = linalg.generic {
    "__Softmax_times_V__",
    indexing_maps = [#map4, #map9, #map10], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%softmax, %arg2 : tensor<64x8x32x32xf32>, tensor<64x32x8x64xf32>) outs(%fill_4 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  return %11 : tensor<64x32x8x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: softmax
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x32x8x64xf32>, %[[ARG1:.+]]: tensor<64x32x8x64xf32>, %[[ARG2:.+]]: tensor<64x32x8x64xf32>
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64x32x8x64xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (%[[C64]], %[[C8]]) shared_outs(%[[ARG5:.+]] = %[[EMPTY]])
// TODO: EMPTY_SOFTMAX we should not have this allocation, but we cannot remove unit dims for softmax.
// CHECK: %[[EMPTY_SOFTMAX:.+]] = tensor.empty() : tensor<1x1x32x32xf32>
// CHECK: %[[EMPTY_GEN:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[FILL_GEN:.+]] = linalg.fill ins(%[[CST]] : f32) 
// CHECK-SAME:  outs(%[[EMPTY_GEN]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME: : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %[[SLICE_0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %[[GEN:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[SLICE]], %[[SLICE_0]]
// CHECK-SAME: outs(%[[FILL_GEN]]
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[GEN]] into %[[EMPTY_SOFTMAX]][0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<32x32xf32> into tensor<1x1x32x32xf32>
// CHECK: %[[OUT_SOFTMAX:.+]] = tensor.empty() : tensor<1x1x32x32xf32>
// CHECK: %[[SOFTMAX:.+]] = linalg.softmax dimension(3) ins(%[[INSERT]] : tensor<1x1x32x32xf32>) 
// CHECK-SAME:  outs(%[[OUT_SOFTMAX]] : tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %[[FILL_GEN_1:.+]] = linalg.fill ins(%[[CST]] : f32) 
// CHECK-SAME:  outs(%[[EXTRACT]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK: %[[EXTRACT_2:.+]] = tensor.extract_slice %[[SOFTMAX]][0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<1x1x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[EXTRACT_3:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<64x32x8x64xf32> to tensor<32x64xf32>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[EXTRACT_2]], %[[EXTRACT_3]]
// CHECK-SAME: outs(%[[FILL_GEN_1]]

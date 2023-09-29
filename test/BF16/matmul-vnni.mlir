// RUN: tpp-opt %s -tpp-mapping | FileCheck %s

!A_tensor_t = tensor<256x512xbf16>
!B_tensor_t = tensor<512x1024xbf16>
!C_tensor_t = tensor<256x1024xbf16>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) -> !C_tensor_t {
   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                           outs(%C: !C_tensor_t) -> !C_tensor_t
   return %matmul : !C_tensor_t
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

// CHECK-LABEL: matmul_static
// CHECK: %{{.+}} = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (8, 32)
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[ARG3]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %{{.+}}[%[[ARG4]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<32x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %{{.+}}[%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x32x32x32xbf16> to tensor<32x32xbf16>
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[SLICE]], %[[SLICE1]]
// CHECK-SAME:  outs(%[[SLICE2]]

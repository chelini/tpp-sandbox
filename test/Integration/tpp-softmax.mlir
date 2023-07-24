// RUN: tpp-opt %s -decompose-aggregated-ops -bufferize -convert-linalg-to-xsmm | \
// RUN: tpp-run -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -decompose-aggregated-ops -bufferize -convert-linalg-to-xsmm | \
// RUN: tpp-run -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s | \
// RUN: tpp-run -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -decompose-aggregated-ops -bufferize -convert-linalg-to-xsmm | \
// RUN: FileCheck %s -check-prefix=IR

!A_tensor_t = tensor<16x8xf32>
!B_tensor_t = tensor<16x8xf32>
!C_tensor_t = tensor<16x8xf32>
!D_tensor_t = tensor<16x8xf32>

#map = affine_map<(b, i, h, k, j) -> (b, h, i, k)>
#map1 = affine_map<(b, i, h, k, j) -> (b, h, k, j)>
#map2 = affine_map<(b, i, h, k, j) -> (b, h, i, j)>

func.func @matmul_static(%A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %D: !D_tensor_t) {
  %A_exp = tensor.expand_shape %A [[0, 1], [2, 3]] :
    !A_tensor_t into tensor<2x8x2x4xf32>
  %B_exp = tensor.expand_shape %B [[0, 1], [2, 3]] :
    !B_tensor_t into tensor<2x8x4x2xf32>
  %C_exp = tensor.expand_shape %C [[0, 1], [2, 3]] :
    !C_tensor_t into tensor<2x8x2x4xf32>
  %D_exp = tensor.expand_shape %D [[0, 1], [2, 3]] :
    !D_tensor_t into tensor<2x8x2x4xf32>

  %cst_fill = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x8x2x2xf32>
  %fill = linalg.fill ins(%cst_fill : f32) outs(%empty: tensor<2x8x2x2xf32>) -> tensor<2x8x2x2xf32>

  // IR: scf.forall (%{{.+}}, %{{.+}}) in (2, 8) 
  // IR-NOT: scf.forall
  // IR: xsmm.brgemm.dispatch [2, 2, 4, 4, 2, 2, 1, 1] flags = (none) data_type = f32 
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%A_exp, %B_exp: tensor<2x8x2x4xf32>, tensor<2x8x4x2xf32>)
    outs(%fill: tensor<2x8x2x2xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<2x8x2x2xf32>

  %softmax_empty = tensor.empty() : tensor<2x8x2x2xf32>
  %softmax = linalg.softmax dimension(3) ins(%1: tensor<2x8x2x2xf32>) 
    outs(%softmax_empty: tensor<2x8x2x2xf32>) -> tensor<2x8x2x2xf32>

  %2 = linalg.fill ins(%cst_fill : f32) outs(%C_exp: tensor<2x8x2x4xf32>) -> tensor<2x8x2x4xf32>
  // IR: xsmm.brgemm.dispatch [2, 4, 2, 2, 4, 4, 1, 1] flags = (none) data_type = f32
  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%softmax, %D_exp: tensor<2x8x2x2xf32>, tensor<2x8x2x4xf32>)
    outs(%C_exp: tensor<2x8x2x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<2x8x2x4xf32>
  
  %gemm_clps = tensor.collapse_shape %3 [[0, 1], [2, 3]] :
    tensor<2x8x2x4xf32> into !C_tensor_t
  %cst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %gemm_clps[%cst, %cst], %d1 : tensor<16x8xf32>, vector<16x8xf32>

  //
  // CHECK:     ( ( 5.09988, 6.09988, 7.09988, 8.09988, 5.1, 6.1, 7.1, 8.1 ), 
  // CHECK-SAME:  ( 5.19992, 6.19992, 7.19992, 8.19992, 5.2, 6.2, 7.2, 8.2 ), 
  // CHECK-SAME:  ( 5.29995, 6.29995, 7.29995, 8.29994, 5.3, 6.3, 7.3, 8.3 ), 
  // CHECK-SAME:  ( 5.39996, 6.39996, 7.39996, 8.39996, 5.4, 6.4, 7.4, 8.4 ), 
  // CHECK-SAME:  ( 5.49998, 6.49998, 7.49998, 8.49998, 5.5, 6.5, 7.5, 8.5 ), 
  // CHECK-SAME:  ( 5.59998, 6.59998, 7.59998, 8.59998, 5.6, 6.6, 7.6, 8.6 ), 
  // CHECK-SAME:  ( 5.69999, 6.69999, 7.69999, 8.69999, 5.7, 6.7, 7.7, 8.7 ), 
  // CHECK-SAME:  ( 5.79999, 6.79999, 7.79999, 8.79999, 5.8, 6.8, 7.8, 8.8 ), 
  // CHECK-SAME:  ( 5.9, 6.9, 7.9, 8.89999, 5.9, 6.9, 7.9, 8.9 ), 
  // CHECK-SAME:  ( 5.09988, 6.09988, 7.09988, 8.09988, 5.1, 6.1, 7.1, 8.1 ), 
  // CHECK-SAME:  ( 5.10988, 6.10988, 7.10988, 8.10988, 5.11, 6.11, 7.11, 8.11 ), 
  // CHECK-SAME:  ( 5.11989, 6.11989, 7.11989, 8.11989, 5.12, 6.12, 7.12, 8.12 ), 
  // CHECK-SAME:  ( 5.12989, 6.12989, 7.12989, 8.12989, 5.13, 6.13, 7.13, 8.13 ), 
  // CHECK-SAME:  ( 5.1399, 6.1399, 7.1399, 8.1399, 5.14, 6.14, 7.14, 8.14 ), 
  // CHECK-SAME:  ( 5.1499, 6.1499, 7.1499, 8.1499, 5.15, 6.15, 7.15, 8.15 ), 
  // CHECK-SAME:  ( 5.1599, 6.1599, 7.1599, 8.1599, 5.16, 6.16, 7.16, 8.16 ) )
  // 
  vector.print %v0 : vector<16x8xf32>
  
  return
}

func.func @entry() {
  %C = arith.constant dense<0.0> : !C_tensor_t
  %A = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9 ],
        [ 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10 ],
        [ 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11 ],
        [ 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12 ],
        [ 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13 ],
        [ 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14 ],
        [ 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15 ],
        [ 1.16, 2.16, 3.16, 4.16, 5.16, 6.16, 7.16, 8.16 ]
  ]> : !A_tensor_t
  %B = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9 ],
        [ 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10 ],
        [ 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11 ],
        [ 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12 ],
        [ 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13 ],
        [ 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14 ],
        [ 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15 ],
        [ 1.16, 2.16, 3.16, 4.16, 5.16, 6.16, 7.16, 8.16 ]
  ]> : !B_tensor_t
  %D = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9 ],
        [ 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10 ],
        [ 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11 ],
        [ 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12 ],
        [ 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13 ],
        [ 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14 ],
        [ 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15 ],
        [ 1.16, 2.16, 3.16, 4.16, 5.16, 6.16, 7.16, 8.16 ]
  ]> : !D_tensor_t

  call @matmul_static(%A, %B, %C, %D) : (!A_tensor_t, !B_tensor_t, !C_tensor_t, !D_tensor_t) -> ()
  return
}

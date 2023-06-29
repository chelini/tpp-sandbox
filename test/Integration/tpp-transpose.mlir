// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @tpp_transpose(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = tpp.transpose(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %cst = arith.constant 0.0 : f32

  // Initialize various matrices.
  %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
  ]> : tensor<4x8xf32>

  %db = tensor.empty() : tensor<8x4xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%db : tensor<8x4xf32>) -> tensor<8x4xf32>

  // Call kernel.
  %0 = call @tpp_transpose(%da, %db) 
    : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

  //
  // CHECK:     ( ( 1.1, 1.2, 1.3, 1.4 ),
  // CHECK-SAME:  ( 2.1, 2.2, 2.3, 2.4 ),
  // CHECK-SAME:  ( 3.1, 3.2, 3.3, 3.4 ),
  // CHECK-SAME:  ( 4.1, 4.2, 4.3, 4.4 ),
  // CHECK-SAME:  ( 5.1, 5.2, 5.3, 5.4 ),
  // CHECK-SAME:  ( 6.1, 6.2, 6.3, 6.4 ),
  // CHECK-SAME:  ( 7.1, 7.2, 7.3, 7.4 ),
  // CHECK-SAME:  ( 8.1, 8.2, 8.3, 8.4 ) )
  //  
  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<8x4xf32>, vector<8x4xf32>
  vector.print %v0 : vector<8x4xf32>

  return
}

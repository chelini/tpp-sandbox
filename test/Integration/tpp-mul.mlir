// This should really be in the passes directory, not here
// RUN: tpp-opt %s -bufferize -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -tpp-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// TPP: func.func @tpp_mul(
// TPP-SAME: %[[A:.+]]: memref<9x6xf32>, %[[B:.+]]: memref<9x6xf32>)
func.func @tpp_mul(%A: tensor<9x6xf32>, %B: tensor<9x6xf32>) -> tensor<9x6xf32>  {
  // TPP: tpp.mul ins(%[[A]] : memref<9x6xf32>, %[[B]] : memref<9x6xf32>) outs(%[[B]] : memref<9x6xf32>)
  %O = linalg.generic { indexing_maps = [#map, #map], 
                        iterator_types = ["parallel", "parallel"] }
    ins(%A: tensor<9x6xf32>)
    outs(%B: tensor<9x6xf32>) {
      ^bb0(%a: f32, %b: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0: f32
  } -> tensor<9x6xf32>
  return %O: tensor<9x6xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1    ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2    ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3    ],
        [ 1.4, -2.4, -3.4, -4.4, 5.4, 6.6 ],
        [ 1.5, -2.5, -3.5, -4.5, 5.5, 6.5 ],
        [ 1.6, -2.6, -3.6, -4.6, 5.6, 6.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7    ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8    ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9    ]
  ]> : tensor<9x6xf32>

  %db = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1    ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2    ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3    ],
        [ 1.4, -2.4, -3.4, -4.4, 5.4, 6.6 ],
        [ 1.5, -2.5, -3.5, -4.5, 5.5, 6.5 ],
        [ 1.6, -2.6, -3.6, -4.6, 5.6, 6.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7    ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8    ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9    ]
  ]> : tensor<9x6xf32>

  %0 = call @tpp_mul(%da, %db) : (tensor<9x6xf32>, tensor<9x6xf32>) -> tensor<9x6xf32>

  //
  // CHECK:       ( ( 1.21, 4.41, 9.61, 16.81, 26.01, 37.21 ),
  // CHECK-SAME:    ( 1.44, 4.84, 10.24, 17.64, 27.04, 38.44 ),
  // CHECK-SAME:    ( 1.69, 5.29, 10.89, 18.49, 28.09, 39.69 ),
  // CHECK-SAME:    ( 1.96, 5.76, 11.56, 19.36, 29.16, 43.56 ),
  // CHECK-SAME:    ( 2.25, 6.25, 12.25, 20.25, 30.25, 42.25 ),
  // CHECK-SAME:    ( 2.56, 6.76, 12.96, 21.16, 31.36, 43.56 ),
  // CHECK-SAME:    ( 2.89, 7.29, 13.69, 22.09, 32.49, 44.89 ),
  // CHECK-SAME:    ( 3.24, 7.84, 14.44, 23.04, 33.64, 46.24 ),
  // CHECK-SAME:    ( 3.61, 8.41, 15.21, 24.01, 34.81, 47.61 ) )
  //

  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<9x6xf32>, vector<9x6xf32>
  vector.print %v0 : vector<9x6xf32>

  return
}

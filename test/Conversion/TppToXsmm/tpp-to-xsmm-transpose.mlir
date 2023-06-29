// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: tranpose
// CHECK-SAME: %[[ARG0:.+]]: memref<2x3xf32>, %[[ARG1:.+]]: memref<3x2xf32>
func.func @tranpose(%arg0: memref<2x3xf32>, %arg1: memref<3x2xf32>) {
  // m = 3
  // n = 2
  // ldi = 3
  // ldo = 2
  // CHECK: %[[DIS:.+]] = xsmm.unary.dispatch transpose [3, 2, 3, 2] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary transpose
  // CHECK-SAME: (data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])
  tpp.transpose ins(%arg0: memref<2x3xf32>, %arg1: memref<3x2xf32>) outs(%arg1: memref<3x2xf32>)
  return
}

// -----

// CHECK-LABEL: trivial_transpose
// CHECK-SAME:  %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
func.func @trivial_transpose(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  // m = 2
  // n = 2
  // ldi = 2
  // ldo = 2
  // CHECK: %[[DIS:.+]] = xsmm.unary.dispatch transpose [2, 2, 2, 2] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary transpose
  // CHECK-SAME:  (data_type = f32, %[[DIS]], %[[ARG0:.+]], %[[ARG1:.+]])
  tpp.transpose ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) outs(%arg1: memref<2x2xf32>)
  return 
}

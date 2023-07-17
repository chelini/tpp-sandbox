// RUN: tpp-opt %s -linalg-make-ops-2d | FileCheck %s

// CHECK-LABEL: transpose
func.func @transpose(%arg0: tensor<64x32x16x32xf32>) -> tensor<64x16x32x32xf32> {
  %1 = tensor.empty() : tensor<64x16x32x32xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<64x32x16x32xf32>) outs(%1 : tensor<64x16x32x32xf32>) permutation = [0, 2, 1, 3] 
  return %transposed : tensor<64x16x32x32xf32>
}

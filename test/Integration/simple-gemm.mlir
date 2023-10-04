// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-xsmm \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes="linalg-to-xsmm" | \
// RUN: FileCheck %s -check-prefix=IR

// RUN: tpp-opt %s -default-tpp-passes | \
// RUN: FileCheck %s -check-prefix=IR

func.func @entry(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // IR: xsmm_gemm_invoke
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// CHECK-COUNT-4: ( 9, 9, 9, 9 )

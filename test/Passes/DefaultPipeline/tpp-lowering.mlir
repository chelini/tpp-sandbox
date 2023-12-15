// RUN: tpp-opt %s -tpp-lowering | FileCheck %s

func.func @tpp_ops(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
  tpp.brgemm ins(%arg0 : memref<3x5x4xf32>, %arg1 : memref<3x4x5xf32>, %arg2 : memref<5x5xf32>)
             outs(%arg2 : memref<5x5xf32>)
  tpp.relu ins(%arg2 : memref<5x5xf32>) outs(%arg2 : memref<5x5xf32>)
  tpp.gemm ins(%arg2 : memref<5x5xf32>, %arg3 : memref<5x5xf32>, %arg2 : memref<5x5xf32>)
           outs(%arg2 : memref<5x5xf32>)
  return
}

// CHECK-LABEL: func.func @tpp_ops(
// CHECK-NOT: tpp.brgemm
// CHECK: xsmm.brgemm
// CHECK-NOT: tpp.relu
// CHECK: xsmm.unary relu
// CHECK-NOT: tpp.gemm
// CHECK: xsmm.gemm

// CHECK-LABEL: copy_memref
func.func @copy_memref(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  // CHECK: xsmm.unary.dispatch identity
  // CHECK-NEXT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}

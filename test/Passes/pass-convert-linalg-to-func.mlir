// RUN: tpp-opt %s -convert-linalg-to-func | FileCheck %s


func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
                outs(%arg2 : memref<64x64xf32>)
  return
}

// CHECK-LABEL: matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<64x64xf32>, %[[ARG1:.+]]: memref<64x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[C64:.+]] = arith.constant 64 : index
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[PTR_ARG0:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG0:.+]] = arith.index_cast %[[PTR_ARG0]] : index to i64
// CHECK: %[[LLVM_PTR_ARG0:.+]] = llvm.inttoptr %[[PTR_CAST_ARG0]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG1:.+]] = arith.index_cast %[[PTR_ARG1]] : index to i64
// CHECK: %[[LLVM_PTR_ARG1:.+]] = llvm.inttoptr %[[PTR_CAST_ARG1]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG2:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG2:.+]] = arith.index_cast %[[PTR_ARG2]] : index to i64
// CHECK: %[[LLVM_PTR_ARG2:.+]] = llvm.inttoptr %[[PTR_CAST_ARG2]] : i64 to !llvm.ptr<f32>
// CHECK: call @linalg_matmul_blas(%[[C64]], %[[C64]], %[[C64]], %[[LLVM_PTR_ARG0]], %[[C0]], %[[C64]], %[[LLVM_PTR_ARG1]], %[[C0]], %[[C64]], %[[LLVM_PTR_ARG2]], %[[C0]], %[[C64]])

// RUN: tpp-opt %s -fold-xsmm-ops | FileCheck %s
// XFAIL: *
func.func @simple_zero(%A: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                       %B: memref<512x64xf32, strided<[512, 1], offset: ?>>,
                       %C: memref<32x64xf32>) {
  %cst = arith.constant 0.0 : f32
  %c1_i64 = arith.constant 1 : i64

  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %C) : (i64, f32, memref<32x64xf32>) -> ()

  %1 = xsmm.brgemm.dispatch [32, 64, 512, 512, 512, 64, 1, 1] flags = (none) data_type = f32
  xsmm.brgemm(data_type = f32, %1, %A, %B, %C, %c1_i64) 
    : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, 
            memref<512x64xf32, strided<[512, 1], offset: ?>>, 
            memref<32x64xf32>, i64) -> ()
  return
}

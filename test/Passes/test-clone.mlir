// RUN: tpp-opt %s -mlir-disable-threading=true -pass-pipeline="builtin.module(func.func(test-clone))" -o /dev/null 2>&1 | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @entry(%arg0: tensor<256x1024xf32>, %arg1: tensor<1024x1024xf32>, 
                 %arg2: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  // CHECK: notifyOperationInserted: arith.mulf
  // CHECK-NEXT: notifyOperationInserted: arith.addf
  // CHECK-NEXT: notifyOperationInserted: linalg.yield
  // CHECK-NEXT: notifyOperationInserted: linalg.generic
  // CHECK-NEXT: ------
  // CHECK-NOT: notifyOperationInserted: arith.mulf
  // CHECK-NOT: notifyOperationInserted: arith.addf
  // CHECK-NOT: notifyOperationInserted: linalg.yield
  // CHECK-NEXT: notifyOperationInserted: linalg.generic
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%arg2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<256x1024xf32>
  return %0 : tensor<256x1024xf32>
}

// RUN: tpp-opt %s -convert-to-dps | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: concat
func.func @concat(%arg1: tensor<384x32xf32>, %arg2: tensor<32x384xf32>,
                  %arg3: tensor<32x384xf32>, %arg4: tensor<384x32xf32>) -> tensor<1536x384xf32> {

  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<0.00164203614> : tensor<384xf32>
  // CHECK: %[[CONCAT:.+]] = tensor.empty() : tensor<1536x384xf32> 
  %0 = tensor.empty() : tensor<384x384xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<384x384xf32>) -> tensor<384x384xf32>
  %2 = linalg.matmul ins(%arg1, %arg2 : tensor<384x32xf32>, tensor<32x384xf32>) outs(%1 : tensor<384x384xf32>) -> tensor<384x384xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<384xf32>) outs(%2 : tensor<384x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.addf %in, %out : f32
      linalg.yield %17 : f32
  } -> tensor<384x384xf32>
  // CHECK: %[[SLICE:.+]] = tensor.insert_slice %{{.+}} into %[[CONCAT]][0, 0] [384, 384] [1, 1] : tensor<384x384xf32> into tensor<1536x384xf32> 
  %4 = tensor.empty() : tensor<384x384xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<384x384xf32>) -> tensor<384x384xf32>
  %6 = linalg.matmul ins(%arg4, %arg3 : tensor<384x32xf32>, tensor<32x384xf32>) outs(%5 : tensor<384x384xf32>) -> tensor<384x384xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<384xf32>) outs(%6 : tensor<384x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.addf %in, %out : f32
      linalg.yield %17 : f32
  } -> tensor<384x384xf32>

  // Concat.
  %8 = tensor.empty() : tensor<1536x384xf32>
  %inserted_slice = tensor.insert_slice %3 into %8[0, 0] [384, 384] [1, 1] : tensor<384x384xf32> into tensor<1536x384xf32>
  // CHECK: %{{.+}} = tensor.insert_slice %{{.+}} into %[[SLICE]][384, 0] [384, 384] [1, 1] : tensor<384x384xf32> into tensor<1536x384xf32>
  %inserted_slice_1 = tensor.insert_slice %7 into %inserted_slice[384, 0] [384, 384] [1, 1] : tensor<384x384xf32> into tensor<1536x384xf32>
  return %inserted_slice_1 : tensor<1536x384xf32>
}

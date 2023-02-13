// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack -element-wise-fusion -transform-dialect-interpreter -cse -canonicalize | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @main(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %cst_0 = arith.constant dense<0.142857149> : tensor<1x1x64x64xf32>
  %cst_1 = arith.constant dense<1.250000e-01> : tensor<64xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, metadata = "expect_to_map2", strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst_0 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %6 = arith.addf %in, %in_3 : f32
      linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %6 = arith.maxf %in, %in_3 : f32
      linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %5 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_2 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %padded : tensor<1x58x58x64xf32>
}

// Preconditions:
// - Packing 
// - Packing propagation
// - Element-wise fusion
//
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!pdl.operation) -> !pdl.operation

    // Filter to get only blocked convolutions.
    %blocked_convolution = transform.structured.get_blocked_convolutions %generics
      : (!pdl.operation) -> !transform.op<"linalg.generic">
    
    // Well...
    %casted_blocked_convolution = transform.cast %blocked_convolution
      : !transform.op<"linalg.generic"> to !pdl.operation

    // transform.print %casted_blocked_convolution : !pdl.operation
 
    // Get consumer (aka fused element-wise operation - relu + add).
    %fused_eltwise = transform.get_consumers_of_result %casted_blocked_convolution[0]
      : (!pdl.operation) -> !pdl.operation

    // transform.print %fused_eltwise : !pdl.operation

    // Tile along the three-outermost parallel loops.
    %tiled_eltwise:2 = transform.structured.tile_to_forall_op %fused_eltwise 
      tile_sizes [1, 1, 1]
    
    // Fusion work on produers only not consumers, thus tile the fused eltwise producer.
    // Then fuse the convolution inside.
    %fused_convolution = transform.structured.fuse_into_containing_op %casted_blocked_convolution 
      into %tiled_eltwise#0
   
    // Collapse R and S, since they are 1s.
    %collapsed_filter_convolution = transform.structured.collapse %fused_convolution 
      [[0], [1], [2], [3], [4], [5, 6, 7], [8]]

    // Additional collapse on the image to align the BRGEMM dimension with
    // the filter.
    %collapsed_img_convolution = transform.structured.collapse %collapsed_filter_convolution 
      [[0], [1], [2, 3], [4], [5], [6]]

    // Expose BRGEMM by swapping loops.
    %interchanged_convolution = transform.structured.interchange %collapsed_img_convolution 
      iterator_interchange = [0, 1, 4, 2, 3, 5]

    // Rewrite to BRGEMM
    transform.structured.rewrite_to_brgemm %interchanged_convolution 
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @main(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x56x56x64xf32>)
// CHECK-DAG: %[[CST_0:.+]] = arith.constant dense<0.142857149> : tensor<2x32x32xf32>
// CHECK: %[[PACKED:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK:.+]] = tensor.pack %[[ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[PACKED]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[LOOP:.+]] = scf.forall 
// CHECK-SAME:  (%[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]]) in (1, 2, 56) 
// CHECK-SAME:  shared_outs(%[[ARG4:.+]] = %[[PACKED]]) -> (tensor<1x2x56x56x32xf32>)
// CHECK: %[[SLICE:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[PACK]][%[[ARG1]], 0, %[[ARG3]], 0, 0] [1, 2, 1, 56, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<1x2x56x56x32xf32> to tensor<1x2x1x56x32xf32>
// CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape 
// CHECK-SAME:  %[[SLICE]] {{\[}}[0], [1], [2, 3], [4]] : tensor<1x2x1x56x32xf32> into tensor<1x2x56x32xf32>
// CHECK: %[[SLICE_4:.+]] = tensor.extract_slice %[[COLLAPSE]][0, 0, 0, 0] [1, 2, 56, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<1x2x56x32xf32> to tensor<2x56x32xf32>
// CHECK: %[[BRGEMM:.+]] = linalg.batch_reduce_matmul 
// CHECK-SAME:  ins(%[[SLICE_4]], %[[CST_0]] : tensor<2x56x32xf32>, tensor<2x32x32xf32>) 
// CHECK-SAME:  outs(%{{.+}} : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[BRGEMM]] 
// CHECK-SAME:  into %{{.+}}[0, 0, 0, 0] [1, 1, 56, 32] [1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x1x56x32xf32>
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape 
// CHECK-SAME:  %[[INSERT]] {{\[}}[0], [1], [2, 3], [4]] : tensor<1x1x56x32xf32> into tensor<1x1x1x56x32xf32>
// CHECK: %[[SLICE_5:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG4]][%[[ARG1]], %[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<1x2x56x56x32xf32> to tensor<1x1x1x56x32xf32>
// CHECK: %[[ELT:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[EXPAND]]
// CHECK-SAME:  outs(%[[SLICE_5]]
// CHECK: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:  %[[ADDF:.+]] = arith.addf %[[IN]], %{{.+}} : f32
// CHECK-NEXT:  %[[MAXF:.+]] = arith.maxf %[[ADDF]], %{{.+}} : f32
// CHECK-NEXT:  linalg.yield %[[MAXF]] : f32
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice %[[ELT]] 
// CHECK-SAME:  into %[[ARG4]][%[[ARG1]], %[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<1x1x1x56x32xf32> into tensor<1x2x56x56x32xf32>
// CHECK: }

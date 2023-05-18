//RUN: tpp-opt %s -tpp-combine -split-input-file | FileCheck %s

func.func @fused_brgemm_test0(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%0: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test0
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>, 
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<32x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<32x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test1(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<32x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test1
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>, 
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<32x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<32x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test2(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<1x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<1x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test2
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<1x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<1x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test3(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<1x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<1x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test3
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<1x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = none, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<1x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test4(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<1x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<1x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test4
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x32xf32>, %[[ARG1:.+]]: tensor<32x32xf32>, %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<1x32xf32>
// CHECK: %[[EXPAND_A:.+]] = tensor.expand_shape %[[ARG0]] 
// CHECK-SAME:  {{\[}}[0, 1], [2]] : tensor<32x32xf32> into tensor<1x32x32xf32>
// CHECK: %[[EXPAND_B:.+]] = tensor.expand_shape %[[ARG1]]
// CHECK-SAME:  {{\[}}[0, 1], [2]] : tensor<32x32xf32> into tensor<1x32x32xf32>
// CHECK: %{{.+}} = tpp.fused_brgemm [unary = none, binary = add]
// CHECK-SAME: (%[[EXPAND_A]] : tensor<1x32x32xf32>, %[[EXPAND_B]] : tensor<1x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<1x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test5(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>,
                              %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<32x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test5
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xf32>, %[[ARG1:.+]]: tensor<32x32xf32>, %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<32x32xf32>
// CHECK: %[[EXPAND_A:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:  {{\[}}[0, 1], [2]] : tensor<32x32xf32> into tensor<1x32x32xf32>
// CHECK: %[[EXPAND_B:.+]] = tensor.expand_shape %[[ARG1]]
// CHECK-SAME:  {{\[}}[0, 1], [2]] : tensor<32x32xf32> into tensor<1x32x32xf32>
// CHECK: %{{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[EXPAND_A]] : tensor<1x32x32xf32>, %[[EXPAND_B]] : tensor<1x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<32x32xf32>) -> (tensor<32x32xf32>)

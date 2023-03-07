// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack -tile-consumer-and-fuse-producers="tile-sizes=1,1,1" -cse -canonicalize -rewrite-to-brgemm | FileCheck -check-prefix=BRGEMM %s

// BRGEMM-COUNT-16: linalg.batch_reduce_matmul

// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack | FileCheck %s --check-prefix=PACK

// PACK-COUNT-1: tensor.pack
// PACK-COUNT-1: tensor.unpack

// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack -tile-consumer-and-fuse-producers="tile-sizes=1,1,1" -cse -canonicalize -rewrite-to-brgemm -rewrite-to-gemm | FileCheck -check-prefix=GEMM %s

// GEMM-COUNT-12: linalg.matmul

// RUN: tpp-opt %s | FileCheck %s --check-prefix=CONV

// CONV-COUNT-29: linalg.conv_2d_nhwc_hwcf

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet50v1(%arg0: tensor<1x224x224x3xi64>) -> tensor<1x14x14x256xi64> {
    %cst = arith.constant 0 : i64
    %cst_0 = arith.constant 0 : i64
    %cst_1 = arith.constant dense<0> : tensor<1x14x14x1024xi64>
    %cst_2 = arith.constant dense<0> : tensor<1x14x14x256xi64>
    %cst_3 = arith.constant dense<0> : tensor<1x28x28x512xi64>
    %cst_4 = arith.constant dense<0> : tensor<1x28x28x128xi64>
    %cst_5 = arith.constant dense<0> : tensor<1x56x56x256xi64>
    %cst_6 = arith.constant dense<0> : tensor<1x56x56x64xi64>
    %cst_7 = arith.constant dense<0> : tensor<1x112x112x64xi64>
    %cst_8 = arith.constant dense<1> : tensor<7x7x3x64xi64>
    %cst_9 = arith.constant dense<5> : tensor<64xi64>
    %cst_10 = arith.constant dense<0> : tensor<1x1x64x64xi64>
    %cst_11 = arith.constant dense<1> : tensor<64xi64>
    %cst_12 = arith.constant dense<0> : tensor<3x3x64x64xi64>
    %cst_13 = arith.constant dense<1> : tensor<64xi64>
    %cst_14 = arith.constant dense<1> : tensor<1x1x64x256xi64>
    %cst_15 = arith.constant dense<5> : tensor<256xi64>
    %cst_16 = arith.constant dense<0> : tensor<1x1x64x256xi64>
    %cst_17 = arith.constant dense<1> : tensor<256xi64>
    %cst_18 = arith.constant dense<1> : tensor<1x1x256x64xi64>
    %cst_19 = arith.constant dense<3> : tensor<64xi64>
    %cst_20 = arith.constant dense<0> : tensor<3x3x64x64xi64>
    %cst_21 = arith.constant dense<0> : tensor<64xi64>
    %cst_22 = arith.constant dense<0> : tensor<1x1x64x256xi64>
    %cst_23 = arith.constant dense<0> : tensor<256xi64>
    %cst_24 = arith.constant dense<0> : tensor<1x1x256x64xi64>
    %cst_25 = arith.constant dense<2> : tensor<64xi64>
    %cst_26 = arith.constant dense<0> : tensor<3x3x64x64xi64>
    %cst_27 = arith.constant dense<0> : tensor<64xi64>
    %cst_28 = arith.constant dense<1> : tensor<1x1x64x256xi64>
    %cst_29 = arith.constant dense<1> : tensor<256xi64>
    %cst_30 = arith.constant dense<1> : tensor<1x1x256x128xi64>
    %cst_31 = arith.constant dense<1> : tensor<128xi64>
    %cst_32 = arith.constant dense<1> : tensor<3x3x128x128xi64>
    %cst_33 = arith.constant dense<1> : tensor<128xi64>
    %cst_34 = arith.constant dense<0> : tensor<1x1x256x512xi64>
    %cst_35 = arith.constant dense<1> : tensor<512xi64>
    %cst_36 = arith.constant dense<0> : tensor<1x1x128x512xi64>
    %cst_37 = arith.constant dense<0> : tensor<512xi64>
    %cst_38 = arith.constant dense<1> : tensor<1x1x512x128xi64>
    %cst_39 = arith.constant dense<1> : tensor<128xi64>
    %cst_40 = arith.constant dense<1> : tensor<3x3x128x128xi64>
    %cst_41 = arith.constant dense<1> : tensor<128xi64>
    %cst_42 = arith.constant dense<1> : tensor<1x1x128x512xi64>
    %cst_43 = arith.constant dense<1> : tensor<512xi64>
    %cst_44 = arith.constant dense<1> : tensor<1x1x512x128xi64>
    %cst_45 = arith.constant dense<1> : tensor<128xi64>
    %cst_46 = arith.constant dense<1> : tensor<3x3x128x128xi64>
    %cst_47 = arith.constant dense<8> : tensor<128xi64>
    %cst_48 = arith.constant dense<1> : tensor<1x1x128x512xi64>
    %cst_49 = arith.constant dense<1> : tensor<512xi64>
    %cst_50 = arith.constant dense<1> : tensor<1x1x512x128xi64>
    %cst_51 = arith.constant dense<7> : tensor<128xi64>
    %cst_52 = arith.constant dense<1> : tensor<3x3x128x128xi64>
    %cst_53 = arith.constant dense<1> : tensor<128xi64>
    %cst_54 = arith.constant dense<1> : tensor<1x1x128x512xi64>
    %cst_55 = arith.constant dense<1> : tensor<512xi64>
    %cst_56 = arith.constant dense<1> : tensor<1x1x512x256xi64>
    %cst_57 = arith.constant dense<1> : tensor<256xi64>
    %cst_58 = arith.constant dense<1> : tensor<3x3x256x256xi64>
    %cst_59 = arith.constant dense<1> : tensor<256xi64>
    %cst_60 = arith.constant dense<1> : tensor<1x1x512x1024xi64>
    %cst_61 = arith.constant dense<1> : tensor<1024xi64>
    %cst_62 = arith.constant dense<1> : tensor<1x1x256x1024xi64>
    %cst_63 = arith.constant dense<6> : tensor<1024xi64>
    %cst_64 = arith.constant dense<5> : tensor<1x1x1024x256xi64>
    %cst_65 = arith.constant dense<0> : tensor<256xi64>
    %padded = tensor.pad %arg0 low[0, 3, 3, 0] high[0, 3, 3, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x224x224x3xi64> to tensor<1x230x230x3xi64>
    %0 = tensor.empty() : tensor<1x112x112x64xi64>
    %1 = linalg.fill ins(%cst_0 : i64) outs(%0 : tensor<1x112x112x64xi64>) -> tensor<1x112x112x64xi64>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %cst_8 : tensor<1x230x230x3xi64>, tensor<7x7x3x64xi64>) outs(%1 : tensor<1x112x112x64xi64>) -> tensor<1x112x112x64xi64>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_9 : tensor<64xi64>) outs(%0 : tensor<1x112x112x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x112x112x64xi64>
    %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add"} ins(%2, %3 : tensor<1x112x112x64xi64>, tensor<1x112x112x64xi64>) outs(%0 : tensor<1x112x112x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x112x112x64xi64>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_7 : tensor<1x112x112x64xi64>, tensor<1x112x112x64xi64>) outs(%0 : tensor<1x112x112x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x112x112x64xi64>
    %padded_66 = tensor.pad %5 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x112x112x64xi64> to tensor<1x114x114x64xi64>
    %6 = tensor.empty() : tensor<3x3xi64>
    %7 = tensor.empty() : tensor<1x56x56x64xi64>
    %8 = linalg.fill ins(%cst : i64) outs(%7 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %9 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_66, %6 : tensor<1x114x114x64xi64>, tensor<3x3xi64>) outs(%8 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %10 = tensor.empty() : tensor<1x56x56x256xi64>
    %11 = linalg.fill ins(%cst_0 : i64) outs(%10 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
    %12 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map1"} ins(%9, %cst_14 : tensor<1x56x56x64xi64>, tensor<1x1x64x256xi64>) outs(%11 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map1"} ins(%cst_15 : tensor<256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x256xi64>
    %14 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map1"} ins(%12, %13 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %15 = linalg.fill ins(%cst_0 : i64) outs(%7 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %16 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map2"} ins(%9, %cst_10 : tensor<1x56x56x64xi64>, tensor<1x1x64x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %17 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map2"} ins(%cst_11 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map2"} ins(%16, %17 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %padded_67 = tensor.pad %19 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x56x56x64xi64> to tensor<1x58x58x64xi64>
    %20 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map3"} ins(%padded_67, %cst_12 : tensor<1x58x58x64xi64>, tensor<3x3x64x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %21 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_13 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %23 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %24 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map4"} ins(%23, %cst_16 : tensor<1x56x56x64xi64>, tensor<1x1x64x256xi64>) outs(%11 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
    %25 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map4"} ins(%cst_17 : tensor<256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x256xi64>
    %26 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map4"} ins(%24, %25 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %27 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %26 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %cst_5 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %29 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map5"} ins(%28, %cst_18 : tensor<1x56x56x256xi64>, tensor<1x1x256x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %30 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map5"} ins(%cst_19 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %31 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map5"} ins(%29, %30 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %padded_68 = tensor.pad %32 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x56x56x64xi64> to tensor<1x58x58x64xi64>
    %33 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map6"} ins(%padded_68, %cst_20 : tensor<1x58x58x64xi64>, tensor<3x3x64x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %34 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_21 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %35 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %34 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %37 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map7"} ins(%36, %cst_22 : tensor<1x56x56x64xi64>, tensor<1x1x64x256xi64>) outs(%11 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
    %38 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map7"} ins(%cst_23 : tensor<256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x256xi64>
    %39 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map7"} ins(%37, %38 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %39 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %41 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %cst_5 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %42 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map8"} ins(%41, %cst_24 : tensor<1x56x56x256xi64>, tensor<1x1x256x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %43 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map8"} ins(%cst_25 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %44 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map8"} ins(%42, %43 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %45 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %padded_69 = tensor.pad %45 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x56x56x64xi64> to tensor<1x58x58x64xi64>
    %46 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map8"} ins(%padded_69, %cst_26 : tensor<1x58x58x64xi64>, tensor<3x3x64x64xi64>) outs(%15 : tensor<1x56x56x64xi64>) -> tensor<1x56x56x64xi64>
    %47 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_27 : tensor<64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x64xi64>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %47 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %49 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48, %cst_6 : tensor<1x56x56x64xi64>, tensor<1x56x56x64xi64>) outs(%7 : tensor<1x56x56x64xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x64xi64>
    %50 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map9"} ins(%49, %cst_28 : tensor<1x56x56x64xi64>, tensor<1x1x64x256xi64>) outs(%11 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
    %51 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map9"} ins(%cst_29 : tensor<256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x56x56x256xi64>
    %52 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map9"} ins(%50, %51 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %53 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %52 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %54 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %cst_5 : tensor<1x56x56x256xi64>, tensor<1x56x56x256xi64>) outs(%10 : tensor<1x56x56x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x56x56x256xi64>
    %55 = tensor.empty() : tensor<1x28x28x512xi64>
    %56 = linalg.fill ins(%cst_0 : i64) outs(%55 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %57 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%54, %cst_34 : tensor<1x56x56x256xi64>, tensor<1x1x256x512xi64>) outs(%56 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %58 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_35 : tensor<512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x512xi64>
    %59 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %58 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %60 = tensor.empty() : tensor<1x28x28x128xi64>
    %61 = linalg.fill ins(%cst_0 : i64) outs(%60 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %62 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%54, %cst_30 : tensor<1x56x56x256xi64>, tensor<1x1x256x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %63 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_31 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %64 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %63 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %65 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %padded_70 = tensor.pad %65 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x28x28x128xi64> to tensor<1x30x30x128xi64>
    %66 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map10"} ins(%padded_70, %cst_32 : tensor<1x30x30x128xi64>, tensor<3x3x128x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %67 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_33 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %67 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %69 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %70 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map11"} ins(%69, %cst_36 : tensor<1x28x28x128xi64>, tensor<1x1x128x512xi64>) outs(%56 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %71 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map11"} ins(%cst_37 : tensor<512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x512xi64>
    %72 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map11"} ins(%70, %71 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %73 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%59, %72 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %74 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %cst_3 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %75 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map12"} ins(%74, %cst_38 : tensor<1x28x28x512xi64>, tensor<1x1x512x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %76 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map12"} ins(%cst_39 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %77 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map12"} ins(%75, %76 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %78 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %padded_71 = tensor.pad %78 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x28x28x128xi64> to tensor<1x30x30x128xi64>
    %79 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map13"} ins(%padded_71, %cst_40 : tensor<1x30x30x128xi64>, tensor<3x3x128x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %80 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_41 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %81 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %82 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %83 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map14"} ins(%82, %cst_42 : tensor<1x28x28x128xi64>, tensor<1x1x128x512xi64>) outs(%56 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %84 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map14"} ins(%cst_43 : tensor<512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x512xi64>
    %85 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map14"} ins(%83, %84 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %86 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85, %cst_3 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %87 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map15"} ins(%86, %cst_44 : tensor<1x28x28x512xi64>, tensor<1x1x512x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %88 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map15"} ins(%cst_45 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %89 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map15"} ins(%87, %88 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %90 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %padded_72 = tensor.pad %90 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x28x28x128xi64> to tensor<1x30x30x128xi64>
    %91 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map16"} ins(%padded_72, %cst_46 : tensor<1x30x30x128xi64>, tensor<3x3x128x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %92 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_47 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %93 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %92 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %94 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %95 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map17"} ins(%94, %cst_48 : tensor<1x28x28x128xi64>, tensor<1x1x128x512xi64>) outs(%56 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %96 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map17"} ins(%cst_49 : tensor<512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x512xi64>
    %97 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map17"} ins(%95, %96 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %98 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %97 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %99 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %cst_3 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %100 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map18"} ins(%99, %cst_50 : tensor<1x28x28x512xi64>, tensor<1x1x512x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %101 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map18"} ins(%cst_51 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %102 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map18"} ins(%100, %101 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %103 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %padded_73 = tensor.pad %103 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x28x28x128xi64> to tensor<1x30x30x128xi64>
    %104 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map19"} ins(%padded_73, %cst_52 : tensor<1x30x30x128xi64>, tensor<3x3x128x128xi64>) outs(%61 : tensor<1x28x28x128xi64>) -> tensor<1x28x28x128xi64>
    %105 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map19"} ins(%cst_53 : tensor<128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x128xi64>
    %106 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map19"} ins(%104, %105 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %107 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %cst_4 : tensor<1x28x28x128xi64>, tensor<1x28x28x128xi64>) outs(%60 : tensor<1x28x28x128xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x128xi64>
    %108 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map20"} ins(%107, %cst_54 : tensor<1x28x28x128xi64>, tensor<1x1x128x512xi64>) outs(%56 : tensor<1x28x28x512xi64>) -> tensor<1x28x28x512xi64>
    %109 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map20"} ins(%cst_55 : tensor<512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x28x28x512xi64>
    %110 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map20"} ins(%108, %109 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %111 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %110 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %112 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %cst_3 : tensor<1x28x28x512xi64>, tensor<1x28x28x512xi64>) outs(%55 : tensor<1x28x28x512xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x28x28x512xi64>
    %113 = tensor.empty() : tensor<1x14x14x1024xi64>
    %114 = linalg.fill ins(%cst_0 : i64) outs(%113 : tensor<1x14x14x1024xi64>) -> tensor<1x14x14x1024xi64>
    %115 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%112, %cst_60 : tensor<1x28x28x512xi64>, tensor<1x1x512x1024xi64>) outs(%114 : tensor<1x14x14x1024xi64>) -> tensor<1x14x14x1024xi64>
    %116 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_61 : tensor<1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x14x14x1024xi64>
    %117 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %116 : tensor<1x14x14x1024xi64>, tensor<1x14x14x1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x1024xi64>
    %118 = tensor.empty() : tensor<1x14x14x256xi64>
    %119 = linalg.fill ins(%cst_0 : i64) outs(%118 : tensor<1x14x14x256xi64>) -> tensor<1x14x14x256xi64>
    %120 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%112, %cst_56 : tensor<1x28x28x512xi64>, tensor<1x1x512x256xi64>) outs(%119 : tensor<1x14x14x256xi64>) -> tensor<1x14x14x256xi64>
    %121 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_57 : tensor<256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x14x14x256xi64>
    %122 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %121 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    %123 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %cst_2 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    %padded_74 = tensor.pad %123 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : i64
    } : tensor<1x14x14x256xi64> to tensor<1x16x16x256xi64>
    %124 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_not_to_map21"} ins(%padded_74, %cst_58 : tensor<1x16x16x256xi64>, tensor<3x3x256x256xi64>) outs(%119 : tensor<1x14x14x256xi64>) -> tensor<1x14x14x256xi64>
    %125 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_59 : tensor<256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x14x14x256xi64>
    %126 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %125 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    %127 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %cst_2 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    %128 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map22"} ins(%127, %cst_62 : tensor<1x14x14x256xi64>, tensor<1x1x256x1024xi64>) outs(%114 : tensor<1x14x14x1024xi64>) -> tensor<1x14x14x1024xi64>
    %129 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map22"} ins(%cst_63 : tensor<1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x14x14x1024xi64>
    %130 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map22"} ins(%128, %129 : tensor<1x14x14x1024xi64>, tensor<1x14x14x1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x1024xi64>
    %131 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %130 : tensor<1x14x14x1024xi64>, tensor<1x14x14x1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x1024xi64>
    %132 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131, %cst_1 : tensor<1x14x14x1024xi64>, tensor<1x14x14x1024xi64>) outs(%113 : tensor<1x14x14x1024xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x1024xi64>
    %133 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, metadata = "expect_to_map23"} ins(%132, %cst_64 : tensor<1x14x14x1024xi64>, tensor<1x1x1024x256xi64>) outs(%119 : tensor<1x14x14x256xi64>) -> tensor<1x14x14x256xi64>
    %134 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "bias_expect_to_map23"} ins(%cst_65 : tensor<256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x14x14x256xi64>
    %135 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], metadata = "add_expect_to_map23"} ins(%133, %134 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.addi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    %136 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%135, %cst_2 : tensor<1x14x14x256xi64>, tensor<1x14x14x256xi64>) outs(%118 : tensor<1x14x14x256xi64>) {
    ^bb0(%in: i64, %in_75: i64, %out: i64):
      %137 = arith.maxsi %in, %in_75 : i64
      linalg.yield %137 : i64
    } -> tensor<1x14x14x256xi64>
    return %136 : tensor<1x14x14x256xi64>
}

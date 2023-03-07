// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack -interchange-conv-to-expose-matmul -element-wise-fusion -tile-consumer-and-fuse-producers="tile-sizes=1,1,1" -cse -canonicalize -rewrite-to-brgemm | FileCheck --check-prefix=BRGEMM %s

// BRGEMM-COUNT-30: linalg.batch_reduce_matmul

// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -propagate-pack-and-unpack -canonicalize -constant-fold-pack | FileCheck %s --check-prefix=PACK

// PACK-COUNT-1: tensor.pack
// PACK-COUNT-1: tensor.unpack

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @resnet50v1(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.001000e-05> : tensor<64xf32>
    %cst_2 = arith.constant dense<1.001000e-05> : tensor<256xf32>
    %cst_3 = arith.constant dense<1.001000e-05> : tensor<512xf32>
    %cst_4 = arith.constant dense<1.001000e-05> : tensor<128xf32>
    %cst_5 = arith.constant dense<1.001000e-05> : tensor<1024xf32>
    %cst_6 = arith.constant dense<1.001000e-05> : tensor<2048xf32>
    %cst_7 = arith.constant dense<4.900000e+01> : tensor<1x2048xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<1x7x7x2048xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<1x7x7x512xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<1x14x14x1024xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<1x14x14x256xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x28x28x512xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<1x28x28x128xf32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<1x56x56x256xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<1x112x112x64xf32>
    %cst_17 = arith.constant dense<1.000000e+00> : tensor<7x7x3x64xf32>
    %cst_18 = arith.constant dense<5.000000e-01> : tensor<64xf32>
    %cst_19 = arith.constant dense<0.142857149> : tensor<1x1x64x64xf32>
    %cst_20 = arith.constant dense<1.250000e-01> : tensor<64xf32>
    %cst_21 = arith.constant dense<0.0769230798> : tensor<3x3x64x64xf32>
    %cst_22 = arith.constant dense<0.0714285746> : tensor<64xf32>
    %cst_23 = arith.constant dense<0.0526315793> : tensor<1x1x64x256xf32>
    %cst_24 = arith.constant dense<5.000000e-02> : tensor<256xf32>
    %cst_25 = arith.constant dense<0.0476190485> : tensor<1x1x64x256xf32>
    %cst_26 = arith.constant dense<0.0454545468> : tensor<256xf32>
    %cst_27 = arith.constant dense<0.0322580636> : tensor<1x1x256x64xf32>
    %cst_28 = arith.constant dense<3.125000e-02> : tensor<64xf32>
    %cst_29 = arith.constant dense<0.0270270277> : tensor<3x3x64x64xf32>
    %cst_30 = arith.constant dense<0.0263157897> : tensor<64xf32>
    %cst_31 = arith.constant dense<0.0232558139> : tensor<1x1x64x256xf32>
    %cst_32 = arith.constant dense<0.0227272734> : tensor<256xf32>
    %cst_33 = arith.constant dense<0.0204081628> : tensor<1x1x256x64xf32>
    %cst_34 = arith.constant dense<2.000000e-02> : tensor<64xf32>
    %cst_35 = arith.constant dense<0.0181818176> : tensor<3x3x64x64xf32>
    %cst_36 = arith.constant dense<0.0178571437> : tensor<64xf32>
    %cst_37 = arith.constant dense<0.0163934417> : tensor<1x1x64x256xf32>
    %cst_38 = arith.constant dense<0.0161290318> : tensor<256xf32>
    %cst_39 = arith.constant dense<0.0149253728> : tensor<1x1x256x128xf32>
    %cst_40 = arith.constant dense<0.0147058824> : tensor<128xf32>
    %cst_41 = arith.constant dense<0.01369863> : tensor<3x3x128x128xf32>
    %cst_42 = arith.constant dense<0.0135135138> : tensor<128xf32>
    %cst_43 = arith.constant dense<0.0126582282> : tensor<1x1x256x512xf32>
    %cst_44 = arith.constant dense<1.250000e-02> : tensor<512xf32>
    %cst_45 = arith.constant dense<0.0123456791> : tensor<1x1x128x512xf32>
    %cst_46 = arith.constant dense<0.0121951215> : tensor<512xf32>
    %cst_47 = arith.constant dense<0.0109890113> : tensor<1x1x512x128xf32>
    %cst_48 = arith.constant dense<0.0108695654> : tensor<128xf32>
    %cst_49 = arith.constant dense<0.010309278> : tensor<3x3x128x128xf32>
    %cst_50 = arith.constant dense<0.0102040814> : tensor<128xf32>
    %cst_51 = arith.constant dense<0.00970873795> : tensor<1x1x128x512xf32>
    %cst_52 = arith.constant dense<0.00961538497> : tensor<512xf32>
    %cst_53 = arith.constant dense<0.00917431153> : tensor<1x1x512x128xf32>
    %cst_54 = arith.constant dense<0.0090909088> : tensor<128xf32>
    %cst_55 = arith.constant dense<0.00869565178> : tensor<3x3x128x128xf32>
    %cst_56 = arith.constant dense<8.620690e-03> : tensor<128xf32>
    %cst_57 = arith.constant dense<0.00826446246> : tensor<1x1x128x512xf32>
    %cst_58 = arith.constant dense<0.00819672085> : tensor<512xf32>
    %cst_59 = arith.constant dense<0.00787401571> : tensor<1x1x512x128xf32>
    %cst_60 = arith.constant dense<7.812500e-03> : tensor<128xf32>
    %cst_61 = arith.constant dense<0.00751879718> : tensor<3x3x128x128xf32>
    %cst_62 = arith.constant dense<0.00746268639> : tensor<128xf32>
    %cst_63 = arith.constant dense<0.00719424477> : tensor<1x1x128x512xf32>
    %cst_64 = arith.constant dense<0.00714285718> : tensor<512xf32>
    %cst_65 = arith.constant dense<0.0068965517> : tensor<1x1x512x256xf32>
    %cst_66 = arith.constant dense<0.00684931502> : tensor<256xf32>
    %cst_67 = arith.constant dense<0.00662251655> : tensor<3x3x256x256xf32>
    %cst_68 = arith.constant dense<0.00657894742> : tensor<256xf32>
    %cst_69 = arith.constant dense<0.00636942684> : tensor<1x1x512x1024xf32>
    %cst_70 = arith.constant dense<0.00632911408> : tensor<1024xf32>
    %cst_71 = arith.constant dense<0.00628930796> : tensor<1x1x256x1024xf32>
    %cst_72 = arith.constant dense<6.250000e-03> : tensor<1024xf32>
    %cst_73 = arith.constant dense<5.917160e-03> : tensor<1x1x1024x256xf32>
    %cst_74 = arith.constant dense<0.00588235306> : tensor<256xf32>
    %cst_75 = arith.constant dense<0.00571428565> : tensor<3x3x256x256xf32>
    %cst_76 = arith.constant dense<0.00568181835> : tensor<256xf32>
    %cst_77 = arith.constant dense<0.00552486209> : tensor<1x1x256x1024xf32>
    %cst_78 = arith.constant dense<0.00549450563> : tensor<1024xf32>
    %cst_79 = arith.constant dense<0.00534759369> : tensor<1x1x1024x256xf32>
    %cst_80 = arith.constant dense<0.00531914877> : tensor<256xf32>
    %cst_81 = arith.constant dense<0.00518134702> : tensor<3x3x256x256xf32>
    %cst_82 = arith.constant dense<0.00515463902> : tensor<256xf32>
    %cst_83 = arith.constant dense<0.00502512557> : tensor<1x1x256x1024xf32>
    %cst_84 = arith.constant dense<5.000000e-03> : tensor<1024xf32>
    %cst_85 = arith.constant dense<0.00487804879> : tensor<1x1x1024x256xf32>
    %cst_86 = arith.constant dense<0.00485436898> : tensor<256xf32>
    %cst_87 = arith.constant dense<0.00473933667> : tensor<3x3x256x256xf32>
    %cst_88 = arith.constant dense<0.0047169812> : tensor<256xf32>
    %cst_89 = arith.constant dense<0.00460829493> : tensor<1x1x256x1024xf32>
    %cst_90 = arith.constant dense<0.00458715577> : tensor<1024xf32>
    %cst_91 = arith.constant dense<0.00448430516> : tensor<1x1x1024x256xf32>
    %cst_92 = arith.constant dense<0.00446428591> : tensor<256xf32>
    %cst_93 = arith.constant dense<0.0043668123> : tensor<3x3x256x256xf32>
    %cst_94 = arith.constant dense<0.00434782589> : tensor<256xf32>
    %cst_95 = arith.constant dense<0.00425531901> : tensor<1x1x256x1024xf32>
    %cst_96 = arith.constant dense<0.00423728814> : tensor<1024xf32>
    %cst_97 = arith.constant dense<0.00414937781> : tensor<1x1x1024x256xf32>
    %cst_98 = arith.constant dense<0.00413223123> : tensor<256xf32>
    %cst_99 = arith.constant dense<0.0040485831> : tensor<3x3x256x256xf32>
    %cst_100 = arith.constant dense<0.00403225794> : tensor<256xf32>
    %cst_101 = arith.constant dense<0.00395256933> : tensor<1x1x256x1024xf32>
    %cst_102 = arith.constant dense<0.00393700786> : tensor<1024xf32>
    %cst_103 = arith.constant dense<0.00386100379> : tensor<1x1x1024x512xf32>
    %cst_104 = arith.constant dense<0.00384615385> : tensor<512xf32>
    %cst_105 = arith.constant dense<0.00377358496> : tensor<3x3x512x512xf32>
    %cst_106 = arith.constant dense<0.00375939859> : tensor<512xf32>
    %cst_107 = arith.constant dense<0.00369003695> : tensor<1x1x1024x2048xf32>
    %cst_108 = arith.constant dense<0.0036764706> : tensor<2048xf32>
    %cst_109 = arith.constant dense<0.00366300368> : tensor<1x1x512x2048xf32>
    %cst_110 = arith.constant dense<0.00364963501> : tensor<2048xf32>
    %cst_111 = arith.constant dense<0.00353356893> : tensor<1x1x2048x512xf32>
    %cst_112 = arith.constant dense<0.00352112669> : tensor<512xf32>
    %cst_113 = arith.constant dense<0.00346020772> : tensor<3x3x512x512xf32>
    %cst_114 = arith.constant dense<0.00344827585> : tensor<512xf32>
    %cst_115 = arith.constant dense<0.00338983047> : tensor<1x1x512x2048xf32>
    %cst_116 = arith.constant dense<0.00337837846> : tensor<2048xf32>
    %cst_117 = arith.constant dense<0.00332225906> : tensor<1x1x2048x512xf32>
    %cst_118 = arith.constant dense<0.00331125828> : tensor<512xf32>
    %cst_119 = arith.constant dense<0.00325732888> : tensor<3x3x512x512xf32>
    %cst_120 = arith.constant dense<0.00324675324> : tensor<512xf32>
    %cst_121 = arith.constant dense<0.00319488812> : tensor<1x1x512x2048xf32>
    %cst_122 = arith.constant dense<0.00318471342> : tensor<2048xf32>
    %cst_123 = arith.constant dense<0.00313479616> : tensor<2048x1000xf32>
    %cst_124 = arith.constant dense<3.125000e-03> : tensor<1000xf32>
    %padded = tensor.pad %arg0 low[0, 3, 3, 0] high[0, 3, 3, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x224x224x3xf32> to tensor<1x230x230x3xf32>
    %0 = tensor.empty() : tensor<1x112x112x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %cst_17 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%1 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %3 = tensor.empty() : tensor<1x112x112x64xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_18 : tensor<64xf32>) outs(%3 : tensor<1x112x112x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x112x112x64xf32>
    %5 = tensor.empty() : tensor<1x112x112x64xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %4 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%5 : tensor<1x112x112x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x112x112x64xf32>
    %7 = tensor.empty() : tensor<1x112x112x64xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %cst_16 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%7 : tensor<1x112x112x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x112x112x64xf32>
    %padded_125 = tensor.pad %8 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x112x112x64xf32> to tensor<1x114x114x64xf32>
    %9 = tensor.empty() : tensor<3x3xf32>
    %10 = tensor.empty() : tensor<1x56x56x64xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %12 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_125, %9 : tensor<1x114x114x64xf32>, tensor<3x3xf32>) outs(%11 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %13 = tensor.empty() : tensor<1x56x56x256xf32>
    %14 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%12, %cst_23 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%14 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %16 = tensor.empty() : tensor<1x56x56x256xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_24 : tensor<256xf32>) outs(%16 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x256xf32>
    %18 = tensor.empty() : tensor<1x56x56x256xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %17 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%18 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %20 = tensor.empty() : tensor<1x56x56x64xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%12, %cst_19 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%21 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %23 = tensor.empty() : tensor<1x56x56x64xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_20 : tensor<64xf32>) outs(%23 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %25 = tensor.empty() : tensor<1x56x56x64xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %24 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%25 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %27 = tensor.empty() : tensor<1x56x56x64xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%27 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %29 = tensor.empty() : tensor<1x56x56x64xf32>
    %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %padded_126 = tensor.pad %28 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
    %31 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_126, %cst_21 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%30 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %32 = tensor.empty() : tensor<1x56x56x64xf32>
    %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_22 : tensor<64xf32>) outs(%32 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %34 = tensor.empty() : tensor<1x56x56x64xf32>
    %35 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %33 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%34 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %36 = tensor.empty() : tensor<1x56x56x64xf32>
    %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%36 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %38 = tensor.empty() : tensor<1x56x56x256xf32>
    %39 = linalg.fill ins(%cst_0 : f32) outs(%38 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %40 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%37, %cst_25 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%39 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %41 = tensor.empty() : tensor<1x56x56x256xf32>
    %42 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_26 : tensor<256xf32>) outs(%41 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x256xf32>
    %43 = tensor.empty() : tensor<1x56x56x256xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %42 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%43 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %45 = tensor.empty() : tensor<1x56x56x256xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %44 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%45 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %47 = tensor.empty() : tensor<1x56x56x256xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%47 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %49 = tensor.empty() : tensor<1x56x56x64xf32>
    %50 = linalg.fill ins(%cst_0 : f32) outs(%49 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %51 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%48, %cst_27 : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%50 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %52 = tensor.empty() : tensor<1x56x56x64xf32>
    %53 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_28 : tensor<64xf32>) outs(%52 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %54 = tensor.empty() : tensor<1x56x56x64xf32>
    %55 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %53 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%54 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %56 = tensor.empty() : tensor<1x56x56x64xf32>
    %57 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%55, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%56 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %58 = tensor.empty() : tensor<1x56x56x64xf32>
    %59 = linalg.fill ins(%cst_0 : f32) outs(%58 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %padded_127 = tensor.pad %57 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
    %60 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_127, %cst_29 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%59 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %61 = tensor.empty() : tensor<1x56x56x64xf32>
    %62 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_30 : tensor<64xf32>) outs(%61 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %63 = tensor.empty() : tensor<1x56x56x64xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %62 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%63 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %65 = tensor.empty() : tensor<1x56x56x64xf32>
    %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%65 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %67 = tensor.empty() : tensor<1x56x56x256xf32>
    %68 = linalg.fill ins(%cst_0 : f32) outs(%67 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %69 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%66, %cst_31 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%68 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %70 = tensor.empty() : tensor<1x56x56x256xf32>
    %71 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_32 : tensor<256xf32>) outs(%70 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x256xf32>
    %72 = tensor.empty() : tensor<1x56x56x256xf32>
    %73 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %71 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%72 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %74 = tensor.empty() : tensor<1x56x56x256xf32>
    %75 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48, %73 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%74 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %76 = tensor.empty() : tensor<1x56x56x256xf32>
    %77 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%76 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %78 = tensor.empty() : tensor<1x56x56x64xf32>
    %79 = linalg.fill ins(%cst_0 : f32) outs(%78 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %80 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%77, %cst_33 : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%79 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %81 = tensor.empty() : tensor<1x56x56x64xf32>
    %82 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_34 : tensor<64xf32>) outs(%81 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %83 = tensor.empty() : tensor<1x56x56x64xf32>
    %84 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80, %82 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%83 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %85 = tensor.empty() : tensor<1x56x56x64xf32>
    %86 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%84, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%85 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %87 = tensor.empty() : tensor<1x56x56x64xf32>
    %88 = linalg.fill ins(%cst_0 : f32) outs(%87 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %padded_128 = tensor.pad %86 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
    %89 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_128, %cst_35 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%88 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %90 = tensor.empty() : tensor<1x56x56x64xf32>
    %91 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_36 : tensor<64xf32>) outs(%90 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x64xf32>
    %92 = tensor.empty() : tensor<1x56x56x64xf32>
    %93 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %91 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%92 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %94 = tensor.empty() : tensor<1x56x56x64xf32>
    %95 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%94 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x64xf32>
    %96 = tensor.empty() : tensor<1x56x56x256xf32>
    %97 = linalg.fill ins(%cst_0 : f32) outs(%96 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %98 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%95, %cst_37 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%97 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %99 = tensor.empty() : tensor<1x56x56x256xf32>
    %100 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_38 : tensor<256xf32>) outs(%99 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x56x256xf32>
    %101 = tensor.empty() : tensor<1x56x56x256xf32>
    %102 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %100 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%101 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %103 = tensor.empty() : tensor<1x56x56x256xf32>
    %104 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77, %102 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%103 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %105 = tensor.empty() : tensor<1x56x56x256xf32>
    %106 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%105 : tensor<1x56x56x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x56x56x256xf32>
    %107 = tensor.empty() : tensor<1x28x28x512xf32>
    %108 = linalg.fill ins(%cst_0 : f32) outs(%107 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %109 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%106, %cst_43 : tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) outs(%108 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %110 = tensor.empty() : tensor<1x28x28x512xf32>
    %111 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_44 : tensor<512xf32>) outs(%110 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x512xf32>
    %112 = tensor.empty() : tensor<1x28x28x512xf32>
    %113 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %111 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%112 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %114 = tensor.empty() : tensor<1x28x28x128xf32>
    %115 = linalg.fill ins(%cst_0 : f32) outs(%114 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %116 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%106, %cst_39 : tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) outs(%115 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %117 = tensor.empty() : tensor<1x28x28x128xf32>
    %118 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_40 : tensor<128xf32>) outs(%117 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %119 = tensor.empty() : tensor<1x28x28x128xf32>
    %120 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116, %118 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%119 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %121 = tensor.empty() : tensor<1x28x28x128xf32>
    %122 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%121 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %123 = tensor.empty() : tensor<1x28x28x128xf32>
    %124 = linalg.fill ins(%cst_0 : f32) outs(%123 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %padded_129 = tensor.pad %122 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
    %125 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_129, %cst_41 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%124 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %126 = tensor.empty() : tensor<1x28x28x128xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_42 : tensor<128xf32>) outs(%126 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %128 = tensor.empty() : tensor<1x28x28x128xf32>
    %129 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%128 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %130 = tensor.empty() : tensor<1x28x28x128xf32>
    %131 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%130 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %132 = tensor.empty() : tensor<1x28x28x512xf32>
    %133 = linalg.fill ins(%cst_0 : f32) outs(%132 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %134 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%131, %cst_45 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%133 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %135 = tensor.empty() : tensor<1x28x28x512xf32>
    %136 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_46 : tensor<512xf32>) outs(%135 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x512xf32>
    %137 = tensor.empty() : tensor<1x28x28x512xf32>
    %138 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134, %136 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%137 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %139 = tensor.empty() : tensor<1x28x28x512xf32>
    %140 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %138 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%139 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %141 = tensor.empty() : tensor<1x28x28x512xf32>
    %142 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%141 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %143 = tensor.empty() : tensor<1x28x28x128xf32>
    %144 = linalg.fill ins(%cst_0 : f32) outs(%143 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %145 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%142, %cst_47 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%144 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %146 = tensor.empty() : tensor<1x28x28x128xf32>
    %147 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_48 : tensor<128xf32>) outs(%146 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %148 = tensor.empty() : tensor<1x28x28x128xf32>
    %149 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %147 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%148 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %150 = tensor.empty() : tensor<1x28x28x128xf32>
    %151 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%150 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %152 = tensor.empty() : tensor<1x28x28x128xf32>
    %153 = linalg.fill ins(%cst_0 : f32) outs(%152 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %padded_130 = tensor.pad %151 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
    %154 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_130, %cst_49 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%153 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %155 = tensor.empty() : tensor<1x28x28x128xf32>
    %156 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_50 : tensor<128xf32>) outs(%155 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %157 = tensor.empty() : tensor<1x28x28x128xf32>
    %158 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154, %156 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%157 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %159 = tensor.empty() : tensor<1x28x28x128xf32>
    %160 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%158, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%159 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %161 = tensor.empty() : tensor<1x28x28x512xf32>
    %162 = linalg.fill ins(%cst_0 : f32) outs(%161 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %163 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%160, %cst_51 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%162 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %164 = tensor.empty() : tensor<1x28x28x512xf32>
    %165 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_52 : tensor<512xf32>) outs(%164 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x512xf32>
    %166 = tensor.empty() : tensor<1x28x28x512xf32>
    %167 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163, %165 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%166 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %168 = tensor.empty() : tensor<1x28x28x512xf32>
    %169 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%167, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%168 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %170 = tensor.empty() : tensor<1x28x28x128xf32>
    %171 = linalg.fill ins(%cst_0 : f32) outs(%170 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %172 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%169, %cst_53 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%171 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %173 = tensor.empty() : tensor<1x28x28x128xf32>
    %174 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_54 : tensor<128xf32>) outs(%173 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %175 = tensor.empty() : tensor<1x28x28x128xf32>
    %176 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172, %174 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%175 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %177 = tensor.empty() : tensor<1x28x28x128xf32>
    %178 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%177 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %179 = tensor.empty() : tensor<1x28x28x128xf32>
    %180 = linalg.fill ins(%cst_0 : f32) outs(%179 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %padded_131 = tensor.pad %178 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
    %181 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_131, %cst_55 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%180 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %182 = tensor.empty() : tensor<1x28x28x128xf32>
    %183 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_56 : tensor<128xf32>) outs(%182 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %184 = tensor.empty() : tensor<1x28x28x128xf32>
    %185 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181, %183 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%184 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %186 = tensor.empty() : tensor<1x28x28x128xf32>
    %187 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%186 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %188 = tensor.empty() : tensor<1x28x28x512xf32>
    %189 = linalg.fill ins(%cst_0 : f32) outs(%188 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %190 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%187, %cst_57 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%189 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %191 = tensor.empty() : tensor<1x28x28x512xf32>
    %192 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_58 : tensor<512xf32>) outs(%191 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x512xf32>
    %193 = tensor.empty() : tensor<1x28x28x512xf32>
    %194 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%190, %192 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%193 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %195 = tensor.empty() : tensor<1x28x28x512xf32>
    %196 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169, %194 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%195 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %197 = tensor.empty() : tensor<1x28x28x512xf32>
    %198 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%196, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%197 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %199 = tensor.empty() : tensor<1x28x28x128xf32>
    %200 = linalg.fill ins(%cst_0 : f32) outs(%199 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %201 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%198, %cst_59 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%200 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %202 = tensor.empty() : tensor<1x28x28x128xf32>
    %203 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_60 : tensor<128xf32>) outs(%202 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %204 = tensor.empty() : tensor<1x28x28x128xf32>
    %205 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%201, %203 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%204 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %206 = tensor.empty() : tensor<1x28x28x128xf32>
    %207 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%205, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%206 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %208 = tensor.empty() : tensor<1x28x28x128xf32>
    %209 = linalg.fill ins(%cst_0 : f32) outs(%208 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %padded_132 = tensor.pad %207 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
    %210 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_132, %cst_61 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%209 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %211 = tensor.empty() : tensor<1x28x28x128xf32>
    %212 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_62 : tensor<128xf32>) outs(%211 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x128xf32>
    %213 = tensor.empty() : tensor<1x28x28x128xf32>
    %214 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210, %212 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%213 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %215 = tensor.empty() : tensor<1x28x28x128xf32>
    %216 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%214, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%215 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x128xf32>
    %217 = tensor.empty() : tensor<1x28x28x512xf32>
    %218 = linalg.fill ins(%cst_0 : f32) outs(%217 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %219 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%216, %cst_63 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%218 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %220 = tensor.empty() : tensor<1x28x28x512xf32>
    %221 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_64 : tensor<512xf32>) outs(%220 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x28x28x512xf32>
    %222 = tensor.empty() : tensor<1x28x28x512xf32>
    %223 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%219, %221 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%222 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %224 = tensor.empty() : tensor<1x28x28x512xf32>
    %225 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%198, %223 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%224 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %226 = tensor.empty() : tensor<1x28x28x512xf32>
    %227 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%225, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%226 : tensor<1x28x28x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x28x28x512xf32>
    %228 = tensor.empty() : tensor<1x14x14x1024xf32>
    %229 = linalg.fill ins(%cst_0 : f32) outs(%228 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %230 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%227, %cst_69 : tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) outs(%229 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %231 = tensor.empty() : tensor<1x14x14x1024xf32>
    %232 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_70 : tensor<1024xf32>) outs(%231 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %233 = tensor.empty() : tensor<1x14x14x1024xf32>
    %234 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %232 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%233 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %235 = tensor.empty() : tensor<1x14x14x256xf32>
    %236 = linalg.fill ins(%cst_0 : f32) outs(%235 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %237 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%227, %cst_65 : tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) outs(%236 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %238 = tensor.empty() : tensor<1x14x14x256xf32>
    %239 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_66 : tensor<256xf32>) outs(%238 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %240 = tensor.empty() : tensor<1x14x14x256xf32>
    %241 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%237, %239 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%240 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %242 = tensor.empty() : tensor<1x14x14x256xf32>
    %243 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%242 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %244 = tensor.empty() : tensor<1x14x14x256xf32>
    %245 = linalg.fill ins(%cst_0 : f32) outs(%244 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_133 = tensor.pad %243 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %246 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_133, %cst_67 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%245 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %247 = tensor.empty() : tensor<1x14x14x256xf32>
    %248 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_68 : tensor<256xf32>) outs(%247 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %249 = tensor.empty() : tensor<1x14x14x256xf32>
    %250 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%246, %248 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%249 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %251 = tensor.empty() : tensor<1x14x14x256xf32>
    %252 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%250, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%251 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %253 = tensor.empty() : tensor<1x14x14x1024xf32>
    %254 = linalg.fill ins(%cst_0 : f32) outs(%253 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %255 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%252, %cst_71 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%254 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %256 = tensor.empty() : tensor<1x14x14x1024xf32>
    %257 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_72 : tensor<1024xf32>) outs(%256 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %258 = tensor.empty() : tensor<1x14x14x1024xf32>
    %259 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%255, %257 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%258 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %260 = tensor.empty() : tensor<1x14x14x1024xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234, %259 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%260 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %262 = tensor.empty() : tensor<1x14x14x1024xf32>
    %263 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%262 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %264 = tensor.empty() : tensor<1x14x14x256xf32>
    %265 = linalg.fill ins(%cst_0 : f32) outs(%264 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %266 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%263, %cst_73 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%265 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %267 = tensor.empty() : tensor<1x14x14x256xf32>
    %268 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_74 : tensor<256xf32>) outs(%267 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %269 = tensor.empty() : tensor<1x14x14x256xf32>
    %270 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%266, %268 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%269 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %271 = tensor.empty() : tensor<1x14x14x256xf32>
    %272 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%270, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%271 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %273 = tensor.empty() : tensor<1x14x14x256xf32>
    %274 = linalg.fill ins(%cst_0 : f32) outs(%273 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_134 = tensor.pad %272 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %275 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_134, %cst_75 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%274 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %276 = tensor.empty() : tensor<1x14x14x256xf32>
    %277 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_76 : tensor<256xf32>) outs(%276 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %278 = tensor.empty() : tensor<1x14x14x256xf32>
    %279 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%275, %277 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%278 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %280 = tensor.empty() : tensor<1x14x14x256xf32>
    %281 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%280 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %282 = tensor.empty() : tensor<1x14x14x1024xf32>
    %283 = linalg.fill ins(%cst_0 : f32) outs(%282 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %284 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%281, %cst_77 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%283 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %285 = tensor.empty() : tensor<1x14x14x1024xf32>
    %286 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_78 : tensor<1024xf32>) outs(%285 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %287 = tensor.empty() : tensor<1x14x14x1024xf32>
    %288 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284, %286 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%287 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %289 = tensor.empty() : tensor<1x14x14x1024xf32>
    %290 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%263, %288 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%289 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %291 = tensor.empty() : tensor<1x14x14x1024xf32>
    %292 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%291 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %293 = tensor.empty() : tensor<1x14x14x256xf32>
    %294 = linalg.fill ins(%cst_0 : f32) outs(%293 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %295 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%292, %cst_79 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%294 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %296 = tensor.empty() : tensor<1x14x14x256xf32>
    %297 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_80 : tensor<256xf32>) outs(%296 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %298 = tensor.empty() : tensor<1x14x14x256xf32>
    %299 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295, %297 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%298 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %300 = tensor.empty() : tensor<1x14x14x256xf32>
    %301 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%300 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %302 = tensor.empty() : tensor<1x14x14x256xf32>
    %303 = linalg.fill ins(%cst_0 : f32) outs(%302 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_135 = tensor.pad %301 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %304 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_135, %cst_81 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%303 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %305 = tensor.empty() : tensor<1x14x14x256xf32>
    %306 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_82 : tensor<256xf32>) outs(%305 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %307 = tensor.empty() : tensor<1x14x14x256xf32>
    %308 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304, %306 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%307 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %309 = tensor.empty() : tensor<1x14x14x256xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%309 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %311 = tensor.empty() : tensor<1x14x14x1024xf32>
    %312 = linalg.fill ins(%cst_0 : f32) outs(%311 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %313 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%310, %cst_83 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%312 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %314 = tensor.empty() : tensor<1x14x14x1024xf32>
    %315 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_84 : tensor<1024xf32>) outs(%314 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %316 = tensor.empty() : tensor<1x14x14x1024xf32>
    %317 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%313, %315 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%316 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %318 = tensor.empty() : tensor<1x14x14x1024xf32>
    %319 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%292, %317 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%318 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %320 = tensor.empty() : tensor<1x14x14x1024xf32>
    %321 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%319, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%320 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %322 = tensor.empty() : tensor<1x14x14x256xf32>
    %323 = linalg.fill ins(%cst_0 : f32) outs(%322 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %324 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%321, %cst_85 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%323 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %325 = tensor.empty() : tensor<1x14x14x256xf32>
    %326 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_86 : tensor<256xf32>) outs(%325 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %327 = tensor.empty() : tensor<1x14x14x256xf32>
    %328 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %326 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%327 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %329 = tensor.empty() : tensor<1x14x14x256xf32>
    %330 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%328, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%329 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %331 = tensor.empty() : tensor<1x14x14x256xf32>
    %332 = linalg.fill ins(%cst_0 : f32) outs(%331 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_136 = tensor.pad %330 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %333 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_136, %cst_87 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%332 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %334 = tensor.empty() : tensor<1x14x14x256xf32>
    %335 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_88 : tensor<256xf32>) outs(%334 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %336 = tensor.empty() : tensor<1x14x14x256xf32>
    %337 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333, %335 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%336 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %338 = tensor.empty() : tensor<1x14x14x256xf32>
    %339 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%338 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %340 = tensor.empty() : tensor<1x14x14x1024xf32>
    %341 = linalg.fill ins(%cst_0 : f32) outs(%340 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %342 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%339, %cst_89 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%341 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %343 = tensor.empty() : tensor<1x14x14x1024xf32>
    %344 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_90 : tensor<1024xf32>) outs(%343 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %345 = tensor.empty() : tensor<1x14x14x1024xf32>
    %346 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%342, %344 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%345 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %347 = tensor.empty() : tensor<1x14x14x1024xf32>
    %348 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321, %346 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%347 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %349 = tensor.empty() : tensor<1x14x14x1024xf32>
    %350 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%349 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %351 = tensor.empty() : tensor<1x14x14x256xf32>
    %352 = linalg.fill ins(%cst_0 : f32) outs(%351 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %353 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%350, %cst_91 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%352 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %354 = tensor.empty() : tensor<1x14x14x256xf32>
    %355 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_92 : tensor<256xf32>) outs(%354 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %356 = tensor.empty() : tensor<1x14x14x256xf32>
    %357 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %355 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%356 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %358 = tensor.empty() : tensor<1x14x14x256xf32>
    %359 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%358 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %360 = tensor.empty() : tensor<1x14x14x256xf32>
    %361 = linalg.fill ins(%cst_0 : f32) outs(%360 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_137 = tensor.pad %359 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %362 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_137, %cst_93 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%361 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %363 = tensor.empty() : tensor<1x14x14x256xf32>
    %364 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_94 : tensor<256xf32>) outs(%363 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %365 = tensor.empty() : tensor<1x14x14x256xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%362, %364 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%365 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %367 = tensor.empty() : tensor<1x14x14x256xf32>
    %368 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%366, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%367 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %369 = tensor.empty() : tensor<1x14x14x1024xf32>
    %370 = linalg.fill ins(%cst_0 : f32) outs(%369 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %371 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%368, %cst_95 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%370 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %372 = tensor.empty() : tensor<1x14x14x1024xf32>
    %373 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_96 : tensor<1024xf32>) outs(%372 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %374 = tensor.empty() : tensor<1x14x14x1024xf32>
    %375 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371, %373 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%374 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %376 = tensor.empty() : tensor<1x14x14x1024xf32>
    %377 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%350, %375 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%376 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %378 = tensor.empty() : tensor<1x14x14x1024xf32>
    %379 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%377, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%378 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %380 = tensor.empty() : tensor<1x14x14x256xf32>
    %381 = linalg.fill ins(%cst_0 : f32) outs(%380 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %382 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%379, %cst_97 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%381 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %383 = tensor.empty() : tensor<1x14x14x256xf32>
    %384 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_98 : tensor<256xf32>) outs(%383 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %385 = tensor.empty() : tensor<1x14x14x256xf32>
    %386 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %384 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%385 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %387 = tensor.empty() : tensor<1x14x14x256xf32>
    %388 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%387 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %389 = tensor.empty() : tensor<1x14x14x256xf32>
    %390 = linalg.fill ins(%cst_0 : f32) outs(%389 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %padded_138 = tensor.pad %388 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %391 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_138, %cst_99 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%390 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %392 = tensor.empty() : tensor<1x14x14x256xf32>
    %393 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_100 : tensor<256xf32>) outs(%392 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x256xf32>
    %394 = tensor.empty() : tensor<1x14x14x256xf32>
    %395 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%391, %393 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%394 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %396 = tensor.empty() : tensor<1x14x14x256xf32>
    %397 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%396 : tensor<1x14x14x256xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x256xf32>
    %398 = tensor.empty() : tensor<1x14x14x1024xf32>
    %399 = linalg.fill ins(%cst_0 : f32) outs(%398 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %400 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%397, %cst_101 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%399 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %401 = tensor.empty() : tensor<1x14x14x1024xf32>
    %402 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_102 : tensor<1024xf32>) outs(%401 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x14x14x1024xf32>
    %403 = tensor.empty() : tensor<1x14x14x1024xf32>
    %404 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %402 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%403 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %405 = tensor.empty() : tensor<1x14x14x1024xf32>
    %406 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%379, %404 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%405 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %407 = tensor.empty() : tensor<1x14x14x1024xf32>
    %408 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%407 : tensor<1x14x14x1024xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x14x14x1024xf32>
    %409 = tensor.empty() : tensor<1x7x7x2048xf32>
    %410 = linalg.fill ins(%cst_0 : f32) outs(%409 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %411 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%408, %cst_107 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) outs(%410 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %412 = tensor.empty() : tensor<1x7x7x2048xf32>
    %413 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_108 : tensor<2048xf32>) outs(%412 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x2048xf32>
    %414 = tensor.empty() : tensor<1x7x7x2048xf32>
    %415 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411, %413 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%414 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %416 = tensor.empty() : tensor<1x7x7x512xf32>
    %417 = linalg.fill ins(%cst_0 : f32) outs(%416 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %418 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%408, %cst_103 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) outs(%417 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %419 = tensor.empty() : tensor<1x7x7x512xf32>
    %420 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_104 : tensor<512xf32>) outs(%419 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %421 = tensor.empty() : tensor<1x7x7x512xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418, %420 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%421 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %423 = tensor.empty() : tensor<1x7x7x512xf32>
    %424 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%422, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%423 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %425 = tensor.empty() : tensor<1x7x7x512xf32>
    %426 = linalg.fill ins(%cst_0 : f32) outs(%425 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %padded_139 = tensor.pad %424 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
    %427 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_139, %cst_105 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%426 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %428 = tensor.empty() : tensor<1x7x7x512xf32>
    %429 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_106 : tensor<512xf32>) outs(%428 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %430 = tensor.empty() : tensor<1x7x7x512xf32>
    %431 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%427, %429 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%430 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %432 = tensor.empty() : tensor<1x7x7x512xf32>
    %433 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%432 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %434 = tensor.empty() : tensor<1x7x7x2048xf32>
    %435 = linalg.fill ins(%cst_0 : f32) outs(%434 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %436 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%433, %cst_109 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%435 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %437 = tensor.empty() : tensor<1x7x7x2048xf32>
    %438 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_110 : tensor<2048xf32>) outs(%437 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x2048xf32>
    %439 = tensor.empty() : tensor<1x7x7x2048xf32>
    %440 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %438 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%439 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %441 = tensor.empty() : tensor<1x7x7x2048xf32>
    %442 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %440 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%441 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %443 = tensor.empty() : tensor<1x7x7x2048xf32>
    %444 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%443 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %445 = tensor.empty() : tensor<1x7x7x512xf32>
    %446 = linalg.fill ins(%cst_0 : f32) outs(%445 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %447 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%444, %cst_111 : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%446 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %448 = tensor.empty() : tensor<1x7x7x512xf32>
    %449 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_112 : tensor<512xf32>) outs(%448 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %450 = tensor.empty() : tensor<1x7x7x512xf32>
    %451 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %449 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%450 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %452 = tensor.empty() : tensor<1x7x7x512xf32>
    %453 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%451, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%452 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %454 = tensor.empty() : tensor<1x7x7x512xf32>
    %455 = linalg.fill ins(%cst_0 : f32) outs(%454 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %padded_140 = tensor.pad %453 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
    %456 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_140, %cst_113 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%455 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %457 = tensor.empty() : tensor<1x7x7x512xf32>
    %458 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_114 : tensor<512xf32>) outs(%457 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %459 = tensor.empty() : tensor<1x7x7x512xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %458 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%459 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %461 = tensor.empty() : tensor<1x7x7x512xf32>
    %462 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%461 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %463 = tensor.empty() : tensor<1x7x7x2048xf32>
    %464 = linalg.fill ins(%cst_0 : f32) outs(%463 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %465 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%462, %cst_115 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%464 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %466 = tensor.empty() : tensor<1x7x7x2048xf32>
    %467 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_116 : tensor<2048xf32>) outs(%466 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x2048xf32>
    %468 = tensor.empty() : tensor<1x7x7x2048xf32>
    %469 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%465, %467 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%468 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %470 = tensor.empty() : tensor<1x7x7x2048xf32>
    %471 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444, %469 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%470 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %472 = tensor.empty() : tensor<1x7x7x2048xf32>
    %473 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%472 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %474 = tensor.empty() : tensor<1x7x7x512xf32>
    %475 = linalg.fill ins(%cst_0 : f32) outs(%474 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %476 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%473, %cst_117 : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%475 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %477 = tensor.empty() : tensor<1x7x7x512xf32>
    %478 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_118 : tensor<512xf32>) outs(%477 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %479 = tensor.empty() : tensor<1x7x7x512xf32>
    %480 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %478 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%479 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %481 = tensor.empty() : tensor<1x7x7x512xf32>
    %482 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%481 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %483 = tensor.empty() : tensor<1x7x7x512xf32>
    %484 = linalg.fill ins(%cst_0 : f32) outs(%483 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %padded_141 = tensor.pad %482 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
    %485 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_141, %cst_119 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%484 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %486 = tensor.empty() : tensor<1x7x7x512xf32>
    %487 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_120 : tensor<512xf32>) outs(%486 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x512xf32>
    %488 = tensor.empty() : tensor<1x7x7x512xf32>
    %489 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%485, %487 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%488 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %490 = tensor.empty() : tensor<1x7x7x512xf32>
    %491 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%490 : tensor<1x7x7x512xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x512xf32>
    %492 = tensor.empty() : tensor<1x7x7x2048xf32>
    %493 = linalg.fill ins(%cst_0 : f32) outs(%492 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %494 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%491, %cst_121 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%493 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %495 = tensor.empty() : tensor<1x7x7x2048xf32>
    %496 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_122 : tensor<2048xf32>) outs(%495 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x7x7x2048xf32>
    %497 = tensor.empty() : tensor<1x7x7x2048xf32>
    %498 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %496 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%497 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %499 = tensor.empty() : tensor<1x7x7x2048xf32>
    %500 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %498 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%499 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %501 = tensor.empty() : tensor<1x7x7x2048xf32>
    %502 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%501 : tensor<1x7x7x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.maxf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x7x7x2048xf32>
    %503 = tensor.empty() : tensor<1x2048xf32>
    %504 = linalg.fill ins(%cst_0 : f32) outs(%503 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %505 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%502 : tensor<1x7x7x2048xf32>) outs(%504 : tensor<1x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %529 = arith.addf %out, %in : f32
      linalg.yield %529 : f32
    } -> tensor<1x2048xf32>
    %506 = tensor.empty() : tensor<1x2048xf32>
    %507 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%505, %cst_7 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%506 : tensor<1x2048xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.divf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x2048xf32>
    %508 = tensor.empty() : tensor<1x1000xf32>
    %509 = linalg.fill ins(%cst_0 : f32) outs(%508 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %510 = linalg.matmul ins(%507, %cst_123 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%509 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %expanded = tensor.expand_shape %cst_124 [[0, 1]] : tensor<1000xf32> into tensor<1x1000xf32>
    %511 = tensor.empty() : tensor<1x1000xf32>
    %512 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%510, %expanded : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%511 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.addf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x1000xf32>
    %513 = tensor.empty() : tensor<1xf32>
    %514 = linalg.fill ins(%cst : f32) outs(%513 : tensor<1xf32>) -> tensor<1xf32>
    %515 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%512 : tensor<1x1000xf32>) outs(%514 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %529 = arith.maxf %out, %in : f32
      linalg.yield %529 : f32
    } -> tensor<1xf32>
    %516 = tensor.empty() : tensor<1x1000xf32>
    %517 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%515 : tensor<1xf32>) outs(%516 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1000xf32>
    %518 = tensor.empty() : tensor<1x1000xf32>
    %519 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%512, %517 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%518 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.subf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x1000xf32>
    %520 = tensor.empty() : tensor<1x1000xf32>
    %521 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%519 : tensor<1x1000xf32>) outs(%520 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %529 = math.exp %in : f32
      linalg.yield %529 : f32
    } -> tensor<1x1000xf32>
    %522 = tensor.empty() : tensor<1xf32>
    %523 = linalg.fill ins(%cst_0 : f32) outs(%522 : tensor<1xf32>) -> tensor<1xf32>
    %524 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%521 : tensor<1x1000xf32>) outs(%523 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %529 = arith.addf %out, %in : f32
      linalg.yield %529 : f32
    } -> tensor<1xf32>
    %525 = tensor.empty() : tensor<1x1000xf32>
    %526 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%524 : tensor<1xf32>) outs(%525 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1000xf32>
    %527 = tensor.empty() : tensor<1x1000xf32>
    %528 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%521, %526 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%527 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %529 = arith.divf %in, %in_142 : f32
      linalg.yield %529 : f32
    } -> tensor<1x1000xf32>
    return %528 : tensor<1x1000xf32>
  }

  func.func @entry() {
    %cst = arith.constant 0.0 : f32
    %input_tensor = bufferization.alloc_tensor() : tensor<1x224x224x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%input_tensor : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>

    %t = perf.start_timer : !perf.timer
    %result = call @resnet50v1(%1) : (tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> 
    %delta = perf.stop_timer(%t : !perf.timer) : f64

    %to_print = tensor.cast %result : tensor<1x1000xf32> to tensor<*xf32>
    call @printMemrefF32(%to_print) : (tensor<*xf32>) -> ()
    call @printF64(%delta) : (f64) -> ()
    return
  }

  func.func private @printF64(%delta : f64) 
  func.func private @printMemrefF32(%ptr : tensor<*xf32>) attributes {llvm.emit_c_interface}
} // ModuleOp

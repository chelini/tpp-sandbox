//===- LinalgMake2D.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct LinalgMake2D : LinalgMake2DBase<LinalgMake2D> {
  void runOnOperation() override {
    {
      RewritePatternSet patterns(&getContext());
      linalg::populateLinalg2DPatterns(patterns);
      // Fold unit-extent dims for linalg on tensors. Since
      // `populateFoldUnitExtentDimsViaSlicesPatterns` works only with
      // linalg.generic we need to generalize first using
      // `populateLinalgNamedOpsGeneralizationPatterns`.
      linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
      linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&getContext());
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }
};

struct TransposeOp2DPattern : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!transposeOp.hasTensorSemantics())
      return failure();

    // Already tiled.
    // (lorenzo) scf.forallOp should have `LoopLikeOpInterface`.
    if (transposeOp->getParentOfType<LoopLikeOpInterface>() ||
        transposeOp->getParentOfType<scf::ForallOp>()) {
      return failure();
    }
#if 0
    // The original idea here was to preserve the transpose by
    // tiling the non-transposed dimensions but this introduce
    // non-unit stride in the fastet varying dimension and it 
    // fails the verifier for tpp.
    ArrayRef<int64_t> permutation = transposeOp.getPermutation();
    int64_t permutedElements = 0;
    llvm::SmallSet<int64_t, 2> nonPermutedPositions;
    int64_t rankSource =
        transposeOp.getInput().getType().cast<ShapedType>().getRank();
    auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, rankSource));
    for (int64_t i = 0; i < rankSource; i++) {
      if (sequence[i] != permutation[i])
        permutedElements++;
      else
        nonPermutedPositions.insert(i);
    }

    // We cannot map to a 2d transpose.
    if (permutedElements != 2)
      return failure();

    // Tile with a factor of 1 the non-permuted positions.
    auto *ctx = rewriter.getContext();
    SmallVector<OpFoldResult> tiles(rankSource, getAsIndexOpFoldResult(ctx, 0));
    // for (int64_t i = 0; i < rankSource; i++) {
    //   if (nonPermutedPositions.count(i))
    //     tiles[i] = getAsIndexOpFoldResult(ctx, 1);
    // }
#endif
    int64_t rankSource =
        transposeOp.getInput().getType().cast<ShapedType>().getRank();
    auto *ctx = rewriter.getContext();
    SmallVector<OpFoldResult> tiles(rankSource, getAsIndexOpFoldResult(ctx, 0));
    tiles[0] = getAsIndexOpFoldResult(ctx, 1);
    tiles[1] = getAsIndexOpFoldResult(ctx, 1);

    FailureOr<linalg::ForallTilingResult> tiledOp =
        linalg::tileToForallOpUsingTileSizes(
            rewriter, cast<TilingInterface>(transposeOp.getOperation()), tiles,
            /*mapping=*/std::nullopt);
    if (failed(tiledOp))
      return failure();
    rewriter.replaceOp(transposeOp, tiledOp->tileOp->getResults());
    return success();
  }
};

} // namespace

void mlir::linalg::populateLinalg2DPatterns(RewritePatternSet &patterns) {
  patterns.add<TransposeOp2DPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::linalg::createLinalgMake2DPass() {
  return std::make_unique<LinalgMake2D>();
}

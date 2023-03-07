//===- TransformUtils.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMUTILS_H
#define TPP_TRANSFORMUTILS_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class Operation;
class OpBuilder;
struct Range;
class RewriterBase;
class TilingInterface;

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace linalgx {
namespace utils {

// Given an opOperand and a range of ivs return the one used by the operands.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs);

// Extract and return a slice for operand. Offsets are the induction variable
// touched by the operand. Sizes are: '1' in [0 to rank - desiredResultRank]
// while the full chunk in [rank - desiredResultRank to rank). Strides are
// assumed to be always 1. The methods effectively peel out the outermost [0 to
// rank - desiredResultRank] dimensions that are materialized as loops.
FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                 linalg::LinalgOp linalgOp, ValueRange ivs,
                                 ValueRange valuesToUse,
                                 unsigned desiredResultRank);

// Extract a slice of `operand` based on `offset`, `sizes` and
// `strides`.
Value getSliceOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                      Value operand, ArrayRef<OpFoldResult> offset,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      unsigned desiredResultRank);

// Return the loop range to materialize as loops from '0' to 'upTo'.
// '0' is the outermost loop.
FailureOr<SmallVector<Range>> getLoopsToMaterialize(RewriterBase &rewriter,
                                                    linalg::LinalgOp linalgOp,
                                                    unsigned upTo);
// Return true if the convolution is blocked.
enum class BlockedConvKind {
  NotABlockedConv = 0,
  BlockedConvWithBatchDim,
  BlockedConvWithoutBatchDim
};
// Checks whether `op` conforms to a blocked convolution with the following
// block layout: [N][K’][P][Q][k] += [N][C’][H][W][c] * [K’][C’][R][S][c][k].
// The method populates `dimensions` with indexes of the different kinds of
// dimensions when present.
BlockedConvKind isaBlockedConvolutionOpInterface(
    Operation *op, linalg::detail::ConvolutionDimensions *dimensions = nullptr);

// Return true if the matmul is blocked.
bool isBlockedMatmul(Operation *op);

// Validate a tile configuration for a linalgOp when we can statically do that.
// Specific dims can be passed using 'dims'. If dims is empty the validation
// will start from the outermost dimension, moving to innermost ones up to the
// number of tiles.
bool validateFullTilesOnDims(TilingInterface tileOp,
                             ArrayRef<OpFoldResult> tiles,
                             ArrayRef<size_t> dims = {});

// Return the size of the range as int64_t if the range is constant.
std::optional<int64_t> getConstantRange(const Range &range);

// Given an affineExpr, return true if tha affine expression is:
// 1. AffineDimExpr
// 2. AffineDimExpr + AffineDimExpr
// 3. AffineDimExpr * AffineConstantExpr/AffineSymbolExpr + AffineDimExpr
// For case 3. return the multiplicative factor.
LogicalResult walkConvExpr(AffineExpr expr, AffineExpr &multiplicativeFactor);

} // namespace utils
} // namespace linalgx
} // namespace mlir

#endif

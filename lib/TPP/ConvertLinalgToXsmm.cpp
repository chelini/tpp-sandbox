//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "convert-linalg-to-xsmm"

namespace {

struct ConvertLinalgToXsmm
    : public ConvertLinalgToXsmmBase<ConvertLinalgToXsmm> {
  void runOnOperation() override;
};

namespace {
struct BrgemmInfo {
  unsigned m;
  unsigned n;
  unsigned k;
  unsigned batch;

  SmallVector<int64_t> stridesOnA;
  SmallVector<int64_t> stridesOnB;
  SmallVector<int64_t> stridesOnC;
};
} // namespace

// Return the position of `dim` in the codomain.
static std::optional<unsigned> getPosInCodomain(unsigned dim,
                                                OpOperand *operand,
                                                linalg::GenericOp genericOp) {
  return genericOp.getMatchingIndexingMap(operand).getResultPosition(
      getAffineDimExpr(dim, genericOp.getContext()));
}

// Check if the strides associated with `operand` are valid strides
// for XSMM:
// - Strides must be statically known.
// - Stride at pos `dim` must be 1. Dim is the fastest varying dimension
// in the BRGEMM.
static FailureOr<SmallVector<int64_t>> verifyStrides(OpOperand *operand,
                                                     unsigned dim) {
  auto operandType = operand->get().getType();
  if (!isa<MemRefType>(operandType))
    return failure();
  auto memref = cast<MemRefType>(operandType);

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Strides for dim: " << dim
                          << " on operand: " << operand->get() << "\n");
  LLVM_DEBUG(llvm::interleaveComma(strides, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (strides[dim] != 1) {
    LLVM_DEBUG(llvm::dbgs() << "VERIFY STRIDES: failed\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "VERIFY STRIDES: ok\n");
  return strides;
}

// Check if the given generic is mappable to a brgemm call.
// - It is a contraction, with:
// -- 1 m and n and 2 k dimensions.
// -- m appears on the LHS and OUT but not in RHS.
// -- n appears on the RHS and OUT but not in LHS.
// -- k and k' appear on the RHS and LHS but not OUT.
// -- the stride of the minor dimension for A, k is 1.
// -- the stride of the minor dimension for B, j is 1.
// -- the stride of the minor dimension for C, j is 1.
static FailureOr<BrgemmInfo> isMappableToBrgemm(linalg::GenericOp genericOp) {
  if (!genericOp)
    return failure();
  auto contractionDims = linalgx::utils::isContraction(genericOp);
  if (failed(contractionDims))
    return failure();
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      contractionDims->k.size() != 2 || contractionDims->batch.size() != 0) {
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (genericOp.getNumLoops() != classifiedLoops)
    return failure();

  unsigned m = contractionDims->m[0];
  unsigned n = contractionDims->n[0];
  unsigned k = contractionDims->k.back();
  unsigned batch = contractionDims->k.front();

  LLVM_DEBUG(llvm::dbgs() << "Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "m: " << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "n: " << n << "\n");
  LLVM_DEBUG(llvm::dbgs() << "k: " << k << "\n");
  LLVM_DEBUG(llvm::dbgs() << "batch: " << batch << "\n");

  // A[I, K]
  OpOperand *operandA = genericOp.getDpsInputOperands()[0];
  auto posKInCodomain = getPosInCodomain(k, operandA, genericOp);
  if (!posKInCodomain)
    return failure();
  auto stridesOnA = verifyStrides(operandA, *posKInCodomain);
  if (failed(stridesOnA))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Strides on A: OK\n");

  // B[K, J]
  OpOperand *operandB = genericOp.getDpsInputOperands()[1];
  auto posNInCodomain = getPosInCodomain(n, operandB, genericOp);
  if (!posNInCodomain)
    return failure();
  auto stridesOnB = verifyStrides(operandB, *posNInCodomain);
  if (failed(stridesOnB))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Strides on B: OK\n");

  // C[I, J]
  OpOperand *operandC = genericOp.getDpsInitOperands()[0];
  posNInCodomain = getPosInCodomain(n, operandC, genericOp);
  if (!posNInCodomain)
    return failure();
  auto stridesOnC =
      verifyStrides(genericOp.getDpsInitOperands()[0], *posNInCodomain);
  if (failed(stridesOnC))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Strides on C: OK\n");

  BrgemmInfo info{m, n, k, batch, *stridesOnA, *stridesOnB, *stridesOnC};
  return info;
}

// Check if we can map `genericOp` to a BRGEMM and rewrite it to XSMM.
struct ConvertGenericToBrgemm : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasBufferSemantics())
      return failure();
    auto brgemmInfo = isMappableToBrgemm(genericOp);
    if (failed(brgemmInfo))
      return failure();

    auto loops = cast<linalg::LinalgOp>(genericOp.getOperation())
                     .computeStaticLoopSizes();
    auto operandA = genericOp.getDpsInputOperands()[0];
    auto operandB = genericOp.getDpsInputOperands()[1];
    auto operandC = genericOp.getDpsInitOperands()[0];

    unsigned m = brgemmInfo->m;
    unsigned n = brgemmInfo->n;
    unsigned k = brgemmInfo->k;
    unsigned batch = brgemmInfo->batch;
    auto mPosCodomainA = getPosInCodomain(m, operandA, genericOp);
    auto kPosCodomainB = getPosInCodomain(k, operandB, genericOp);
    auto mPosCodomainC = getPosInCodomain(m, operandC, genericOp);
    auto batchPosCodomainA = getPosInCodomain(batch, operandA, genericOp);
    auto batchPosCodomainB = getPosInCodomain(batch, operandB, genericOp);
    if (!mPosCodomainA || !kPosCodomainB || !mPosCodomainC ||
        !batchPosCodomainA || !batchPosCodomainB) {
      return failure();
    }
    int64_t lda = brgemmInfo->stridesOnA[*mPosCodomainA];
    int64_t ldb = brgemmInfo->stridesOnB[*kPosCodomainB];
    int64_t ldc = brgemmInfo->stridesOnC[*mPosCodomainC];
    int64_t strideA = brgemmInfo->stridesOnA[*batchPosCodomainA];
    int64_t strideB = brgemmInfo->stridesOnB[*batchPosCodomainB];

    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{loops[m], loops[n], loops[k], lda, ldb, ldc, strideA,
                          strideB});
    auto dtype =
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    Location loc = genericOp.getLoc();
    auto flags = rewriter.getArrayAttr(
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, flags, dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, loops[batch]));
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(genericOp->getOperands().begin(),
                          genericOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(genericOp, dtype,
                                                invokeOperands);
    return success();
  }
};

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Preliminary walk to understand if an operation is mappable to BRGEMM.
  // If this is the case, tile it and let `ConvertGenericToBrgemm` rewrite it.
  funcOp->walk([&](linalg::GenericOp genericOp) {
    if (!genericOp.hasBufferSemantics())
      return WalkResult::skip();
    // Verify it is a contraction with enough dimensions for BRGEMM.
    auto contractionDims = linalgx::utils::isContraction(genericOp);
    if (failed(contractionDims))
      return WalkResult::skip();
    if (contractionDims->m.size() < 1 || contractionDims->n.size() < 1 ||
        (contractionDims->k.size() != 2 && contractionDims->k.size() != 1)) {
      return WalkResult::skip();
    }
    unsigned classifiedLoops =
        contractionDims->m.size() + contractionDims->n.size() +
        contractionDims->k.size() + contractionDims->batch.size();
    if (genericOp.getNumLoops() != classifiedLoops)
      return WalkResult::skip();

    // Verify the strides on the minor dimensions (the dimension involved in the
    // BRGEMM). Fail if the stride is not 1.
    unsigned minorDimN = contractionDims->n.back();
    unsigned minorDimK = contractionDims->k.back();
    OpOperand *operandA = genericOp.getDpsInputOperands()[0];
    OpOperand *operandB = genericOp.getDpsInputOperands()[1];
    OpOperand *operandC = genericOp.getDpsInitOperands()[0];
    auto minorKInCodomainOpA = getPosInCodomain(minorDimK, operandA, genericOp);
    auto minorNInCodomainOpB = getPosInCodomain(minorDimN, operandB, genericOp);
    auto minorNInCodomainOpC = getPosInCodomain(minorDimN, operandC, genericOp);
    if (!minorKInCodomainOpA || !minorNInCodomainOpB || !minorNInCodomainOpC)
      return WalkResult::skip();

    if (failed(verifyStrides(operandA, *minorKInCodomainOpA)) ||
        failed(verifyStrides(operandB, *minorNInCodomainOpB)) ||
        failed(verifyStrides(operandC, *minorNInCodomainOpC))) {
      return WalkResult::skip();
    }
    AffineMap outputMap = genericOp.getMatchingIndexingMap(operandC);
    if (!outputMap.isProjectedPermutation())
      return WalkResult::skip();

    // At this point we know that the operation is mappable to BRGEMM
    // tile all the parallel loops.
    // Drop all the minor dimensions, as they are part of the BRGEMM.
    contractionDims->m.pop_back();
    contractionDims->n.pop_back();
    llvm::DenseSet<unsigned> otherParallelDims(contractionDims->m.begin(),
                                               contractionDims->m.end());
    otherParallelDims.insert(contractionDims->n.begin(),
                             contractionDims->n.end());
    otherParallelDims.insert(contractionDims->batch.begin(),
                             contractionDims->batch.end());

    SmallVector<OpFoldResult> tiles(
        genericOp.getNumLoops(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    for (auto expr : outputMap.getResults()) {
      auto dim = expr.cast<AffineDimExpr>().getPosition();
      if (otherParallelDims.count(dim))
        tiles[dim] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
    }

    if (llvm::all_of(tiles, [](OpFoldResult tile) {
          return isConstantIntValue(tile, 0);
        })) {
      return WalkResult::skip();
    }

    rewriter.setInsertionPoint(genericOp);
    FailureOr<linalg::ForallTilingResult> tiledOp =
        linalg::tileToForallOpUsingTileSizes(
            rewriter, cast<TilingInterface>(genericOp.getOperation()), tiles,
            /*mapping=*/std::nullopt);
    assert(succeeded(tiledOp));
    rewriter.replaceOp(genericOp, tiledOp->tiledOp->getResults());
    return WalkResult::advance();
  });
  LLVM_DEBUG(llvm::dbgs() << "=================================\n");

  RewritePatternSet patterns(ctx);
  tpp::populateLinalgToXsmmPatterns(patterns);
  linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGenericToBrgemm>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}

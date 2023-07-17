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
#include "mlir/Dialect/Utils/IndexingUtils.h"
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
// for XSMM: Strides must be statically known.
static FailureOr<SmallVector<int64_t>> verifyStrides(OpOperand *operand) {
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
  auto contractionDims = linalgx::utils::isContraction(genericOp);
  if (failed(contractionDims))
    return failure();
  OpOperand *operandA = genericOp.getDpsInputOperands()[0];
  OpOperand *operandB = genericOp.getDpsInputOperands()[1];
  OpOperand *operandC = genericOp.getDpsInitOperands()[0];

  if (!llvm::all_of(SmallVector<OpOperand *>{operandA, operandB, operandC},
                    [](OpOperand *operand) {
                      Type typeOperand = operand->get().getType();
                      if (!isa<ShapedType>(typeOperand))
                        return false;
                      auto shapedType = cast<ShapedType>(typeOperand);
                      return shapedType.hasStaticShape();
                    })) {
    return failure();
  }

  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1) ||
      contractionDims->batch.size() != 0) {
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
  unsigned batch = (contractionDims->k.size() == 2)
                       ? contractionDims->k.front()
                       : std::numeric_limits<unsigned>::max();

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] m: " << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] n: " << n << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] k: " << k << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] batch: " << batch << "\n");

  // A(i, k)
  auto posKInCodomain = getPosInCodomain(k, operandA, genericOp);
  if (!posKInCodomain)
    return failure();
  auto stridesOnA = verifyStrides(operandA);
  if (failed(stridesOnA) || (*stridesOnA)[*posKInCodomain] != 1)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on A: OK\n");

  // B(k, j)
  auto posNInCodomain = getPosInCodomain(n, operandB, genericOp);
  if (!posNInCodomain)
    return failure();
  auto stridesOnB = verifyStrides(operandB);
  if (failed(stridesOnB) || (*stridesOnB)[*posNInCodomain] != 1)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on B: OK\n");

  // C(i, j)
  posNInCodomain = getPosInCodomain(n, operandC, genericOp);
  if (!posNInCodomain)
    return failure();
  auto stridesOnC = verifyStrides(genericOp.getDpsInitOperands()[0]);
  if (failed(stridesOnC) || (*stridesOnC)[*posNInCodomain] != 1)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on C: OK\n");

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
    if (!mPosCodomainA || !kPosCodomainB || !mPosCodomainC) {
      return failure();
    }
    int64_t lda = brgemmInfo->stridesOnA[*mPosCodomainA];
    int64_t ldb = brgemmInfo->stridesOnB[*kPosCodomainB];
    int64_t ldc = brgemmInfo->stridesOnC[*mPosCodomainC];
    int64_t strideA = 1;
    if (batchPosCodomainA)
      strideA = brgemmInfo->stridesOnA[*batchPosCodomainA];
    int64_t strideB = 1;
    if (batchPosCodomainB)
      strideB = brgemmInfo->stridesOnB[*batchPosCodomainB];

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

    unsigned batchVal = 1;
    if (batch != std::numeric_limits<unsigned>::max())
      batchVal = loops[batch];
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchVal));
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

// Return true if the `genericOp` has enough parallel and reductions dimension
// for a BRGEMM operation.
static FailureOr<linalg::ContractionDimensions>
isBrgemmLike(linalg::GenericOp genericOp) {
  if (!genericOp.hasBufferSemantics())
    return failure();

  auto contractionDims = linalgx::utils::isContraction(genericOp);
  if (failed(contractionDims))
    return failure();
  if (contractionDims->m.size() < 1 || contractionDims->n.size() < 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1)) {
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (genericOp.getNumLoops() != classifiedLoops)
    return failure();
  return contractionDims;
}

// Emit a transpose operation for `operand` by swapping the dimensions at index
// `posMinorDim` with `newPosMinorDim`.
static void emitTransposeOnOperand(RewriterBase &rewriter,
                                   linalg::GenericOp genericOp,
                                   OpOperand *operand, unsigned posMinorDim,
                                   unsigned newPosMinorDim) {
  MemRefType operandType = operand->get().getType().cast<MemRefType>();
  auto rank = operandType.getRank();
  SmallVector<int64_t> shape = llvm::to_vector(operandType.getShape());
  auto permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  std::swap(permutation[posMinorDim], permutation[newPosMinorDim]);
  assert(isPermutationVector(permutation));
  LLVM_DEBUG(llvm::interleaveComma(
      permutation, llvm::dbgs() << "[emitTransposeOnOperand] Perm: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  rewriter.setInsertionPoint(genericOp);
  applyPermutationToVector<int64_t>(shape, permutation);
  Value alloc = rewriter.create<memref::AllocOp>(
      genericOp.getLoc(), MemRefType::get(shape, operandType.getElementType()));
  rewriter.create<linalg::TransposeOp>(genericOp.getLoc(), operand->get(),
                                       alloc, permutation);

  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  AffineMap operandMap = indexingMaps[operand->getOperandNumber()];
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] Old map: " << operandMap
                          << "\n");
  SmallVector<AffineExpr> mapResults = llvm::to_vector(operandMap.getResults());
  applyPermutationToVector<AffineExpr>(mapResults, permutation);
  AffineMap newMap =
      AffineMap::get(operandMap.getNumDims(), operandMap.getNumSymbols(),
                     mapResults, genericOp.getContext());
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] New map: " << newMap
                          << "\n");
  indexingMaps[operand->getOperandNumber()] = newMap;
  rewriter.updateRootInPlace(genericOp, [&]() {
    genericOp->setOperand(operand->getOperandNumber(), alloc);
    genericOp.setIndexingMapsAttr(
        ArrayAttr::get(genericOp.getContext(),
                       llvm::to_vector(llvm::map_range(
                           indexingMaps, [](AffineMap map) -> Attribute {
                             return AffineMapAttr::get(map);
                           }))));
  });
  rewriter.setInsertionPointAfter(genericOp);
  rewriter.create<memref::DeallocOp>(genericOp.getLoc(), alloc);
}

static bool isInnerMostDim(OpOperand *operand, unsigned minorDim) {
  MemRefType memref = operand->get().getType().cast<MemRefType>();
  unsigned rank = memref.getRank();
  return minorDim == rank - 1;
}

static FailureOr<linalg::GenericOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter, Operation *op,
                             unsigned minorDimM, unsigned minorDimN,
                             unsigned minorDimK) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return failure();

  OpOperand *operandA = genericOp.getDpsInputOperands()[0];
  OpOperand *operandB = genericOp.getDpsInputOperands()[1];
  OpOperand *operandC = genericOp.getDpsInitOperands()[0];

  // C(m,n) += A(m,k) * B(k,n)
  // n is expected to be the innermost for C
  // k is expected to be the innermost for A
  // n is expected to be the innermost for B
  auto minorKInCodomainOpA = getPosInCodomain(minorDimK, operandA, genericOp);
  auto minorMInCodomainOpA = getPosInCodomain(minorDimM, operandA, genericOp);
  if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for A\n");
    return failure();
  }

  auto minorNInCodomainOpB = getPosInCodomain(minorDimN, operandB, genericOp);
  auto minorKInCodomainOpB = getPosInCodomain(minorDimK, operandB, genericOp);
  if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for B\n");
    return failure();
  }

  auto minorNInCodomainOpC = getPosInCodomain(minorDimN, operandC, genericOp);
  auto minorMInCodomainOpC = getPosInCodomain(minorDimM, operandC, genericOp);
  if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for C\n");
    return failure();
  }

  if (!isInnerMostDim(operandC, *minorNInCodomainOpC)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for C\n");
    assert(isInnerMostDim(operandC, *minorMInCodomainOpC));
    if (isInnerMostDim(operandA, *minorKInCodomainOpA)) {
      emitTransposeOnOperand(rewriter, genericOp, operandA,
                             *minorKInCodomainOpA, *minorMInCodomainOpA);
    }
    if (isInnerMostDim(operandB, *minorNInCodomainOpB)) {
      emitTransposeOnOperand(rewriter, genericOp, operandB,
                             *minorNInCodomainOpB, *minorKInCodomainOpB);
    }
    // Avoid transpose on the output by swapping A and B.
    OpOperand *operandA = genericOp.getDpsInputOperands()[0];
    OpOperand *operandB = genericOp.getDpsInputOperands()[1];
    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    std::swap(indexingMaps[0], indexingMaps[1]);
    rewriter.updateRootInPlace(genericOp, [&]() {
      Value operandATmp = operandA->get();
      genericOp->setOperand(operandA->getOperandNumber(), operandB->get());
      genericOp->setOperand(operandB->getOperandNumber(), operandATmp);
      genericOp.setIndexingMapsAttr(
          ArrayAttr::get(genericOp.getContext(),
                         llvm::to_vector(llvm::map_range(
                             indexingMaps, [](AffineMap map) -> Attribute {
                               return AffineMapAttr::get(map);
                             }))));
    });
    return genericOp;
  }

  if (!isInnerMostDim(operandA, *minorKInCodomainOpA)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for A\n");
    assert(isInnerMostDim(operandA, *minorMInCodomainOpA));
    emitTransposeOnOperand(rewriter, genericOp, operandA, *minorKInCodomainOpA,
                           *minorMInCodomainOpA);
  }
  if (!isInnerMostDim(operandB, *minorNInCodomainOpB)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for B\n");
    assert(isInnerMostDim(operandB, *minorKInCodomainOpB));
    emitTransposeOnOperand(rewriter, genericOp, operandB, *minorKInCodomainOpB,
                           *minorNInCodomainOpB);
  }
  return genericOp;
}

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  auto res = funcOp->walk([&](linalg::GenericOp genericOp) {
    // Step1. The operation should be a contraction.
    auto contractionDims = isBrgemmLike(genericOp);
    if (failed(contractionDims)) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] not a contraction\n");
      return WalkResult::skip();
    }

    // Step2. Verify the strides on each operands. Need to be
    // statically known and at least one of the contraction
    // dimensions must be the fastest-varying one.
    OpOperand *operandA = genericOp.getDpsInputOperands()[0];
    OpOperand *operandB = genericOp.getDpsInputOperands()[1];
    OpOperand *operandC = genericOp.getDpsInitOperands()[0];
    auto stridesOnA = verifyStrides(operandA);
    auto stridesOnB = verifyStrides(operandB);
    auto stridesOnC = verifyStrides(operandC);

    if (failed(stridesOnA) || failed(stridesOnB) || failed(stridesOnC)) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] no constant strides\n");
      return WalkResult::skip();
    }

    unsigned minorDimM = contractionDims->m.back();
    unsigned minorDimN = contractionDims->n.back();
    unsigned minorDimK = contractionDims->k.back();
    auto minorKInCodomainOpA = getPosInCodomain(minorDimK, operandA, genericOp);
    auto minorMInCodomainOpA = getPosInCodomain(minorDimM, operandA, genericOp);
    if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for A\n");
      return WalkResult::skip();
    }
    if ((*stridesOnA)[*minorKInCodomainOpA] != 1 &&
        (*stridesOnA)[*minorMInCodomainOpA] != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Walk] minor dims for A are not fastest-varying\n");
      return WalkResult::skip();
    }

    auto minorNInCodomainOpB = getPosInCodomain(minorDimN, operandB, genericOp);
    auto minorKInCodomainOpB = getPosInCodomain(minorDimK, operandB, genericOp);
    if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for B\n");
      return WalkResult::skip();
    }
    if ((*stridesOnB)[*minorNInCodomainOpB] != 1 &&
        (*stridesOnB)[*minorKInCodomainOpB] != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Walk] minor dims for B are not fastest-varying\n");
      return WalkResult::skip();
    }

    auto minorNInCodomainOpC = getPosInCodomain(minorDimN, operandC, genericOp);
    auto minorMInCodomainOpC = getPosInCodomain(minorDimM, operandC, genericOp);
    if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for C\n");
      return WalkResult::skip();
    }
    if ((*stridesOnC)[*minorNInCodomainOpC] != 1 &&
        (*stridesOnC)[*minorMInCodomainOpC] != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Walk] minor dims for C are not fastest-varying\n");
      return WalkResult::skip();
    }

    AffineMap outputMap = genericOp.getMatchingIndexingMap(operandC);
    if (!outputMap.isProjectedPermutation()) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] expect permutation for output map\n");
      return WalkResult::skip();
    }

    // Step3. Tile all the parallel loops not involved in the contraction.
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

    // Step4. Check if we need to tile. If we don't emit transpose to bring the
    // GEMM in the canonical form: j fastest-varying for C and B and k
    // fastest-varying for A.
    if (llvm::all_of(tiles, [](OpFoldResult tile) {
          return isConstantIntValue(tile, 0);
        })) {
      if (failed(makeMinorDimensionsInnerMost(rewriter, genericOp, minorDimM,
                                              minorDimN, minorDimK)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    rewriter.setInsertionPoint(genericOp);
    FailureOr<linalg::ForallTilingResult> tiledOp =
        linalg::tileToForallOpUsingTileSizes(
            rewriter, cast<TilingInterface>(genericOp.getOperation()), tiles,
            /*mapping=*/std::nullopt);
    // Step4b. Tile and emit transposes.
    if ((failed(tiledOp)) ||
        failed(makeMinorDimensionsInnerMost(rewriter, tiledOp->tiledOp,
                                            minorDimM, minorDimN, minorDimK))) {
      return WalkResult::interrupt();
    }
    rewriter.replaceOp(genericOp, tiledOp->tileOp->getResults());
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return signalPassFailure();

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

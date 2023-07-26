//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
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

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;
};
} // namespace

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
static FailureOr<BrgemmInfo> isMappableToBrgemm(linalg::LinalgOp linalgOp) {
  if (!linalgOp.hasBufferSemantics())
    return failure();

  auto contractionDims = linalgx::utils::isContraction(linalgOp);
  if (failed(contractionDims))
    return failure();
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = linalgOp.getDpsInitOperands()[0];

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
  if (linalgOp.getNumLoops() != classifiedLoops)
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

  // A(m, k)
  auto posKInCodomain = linalgx::utils::getPosInCodomain(k, operandA, linalgOp);
  auto posMInCodomain = linalgx::utils::getPosInCodomain(m, operandA, linalgOp);
  if (!posKInCodomain || !posMInCodomain)
    return failure();
  auto stridesOnA = verifyStrides(operandA);
  if (failed(stridesOnA) || (*stridesOnA)[*posKInCodomain] != 1)
    return failure();
  int64_t lda = (*stridesOnA)[*posMInCodomain];
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on A: OK\n");

  // B(k, n)
  auto posNInCodomain = linalgx::utils::getPosInCodomain(n, operandB, linalgOp);
  posKInCodomain = linalgx::utils::getPosInCodomain(k, operandB, linalgOp);
  if (!posNInCodomain || !posKInCodomain)
    return failure();
  auto stridesOnB = verifyStrides(operandB);
  if (failed(stridesOnB) || (*stridesOnB)[*posNInCodomain] != 1)
    return failure();
  int64_t ldb = (*stridesOnB)[*posKInCodomain];
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on B: OK\n");

  // C(m, n)
  posNInCodomain = linalgx::utils::getPosInCodomain(n, operandC, linalgOp);
  posMInCodomain = linalgx::utils::getPosInCodomain(m, operandC, linalgOp);
  if (!posNInCodomain || !posMInCodomain)
    return failure();
  auto stridesOnC = verifyStrides(linalgOp.getDpsInitOperands()[0]);
  if (failed(stridesOnC) || (*stridesOnC)[*posNInCodomain] != 1)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on C: OK\n");
  int ldc = (*stridesOnC)[*posMInCodomain];

  auto batchPosCodomainA =
      linalgx::utils::getPosInCodomain(batch, operandA, linalgOp);
  auto batchPosCodomainB =
      linalgx::utils::getPosInCodomain(batch, operandB, linalgOp);
  int64_t strideA = 1;
  if (batchPosCodomainA)
    strideA = (*stridesOnA)[*batchPosCodomainA];
  int64_t strideB = 1;
  if (batchPosCodomainB)
    strideB = (*stridesOnB)[*batchPosCodomainB];

  BrgemmInfo info{m, n, k, batch, lda, ldb, ldc, strideA, strideB};
  return info;
}

static void replaceOpWithBrgemm(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                BrgemmInfo brgemmInfo) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto loops = linalgOp.computeStaticLoopSizes();
  unsigned m = brgemmInfo.m;
  unsigned n = brgemmInfo.n;
  unsigned k = brgemmInfo.k;
  unsigned batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;

  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{loops[m], loops[n], loops[k], lda, ldb, ldc, strideA,
                        strideB});
  // TODO: (lorenzo) support other element types.
  auto dtype =
      xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = linalgOp.getLoc();
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
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  invokeOperands.push_back(batchDim);
  rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(linalgOp, dtype, invokeOperands);
}

struct ConvertMatmulToMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto gemmInfo = isMappableToBrgemm(matmulOp);
    if (failed(gemmInfo))
      return failure();
    replaceOpWithBrgemm(rewriter, matmulOp, *gemmInfo);
    return success();
  }
};

// Check if we can map `genericOp` to a BRGEMM and rewrite it to XSMM.
struct ConvertGenericToBrgemm : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = isMappableToBrgemm(genericOp);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithBrgemm(rewriter, genericOp, *brgemmInfo);
    return success();
  }
};

struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp.hasBufferSemantics() || fillOp.hasDynamicShape())
      return failure();
    auto input = fillOp.getDpsInputOperands()[0];
    if (!tpp::utils::isZeroTensor(input->get()))
      return failure();
    auto output = fillOp.getDpsInitOperands()[0];
    ShapedType outputType = output->get().getType().cast<ShapedType>();
    auto outputRank = outputType.getRank();
    if (outputRank != 2)
      return failure();

    // Verify strides and minor dimensions.
    auto stridesOnOutput = verifyStrides(output);
    if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
      return failure();

    int64_t m = outputType.getShape()[0];
    int64_t n = outputType.getShape()[1];
    int64_t ldo = stridesOnOutput->front();
    // fillOp has a scalar input.
    int64_t ldi = 1;

    Location loc = fillOp.getLoc();
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, ldi, ldo});
    // TODO: (lorenzo) support other element types.
    auto dtype =
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
        loc, integer64, kind, dims, flags, dtype);
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(fillOp->getOperands().begin(),
                          fillOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(fillOp, dtype, kind,
                                               invokeOperands);
    return success();
  }
};

struct ConvertTransposeOpToUnaryTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!transposeOp.hasBufferSemantics() || transposeOp.hasDynamicShape())
      return failure();
    auto input = transposeOp.getDpsInputOperands()[0];
    auto output = transposeOp.getDpsInitOperands()[0];
    ShapedType inputType = input->get().getType().cast<ShapedType>();
    ShapedType outputType = output->get().getType().cast<ShapedType>();
    auto inputRank = inputType.getRank();
    auto outputRank = outputType.getRank();
    if (inputRank != 2 || outputRank != 2)
      return failure();

    auto stridesOnInput = verifyStrides(input);
    auto stridesOnOutput = verifyStrides(output);
    if (failed(stridesOnInput) || failed(stridesOnOutput))
      return failure();
    if (stridesOnInput->back() != 1 || stridesOnOutput->back() != 1)
      return failure();

    int64_t m = inputType.getShape()[0];
    int64_t n = inputType.getShape()[1];
    int64_t ldi = stridesOnInput->front();
    int64_t ldo = stridesOnOutput->front();

    Location loc = transposeOp.getLoc();
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, ldi, ldo});
    // TODO: (lorenzo) support other element types.
    auto dtype =
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
        loc, integer64, kind, dims, flags, dtype);
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(transposeOp->getOperands().begin(),
                          transposeOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(transposeOp, dtype, kind,
                                               invokeOperands);
    return success();
  }
};

// Return true if the `genericOp` has enough parallel and reductions dimension
// for a BRGEMM operation.
static FailureOr<linalg::ContractionDimensions>
isBrgemmLike(linalg::GenericOp genericOp) {
  if (!genericOp.hasBufferSemantics()) {
    LLVM_DEBUG(llvm::dbgs() << "[isBrgemmLike] not bufferized!\n");
    return failure();
  }

  auto contractionDims = linalgx::utils::isContraction(genericOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[isBrgemmLike] not a contraction\n");
    return failure();
  }
  if (contractionDims->m.size() < 1 || contractionDims->n.size() < 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1)) {
    LLVM_DEBUG(llvm::dbgs() << "[isBrgemmLike] not enough m, n and k dims\n");
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (genericOp.getNumLoops() != classifiedLoops) {
    LLVM_DEBUG(llvm::dbgs() << "[isBrgemmLike] not all loops classified\n");
    return failure();
  }
  return contractionDims;
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
    unsigned constexpr strideOne = 1;
    auto minorKInCodomainOpA =
        linalgx::utils::getPosInCodomain(minorDimK, operandA, genericOp);
    auto minorMInCodomainOpA =
        linalgx::utils::getPosInCodomain(minorDimM, operandA, genericOp);
    if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for A\n");
      return WalkResult::skip();
    }
    if ((*stridesOnA)[*minorKInCodomainOpA] != strideOne &&
        (*stridesOnA)[*minorMInCodomainOpA] != strideOne) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Walk] minor dims for A are not fastest-varying\n");
      return WalkResult::skip();
    }

    auto minorNInCodomainOpB =
        linalgx::utils::getPosInCodomain(minorDimN, operandB, genericOp);
    auto minorKInCodomainOpB =
        linalgx::utils::getPosInCodomain(minorDimK, operandB, genericOp);
    if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for B\n");
      return WalkResult::skip();
    }
    if ((*stridesOnB)[*minorNInCodomainOpB] != strideOne &&
        (*stridesOnB)[*minorKInCodomainOpB] != strideOne) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[Walk] minor dims for B are not fastest-varying\n");
      return WalkResult::skip();
    }

    auto minorNInCodomainOpC =
        linalgx::utils::getPosInCodomain(minorDimN, operandC, genericOp);
    auto minorMInCodomainOpC =
        linalgx::utils::getPosInCodomain(minorDimM, operandC, genericOp);
    if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
      LLVM_DEBUG(llvm::dbgs() << "[Walk] did not find minor dims for C\n");
      return WalkResult::skip();
    }
    if ((*stridesOnC)[*minorNInCodomainOpC] != strideOne &&
        (*stridesOnC)[*minorMInCodomainOpC] != strideOne) {
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
      if (failed(linalgx::utils::makeMinorDimensionsInnerMost(
              rewriter, genericOp, minorDimM, minorDimN, minorDimK)))
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
        failed(linalgx::utils::makeMinorDimensionsInnerMost(
            rewriter, cast<linalg::GenericOp>(tiledOp->tiledOp), minorDimM,
            minorDimN, minorDimK))) {
      return WalkResult::interrupt();
    }
    rewriter.replaceOp(genericOp, tiledOp->tileOp->getResults());
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return signalPassFailure();

  RewritePatternSet patterns(ctx);
  tpp::populateLinalgToXsmmPatterns(patterns);
  linalg::ControlDropUnitDims options;
  options.rankReductionStrategy =
      linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
  linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGenericToBrgemm, ConvertMatmulToMatmul/*,
               ConvertFillOpToUnaryZero, ConvertTransposeOpToUnaryTranspose*/>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}

//===- CombineTpp.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct FusedBrgemmOp {
  SmallVector<Value> operands;
  Value bias;
  tpp::FusedBinaryOpKindAttr binaryKind;
};

static FailureOr<tpp::FusedBinaryOpKindAttr>
getFusedBinaryAttr(Operation *operation, MLIRContext *ctx) {
  if (isa<tpp::AddOp>(operation))
    return tpp::FusedBinaryOpKindAttr::get(ctx, tpp::FusedBinaryOpKind::ADD);
  return failure();
}

// Insert a tensor expand shape by adding a leading 1.
static Value insertExpand(RewriterBase &rewriter, Location loc, Value operand) {
  assert(operand.getType().isa<RankedTensorType>());
  RankedTensorType sourceType = operand.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape = llvm::to_vector(sourceType.getShape());
  shape.insert(shape.begin(), 1);
  RankedTensorType destType =
      RankedTensorType::get(shape, sourceType.getElementType());
  auto reassociation = getReassociationIndicesForReshape(sourceType, destType);
  assert(reassociation &&
         "expect getReassociationIndicesForReshape not to fail");
  return rewriter.create<tensor::ExpandShapeOp>(loc, destType, operand,
                                                *reassociation);
}

// Add leading 1s to A and B, leave C as is.
static SmallVector<Value> reshapeOperandForBrgemm(RewriterBase &rewriter,
                                                  Location loc,
                                                  OperandRange operands) {
  assert(operands.size() == 3);
  Value operandA = insertExpand(rewriter, loc, operands.front());
  Value operandB = insertExpand(rewriter, loc, operands[1]);
  Value operandC = operands.back();
  return {operandA, operandB, operandC};
}

static FailureOr<FusedBrgemmOp> getFusedBrgemmInfo(RewriterBase &rewriter,
                                                   tpp::TppOp tppOp) {
  if (!tppOp.hasTensorSemantics() || !tppOp.isBinary())
    return failure();

  SmallVector<Value> brgemmOperands;
  Value biasOperand;
  bool hasBrgemmProducer = false;
  for (Value operand : tppOp.getInputs()) {
    if (auto brgemmOp = operand.getDefiningOp<tpp::BrgemmOp>()) {
      if (hasBrgemmProducer)
        return failure();
      brgemmOperands = brgemmOp.getInputs();
      hasBrgemmProducer = true;
      continue;
    }
    if (auto gemmOp = operand.getDefiningOp<tpp::GemmOp>()) {
      if (hasBrgemmProducer)
        return failure();
      brgemmOperands =
          reshapeOperandForBrgemm(rewriter, tppOp.getLoc(), gemmOp.getInputs());
      hasBrgemmProducer = true;
      continue;
    }
    biasOperand = operand;
  }
  if (!hasBrgemmProducer)
    return failure();
  auto binaryType = getFusedBinaryAttr(tppOp, rewriter.getContext());
  if (failed(binaryType))
    return failure();
  assert(brgemmOperands.size() == 3);
  return FusedBrgemmOp{brgemmOperands, biasOperand, *binaryType};
}

template <typename OpTy>
struct CombineBrgemmWithBinary : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy binaryOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = getFusedBrgemmInfo(rewriter, binaryOp);
    if (failed(brgemmInfo))
      return failure();
    auto unaryKind = tpp::FusedUnaryOpKindAttr::get(
        rewriter.getContext(), tpp::FusedUnaryOpKind::NONE);

    rewriter.replaceOpWithNewOp<tpp::FusedBrgemmOp>(
        binaryOp, brgemmInfo->operands, brgemmInfo->operands.back(),
        brgemmInfo->bias, unaryKind, brgemmInfo->binaryKind);
    return success();
  }
};

template <typename OpTy>
struct CombineBrgemmWithOptionalBinaryAndUnary : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  FailureOr<tpp::FusedUnaryOpKindAttr>
  getFusedUnaryAttr(Operation *operation, MLIRContext *ctx) const {
    if (isa<tpp::ReluOp>(operation))
      return tpp::FusedUnaryOpKindAttr::get(ctx, tpp::FusedUnaryOpKind::RELU);
    return failure();
  }

  FailureOr<tpp::FusedBinaryOpKindAttr>
  getFusedBinaryAttr(Operation *operation, MLIRContext *ctx) const {
    if (isa<tpp::AddOp>(operation))
      return tpp::FusedBinaryOpKindAttr::get(ctx, tpp::FusedBinaryOpKind::ADD);
    return failure();
  }

  LogicalResult matchAndRewrite(OpTy unaryOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<tpp::TppOp>(unaryOp.getOperation()))
      return failure();
    auto tppUnaryOp = cast<tpp::TppOp>(unaryOp.getOperation());
    if (!tppUnaryOp.hasTensorSemantics() || !tppUnaryOp.isUnary())
      return failure();
    Value operandUnary = tppUnaryOp.getInputs()[0];
    auto tppBinaryOp = operandUnary.getDefiningOp();
    if (!isa<tpp::TppOp>(tppBinaryOp))
      return failure();
    auto unaryKind = getFusedUnaryAttr(tppUnaryOp, rewriter.getContext());
    auto brgemmInfo =
        getFusedBrgemmInfo(rewriter, cast<tpp::TppOp>(tppBinaryOp));
    if (failed(unaryKind) || failed(brgemmInfo))
      return failure();
    rewriter.replaceOpWithNewOp<tpp::FusedBrgemmOp>(
        unaryOp, brgemmInfo->operands, brgemmInfo->operands.back(),
        brgemmInfo->bias, *unaryKind, brgemmInfo->binaryKind);
    return success();
  }
};

void populatePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineBrgemmWithOptionalBinaryAndUnary<tpp::ReluOp>,
               CombineBrgemmWithBinary<tpp::AddOp>>(patterns.getContext());
}

struct CombineTppOps : public CombineTppOpsBase<CombineTppOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createCombineTppPass() {
  return std::make_unique<CombineTppOps>();
}

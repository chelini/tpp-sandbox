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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

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
    if (!isa<tpp::TppOp>(tppBinaryOp) ||
        !cast<tpp::TppOp>(tppBinaryOp).isBinary())
      return failure();

    SmallVector<Value> brgemmOperands;
    Value addOperand;
    bool hasBrgemmProducer = false;
    for (Value operand : cast<tpp::TppOp>(tppBinaryOp).getInputs()) {
      if (auto brgemmOp = operand.getDefiningOp<tpp::BrgemmOp>()) {
        brgemmOperands = brgemmOp.getInputs();
        hasBrgemmProducer = true;
        continue;
      }
      addOperand = operand;
    }
    if (!hasBrgemmProducer)
      return failure();
    Value outputBrgemm = brgemmOperands[brgemmOperands.size() - 1];
    brgemmOperands.push_back(addOperand);
    auto ctx = rewriter.getContext();
    auto unaryType = getFusedUnaryAttr(tppUnaryOp, ctx);
    auto binaryType = getFusedBinaryAttr(tppBinaryOp, ctx);
    if (failed(unaryType) || failed(binaryType))
      return failure();
    rewriter.replaceOpWithNewOp<tpp::FusedBrgemmOp>(
        unaryOp, brgemmOperands, outputBrgemm, *unaryType, *binaryType);
    return success();
  }
};

void populatePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineBrgemmWithOptionalBinaryAndUnary<tpp::ReluOp>>(
      patterns.getContext());
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

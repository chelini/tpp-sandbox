//===- ConvInitSimplify.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVINITSIMPLIFY
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Returns true if: 1) the region has a single block. 2) The block has a single
// operation `OP`. 3) The operation result types are int or float.
template <typename OP> static bool hasOnlyOp(Region &region) {
  if (!region.hasOneBlock())
    return false;
  unsigned numberOfOpsInRegion = 2;
  if (std::is_same<OP, linalg::YieldOp>::value)
    numberOfOpsInRegion = 1;
  if (std::distance(region.front().begin(), region.front().end()) !=
      numberOfOpsInRegion)
    return false;
  for (Operation &op : region.front()) {
    if (!isa<OP, linalg::YieldOp>(op) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrFloat(); }))
      return false;
  }
  return true;
}

static bool isBroadCastOp(linalg::GenericOp linalgOp) {
  if (linalgOp->getNumOperands() != 2 || linalgOp->getNumResults() != 1)
    return false;
  if (!hasOnlyOp<linalg::YieldOp>(linalgOp.getRegion()))
    return false;
  SmallVector<unsigned> perm;
  return linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0))
      .isPermutationOfMinorIdentityWithBroadcasting(perm);
}

static std::optional<linalg::GenericOp> getBroadCastProdcuer(OpOperand *rhs) {
  linalg::GenericOp broadcastOp = rhs->get().getDefiningOp<linalg::GenericOp>();
  if (!broadcastOp || !isBroadCastOp(broadcastOp))
    return std::nullopt;
  return broadcastOp;
}

// Instead of initializing the output of a convolution with zero and then add a
// bias, initialize the output of the convolution with the bias.
struct EliminateZeroInitAndAddBiasToInit
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics())
      return failure();
    if (linalgOp->getNumOperands() != 3 || linalgOp->getNumResults() != 1)
      return failure();

    if (!linalg::isElementwise(linalgOp) ||
        !hasOnlyOp<arith::AddFOp>(linalgOp.getRegion()))
      return failure();

    OpOperand *lhs = linalgOp.getDpsInputOperand(0);
    OpOperand *rhs = linalgOp.getDpsInputOperand(1);

    auto convProducer = lhs->get().getDefiningOp<linalg::Conv2DNhwcHwcfOp>();
    auto broadCastProducer = getBroadCastProdcuer(rhs);
    if (!convProducer || !broadCastProducer ||
        !utils::isZeroTensor(convProducer.getDpsInitOperand(0)->get()) ||
        !convProducer.getTiedOpResult(convProducer.getDpsInitOperand(0)))
      return failure();

    SmallVector<Value> convInputs;
    for (OpOperand *operand : convProducer.getDpsInputOperands())
      convInputs.push_back(operand->get());
    Value broadCastOutput = broadCastProducer->getTiedOpResult(
        broadCastProducer->getDpsInitOperand(0));

    auto replOp = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        linalgOp.getLoc(), broadCastOutput.getType(), convInputs,
        broadCastOutput, convProducer.getStrides(),
        convProducer.getDilations());
    if (auto metadata = convProducer->getAttr("metadata"))
      replOp->setAttr("metadata", metadata);

    rewriter.replaceOp(linalgOp, replOp.getResults());
    return success();
  }
};

struct ConvInitSimplify
    : public tpp::impl::ConvInitSimplifyBase<ConvInitSimplify> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<EliminateZeroInitAndAddBiasToInit>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

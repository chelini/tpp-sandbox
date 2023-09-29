//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace xsmm {

namespace {

struct FusionOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          FusionOpBufferizationInterface, xsmm::FusionOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    auto fusionOp = cast<xsmm::FusionOp>(op);
    SmallVector<Type> newTypes;
    for (Value result : fusionOp->getResults()) {
      auto bufferType = bufferization::getBufferType(result, options);
      if (failed(bufferType))
        return failure();
      newTypes.push_back(*bufferType);
    }

    xsmm::FusionOp newFusionOp =
        rewriter.create<xsmm::FusionOp>(fusionOp.getLoc(), newTypes);
    Region &region = newFusionOp->getRegion(0);
    rewriter.createBlock(&region);
    rewriter.mergeBlocks(&fusionOp.getRegion().front(),
                         &newFusionOp.getRegion().front());
    // Operation *term = newFusionOp.getRegion().front().getTerminator();
    // xsmm::YieldOp yield = cast<xsmm::YieldOp>(term);
    // for (Value v : yield.getResults())
    //  v.dump();
    replaceOpWithBufferizedValues(rewriter, op, newFusionOp->getResults());

    return success();
  }
};

struct YieldOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          YieldOpBufferizationInterface, xsmm::YieldOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    auto yieldOp = cast<xsmm::YieldOp>(op);
    SmallVector<Value> newResults;
    for (Value val : yieldOp.getResults())
      llvm::errs() << val << "\n";
    newResults.append(yieldOp.getResults().begin(), yieldOp.getResults().end());
    replaceOpWithNewBufferizedOp<xsmm::YieldOp>(rewriter, op, newResults);
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, xsmm::XsmmDialect *dialect) {
    FusionOp::attachInterface<xsmm::FusionOpBufferizationInterface>(*ctx);
    YieldOp::attachInterface<xsmm::YieldOpBufferizationInterface>(*ctx);
  });
}

} // end namespace xsmm
} // namespace mlir

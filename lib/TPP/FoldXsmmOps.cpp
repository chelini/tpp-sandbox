//===- FoldXsmmOps.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

static void foldXsmmZero(RewriterBase &rewriter, Value val) {
  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(val, &forwardSlice);

  llvm::errs() << "Root: " << val << "\n";
  for (Operation *user : forwardSlice)
    llvm::errs() << "User " << *user << "\n";
}

struct FoldXsmmOps : public FoldXsmmOpsBase<FoldXsmmOps> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    funcOp->walk([&](xsmm::BrgemmOp brgemmOp) {
      foldXsmmZero(rewriter, brgemmOp.getInputs()[3]);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createFoldXsmmOpsPass() {
  return std::make_unique<FoldXsmmOps>();
}

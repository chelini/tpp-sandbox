//===- CanonicalizeContractions.cpp ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "canonicalize-contractions"

namespace {

struct CanonicalizeContractions
    : CanonicalizeContractionsBase<CanonicalizeContractions> {
  void runOnOperation() override;
};

// Return true if the `genericOp` has enough parallel and reductions dimension
// for a BRGEMM operation.
static FailureOr<linalg::ContractionDimensions>
isBrgemmLike(linalg::GenericOp genericOp) {
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

void CanonicalizeContractions::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  funcOp->walk([&](linalg::GenericOp genericOp) {
    auto contractionDims = isBrgemmLike(genericOp);
    if (failed(contractionDims))
      return;
    LLVM_DEBUG(llvm::dbgs() << "Candidate: " << genericOp << "\n");
    unsigned minorDimM = contractionDims->m.back();
    unsigned minorDimN = contractionDims->n.back();
    unsigned minorDimK = contractionDims->k.back();
    if (failed(linalgx::utils::makeMinorDimensionsInnerMost(
            rewriter, genericOp, minorDimM, minorDimN, minorDimK)))
      return;
  });
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createCanonicalizeContractionsPass() {
  return std::make_unique<CanonicalizeContractions>();
}

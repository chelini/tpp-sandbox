//===- Bufferize.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct Bufferize : public BufferizeBase<Bufferize> {
  Bufferize() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                check::CheckDialect,
                perf::PerfDialect,
                scf::SCFDialect,
                tpp::TppDialect,
                tensor::TensorDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    tpp::registerBufferizableOpInterfaceExternalModels(registry);
  }
  void runOnOperation() override;
};

struct ConvertToDestinationPassingStyle
    : public ConvertToDestinationPassingStyleBase<
          ConvertToDestinationPassingStyle> {
  ConvertToDestinationPassingStyle() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void Bufferize::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  OpPassManager passManager;

  // Pre-processing.
  passManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  // One-shot.
  bufferization::OneShotBufferizationOptions buffOpts;
  buffOpts.allowReturnAllocs = true;
  buffOpts.bufferizeFunctionBoundaries = true;
  buffOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bool runOnlyAnalysis = this->testAnalysisOnly || this->printConflicts;
  if (runOnlyAnalysis) {
    buffOpts.printConflicts = this->printConflicts;
    buffOpts.testAnalysisOnly = this->testAnalysisOnly;
  }
  passManager.addPass(bufferization::createOneShotBufferizePass(buffOpts));

  if (!runOnlyAnalysis) {
    passManager.addPass(bufferization::createDropEquivalentBufferResultsPass());
    passManager.addNestedPass<func::FuncOp>(
        bufferization::createFinalizingBufferizePass());

    // Post-processing.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
    // There are redundant memcpy (with linalg.generic form) ops created, which
    // can be deleted by canonicalizer. We have to run it again because the
    // memrefs are unified in CSE pass, so we can truly remove redundant memcpy.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }

  if (failed(runPipeline(passManager, moduleOp)))
    return signalPassFailure();
}

// Return true if `inputOperand` can be used to store in-place `initOperand`.
static bool canUseOperand(OpOperand *inputOperand, OpOperand *initOperand) {
  if (auto defOp = inputOperand->get().getDefiningOp<arith::ConstantOp>())
    return false;

  if (inputOperand->getOwner() != initOperand->getOwner())
    return false;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(inputOperand->getOwner());
  if (!linalgOp)
    return false;

  if (linalgOp.getMatchingIndexingMap(inputOperand) !=
      linalgOp.getMatchingIndexingMap(initOperand)) {
    return false;
  }

  return inputOperand->get().getType() == initOperand->get().getType();
}

// Return the input operand that can be used to store the result if any.
static std::optional<OpOperand *>
getReusableInputOperands(OpOperand *initOperand, linalg::LinalgOp linalgOp) {
  assert(linalgOp.isDpsInit(initOperand) && "Expect an init operand");

  // Used in the region, cannot do in-place.
  if (linalgOp.payloadUsesValueFromOperand(initOperand))
    return std::nullopt;

  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (canUseOperand(inputOperand, initOperand))
      return inputOperand;
  }
  return std::nullopt;
}

static void makeGenericOpInPlace(RewriterBase &rewriter, OpOperand *inOperand,
                                 OpOperand *initOperand) {
  auto genericOp = cast<linalg::GenericOp>(inOperand->getOwner());
  assert(genericOp == initOperand->getOwner() &&
         "expected in operand and out operand to be the same op");
  SmallVector<Value> newInputs;
  SmallVector<Value> newOutputs;
  SmallVector<Type> newResultTypes;
  SmallVector<AffineMap> maps;
  for (OpOperand *in : genericOp.getDpsInputOperands()) {
    if (in != inOperand) {
      newInputs.push_back(in->get());
      maps.push_back(genericOp.getMatchingIndexingMap(in));
    }
  }
  for (OpOperand *out : genericOp.getDpsInitOperands()) {
    maps.push_back(genericOp.getMatchingIndexingMap(out));
    if (initOperand == out) {
      newOutputs.push_back(inOperand->get());
      newResultTypes.push_back(inOperand->get().getType());
    } else {
      newOutputs.push_back(out->get());
      newResultTypes.push_back(out->get().getType());
    }
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(genericOp);

  Location loc = genericOp.getLoc();
  SmallVector<utils::IteratorType> iterTypes(genericOp.getNumLoops(),
                                             utils::IteratorType::parallel);
  auto newOp = rewriter.create<linalg::GenericOp>(
      loc, newResultTypes, newInputs, newOutputs, maps, iterTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(), newOp.getRegion(),
                              newOp.getRegion().begin());

  // Repair the payload entry block.
  Block &payload = newOp.getRegion().front();
  payload.getArgument(inOperand->getOperandNumber())
      .replaceAllUsesWith(payload.getArgument(initOperand->getOperandNumber()));
  payload.eraseArgument(inOperand->getOperandNumber());

  rewriter.replaceOp(genericOp, newOp.getResults());
}

static void adaptComputeConsumerToAvoidAllocation(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return WalkResult::skip();

    for (OpOperand *initOperand : linalgOp.getDpsInitOperands()) {
      std::optional<OpOperand *> reusableOperand =
          getReusableInputOperands(initOperand, linalgOp);
      if (!reusableOperand)
        continue;
      makeGenericOpInPlace(rewriter, *reusableOperand, initOperand);
    }
    return WalkResult::advance();
  });
}

static void moveBefore(Operation *opToMove, Operation *existingOp) {
  // Avoid breaking dominance.
  for (Value operand : opToMove->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || !defOp->isBeforeInBlock(existingOp))
      return;
  }
  opToMove->moveBefore(existingOp);
}

static void moveTensorInsertSlice(func::FuncOp funcOp) {
  funcOp.walk([&](tensor::InsertSliceOp insertSliceOp) {
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(
        insertSliceOp.getSource(), &backwardSlice, [](Operation *op) {
          return isa<DestinationStyleOpInterface, tensor::EmptyOp>(op);
        });
    if (backwardSlice.empty() || !isa<tensor::EmptyOp>(backwardSlice.front()))
      return WalkResult::skip();

    Operation *destOp = insertSliceOp.getDest().getDefiningOp();
    if (!destOp)
      return WalkResult::skip();
    moveBefore(destOp, backwardSlice.front());
    return WalkResult::advance();
  });
}

void ConvertToDestinationPassingStyle::runOnOperation() {
  // Still TBD if this is profitable.
  adaptComputeConsumerToAvoidAllocation(getOperation());
  moveTensorInsertSlice(getOperation());
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createBufferizePass() {
  return std::make_unique<Bufferize>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertToDestinationPassingStylePass() {
  return std::make_unique<ConvertToDestinationPassingStyle>();
}

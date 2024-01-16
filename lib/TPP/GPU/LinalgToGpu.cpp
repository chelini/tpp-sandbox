//===- LinalgToGpu.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/IR/MatcherUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGTOGPU
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Creates an outermost parallel loop wrapper around an operation to represent
// number of GPU blocks.
// If there is already a parallel loop present, no operation is created and
// a nullopt is returned instead.
static std::optional<scf::ParallelOp>
createGpuBlocksWrapper(Operation *op, ArrayRef<int64_t> blockDims,
                       PatternRewriter &rewriter) {
  assert(blockDims.size() <= 3 && "Too many GPU blocks dimensions");

  auto loc = op->getLoc();

  auto *parentOp = op->getParentOp();
  if (isa<scf::ParallelOp>(parentOp))
    return std::nullopt;

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> gpuBlocks;
  SmallVector<Value> lbs;
  SmallVector<Value> steps;
  for (auto blockDim : blockDims) {
    auto blockSize = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);
    gpuBlocks.push_back(blockSize);
    // Add matching number of lbs and steps.
    lbs.push_back(zero);
    steps.push_back(one);
  }

  return rewriter.create<scf::ParallelOp>(loc, lbs, gpuBlocks, steps);
}

// Return true if the operation can be represented with WMMA compute.
static bool supportsMMACompute(linalg::LinalgOp linalgOp) {
  if (!(isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
        isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp))) {
    return false;
  }

  // Only static shapes are supported.
  if (linalgOp.hasDynamicShape())
    return false;

  auto aType = linalgOp.getDpsInputs()[0].getType().cast<ShapedType>();
  auto bType = linalgOp.getDpsInputs()[1].getType().cast<ShapedType>();
  auto cType = linalgOp.getDpsInits()[0].getType().cast<ShapedType>();

  ArrayRef<int64_t> shapeA = aType.getShape();
  ArrayRef<int64_t> shapeC = cType.getShape();
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  // Buffer A might be 2D (gemm) or 3D (brgemm) but the last dimension will
  // always be reduction.
  int64_t k = shapeA.back();

  // For now, only M-N-K F16[16] x F16[16] x F16[16] WMMA variant is supported.
  // TODO: add more WMMA combinations.
  return aType.getElementType().isF16() && bType.getElementType().isF16() &&
         cType.getElementType().isF16() && m == 16 && n == 16 && k == 16;
}

// Fuse a consumer using WMMA operations.
// Returns updated store op or nullopt if the fusion fails.
static std::optional<Operation *>
mmaFusion(linalg::LinalgOp rootOp, linalg::LinalgOp consumer,
          gpu::SubgroupMmaStoreMatrixOp rootStoreOp, ValueRange storeIndices,
          PatternRewriter &rewriter) {
  Location loc = rootOp.getLoc();

  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();
  // Must be a floating point type.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  gpu::MMAMatrixType mmaOutputType = gpu::MMAMatrixType::get(
      outputType.getShape(), outputType.getElementType(), "COp");
  auto leadingDim = rootStoreOp.getLeadDimension();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOp);

  std::optional<Operation *> newStore = std::nullopt;
  SmallVector<Value> operands;
  if (structured_match::utils::isTwoDAddOp(consumer, &operands)) {
    // Get the value to be added - load the tile first.
    // Must be a buffer of the same type - scalar broadcast is not supported.
    auto addValue = (operands[0] != rootOutput) ? operands[0] : operands[1];
    if (addValue.getType() != rootOutput.getType())
      return std::nullopt;
    // Fuse the add into the matmul body.
    addValue = rewriter
                   .create<gpu::SubgroupMmaLoadMatrixOp>(
                       loc, mmaOutputType, addValue, ValueRange{zero, zero},
                       leadingDim,
                       /*transpose=*/UnitAttr())
                   .getRes();
    auto eltwiseAttr = gpu::MMAElementwiseOp::ADDF;
    auto addRes =
        rewriter
            .create<gpu::SubgroupMmaElementwiseOp>(
                loc, mmaOutputType, ValueRange{rootStoreOp.getSrc(), addValue},
                eltwiseAttr)
            .getRes();
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        rootStoreOp, addRes, rootStoreOp.getDstMemref(), ValueRange{zero, zero},
        leadingDim,
        /*transpose=*/UnitAttr());
  } else if (structured_match::utils::isTwoDReluOp(consumer, &operands)) {
    // Fuse the relu into the matmul body.
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    Value zeroTile = rewriter.create<gpu::SubgroupMmaConstantMatrixOp>(
        loc, mmaOutputType, zeroFloat);
    auto eltwiseAttr = gpu::MMAElementwiseOp::MAXF;
    auto maxRes =
        rewriter
            .create<gpu::SubgroupMmaElementwiseOp>(
                loc, mmaOutputType, ValueRange{rootStoreOp.getSrc(), zeroTile},
                eltwiseAttr)
            .getRes();
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        rootStoreOp, maxRes, rootStoreOp.getDstMemref(), ValueRange{zero, zero},
        leadingDim,
        /*transpose=*/UnitAttr());
  } else {
    // Not a fusable operation. Bail out.
    return std::nullopt;
  }

  rewriter.eraseOp(consumer);

  return newStore;
}

// Fuse a consumer using scalar operations.
// Returns updated store op or nullopt if the fusion fails.
static std::optional<Operation *> scalarFusion(linalg::LinalgOp rootOp,
                                               linalg::LinalgOp consumer,
                                               memref::StoreOp rootStoreOp,
                                               ValueRange storeIndices,
                                               PatternRewriter &rewriter) {
  Location loc = rootOp.getLoc();
  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();
  // Must be a floating point type.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOp);

  std::optional<Operation *> newStore = std::nullopt;
  SmallVector<Value> operands;
  if (structured_match::utils::isTwoDAddOp(consumer, &operands)) {
    // Get the value to be added. Load the element first, if necessary.
    auto addValue = (operands[0] != rootOutput) ? operands[0] : operands[1];
    if (addValue.getType().isa<ShapedType>()) {
      addValue = rewriter.create<memref::LoadOp>(loc, addValue, storeIndices)
                     .getResult();
    }
    // Fuse the add into the matmul body.
    auto addOp =
        rewriter.create<arith::AddFOp>(loc, rootStoreOp.getValue(), addValue);
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        rootStoreOp, addOp.getResult(), rootOutput, storeIndices);
  } else if (structured_match::utils::isTwoDReluOp(consumer, &operands)) {
    // Fuse the relu into the matmul body.
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    auto maxOp = rewriter.create<arith::MaximumFOp>(loc, rootStoreOp.getValue(),
                                                    zeroFloat);
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        rootStoreOp, maxOp.getResult(), rootOutput, storeIndices);
  } else {
    // Not a fusable operation. Bail out.
    return std::nullopt;
  }

  rewriter.eraseOp(consumer);

  return newStore;
}

// Fuse elementwise consumers.
// A naive fusion strategy that looks at the other operations after the root
// linalg op and tries to fuse them.
// Attemps bails on the first mismatch.
// Returns updated store op.
static Operation *fuseEltwiseConsumers(linalg::LinalgOp rootOp,
                                       Operation *rootStoreOp,
                                       ValueRange storeIndices,
                                       PatternRewriter &rewriter) {
  auto *parentOp = rootOp->getParentOp();
  auto rootOutput = rootOp.getDpsInits()[0];

  // Traverse other ops within the same region and collect consumers.
  SmallVector<linalg::LinalgOp> consumers;
  Operation *nextOp = rootOp;
  while ((nextOp = nextOp->getNextNode())) {
    // Potential consumers must be within the same region.
    if (nextOp->getParentOp() != parentOp)
      break;

    // Only other linalg ops are expected as consumers.
    // TODO: might need to be relaxed to skip over ops without side effects
    auto consumer = dyn_cast<linalg::LinalgOp>(nextOp);
    if (!consumer || !linalg::isElementwise(consumer))
      break;
    // Require the same iteration space.
    if (consumer.getNumParallelLoops() != rootOp.getNumParallelLoops())
      break;

    auto outBuf = consumer.getDpsInitOperand(0)->get();
    // Check that the op reuses the same output buffer as the root op.
    // Otherwise, it is assumed that the op cannot be fused.
    if (outBuf != rootOutput)
      break;

    consumers.push_back(consumer);
  }

  for (auto op : consumers) {
    std::optional<Operation *> updatedStoreOp = std::nullopt;
    if (auto storeOp = dyn_cast<memref::StoreOp>(rootStoreOp)) {
      updatedStoreOp =
          scalarFusion(rootOp, op, storeOp, storeIndices, rewriter);
    } else if (auto mmaStore =
                   dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(rootStoreOp)) {
      updatedStoreOp = mmaFusion(rootOp, op, mmaStore, storeIndices, rewriter);
    }

    // Not a fusable operation. Bail out.
    if (!updatedStoreOp)
      break;

    rootStoreOp = *updatedStoreOp;
  }

  return rootStoreOp;
}

// Create WMMA instructions out of matmul-like operation.
static LogicalResult gemmToGpuMMA(linalg::LinalgOp linalgOp,
                                  PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for MMA lowering");

  Location loc = linalgOp.getLoc();

  // If there is no parallel loop, create a unit blocks wrapper around the
  // current op.
  // This ensures that WMMA operations are created at the thread level (inner
  // nested parallel loops).
  auto blocksLoop = createGpuBlocksWrapper(linalgOp, {1, 1}, rewriter);
  if (blocksLoop)
    rewriter.setInsertionPoint(blocksLoop->getBody()->getTerminator());

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeB = matB.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

  gpu::MMAMatrixType mmaTypeA = gpu::MMAMatrixType::get(
      typeA.getShape().take_back(2), typeA.getElementType(), "AOp");
  gpu::MMAMatrixType mmaTypeB = gpu::MMAMatrixType::get(
      typeB.getShape().take_back(2), typeB.getElementType(), "BOp");
  gpu::MMAMatrixType mmaTypeC =
      gpu::MMAMatrixType::get(typeC.getShape(), typeC.getElementType(), "COp");

  auto stridesA = utils::getStaticStrides(matA);
  auto stridesB = utils::getStaticStrides(matB);
  auto stridesC = utils::getStaticStrides(matC);

  if (failed(stridesA) || failed(stridesB) || failed(stridesC)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect static strides for MMA lowering");
  }
  if (stridesA->back() != 1 || stridesB->back() != 1 || stridesC->back() != 1) {
    return rewriter.notifyMatchFailure(
        linalgOp,
        "Expect unit stride in the innermost dimension for MMA operations");
  }

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  // Skip batch dimension stride in case of brgemm.
  auto lda = rewriter.getIndexAttr(stridesA->begin()[isBrgemm ? 1 : 0]);
  auto ldb = rewriter.getIndexAttr(stridesB->begin()[isBrgemm ? 1 : 0]);
  auto ldc = rewriter.getIndexAttr(stridesC->front());

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  // WMMA requires warp/subgroup size of 32 threads/work items.
  Value subgroupSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);

  // Create parallel loop to indicate that the whole subgroup is performing MMA
  // operations together. It also ensures that the kernel is outlined with
  // the correct number of threads.
  auto parallelLoop = rewriter.create<scf::ParallelOp>(
      loc, ValueRange{zero}, ValueRange{subgroupSize}, ValueRange{one});

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());

  // Fetch the inital value of the output element.
  Value tileC = rewriter
                    .create<gpu::SubgroupMmaLoadMatrixOp>(
                        loc, mmaTypeC, matC, ValueRange{zero, zero}, ldc,
                        /*transpose=*/UnitAttr())
                    .getRes();

  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value batch =
        rewriter.create<arith::ConstantIndexOp>(loc, typeA.getShape()[0]);
    batchLoop =
        rewriter.create<scf::ForOp>(loc, zero, batch, one, ValueRange{tileC});
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();
    tileC = batchLoop.getRegionIterArg(0);
  }

  Value tileA = rewriter
                    .create<gpu::SubgroupMmaLoadMatrixOp>(
                        loc, mmaTypeA, matA,
                        isBrgemm ? ValueRange{batchIv, zero, zero}
                                 : ValueRange{zero, zero},
                        lda,
                        /*transpose=*/UnitAttr())
                    .getRes();
  Value tileB = rewriter
                    .create<gpu::SubgroupMmaLoadMatrixOp>(
                        loc, mmaTypeB, matB,
                        isBrgemm ? ValueRange{batchIv, zero, zero}
                                 : ValueRange{zero, zero},
                        ldb, /*transpose=*/UnitAttr())
                    .getRes();

  Value result =
      rewriter
          .create<gpu::SubgroupMmaComputeOp>(loc, tileC.getType(), tileA, tileB,
                                             tileC, /*a_transpose=*/UnitAttr(),
                                             /*b_transpose=*/UnitAttr())
          .getRes();

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{result});
    result = batchLoop.getResults()[0];
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  auto storeOp = rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
      loc, result, matC, ValueRange{zero, zero}, ldc, /*transpose=*/UnitAttr());

  (void)fuseEltwiseConsumers(linalgOp, storeOp, ValueRange{zero, zero},
                             rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create loops out of matmul-like operation.
static LogicalResult gemmToGpuLoops(linalg::LinalgOp linalgOp,
                                    PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for loop lowering");

  Location loc = linalgOp.getLoc();

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  ArrayRef<int64_t> shapeC = matC.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> shapeA = matA.getType().cast<ShapedType>().getShape();

  // Parallel dims.
  Value i = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[0]);
  Value j = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[1]);
  // Reduction dim.
  Value k = rewriter.create<arith::ConstantIndexOp>(loc, shapeA.back());
  // Lbs.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  // Step.
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> ivs;

  // Create parallel loops over the outer dimensions.
  auto parallelLoop = rewriter.create<scf::ParallelOp>(
      loc, ValueRange{zero, zero}, ValueRange{i, j}, ValueRange{one, one});
  auto parallelIvs = parallelLoop.getInductionVars();
  ivs.append(parallelIvs.begin(), parallelIvs.end());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());

  // Fetch the inital value of the output element.
  Value initVal =
      rewriter.create<memref::LoadOp>(loc, matC, parallelIvs).getResult();

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value batch = rewriter.create<arith::ConstantIndexOp>(loc, shapeA[0]);
    batchLoop =
        rewriter.create<scf::ForOp>(loc, zero, batch, one, ValueRange{initVal});
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();
    initVal = batchLoop.getRegionIterArg(0);
  }

  // Compute matmul with a loop over reduction dimension.
  // Each GPU thread computes a single result element.
  // Accumulate result locally through loop's iter args.
  // This maps to more efficient computation as the accumulation is kept
  // locally by a thread.
  auto bodyBuilder = [&](OpBuilder &b, Location loc, Value localIv,
                         ValueRange iterArgs) {
    SmallVector<Value> loopIvs = ivs;
    loopIvs.push_back(localIv);
    assert(loopIvs.size() == 3);
    Value localI = loopIvs[0];
    Value localJ = loopIvs[1];
    Value localK = loopIvs[2];
    Value scalarA =
        b.create<memref::LoadOp>(loc, matA,
                                 isBrgemm ? ValueRange{batchIv, localI, localK}
                                          : ValueRange{localI, localK});
    Value scalarB =
        b.create<memref::LoadOp>(loc, matB,
                                 isBrgemm ? ValueRange{batchIv, localK, localJ}
                                          : ValueRange{localK, localJ});
    Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
    auto scalarAdd = b.create<arith::AddFOp>(loc, iterArgs[0], scalarMul);

    b.create<scf::YieldOp>(loc, scalarAdd.getResult());
  };
  auto accumulationLoop = rewriter.create<scf::ForOp>(
      loc, zero, k, one, ValueRange{initVal},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        bodyBuilder(b, loc, iv, iterArgs);
      });

  Value result = accumulationLoop.getResults()[0];

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{result});
    result = batchLoop.getResults()[0];
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  auto storeOp =
      rewriter.create<memref::StoreOp>(loc, result, matC, parallelIvs);

  (void)fuseEltwiseConsumers(linalgOp, storeOp, parallelIvs, rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert linalg.matmul to GPU-compatible kernel.
struct ConvertGemmToGpu : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  ConvertGemmToGpu(MLIRContext *ctx, bool useWmma)
      : OpRewritePattern(ctx), useWmma(useWmma) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Linalg gemm to GPU expects memref type");
    }
    if (matmulOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to GPU");
    }

    if (useWmma && supportsMMACompute(matmulOp))
      return gemmToGpuMMA(matmulOp, rewriter);
    return gemmToGpuLoops(matmulOp, rewriter);
  }

private:
  bool useWmma;
};

// Convert linalg.batch_reduce_matmul to GPU-compatible kernel.
struct ConvertBrgemmToGpu
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  ConvertBrgemmToGpu(MLIRContext *ctx, bool useWmma)
      : OpRewritePattern(ctx), useWmma(useWmma) {}

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Linalg brgemm to GPU expects memref type");
    }
    if (brgemmOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Expect static shape when mapping to GPU");
    }

    if (useWmma && supportsMMACompute(brgemmOp))
      return gemmToGpuMMA(brgemmOp, rewriter);
    return gemmToGpuLoops(brgemmOp, rewriter);
  }

private:
  bool useWmma;
};

void populateLinalgToGpuPatterns(RewritePatternSet &patterns, bool useWmma) {
  patterns.add<ConvertGemmToGpu, ConvertBrgemmToGpu>(patterns.getContext(),
                                                     useWmma);
}

struct LinalgToGpu : public tpp::impl::LinalgToGpuBase<LinalgToGpu> {
  using LinalgToGpuBase::LinalgToGpuBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToGpuPatterns(patterns, useWmma);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

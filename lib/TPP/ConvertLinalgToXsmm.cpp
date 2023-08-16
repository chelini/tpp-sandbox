//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
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

struct UnaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldi;
  int64_t ldo;
};

struct BinaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldiLhs;
  int64_t ldiRhs;
  int64_t ldo;
};
} // namespace

// Check if the strides associated with `operand` are valid strides
// for XSMM: Strides must be statically known.
static FailureOr<SmallVector<int64_t>> verifyStrides(Type operandType) {
  assert(!isa<RankedTensorType>(operandType));

  // Scalar type.
  if (!isa<MemRefType>(operandType))
    return SmallVector<int64_t>{1};

  // MemRef type.
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
  if (strides.back() != 1)
    return failure();
  return strides;
}

static FailureOr<SmallVector<int64_t>> verifyStrides(OpOperand *operand) {
  return verifyStrides(operand->get().getType());
}

// Return true if all the operand have the same type.
static bool hasEqualTypes(linalg::LinalgOp linalgOp) {
  OpOperand *outputOperand = linalgOp.getDpsInitOperands().back();
  auto elemType = getElementTypeOrSelf(outputOperand->get().getType());

  if (!llvm::all_of(linalgOp.getDpsInitOperands(), [&](OpOperand *operand) {
        auto currentOperandType =
            getElementTypeOrSelf(operand->get().getType());
        return currentOperandType == elemType;
      })) {
    return false;
  }

  return llvm::all_of(linalgOp.getDpsInputOperands(), [&](OpOperand *operand) {
    auto currentOperandType = getElementTypeOrSelf(operand->get().getType());
    return currentOperandType == elemType;
  });
}

// Structural matcher.
static FailureOr<linalg::ContractionDimensions>
checkStructure(linalg::LinalgOp linalgOp) {
  if (!linalgOp.hasBufferSemantics() || linalgOp.hasDynamicShape() ||
      !hasEqualTypes(linalgOp)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Failed preconditions\n");
    return failure();
  }

  auto contractionDims = linalgx::utils::isContraction(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a contraction\n");
    return failure();
  }
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
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong operands\n");
    return failure();
  }

  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1) ||
      contractionDims->batch.size() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong dimensions\n");
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (linalgOp.getNumLoops() != classifiedLoops) {
    LLVM_DEBUG(llvm::dbgs()
               << "[checkStructure] Not all loops are classified\n");
    return failure();
  }
  return contractionDims;
}

// Access matcher.
static FailureOr<BrgemmInfo> checkAccess(linalg::LinalgOp linalgOp, unsigned m,
                                         unsigned n, unsigned k,
                                         unsigned batch) {
  assert(linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1);
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = linalgOp.getDpsInitOperands()[0];

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
  auto contractionDims = checkStructure(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[isMappableToBrgemm] Failed on checkStructure\n");
    return failure();
  }

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

  return checkAccess(linalgOp, m, n, k, batch);
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
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
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

// Convert a linalg.matmul to a XSMM brgemm op.
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

// Check if we can map `genericOp` to a BRGEMM and rewrite it to XSMM brgemm op.
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

static void replaceOpWithUnary(RewriterBase &rewriter,
                               linalg::LinalgOp linalgOp, UnaryInfo unaryInfo,
                               ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(linalgOp, dtype, kind,
                                             invokeOperands);
}

static void replaceOpWithBinary(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                BinaryInfo binaryInfo, ArrayAttr flags,
                                xsmm::BinaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{binaryInfo.m, binaryInfo.n, binaryInfo.ldiLhs,
                        binaryInfo.ldiRhs, binaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  Value dispatched = rewriter.create<xsmm::BinaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  rewriter.replaceOpWithNewOp<xsmm::BinaryOp>(linalgOp, dtype, kind,
                                              invokeOperands);
}

// Convert a linalg.fill to XSMM zero, if the fill fills with zeros.
struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp.hasBufferSemantics() || fillOp.hasDynamicShape() ||
        !hasEqualTypes(fillOp))
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
    if (failed(stridesOnOutput))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = outputType.getShape()[0];
    unaryInfo.n = outputType.getShape()[1];
    unaryInfo.ldo = stridesOnOutput->front();
    // fillOp has a scalar input.
    unaryInfo.ldi = 1;

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    replaceOpWithUnary(rewriter, fillOp, unaryInfo, flags, kind);
    return success();
  }
};

// Convert a linalg.transpose to a XSMM unary transpose.
struct ConvertTransposeOpToUnaryTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!transposeOp.hasBufferSemantics() || transposeOp.hasDynamicShape() ||
        !hasEqualTypes(transposeOp))
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

    // This looks wired. We should look for m and n on the output
    // buffer, but for transpose it seems not the case.
    UnaryInfo unaryInfo;
    unaryInfo.m = inputType.getShape()[0];
    unaryInfo.n = inputType.getShape()[1];
    unaryInfo.ldi = stridesOnInput->front();
    unaryInfo.ldo = stridesOnOutput->front();

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    replaceOpWithUnary(rewriter, transposeOp, unaryInfo, flags, kind);
    return success();
  }
};

namespace {
enum class BroadCastType { NONE = 0, SCALAR, ROW, COL };
} // namespace

static FailureOr<BroadCastType> getBroadCastFromMap(AffineMap map) {
  if (map.getNumResults() > map.getNumInputs() || map.getNumInputs() != 2 ||
      map.getNumSymbols() != 0) {
    return failure();
  }

  if (map.getNumResults() == 0)
    return BroadCastType::SCALAR;

  // Extend the maps with leading zeros.
  // Example,
  // (d0, d1) -> (d1) --> (d0, d1) -> (0, d1)
  while (map.getNumResults() != map.getNumInputs())
    map = map.insertResult(mlir::getAffineConstantExpr(0, map.getContext()), 0);

  if (!map.isProjectedPermutation(/*allowZeroInResults=*/true))
    return failure();

  SmallVector<unsigned> broadcastedDims;
  if (!map.isMinorIdentityWithBroadcasting(&broadcastedDims))
    return failure();

  if (broadcastedDims.empty())
    return BroadCastType::NONE;

  if (broadcastedDims.size() != 1)
    return failure();

  unsigned broadcastedDim = broadcastedDims[0];
  // Broadcast the cols into the rows.
  if (broadcastedDim == 0)
    return BroadCastType::COL;
  return BroadCastType::ROW;
}

// Get the xsmm unary broadcast flags by looking at the map. Example,
// (d0, d1) -> (d0, d1) = NONE
// (d0, d1) -> (0, d1) = COL
// (d0, d1) -> (d0, 0) = ROW
// (d0, d1) -> () = SCALAR
static FailureOr<xsmm::UnaryFlags> getBroadCastUnaryFlagFromMap(AffineMap map) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return xsmm::UnaryFlags::BCAST_SCALAR;
  case BroadCastType::ROW:
    return xsmm::UnaryFlags::BCAST_ROW;
  case BroadCastType::COL:
    return xsmm::UnaryFlags::BCAST_COL;
  default:
    return xsmm::UnaryFlags::NONE;
  }
}

static FailureOr<xsmm::BinaryFlags>
getBroadCastBinaryFlagFromMap(AffineMap map, unsigned operandIdx) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  assert(operandIdx == 0 || operandIdx == 1);
  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_SCALAR_IN_0
                             : xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  case BroadCastType::ROW:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_ROW_IN_0
                             : xsmm::BinaryFlags::BCAST_ROW_IN_1;
  case BroadCastType::COL:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_COL_IN_0
                             : xsmm::BinaryFlags::BCAST_COL_IN_1;
  default:
    return xsmm::BinaryFlags::NONE;
  }
}

// Get the OpOperand matching 'input', assert if 'input' is not found.
static OpOperand *getOperandFromValue(linalg::GenericOp genericOp, Value val) {
  SmallVector<OpOperand *> allOperands = genericOp.getDpsInputOperands();
  SmallVector<OpOperand *> initOperands = genericOp.getDpsInitOperands();
  allOperands.append(initOperands.begin(), initOperands.end());

  OpOperand *valAsOperand = nullptr;
  for (OpOperand *operand : allOperands) {
    if (operand->get() == val) {
      valAsOperand = operand;
      break;
    }
  }
  assert(valAsOperand && "expect to find input");
  return valAsOperand;
}

struct ConvertGenericToUnaryRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasBufferSemantics() || genericOp.hasDynamicShape() ||
        !hasEqualTypes(genericOp) || !linalg::isElementwise(genericOp)) {
      return failure();
    }
    SmallVector<Value> operands;
    if (!tpp::utils::isTppRelu(genericOp, &operands) || operands.size() != 2)
      return failure();

    auto input = operands[0];
    auto output = operands[1];
    Type inputType = input.getType();
    Type outputType = output.getType();
    auto stridesOnInput = verifyStrides(inputType);
    auto stridesOnOutput = verifyStrides(outputType);
    if (failed(stridesOnInput) || failed(stridesOnInput))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = outputType.cast<ShapedType>().getShape()[0];
    unaryInfo.n = outputType.cast<ShapedType>().getShape()[1];
    unaryInfo.ldi = stridesOnInput->front();
    unaryInfo.ldo = stridesOnOutput->front();

    // Wired way to get back the OpOperand from the captured value,
    // maybe we should capture OpOperand*?
    OpOperand *inputOperand = getOperandFromValue(genericOp, input);
    auto broadCastFlag = getBroadCastUnaryFlagFromMap(
        genericOp.getMatchingIndexingMap(inputOperand));
    if (failed(broadCastFlag))
      return failure();
    auto flags = rewriter.getArrayAttr(
        xsmm::UnaryFlagsAttr::get(rewriter.getContext(), *broadCastFlag));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::RELU);
    replaceOpWithUnary(rewriter, genericOp, unaryInfo, flags, kind);
    return success();
  }
};

struct ConvertGenericToBinaryAdd : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasBufferSemantics() || genericOp.hasDynamicShape() ||
        !hasEqualTypes(genericOp) || !linalg::isElementwise(genericOp)) {
      return failure();
    }
    SmallVector<Value> operands;
    if (!tpp::utils::isTppAdd(genericOp, &operands) || operands.size() != 3)
      return failure();

    auto lhs = operands[0];
    auto rhs = operands[1];
    auto output = operands[2];
    Type outputType = output.getType();

    auto stridesOnLhs = verifyStrides(lhs.getType());
    auto stridesOnRhs = verifyStrides(rhs.getType());
    auto stridesOnOutput = verifyStrides(outputType);
    if (failed(stridesOnLhs) || failed(stridesOnRhs) || failed(stridesOnOutput))
      return failure();

    BinaryInfo binaryInfo;
    binaryInfo.m = outputType.cast<ShapedType>().getShape()[0];
    binaryInfo.n = outputType.cast<ShapedType>().getShape()[1];
    binaryInfo.ldiLhs = stridesOnLhs->front();
    binaryInfo.ldiRhs = stridesOnRhs->front();
    binaryInfo.ldo = stridesOnOutput->front();

    OpOperand *rhsOperand = getOperandFromValue(genericOp, rhs);
    // TODO: Handle LHS.
    auto broadCastFlag = getBroadCastBinaryFlagFromMap(
        genericOp.getMatchingIndexingMap(rhsOperand), /*operandIdx=*/1);
    if (failed(broadCastFlag))
      return failure();
    auto flags = rewriter.getArrayAttr(
        xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *broadCastFlag));
    xsmm::BinaryKindAttr kind =
        xsmm::BinaryKindAttr::get(rewriter.getContext(), xsmm::BinaryKind::ADD);
    replaceOpWithBinary(rewriter, genericOp, binaryInfo, flags, kind);
    return failure();
  }
};

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Enable conversion for linalg.generic to XSMM Brgemm if possible.
  auto res = funcOp->walk([&](linalg::GenericOp genericOp) {
    auto contractionDims = checkStructure(genericOp);
    // If the generic does not match the structure of a Brgemm op, skip it.
    if (failed(contractionDims))
      return WalkResult::skip();
    unsigned m = contractionDims->m[0];
    unsigned n = contractionDims->n[0];
    unsigned k = contractionDims->k.back();
    unsigned batch = (contractionDims->k.size() == 2)
                         ? contractionDims->k.front()
                         : std::numeric_limits<unsigned>::max();
    if (failed(checkAccess(genericOp, m, n, k, batch))) {
      // The generic is a Brgemm but the strides of the selected dims (m, n, k)
      // are not unit strides. Inject transposes to bring them innermost.
      if (failed(linalgx::utils::makeMinorDimensionsInnerMost(
              rewriter, genericOp, m, n, k))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "pass failed!\n");
    return signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  tpp::populateLinalgToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGenericToBrgemm, ConvertMatmulToMatmul,
               ConvertFillOpToUnaryZero, ConvertTransposeOpToUnaryTranspose,
               ConvertGenericToUnaryRelu, ConvertGenericToBinaryAdd>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}

//===- ConvertTppToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTTPPTOXSMM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "convert-tpp-to-xsmm"

namespace {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static FailureOr<int64_t> getLeadingDim(Type type, size_t pos = 0) {
  // Not shaped type, the leading dimension is the single scalar.
  if (!isa<ShapedType>(type))
    return 1;
  MemRefType memref = type.cast<MemRefType>();
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

template <class OpKind, class OpFlags, class KindAttr, class FlagsAttr,
          class DispatchOp, class Op>
static LogicalResult lowerTPPtoXSMM(tpp::TppOp op, PatternRewriter &rewriter,
                                    Type elmTy, OpKind kind, OpFlags flags,
                                    ArrayRef<int64_t> dims) {
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  KindAttr kindAttr = KindAttr::get(ctx, kind);
  DenseI64ArrayAttr dimsAttr =
      DenseI64ArrayAttr::get(rewriter.getContext(), dims);
  auto flagsAttr = FlagsAttr::get(ctx, flags);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  xsmm::DataTypeAttr dtype =
      xsmm::utils::getDataType(rewriter, op.getOutputType());

  Value dispatched =
      rewriter.create<DispatchOp>(loc, integer64, kindAttr, dimsAttr,
                                  rewriter.getArrayAttr(flagsAttr), dtype);

  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(op.getInputs().begin(), op.getInputs().end());
  invokeOperands.push_back(op.getOutput());

  rewriter.replaceOpWithNewOp<Op>(op, dtype, kindAttr, invokeOperands);
  return success();
}

static LogicalResult lowerUnaryTPPtoXSMM(PatternRewriter &rewriter,
                                         Operation *op, xsmm::UnaryKind kind) {
  auto tppOp = cast<tpp::TppOp>(op);
  if (!tppOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(tppOp, "xsmm expects a memref type");

  MemRefType outputMemRef = tppOp.getOutputType();
  int64_t m = outputMemRef.getShape()[0];
  int64_t n = outputMemRef.getShape()[1];
  auto ldo = getLeadingDim(outputMemRef);
  if (failed(ldo))
    return rewriter.notifyMatchFailure(tppOp, "cannot compute ldo");
  auto ldi = getLeadingDim(tppOp.getInputs()[0].getType());
  if (failed(ldi))
    return rewriter.notifyMatchFailure(tppOp, "cannot compute ldi");
  auto flags = xsmm::utils::getUnaryFlags(tppOp.getInputs()[0].getType(),
                                          tppOp.getOutputType());
  if (failed(flags))
    return failure();
  return lowerTPPtoXSMM<xsmm::UnaryKind, xsmm::UnaryFlags, xsmm::UnaryKindAttr,
                        xsmm::UnaryFlagsAttr, xsmm::UnaryDispatchOp,
                        xsmm::UnaryOp>(tppOp, rewriter,
                                       outputMemRef.getElementType(), kind,
                                       *flags, {m, n, *ldi, *ldo});
}

struct ConvertTppIdentityOp : public OpRewritePattern<tpp::IdentityOp> {
  using OpRewritePattern<tpp::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, identityOp, xsmm::UnaryKind::IDENTITY);
  }
};

struct ConvertTppReluOp : public OpRewritePattern<tpp::ReluOp> {
  using OpRewritePattern<tpp::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, reluOp, xsmm::UnaryKind::RELU);
  }
};

struct ConvertTppZeroOp : public OpRewritePattern<tpp::ZeroOp> {
  using OpRewritePattern<tpp::ZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ZeroOp zeroOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, zeroOp, xsmm::UnaryKind::ZERO);
  }
};

struct ConvertTppToXsmm
    : public tpp::impl::ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTppIdentityOp, ConvertTppReluOp, ConvertTppZeroOp>(
        patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

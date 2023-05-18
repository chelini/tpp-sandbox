//===- TppOps.cpp - Tpp dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/VNNIUtils.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Tpp/TppOps.cpp.inc"

using namespace mlir;
using namespace mlir::tpp;

namespace {
constexpr std::string_view INS = "ins";
constexpr std::string_view OUTS = "outs";
constexpr std::string_view OPERAND_SEGMENT_SIZE = "operand_segment_sizes";
constexpr std::string_view UNARY_KIND = "unary_kind";
constexpr std::string_view BINARY_KIND = "binary_kind";
constexpr std::string_view UNARY = "unary";
constexpr std::string_view BINARY = "binary";
} // namespace

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static void printCommaSeparatedList(OpAsmPrinter &printer, ValueRange values) {
  printer << '(';
  for (auto [idx, value] : llvm::enumerate(values)) {
    printer << value << " : " << value.getType();
    if (idx != values.size() - 1)
      printer << ", ";
  }
  printer << ')';
}

static ParseResult parseTppOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandsTypes;

  bool isMemRef = false;
  if (succeeded(parser.parseOptionalKeyword(INS)))
    isMemRef = true;

  // Parse operands.
  SmallVector<llvm::SMLoc> locsOperands;
  auto parseOperand = [&]() -> ParseResult {
    locsOperands.push_back(parser.getCurrentLocation());
    if (parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(operandsTypes.emplace_back()))
      return failure();
    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperand)) {
    return failure();
  }
  int numberOfInputs = operands.size();
  int numberOfOutputs = 0;

  if (isMemRef) {
    locsOperands.push_back(parser.getCurrentLocation());
    if (parser.parseKeyword(OUTS) || parser.parseLParen() ||
        parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(operandsTypes.emplace_back()) ||
        parser.parseRParen())
      return failure();
    numberOfOutputs = operands.size() - numberOfInputs;
  } else {
    // Parse result types.
    SmallVector<Type> resultTypes;
    llvm::SMLoc resultTypeLoc = parser.getCurrentLocation();
    if (parser.parseArrowTypeList(resultTypes) ||
        parser.addTypesToList(resultTypes, result.types))
      return failure();

    if (resultTypes.size() != 1) {
      return parser.emitError(resultTypeLoc,
                              "expect single result at tensor abstraction");
    }
  }

  // Validate operands. Scan each operand one-by-one to emit
  // better diagnostic.
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    if (parser.resolveOperand(operand, operandsTypes[idx], result.operands))
      return failure();
    if (isMemRef && operandsTypes[idx].isa<RankedTensorType>())
      return parser.emitError(locsOperands[idx], "expect memref type");
    if (!isMemRef && operandsTypes[idx].isa<MemRefType>())
      return parser.emitError(locsOperands[idx], "expect tensor type");
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  // Check if we parsed `operand_segment_sizes` already, otherwise add it.
  if (!attrs.get(OPERAND_SEGMENT_SIZE)) {
    auto operandSegmentSize = parser.getBuilder().getDenseI32ArrayAttr(
        {numberOfInputs, numberOfOutputs});
    result.addAttribute(OPERAND_SEGMENT_SIZE, operandSegmentSize);
  }
  result.addAttributes(attrs);
  return success();
}

// Print a tpp op. Note that `out` can be null. It is null for unary and binary
// at tensor abstraction.
static void printTppOp(OpAsmPrinter &printer, ValueRange operands,
                       ValueRange outs, TypeRange results, Operation *op) {
  printer << ' ';
  if (results.empty()) {
    printer << INS;
    printCommaSeparatedList(printer, operands);
    printer << ' ';
    printer << OUTS;
    printCommaSeparatedList(printer, outs);
  } else {
    printCommaSeparatedList(printer, operands);
    printer << " -> (" << results << ")";
  }
  printer.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{OPERAND_SEGMENT_SIZE, UNARY_KIND, BINARY_KIND});
}

static void tppOpBuilder(OpBuilder &builder, OperationState &state,
                         ValueRange inputs, ValueRange outputs) {
  assert(outputs.size() >= 1);
  state.addOperands(inputs);
  if (auto rankedOutput =
          outputs[0].getType().dyn_cast_or_null<RankedTensorType>()) {
    state.addTypes(outputs.getTypes());
    state.addAttribute(
        OPERAND_SEGMENT_SIZE,
        builder.getDenseI32ArrayAttr(
            {static_cast<int>(inputs.size()), /*numOutputs=*/0}));
  } else {
    state.addOperands(outputs);
    state.addAttribute(
        OPERAND_SEGMENT_SIZE,
        builder.getDenseI32ArrayAttr({static_cast<int>(inputs.size()),
                                      static_cast<int>(outputs.size())}));
  }
}

static void getEffectsImpl(
    TppOp tppOp,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (tppOp.hasTensorSemantics())
    return;
  for (auto operand : tppOp.getInputs()) {
    if (!operand.getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), tppOp.getOutput(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

void IdentityOp::build(OpBuilder &builder, OperationState &state, Value input,
                       Value output) {
  tppOpBuilder(builder, state, input, output);
}

void IdentityOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult IdentityOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void IdentityOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilder(builder, state, input, output);
}

void ReluOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// ZeroOp
//===----------------------------------------------------------------------===//

void ZeroOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilder(builder, state, input, output);
}

void ZeroOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult ZeroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

LogicalResult ZeroOp::verify() {
  // At tensor abstraction computation result is always placed in a new tensor
  // so skip validation.
  if (hasTensorSemantics())
    return success();

  auto input = getInputs()[0];
  auto output = getOutputs()[0];

  if (input != output)
    return emitOpError("fails to verify in-place computation");

  return success();
}

void ZeroOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  Value output) {
  tppOpBuilder(builder, state, inputs, output);
}

void AddOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void AddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult
validateVnniGemmOperand(OpTy operation, ArrayRef<int64_t> shape,
                        Type elementType, int dimI, int dimJ) {
  if (shape.size() == 3) {
    if (!elementType.isBF16()) {
      return operation->emitOpError() << "operand 1 invalid element type for "
                                         "VNNI layout expect bf16, but got: "
                                      << elementType << "\n";
    }
    if (shape[shape.size() - 1] !=
        vnni::utils::getVnniBlockingFactor(elementType)) {
      return operation->emitOpError() << "operand 1 invalid VNNI layout expect "
                                         "inner dims to be 2 or 4, but got: "
                                      << shape[shape.size() - 1] << "\n";
    }
    // VNNI layout: [K/VNNI][J][VNNI]
    if (shape[0] * shape[shape.size() - 1] != dimI || shape[1] != dimJ)
      return operation->emitOpError("operand 1 fails to verify expected shape");
    return success();
  }
  if (shape.size() != 2) {
    return operation->emitOpError()
           << "operand 1 expects rank 2, but got: " << shape.size() << "\n";
  }
  if (shape[0] != dimI || shape[1] != dimJ)
    return operation->emitOpError("operand 1 fails to verify expected shape");
  return success();
}

// Verify gemm operation.
LogicalResult GemmOp::verify() {
  auto shapedA = getInputs()[0].getType().cast<ShapedType>();
  auto shapedB = getInputs()[1].getType().cast<ShapedType>();
  auto shapedC = getInputs()[2].getType().cast<ShapedType>();
  auto shapedResult = (hasTensorSemantics())
                          ? getResultType().cast<ShapedType>()
                          : getOutputType().cast<ShapedType>();
  if (shapedC != shapedResult) {
    return emitOpError() << "result type differs from destination operand type";
  }

  // Validate operand C.
  if (shapedC.getRank() != 2) {
    return emitOpError() << "operand 2 expects rank 2, but got: "
                         << shapedC.getRank() << "\n";
  }
  int64_t m = shapedC.getShape()[0];
  int64_t n = shapedC.getShape()[1];

  // Validate operand A.
  if (shapedA.getRank() != 2) {
    return emitOpError() << "operand 0 expects rank 2, but got: "
                         << shapedA.getRank() << "\n";
  }
  if (shapedA.getShape()[0] != m)
    return emitOpError("operand 0 fails to verify expected shape");
  int64_t k = shapedA.getShape()[1];

  // Validate operand B.
  return validateVnniGemmOperand(this, shapedB.getShape(),
                                 shapedB.getElementType(), k, n);
}

void GemmOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                   Value output) {
  tppOpBuilder(builder, state, inputs, output);
}

ParseResult GemmOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void GemmOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyBgemmOperands(OpTy operation) {

  static_assert(llvm::is_one_of<OpTy, BrgemmOp, FusedBrgemmOp>::value,
                "applies to brgemm or fused_brgemm operations");

  auto shapedA = operation.getInputs()[0].getType().template cast<ShapedType>();
  auto shapedB = operation.getInputs()[1].getType().template cast<ShapedType>();
  auto shapedC = operation.getInputs()[2].getType().template cast<ShapedType>();
  auto shapedResult =
      (operation.hasTensorSemantics())
          ? operation.getResultType().template cast<ShapedType>()
          : operation.getOutputType().template cast<ShapedType>();

  if (shapedC != shapedResult) {
    return operation.emitOpError()
           << "result type differs from destination operand type";
  }

  // Validate operand C.
  if (shapedC.getRank() != 2) {
    return operation.emitOpError()
           << "operand 2 expects rank 2, but got: " << shapedC.getRank()
           << "\n";
  }
  int64_t m = shapedC.getShape()[0];
  int64_t n = shapedC.getShape()[1];

  // Validate operand A.
  if (shapedA.getRank() != 3) {
    return operation.emitOpError()
           << "operand 0 expects rank 2, but got: " << shapedA.getRank()
           << "\n";
  }
  int64_t batch = shapedA.getShape()[0];
  int64_t k = shapedA.getShape()[shapedA.getRank() - 1];
  if (shapedA.getShape()[1] != m)
    return operation.emitOpError("operand 0 fails to verify expected shape");

  // Validate operand B.
  if (shapedB.getShape()[0] != batch)
    return operation.emitOpError("operand 1 fails to verify expected shape");
  return validateVnniGemmOperand(operation, shapedB.getShape().drop_front(),
                                 shapedB.getElementType(), k, n);
}

LogicalResult BrgemmOp::verify() { return verifyBgemmOperands(*this); }

void BrgemmOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  assert(inputs.size() == 3);
  tppOpBuilder(builder, state, inputs, output);
}

ParseResult BrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void BrgemmOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void BrgemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

LogicalResult FusedBrgemmOp::verify() { return verifyBgemmOperands(*this); }

void FusedBrgemmOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output,
                          Optional<Value> biasInput,
                          FusedUnaryOpKindAttr unaryType,
                          FusedBinaryOpKindAttr binaryType) {
  assert(inputs.size() == 3);

  int numberOfInputs = inputs.size();
  int numberOfOutputs = 0;
  int numberOfOptionalInputs = 0;

  state.addOperands(inputs);
  if (biasInput.has_value()) {
    numberOfOptionalInputs = 1;
    state.addOperands(*biasInput);
  }
  if (auto rankedOutput =
          output.getType().dyn_cast_or_null<RankedTensorType>()) {
    state.addTypes(output.getType());
  } else {
    numberOfOutputs = 1;
    state.addOperands(output);
  }
  state.attributes.set(
      OPERAND_SEGMENT_SIZE,
      builder.getDenseI32ArrayAttr(
          {numberOfInputs, numberOfOptionalInputs, numberOfOutputs}));
  state.addAttribute(UNARY_KIND, unaryType);
  state.addAttribute(BINARY_KIND, binaryType);
}

void FusedBrgemmOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output,
                          FusedUnaryOpKindAttr unaryType,
                          FusedBinaryOpKindAttr binaryType) {
  assert(inputs.size() == 3);
  return build(builder, state, inputs, output, std::nullopt, unaryType,
               binaryType);
}

template <typename EnumClass>
static ParseResult parseEnum(EnumClass &value, OpAsmParser &parser) {
  StringRef flag;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&flag))
    return failure();
  auto flagAttr = symbolizeEnum<EnumClass>(flag);
  if (!flagAttr)
    return parser.emitError(loc, "invalid enum ") << flag;
  value = *flagAttr;
  return success();
}

ParseResult FusedBrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  auto loc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseKeyword(UNARY) ||
      parser.parseEqual())
    return failure();
  FusedUnaryOpKind unaryKind;
  if (parseEnum(unaryKind, parser))
    return failure();
  if (parser.parseComma() || parser.parseKeyword(BINARY) || parser.parseEqual())
    return failure();
  FusedBinaryOpKind binaryKind;
  if (parseEnum(binaryKind, parser))
    return failure();
  if (parser.parseRSquare())
    return failure();
  auto ctx = parser.getBuilder().getContext();
  result.addAttribute(UNARY_KIND, FusedUnaryOpKindAttr::get(ctx, unaryKind));
  result.addAttribute(BINARY_KIND, FusedBinaryOpKindAttr::get(ctx, binaryKind));

  if (failed(parseTppOp(parser, result)))
    return failure();
  bool isMemRefOp = (result.types.size() == 0) ? true : false;
  int outputOperands = (isMemRefOp) ? 1 : 0;
  int inputOperands = 3;
  int numberOfOperands = result.operands.size();
  if (numberOfOperands < 3) {
    return parser.emitError(loc)
           << "expect at least 3 operands, but got: " << numberOfOperands
           << "\n";
  }
  int optionalOperand = numberOfOperands - (outputOperands + inputOperands);
  auto operandSegmentSize = parser.getBuilder().getDenseI32ArrayAttr(
      {inputOperands, optionalOperand, outputOperands});
  result.attributes.set(OPERAND_SEGMENT_SIZE, operandSegmentSize);
  return success();
}

void FusedBrgemmOp::print(OpAsmPrinter &printer) {
  printer << " [" << UNARY << " = "
          << tpp::stringifyFusedUnaryOpKind(getUnaryKind()) << ", " << BINARY
          << " = " << tpp::stringifyFusedBinaryOpKind(getBinaryKind()) << "]";
  SmallVector<Value> inputs = getInputs();
  if (auto biasInput = getBiasInput())
    inputs.push_back(biasInput);
  printTppOp(printer, inputs, getOutputs(), getResultTypes(), *this);
}

void FusedBrgemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

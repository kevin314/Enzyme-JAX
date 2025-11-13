//===- ArithRaising.cpp - Raise to Arith dialect --------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BLASRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct BlasRaisingPass
    : public enzyme::impl::BlasRaisingPassBase<BlasRaisingPass> {
  using BlasRaisingPassBase::BlasRaisingPassBase;

  // Helper to extract and validate cublasSgemm operands
  struct CublasSgemmOperands {
    Value handle;
    Value transA;
    Value transB;
    Value m;
    Value n;
    Value k;
    Value alphaPtr;
    Value A;
    Value lda;
    Value B;
    Value ldb;
    Value betaPtr;
    Value C;
    Value ldc;
  };

  bool extractCublasSgemmOperands(LLVM::CallOp call,
                                   CublasSgemmOperands &operands) const {
    if (call.getNumOperands() != 14) {
      llvm::errs() << "Warning: cublasSgemm_v2 has " << call.getNumOperands()
                   << " operands, expected 14\n";
      return false;
    }

    operands.handle = call.getOperand(0);
    operands.transA = call.getOperand(1);
    operands.transB = call.getOperand(2);
    operands.m = call.getOperand(3);
    operands.n = call.getOperand(4);
    operands.k = call.getOperand(5);
    operands.alphaPtr = call.getOperand(6);
    operands.A = call.getOperand(7);
    operands.lda = call.getOperand(8);
    operands.B = call.getOperand(9);
    operands.ldb = call.getOperand(10);
    operands.betaPtr = call.getOperand(11);
    operands.C = call.getOperand(12);
    operands.ldc = call.getOperand(13);

    llvm::errs() << "Successfully extracted cublasSgemm_v2 operands\n";
    return true;
  }

  Value loadScalarFromPointer(Value ptr, OpBuilder &builder, Location loc) const {
    auto f32Type = builder.getF32Type();
    auto scalarMemrefType = MemRefType::get({}, f32Type);

    auto p2m = builder.create<enzymexla::Pointer2MemrefOp>(
        loc, scalarMemrefType, ptr, ValueRange{});

    auto loaded = builder.create<memref::LoadOp>(loc, p2m.getResult());

    return loaded.getResult();
  }

  Value pointerToMemref(Value ptr, Value rows, Value cols, OpBuilder &builder,
                       Location loc) const {
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);

    auto p2m = builder.create<enzymexla::Pointer2MemrefOp>(
        loc, memrefType, ptr, ValueRange{rows, cols});

    return p2m.getResult();
  }

  Value createDotGeneral(Value A, Value B, OpBuilder &builder,
                        Location loc) const {
    auto ctx = builder.getContext();
    llvm::SmallVector<int64_t> batchDimsA, batchDimsB;
    llvm::SmallVector<int64_t> contractingDimsA{1};
    llvm::SmallVector<int64_t> contractingDimsB{0};

    auto dotDimNumbers = stablehlo::DotDimensionNumbersAttr::get(
        ctx,
        batchDimsA, batchDimsB,
        contractingDimsA, contractingDimsB);

    auto aType = llvm::dyn_cast<ShapedType>(A.getType());
    auto bType = llvm::dyn_cast<ShapedType>(B.getType());
    if (!aType || !bType) {
      llvm::errs() << "Error: A or B type is not a ShapedType\n";
      return A;
    }

    // Result shape is (m, n)
    llvm::SmallVector<int64_t> resultShape{aType.getShape()[0], bType.getShape()[1]};
    auto f32Type = builder.getF32Type();
    auto resultType = RankedTensorType::get(resultShape, f32Type);

    auto dotGeneralOp = builder.create<stablehlo::DotGeneralOp>(
        loc, resultType, A, B, dotDimNumbers,
        nullptr,  // precision_config
        nullptr); // algorithm

    return dotGeneralOp.getResult();
  }

  Value applyAlphaScaling(Value result, Value alpha,
                         OpBuilder &builder, Location loc) const {
    auto resultType = llvm::dyn_cast<ShapedType>(result.getType());
    if (!resultType) {
      llvm::errs() << "Error: result type is not a ShapedType\n";
      return result;
    }
    auto resultShape = resultType.getShape();

    auto alphaBroadcasted = builder.create<stablehlo::BroadcastOp>(
        loc, resultType, alpha,
        DenseI64ArrayAttr::get(builder.getContext(), resultShape));

    auto scaled = builder.create<stablehlo::MulOp>(
        loc, result, alphaBroadcasted);

    return scaled.getResult();
  }

  Value applyBetaAccumulation(Value result, Value C, Value beta,
                             OpBuilder &builder, Location loc) const {
    auto cType = llvm::dyn_cast<ShapedType>(C.getType());
    if (!cType) {
      llvm::errs() << "Error: C type is not a ShapedType\n";
      return result;
    }
    auto cShape = cType.getShape();

    auto betaBroadcasted = builder.create<stablehlo::BroadcastOp>(
        loc, cType, beta,
        DenseI64ArrayAttr::get(builder.getContext(), cShape));

    auto scaled = builder.create<stablehlo::MulOp>(
        loc, C, betaBroadcasted);

    auto accumulated = builder.create<stablehlo::AddOp>(
        loc, result, scaled);

    return accumulated.getResult();
  }

  void runOnOperation() override {
    auto op = getOperation();
    llvm::errs() << "=== BlasRaisingPass running ===\n";
    llvm::errs().flush();

    SmallVector<LLVM::CallOp, 4> cublasCalls;

    op->walk([&](LLVM::CallOp callOp) {
      auto calleeName = callOp.getCallee().value_or("");
      if (calleeName == "cublasSgemm_v2") {
        llvm::errs() << "Found cublasSgemm_v2 call\n";
        cublasCalls.push_back(callOp);
      }
    });

    llvm::errs() << "Found " << cublasCalls.size() << " cublasSgemm_v2 calls\n";
    llvm::errs().flush();

    for (auto call : cublasCalls) {
      CublasSgemmOperands operands;
      if (!extractCublasSgemmOperands(call, operands)) {
        llvm::errs() << "Failed to extract operands for cublasSgemm_v2\n";
        continue;
      }

      llvm::errs() << "Transforming cublasSgemm_v2 call\n";

      OpBuilder builder(call);
      builder.setInsertionPoint(call);

      auto m_index = builder.create<arith::IndexCastOp>(
          call.getLoc(), builder.getIndexType(), operands.m);
      auto n_index = builder.create<arith::IndexCastOp>(
          call.getLoc(), builder.getIndexType(), operands.n);
      auto k_index = builder.create<arith::IndexCastOp>(
          call.getLoc(), builder.getIndexType(), operands.k);

      Value A_memref = pointerToMemref(operands.A, m_index.getResult(), k_index.getResult(), builder, call.getLoc());
      Value B_memref = pointerToMemref(operands.B, k_index.getResult(), n_index.getResult(), builder, call.getLoc());
      Value C_memref = pointerToMemref(operands.C, m_index.getResult(), n_index.getResult(), builder, call.getLoc());

      Value alpha = loadScalarFromPointer(operands.alphaPtr, builder, call.getLoc());
      Value beta = loadScalarFromPointer(operands.betaPtr, builder, call.getLoc());

      llvm::errs() << "Successfully loaded alpha and beta scalars\n";

      builder.create<enzymexla::GemmOp>(
          call.getLoc(),
          A_memref,
          B_memref,
          alpha,
          beta,
          C_memref);

      llvm::errs() << "Emitted enzymexla.linalg.gemm operation\n";

      auto statusCode = builder.create<arith::ConstantOp>(
          call.getLoc(),
          builder.getI32Type(),
          builder.getI32IntegerAttr(0));

      call.replaceAllUsesWith(statusCode);
      call.erase();

      llvm::errs() << "Replaced cublasSgemm_v2 call with enzymexla.linalg.gemm\n";
    }

    llvm::errs() << "=== BlasRaisingPass done ===\n";
    llvm::errs().flush();
  }
};

} // end anonymous namespace


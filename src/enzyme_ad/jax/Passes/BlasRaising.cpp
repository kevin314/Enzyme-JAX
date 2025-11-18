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
  
  Value getPairFromScalars(OpBuilder &builder, Location &loc, Value a, Value b) {
    auto elemTy = cast<TensorType>(a.getType()).getElementType();
    auto aTensor = stablehlo::ReshapeOp::create(
      builder,
      loc, RankedTensorType::get({1}, elemTy), a
    );
    auto bTensor = stablehlo::ReshapeOp::create(
      builder,
      loc, RankedTensorType::get({1}, elemTy), b
    );

    return stablehlo::ConcatenateOp::create(
      builder, loc, RankedTensorType::get({2}, elemTy), ValueRange{aTensor, bTensor}, builder.getI64IntegerAttr(0)
    );
  }

  Value getIsEnum(OpBuilder &builder, Location &loc, Value tensorVal, SmallVector<int> enums) {
    auto i32Tensor = RankedTensorType::get({}, builder.getI32Type());
    auto cmp = [&](Value a, Value b) {
      return stablehlo::CompareOp::create(
          builder,
          loc,
          a, b,
          stablehlo::ComparisonDirection::EQ);
    };

    auto enumTensor = stablehlo::ConstantOp::create(
      builder,
      loc, DenseIntElementsAttr::get(i32Tensor, enums[0])
    );
    Value valCmp = cmp(tensorVal, enumTensor);
    for (int i = 1; i < enums.size(); i++) {
      enumTensor = stablehlo::ConstantOp::create(
        builder,
        loc, DenseIntElementsAttr::get(i32Tensor, enums[i])
      );
      Value newValCmp = cmp(tensorVal, enumTensor);
      valCmp = builder.create<stablehlo::OrOp>(loc, valCmp, newValCmp);
    }
    return valCmp;
  }

  Value getDynamicSlice(OpBuilder &builder, Location &loc, Value tensor, Value dim0, Value dim1) {
    auto i32 = builder.getI32Type();
    auto i32DoubleTensor = RankedTensorType::get({2}, i32);
    // Type dynTensor = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    Value zero_tensor = stablehlo::ConstantOp::create(
        builder,
        loc,
        DenseIntElementsAttr::get(i32DoubleTensor, {0, 0})
      );

    Value one_tensor = stablehlo::ConstantOp::create(
        builder,
        loc,
        DenseIntElementsAttr::get(i32DoubleTensor, {1, 1})
      );

    auto limit_tensor = getPairFromScalars(builder, loc, dim0, dim1);

    return stablehlo::RealDynamicSliceOp::create(builder, loc, tensor.getType(), tensor, zero_tensor, limit_tensor, one_tensor);
  }

  // TODO: Handle differing leading dimensions, update C in place instead of returning it
  func::FuncOp getCublasSGemm_v2(LLVM::LLVMFuncOp func) {
    constexpr StringRef fnName = "raised_cublasSGemm_v2";
    auto module = func->getParentOfType<ModuleOp>();

    if (func::FuncOp existing =
            module.lookupSymbol<func::FuncOp>(fnName)) {
      llvm::errs() << "early exit\n";
      return existing;
    }

    MLIRContext *ctx = func.getContext();
    OpBuilder builder(ctx);
    auto loc = func.getLoc();

    auto f32 = builder.getF32Type();
    auto i32 = builder.getI32Type();
    Type dynTensor = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    Type i32Tensor = RankedTensorType::get({}, i32);
    Type f32Tensor = RankedTensorType::get({}, f32);

    // original type
    auto inputs = func.getArgumentTypes();
    // auto results = func.getResultTypes();

    // Construct new input type list excluding the first one
    SmallVector<Type> newInputs;
    SmallVector<Type> results;

    for (int i = 0; i < inputs.size()-1; i++) {
      if (i == 6 || i == 8 || i == 11) {
        newInputs.push_back(dynTensor);
        results.push_back(dynTensor);
      } else if (i == 5 || i == 10) {
        newInputs.push_back(f32Tensor);
        results.push_back(f32Tensor);
      } else {
        newInputs.push_back(i32Tensor);
        results.push_back(i32Tensor);
      }
    }

    // Create new function type
    auto newFuncType = mlir::FunctionType::get(ctx, newInputs, results);
      // return existing;
    

    auto fn = func::FuncOp::create(func.getLoc(), fnName, newFuncType);
    fn.setPrivate();
    module.push_back(fn);


    // Construct replacement function

    Block *entry = fn.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());

    // Extract arguments
    auto args = entry->getArguments();
    Value transAenum = args[0];
    Value transBenum = args[1];
    Value m = args[2];
    Value n = args[3];
    Value k = args[4];
    Value alpha = args[5];
    Value A = args[6];
    Value lda = args[7];
    Value B = args[8];
    Value ldb = args[9];
    Value beta = args[10];
    Value C = args[11];
    Value ldc = args[12];

    // Value m = stablehlo::ReshapeOp::create(
    //   bodyBuilder,
    //   loc, RankedTensorType::get({1}, i32), mScalar);
    // Value n = stablehlo::ReshapeOp::create(
    //   bodyBuilder,
    //   loc, RankedTensorType::get({1}, i32), nScalar);
    // Value k = stablehlo::ReshapeOp::create(
    //   bodyBuilder,
    //   loc, RankedTensorType::get({1}, i32), kScalar);

    // If transA or transB matches any of these enums, take the transpose
    SmallVector<int> transposeEnums = {'T', 't', 'C', 'c'};
    Value transA = getIsEnum(bodyBuilder, loc, transAenum, transposeEnums);
    Value transB = getIsEnum(bodyBuilder, loc, transBenum, transposeEnums);

    // Value alpha = bodyBuilder.create<LLVM::LoadOp>(loc, f32, alpha_ptr);
    // Value beta = bodyBuilder.create<LLVM::LoadOp>(loc, f32, beta_ptr);
    // STEP 1. Slice A and B based on transpose and leading dimensions.
    // Use stablehlo.slice & stablehlo.transpose.
    // Compute shapes: [m, k] or [k, m] depending on trans flags.
    auto idxTy = bodyBuilder.getIndexType();

    // Zero constant index


    // Slice true shapes before transpose
    // Value A_sliced = buildSlice(A, m, k, lda);
    // Value B_sliced = buildSlice(B, k, n, ldb);
    Value A_sliced = getDynamicSlice(bodyBuilder, loc, A, m, k);
    Value B_sliced = getDynamicSlice(bodyBuilder, loc, B, k, n);

    // Transpose conditionally
    auto transpose2D = [&](OpBuilder &myBuilder, Value t) -> Value {
      SmallVector<int64_t> perm{1, 0};
      return stablehlo::TransposeOp::create(
          myBuilder,
          loc, dynTensor, t,
          perm
        );
    };


    auto A_if = stablehlo::IfOp::create(
        bodyBuilder,
        loc,
        A_sliced.getType(),   // result type
        transA
    );

    // Fill in the "then" region
    auto &thenRegion = A_if.getTrueBranch();
    Block *thenBlock = new mlir::Block();
    thenRegion.push_back(thenBlock);
    OpBuilder ifBuilder(thenBlock, thenBlock->begin());
    Value thenVal = transpose2D(ifBuilder, A_sliced); // produce Value of type resultType
    ifBuilder.create<stablehlo::ReturnOp>(loc, thenVal);

    // Fill in the "else" region
    auto &elseRegion = A_if.getFalseBranch();
    Block *elseBlock = new mlir::Block();
    elseRegion.push_back(elseBlock);
    OpBuilder elseBuilder(elseBlock, elseBlock->begin());
    elseBuilder.create<stablehlo::ReturnOp>(loc, A_sliced);

    Value A_eff = A_if.getResult(0);

    auto B_if = stablehlo::IfOp::create(
        bodyBuilder,
        loc,
        B_sliced.getType(),   // result type
        transB
    );
    // Fill in the "then" region
    auto &thenRegionB = B_if.getTrueBranch();
    Block *thenBlockB = new mlir::Block();
    thenRegionB.push_back(thenBlockB);
    OpBuilder ifBuilderB(thenBlockB, thenBlockB->begin());
    Value thenValB = transpose2D(ifBuilderB, B_sliced); // produce Value of type resultType
    ifBuilderB.create<stablehlo::ReturnOp>(loc, thenValB);

    // Fill in the "else" region
    auto &elseRegionB = B_if.getFalseBranch();
    Block *elseBlockB = new mlir::Block();
    elseRegionB.push_back(elseBlockB);
    OpBuilder elseBuilderB(elseBlockB, elseBlockB->begin());
    elseBuilderB.create<stablehlo::ReturnOp>(loc, B_sliced);

    Value B_eff = B_if.getResult(0);
    // Value B_eff = stablehlo::IfOp::create(
    //     bodyBuilder,
    //     loc, transB, transpose2D(B_sliced), B_sliced
    //   );

    // STEP 2: Dot general: A_eff [m,k], B_eff [k,n] => [m,n]
    // Mixed batch dims are empty; contracting dimension is {1}.
    // auto resultType = UnrankedTensorType::get(f32);

    auto dotDimNumbers = stablehlo::DotDimensionNumbersAttr::get(
        bodyBuilder.getContext(),
        /*lhsBatchingDims=*/{},
        /*rhsBatchingDims=*/{},
        /*lhsContractingDims=*/{1},
        /*rhsContractingDims=*/{0}
      );

    Value dot =
        stablehlo::DotGeneralOp::create(
            bodyBuilder,
            loc, dynTensor, A_eff, B_eff,
            dotDimNumbers, nullptr, nullptr);
    // Value dot = builder.create<stablehlo::DotGeneralOp>(
    //     loc, resultType, A, B, dotDimNumbers,
    //     nullptr,  // precision_config
    //     nullptr); // algorithm

    // STEP 3: alpha * dot + beta * C
    // Scale dot
    Value broadcastSize = getPairFromScalars(bodyBuilder, loc, m, n);
    llvm::ArrayRef<int64_t> b_dims = {0, 1};
    auto b_dimsAttr = mlir::DenseI64ArrayAttr::get(ctx, b_dims);
    auto alphaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), alpha
    );
    Value alphaBroadcast = stablehlo::DynamicBroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), alphaTensor, broadcastSize, b_dimsAttr);
    // alphaBroadcast.dump();
    Value scaledDot = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      dynTensor,
      dot,
      alphaBroadcast);

    // Slice C to correct [m,n] shape
    // Value C_sliced = buildSlice(C, m, n, ldc);
    Value C_sliced = getDynamicSlice(bodyBuilder, loc, C, m, n);

    // Scale C
    auto betaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), beta
    );
    Value betaBroadcast = stablehlo::DynamicBroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), betaTensor, broadcastSize, b_dimsAttr);
    betaBroadcast.dump();
    Value scaledC = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      dynTensor,
      C_sliced,
      betaBroadcast);

    // Add: out = scaledDot + scaledC
    Value out =
        bodyBuilder.create<stablehlo::AddOp>(loc, dynTensor, scaledDot,
                                            scaledC);

    func::ReturnOp::create(bodyBuilder, loc, ValueRange{transAenum, transBenum, m, n, k, alpha, A, lda, B, ldb, beta, out, ldc});
    return fn;
  }

  void runOnOperation() override {
    auto op = getOperation();
    llvm::errs() << "=== BlasRaisingPass running ===\n";
    llvm::errs().flush();

    op->walk([&](LLVM::LLVMFuncOp callOp) {
      auto calleeName = callOp.getName();
      llvm::errs() << "name: " << calleeName << "\n";

      if (calleeName == "cublasSgemm_v2") {
        llvm::errs() << "one call found\n";
        getCublasSGemm_v2(callOp);

        // cublasCalls.push_back(callOp);
      }
    });

    llvm::errs() << "=== BlasRaisingPass done ===\n";
    // llvm::errs().flush();
  }
};

} // end anonymous namespace


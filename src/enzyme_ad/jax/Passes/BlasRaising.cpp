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

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BLASRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

  // TODO: better input mapping, reshape 1D inputs to be 2D
struct BlasRaisingPass
    : public enzyme::impl::BlasRaisingPassBase<BlasRaisingPass> {
  using BlasRaisingPassBase::BlasRaisingPassBase;

  llvm::StringMap<std::vector<Type>> typeMap;
  void initializeInputMap(MLIRContext *ctx) {
    Type i32 = IntegerType::get(ctx, 32);
    Type f32 = Float32Type::get(ctx);
    Type f32DynTensor2D = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);

    // cublasSGemm_v2
    typeMap["cublasSGemm_v2"].push_back(i32); // transA
    typeMap["cublasSGemm_v2"].push_back(i32); // transB
    typeMap["cublasSGemm_v2"].push_back(i32); // m
    typeMap["cublasSGemm_v2"].push_back(i32); // n
    typeMap["cublasSGemm_v2"].push_back(i32); // k
    typeMap["cublasSGemm_v2"].push_back(f32); // alpha
    typeMap["cublasSGemm_v2"].push_back(f32DynTensor2D); // A
    typeMap["cublasSGemm_v2"].push_back(i32); // lda
    typeMap["cublasSGemm_v2"].push_back(f32DynTensor2D); // B
    typeMap["cublasSGemm_v2"].push_back(i32); // ldb
    typeMap["cublasSGemm_v2"].push_back(f32); // beta
    typeMap["cublasSGemm_v2"].push_back(f32DynTensor2D); // C
    typeMap["cublasSGemm_v2"].push_back(i32); // ldc
  }

  SmallVector<Value> transformOperands(LLVM::CallOp call, StringRef name) {
    auto modOp = call->getParentOfType<ModuleOp>();
    OpBuilder builder(call);
    ArrayRef<Type> targetTypes(typeMap[name]);
    Location loc = call.getLoc();

    SmallVector<Value> newOperands;
    int idx = 0;
    for (auto it = std::next(call.getOperands().begin());
        it != call.getOperands().end(); ++it, ++idx) {
      Value arg = *it;
      Type desiredType = targetTypes[idx];

      // Largely copied from AffineToStableHLORaising.cpp
      // Is tensor, just convert to memref
      if (auto tensorType = dyn_cast<TensorType>(desiredType)) {
        auto MT =
            MemRefType::get(
              {ShapedType::kDynamic},
              tensorType.getElementType()
            );
        newOperands.push_back(enzymexla::Pointer2MemrefOp::create(builder, loc, MT, arg));
        continue;
      }

      // Not a tensor, must check if you have to load the ptr
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        arg = LLVM::LoadOp::create(builder, loc, desiredType, arg);
      }
      // convert scalar value into appropriate memref
      auto MT0 =
          MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                          builder.getI64IntegerAttr(0));
      auto MT =
          MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                          builder.getI64IntegerAttr(1));

      auto res =
          gpu::AllocOp::create(builder, loc, MT, (mlir::Type) nullptr,
                                ValueRange(), ValueRange(), ValueRange())
              ->getResult(0);

      auto res0 = memref::AllocaOp::create(builder, loc, MT0);
      affine::AffineStoreOp::create(builder, loc, arg, res0,
                                    builder.getMultiDimIdentityMap(0),
                                    ValueRange());
      auto c1 = arith::ConstantIndexOp::create(builder, loc, 1);
      enzymexla::MemcpyOp::create(builder, loc, (mlir::Type) nullptr,
                                  ValueRange(), res, res0, c1);

      builder.setInsertionPointAfter(call);
      // gpu::DeallocOp::create(builder, loc, (mlir::Type) nullptr,
      //                         ValueRange(), res);
      builder.setInsertionPoint(call);
      newOperands.push_back(res);
    }
    return newOperands;
  }

  Type getElemType(Value tensor) {
    Type inputTy = tensor.getType();
    auto vecInputTy = cast<TensorType>(inputTy);
    return vecInputTy.getElementType();
  }

  Value makePairFromScalars(OpBuilder &builder, Location &loc, Value a, Value b) {
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

  Value makeDynamicSlice(OpBuilder &builder, Location &loc, Value tensor, Value dim0, Value dim1) {
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

    auto limit_tensor = makePairFromScalars(builder, loc, dim0, dim1);

    return stablehlo::RealDynamicSliceOp::create(builder, loc, tensor.getType(), tensor, zero_tensor, limit_tensor, one_tensor);
  }

  Value makeDynamicUnflatten(OpBuilder &builder, Location &loc, Value tensor, Value ldim) {
    Type inputTy = tensor.getType();              // tensor<?xf32>
    auto vecInputTy = cast<RankedTensorType>(inputTy);
    Type elemTy = vecInputTy.getElementType();

    // --- Get total length ---
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value total = builder.create<tensor::DimOp>(loc, tensor, c0);  // index type

    // --- Extract scalar from ldim: tensor<i32> -> i32 ---
    Value ldimScalar = builder.create<tensor::ExtractOp>(loc, ldim);

    // --- Convert ldim to index ---
    Value ldimAsIndex = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), ldimScalar);

    // --- Compute rest = total / ldim ---
    Value rest = builder.create<arith::DivUIOp>(loc, total, ldimAsIndex);

    // --- Build shape tensor<2xindex> ---
    Value shape = builder.create<tensor::FromElementsOp>(
        loc,
        RankedTensorType::get({2}, builder.getIndexType()),
        ValueRange{ldimAsIndex, rest}
    );

    // --- Result type: tensor<?x?xf32> ---
    auto resultTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic},
        elemTy);

    // --- Perform reshape ---
    return builder.create<tensor::ReshapeOp>(
        loc,
        resultTy,
        tensor,
        shape);
  }

  Value makeDynamicFlatten(OpBuilder &builder, Location &loc, Value tensor) {
    // types
    Type elemType = getElemType(tensor);
    auto flatTy = RankedTensorType::get({ShapedType::kDynamic}, elemType);

    // index constants
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    // dims
    auto dim0 = builder.create<tensor::DimOp>(loc, tensor, c0);
    auto dim1 = builder.create<tensor::DimOp>(loc, tensor, c1);

    // total size = dim0 * dim1
    auto total = builder.create<arith::MulIOp>(loc, dim0, dim1);

    // shape tensor for 1D result
    Value shape = builder.create<tensor::FromElementsOp>(loc, ValueRange{total});

    // reshape back to 1D
    return builder.create<tensor::ReshapeOp>(
        loc,
        flatTy,
        tensor,
        shape
      );
  }

  Value makeDynamicUpdateSlice(OpBuilder &builder, Location &loc, Value orig, Value update, Value dim0, Value dim1) {
    auto i32 = builder.getI32Type();
    auto i32ZeroTensor = RankedTensorType::get({}, i32);
    // Type dynTensor = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    Value zero_tensor = stablehlo::ConstantOp::create(
        builder,
        loc,
        DenseIntElementsAttr::get(i32ZeroTensor, {0})
      );

    // auto limit_tensor = makePairFromScalars(builder, loc, dim0, dim1);

    return stablehlo::DynamicUpdateSliceOp::create(builder, loc, orig.getType(), ValueRange{orig, update, zero_tensor, zero_tensor});
  }


  void replaceCublasSGemm_v2(LLVM::CallOp call) {
    constexpr StringRef fnName = "raised_cublasSGemm_v2";
    auto module = call->getParentOfType<ModuleOp>();
    func::FuncOp f = module.lookupSymbol<func::FuncOp>(fnName);
    if (!f) {
      llvm::errs() << "replacing function not found\n";
    }
    SmallVector<Value> newOperands = transformOperands(call, "cublasSGemm_v2");

    OpBuilder builder(call);
    enzymexla::XLAWrapperOp::create(
      builder, call->getLoc(), SymbolRefAttr::get(f),
      llvm::to_vector(newOperands), nullptr, nullptr);
    
    // call.getResult().getType().dump();
  }
  // TODO: Handle differing leading dimensions
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
    Type flatDynTensor = RankedTensorType::get({ShapedType::kDynamic}, f32);
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
        newInputs.push_back(flatDynTensor);
        results.push_back(flatDynTensor);
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
    Value A_flat = args[6];
    Value lda = args[7];
    Value B_flat = args[8];
    Value ldb = args[9];
    Value beta = args[10];
    Value C_flat = args[11];
    Value ldc = args[12];

    Value A = makeDynamicUnflatten(bodyBuilder, loc, A_flat, lda);
    Value B = makeDynamicUnflatten(bodyBuilder, loc, B_flat, ldb);
    Value C = makeDynamicUnflatten(bodyBuilder, loc, C_flat, ldc);
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
    Value A_sliced = makeDynamicSlice(bodyBuilder, loc, A, m, k);
    Value B_sliced = makeDynamicSlice(bodyBuilder, loc, B, k, n);

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
    Value broadcastSize = makePairFromScalars(bodyBuilder, loc, m, n);
    llvm::ArrayRef<int64_t> b_dims = {0, 1};
    auto b_dimsAttr = mlir::DenseI64ArrayAttr::get(ctx, b_dims);
    auto alphaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), alpha
    );
    Value alphaBroadcast = stablehlo::DynamicBroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), alphaTensor, broadcastSize, b_dimsAttr);
    Value scaledDot = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      dynTensor,
      dot,
      alphaBroadcast);

    // Slice C to correct [m,n] shape
    // Value C_sliced = buildSlice(C, m, n, ldc);
    Value C_sliced = makeDynamicSlice(bodyBuilder, loc, C, m, n);

    // Scale C
    auto betaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), beta
    );
    Value betaBroadcast = stablehlo::DynamicBroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), betaTensor, broadcastSize, b_dimsAttr);
    Value scaledC = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      dynTensor,
      C_sliced,
      betaBroadcast);

    // Add: out = scaledDot + scaledC
    Value update =
        bodyBuilder.create<stablehlo::AddOp>(loc, dynTensor, scaledDot,
                                            scaledC);
    Value out2D = makeDynamicUpdateSlice(bodyBuilder, loc, C, update, m, n);
    Value outFlat = makeDynamicFlatten(bodyBuilder, loc, out2D);

    func::ReturnOp::create(bodyBuilder, loc, ValueRange{transAenum, transBenum, m, n, k, alpha, A_flat, lda, B_flat, ldb, beta, outFlat, ldc});
    return fn;
  }

  void runOnOperation() override {
    auto op = getOperation();
    llvm::errs() << "=== BlasRaisingPass running ===\n";
    llvm::errs().flush();
    initializeInputMap(op->getContext());

    op->walk([&](LLVM::LLVMFuncOp callOp) {
      auto calleeName = callOp.getName();
      if (calleeName == "cublasSgemm_v2") {
        getCublasSGemm_v2(callOp);
      }
    });

    // SmallVector<Value> oldCalls;
    // op->walk([&](LLVM::CallOp callOp) {
    //   llvm::StringRef calleeName = callOp.getCallee().value_or("");
    //   llvm::errs() << "call: " << calleeName << "\n";
    //   if (calleeName == "cublasSgemm_v2") {
    //     llvm::errs() << "call found\n";
    //     replaceCublasSGemm_v2(callOp);
    //     oldCalls.push_back(callOp);
    //   }
    // });
    SmallVector<LLVM::CallOp, 4> cublasCalls;

    op->walk([&](LLVM::CallOp callOp) {
      auto calleeName = callOp.getCallee().value_or("");
      if (calleeName == "cublasSgemm_v2") {
        llvm::errs() << "Found cublasSgemm_v2 call\n";
        replaceCublasSGemm_v2(callOp);

        cublasCalls.push_back(callOp);
      }
    });

    for (auto call : cublasCalls) {
      OpBuilder builder(call);
      Value zero = LLVM::ConstantOp::create(builder, call->getLoc(), builder.getI32Type(), 0);

      for (auto result : call.getResults()) {
        result.replaceAllUsesWith(zero);
      }
      call.erase();
    }

    llvm::errs() << "=== BlasRaisingPass done ===\n";
    op->dump();
    // llvm::errs().flush();
  }
};

} // end anonymous namespace


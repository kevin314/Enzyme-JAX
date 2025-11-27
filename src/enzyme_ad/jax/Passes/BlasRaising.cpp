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

#define CUBLAS_OP_N (0)
#define CUBLAS_OP_T (1)
#define CUBLAS_OP_C (2)

namespace {

  // TODO: better input mapping, reshape 1D inputs to be 2D
struct BlasRaisingPass
    : public enzyme::impl::BlasRaisingPassBase<BlasRaisingPass> {
  using BlasRaisingPassBase::BlasRaisingPassBase;

  struct ValueWrapper {
    Value val;
    bool isConst;
  };

  using CublasConstructor = std::function<func::FuncOp(LLVM::CallOp, SmallVector<ValueWrapper>, func::FuncOp)>;

  llvm::StringMap<SmallVector<Type>> typeMap;
  void initializeInputMap(MLIRContext *ctx) {
    Type i32 = IntegerType::get(ctx, 32);
    Type f32 = Float32Type::get(ctx);
    Type f32DynTensor2D = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);

    // cublasSgemm_v2
    typeMap["cublasSgemm_v2"].push_back(i32); // transA
    typeMap["cublasSgemm_v2"].push_back(i32); // transB
    typeMap["cublasSgemm_v2"].push_back(i32); // m
    typeMap["cublasSgemm_v2"].push_back(i32); // n
    typeMap["cublasSgemm_v2"].push_back(i32); // k
    typeMap["cublasSgemm_v2"].push_back(f32); // alpha
    typeMap["cublasSgemm_v2"].push_back(f32DynTensor2D); // A
    typeMap["cublasSgemm_v2"].push_back(i32); // lda
    typeMap["cublasSgemm_v2"].push_back(f32DynTensor2D); // B
    typeMap["cublasSgemm_v2"].push_back(i32); // ldb
    typeMap["cublasSgemm_v2"].push_back(f32); // beta
    typeMap["cublasSgemm_v2"].push_back(f32DynTensor2D); // C
    typeMap["cublasSgemm_v2"].push_back(i32); // ldc
  }
  
  StringRef getRaisedFuncName(StringRef funcName) {
    static uint64_t counter = 0;
    return StringRef(funcName.str() + std::to_string(counter++));
  }

  SmallVector<ValueWrapper> transformOperands(LLVM::CallOp call, StringRef name) {
    auto modOp = call->getParentOfType<ModuleOp>();
    OpBuilder builder(call);
    Location loc = call.getLoc();

    SmallVector<Type> targetTypes(typeMap[name.str()]);
    SmallVector<ValueWrapper> newOperands;
    int idx = 0;
    for (auto it = std::next(call.getOperands().begin());
        it != call.getOperands().end(); ++it, ++idx) {
      Value arg = *it;
      Type desiredType = targetTypes[idx];

      Attribute attr;
      if (matchPattern(arg, m_Constant(&attr))) {
        newOperands.push_back(ValueWrapper{arg, true});
        continue;
      }

      // Largely copied from AffineToStableHLORaising.cpp
      // Is tensor, just convert to memref
      if (auto tensorType = dyn_cast<TensorType>(desiredType)) {
        auto MT =
            MemRefType::get(
              {ShapedType::kDynamic},
              tensorType.getElementType()
            );
        newOperands.push_back(ValueWrapper{enzymexla::Pointer2MemrefOp::create(builder, loc, MT, arg), false});
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
      // TODO: add check for size of datatype, and set 4 to instead be that size
      auto c1 = arith::ConstantIndexOp::create(builder, loc, 4);
      enzymexla::MemcpyOp::create(builder, loc, (mlir::Type) nullptr,
                                  ValueRange(), res, res0, c1);

      builder.setInsertionPointAfter(call);
      gpu::DeallocOp::create(builder, loc, (mlir::Type) nullptr,
                              ValueRange(), res);
      builder.setInsertionPoint(call);
      newOperands.push_back(ValueWrapper{res, false});
    }
    return newOperands;
  }

  Type getElemType(Value tensor) {
    Type inputTy = tensor.getType();
    auto vecInputTy = cast<TensorType>(inputTy);
    return vecInputTy.getElementType();
  }

  int64_t getConstantValue(Value constant) {
    auto dim0C = constant.getDefiningOp<stablehlo::ConstantOp>();
    auto dim0Dense = cast<DenseElementsAttr>(dim0C.getValue());
    return dim0Dense.getSplatValue<IntegerAttr>().getInt();
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

  Value makePairFromScalars(OpBuilder &builder, Location &loc, Value a, Value b) {
    auto i64 = builder.getI64Type();
    auto a64 = stablehlo::ConvertOp::create(builder, loc, RankedTensorType::get({}, i64), a);
    auto b64 = stablehlo::ConvertOp::create(builder, loc, RankedTensorType::get({}, i64), b);

    auto aTensor = stablehlo::ReshapeOp::create(
      builder,
      loc, RankedTensorType::get({1}, i64), a64
    );
    auto bTensor = stablehlo::ReshapeOp::create(
      builder,
      loc, RankedTensorType::get({1}, i64), b64
    );
    return stablehlo::ConcatenateOp::create(
      builder, loc, RankedTensorType::get({2}, i64), ValueRange{aTensor, bTensor}, builder.getI64IntegerAttr(0)
    );
  }

  Value make2DTensor(OpBuilder &builder, Location &loc, Value tensor, int64_t ldim, int64_t num_rows, int64_t num_cols) {
    Type elemTy = getElemType(tensor);
    auto ctx = builder.getContext();

    auto sliced = stablehlo::SliceOp::create(builder, loc, RankedTensorType::get({ldim * num_cols}, elemTy), tensor,
      DenseI64ArrayAttr::get(ctx, {(int64_t) 0}),
      DenseI64ArrayAttr::get(ctx, {ldim * num_cols}),
      DenseI64ArrayAttr::get(ctx, {(int64_t) 1})
    );

    auto reshaped = stablehlo::ReshapeOp::create(
      builder,
        loc,
        RankedTensorType::get(
          {ldim, num_cols},
          elemTy
        ),
        sliced
      );
    
    return stablehlo::SliceOp::create(builder, loc, RankedTensorType::get({num_rows, num_cols}, elemTy), reshaped,
      DenseI64ArrayAttr::get(ctx, {(int64_t) 0, (int64_t) 0}),
      DenseI64ArrayAttr::get(ctx, {num_rows, num_cols}),
      DenseI64ArrayAttr::get(ctx, {(int64_t) 1, (int64_t) 1})
    );
  }

  Value make1DTensor(OpBuilder &builder, Location loc,
                          Value orig,
                          Value update,
                          int64_t ldim,
                          int64_t num_rows,
                          int64_t num_cols) {
    auto ctx = builder.getContext();
    Type elemTy = getElemType(orig);

    auto startIndices = DenseI64ArrayAttr::get(ctx, {0, 0});

    Value zero_tensor = stablehlo::ConstantOp::create(
        builder,
        loc,
        DenseIntElementsAttr::get(RankedTensorType::get({}, builder.getI64Type()), {(int64_t) 0})
      );

    Value flattened =
        stablehlo::ReshapeOp::create(builder, loc, RankedTensorType::get({num_rows*num_cols}, elemTy), update);

    return stablehlo::DynamicUpdateSliceOp::create(builder, loc, orig.getType(), ValueRange{orig, flattened, zero_tensor});
  }

  void replaceCall(LLVM::CallOp call, CublasConstructor constructor, StringRef name) {
    MLIRContext *ctx = call.getContext();
    auto loc = call.getLoc();
    auto module = call->getParentOfType<ModuleOp>();

    StringRef fnName = getRaisedFuncName(name);

    // transform operands
    SmallVector<ValueWrapper> operands = transformOperands(call, StringRef(name));

    // Construct new function type
    SmallVector<Type> newInputs;
    for (auto &value_wrapper : operands) {
      if (!value_wrapper.isConst) {
        auto argTy = cast<MemRefType>(value_wrapper.val.getType());
        newInputs.push_back(RankedTensorType::get(argTy.getShape(), argTy.getElementType()));
      }
    }
    SmallVector<Type> results(newInputs);
    auto newFuncType = mlir::FunctionType::get(ctx, newInputs, results);

    // Construct new function
    auto fn = func::FuncOp::create(loc, fnName, newFuncType);
    fn.setPrivate();
    module.push_back(fn);

    // fill in function with corresponding constructor
    func::FuncOp f = constructor(call, operands, fn);

    // create call to new function
    SmallVector<Value> args;
    for (auto value_wrapper : operands) {
      if (!value_wrapper.isConst) {
        args.push_back(value_wrapper.val);
      }
    }
    OpBuilder builder(call);
    enzymexla::XLAWrapperOp::create(
      builder, call->getLoc(), SymbolRefAttr::get(f),
      llvm::to_vector(args), nullptr, nullptr);
    
  }

  SmallVector<Value> makeArgs(OpBuilder builder, Location loc, mlir::Block *entry, SmallVector<ValueWrapper> operands) {
    // Merge passed args and constants into one array
    auto args = entry->getArguments();
    SmallVector<Value> allArgs;
    int argIdx = 0;
    for (auto &value_wrapper : operands) {
      if (value_wrapper.isConst) {
        auto unrankedTensorType = RankedTensorType::get({}, value_wrapper.val.getType());
        Attribute attr;
        matchPattern(value_wrapper.val, m_Constant(&attr));

        allArgs.push_back(stablehlo::ConstantOp::create(
          builder, value_wrapper.val.getLoc(), unrankedTensorType,
          SplatElementsAttr::get(
              unrankedTensorType,
              ArrayRef<Attribute>(attr))
          ));
      } else {
        allArgs.push_back(args[argIdx]);
        ++argIdx;
      }
    }
    return allArgs;
  }

  func::FuncOp getCublasSGemm_v2(LLVM::CallOp call, SmallVector<ValueWrapper> operands, func::FuncOp fn) {
    auto module = call->getParentOfType<ModuleOp>();
    auto loc = call.getLoc();
    auto ctx = call.getContext();

    Block *entry = fn.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());
  
    // Extract arguments
    SmallVector<Value> allArgs(makeArgs(bodyBuilder, loc, entry, operands));
    int i = 0;
    Value transAenum = allArgs[i++];
    Value transBenum = allArgs[i++];
    Value m = allArgs[i++];
    Value n = allArgs[i++];
    Value k = allArgs[i++];
    Value alpha = allArgs[i++];
    Value A_flat = allArgs[i++];
    Value lda = allArgs[i++];
    Value B_flat = allArgs[i++];
    Value ldb = allArgs[i++];
    Value beta = allArgs[i++];
    Value C_flat = allArgs[i++];
    Value ldc = allArgs[i++];

    Type elemTy = getElemType(A_flat);

    int64_t m_const = getConstantValue(m);
    int64_t n_const = getConstantValue(n);
    int64_t k_const = getConstantValue(k);
    int64_t lda_const = getConstantValue(lda);
    int64_t ldb_const = getConstantValue(ldb);
    int64_t ldc_const = getConstantValue(ldc);

    int64_t transAenum_const = getConstantValue(transAenum);
    int64_t transBenum_const = getConstantValue(transBenum);


    // If transA or transB matches any of these enums, take the transpose
    SmallVector<int64_t> transposeEnums = {1, 2};
    bool transA = llvm::is_contained(transposeEnums, transAenum_const);
    bool transB = llvm::is_contained(transposeEnums, transBenum_const);
    // Value transA = getIsEnum(bodyBuilder, loc, transAenum, transposeEnums);
    // Value transB = getIsEnum(bodyBuilder, loc, transBenum, transposeEnums);

    // Column-major matrix in memory has same layout as row-major transpose
    // So column-major A[m,k] = row-major A^T[k,m] when reshaped
    // make2DTensor slices ldim*cols elements, reshapes to [ldim, cols], slices to [rows, cols]
    // For A: want row-major [k,m], so pass ldim=k, rows=k, cols=m
    Value A_eff;
    if (transA) {
      // transA: column-major is A^T[k,m], so row-major is A[m,k]
      A_eff = make2DTensor(bodyBuilder, loc, A_flat, lda_const, m_const, k_const);
    } else {
      // Normal: column-major A[m,k], so row-major is A^T[k,m]
      // ldim*cols = k*m, reshape to [k,m], slice to [k,m]
      // But make2DTensor uses ldim as first reshape dim, so use ldim=m
      Type elemTy = getElemType(A_flat);
      auto ctx = bodyBuilder.getContext();
      auto sliced = stablehlo::SliceOp::create(bodyBuilder, loc,
        RankedTensorType::get({m_const * k_const}, elemTy), A_flat,
        DenseI64ArrayAttr::get(ctx, {(int64_t)0}),
        DenseI64ArrayAttr::get(ctx, {m_const * k_const}),
        DenseI64ArrayAttr::get(ctx, {(int64_t)1}));
      A_eff = stablehlo::ReshapeOp::create(bodyBuilder, loc,
        RankedTensorType::get({k_const, m_const}, elemTy), sliced);
    }

    Value B_eff;
    if (transB) {
      // transB: column-major is B^T[n,k], so row-major is B[k,n]
      B_eff = make2DTensor(bodyBuilder, loc, B_flat, ldb_const, k_const, n_const);
    } else {
      // Normal: column-major B[k,n], so row-major is B^T[n,k]
      Type elemTy = getElemType(B_flat);
      auto ctx = bodyBuilder.getContext();
      auto sliced = stablehlo::SliceOp::create(bodyBuilder, loc,
        RankedTensorType::get({k_const * n_const}, elemTy), B_flat,
        DenseI64ArrayAttr::get(ctx, {(int64_t)0}),
        DenseI64ArrayAttr::get(ctx, {k_const * n_const}),
        DenseI64ArrayAttr::get(ctx, {(int64_t)1}));
      B_eff = stablehlo::ReshapeOp::create(bodyBuilder, loc,
        RankedTensorType::get({n_const, k_const}, elemTy), sliced);
    }
    // Value A_sliced = makeDynamicSlice(bodyBuilder, loc, A, m_const, k_const);
    // Value B_sliced = makeDynamicSlice(bodyBuilder, loc, B, k_const, n_const);

    // Transpose conditionally
    // auto transpose2D = [&](OpBuilder &myBuilder, Value t, int64_t dim0_const, int64_t dim1_const) -> Value {
    //   SmallVector<int64_t> perm{1, 0};
    //   Type elemTy = getElemType(t);
    //   return stablehlo::TransposeOp::create(
    //       myBuilder,
    //       loc, RankedTensorType::get({dim1_const, dim0_const}, elemTy), t,
    //       perm
    //     );
    // };


    // auto A_if = stablehlo::IfOp::create(
    //     bodyBuilder,
    //     loc,
    //     A_sliced.getType(),   // result type
    //     transA
    // );

    // // Fill in the "then" region
    // auto &thenRegion = A_if.getTrueBranch();
    // Block *thenBlock = new mlir::Block();
    // thenRegion.push_back(thenBlock);
    // OpBuilder ifBuilder(thenBlock, thenBlock->begin());
    // Value thenVal = transpose2D(ifBuilder, A_sliced); // produce Value of type resultType
    // ifBuilder.create<stablehlo::ReturnOp>(loc, thenVal);

    // // Fill in the "else" region
    // auto &elseRegion = A_if.getFalseBranch();
    // Block *elseBlock = new mlir::Block();
    // elseRegion.push_back(elseBlock);
    // OpBuilder elseBuilder(elseBlock, elseBlock->begin());
    // elseBuilder.create<stablehlo::ReturnOp>(loc, A_sliced);

    // Value A_eff = A_if.getResult(0);

    // auto B_if = stablehlo::IfOp::create(
    //     bodyBuilder,
    //     loc,
    //     B_sliced.getType(),   // result type
    //     transB
    // );
    // // Fill in the "then" region
    // auto &thenRegionB = B_if.getTrueBranch();
    // Block *thenBlockB = new mlir::Block();
    // thenRegionB.push_back(thenBlockB);
    // OpBuilder ifBuilderB(thenBlockB, thenBlockB->begin());
    // Value thenValB = transpose2D(ifBuilderB, B_sliced); // produce Value of type resultType
    // ifBuilderB.create<stablehlo::ReturnOp>(loc, thenValB);

    // // Fill in the "else" region
    // auto &elseRegionB = B_if.getFalseBranch();
    // Block *elseBlockB = new mlir::Block();
    // elseRegionB.push_back(elseBlockB);
    // OpBuilder elseBuilderB(elseBlockB, elseBlockB->begin());
    // elseBuilderB.create<stablehlo::ReturnOp>(loc, B_sliced);

    // Value B_eff = B_if.getResult(0);
    // Value B_eff = stablehlo::IfOp::create(
    //     bodyBuilder,
    //     loc, transB, transpose2D(B_sliced), B_sliced
    //   );

    // STEP 2: Dot general: A_eff [m,k], B_eff [k,n] => [m,n]
    // Mixed batch dims are empty; contracting dimension is {1}.
    // auto resultType = UnrankedTensorType::get(f32);

    // cuBLAS uses column-major, StableHLO uses row-major
    // Column-major: C(m,n) = A(m,k) * B(k,n)
    // Row-major: C_row(n,m) = B_row(n,k) * A_row(k,m)
    // A_eff is now [k, m] and B_eff is now [n, k]
    // Compute B_eff[n,k] * A_eff[k,m] = [n,m]
    // This [n,m] row-major result corresponds to [m,n] column-major output
    auto dotDimNumbers = stablehlo::DotDimensionNumbersAttr::get(
        bodyBuilder.getContext(),
        /*lhsBatchingDims=*/{},
        /*rhsBatchingDims=*/{},
        /*lhsContractingDims=*/{1},  // B_eff[n,k] contracts on dim 1 (k)
        /*rhsContractingDims=*/{0}   // A_eff[k,m] contracts on dim 0 (k)
      );

    Value dot =
        stablehlo::DotGeneralOp::create(
            bodyBuilder,
            loc, RankedTensorType::get({n_const, m_const}, elemTy), B_eff, A_eff,
            dotDimNumbers, nullptr, nullptr);

    // Value dot = builder.create<stablehlo::DotGeneralOp>(
    //     loc, resultType, A, B, dotDimNumbers,
    //     nullptr,  // precision_config
    //     nullptr); // algorithm

    // STEP 3: alpha * dot + beta * C
    // All operations are in row-major [n, m] space
    // Scale dot
    Value broadcastSize = makePairFromScalars(bodyBuilder, loc, n, m);
    SmallVector<int64_t> b_dims = {0, 1};
    auto b_dimsAttr = mlir::DenseI64ArrayAttr::get(ctx, b_dims);
    auto alphaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, getElemType(alpha)), alpha
    );
    Value alphaBroadcast = stablehlo::BroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), alphaTensor, b_dimsAttr);
    Value scaledDot = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      alphaBroadcast.getType(),
      dot,
      alphaBroadcast);

    // C: column-major [m,n] = row-major [n,m] when reshaped
    auto C_sliced_1d = stablehlo::SliceOp::create(bodyBuilder, loc,
      RankedTensorType::get({m_const * n_const}, elemTy), C_flat,
      DenseI64ArrayAttr::get(ctx, {(int64_t)0}),
      DenseI64ArrayAttr::get(ctx, {m_const * n_const}),
      DenseI64ArrayAttr::get(ctx, {(int64_t)1}));
    Value C_sliced = stablehlo::ReshapeOp::create(bodyBuilder, loc,
      RankedTensorType::get({n_const, m_const}, elemTy), C_sliced_1d);

    // Scale C
    auto betaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, getElemType(beta)), beta
    );
    Value betaBroadcast = stablehlo::DynamicBroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), betaTensor, broadcastSize, b_dimsAttr);
    Value scaledC = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      C_sliced.getType(),
      C_sliced,
      betaBroadcast);

    // Add: out = scaledDot + scaledC
    Value update =
        bodyBuilder.create<stablehlo::AddOp>(loc, scaledDot.getType(), scaledDot,
                                            scaledC);
    // Result is row-major [n,m] which has same memory layout as column-major [m,n]
    // Just flatten and write back
    Value outFlat = stablehlo::ReshapeOp::create(bodyBuilder, loc,
      RankedTensorType::get({n_const * m_const}, elemTy), update);
    Value zero_i64 = stablehlo::ConstantOp::create(bodyBuilder, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({}, bodyBuilder.getI64Type()), {(int64_t)0}));
    outFlat = stablehlo::DynamicUpdateSliceOp::create(bodyBuilder, loc,
      C_flat.getType(), ValueRange{C_flat, outFlat, zero_i64});
    
    allArgs[11] = outFlat;
    SmallVector<Value> result;
    int idx = 0;
    for (auto &value_wrapper : operands) {
      if (!value_wrapper.isConst) {
        result.push_back(allArgs[idx]);
      }
      ++idx;
    }
    func::ReturnOp::create(bodyBuilder, loc, ValueRange{result});
    return fn;
  }

  void runOnOperation() override {
    auto op = getOperation();
    llvm::errs() << "=== BlasRaisingPass running ===\n";
    llvm::errs().flush();

    initializeInputMap(op->getContext());

    // op->walk([&](LLVM::LLVMFuncOp callOp) {
    //   auto calleeName = callOp.getName();
    //   if (calleeName == "cublasSgemm_v2") {
    //     getCublasSGemm_v2(callOp);
    //   }
    // });

    SmallVector<LLVM::CallOp, 4> cublasCalls;

    op->walk([&](LLVM::CallOp callOp) {
      auto calleeName = callOp.getCallee().value_or("");
      if (calleeName == "cublasSgemm_v2") {
        replaceCall(callOp, [this](LLVM::CallOp call, SmallVector<ValueWrapper> ops, func::FuncOp fn) {
          return this->getCublasSGemm_v2(call, ops, fn);
        }, StringRef("cublasSgemm_v2"));

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


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

#define GET_NEXT_ARG(NAME)                         \
  Value NAME;                                      \
  do {                                             \
    if (constants.count(idx) > 0) {                \
      NAME = constants.at(idx);                    \
    } else {                                       \
      NAME = args[tracker];                        \
      outputs[idx] = NAME;                         \
      tracker++;                                   \
    }                                              \
    idx++;                                         \
  } while (0)


namespace {

  // TODO: better input mapping, reshape 1D inputs to be 2D
struct BlasRaisingPass
    : public enzyme::impl::BlasRaisingPassBase<BlasRaisingPass> {
  using BlasRaisingPassBase::BlasRaisingPassBase;

  // struct MatrixArgsIdx {
  //   int64_t memref;
  //   int64_t num_rows;
  //   int64_t num_cols;
  //   int64_t ldim;
  //   int64_t op_enum;
  // };
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

  
  std::string getRaisedFuncName(const std::string &funcName) {
      static uint64_t counter = 0;
      return funcName + std::to_string(counter++);
  }

  SmallVector<Value> transformOperands(LLVM::CallOp call, StringRef name, std::map<int, Value> &constants) {
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
      // Is tensor, allocate XLA GPU memory and copy CUDA data to it
      if (auto tensorType = dyn_cast<TensorType>(desiredType)) {
        // Compute the buffer size for this tensor
        // For cublasSGemm_v2: idx 6=A (m*k), idx 8=B (k*n), idx 11=C (m*n)
        Value sizeVal;
        int64_t staticSize = -1;  // -1 means dynamic

        if (name == "cublasSGemm_v2") {
          // Get m, n, k from constants (indices 2, 3, 4)
          // Try to extract compile-time constant values
          auto getConstantInt = [](Value v) -> std::optional<int64_t> {
            Attribute attr;
            if (matchPattern(v, m_Constant(&attr))) {
              if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                return intAttr.getInt();
              }
            }
            return std::nullopt;
          };

          if (idx == 6) {  // A matrix: m * k
            if (constants.count(2) && constants.count(4)) {
              Value m = constants[2];
              Value k = constants[4];
              auto m_const = getConstantInt(m);
              auto k_const = getConstantInt(k);
              if (m_const && k_const) {
                staticSize = (*m_const) * (*k_const);
              }
              Value m_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), m);
              Value k_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), k);
              sizeVal = builder.create<arith::MulIOp>(loc, m_idx, k_idx);
            }
          } else if (idx == 8) {  // B matrix: k * n
            if (constants.count(3) && constants.count(4)) {
              Value k = constants[4];
              Value n = constants[3];
              auto k_const = getConstantInt(k);
              auto n_const = getConstantInt(n);
              if (k_const && n_const) {
                staticSize = (*k_const) * (*n_const);
              }
              Value k_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), k);
              Value n_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), n);
              sizeVal = builder.create<arith::MulIOp>(loc, k_idx, n_idx);
            }
          } else if (idx == 11) {  // C matrix: m * n
            if (constants.count(2) && constants.count(3)) {
              Value m = constants[2];
              Value n = constants[3];
              auto m_const = getConstantInt(m);
              auto n_const = getConstantInt(n);
              if (m_const && n_const) {
                staticSize = (*m_const) * (*n_const);
              }
              Value m_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), m);
              Value n_idx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), n);
              sizeVal = builder.create<arith::MulIOp>(loc, m_idx, n_idx);
            }
          }
        }

        if (!sizeVal) {
          llvm::errs() << "WARNING: Could not compute size for tensor at idx " << idx << "\n";
          sizeVal = builder.create<arith::ConstantIndexOp>(loc, 0);
          staticSize = 0;
        }

        // Allocate a new XLA buffer with the correct size
        MemRefType MT_xla;
        SmallVector<Value> dynamicSizes;
        if (staticSize >= 0) {
          MT_xla = MemRefType::get(
              {staticSize},
              tensorType.getElementType(),
              MemRefLayoutAttrInterface{},
              builder.getI64IntegerAttr(1)
            );
          llvm::errs() << "Allocating new XLA buffer with size " << staticSize << " at idx " << idx << "\n";
        } else {
          MT_xla = MemRefType::get(
              {ShapedType::kDynamic},
              tensorType.getElementType(),
              MemRefLayoutAttrInterface{},
              builder.getI64IntegerAttr(1)
            );
          dynamicSizes.push_back(sizeVal);
        }

        auto xla_memref = gpu::AllocOp::create(builder, loc, MT_xla, (mlir::Type) nullptr,
                              ValueRange(), dynamicSizes, ValueRange())
            ->getResult(0);

        // Create a dynamic memref from the source pointer for copying
        // Use dynamic shape to avoid type conflicts
        MemRefType MT_src_dynamic = MemRefType::get(
            {ShapedType::kDynamic},
            tensorType.getElementType(),
            MemRefLayoutAttrInterface{},
            builder.getI64IntegerAttr(1)
          );
        Value src_memref = enzymexla::Pointer2MemrefOp::create(builder, loc, MT_src_dynamic, arg);

        // Convert element count to byte size (sizeof(f32) = 4)
        Value elemSize = builder.create<arith::ConstantIndexOp>(loc, 4);
        Value byteSizeVal = builder.create<arith::MulIOp>(loc, sizeVal, elemSize);

        // Copy from the source (might be oversized) to the correctly-sized XLA buffer
        // The byte size ensures we only copy the correct amount
        enzymexla::MemcpyOp::create(builder, loc, (mlir::Type) nullptr,
                                    ValueRange(), xla_memref, src_memref, byteSizeVal);

        newOperands.push_back(xla_memref);
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
      // Compute byte size for scalar (element count * element size in bytes)
      unsigned elemBitWidth = arg.getType().getIntOrFloatBitWidth();
      unsigned elemByteSize = (elemBitWidth + 7) / 8;  // Round up to nearest byte
      auto byteSize = arith::ConstantIndexOp::create(builder, loc, elemByteSize);
      enzymexla::MemcpyOp::create(builder, loc, (mlir::Type) nullptr,
                                  ValueRange(), res, res0, byteSize);

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
    // auto limit_tensor = makePairFromScalars(builder, loc, dim0, dim1);

    auto updated2D = stablehlo::DynamicUpdateSliceOp::create(builder, loc, orig.getType(), ValueRange{orig, flattened, zero_tensor});



    return updated2D;
  }

  void replaceCublasSGemm_v2(LLVM::CallOp call) {
    std::string fnName = getRaisedFuncName("raised_cublasSGemm_v2");
    auto module = call->getParentOfType<ModuleOp>();
    // auto origFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("cublasSgemm_v2");
    // if (!origFunc) {
    //   llvm::errs() << "replacing function not found\n";
    // }

    int i = 0;
    std::map<int, Value> constants;
    for (auto arg : call->getOperands()) {
      if (i == 0) {
        ++i;
        continue; // skip cublasHandle arg
      }
      Attribute attr;
      if (matchPattern(arg, m_Constant(&attr))) {
        // llvm::errs() << "matches constant\n";
        constants[i-1] = arg;
      }
      ++i;
    }
    
    func::FuncOp f = getCublasSGemm_v2(call, constants);

    llvm::errs() << "replacing cublasSGemm\n";

    // Before transforming, get the original C pointer operand
    // cublasSgemm args: handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    // C is at operand index 12 (0-indexed)
    Value orig_C_ptr = call.getOperand(12);

    // Trace back to find the GPU memref: C_ptr came from memref2pointer(C_gpu_memref)
    Value C_gpu_memref;
    if (auto memref2ptr = orig_C_ptr.getDefiningOp<enzymexla::Memref2PointerOp>()) {
      C_gpu_memref = memref2ptr.getSource();
      llvm::errs() << "Found C GPU memref: " << C_gpu_memref << "\n";
    } else {
      llvm::errs() << "WARNING: Could not trace C back to GPU memref\n";
    }

    SmallVector<Value> newOperands = transformOperands(call, "cublasSGemm_v2", constants);

    for (int i = newOperands.size() - 1; i >= 0; --i) {
      if (constants.count(i) != 0) {
        newOperands.erase(newOperands.begin() + i);
      }
    }

    OpBuilder builder(call);
    // Get result types from function (all same as input types for in/out params)
    SmallVector<Type> resultTypes;
    for (auto operand : newOperands) {
      resultTypes.push_back(operand.getType());
    }

    // Create the XLA wrapper operation
    auto wrapperOp = enzymexla::XLAWrapperOp::create(
      builder, call->getLoc(), TypeRange(resultTypes), SymbolRefAttr::get(f),
      llvm::to_vector(newOperands), nullptr, nullptr);

    llvm::errs() << "Created XLA wrapper for GEMM\n";
    llvm::errs() << "  Wrapper has " << wrapperOp->getNumResults() << " results\n";
    llvm::errs() << "  newOperands has " << newOperands.size() << " operands\n";

    // Copy the result from XLA GPU memory back to the original CUDA GPU buffer
    // Result #4 is the C matrix output (in XLA GPU memory, address space 1)
    if (C_gpu_memref && wrapperOp->getNumResults() >= 5) {
      builder.setInsertionPointAfter(wrapperOp);

      // Compute C size: m * n elements, then convert to bytes
      Value c_size_elements;
      if (constants.count(2) && constants.count(3)) {
        Value m = constants[2];
        Value n = constants[3];
        Value m_idx = arith::IndexCastOp::create(builder, call->getLoc(), builder.getIndexType(), m);
        Value n_idx = arith::IndexCastOp::create(builder, call->getLoc(), builder.getIndexType(), n);
        c_size_elements = builder.create<arith::MulIOp>(call->getLoc(), m_idx, n_idx);
      } else {
        c_size_elements = builder.create<arith::ConstantIndexOp>(call->getLoc(), 6);  // fallback: 2x3 = 6 elements
      }

      // Convert element count to byte size (sizeof(f32) = 4)
      Value elemSize = builder.create<arith::ConstantIndexOp>(call->getLoc(), 4);
      Value c_size_bytes = builder.create<arith::MulIOp>(call->getLoc(), c_size_elements, elemSize);

      // Copy from XLA GPU memory (address space 1) to CUDA GPU memory (address space 1)
      // This becomes direction=3 (device to device)
      Value xla_result = wrapperOp->getResult(4);  // C output in XLA GPU memory (address space 1)
      enzymexla::MemcpyOp::create(
        builder, call->getLoc(), (mlir::Type)nullptr, ValueRange(),
        C_gpu_memref,      // destination: CUDA GPU buffer (address space 1)
        xla_result,        // source: XLA GPU buffer (address space 1)
        c_size_bytes
      );

      llvm::errs() << "Added memcpy from XLA GPU result to CUDA GPU buffer\n";
    }

  }
  // TODO: Handle differing leading dimensions
  func::FuncOp getCublasSGemm_v2(LLVM::CallOp call, std::map<int, Value> constants) {
    constexpr StringRef fnName = "raised_cublasSGemm_v2";
    auto module = call->getParentOfType<ModuleOp>();

    // if (func::FuncOp existing =
    //         module.lookupSymbol<func::FuncOp>(fnName)) {
    //   llvm::errs() << "early exit\n";
    //   return existing;
    // }
    

    MLIRContext *ctx = call.getContext();
    OpBuilder builder(ctx);
    auto loc = call.getLoc();

    auto f32 = builder.getF32Type();
    auto i32 = builder.getI32Type();
    Type flatDynTensor = RankedTensorType::get({ShapedType::kDynamic}, f32);
    Type i32Tensor = RankedTensorType::get({}, i32);
    Type f32Tensor = RankedTensorType::get({}, f32);

    // Construct new input type list

    SmallVector<Type> newInputs;
    SmallVector<Type> results;

    for (int i = 0; i < 13; i++) {
      if (constants.count(i) > 0) {
        // do nothing
      } else if (i == 6 || i == 8 || i == 11) {
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
    

    auto fn = func::FuncOp::create(loc, fnName, newFuncType);
    fn.setPrivate();
    module.push_back(fn);


    // Construct replacement function

    Block *entry = fn.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());
    
    for (auto &entry : constants) {
      Value &val = entry.second;   // mutable reference
      auto unrankedTensorType = RankedTensorType::get({}, val.getType());
      Attribute attr;
      matchPattern(val, m_Constant(&attr));

      val = stablehlo::ConstantOp::create(
        bodyBuilder, val.getLoc(), unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(attr)));
    }

    // Extract arguments
    auto args = entry->getArguments();
    int idx = 0;
    int tracker = 0;
    std::map<int, Value> outputs;

    GET_NEXT_ARG(transAenum);
    GET_NEXT_ARG(transBenum);
    GET_NEXT_ARG(m);
    GET_NEXT_ARG(n);
    GET_NEXT_ARG(k);
    GET_NEXT_ARG(alpha);
    GET_NEXT_ARG(A_flat);
    GET_NEXT_ARG(lda);
    GET_NEXT_ARG(B_flat);
    GET_NEXT_ARG(ldb);
    GET_NEXT_ARG(beta);
    GET_NEXT_ARG(C_flat);
    GET_NEXT_ARG(ldc);

    // Value m = args[2];
    // Value n = args[3];
    // Value k = args[4];
    // Value alpha = args[5];
    // Value A_flat = args[6];
    // Value lda = args[7];
    // Value B_flat = args[8];
    // Value ldb = args[9];
    // Value beta = args[10];
    // Value C_flat = args[11];
    // Value ldc = args[12];
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
    SmallVector<int64_t> transposeEnums = {'T', 't', 'C', 'c'};
    bool transA = llvm::is_contained(transposeEnums, transAenum_const);
    bool transB = llvm::is_contained(transposeEnums, transBenum_const);
    // Value transA = getIsEnum(bodyBuilder, loc, transAenum, transposeEnums);
    // Value transB = getIsEnum(bodyBuilder, loc, transBenum, transposeEnums);

    Value A_eff;
    if (transA) {
      auto A = make2DTensor(bodyBuilder, loc, A_flat, lda_const, k_const, m_const);
      A_eff = stablehlo::TransposeOp::create(
        bodyBuilder, loc, RankedTensorType::get({m_const, k_const}, elemTy), A, SmallVector<int64_t>{1, 0}
      );
    } else {
      A_eff = make2DTensor(bodyBuilder, loc, A_flat, lda_const, m_const, k_const);
    }


    Value B_eff;
    if (transA) {
      auto B = make2DTensor(bodyBuilder, loc, B_flat, ldb_const, n_const, k_const);
      B_eff = stablehlo::TransposeOp::create(
        bodyBuilder, loc, RankedTensorType::get({k_const, n_const}, elemTy), B_flat, SmallVector<int64_t>{1, 0}
      );
    } else {
      B_eff = make2DTensor(bodyBuilder, loc, B_flat, ldb_const, k_const, n_const);
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
            loc, RankedTensorType::get({m_const, n_const}, elemTy), A_eff, B_eff,
            dotDimNumbers, nullptr, nullptr);
    // Value dot = builder.create<stablehlo::DotGeneralOp>(
    //     loc, resultType, A, B, dotDimNumbers,
    //     nullptr,  // precision_config
    //     nullptr); // algorithm

    // STEP 3: alpha * dot + beta * C
    // Scale dot
    Value broadcastSize = makePairFromScalars(bodyBuilder, loc, m, n);
    SmallVector<int64_t> b_dims = {0, 1};
    auto b_dimsAttr = mlir::DenseI64ArrayAttr::get(ctx, b_dims);
    auto alphaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), alpha
    );
    Value alphaBroadcast = stablehlo::BroadcastInDimOp::create(
        bodyBuilder,
        loc, dot.getType(), alphaTensor, b_dimsAttr);
    Value scaledDot = bodyBuilder.create<stablehlo::MulOp>(
      loc,
      alphaBroadcast.getType(),
      dot,
      alphaBroadcast);

    // Slice C to correct [m,n] shape
    // Value C_sliced = buildSlice(C, m, n, ldc);
    Value C_sliced = make2DTensor(bodyBuilder, loc, C_flat, ldc_const, m_const, n_const);

    // Scale C
    auto betaTensor = stablehlo::ReshapeOp::create(
      bodyBuilder, loc, RankedTensorType::get({1, 1}, f32), beta
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
    Value outFlat = make1DTensor(bodyBuilder, loc, C_flat, update, ldc_const, m_const, n_const);
    
    outputs.at(11) = outFlat;

    SmallVector<Value> result;
    for (auto &entry : outputs) {
        result.push_back(entry.second);
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


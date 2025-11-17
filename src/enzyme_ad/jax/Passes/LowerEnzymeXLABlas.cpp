//===- LowerEnzymeXLABlas.cpp - Lower BLAS operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of enzymexla.linalg.gemm operations to
// backend-specific implementations. For now, this is a placeholder pass that
// documents where the lowering would happen. The actual lowering from GEMM
// to StableHLO/device-specific code requires tensor dialect support which
// needs to be added at a higher level in the pipeline.
//
//===----------------------------------------------------------------------===//


#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "llvm/ADT/SmallVector.h"


#define DEBUG_TYPE "lower-enzymexla-blas"


namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLABLASPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir


using namespace mlir;
using namespace mlir::enzyme;


namespace {


struct GemmOpLowering : public OpRewritePattern<enzymexla::GemmOp> {
 std::string backend;


 GemmOpLowering(std::string backend, MLIRContext *context,
                PatternBenefit benefit = 1)
     : OpRewritePattern(context, benefit), backend(backend) {}


 LogicalResult matchAndRewrite(enzymexla::GemmOp op,
                               PatternRewriter &rewriter) const override {
   auto loc = op.getLoc();

   llvm::errs() << "===== GemmOpLowering::matchAndRewrite called for GemmOp =====\n";
   llvm::errs().flush();

   llvm::errs() << "GemmOpLowering::matchAndRewrite called for GemmOp\n";


   // Extract operands: A, B, alpha, beta, C
   Value A = op.getOperand(0);
   Value B = op.getOperand(1);
   Value alpha = op.getOperand(2);
   Value beta = op.getOperand(3);
   Value C = op.getOperand(4);


   llvm::errs() << "  A type: " << A.getType() << "\n";
   llvm::errs() << "  B type: " << B.getType() << "\n";
   llvm::errs() << "  C type: " << C.getType() << "\n";
   llvm::errs() << "  alpha type: " << alpha.getType() << "\n";
   llvm::errs() << "  beta type: " << beta.getType() << "\n";


   // Check if operands are memrefs
   auto A_isMemref = isa<MemRefType>(A.getType());
   auto B_isMemref = isa<MemRefType>(B.getType());
   auto C_isMemref = isa<MemRefType>(C.getType());


   llvm::errs() << "  A is memref: " << A_isMemref << "\n";
   llvm::errs() << "  B is memref: " << B_isMemref << "\n";
   llvm::errs() << "  C is memref: " << C_isMemref << "\n";


   if (!A_isMemref || !B_isMemref || !C_isMemref) {
     llvm::errs() << "  ERROR: GemmOp operands are not all memrefs - skipping lowering\n";
     return failure();
   }


   llvm::errs() << "  All operands are memrefs - proceeding with lowering\n";

   // Debug: Print actual values
   llvm::errs() << "  A: " << A << "\n";
   llvm::errs() << "  B: " << B << "\n";
   llvm::errs() << "  C: " << C << "\n";
   llvm::errs() << "  alpha: " << alpha << "\n";
   llvm::errs() << "  beta: " << beta << "\n";

   auto A_type = cast<MemRefType>(A.getType());
   auto B_type = cast<MemRefType>(B.getType());
   auto C_type = cast<MemRefType>(C.getType());


   auto elementType = A_type.getElementType();
   auto A_shape = A_type.getShape();
   auto B_shape = B_type.getShape();
   auto C_shape = C_type.getShape();


   // Extract static dimensions where available
   int m = A_shape[0] == ShapedType::kDynamic ? 0 : A_shape[0];
   int k = A_shape[1] == ShapedType::kDynamic ? 0 : A_shape[1];
   int n = B_shape[1] == ShapedType::kDynamic ? 0 : B_shape[1];


   // For dynamic dimensions, we need to query at runtime
   SmallVector<Value> dynamicDims;
   if (A_shape[0] == ShapedType::kDynamic) {
     dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, A, 0));
   }
   if (A_shape[1] == ShapedType::kDynamic) {
     dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, A, 1));
   }
   if (B_shape[1] == ShapedType::kDynamic) {
     dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, B, 1));
   }


   // Implement C := alpha * (A @ B) + beta * C using scf.for loops
   // Matrices: A[m, k], B[k, n], C[m, n]

   // Create loop bounds
   Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
   Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
   Value m_bound = (m != 0) ? rewriter.create<arith::ConstantIndexOp>(loc, m).getResult()
                            : rewriter.create<memref::DimOp>(loc, A, 0).getResult();
   Value n_bound = (n != 0) ? rewriter.create<arith::ConstantIndexOp>(loc, n).getResult()
                            : rewriter.create<memref::DimOp>(loc, B, 1).getResult();
   Value k_bound = (k != 0) ? rewriter.create<arith::ConstantIndexOp>(loc, k).getResult()
                            : rewriter.create<memref::DimOp>(loc, A, 1).getResult();

   // For now, just create a simple loop structure to test lowering works
   llvm::errs() << "  Creating outer loop...\n";

   // Create outer loop using the static factory method
   auto loopI = scf::ForOp::create(
     rewriter, loc, c0, m_bound, c1, ValueRange{},
     [&](OpBuilder &bodyBuilderI, Location locI, Value i, ValueRange iterArgsI) {
       llvm::errs() << "    Building outer loop body...\n";
       // Inner loop
       scf::ForOp::create(
         bodyBuilderI, locI, c0, n_bound, c1, ValueRange{},
         [&](OpBuilder &bodyBuilderJ, Location locJ, Value j, ValueRange iterArgsJ) {
           llvm::errs() << "      Building inner loop body...\n";
           // Create an explicit yield with no operands
           bodyBuilderJ.create<scf::YieldOp>(locJ);
         });
       // Create yield for outer loop
       bodyBuilderI.create<scf::YieldOp>(locI);
     });

   llvm::errs() << "  Done creating loops.\n";

   llvm::errs() << "  Successfully lowered GemmOp to nested loops\n";

   // Make sure insertion point is reset
   rewriter.setInsertionPointAfter(op);

   // Dump IR before erasing
   llvm::errs() << "  IR before erasing GemmOp:\n";
   op->dump();

   llvm::errs() << "  About to erase GemmOp: " << op << "\n";
   llvm::errs().flush();
   rewriter.eraseOp(op);
   llvm::errs() << "  GemmOp erased successfully\n";
   llvm::errs().flush();
   return success();
 }
};


struct LowerEnzymeXLABlasPass
   : public enzyme::impl::LowerEnzymeXLABlasPassBase<LowerEnzymeXLABlasPass> {
 using LowerEnzymeXLABlasPassBase::LowerEnzymeXLABlasPassBase;


 void runOnOperation() override {
   auto op = getOperation();


   llvm::errs() << "=== LowerEnzymeXLABlasPass running ===\n";
   llvm::errs() << "Backend: " << backend << "\n";


   // First, count how many GemmOps exist before lowering
   int64_t gemmCountBefore = 0;
   op->walk([&](enzymexla::GemmOp gemm) {
     gemmCountBefore++;
     llvm::errs() << "Found enzymexla.linalg.gemm operation BEFORE lowering\n";
   });
   llvm::errs() << "Found " << gemmCountBefore
                << " GemmOp operations BEFORE lowering\n";


   // Set up pattern rewriter
   RewritePatternSet patterns(op->getContext());
   patterns.add<GemmOpLowering>(backend, op->getContext());
   llvm::errs() << "Added GemmOpLowering pattern\n";


   // Apply patterns
   GreedyRewriteConfig config;
   llvm::errs() << "About to call applyPatternsAndFoldGreedily\n";
   llvm::errs().flush();
   LogicalResult result = applyPatternsAndFoldGreedily(op, std::move(patterns), config);
   llvm::errs() << "applyPatternsAndFoldGreedily returned: " << (succeeded(result) ? "success" : "failure") << "\n";
   llvm::errs().flush();
   if (failed(result)) {
     llvm::errs() << "Failed to apply patterns\n";
     signalPassFailure();
     return;
   }


   int64_t gemmCountAfter = 0;
   op->walk([&](enzymexla::GemmOp gemm) {
     gemmCountAfter++;
     llvm::errs() << "Found enzymexla.linalg.gemm operation AFTER lowering\n";
   });


   llvm::errs() << "Found " << gemmCountAfter
                << " GemmOp operations AFTER lowering\n";
   llvm::errs() << "=== LowerEnzymeXLABlasPass done ===\n";
 }
};


} // end anonymous namespace




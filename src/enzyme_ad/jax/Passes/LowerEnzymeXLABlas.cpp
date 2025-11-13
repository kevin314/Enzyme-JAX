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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

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
struct LowerEnzymeXLABlasPass
    : public enzyme::impl::LowerEnzymeXLABlasPassBase<LowerEnzymeXLABlasPass> {
  using LowerEnzymeXLABlasPassBase::LowerEnzymeXLABlasPassBase;

  void runOnOperation() override {
    auto op = getOperation();

    llvm::errs() << "=== LowerEnzymeXLABlasPass running ===\n";
    llvm::errs() << "Backend: " << backend << "\n";

    int64_t gemmCount = 0;
    op->walk([&](enzymexla::GemmOp gemm) {
      gemmCount++;
      llvm::errs() << "Found enzymexla.linalg.gemm operation\n";
    });

    llvm::errs() << "Found " << gemmCount << " GemmOp operations\n";
    llvm::errs() << "=== LowerEnzymeXLABlasPass done ===\n";
  }
};

} // end anonymous namespace

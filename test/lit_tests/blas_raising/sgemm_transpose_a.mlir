// RUN: enzymexlamlir-opt %s -blas-raise 2>&1 | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "p1 _ZTS13cublasContext", members = {<#tbaa_type_desc2, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %1 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %0 : !llvm.ptr
    %3 = llvm.call @cublasCreate_v2(%0) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> i32
    llvm.intr.lifetime.start %1 : !llvm.ptr
    %4 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xf32>
    affine.store %cst, %4[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf32>
    llvm.intr.lifetime.start %2 : !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xf32>
    affine.store %cst_0, %5[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf32>
    %memref = gpu.alloc  () : memref<32xf32, 1>
    %6 = "enzymexla.memref2pointer"(%memref) : (memref<32xf32, 1>) -> !llvm.ptr
    %memref_1 = gpu.alloc  () : memref<48xf32, 1>
    %7 = "enzymexla.memref2pointer"(%memref_1) : (memref<48xf32, 1>) -> !llvm.ptr
    %memref_2 = gpu.alloc  () : memref<24xf32, 1>
    %8 = "enzymexla.memref2pointer"(%memref_2) : (memref<24xf32, 1>) -> !llvm.ptr
    %9 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %10 = affine.load %9[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?x!llvm.ptr>
    %11 = llvm.call @cublasSgemm_v2(%10, %c1_i32, %c0_i32, %c2_i32, %c3_i32, %c4_i32, %1, %6, %c2_i32, %7, %c4_i32, %2, %8, %c2_i32) {no_unwind} : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %12 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %13 = affine.load %12[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?x!llvm.ptr>
    %14 = llvm.call @cublasDestroy_v2(%13) {no_unwind} : (!llvm.ptr {llvm.noundef}) -> i32
    %15 = llvm.call @cudaFree(%6) {no_unwind} : (!llvm.ptr {llvm.noundef}) -> i32
    %16 = llvm.call @cudaFree(%7) {no_unwind} : (!llvm.ptr {llvm.noundef}) -> i32
    %17 = llvm.call @cudaFree(%8) {no_unwind} : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.intr.lifetime.end %2 : !llvm.ptr
    llvm.intr.lifetime.end %1 : !llvm.ptr
    llvm.intr.lifetime.end %0 : !llvm.ptr
    llvm.return %c0_i32 : i32
  }
  llvm.func local_unnamed_addr @cublasCreate_v2(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cublasSgemm_v2(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cublasDestroy_v2(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaFree(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK: enzymexla.xla_wrapper
// CHECK: func.func private
// CHECK-SAME: tensor<f32>, tensor<?xf32>, tensor<?xf32>, tensor<f32>, tensor<?xf32>
// CHECK: stablehlo.slice
// CHECK: stablehlo.reshape {{.*}} tensor<2x4xf32>
// CHECK: stablehlo.transpose {{.*}} dims = [1, 0]
// CHECK: stablehlo.slice
// CHECK: stablehlo.reshape {{.*}} tensor<3x4xf32>
// CHECK: stablehlo.dot_general

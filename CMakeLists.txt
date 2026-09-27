cmake_minimum_required(VERSION 3.16)
project(rknpu2_matmul_open LANGUAGES C CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenMP REQUIRED)
include(CheckCXXCompilerFlag)

set(RK_NPU_AARCH64_FP16_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  check_cxx_compiler_flag("-march=armv8.2-a+fp16" RK_NPU_HAS_ARMV82_FP16_FLAG)
  if(RK_NPU_HAS_ARMV82_FP16_FLAG)
    set(RK_NPU_AARCH64_FP16_FLAGS "-march=armv8.2-a+fp16")
  endif()
endif()

# ---- module sources (one translation unit per datapath + shared core) ----
set(RK_NPU_SRCS
  src/rk_npu_core.cpp          # device / memory / submit / PC-chain (shared)
  src/rk_npu_cpu_kernels.cpp   # CPU-side layout pack/unpack kernels
  src/rk_npu_dcomp.cpp         # standalone lossless normal codec + basic RTN
  src/rk_npu_kn_plan.cpp       # persistent split-K / multi-core-N plan
  src/rk_npu_autotune_common.cpp # shared resources, timing and ranking
  src/rk_npu_autotune_cache.cpp  # shared cache paths and atomic I/O
  src/rk_npu_quant_matmul.cpp  # typed single-GEMM autotune/execution API
  src/rk_npu_matmul_i8.cpp     # int8 x int8 -> fp32 matmul
  src/rk_npu_moe_w8.cpp        # routed W8 SwiGLU, shared weights/workspace
  src/rk_npu_i4_cpu.cpp        # INT4 packing + INT16 partial reduction
  src/rk_npu_i4_bounds.cpp     # weight-only INT16 bounds and safe K selection
  src/rk_npu_i4_regs.cpp       # native W4A4 register body only
  src/rk_npu_i4_plan.cpp       # persistent K/M/N tiles + shared packed B
  src/rk_npu_i4_executor.cpp   # optional CPU/NPU K-wave pipeline
  src/rk_npu_matmul_i4.cpp     # integer-only C ABI + input producer boundary
  src/rk_npu_flatquant_regs.cpp # FP16 GEMM with fold2 transpose writeback
  src/rk_npu_flatquant_cpu.cpp # initial pack and native-output scan/INT4 packing
  src/rk_npu_flatquant.cpp     # reusable transform weights and double-buffer pipeline
  src/rk_npu_w4a4_linear_cpu.cpp # packed INT4 input producer
  src/rk_npu_w4a4_output.cpp   # native INT32 accumulation and cache-line output
  src/rk_npu_w4a4_linear.cpp   # fused FP32/FP16 W4A4 Linear C ABI
  src/rk_npu_w4a8_quant.cpp    # full-token A8 scale + fused MSD/native packing
  src/rk_npu_w4a8_output.cpp   # partial MSD reconstruction + floating output
  src/rk_npu_w4a8_npu_reduce.cpp # optional retained-partial INT32 Conv consumer
  src/rk_npu_w4a8.cpp          # shared scaled W4 weights + f32/f16 entry points
  src/rk_npu_w4a8_tune_policy.cpp # weight-aware candidate and cache identity policy
  src/rk_npu_w4a8_tune_data.cpp   # deterministic calibration and independent oracle
  src/rk_npu_w4a8_tune_cache.cpp  # W4 strategy serialization
  src/rk_npu_w4a8_tune.cpp        # W4 complete-call tuning
  src/rk_npu_w4a8_tune_family.cpp # shared K selection across M values
  src/rk_npu_matmul_f16.cpp    # fp16 x fp16 -> fp16 matmul + fused MUL/ADD
  src/rk_npu_add_rmsnorm_f16.cpp)

set(RK_NPU_PUBLIC_HEADERS
  include/rk_npu_common.h
  include/rk_npu_matmul.h
  include/rk_npu_moe_w8.h
  include/rk_npu_matmul_i4.h
  include/rk_npu_w4a4_linear.h
  include/rk_npu_w4a8.h
  include/rk_npu_w4a8_tune.h
  include/rk_npu_quant_matmul.h
  include/rk_npu_matmul_f16.h
  include/rk_npu_add_rmsnorm_f16.h)

# ---- the C-ABI shared library (callable like librknnrt.so) ----
add_library(rk_npu_matmul SHARED ${RK_NPU_SRCS})
target_include_directories(rk_npu_matmul PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(rk_npu_matmul PROPERTIES
  PUBLIC_HEADER "${RK_NPU_PUBLIC_HEADERS}")
target_compile_options(rk_npu_matmul PRIVATE ${RK_NPU_AARCH64_FP16_FLAGS})
target_link_libraries(rk_npu_matmul PUBLIC OpenMP::OpenMP_CXX)

# also a static lib for easy embedding
add_library(rk_npu_matmul_static STATIC ${RK_NPU_SRCS})
target_include_directories(rk_npu_matmul_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(rk_npu_matmul_static PROPERTIES OUTPUT_NAME rk_npu_matmul)
target_compile_options(rk_npu_matmul_static PRIVATE ${RK_NPU_AARCH64_FP16_FLAGS})
target_link_libraries(rk_npu_matmul_static PUBLIC OpenMP::OpenMP_CXX)

# ---- demo executables ----
add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE rk_npu_matmul)

add_executable(bench examples/bench.cpp)
target_link_libraries(bench PRIVATE rk_npu_matmul)

add_executable(bench_llm_i8 examples/bench_llm_i8.cpp)
target_link_libraries(bench_llm_i8 PRIVATE rk_npu_matmul)

add_executable(bench_i4 examples/bench_i4.cpp)
target_link_libraries(bench_i4 PRIVATE rk_npu_matmul)

add_executable(bench_w4a8 examples/bench_w4a8.cpp)
target_link_libraries(bench_w4a8 PRIVATE rk_npu_matmul)
add_executable(bench_w4a4_flatquant examples/bench_w4a4_flatquant.cpp)
target_link_libraries(bench_w4a4_flatquant PRIVATE rk_npu_matmul)
add_executable(tune_w4a8 examples/tune_w4a8.cpp)
target_link_libraries(tune_w4a8 PRIVATE rk_npu_matmul)

add_executable(demo_f16 examples/demo_f16.cpp)
target_link_libraries(demo_f16 PRIVATE rk_npu_matmul)

# ---- test framework (int8 correctness + performance sweeps) ----
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE rk_npu_matmul)

add_executable(test_perf tests/test_perf.cpp)
target_link_libraries(test_perf PRIVATE rk_npu_matmul)

# Optional: build test_perf --official against the vendor rknn_matmul runtime.
# Board example:
#   cmake -B build -DRK_NPU_OFFICIAL=ON \
#     -DRKNN_INCLUDE_DIR=/path/to/runtime/Linux/librknn_api/include \
#     -DRKNN_LIB=/usr/lib/librknnrt.so
option(RK_NPU_OFFICIAL "Bench test_perf against the vendor librknnrt rknn_matmul" OFF)
set(RKNN_INCLUDE_DIR "" CACHE PATH   "Directory containing rknn_matmul_api.h / rknn_api.h")
set(RKNN_LIB         "rknnrt" CACHE STRING "Vendor runtime lib (name or full path to librknnrt.so)")
if(RK_NPU_OFFICIAL)
  target_compile_definitions(test_perf PRIVATE RK_WITH_OFFICIAL)
  if(RKNN_INCLUDE_DIR)
    target_include_directories(test_perf PRIVATE ${RKNN_INCLUDE_DIR})
  endif()
  target_link_libraries(test_perf PRIVATE ${RKNN_LIB})
endif()

add_executable(test_mem_api tests/test_mem_api.cpp)
target_link_libraries(test_mem_api PRIVATE rk_npu_matmul)

add_executable(test_batch tests/test_batch.cpp)
target_link_libraries(test_batch PRIVATE rk_npu_matmul)

add_executable(test_batch_f16 tests/test_batch_f16.cpp)
target_link_libraries(test_batch_f16 PRIVATE rk_npu_matmul)

add_executable(bench_f16 tests/bench_f16.cpp)
target_link_libraries(bench_f16 PRIVATE rk_npu_matmul)

add_executable(test_add_rmsnorm_f16 tests/test_add_rmsnorm_f16.cpp)
target_link_libraries(test_add_rmsnorm_f16 PRIVATE rk_npu_matmul)

add_executable(test_cpu_kernels tests/test_cpu_kernels.cpp)
target_link_libraries(test_cpu_kernels PRIVATE rk_npu_matmul)
if(BUILD_TESTING)
  add_test(NAME cpu_kernels COMMAND test_cpu_kernels)
endif()

add_executable(test_quant_matmul tests/test_quant_matmul.cpp)
target_link_libraries(test_quant_matmul PRIVATE rk_npu_matmul)

add_executable(test_moe_w8 tests/test_moe_w8.cpp)
target_link_libraries(test_moe_w8 PRIVATE rk_npu_matmul)
add_library(test_moe_w8_header_c OBJECT tests/test_moe_w8_header.c)
target_include_directories(test_moe_w8_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_i4_cpu tests/test_i4_cpu.cpp)
target_link_libraries(test_i4_cpu PRIVATE rk_npu_matmul)
add_executable(test_i4 tests/test_i4.cpp)
target_link_libraries(test_i4 PRIVATE rk_npu_matmul)
add_library(test_i4_header_c OBJECT tests/test_i4_header.c)
target_include_directories(test_i4_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(test_w4a4_linear_cpu tests/test_w4a4_linear_cpu.cpp)
target_link_libraries(test_w4a4_linear_cpu PRIVATE rk_npu_matmul)
add_executable(test_w4a4_linear tests/test_w4a4_linear.cpp)
target_link_libraries(test_w4a4_linear PRIVATE rk_npu_matmul)
add_library(test_w4a4_linear_header_c OBJECT tests/test_w4a4_linear_header.c)
target_include_directories(test_w4a4_linear_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(BUILD_TESTING)
  add_test(NAME i4_cpu COMMAND test_i4_cpu)
  add_test(NAME w4a4_linear_cpu COMMAND test_w4a4_linear_cpu)
endif()

add_executable(test_w4a8_cpu tests/test_w4a8_cpu.cpp)
target_link_libraries(test_w4a8_cpu PRIVATE rk_npu_matmul)
add_executable(test_w4a8_reduction tests/test_w4a8_reduction.cpp)
target_link_libraries(test_w4a8_reduction PRIVATE rk_npu_matmul)
add_executable(test_w4a8 tests/test_w4a8.cpp)
target_link_libraries(test_w4a8 PRIVATE rk_npu_matmul)
add_executable(test_i4_bounds tests/test_i4_bounds.cpp)
target_link_libraries(test_i4_bounds PRIVATE rk_npu_matmul)
add_executable(test_w4a8_multiplier tests/test_w4a8_multiplier.cpp)
target_link_libraries(test_w4a8_multiplier PRIVATE rk_npu_matmul)
add_executable(test_autotune_cpu tests/test_autotune_cpu.cpp)
target_link_libraries(test_autotune_cpu PRIVATE rk_npu_matmul)
add_executable(test_w4a8_tune tests/test_w4a8_tune.cpp)
target_link_libraries(test_w4a8_tune PRIVATE rk_npu_matmul)
add_library(test_w4a8_header_c OBJECT tests/test_w4a8_header.c)
target_include_directories(test_w4a8_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(BUILD_TESTING)
  add_test(NAME w4a8_cpu COMMAND test_w4a8_cpu)
  add_test(NAME w4a8_reduction COMMAND test_w4a8_reduction)
  add_test(NAME i4_bounds COMMAND test_i4_bounds)
  add_test(NAME autotune_cpu COMMAND test_autotune_cpu)
endif()

add_executable(test_panel_layout tests/test_panel_layout.cpp)
target_link_libraries(test_panel_layout PRIVATE rk_npu_matmul)
if(BUILD_TESTING)
  add_test(NAME panel_layout COMMAND test_panel_layout)
endif()

add_executable(test_dcomp_codec tests/test_dcomp_codec.cpp)
target_link_libraries(test_dcomp_codec PRIVATE rk_npu_matmul)
add_executable(test_compress tests/test_compress.cpp)
target_link_libraries(test_compress PRIVATE rk_npu_matmul)
if(BUILD_TESTING)
  add_test(NAME dcomp_codec COMMAND test_dcomp_codec)
endif()

# Compile the public quantized API as C on every build.  The implementation is
# C++, but callers must not need a C++ compiler to include its C ABI header.
add_library(test_quant_header_c OBJECT tests/test_quant_header.c)
target_include_directories(test_quant_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Keep the raw FP16 + split-K/dmabuf public surface valid for C callers too.
add_library(test_f16_header_c OBJECT tests/test_f16_header.c)
target_include_directories(test_f16_header_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(probe_n_tiling tests/probe_n_tiling.cpp)
target_link_libraries(probe_n_tiling PRIVATE rk_npu_matmul)

add_executable(tune_quant_matmul tests/tune_quant_matmul.cpp)
target_link_libraries(tune_quant_matmul PRIVATE rk_npu_matmul)

add_executable(probe_ac_layout tests/probe_ac_layout.cpp)
target_link_libraries(probe_ac_layout PRIVATE rk_npu_matmul)

include(GNUInstallDirs)
install(TARGETS rk_npu_matmul
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(test_w4a8_low_cpu tests/test_w4a8_low_cpu.cpp)
target_link_libraries(test_w4a8_low_cpu PRIVATE rk_npu_matmul ${CMAKE_DL_LIBS})
if(BUILD_TESTING)
  add_test(NAME w4a8_low_cpu_host COMMAND test_w4a8_low_cpu --host)
endif()

add_executable(test_w4a8_tune_layout_cpu tests/test_w4a8_tune_layout_cpu.cpp)
target_link_libraries(test_w4a8_tune_layout_cpu PRIVATE rk_npu_matmul)
if(BUILD_TESTING)
  add_test(NAME w4a8_tune_layout_cpu COMMAND test_w4a8_tune_layout_cpu)
endif()

add_executable(test_w4a8_tune_layout tests/test_w4a8_tune_layout.cpp)
target_link_libraries(test_w4a8_tune_layout PRIVATE rk_npu_matmul)

# CPU-only validation, including public C headers. Some hardware-only FP16
# programs use __fp16 and therefore cannot be built by an x86 compiler.
add_custom_target(rknpu2_matmul_open_cpu_tests DEPENDS
  test_cpu_kernels test_i4_cpu test_w4a4_linear_cpu test_w4a8_cpu
  test_w4a8_reduction test_i4_bounds test_autotune_cpu test_panel_layout
  test_dcomp_codec test_w4a8_low_cpu test_w4a8_tune_layout_cpu
  test_quant_header_c test_f16_header_c test_i4_header_c test_w4a8_header_c
  test_w4a4_linear_header_c test_moe_w8_header_c)

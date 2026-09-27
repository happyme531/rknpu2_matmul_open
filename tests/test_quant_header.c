#include "rk_npu_quant_matmul.h"

/* This translation unit intentionally has no main().  Its only job is to make
 * the regular CMake build reject accidental C++ syntax in the public C ABI. */
void rk_npu_quant_header_c_compile_test(void) {
    rk_npu_matmul_autotune_config config;
    int m_values[1] = {1};
    rk_npu_matmul_family_autotune_config family_config;
    rk_npu_matmul_family_summary family_summary;
    rk_npu_matmul_strategy fastest;
    rk_npu_matmul_workspace_requirements memory;
    rk_npu_matmul_weight_requirements weight_memory;
    rk_npu_matmul_weight_config weight = {32, 32, 32};
    rk_npu_matmul_autotune_config_init(&config, 1, 32, 32);
    rk_npu_matmul_family_autotune_config_init(
        &family_config, 32, 32, 1, m_values);
    fastest.a_layout = RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    (void)rk_npu_i8i8i32_autotune((rk_npu_ctx*)0, &config, &fastest, 0);
    (void)rk_npu_matmul_autotune_cached(
        (rk_npu_ctx*)0, &config, RK_NPU_MATMUL_I8I8I32, 0, 0,
        &fastest, 0, 0);
    (void)rk_npu_matmul_autotune_cache_load(
        (rk_npu_ctx*)0, "cache.tune", RK_NPU_MATMUL_I8I8I32,
        &config, &fastest, &fastest);
    (void)rk_npu_matmul_autotune_family_cached(
        (rk_npu_ctx*)0, &family_config, RK_NPU_MATMUL_I8I8I32,
        0, 0, &family_summary, &fastest, 0, 0, 0, 0);
    (void)rk_npu_matmul_workspace_memory_query(&fastest, &memory);
    (void)rk_npu_i8i8i32_weights_memory_query(&weight, &weight_memory);
    (void)rk_npu_f32i8f32_dynamic_autotune(
        (rk_npu_ctx*)0, &config, &fastest, 0);
    (void)rk_npu_f32i8f32_weights_memory_query(&weight, &weight_memory);
    (void)rk_npu_f32i8f32_run_dynamic(
        (rk_npu_matmul_workspace*)0, (rk_npu_f32i8f32_weights*)0,
        (const float*)0, (float*)0);
}

/* Keep both constructor families visible and correctly typed to C callers. */
static void compressed_api_signatures(rk_npu_iommu_domain* d,
    const rk_npu_matmul_weight_config* cfg, const int8_t* b,
    const float* scales, const float* w) {
    rk_npu_i8i8i32_weights_free(rk_npu_i8i8i32_weights_create_compress(d,cfg,b));
    rk_npu_f16i8f16_weights_free(rk_npu_f16i8f16_weights_create_compress(d,cfg,b,scales));
    rk_npu_f32i8f32_weights_free(rk_npu_f32i8f32_weights_create_compress(d,cfg,b,scales));
    rk_npu_f16i8f16_weights_free(rk_npu_f16i8f16_weights_create_from_f32_compress(d,cfg,w,6.5f));
    rk_npu_f32i8f32_weights_free(rk_npu_f32i8f32_weights_create_from_f32_compress(d,cfg,w,6.5f));
}

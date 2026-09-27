#include "rk_npu_matmul_f16.h"
#include "rk_npu_add_rmsnorm_f16.h"

void rk_npu_test_f16_header_c(rk_npu_iommu_domain* domain,
                              rk_npu_mem* input,
                              rk_npu_mem* weight,
                              rk_npu_mem* output) {
    rk_npu_matmul_f16_config cfg;
    rk_npu_matmul_sizes sizes;
    rk_npu_matmul_f16_config_init(&cfg, 128, 1280, 5120,
                                  RK_NPU_FUSE_NONE);
    cfg.core_mask = 7;
    (void)rk_npu_matmul_f16_query(&cfg, &sizes);
    (void)rk_npu_matmul_f16_splitk_query(3, &cfg, &sizes);
    rk_npu_matmul_f16_splitk_plan* plan =
        rk_npu_matmul_f16_splitk_prepare(domain, 3, &cfg);
    (void)input;
    (void)weight;
    (void)output;
    rk_npu_matmul_f16_splitk_plan_free(plan);
}

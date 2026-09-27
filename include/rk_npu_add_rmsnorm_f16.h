/* Direct-register FP16 native Add + RMSNorm for validated RK3588 LLM shapes. */
#ifndef RK_NPU_ADD_RMSNORM_F16_H
#define RK_NPU_ADD_RMSNORM_F16_H

#include "rk_npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int M;
    int D;
    float eps;
    uint32_t timeout_ms;
} rk_npu_add_rmsnorm_f16_config;

typedef struct {
    uint64_t tensor_bytes;
    uint64_t internal_bytes;
    uint64_t weight_bytes;
    uint64_t regcmd_bytes;
    uint64_t task_bytes;
    int num_tasks;
} rk_npu_add_rmsnorm_f16_sizes;

typedef struct rk_npu_add_rmsnorm_f16_plan rk_npu_add_rmsnorm_f16_plan;

/* The fast templates are currently validated only for M=1/128, D=4096 and
 * eps=1e-5.  Every tensor uses physical layout `(D/8, M, 8)` FP16 bits. */
void rk_npu_add_rmsnorm_f16_config_init(
    rk_npu_add_rmsnorm_f16_config* cfg, int M, int D);
int rk_npu_add_rmsnorm_f16_query(
    const rk_npu_add_rmsnorm_f16_config* cfg,
    rk_npu_add_rmsnorm_f16_sizes* out);

/* gamma[D] is copied into immutable plan storage as FP16 bit patterns. */
rk_npu_add_rmsnorm_f16_plan* rk_npu_add_rmsnorm_f16_plan_create(
    rk_npu_iommu_domain* domain,
    const rk_npu_add_rmsnorm_f16_config* cfg,
    const uint16_t* gamma);

/* All four buffers are caller-owned native block-8 tensors in the plan's
 * IOMMU domain. Outputs must not overlap either input or each other. */
int rk_npu_add_rmsnorm_f16_run(
    rk_npu_ctx* ctx, rk_npu_add_rmsnorm_f16_plan* plan,
    const rk_npu_mem* x_native,
    const rk_npu_mem* residual_native,
    rk_npu_mem* residual_out_native,
    rk_npu_mem* norm_out_native);

void rk_npu_add_rmsnorm_f16_plan_free(
    rk_npu_add_rmsnorm_f16_plan* plan);

#ifdef __cplusplus
}
#endif
#endif

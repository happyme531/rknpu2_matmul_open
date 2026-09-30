#ifndef RK_NPU_FLOAT_BACKEND_H
#define RK_NPU_FLOAT_BACKEND_H

// Private shared floating execution backend. Geometry and bit-preserving
// packing use the existing FP16 config; precision is fixed at preparation.
#include "rk_npu_matmul_f16.h"

namespace rknpu2_matmul_open::detail {
enum class FloatPrecision : uint32_t { F16 = 2, BF16 = 3, TF32 = 7 };
struct FloatBatchPlan;

int float_batch_query(int batch_count, const rk_npu_matmul_f16_config* cfg,
    FloatPrecision precision, rk_npu_matmul_sizes* out);
int float_batch_prepare(rk_npu_iommu_domain* domain, int batch_count,
    const rk_npu_matmul_f16_config* cfg, FloatPrecision precision, FloatBatchPlan** out);
int float_batch_run(rk_npu_ctx* ctx, FloatBatchPlan* plan,
    rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output);
void float_batch_free(FloatBatchPlan* plan);
}
#endif

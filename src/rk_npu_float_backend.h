#ifndef RK_NPU_FLOAT_BACKEND_H
#define RK_NPU_FLOAT_BACKEND_H

// Private shared floating execution backend. Geometry and bit-preserving
// packing use the existing FP16 config; precision is fixed at preparation.
#include "rk_npu_matmul_f16.h"
#include <vector>

struct rk_npu_mm_packed_b;
namespace rknpu2_matmul_open::detail {
int float_packed_b_view(const rk_npu_mm_packed_b*, rk_npu_iommu_domain*,
    int N, int K, const rk_npu_mem**);
enum class FloatPrecision : uint32_t { F16 = 2, BF16 = 3, TF32 = 7 };
/* Bound native A/B/output register body, no allocation or submit. Dimensions
 * must be positive multiples of 16 and supported by the floating backend. */
void float_native_regs(std::vector<uint64_t>& regs, int m, int n, int k,
    uint64_t a, uint64_t b, uint64_t output, FloatPrecision precision);
struct FloatBatchPlan;

int float_batch_query(int batch_count, const rk_npu_matmul_f16_config* cfg,
    FloatPrecision precision, rk_npu_matmul_sizes* out, int task_m_tile = 0);
int float_batch_prepare(rk_npu_iommu_domain* domain, int batch_count,
    const rk_npu_matmul_f16_config* cfg, FloatPrecision precision, FloatBatchPlan** out,
    int task_m_tile = 0);
int float_batch_run(rk_npu_ctx* ctx, FloatBatchPlan* plan,
    rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output);
void float_batch_free(FloatBatchPlan* plan);
}
#endif

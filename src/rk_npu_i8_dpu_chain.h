#pragma once
#include "rk_npu_internal.h"

namespace rknpu2_matmul_open::detail {
struct I8KnPlanConfig;
class I8KnWeights;

// SDK-style offline INT32 EW reduction. All K waves of an N shard stay on
// one core, followed optionally by the MoE FP16-coefficient dequant recipe.
class I8DpuChain {
public:
    ~I8DpuChain();
    int prepare(rk_npu_iommu_domain* domain, const I8KnPlanConfig& config,
                const std::vector<rk_npu_matmul_i8_plan*>& waves,
                const std::vector<rk_npu_mem>& outputs);
    int run(const I8KnWeights& weights, const float* a_scale,
            const float* w_scale, bool dequant);
    int copy_dequant(void* output, bool fp16);
    // Extra command/data storage, including the copied GEMM bodies.
    static void memory(const I8KnPlanConfig& config, uint64_t gemm_commands,
                        uint64_t gemm_tasks, uint64_t& control, uint64_t& data);
private:
    struct WeightPatch { int index, wave; uint64_t offset; };
    struct RowPatch { int index, row; };
    rk_npu_iommu_domain* domain_ = nullptr;
    rk_npu_mem commands_{}, tasks_{};
    DomainDataBuffer coefficients_, result_;
    std::vector<WeightPatch> weights_;
    std::vector<RowPatch> rows_;
    // Compare scale contents, never object addresses: callers can reuse a
    // workspace with different weights or mutate internal probe inputs.
    std::vector<float> coefficient_scales_;
    std::vector<uint32_t> row_scales_;
    bool coefficients_valid_ = false;
    int M_ = 0, N_ = 0, padded_n_ = 0, task_count_ = 0;
    uint32_t core_mask_ = 0, timeout_ms_ = 0;
    uint32_t starts_[3]{}, counts_[3]{};
};
}

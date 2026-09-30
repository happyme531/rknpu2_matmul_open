#ifndef RK_NPU_KN_PLAN_H
#define RK_NPU_KN_PLAN_H

#include "rk_npu_internal.h"
#include "rk_npu_matmul.h"
#include "rk_npu_i8_dpu_chain.h"

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

namespace rknpu2_matmul_open::detail {

struct I8KnPlanConfig {
    int M = 0;
    int N = 0;
    int K = 0;
    int k_tile = 0;
    int n_tile = 0;
    uint32_t npu_core_mask = 1;
    uint32_t timeout_ms = 500;
    rk_npu_matmul_i8_a_layout a_layout = RK_NPU_I8_A_LAYOUT_NORMAL;
    rk_npu_matmul_i8_c_layout c_layout = RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;
    bool npu_reduce = false;
    bool npu_dequant = false; // explicit FP16 coefficient precision opt-in
};

struct I8KnWeightConfig {
    int K = 0;
    int N = 0;
    int k_tile = 0;
};

struct I8KnMemoryInfo {
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t control_bytes = 0;
    uint32_t wave_count = 0;
    uint32_t workspace_buffer_count = 0;
    uint32_t weight_buffer_count = 0;
};

int query_i8_kn_memory(const I8KnPlanConfig& config, I8KnMemoryInfo* out);
int query_i8_kn_weight_memory(const I8KnWeightConfig& config,
                              I8KnMemoryInfo* out);

class I8KnWeights;

/* Persistent K-wave x N-shard execution plan used by the autotuner and future
 * production integration.  It owns only shared execution resources: packed-A
 * and partial-C arenas, register plans, and activation scales. Matrix-specific
 * packed B lives in I8KnWeights and is selected for each run. */
class I8KnPlan {
public:
    I8KnPlan() = default;
    ~I8KnPlan();
    I8KnPlan(const I8KnPlan&) = delete;
    I8KnPlan& operator=(const I8KnPlan&) = delete;

    int prepare(rk_npu_iommu_domain* domain, const I8KnPlanConfig& config);
    int prepare_f16_dynamic(const uint16_t* A_fp16_rowmajor);
    int prepare_and_pack_wave0_f16_dynamic(
        const uint16_t* A_fp16_rowmajor);
    int pack_wave_i8(int wave, const int8_t* A_rowmajor);
    int pack_wave_f16_dynamic(int wave,
                              const uint16_t* A_fp16_rowmajor);
    int pack_wave_f16_static(int wave,
                             const uint16_t* A_fp16_rowmajor,
                             const float* per_token_scale);
    int prepare_f32_dynamic(const float* A_fp32_rowmajor);
    int prepare_and_pack_wave0_f32_dynamic(const float* A_fp32_rowmajor);
    int pack_wave_f32_dynamic(int wave, const float* A_fp32_rowmajor);
    int pack_wave_f32_static(int wave, const float* A_fp32_rowmajor,
                             const float* per_token_scale);
    // Device chains retain every A slice, so dynamic quantization can pack
    // them together in one OpenMP region without changing any slice layout.
    int pack_chain_dynamic(const uint16_t* A_fp16_rowmajor);
    int pack_chain_dynamic(const float* A_fp32_rowmajor);
    int run_wave(int wave, const I8KnWeights& weights);
    int run_chain(const I8KnWeights& weights, const float* a_scale,
                   const float* w_scale, bool dequant);
    int copy_dequant(void* output, bool fp16);
    int begin_wave_output_cpu_read(int wave);
    int begin_wave_output_cpu_readwrite(int wave);
    int end_wave_output_cpu_access(int wave);
    int unpack_partial_i32(int wave, std::vector<int32_t>& dst) const;
    int release();

    int wave_count() const { return (int)waves_.size(); }
    int k0(int wave) const;
    int kt(int wave) const;
    const int32_t* partial_i32(int wave) const;
    const float* a_scale() const;
    const I8KnPlanConfig& config() const { return config_; }
    int a_panel_width() const {
        return config_.a_layout == RK_NPU_I8_A_LAYOUT_PANEL8 ? 8 :
               config_.a_layout == RK_NPU_I8_A_LAYOUT_PANEL16 ? 16 : 0;
    }
    int c_panel_width() const {
        return config_.c_layout == RK_NPU_I8_C_LAYOUT_PANEL8 ? 8 :
               config_.c_layout == RK_NPU_I8_C_LAYOUT_PANEL16 ? 16 : 0;
    }

private:
    friend class I8KnWeights;

    struct Wave {
        int k0 = 0;
        int kt = 0;
        rk_npu_matmul_i8_config mm_cfg{};
        rk_npu_matmul_sizes sizes{};
        rk_npu_matmul_i8_plan* plan = nullptr;
        rk_npu_mem input{};
        rk_npu_mem output{};
    };

    int input_slot_for_wave(int wave) const;
    int output_slot_for_wave(int wave) const;
    int pack_wave_i8_impl(Wave& wave, const int8_t* A_rowmajor);
    template<class T> int pack_chain_dynamic_impl(const T* A_rowmajor);

    rk_npu_ctx* ctx_ = nullptr;
    rk_npu_iommu_domain* domain_ = nullptr;
    I8KnPlanConfig config_{};
    std::vector<Wave> waves_;
    /* Two A slots let the CPU pack wave i+1 while the NPU reads wave i. */
    // The CPU pipeline uses two slots; a device chain retains all K inputs.
    std::vector<DomainDataBuffer> input_slot_;
    int input_slot_count_ = 0;
    /* ACC plus two ping-pong partials.  Separate dma-bufs are required so a
     * CPU access interval on one partial can overlap an NPU write to another. */
    DomainDataBuffer output_slot_[3];
    int output_slot_count_ = 0;
    std::vector<float> a_scale_;
    std::vector<uint8_t> wave_packed_;
    std::vector<int> slice_k0_, slice_k_, slice_align_in_;
    std::vector<int8_t*> slice_dst_;
    std::unique_ptr<I8DpuChain> chain_;
};

/* One immutable packed-weight arena. Compatibility depends only on K/N and the
 * K-wave partition, so one object can serve execution plans with different M,
 * N tiles, and CPU/NPU masks. */
class I8KnWeights {
public:
    I8KnWeights() = default;
    ~I8KnWeights();
    I8KnWeights(const I8KnWeights&) = delete;
    I8KnWeights& operator=(const I8KnWeights&) = delete;

    int prepare(rk_npu_iommu_domain* domain, const I8KnWeightConfig& config,
                const int8_t* B_rowmajor);
    int prepare_compress(rk_npu_iommu_domain* domain, const I8KnWeightConfig& config,
                         const int8_t* B_rowmajor);
    bool compressed() const { return compressed_; }
    int run_compressed(int wave, int n_tile, rk_npu_matmul_i8_plan* plan,
                        uint32_t timeout_ms) const;
    uint64_t stored_bytes() const;
    int release();
    bool same_domain(const I8KnPlan& workspace) const;
    bool compatible(const I8KnPlan& workspace) const;
    bool compatible_wave(const I8KnPlan& workspace, int wave) const;
    const rk_npu_mem* wave(int index) const;

private:
    struct CompressedLayout {
        int n_tile = 0;
        bool used = false;
        DomainDataBuffer arena;
        std::vector<std::vector<I8CompressedTile>> waves;
        ~CompressedLayout() {
            if (arena.domain) arena.release(arena.domain->ctx);
        }
    };
    int build_compressed_layout(int n_tile, const int8_t* B,
                                 const CompressedLayout* source,
                                 std::shared_ptr<CompressedLayout>& result) const;
    bool compressed_ = false;
    mutable std::mutex compressed_mutex_;
    mutable std::map<int, std::shared_ptr<CompressedLayout>> compressed_layouts_;
    rk_npu_ctx* ctx_ = nullptr;
    rk_npu_iommu_domain* domain_ = nullptr;
    I8KnWeightConfig config_{};
    DomainDataBuffer arena_;
    std::vector<rk_npu_mem> waves_;
};

} /* namespace rknpu2_matmul_open::detail */

#endif /* RK_NPU_KN_PLAN_H */

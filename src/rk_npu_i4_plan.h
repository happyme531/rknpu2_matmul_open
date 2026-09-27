#ifndef RK_NPU_I4_PLAN_H
#define RK_NPU_I4_PLAN_H
#include "rk_npu_internal.h"
#include "rk_npu_matmul_i4.h"
#include "rk_npu_i4_bounds.h"
#include <vector>
#include "rk_npu_i4_layout.h"

namespace rknpu2_matmul_open::detail {
struct I4Tile {
    int m0, m, n0, n, logical_n;
    uint64_t output_offset;
};
struct I4Wave {
    int k0 = 0, k = 0, padded_k = 0;
    uint64_t weight_offset = 0;
    rk_npu_mem regcmd{}, tasks{};
};

int query_i4(const rk_npu_i4_config& cfg, rk_npu_i4_memory_info& out,
             I4InputLayout layout = I4InputLayout::Native);
std::vector<I4Tile> i4_tiles(const rk_npu_i4_config&, uint32_t* starts, uint32_t* counts,
                             I4InputLayout layout = I4InputLayout::Native);
int query_i4_weights(const rk_npu_i4_weight_config& cfg, uint64_t& bytes);

class I4Weights {
public:
    ~I4Weights();
    /* allow_unproven is only for the explicit W4A8 multiplier experiment. */
    int prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_weight_config& cfg, const int8_t* B,
                bool allow_unproven = false);
    const rk_npu_mem& memory() const { return arena_.mem; }
    const rk_npu_i4_weight_config& config() const { return cfg_; }
private:
    rk_npu_i4_weight_config cfg_{};
    DomainDataBuffer arena_;
};

class I4Plan;
/* Called with synchronized native INT16 partials on the execution thread.
 * The plan owns begin/end CPU access; consumers must not retain the pointer. */
using I4PartialConsumer = int (*)(void*, const I4Plan&, int wave, const int16_t*);

/* Persistent execution resources, independent of input producer and reduction
 * policy. Native partials are exposed here for future fused W4A8/FlatQuant
 * consumers; public integer GEMM uses reduce() to compact INT32 C. */
class I4Plan {
public:
    ~I4Plan();
    int prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_config& cfg,
                bool retain_partials = false, I4InputLayout layout = I4InputLayout::Native);
    int bind(const I4Weights& weights);
    int pack(int wave, rk_npu_i4_input_producer producer, void* user, rk_npu_i4_timings& times);
    int submit(int wave);
    int begin_partial(int wave, const int16_t** partial);
    int end_partial(int wave);
    int reduce(int wave, int32_t* C, rk_npu_i4_timings& times);
    int consume(int wave, I4PartialConsumer consumer, void* user, rk_npu_i4_timings& times);
    const std::vector<I4Tile>& tiles() const { return tiles_; }
    const rk_npu_i4_config& config() const { return cfg_; }
    int wave_count() const { return int(waves_.size()); }
    // Optional device consumer storage: [GEMM tile, K wave, native INT16].
    // No CPU partial callback is valid in this mode. Public I4 keeps ping-pong.
    bool retains_partials() const { return retained_; }
    const rk_npu_mem& retained_partials() const { return retained_output_.mem; }
private:
    rk_npu_iommu_domain* domain_ = nullptr;
    rk_npu_i4_config cfg_{};
    DomainDataBuffer input_[2], output_[2];
    DomainDataBuffer retained_output_;
    std::vector<I4Wave> waves_;
    std::vector<I4Tile> tiles_;
    uint32_t starts_[3]{}, counts_[3]{};
    uint64_t bound_weight_dma_ = UINT64_MAX;
    int slots_ = 0;
    bool retained_ = false;
    I4InputLayout layout_ = I4InputLayout::Native;
};
}
#endif

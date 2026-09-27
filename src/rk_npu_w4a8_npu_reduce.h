#ifndef RK_NPU_W4A8_NPU_REDUCE_H
#define RK_NPU_W4A8_NPU_REDUCE_H
#include "rk_npu_w4a8.h"
#include "rk_npu_w4a8_cpu.h"
namespace rknpu2_matmul_open::detail {
// Integer Conv consumer of retained W4A4 partials; owns no GEMM weights.
// Its descriptors, Conv weights, output and command buffers are fixed at prepare.
class W4A8NpuReduce {
  public:
    ~W4A8NpuReduce();
    static int query(const rk_npu_i4_config &, I4InputLayout, rk_npu_w4a8_memory_info_ex &);
    int prepare(rk_npu_iommu_domain *, const I4Plan &);
    int run(const W4A8Reduction &, rk_npu_w4a8_timings &);

  private:
    struct Part {
        I4Tile tile;
        int row0, rows;
        uint64_t offset;
    };
    static int describe(const rk_npu_i4_config &, const std::vector<I4Tile> &, std::vector<Part> &,
                        rk_npu_w4a8_memory_info_ex &);
    void output(const W4A8Reduction &);
    rk_npu_iommu_domain *domain_ = nullptr;
    rk_npu_i4_config cfg_{};
    DomainDataBuffer reduced_, weights_;
    rk_npu_mem regcmd_{}, tasks_{};
    std::vector<Part> parts_;
    uint32_t starts_[3]{}, counts_[3]{};
};
} // namespace rknpu2_matmul_open::detail
#endif

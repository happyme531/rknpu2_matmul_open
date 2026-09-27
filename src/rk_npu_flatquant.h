#ifndef RK_NPU_FLATQUANT_H
#define RK_NPU_FLATQUANT_H
#include "rk_npu_internal.h"
#include "rk_npu_w4a4_linear.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

namespace rknpu2_matmul_open::detail {
struct FlatShape {
    int L = 0, R = 0, lp = 0, rp = 0;
};
struct FlatGroup {
    int m0, rows;
};
struct FlatMetrics {
    double pack = 0, submit = 0, scan = 0, quant = 0, sync = 0, total = 0;
};
int flat_shape(int L, int R, FlatShape &out);
int flat_validate_input(const void *A, bool half, int M, int K, int threads, bool to_half);
void flat_pack_input(const void *A, bool half, int K, int m0, const FlatShape &shape,
                     const std::vector<FlatGroup> &groups, int threads, uint16_t *output);
int flat_quantize_native(const uint16_t *data, const FlatShape &shape,
                         const std::vector<FlatGroup> &groups, int m0, int M, int threads,
                         float negative, float positive, uint8_t *packed, float *scales,
                         float *inverse, FlatMetrics &metrics);
int flat_quantize_identity(const void *A, bool half, int M, int K, int threads, float negative,
                           float positive, uint8_t *packed, float *scales, float *inverse,
                           FlatMetrics &metrics);

class FlatWeights {
  public:
    ~FlatWeights();
    int prepare(rk_npu_iommu_domain *domain, int L, int R, const float *left, const float *right);
    const FlatShape &shape() const { return shape_; }
    const rk_npu_mem &memory() const { return arena_.mem; }

  private:
    FlatShape shape_;
    DomainDataBuffer arena_;
};

class FlatPlan {
  public:
    ~FlatPlan();
    static int query(const rk_npu_w4a4_linear_config &cfg, const FlatShape &shape, uint64_t &bytes);
    int prepare(rk_npu_iommu_domain *domain, const rk_npu_w4a4_linear_config &cfg,
                const FlatShape &shape);
    int run(const FlatWeights &weights, const void *A, bool half, float negative, float positive,
            uint8_t *packed, float *scales, float *inverse, FlatMetrics &metrics);
    const FlatShape &shape() const { return shape_; }

  private:
    struct Commands {
        rk_npu_mem regs{}, tasks{};
        std::vector<FlatGroup> groups;
        std::vector<std::pair<int, uint64_t>> bindings;
        uint32_t starts[3]{}, counts[3]{}, mask = 0;
        int tasks_count = 0;
    };
    struct Slot {
        DomainDataBuffer input, middle, output;
        Commands full, tail;
    };
    int build(Commands &commands, Slot &slot, int rows);
    int bind(const FlatWeights &weights);
    int submit_job(int block);
    int pack_job(int block, const void *A, bool half, FlatMetrics &metrics);
    int quant_job(int block, float negative, float positive, uint8_t *packed, float *scales,
                  float *inverse, FlatMetrics &metrics);
    Commands &commands(int block);
    void worker();
    void launch(int block);
    int wait(FlatMetrics &metrics);
    rk_npu_iommu_domain *domain_ = nullptr;
    rk_npu_w4a4_linear_config cfg_{};
    FlatShape shape_;
    Slot slots_[2];
    int batch_ = 0, ubatch_ = 0, slot_count_ = 0, blocks_ = 0, tail_ = 0;
    uint64_t weight_dma_ = UINT64_MAX;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false, pending_ = false, done_ = true;
    int block_ = 0, result_ = 0;
    double submit_us_ = 0;
};
} // namespace rknpu2_matmul_open::detail
#endif

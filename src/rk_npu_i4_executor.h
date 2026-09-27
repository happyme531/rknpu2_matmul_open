#ifndef RK_NPU_I4_EXECUTOR_H
#define RK_NPU_I4_EXECUTOR_H
#include "rk_npu_i4_plan.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace rknpu2_matmul_open::detail {
/* Owns the blocking submit thread only. I4Plan remains usable by another
 * scheduler without this policy (e.g. future grouped/fused execution). */
class I4Executor {
public:
    ~I4Executor();
    int prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_config& cfg,
                bool retain_partials = false, I4InputLayout layout = I4InputLayout::Native);
    int run(const I4Weights& weights, rk_npu_i4_input_producer producer,
             void* user, int32_t* C, rk_npu_i4_timings* times,
             I4PartialConsumer consumer = nullptr, void* consumer_user = nullptr);
    int K() const { return plan_.config().K; }
    const I4Plan& plan() const { return plan_; }
private:
    void worker();
    void submit(int w);
    int wait(rk_npu_i4_timings& times);
    I4Plan plan_;
    std::atomic_flag busy_=ATOMIC_FLAG_INIT;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_=false, pending_=false, done_=true;
    int wave_=0, result_=RK_NPU_OK;
    double submit_us_=0;
};
}
#endif

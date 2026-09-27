#include "rk_npu_i4_executor.h"
#include <chrono>

namespace rknpu2_matmul_open::detail {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::micro>(Clock::now()-t).count(); }
}
I4Executor::~I4Executor() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_=true; cv_.notify_all(); }
    if (thread_.joinable()) thread_.join();
}
int I4Executor::prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_config& cfg, bool retain_partials, I4InputLayout layout) {
    const int rc=plan_.prepare(domain,cfg,retain_partials,layout);
    if (rc) return rc;
    if (cfg.pipeline) thread_=std::thread(&I4Executor::worker,this);
    return RK_NPU_OK;
}
void I4Executor::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        cv_.wait(lock,[&] { return pending_ || stop_; });
        if (stop_) return;
        const int w=wave_; pending_=false;
        lock.unlock();
        const auto start=Clock::now(); const int rc=plan_.submit(w); const double us=elapsed(start);
        lock.lock(); result_=rc; submit_us_=us; done_=true; cv_.notify_all();
    }
}
void I4Executor::submit(int w) {
    std::lock_guard<std::mutex> lock(mutex_);
    wave_=w; done_=false; pending_=true; cv_.notify_all();
}
int I4Executor::wait(rk_npu_i4_timings& times) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,[&] { return done_; });
    times.submit_us+=submit_us_;
    return result_;
}
int I4Executor::run(const I4Weights& weights, rk_npu_i4_input_producer producer,
                     void* user, int32_t* C, rk_npu_i4_timings* metrics,
                     I4PartialConsumer consumer, void* consumer_user) {
    if (!producer || (!C && !consumer && !plan_.retains_partials()) ||
        (plan_.retains_partials() && (C || consumer))) return RK_NPU_ERR_PARAM;
    if (busy_.test_and_set(std::memory_order_acquire)) return RK_NPU_ERR_BUSY;
    struct Unlock { std::atomic_flag& f; ~Unlock() { f.clear(std::memory_order_release); } } unlock{busy_};
    const auto start=Clock::now(); rk_npu_i4_timings times{};
    int rc=plan_.bind(weights);
    const int count=plan_.wave_count();
    auto reduce=[&](int wave) {
        if (plan_.retains_partials()) return RK_NPU_OK;
        return consumer ? plan_.consume(wave,consumer,consumer_user,times) : plan_.reduce(wave,C,times);
    };
    if (!rc && !plan_.config().pipeline) {
        for (int w=0;!rc && w<count;++w) {
            rc=plan_.pack(w,producer,user,times);
            if (!rc) {
                const auto t=Clock::now(); rc=plan_.submit(w); times.submit_us+=elapsed(t);
            }
            if (!rc) rc=reduce(w);
        }
    } else if (!rc) {
        rc=plan_.pack(0,producer,user,times);
        for (int w=0;!rc && w<count;++w) {
            submit(w);
            // At w+1 the slot last consumed by w-1 is available again. The
            // previous partial is reduced before w+1 may overwrite its slot.
            if (w+1<count) rc=plan_.pack(w+1,producer,user,times);
            if (!rc && w>0) rc=reduce(w-1);
            const int submitted=wait(times); // always drain, including producer failures
            if (!rc) rc=submitted;
        }
        if (!rc) rc=reduce(count-1);
    }
    times.total_us=elapsed(start);
    if (metrics) *metrics=times;
    return rc;
}
}

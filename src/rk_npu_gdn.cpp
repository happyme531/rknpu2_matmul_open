#include "rk_npu_gdn_internal.h"
#include <cmath>
#include <new>
#include <system_error>
#include <sys/resource.h>

namespace rknpu2_matmul_open::gdn {
Workers::Workers(int count):count_(count) {
    try {
        for(int w=1;w<count_;++w) threads.emplace_back([this,w] {
            unsigned seen=0;
            for(;;) {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock,[&] { return stop || seen!=epoch; });
                if(stop) return;
                seen=epoch; lock.unlock();
                job(w);
                done.fetch_add(1,std::memory_order_release);
            }
        });
    } catch(...) {
        { std::lock_guard<std::mutex> lock(mutex); stop=true; }
        ready.notify_all();
        for(auto& thread:threads) thread.join();
        throw;
    }
}
Workers::~Workers() {
    { std::lock_guard<std::mutex> lock(mutex); stop=true; }
    ready.notify_all();
    for(auto& thread:threads) thread.join();
}
void Workers::run(std::function<void(int)> fn) {
    { std::lock_guard<std::mutex> lock(mutex); job=std::move(fn); done=0; ++epoch; }
    ready.notify_all();
    job(0);
    while(done.load(std::memory_order_acquire)!=count_-1) {
#if defined(__aarch64__)
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }
}
int validate(const rk_npu_gdn_config* c) {
    if(!c || c->tokens<1 || c->tokens>2048 || c->key_heads!=HK || c->value_heads!=HV ||
       c->key_dim!=D || c->value_dim!=D || c->cpu_threads<1 || c->cpu_threads>4 ||
       (c->chunk_size!=16 && c->chunk_size!=32 && c->chunk_size!=64 && c->chunk_size!=128) ||
       (c->core_mask!=1 && c->core_mask!=2 && c->core_mask!=4 && c->core_mask!=3 && c->core_mask!=7) ||
       !c->timeout_ms || (c->precision!=RK_NPU_GDN_TF32 && c->precision!=RK_NPU_GDN_COMPENSATED_TF32) ||
       (c->head_mapping!=RK_NPU_GDN_HEAD_GROUPED && c->head_mapping!=RK_NPU_GDN_HEAD_INTERLEAVED))
        return RK_NPU_ERR_PARAM;
    const int products=c->precision==RK_NPU_GDN_COMPENSATED_TF32?5:1;
    if(ceil_div(c->tokens,c->chunk_size)*HK*products>4095) return RK_NPU_ERR_PARAM;
    return RK_NPU_OK;
}
struct Busy {
    std::atomic<bool>& flag;
    explicit Busy(std::atomic<bool>& f):flag(f) {
        bool no=false;
        if(!flag.compare_exchange_strong(no,true)) throw Error{RK_NPU_ERR_BUSY};
    }
    ~Busy() { flag.store(false,std::memory_order_release); }
};
} // namespace rknpu2_matmul_open::gdn

using namespace rknpu2_matmul_open::gdn;
struct rk_npu_gdn_workspace {
    Lease lease;
    rk_npu_gdn_config cfg;
    Workers workers;
    std::unique_ptr<Engine> engine;
    std::vector<float> values,history;
    std::atomic<bool> busy{false};
    bool poisoned=false;
    rk_npu_gdn_workspace(rk_npu_iommu_domain* d,const rk_npu_gdn_config& c)
        :lease(d),cfg(c),workers(c.cpu_threads),engine(prepare_engine(d,c,workers)),
         values(size_t(c.tokens)*HV*D),history((2*HK+HV)*D*3) {}
};

extern "C" {
void rk_npu_gdn_config_init(rk_npu_gdn_config* c,int tokens) {
    if(!c) return;
    *c={tokens,HK,HV,D,D,64,4,7,6000,RK_NPU_GDN_TF32,RK_NPU_GDN_HEAD_GROUPED};
}
int rk_npu_gdn_query(const rk_npu_gdn_config* c,rk_npu_gdn_sizes* s) {
    if(!s) return RK_NPU_ERR_PARAM;
    const int rc=validate(c); if(rc) return rc;
    *s={};
    s->query_bytes=s->key_bytes=uint64_t(c->tokens)*HK*D*4;
    s->value_bytes=s->output_bytes=uint64_t(c->tokens)*HV*D*4;
    s->gate_bytes=s->beta_bytes=uint64_t(c->tokens)*HV*4;
    s->state_bytes=uint64_t(HV)*SS*4;
    s->projected_bytes=uint64_t(c->tokens)*(2*HK+HV)*D*4;
    s->conv_weight_bytes=uint64_t(2*HK+HV)*D*4*4;
    s->conv_history_bytes=uint64_t(2*HK+HV)*D*3*4;
    return RK_NPU_OK;
}
int rk_npu_gdn_prepare(rk_npu_iommu_domain* d,const rk_npu_gdn_config* c,rk_npu_gdn_workspace** out) {
    if(!out) return RK_NPU_ERR_PARAM;
    *out=nullptr;
    const int rc=validate(c); if(rc) return rc;
    if(!d || !d->ctx) return RK_NPU_ERR_PARAM;
    try {
        *out=new rk_npu_gdn_workspace(d,*c);
        return RK_NPU_OK;
    } catch(const Error& e) { return e.code; }
      catch(const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
      catch(const std::system_error&) { return RK_NPU_ERR_NOMEM; }
}
void rk_npu_gdn_workspace_free(rk_npu_gdn_workspace* w) { delete w; }
}

namespace {
int execute(rk_npu_gdn_workspace* w,const Inputs& x,const Projected* producer,
            float* output,float* final_state,float* final_history,rk_npu_gdn_timings* timings) {
    if(timings) *timings={};
    if(!w || !output || !final_state || !x.g || !x.beta || !x.state ||
       (!producer && (!x.q || !x.k || !x.v))) return RK_NPU_ERR_PARAM;
    try {
        Busy lock(w->busy);
        if(w->poisoned) return RK_NPU_ERR_SUBMIT;
        rusage cpu_begin{},cpu_end{};
        getrusage(RUSAGE_SELF,&cpu_begin);
        const auto begin=Clock::now();
        for(int i=0;i<w->cfg.tokens*HV;++i)
            if(!(x.g[i]<=0.f) || !(x.beta[i]>=0.f && x.beta[i]<=1.f)) return RK_NPU_ERR_PARAM;
        Times t;
        try { t=w->engine->run(x,producer,output,final_state); }
        catch(const Error& e) {
            if(e.code==RK_NPU_ERR_SUBMIT || e.code==RK_NPU_ERR_IOCTL) w->poisoned=true;
            throw;
        }
        if(producer) std::copy(w->history.begin(),w->history.end(),final_history);
        t.total=micros(begin,Clock::now());
        getrusage(RUSAGE_SELF,&cpu_end);
        auto elapsed=[](const timeval& a,const timeval& b) {
            return double(b.tv_sec-a.tv_sec)*1e6+double(b.tv_usec-a.tv_usec);
        };
        t.cpu_user=elapsed(cpu_begin.ru_utime,cpu_end.ru_utime);
        t.cpu_system=elapsed(cpu_begin.ru_stime,cpu_end.ru_stime);
        if(timings) *timings={t.total,t.pack_compute,t.coefficients_compute,t.solve_compute,t.collect_compute,
            t.driver,t.gram_driver,t.predict_driver,t.update_driver,t.cpu_user,t.cpu_system};
        return RK_NPU_OK;
    } catch(const Error& e) { return e.code; }
      catch(const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
}
}
extern "C" {
int rk_npu_gdn_run(rk_npu_gdn_workspace* w,const float* q,const float* k,const float* v,
    const float* g,const float* beta,const float* state,float* output,float* final_state,rk_npu_gdn_timings* timings) {
    return execute(w,{q,k,v,g,beta,state},nullptr,output,final_state,nullptr,timings);
}
int rk_npu_gdn_run_projected(rk_npu_gdn_workspace* w,const float* projected,const float* weights,const float* history,
    const float* g,const float* beta,const float* state,float* output,float* final_state,float* final_history,
    rk_npu_gdn_timings* timings) {
    if(!w || !projected || !weights || !history || !final_history) return RK_NPU_ERR_PARAM;
    const Projected producer{projected,weights,history,w->values.data(),w->history.data()};
    return execute(w,{nullptr,nullptr,w->values.data(),g,beta,state},&producer,output,final_state,final_history,timings);
}
}

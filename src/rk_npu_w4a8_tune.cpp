#include "rk_npu_w4a8_tune_internal.h"
#include "rk_npu_w4a8_internal.h"
#include <algorithm>
#include <cstdio>
#include <thread>

namespace rknpu2_matmul_open::w4_tune {
int on_worker(const Resources& r,const std::function<int()>& fn) {
    int rc=RK_NPU_ERR_PARAM;
    try {
        std::thread worker([&] {
            try {
                // Child submit/OpenMP workers inherit the same allowed CPUs,
                // matching a caller that applies strategy.cpu_core_mask before
                // workspace creation. Do not mutate the caller's OpenMP team.
                rc=rknpu2_matmul_open::tune::set_thread_cpu_mask(r.cpu)?fn():RK_NPU_ERR_PARAM;
            }
            catch(const std::bad_alloc&) {rc=RK_NPU_ERR_NOMEM;} catch(...) {rc=RK_NPU_ERR_PARAM;}
        });
        worker.join();
    } catch(...) {return RK_NPU_ERR_NOMEM;}
    return rc;
}
int tune(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config& c,const int8_t* B,const float* scales,
         const Profile& p,const Resources& r,Strategy& fast,Strategy& stable,bool panel8) {
    const auto choices=candidates(c,p,r,panel8);if(choices.empty()) return RK_NPU_ERR_PARAM;
    std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(
        rk_npu_iommu_domain_create(ctx,0),rk_npu_iommu_domain_free);
    if(!domain) return RK_NPU_ERR_DOMAIN;
    std::map<int,Weight> weights;
    auto weight=[&](int kt)->rk_npu_w4a8_weights* {
        auto it=weights.find(kt);
        if(it==weights.end()) {
            Weight w(rknpu2_matmul_open::detail::create_w4a8_weights(domain.get(),p.info(kt),B,scales),rk_npu_w4a8_weights_free);
            if(!w) return nullptr;
            it=weights.emplace(kt,std::move(w)).first;
        }
        return it->second.get();
    };
    Data data(c);auto anchor=choices.front().config;
    anchor.k_tile=std::min(480,align_up(c.K,32));anchor.m_tile=std::min(32,c.M);
    auto* anchor_w=weight(anchor.k_tile);if(!anchor_w) return RK_NPU_ERR_NOMEM;
    Workspace reference(rk_npu_w4a8_workspace_create(domain.get(),&anchor),rk_npu_w4a8_workspace_free);
    if(!reference) return RK_NPU_ERR_NOMEM;
    rknpu2_matmul_open::tune::Sample sample;int rc=data.run(reference.get(),anchor_w,sample);if(rc) return rc;
    data.capture();if(!data.verify_reference(B,scales)) return RK_NPU_ERR_PARAM;
    reference.reset();
    bool found=false;
    if(c.verbose) std::fprintf(stderr,"w4 autotune: %zu candidates, npu=0x%x cpu=0x%llx safe=%d limit=%d multiplier=%g pipeline=1\n",
        choices.size(),r.npu,(unsigned long long)r.cpu,p.safe,p.limit,p.multiplier);
    for(const auto& candidate:choices) {
        const auto& cfg=candidate.config;const auto& execution=candidate.execution;
        auto* w=weight(cfg.k_tile);if(!w) return RK_NPU_ERR_NOMEM;
        Workspace ws(rk_npu_w4a8_workspace_create_ex(domain.get(),&cfg,&execution),rk_npu_w4a8_workspace_free);
        if(!ws) return RK_NPU_ERR_NOMEM;
        bool correct=true;
        for(int trial=0;trial<2;++trial) {
            rc=data.run(ws.get(),w,sample);if(rc) return rc;
            if(!data.same()) {correct=false;break;}
        }
        if(!correct) {
            if(c.verbose) std::fprintf(stderr,"w4 autotune: reject layout=%d M/K/N=%d/%d/%d output mismatch\n",int(execution.input_layout),cfg.m_tile,cfg.k_tile,cfg.n_tile);
            continue;
        }
        rknpu2_matmul_open::tune::Score score;
        rc=rknpu2_matmul_open::tune::measure({c.warmup,c.loops,c.repeats},
            [&](rknpu2_matmul_open::tune::Sample& s){return data.run(ws.get(),w,s);},[&]{return data.same();},score);
        if(rc==RK_NPU_ERR_SUBMIT) return rc;
        if(rc) continue;
        Strategy strategy{};strategy.execution=execution;auto& s=strategy.base;s.config=cfg;s.activation_type=c.activation_type;s.cpu_core_mask=r.cpu;
        s.weights=p.info(cfg.k_tile);s.weight_fingerprint=p.fingerprint;s.tuning_revision=revision;
        s.input_us=score.input_us;s.npu_us=score.npu_us;s.sync_us=score.sync_us;s.output_us=score.output_us;
        s.total_us=score.total_us;s.jitter_pct=score.jitter_pct;s.robust_us=score.robust_us;
        if(c.verbose) std::fprintf(stderr,"w4 autotune: layout=%d M/K/N=%d/%d/%d total=%.2f robust=%.2f us relaxed=%d\n",
            int(execution.input_layout),cfg.m_tile,cfg.k_tile,cfg.n_tile,s.total_us,s.robust_us,s.weights.bound_relaxed);
        if(!found||s.total_us<fast.base.total_us) fast=strategy;
        if(!found||s.robust_us<stable.base.robust_us) stable=strategy;
        found=true;
    }
    return found?RK_NPU_OK:RK_NPU_ERR_PARAM;
}
int cached(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config& c,const int8_t* B,const float* scale,
           const Profile& p,const Resources& r,const char* path,int refresh,
           Strategy& fast,Strategy& stable,int* hit,bool panel8) {
    if(hit) *hit=0;
    const auto k=key(ctx,c,p,r,panel8),file=path?std::string(path):cache_path(k);
    if(!refresh&&!load(file,k,c,p,r,fast,stable,panel8)) {if(hit) *hit=1;return RK_NPU_OK;}
    const int rc=tune(ctx,c,B,scale,p,r,fast,stable,panel8);if(rc) return rc;
    const int saved=save(file,k,fast,stable);
    if(saved && c.verbose) std::fprintf(stderr,"w4 autotune: cache save failed (%d), strategy remains valid\n",saved);
    return RK_NPU_OK;
}
namespace {
template<class S>
int entry(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* c,const int8_t* B,const float* scale,
          const char* path,int refresh,S* fast,S* stable,int* hit,bool cache) {
    constexpr bool panel8=std::is_same_v<S,Strategy>;
    if(hit) *hit=0;
    if(!ctx||!c||(!fast&&!stable)||(path&&!*path)) return RK_NPU_ERR_PARAM;
    try {
        Profile p;int rc=profile(*c,B,scale,p);if(rc) return rc;
        const auto r=resources(*c);if(!r.npu||!r.threads) return RK_NPU_ERR_PARAM;
        Strategy f{},s{};
        if(cache&&!refresh) {
            const auto k=key(ctx,*c,p,r,panel8),file=path?std::string(path):cache_path(k);
            if(!load(file,k,*c,p,r,f,s,panel8)) {publish(fast,f);publish(stable,s);if(hit) *hit=1;return RK_NPU_OK;}
        }
        rc=on_worker(r,[&]{return cache?cached(ctx,*c,B,scale,p,r,path,refresh,f,s,hit,panel8):tune(ctx,*c,B,scale,p,r,f,s,panel8);});
        if(!rc) {publish(fast,f);publish(stable,s);}
        return rc;
    } catch(const std::bad_alloc&) {return RK_NPU_ERR_NOMEM;} catch(...) {return RK_NPU_ERR_PARAM;}
}
}
}
extern "C" int rk_npu_w4a8_autotune(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* c,
    const int8_t* B,const float* scale,rk_npu_w4a8_strategy* f,rk_npu_w4a8_strategy* s) {
    return rknpu2_matmul_open::w4_tune::entry(ctx,c,B,scale,nullptr,0,f,s,nullptr,false);
}
extern "C" int rk_npu_w4a8_autotune_cached(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* c,
    const int8_t* B,const float* scale,const char* path,int refresh,
    rk_npu_w4a8_strategy* f,rk_npu_w4a8_strategy* s,int* hit) {
    return rknpu2_matmul_open::w4_tune::entry(ctx,c,B,scale,path,refresh,f,s,hit,true);
}
extern "C" int rk_npu_w4a8_autotune_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* c,
    const int8_t* B,const float* scale,rk_npu_w4a8_strategy_ex* f,rk_npu_w4a8_strategy_ex* s) {
    return rknpu2_matmul_open::w4_tune::entry(ctx,c,B,scale,nullptr,0,f,s,nullptr,false);
}
extern "C" int rk_npu_w4a8_autotune_cached_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* c,
    const int8_t* B,const float* scale,const char* path,int refresh,
    rk_npu_w4a8_strategy_ex* f,rk_npu_w4a8_strategy_ex* s,int* hit) {
    return rknpu2_matmul_open::w4_tune::entry(ctx,c,B,scale,path,refresh,f,s,hit,true);
}

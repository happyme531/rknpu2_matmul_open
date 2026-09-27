#include "rk_npu_w4a8_tune_internal.h"
#include <climits>
#include <cmath>
#include <set>

namespace {
template<class S>
int family(rk_npu_ctx* ctx,const rk_npu_w4a8_family_config* cfg,const int8_t* B,const float* scale,
           const char* directory,int refresh,rk_npu_w4a8_family_summary* fast,S* fs,
           rk_npu_w4a8_family_summary* stable,S* ss,int* hits,int* misses,bool cache) {
    using namespace rknpu2_matmul_open::w4_tune;
    constexpr bool panel8=std::is_same_v<S,Strategy>;
    if(hits) *hits=0;if(misses) *misses=0;
    if(!ctx||!cfg||cfg->m_count<1||!cfg->m_values||(!fast&&!stable)
       ||bool(fast)!=bool(fs)||bool(stable)!=bool(ss)||(directory&&!*directory)) return RK_NPU_ERR_PARAM;
    try {
        std::set<int> unique;double weight=0;
        for(int i=0;i<cfg->m_count;++i) {
            const int m=cfg->m_values[i];const double f=cfg->frequencies?cfg->frequencies[i]:1;
            if(m<1||m>INT_MAX/2||!unique.insert(m).second||!std::isfinite(f)||f<=0) return RK_NPU_ERR_PARAM;
            weight+=f;
        }
        if(!std::isfinite(weight)) return RK_NPU_ERR_PARAM;
        auto base=cfg->base;base.M=1;
        Profile p;int rc=profile(base,B,scale,p);if(rc) return rc;
        const auto r=resources(base);if(!r.npu||!r.threads) return RK_NPU_ERR_PARAM;
        std::vector<Strategy> best_f,best_s;
        double f_score=0,s_score=0;bool found=false;
        int hit_count=0,miss_count=0;
        rc=on_worker(r,[&] {
            for(int kt:k_candidates(base,p)) {
                std::vector<Strategy> f(cfg->m_count),s(cfg->m_count);
                double f_total=0,s_total=0;int one_rc=0;
                for(int i=0;i<cfg->m_count;++i) {
                    auto c=base;c.M=cfg->m_values[i];c.required_k_tile=kt;
                    if(cache) {
                        const auto path=cache_path(key(ctx,c,p,r,panel8),directory);int hit=0;
                        one_rc=cached(ctx,c,B,scale,p,r,path.c_str(),refresh,f[i],s[i],&hit,panel8);
                        if(!one_rc) {hit_count+=hit;miss_count+=!hit;}
                    } else one_rc=tune(ctx,c,B,scale,p,r,f[i],s[i],panel8);
                    if(one_rc==RK_NPU_ERR_SUBMIT) return one_rc;
                    if(one_rc) break;
                    const double fraction=(cfg->frequencies?cfg->frequencies[i]:1)/weight;
                    f_total+=fraction*f[i].base.total_us;s_total+=fraction*s[i].base.robust_us;
                }
                if(one_rc) continue;
                if(!found||f_total<f_score) {best_f=f;f_score=f_total;}
                if(!found||s_total<s_score) {best_s=s;s_score=s_total;}
                found=true;
            }
            return found?RK_NPU_OK:RK_NPU_ERR_PARAM;
        });
        if(hits) *hits=hit_count;if(misses) *misses=miss_count;
        if(rc) return rc;
        auto result=[&](rk_npu_w4a8_family_summary* summary,S* strategies,
                        const std::vector<Strategy>& selected) {
            if(!summary) return;
            *summary={selected[0].base.weights,p.fingerprint,0,0};
            for(int i=0;i<cfg->m_count;++i) {
                publish(strategies+i,selected[i]);
                const double fraction=(cfg->frequencies?cfg->frequencies[i]:1)/weight;
                summary->weighted_total_us+=fraction*selected[i].base.total_us;
                summary->weighted_robust_us+=fraction*selected[i].base.robust_us;
            }
        };
        result(fast,fs,best_f);result(stable,ss,best_s);return RK_NPU_OK;
    } catch(const std::bad_alloc&) {return RK_NPU_ERR_NOMEM;} catch(...) {return RK_NPU_ERR_PARAM;}
}
}
extern "C" int rk_npu_w4a8_autotune_family(rk_npu_ctx* ctx,const rk_npu_w4a8_family_config* c,
    const int8_t* B,const float* scale,rk_npu_w4a8_family_summary* f,rk_npu_w4a8_strategy* fs,
    rk_npu_w4a8_family_summary* s,rk_npu_w4a8_strategy* ss) {
    return family(ctx,c,B,scale,nullptr,0,f,fs,s,ss,nullptr,nullptr,false);
}
extern "C" int rk_npu_w4a8_autotune_family_cached(rk_npu_ctx* ctx,const rk_npu_w4a8_family_config* c,
    const int8_t* B,const float* scale,const char* directory,int refresh,
    rk_npu_w4a8_family_summary* f,rk_npu_w4a8_strategy* fs,rk_npu_w4a8_family_summary* s,rk_npu_w4a8_strategy* ss,
    int* hits,int* misses) {
    return family(ctx,c,B,scale,directory,refresh,f,fs,s,ss,hits,misses,true);
}
extern "C" int rk_npu_w4a8_autotune_family_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_family_config* c,
    const int8_t* B,const float* scale,rk_npu_w4a8_family_summary* f,rk_npu_w4a8_strategy_ex* fs,
    rk_npu_w4a8_family_summary* s,rk_npu_w4a8_strategy_ex* ss) {
    return family(ctx,c,B,scale,nullptr,0,f,fs,s,ss,nullptr,nullptr,false);
}
extern "C" int rk_npu_w4a8_autotune_family_cached_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_family_config* c,
    const int8_t* B,const float* scale,const char* directory,int refresh,
    rk_npu_w4a8_family_summary* f,rk_npu_w4a8_strategy_ex* fs,rk_npu_w4a8_family_summary* s,rk_npu_w4a8_strategy_ex* ss,
    int* hits,int* misses) {
    return family(ctx,c,B,scale,directory,refresh,f,fs,s,ss,hits,misses,true);
}

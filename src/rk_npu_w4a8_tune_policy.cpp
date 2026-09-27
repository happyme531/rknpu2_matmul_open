#include "rk_npu_w4a8_tune_internal.h"
#include "rk_npu_w4a8_internal.h"
#include "rk_npu_i4_bounds.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>

namespace rknpu2_matmul_open::w4_tune {
bool valid(const rk_npu_w4a8_autotune_config& c) {
    return c.M>0 && c.M<=INT_MAX/2 && c.K>0 && c.K<=INT32_MAX/1024 && c.N>0 && c.N<=INT_MAX-63
        && (c.activation_type==RK_NPU_W4A8_F32||c.activation_type==RK_NPU_W4A8_F16)
        && !(c.allowed_npu_core_mask&~7u) && c.warmup>=0 && c.loops>0 && c.repeats>0 && c.timeout_ms
        && c.required_k_tile>=0 && c.required_k_tile<=std::min(2048,align_up(c.K,32)) && c.required_k_tile%32==0
        && std::isfinite(c.k_tile_multiplier) && (c.k_tile_multiplier==0||c.k_tile_multiplier>=1);
}
int profile(const rk_npu_w4a8_autotune_config& c,const int8_t* B,const float* scale,Profile& out) {
    if(!valid(c)||!B||!scale) return RK_NPU_ERR_PARAM;
    uint64_t bytes=0;const rk_npu_i4_weight_config wc{c.K,c.N,32};
    int rc=rk_npu_i4_weights_memory_query(&wc,&bytes);if(rc) return rc;
    double factor=c.k_tile_multiplier;
    if(factor==0) {
        factor=1;
        if(const char* text=std::getenv("RK_NPU_W4A8_K_TILE_MULTIPLIER")) {
            errno=0;char* end=nullptr;factor=std::strtod(text,&end);
            if(errno||end==text||*end||!std::isfinite(factor)||factor<1) return RK_NPU_ERR_PARAM;
        }
    }
    for(int n=0;n<c.N;++n) if(!std::isfinite(scale[n])||scale[n]<=0) return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::detail::I4WeightBound b;
    rc=rknpu2_matmul_open::detail::bound_i4_weights(c.K,c.N,B,0,b);if(rc) return rc;
    Profile p;p.K=c.K;p.N=c.N;p.safe=b.k_tile;p.safe_tiles=b.safe_tiles;p.multiplier=factor;
    p.limit=int(std::min(double(std::min(2048,align_up(c.K,32))),b.k_tile*factor)/32)*32;
    rknpu2_matmul_open::tune::Hash hash;hash.mix(c.K);hash.mix(c.N);
    hash.bytes(B,size_t(c.K)*c.N);hash.bytes(scale,size_t(c.N)*sizeof(float));p.fingerprint=hash.value;
    if(c.required_k_tile && !p.permits(c.required_k_tile)) return RK_NPU_ERR_PARAM;
    out=p;return RK_NPU_OK;
}
Resources resources(const rk_npu_w4a8_autotune_config& c) {
    Resources r;const auto affinity=rknpu2_matmul_open::tune::process_cpu_mask();
    r.allowed_cpu=(c.allowed_cpu_core_mask?c.allowed_cpu_core_mask:affinity)&affinity;
    const auto cpus=rknpu2_matmul_open::tune::preferred_cpus(r.allowed_cpu);
    r.threads=std::min(4,int(cpus.size()));r.cpu=rknpu2_matmul_open::tune::first_cpu_mask(cpus,r.threads);
    r.npu=rknpu2_matmul_open::tune::fixed_npu_mask(c.allowed_npu_core_mask?c.allowed_npu_core_mask:7,c.N,64);
    return r;
}
std::vector<int> k_candidates(const rk_npu_w4a8_autotune_config& c,const Profile& p) {
    if(c.required_k_tile) return p.permits(c.required_k_tile)?std::vector<int>{c.required_k_tile}:std::vector<int>{};
    std::set<int> choices;
    for(int k:{256,480,p.safe,p.limit,p.limit/2,p.limit*3/4}) {
        k=std::max(32,std::min(k,p.limit)/32*32);
        if(p.permits(k)) choices.insert(k);
    }
    return {choices.begin(),choices.end()};
}
std::vector<Candidate> candidates(const rk_npu_w4a8_autotune_config& c,const Profile& p,
                                   const Resources& r,bool panel8) {
    std::vector<Candidate> out;if(!r.npu||!r.threads) return out;
    const int cores=__builtin_popcount(r.npu),max_shard=ceil_div(ceil_div(c.N,64),cores)*64;
    std::set<int> ns,ms;
    for(int n:{512,1024,2048,4096}) ns.insert(std::min(n,max_shard));
    for(int m:{16,32,64}) ms.insert(std::min(m,c.M));
    auto add=[&](int k,int m,int n,rk_npu_w4a8_input_layout layout) {
        Candidate candidate{{c.M,c.N,c.K,m,k,n,r.npu,c.timeout_ms,1,r.threads},
            {sizeof(rk_npu_w4a8_options),layout,RK_NPU_W4A8_REDUCE_CPU}};
        rk_npu_w4a8_memory_info_ex memory{};
        if(!rk_npu_w4a8_memory_query_ex(&candidate.config,&candidate.execution,&memory))
            out.push_back(candidate);
    };
    const auto ks=k_candidates(c,p);
    // Keep all legacy candidates. An extended search can always select native.
    for(int k:ks) for(int m:ms) for(int n:ns) add(k,m,n,RK_NPU_W4A8_INPUT_NATIVE);
    // On small ragged batches, panel8 adds a native tail task and repeats B
    // traffic (e.g. M17 -> 16+1); native can process these in one task.
    if(panel8 && c.M>=32 && (c.M>64 || c.M%4==0)) {
        ms.clear();
        for(int m:{16,32,64,128,256}) ms.insert(std::min(m,c.M/4*4));
        for(int k:ks) for(int m:ms) for(int n:ns) add(k,m,n,RK_NPU_W4A8_INPUT_PANEL8);
    }
    return out;
}
bool valid_execution(const Strategy& s,bool panel8) {
    const auto& o=s.execution;
    if(o.struct_size!=sizeof(o) || o.reduce_backend!=RK_NPU_W4A8_REDUCE_CPU ||
       (o.input_layout!=RK_NPU_W4A8_INPUT_NATIVE &&
        (!panel8 || o.input_layout!=RK_NPU_W4A8_INPUT_PANEL8))) return false;
    rk_npu_w4a8_memory_info_ex info{};
    return !rk_npu_w4a8_memory_query_ex(&s.base.config,&o,&info);
}
bool valid_strategy(const Strategy& strategy,const rk_npu_w4a8_autotune_config& c,
                    const Profile& p,const Resources& r,bool panel8) {
    const auto& s=strategy.base;
    if(!valid_execution(strategy,panel8)) return false;
    const auto& a=s.config;const auto& w=s.weights;
    if(s.tuning_revision!=revision||s.activation_type!=c.activation_type||s.weight_fingerprint!=p.fingerprint
       ||a.M!=c.M||a.N!=c.N||a.K!=c.K||a.npu_core_mask!=r.npu||a.cpu_threads!=r.threads
       ||s.cpu_core_mask!=r.cpu||a.timeout_ms!=c.timeout_ms||a.pipeline!=1||!p.permits(a.k_tile)
       ||(c.required_k_tile && a.k_tile!=c.required_k_tile)
       ||w.config.K!=c.K||w.config.N!=c.N||w.config.k_tile!=a.k_tile||w.safe_k_tile!=p.safe
       ||w.multiplier!=p.multiplier||w.bound_relaxed!=!p.certified(a.k_tile)) return false;
    for(double t:{s.input_us,s.npu_us,s.sync_us,s.output_us,s.total_us,s.jitter_pct,s.robust_us})
        if(!std::isfinite(t)||t<0) return false;
    return true;
}
std::string key(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config& c,const Profile& p,const Resources& r,bool panel8) {
    std::ostringstream out;out.imbue(std::locale::classic());
    out<<revision<<' '<<(panel8?3:1)<<' '<<ctx->driver_version<<' '<<c.M<<' '<<c.N<<' '<<c.K<<' '<<int(c.activation_type)
       <<' '<<(c.allowed_npu_core_mask?c.allowed_npu_core_mask:7)<<' '<<r.allowed_cpu<<' '<<r.npu<<' '<<r.cpu
       <<' '<<c.warmup<<' '<<c.loops<<' '<<c.repeats<<' '<<c.timeout_ms<<' '<<c.required_k_tile
       <<' '<<p.fingerprint<<' '<<p.safe<<' '<<p.safe_tiles<<' '<<p.limit<<' '<<rknpu2_matmul_open::tune::double_bits(p.multiplier);
    return out.str();
}
} // namespace rknpu2_matmul_open::w4_tune

extern "C" void rk_npu_w4a8_autotune_config_init(rk_npu_w4a8_autotune_config* c,int M,int N,int K) {
    if(c) *c={M,N,K,RK_NPU_W4A8_F32,7,0,2,8,3,500,0,0,0};
}
extern "C" void rk_npu_w4a8_family_config_init(rk_npu_w4a8_family_config* c,int N,int K,int count,const int* ms) {
    if(!c) return;*c={};rk_npu_w4a8_autotune_config_init(&c->base,1,N,K);
    c->m_count=count;c->m_values=ms;
}
namespace {
rk_npu_w4a8_weights* create_tuned(rk_npu_iommu_domain* domain,
    const rknpu2_matmul_open::w4_tune::Strategy& strategy,const int8_t* B,const float* scale,bool panel8) {
    const auto* s=&strategy.base;
    if(!domain||!B||!scale) return nullptr;
    try {
        rk_npu_w4a8_autotune_config c;
        rk_npu_w4a8_autotune_config_init(&c,s->config.M,s->config.N,s->config.K);
        c.activation_type=s->activation_type;c.required_k_tile=s->config.k_tile;
        c.k_tile_multiplier=s->weights.multiplier;c.timeout_ms=s->config.timeout_ms;
        rknpu2_matmul_open::w4_tune::Profile p;if(rknpu2_matmul_open::w4_tune::profile(c,B,scale,p)) return nullptr;
        rknpu2_matmul_open::w4_tune::Resources r{s->config.npu_core_mask,s->cpu_core_mask,s->cpu_core_mask,s->config.cpu_threads};
        if(!rknpu2_matmul_open::w4_tune::valid_strategy(strategy,c,p,r,panel8)) return nullptr;
        return rknpu2_matmul_open::detail::create_w4a8_weights(domain,p.info(c.required_k_tile),B,scale);
    } catch(...) {return nullptr;}
}
}
extern "C" rk_npu_w4a8_weights* rk_npu_w4a8_weights_create_tuned(rk_npu_iommu_domain* domain,
    const rk_npu_w4a8_strategy* s,const int8_t* B,const float* scale) {
    return s?create_tuned(domain,rknpu2_matmul_open::w4_tune::extend(*s),B,scale,false):nullptr;
}
extern "C" rk_npu_w4a8_weights* rk_npu_w4a8_weights_create_tuned_ex(rk_npu_iommu_domain* domain,
    const rk_npu_w4a8_strategy_ex* s,const int8_t* B,const float* scale) {
    return s?create_tuned(domain,*s,B,scale,true):nullptr;
}
extern "C" rk_npu_w4a8_workspace* rk_npu_w4a8_workspace_create_tuned_ex(rk_npu_iommu_domain* domain,
    const rk_npu_w4a8_strategy_ex* s) {
    if(!s || s->base.tuning_revision!=rknpu2_matmul_open::w4_tune::revision || s->base.config.pipeline!=1 ||
       !rknpu2_matmul_open::w4_tune::valid_execution(*s,true) || s->base.config.K!=s->base.weights.config.K ||
       s->base.config.N!=s->base.weights.config.N || s->base.config.k_tile!=s->base.weights.config.k_tile)
        return nullptr;
    return rk_npu_w4a8_workspace_create_ex(domain,&s->base.config,&s->execution);
}

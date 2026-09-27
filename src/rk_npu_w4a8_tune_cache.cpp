#include "rk_npu_w4a8_tune_internal.h"
#include <istream>
#include <ostream>

namespace rknpu2_matmul_open::w4_tune {
namespace {
constexpr const char* magic="rk_npu_w4a8_tuning_cache_v2";
void write_strategy(std::ostream& out,const char* label,const Strategy& strategy) {
    const auto& s=strategy.base;const auto& execution=strategy.execution;
    const auto& c=s.config;const auto& w=s.weights;
    out<<label<<' '<<c.M<<' '<<c.N<<' '<<c.K<<' '<<c.m_tile<<' '<<c.k_tile<<' '<<c.n_tile
       <<' '<<c.npu_core_mask<<' '<<c.timeout_ms<<' '<<c.pipeline<<' '<<c.cpu_threads
       <<' '<<int(s.activation_type)<<' '<<s.cpu_core_mask<<' '<<w.config.K<<' '<<w.config.N
       <<' '<<w.config.k_tile<<' '<<w.safe_k_tile<<' '<<rknpu2_matmul_open::tune::double_bits(w.multiplier)
       <<' '<<w.bound_relaxed<<' '<<s.weight_fingerprint<<' '<<s.tuning_revision;
    for(double d:{s.input_us,s.npu_us,s.sync_us,s.output_us,s.total_us,s.jitter_pct,s.robust_us})
        out<<' '<<rknpu2_matmul_open::tune::double_bits(d);
    out<<' '<<execution.struct_size<<' '<<int(execution.input_layout)<<' '<<int(execution.reduce_backend)<<'\n';
}
bool read_strategy(std::istream& in,const char* label,Strategy& strategy) {
    auto& s=strategy.base;
    std::string name;int dtype;uint64_t factor;
    auto& c=s.config;auto& w=s.weights;
    if(!(in>>name>>c.M>>c.N>>c.K>>c.m_tile>>c.k_tile>>c.n_tile>>c.npu_core_mask
         >>c.timeout_ms>>c.pipeline>>c.cpu_threads>>dtype>>s.cpu_core_mask>>w.config.K>>w.config.N
         >>w.config.k_tile>>w.safe_k_tile>>factor>>w.bound_relaxed>>s.weight_fingerprint>>s.tuning_revision)
       ||name!=label || (dtype!=RK_NPU_W4A8_F32 && dtype!=RK_NPU_W4A8_F16)) return false;
    s.activation_type=static_cast<rk_npu_w4a8_activation_type>(dtype);w.multiplier=rknpu2_matmul_open::tune::bits_double(factor);
    for(double* d:{&s.input_us,&s.npu_us,&s.sync_us,&s.output_us,&s.total_us,&s.jitter_pct,&s.robust_us}) {
        uint64_t bits;if(!(in>>bits)) return false;*d=rknpu2_matmul_open::tune::bits_double(bits);
    }
    int layout,reduce;
    if(!(in>>strategy.execution.struct_size>>layout>>reduce) ||
       (layout!=RK_NPU_W4A8_INPUT_NATIVE && layout!=RK_NPU_W4A8_INPUT_PANEL8) ||
       reduce!=RK_NPU_W4A8_REDUCE_CPU) return false;
    strategy.execution.input_layout=static_cast<rk_npu_w4a8_input_layout>(layout);
    strategy.execution.reduce_backend=static_cast<rk_npu_w4a8_reduce_backend>(reduce);
    return true;
}
}
std::string cache_path(const std::string& key,const char* directory) {
    rknpu2_matmul_open::tune::Hash hash;hash.bytes(key.data(),key.size());
    return rknpu2_matmul_open::tune::cache_path(hash.value,directory,"rk_npu_w4a8");
}
int load(const std::string& path,const std::string& expected,const rk_npu_w4a8_autotune_config& cfg,
         const Profile& profile,const Resources& resources,Strategy& fast,Strategy& stable,bool panel8) {
    Strategy f{},s{};
    const int rc=rknpu2_matmul_open::tune::load_cache(path,[&](std::istream& in) {
        std::string header,key;
        return bool(std::getline(in,header)) && header==magic && bool(std::getline(in,key)) && key==expected
            && read_strategy(in,"fastest",f) && read_strategy(in,"stable",s)
            && valid_strategy(f,cfg,profile,resources,panel8) && valid_strategy(s,cfg,profile,resources,panel8);
    });
    if(!rc) {fast=f;stable=s;}return rc;
}
int save(const std::string& path,const std::string& key,const Strategy& fast,const Strategy& stable) {
    return rknpu2_matmul_open::tune::save_cache(path,[&](std::ostream& out) {
        out<<magic<<'\n'<<key<<'\n';write_strategy(out,"fastest",fast);write_strategy(out,"stable",stable);
    });
}
}

namespace {
template<class S>
int load_entry(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,S* fast,S* stable) {
    if(!ctx||!cfg||!fast||!stable||(path&&!*path)) return RK_NPU_ERR_PARAM;
    constexpr bool panel8=std::is_same_v<S,rknpu2_matmul_open::w4_tune::Strategy>;
    try {
        rknpu2_matmul_open::w4_tune::Profile p;int rc=rknpu2_matmul_open::w4_tune::profile(*cfg,B,scales,p);if(rc) return rc;
        const auto r=rknpu2_matmul_open::w4_tune::resources(*cfg);if(!r.npu||!r.threads) return RK_NPU_ERR_PARAM;
        const auto key=rknpu2_matmul_open::w4_tune::key(ctx,*cfg,p,r,panel8);
        rknpu2_matmul_open::w4_tune::Strategy f{},s{};
        rc=rknpu2_matmul_open::w4_tune::load(path?path:rknpu2_matmul_open::w4_tune::cache_path(key),key,*cfg,p,r,f,s,panel8);
        if(!rc) {rknpu2_matmul_open::w4_tune::publish(fast,f);rknpu2_matmul_open::w4_tune::publish(stable,s);}return rc;
    } catch(const std::bad_alloc&) {return RK_NPU_ERR_NOMEM;} catch(...) {return RK_NPU_ERR_PARAM;}
}
template<class S>
int save_entry(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,const S* fast,const S* stable) {
    if(!ctx||!cfg||!fast||!stable||(path&&!*path)) return RK_NPU_ERR_PARAM;
    constexpr bool panel8=std::is_same_v<S,rknpu2_matmul_open::w4_tune::Strategy>;
    try {
        rknpu2_matmul_open::w4_tune::Profile p;int rc=rknpu2_matmul_open::w4_tune::profile(*cfg,B,scales,p);if(rc) return rc;
        const auto r=rknpu2_matmul_open::w4_tune::resources(*cfg);if(!r.npu||!r.threads) return RK_NPU_ERR_PARAM;
        const auto f=rknpu2_matmul_open::w4_tune::extend(*fast),s=rknpu2_matmul_open::w4_tune::extend(*stable);
        if(!rknpu2_matmul_open::w4_tune::valid_strategy(f,*cfg,p,r,panel8)||!rknpu2_matmul_open::w4_tune::valid_strategy(s,*cfg,p,r,panel8)) return RK_NPU_ERR_PARAM;
        const auto key=rknpu2_matmul_open::w4_tune::key(ctx,*cfg,p,r,panel8);
        return rknpu2_matmul_open::w4_tune::save(path?path:rknpu2_matmul_open::w4_tune::cache_path(key),key,f,s);
    } catch(const std::bad_alloc&) {return RK_NPU_ERR_NOMEM;} catch(...) {return RK_NPU_ERR_PARAM;}
}
}
extern "C" int rk_npu_w4a8_autotune_cache_load(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,rk_npu_w4a8_strategy* fast,rk_npu_w4a8_strategy* stable) {
    return load_entry(ctx,cfg,B,scales,path,fast,stable);
}
extern "C" int rk_npu_w4a8_autotune_cache_save(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,const rk_npu_w4a8_strategy* fast,const rk_npu_w4a8_strategy* stable) {
    return save_entry(ctx,cfg,B,scales,path,fast,stable);
}
extern "C" int rk_npu_w4a8_autotune_cache_load_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,rk_npu_w4a8_strategy_ex* fast,rk_npu_w4a8_strategy_ex* stable) {
    return load_entry(ctx,cfg,B,scales,path,fast,stable);
}
extern "C" int rk_npu_w4a8_autotune_cache_save_ex(rk_npu_ctx* ctx,const rk_npu_w4a8_autotune_config* cfg,
    const int8_t* B,const float* scales,const char* path,const rk_npu_w4a8_strategy_ex* fast,const rk_npu_w4a8_strategy_ex* stable) {
    return save_entry(ctx,cfg,B,scales,path,fast,stable);
}

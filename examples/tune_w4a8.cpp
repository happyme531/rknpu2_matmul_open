#include "rk_npu_w4a8_tune.h"
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void read(const std::string& path,std::vector<T>& data) {
    std::ifstream in(path,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size()*sizeof(T)))||in.peek()!=EOF)
        throw std::runtime_error("invalid size or unreadable file: "+path);
}
static void print(const char* label,const rk_npu_w4a8_strategy_ex& strategy) {
    const auto& s=strategy.base;
    std::printf("%s total=%.3fus robust=%.3fus tiles(M/K/N)=%d/%d/%d npu=0x%x cpu=0x%llx/%d pipeline=%d safeK=%d multiplier=%g relaxed=%d layout=%s\n",
        label,s.total_us,s.robust_us,s.config.m_tile,s.config.k_tile,s.config.n_tile,s.config.npu_core_mask,
        (unsigned long long)s.cpu_core_mask,s.config.cpu_threads,s.config.pipeline,s.weights.safe_k_tile,
        s.weights.multiplier,s.weights.bound_relaxed,
        strategy.execution.input_layout==RK_NPU_W4A8_INPUT_PANEL8?"panel8":"native");
}
int main(int argc,char** argv) {
    try {
        rk_npu_w4a8_autotune_config c;rk_npu_w4a8_autotune_config_init(&c,128,4096,4096);
        std::string weights_path,scales_path,cache;bool refresh=false,no_cache=false,native_only=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() {if(++i>=argc) throw std::runtime_error("missing value: "+arg);return std::string(argv[i]);};
            if(arg=="--shape") {c.M=std::stoi(value());c.K=std::stoi(value());c.N=std::stoi(value());}
            else if(arg=="--native-only") native_only=true;
            else if(arg=="--f16") c.activation_type=RK_NPU_W4A8_F16;
            else if(arg=="--weights") weights_path=value();
            else if(arg=="--scales") scales_path=value();
            else if(arg=="--multiplier") c.k_tile_multiplier=std::stod(value());
            else if(arg=="--k-tile") c.required_k_tile=std::stoi(value());
            else if(arg=="--cpu-mask") c.allowed_cpu_core_mask=std::stoull(value(),nullptr,0);
            else if(arg=="--npu-mask") c.allowed_npu_core_mask=std::stoul(value(),nullptr,0);
            else if(arg=="--warmup") c.warmup=std::stoi(value());
            else if(arg=="--loops") c.loops=std::stoi(value());
            else if(arg=="--repeats") c.repeats=std::stoi(value());
            else if(arg=="--cache") cache=value();
            else if(arg=="--refresh") refresh=true;
            else if(arg=="--no-cache") no_cache=true;
            else if(arg=="--verbose") c.verbose=1;
            else if(arg=="--help") {
                std::puts("tune_w4a8 --shape M K N [--f16] [--weights raw_i4_codes.bin --scales raw_f32.bin]\n"
                          "  [--multiplier 1|1.5|2.1|0(env)] [--k-tile K] [--cpu-mask 0xf0] [--npu-mask 7]\n"
                          "  [--native-only] [--cache PATH] [--refresh|--no-cache] [--warmup 2] [--loops 8] [--repeats 3] [--verbose]\n"
                          "Without files, generate synthetic weights/scales. Search native/panel8, ubatch<=256; CPU reduction, pipeline=1.");
                return 0;
            } else throw std::runtime_error("unknown option: "+arg);
        }
        if(c.M<1||c.K<1||c.N<1||weights_path.empty()!=scales_path.empty()) throw std::runtime_error("invalid shape or weight/scale pair");
        std::vector<int8_t> B(size_t(c.K)*c.N);std::vector<float> scales(c.N,.01f);
        if(weights_path.empty()) {std::mt19937 rng(20260920);for(auto& b:B) b=int(rng()%16)-8;std::puts("SYNTHETIC weights; timings are not model E2E");}
        else {read(weights_path,B);read(scales_path,scales);}
        std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);
        if(!ctx) throw std::runtime_error("cannot open NPU");
        rk_npu_w4a8_strategy_ex fast{},stable{};int hit=0,rc;
        if(native_only) {
            rk_npu_w4a8_options_init(&fast.execution);rk_npu_w4a8_options_init(&stable.execution);
            rc=no_cache?rk_npu_w4a8_autotune(ctx.get(),&c,B.data(),scales.data(),&fast.base,&stable.base)
                :rk_npu_w4a8_autotune_cached(ctx.get(),&c,B.data(),scales.data(),cache.empty()?nullptr:cache.c_str(),refresh,&fast.base,&stable.base,&hit);
        } else rc=no_cache?rk_npu_w4a8_autotune_ex(ctx.get(),&c,B.data(),scales.data(),&fast,&stable)
            :rk_npu_w4a8_autotune_cached_ex(ctx.get(),&c,B.data(),scales.data(),cache.empty()?nullptr:cache.c_str(),refresh,&fast,&stable,&hit);
        if(rc) throw std::runtime_error(rk_npu_strerror(rc));
        std::printf("CACHE %s\n",no_cache?"disabled":hit?"hit":"miss");print("FASTEST",fast);print("STABLE",stable);
    } catch(const std::exception& e) {std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}

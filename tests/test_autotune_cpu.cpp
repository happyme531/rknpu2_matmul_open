#include "../src/rk_npu_w4a8_tune_internal.h"
#include "rk_npu_quant_matmul.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#define CHECK(x) do {if(!(x)) throw std::runtime_error(#x);} while(0)

static void timing() {
    int calls=0,validations=0;rknpu2_matmul_open::tune::Score s;
    CHECK(!rknpu2_matmul_open::tune::measure({2,2,3},[&](rknpu2_matmul_open::tune::Sample& one){one.total_us=++calls;one.input_us=calls*2;return 0;},
        [&]{++validations;return true;},s));
    CHECK(calls==8 && validations==6 && s.total_us==5.5 && s.input_us==11 && s.robust_us==7.5);
    CHECK(std::fabs(s.jitter_pct-400.0/5.5)<1e-12);
    calls=0;validations=0;
    CHECK(rknpu2_matmul_open::tune::measure({0,8,2},[&](rknpu2_matmul_open::tune::Sample&){return ++calls==4?RK_NPU_ERR_SUBMIT:0;},
          [&]{++validations;return true;},s)==RK_NPU_ERR_SUBMIT);
    CHECK(calls==4 && validations==3);
    CHECK(rknpu2_matmul_open::tune::fixed_npu_mask(7,65,64)==3);
    CHECK(rknpu2_matmul_open::tune::fixed_npu_mask(7,65,32)==7);
}
static void w8_cache(rk_npu_ctx& ctx,const std::string& path) {
    rk_npu_matmul_autotune_config c;rk_npu_matmul_autotune_config_init(&c,1,64,64);
    c.allowed_npu_core_mask=1;c.allowed_cpu_core_mask=rknpu2_matmul_open::tune::first_cpu_mask(rknpu2_matmul_open::tune::preferred_cpus(rknpu2_matmul_open::tune::process_cpu_mask()),1);
    rk_npu_matmul_strategy s{};s.op_kind=RK_NPU_MATMUL_I8I8I32;s.M=1;s.N=64;s.K=64;s.k_tile=64;s.n_tile=64;
    s.npu_core_mask=1;s.cpu_core_mask=c.allowed_cpu_core_mask;s.cpu_threads=1;s.wave_count=1;s.n_groups=1;
    s.total_us=3.5;s.robust_us=4.5;
    CHECK(!rk_npu_matmul_autotune_cache_save(&ctx,path.c_str(),s.op_kind,&c,&s,&s));
    std::ifstream in(path);std::string magic,key;std::getline(in,magic);std::getline(in,key);
    CHECK(magic=="rk_npu_matmul_tuning_cache_v2");
    std::ostringstream expected;expected<<"key 2 3 "<<ctx.driver_version<<" 0 1 64 64 1 "
        <<c.allowed_cpu_core_mask<<" 2 8 3 500 0";
    CHECK(key==expected.str());
    rk_npu_matmul_strategy f{},stable{};
    CHECK(!rk_npu_matmul_autotune_cache_load(&ctx,path.c_str(),s.op_kind,&c,&f,&stable));
    CHECK(f.total_us==3.5 && stable.robust_us==4.5);
    {std::ofstream out(path,std::ios::app);out<<"junk";}
    CHECK(rk_npu_matmul_autotune_cache_load(&ctx,path.c_str(),s.op_kind,&c,&f,&stable)==RK_NPU_ERR_CACHE_MISS);
}
static void w4_policy_cache(rk_npu_ctx& ctx,const std::string& path) {
    rk_npu_w4a8_autotune_config c;rk_npu_w4a8_autotune_config_init(&c,4,1,2048);
    c.k_tile_multiplier=1;
    c.allowed_cpu_core_mask=rknpu2_matmul_open::tune::first_cpu_mask(rknpu2_matmul_open::tune::preferred_cpus(rknpu2_matmul_open::tune::process_cpu_mask()),1);
    std::vector<int8_t> b(2048,0);std::fill(b.begin()+624,b.begin()+1424,-8);
    float scale=.1f;rknpu2_matmul_open::w4_tune::Profile p;
    CHECK(!rknpu2_matmul_open::w4_tune::profile(c,b.data(),&scale,p));
    CHECK(p.safe==1120 && !p.certified(768) && p.certified(1024));
    CHECK(!p.permits(768));
    const auto original=c;c.required_k_tile=768;
    CHECK(rknpu2_matmul_open::w4_tune::profile(c,b.data(),&scale,p)==RK_NPU_ERR_PARAM);
    c.k_tile_multiplier=1.5;
    CHECK(!rknpu2_matmul_open::w4_tune::profile(c,b.data(),&scale,p));
    CHECK(p.permits(768) && p.info(768).bound_relaxed);
    c=original;CHECK(!rknpu2_matmul_open::w4_tune::profile(c,b.data(),&scale,p));
    const auto r=rknpu2_matmul_open::w4_tune::resources(c);
    const uint64_t before=rknpu2_matmul_open::tune::process_cpu_mask();
    CHECK(!rknpu2_matmul_open::w4_tune::on_worker(r,[&]{return rknpu2_matmul_open::tune::process_cpu_mask()==r.cpu?0:RK_NPU_ERR_PARAM;}));
    CHECK(rknpu2_matmul_open::tune::process_cpu_mask()==before);
    const auto choices=rknpu2_matmul_open::w4_tune::candidates(c,p,r);CHECK(!choices.empty());
    for(const auto& cfg:choices) CHECK(cfg.config.pipeline==1 && p.certified(cfg.config.k_tile));
    rk_npu_w4a8_strategy s{};s.config=choices.front().config;s.activation_type=c.activation_type;s.cpu_core_mask=r.cpu;
    s.weights=p.info(s.config.k_tile);s.weight_fingerprint=p.fingerprint;s.tuning_revision=rknpu2_matmul_open::w4_tune::revision;
    s.total_us=1.25;s.robust_us=2;
    CHECK(!rk_npu_w4a8_autotune_cache_save(&ctx,&c,b.data(),&scale,path.c_str(),&s,&s));
    rk_npu_w4a8_strategy f{},stable{};
    CHECK(!rk_npu_w4a8_autotune_cache_load(&ctx,&c,b.data(),&scale,path.c_str(),&f,&stable));
    CHECK(f.total_us==1.25 && f.config.pipeline==1 && stable.robust_us==2);
    int hit=0;
    // Fake ctx.fd=-1: a hit must not allocate or submit anything to the device.
    CHECK(!rk_npu_w4a8_autotune_cached(&ctx,&c,b.data(),&scale,path.c_str(),0,&f,&stable,&hit) && hit==1);
    b[0]=1;CHECK(rk_npu_w4a8_autotune_cache_load(&ctx,&c,b.data(),&scale,path.c_str(),&f,&stable)==RK_NPU_ERR_CACHE_MISS);b[0]=0;
    scale=.2f;CHECK(rk_npu_w4a8_autotune_cache_load(&ctx,&c,b.data(),&scale,path.c_str(),&f,&stable)==RK_NPU_ERR_CACHE_MISS);scale=.1f;
    c.k_tile_multiplier=1.5;CHECK(rk_npu_w4a8_autotune_cache_load(&ctx,&c,b.data(),&scale,path.c_str(),&f,&stable)==RK_NPU_ERR_CACHE_MISS);c=original;
    s.config.pipeline=0;CHECK(rk_npu_w4a8_autotune_cache_save(&ctx,&c,b.data(),&scale,path.c_str(),&s,&s)==RK_NPU_ERR_PARAM);s.config.pipeline=1;
    s.weights.safe_k_tile=2048;
    CHECK(!rknpu2_matmul_open::w4_tune::save(path,rknpu2_matmul_open::w4_tune::key(&ctx,c,p,r),rknpu2_matmul_open::w4_tune::extend(s),rknpu2_matmul_open::w4_tune::extend(s)));
    CHECK(rk_npu_w4a8_autotune_cache_load(&ctx,&c,b.data(),&scale,path.c_str(),&f,&stable)==RK_NPU_ERR_CACHE_MISS);
    CHECK(rknpu2_matmul_open::tune::save_cache(path+"/not_a_directory",[](std::ostream& out){out<<1;})==RK_NPU_ERR_IO);
}
int main() {
    char directory[]="/tmp/rk_autotune_cpu_XXXXXX";const char* dir=::mkdtemp(directory);
    if(!dir) return 1;
    try {
        rk_npu_ctx ctx{};ctx.fd=-1;ctx.driver_version=0x908;
        timing();w8_cache(ctx,std::string(dir)+"/w8");w4_policy_cache(ctx,std::string(dir)+"/w4");
        std::filesystem::remove_all(dir);
        std::puts("PASS shared scoring, fail-stop, W8 cache compatibility, W4 bounds/cache identity, affinity");
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());std::filesystem::remove_all(dir);return 1;}
}

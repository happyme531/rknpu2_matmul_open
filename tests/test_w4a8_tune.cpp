#include "rk_npu_w4a8_tune.h"
#include "../src/rk_npu_autotune_common.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#define CHECK(x) do {if(!(x)) throw std::runtime_error(#x);} while(0)
using W=std::unique_ptr<rk_npu_w4a8_weights,decltype(&rk_npu_w4a8_weights_free)>;
using P=std::unique_ptr<rk_npu_w4a8_workspace,decltype(&rk_npu_w4a8_workspace_free)>;
static void verify(rk_npu_iommu_domain* domain,const rk_npu_w4a8_strategy& s,
                   const std::vector<int8_t>& b,const std::vector<float>& scale) {
    const auto& c=s.config;W w(rk_npu_w4a8_weights_create_tuned(domain,&s,b.data(),scale.data()),rk_npu_w4a8_weights_free);
    P p(rk_npu_w4a8_workspace_create(domain,&c),rk_npu_w4a8_workspace_free);CHECK(w&&p&&c.pipeline==1);
    std::vector<float> a(size_t(c.M)*c.K),out(size_t(c.M)*c.N);
    std::vector<uint16_t> ah(a.size()),oh(out.size());
    for(size_t i=0;i<a.size();++i) {a[i]=float(int(i%23)-11)*.0625f;ah[i]=rknpu2_matmul_open::bits::float_to_half(a[i]);}
    CHECK(!rk_npu_w4a8_run_f32(p.get(),w.get(),a.data(),out.data(),nullptr));
    CHECK(!rk_npu_w4a8_run_f16(p.get(),w.get(),ah.data(),oh.data(),nullptr));
    for(int m=0;m<c.M;++m) {
        float maxabs=0;for(int k=0;k<c.K;++k) maxabs=std::max(maxabs,std::fabs(a[size_t(m)*c.K+k]));
        const float as=maxabs==0?1:maxabs/127.f,inv=1.f/as;
        std::vector<int> q(c.K);
        for(int k=0;k<c.K;++k) q[k]=int(std::clamp(std::round(a[size_t(m)*c.K+k]*inv),-127.f,127.f));
        for(int n=0;n<c.N;++n) {
            int64_t sum=0;for(int k=0;k<c.K;++k) sum+=q[k]*int(b[size_t(k)*c.N+n]);
            const float ref=(float(sum)*as)*scale[n];const size_t ix=size_t(m)*c.N+n;
            CHECK(out[ix]==ref && oh[ix]==rknpu2_matmul_open::bits::float_to_half(ref));
        }
    }
}
static void one(rk_npu_ctx* ctx,rk_npu_iommu_domain* domain,int M,int K,int N,bool half,double multiplier,
                int required,const std::string& file) {
    std::mt19937 rng(13);std::vector<int8_t> b(size_t(K)*N);std::vector<float> scale(N,.0078125f);
    for(auto& x:b) x=int(rng()%16)-8;
    rk_npu_w4a8_autotune_config c;rk_npu_w4a8_autotune_config_init(&c,M,N,K);
    c.activation_type=half?RK_NPU_W4A8_F16:RK_NPU_W4A8_F32;c.warmup=0;c.loops=1;c.repeats=1;
    c.allowed_cpu_core_mask=M>=8?0xf0:0x10;c.k_tile_multiplier=multiplier;c.required_k_tile=required;
    rk_npu_w4a8_strategy fast{},stable{};int hit=-1;const auto before=rknpu2_matmul_open::tune::process_cpu_mask();
    CHECK(!rk_npu_w4a8_autotune_cached(ctx,&c,b.data(),scale.data(),file.c_str(),1,&fast,&stable,&hit)&&hit==0);
    CHECK(before==rknpu2_matmul_open::tune::process_cpu_mask());verify(domain,fast,b,scale);verify(domain,stable,b,scale);
    const auto total=fast.total_us;
    CHECK(!rk_npu_w4a8_autotune_cached(ctx,&c,b.data(),scale.data(),file.c_str(),0,&fast,&stable,&hit)&&hit==1);
    CHECK(fast.total_us==total && fast.config.pipeline==1);
    if(multiplier==1) CHECK(!fast.weights.bound_relaxed);
    auto changed=b;changed[0]=changed[0]==7?6:changed[0]+1;
    W bad(rk_npu_w4a8_weights_create_tuned(domain,&fast,changed.data(),scale.data()),rk_npu_w4a8_weights_free);CHECK(!bad);
    std::printf("PASS W4 tune/cache M=%d K=%d N=%d dtype=%s kt=%d relaxed=%d\n",M,K,N,half?"f16":"f32",fast.config.k_tile,fast.weights.bound_relaxed);
}
static void family(rk_npu_ctx* ctx,const std::string& directory) {
    const int ms[]={1,4,32};double freq[]={1,2,1};const int K=513,N=193;
    std::vector<int8_t> b(K*N,1);std::vector<float> scale(N,.01f);
    rk_npu_w4a8_family_config c;rk_npu_w4a8_family_config_init(&c,N,K,1,ms);
    c.base.warmup=0;c.base.loops=1;c.base.repeats=1;c.base.allowed_cpu_core_mask=0xf0;c.frequencies=freq;
    rk_npu_w4a8_family_summary f{},s{};rk_npu_w4a8_strategy fs[3]{},ss[3]{};int hits=0,misses=0;
    auto run=[&] {return rk_npu_w4a8_autotune_family_cached(ctx,&c,b.data(),scale.data(),directory.c_str(),0,&f,fs,&s,ss,&hits,&misses);};
    CHECK(!run() && hits==0 && misses>0);
    c.m_count=3;CHECK(!run() && hits>0 && misses>0);
    CHECK(!run() && hits>0 && misses==0);
    freq[0]=100;CHECK(!run() && hits>0 && misses==0);
    for(int i=0;i<3;++i) CHECK(fs[i].config.k_tile==f.weights.config.k_tile && ss[i].config.k_tile==s.weights.config.k_tile && fs[i].config.pipeline==1);
    std::puts("PASS W4 family shared K, partial cache reuse, changed frequencies without retuning");
}
int main(int argc,char**) {
    char directory[]="/tmp/rk_w4_tune_XXXXXX";if(!::mkdtemp(directory)) return 1;
    try {
        std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free);CHECK(domain);
        one(ctx.get(),domain.get(),3,257,193,false,1,0,std::string(directory)+"/small");
        if(argc==1) {
            one(ctx.get(),domain.get(),65,1057,257,true,1,0,std::string(directory)+"/tail");
            one(ctx.get(),domain.get(),4,4096,193,false,2.1,2048,std::string(directory)+"/relaxed");
            family(ctx.get(),std::string(directory)+"/family");
            rk_npu_w4a8_autotune_config c;rk_npu_w4a8_autotune_config_init(&c,3,64,1024);
            c.k_tile_multiplier=1.5;c.required_k_tile=704;c.warmup=0;c.loops=1;c.repeats=1;c.allowed_cpu_core_mask=0x10;
            std::vector<int8_t> b(1024*64,-8);std::vector<float> scale(64,1);rk_npu_w4a8_strategy s{};
            CHECK(rk_npu_w4a8_autotune(ctx.get(),&c,b.data(),scale.data(),&s,nullptr)==RK_NPU_ERR_PARAM);
            std::puts("PASS sampled saturation rejected during tuning; no runtime fallback added");
        }
        std::filesystem::remove_all(directory);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());std::filesystem::remove_all(directory);return 1;}
}

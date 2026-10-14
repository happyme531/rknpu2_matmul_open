#include "rk_npu_w4a8.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);} while(0)
using namespace rknpu2_matmul_open;
static std::vector<float> oracle(int M,int K,int N,const std::vector<float>& a,
    const std::vector<int8_t>& b,const std::vector<float>& scales) {
    std::vector<float> out(size_t(M)*N);
    for(int m=0;m<M;++m) {
        float maximum=0;for(int k=0;k<K;++k)maximum=std::max(maximum,std::fabs(a[size_t(m)*K+k]));
        const float scale=maximum?std::max(maximum/127.f,std::numeric_limits<float>::min()):1.f;
        const float inv=1.f/scale;std::vector<int64_t> sum(N);
        for(int k=0;k<K;++k) {
            const int q=int(std::clamp(std::round(a[size_t(m)*K+k]*inv),-127.f,127.f));
            for(int n=0;n<N;++n)sum[n]+=q*int64_t(b[size_t(k)*N+n]);
        }
        for(int n=0;n<N;++n)out[size_t(m)*N+n]=(float(sum[n])*scale)*scales[n];
    }
    return out;
}
static void run(rk_npu_iommu_domain* domain,int M,int K,int N,int core,bool panel=false) {
    std::vector<int8_t> b[2];std::vector<float> scale[2];
    for(int w=0;w<2;++w) {
        b[w].resize(size_t(K)*N);scale[w].resize(N);
        for(int k=0;k<K;++k)for(int n=0;n<N;++n)
            b[w][size_t(k)*N+n]=K>8192?-8:((k*(17+w*2)+n*(11+w*2))%16)-8;
        for(int n=0;n<N;++n)scale[w][n]=(.001f+float(n%13)*.000125f)*(w+1);
    }
    const rk_npu_i4_weight_config wc{K,N,480};
    using Weight=std::unique_ptr<rk_npu_w4a8_weights,decltype(&rk_npu_w4a8_weights_free)>;
    Weight weights[]={Weight(rk_npu_w4a8_weights_create(domain,&wc,b[0].data(),scale[0].data()),rk_npu_w4a8_weights_free),
                      Weight(rk_npu_w4a8_weights_create(domain,&wc,b[1].data(),scale[1].data()),rk_npu_w4a8_weights_free)};
    CHECK(weights[0] && weights[1]);
    rk_npu_w4a8_config cfg;rk_npu_w4a8_config_init(&cfg,M,N,K);
    cfg.k_tile=480;cfg.n_tile=256;cfg.m_tile=panel?8:std::min(M,64);
    cfg.pipeline=RK_NPU_W4A8_CHAIN;cfg.cpu_threads=M<=4?1:4;cfg.npu_core_mask=core;
    rk_npu_w4a8_options options;rk_npu_w4a8_options_init(&options);
    options.input_layout=panel?RK_NPU_W4A8_INPUT_PANEL8:RK_NPU_W4A8_INPUT_NATIVE;
    rk_npu_w4a8_memory_info_ex info{};CHECK(!rk_npu_w4a8_memory_query_ex(&cfg,&options,&info));
    CHECK(info.base.backend.input_bytes==uint64_t(M)*((K+31)/32*32));
    CHECK(info.base.backend.partial_bytes==uint64_t((K+479)/480)*4*M*((N+63)/64*64));
    CHECK(info.workspace_bytes==info.base.backend.input_bytes+info.base.backend.partial_bytes+
        info.base.backend.control_bytes+info.base.cpu_scratch_bytes);
    std::unique_ptr<rk_npu_w4a8_workspace,decltype(&rk_npu_w4a8_workspace_free)> ws(
        rk_npu_w4a8_workspace_create_ex(domain,&cfg,&options),rk_npu_w4a8_workspace_free);CHECK(ws);
    std::vector<float> a(size_t(M)*K),c(size_t(M)*N+32,-12345.f);
    std::vector<uint16_t> ah(a.size()),ch(c.size(),0x5555);
    for(int trial=0;trial<4;++trial) {
        for(size_t i=0;i<a.size();++i) {
            a[i]=trial==1?0.f:trial==2?15.875f:float(int((i*(17+trial))%255)-127)*.125f;
            ah[i]=bits::float_to_half(a[i]);
        }
        for(int w:{0,1,0}) {
            const auto expected=oracle(M,K,N,a,b[w],scale[w]);
            CHECK(!rk_npu_w4a8_run_f32(ws.get(),weights[w].get(),a.data(),c.data()+16,nullptr));
            CHECK(!rk_npu_w4a8_run_f16(ws.get(),weights[w].get(),ah.data(),ch.data()+16,nullptr));
            for(size_t i=0;i<expected.size();++i)
                CHECK(c[i+16]==expected[i] && ch[i+16]==bits::float_to_half(expected[i]));
            for(int i=0;i<16;++i)
                CHECK(c[i]==-12345.f && c[expected.size()+16+i]==-12345.f && ch[i]==0x5555 && ch[expected.size()+16+i]==0x5555);
        }
    }
    a[0]=std::numeric_limits<float>::quiet_NaN();rk_npu_w4a8_timings timings{};
    CHECK(rk_npu_w4a8_run_f32(ws.get(),weights[0].get(),a.data(),c.data()+16,&timings)==RK_NPU_ERR_PARAM);
    CHECK(timings.submit_us==0);
    a[0]=0;const auto expected=oracle(M,K,N,a,b[0],scale[0]);
    CHECK(!rk_npu_w4a8_run_f32(ws.get(),weights[0].get(),a.data(),c.data()+16,nullptr));
    for(size_t i=0;i<expected.size();++i)CHECK(c[i+16]==expected[i]);
    std::printf("PASS chain M=%d K=%d N=%d mask=%d layout=%d full f32/f16 INT64 oracle/rebind/zero/reuse/guards/recovery\n",M,K,N,core,int(options.input_layout));
}
int main() {
    try {
        std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free);CHECK(domain);
        for(int core:{1,2,4,3,7})for(int M:{1,2,4,8})run(domain.get(),M,1057,193,core);
        run(domain.get(),1,20003,193,7);
        run(domain.get(),33,1057,257,7,true);
        rk_npu_w4a8_config cfg;rk_npu_w4a8_config_init(&cfg,33,193,1057);cfg.pipeline=RK_NPU_W4A8_CHAIN;
        rk_npu_w4a8_options options;rk_npu_w4a8_options_init(&options);options.reduce_backend=RK_NPU_W4A8_REDUCE_NPU;
        rk_npu_w4a8_memory_info_ex info{};CHECK(rk_npu_w4a8_memory_query_ex(&cfg,&options,&info)==RK_NPU_ERR_PARAM);
        CHECK(!rk_npu_w4a8_workspace_create_ex(domain.get(),&cfg,&options));
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

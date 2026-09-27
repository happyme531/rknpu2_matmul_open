#include "rk_npu_matmul_i4.h"
#include "rk_npu_quant_matmul.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(0)
using W=std::unique_ptr<rk_npu_i4i4i32_weights,decltype(&rk_npu_i4i4i32_weights_free)>;
using P=std::unique_ptr<rk_npu_i4_workspace,decltype(&rk_npu_i4_workspace_free)>;

static void compare(int M,int N,int K,const std::vector<int8_t>& a,const std::vector<int8_t>& b,const int32_t* c) {
    for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
        int64_t sum=0;
        for (int k=0;k<K;++k) sum+=int(a[size_t(m)*K+k])*int(b[size_t(k)*N+n]);
        if (sum!=c[size_t(m)*N+n]) {
            std::fprintf(stderr,"m=%d n=%d expected=%lld got=%d\n",m,n,(long long)sum,c[size_t(m)*N+n]);
            throw std::runtime_error("integer reference mismatch");
        }
    }
}
static int calls=0;
static void ordinary(rk_npu_iommu_domain* domain,int M,int K,int N,int mt,int kt,int nt,int mask,int pipe,bool extreme=false) {
    std::mt19937 rng(91+calls++);
    std::vector<int8_t> a(size_t(M)*K),b(size_t(K)*N);
    for (auto& v:a) v=extreme?-8:int(rng()%16)-8;
    for (size_t i=0;i<b.size();++i) b[i]=extreme?((i%N)&1?7:-8):int(rng()%16)-8;
    rk_npu_i4_config cfg; rk_npu_i4_config_init(&cfg,M,N,K);
    cfg.m_tile=mt;cfg.k_tile=kt;cfg.n_tile=nt;cfg.npu_core_mask=mask;cfg.pipeline=pipe;
    cfg.cpu_threads=M>=128?4:1;
    rk_npu_i4_weight_config wc{K,N,kt};
    W w(rk_npu_i4i4i32_weights_create(domain,&wc,b.data()),rk_npu_i4i4i32_weights_free); CHECK(w);
    P p(rk_npu_i4_workspace_create(domain,&cfg),rk_npu_i4_workspace_free); CHECK(p);
    std::vector<int32_t> output(size_t(M)*N+32,0x12345678);
    for (int repeat=0;repeat<3;++repeat) {
        // Changed inputs detect stale ping-pong buffers on repeated calls.
        if (!extreme) a[repeat%a.size()]=int8_t(repeat-8);
        rk_npu_i4_timings times{};
        const int rc=rk_npu_i4i4i32_run(p.get(),w.get(),a.data(),output.data()+16,&times);
        if (rc) { std::fprintf(stderr,"run rc=%d\n",rc); throw std::runtime_error("submit/run failed: stop"); }
        compare(M,N,K,a,b,output.data()+16);
        for (int j=0;j<16;++j) CHECK(output[j]==0x12345678 && output[output.size()-1-j]==0x12345678);
    }
    std::printf("PASS M=%d K=%d N=%d tiles=%d/%d/%d mask=%d pipeline=%d extreme=%d\n",M,K,N,mt,kt,nt,mask,pipe,extreme);
    std::fflush(stdout);
}

// Producer independently encodes a virtual matrix, no intermediate A buffer:
// verifies the extension boundary intended for fused preprocessing.
struct Producer { int K; int fail_k=-1; rk_npu_i4_workspace* p=nullptr; const rk_npu_i4i4i32_weights* w=nullptr; int32_t* c=nullptr; };
static int generated(int m,int k) { return (m*7+k*3)%16-8; }
static int produce(void* user,const rk_npu_i4_input_tile* t) {
    auto& state=*static_cast<Producer*>(user);
    if (state.p) {
        std::vector<int8_t> dummy(state.K,0);
        CHECK(rk_npu_i4i4i32_run(state.p,state.w,dummy.data(),state.c,nullptr)==RK_NPU_ERR_BUSY);
    }
    if (t->k0==state.fail_k) return RK_NPU_ERR_PARAM;
    for (int k=0;k<t->k;++k) for (int m=0;m<t->rows;++m) {
        const size_t index=(size_t(k/32)*t->rows+m)*32+k%32;
        const uint8_t q=uint8_t(generated(t->m0+m,t->k0+k))&15;
        if (index&1) t->dst[index/2]|=q; else t->dst[index/2]=q<<4;
    }
    return RK_NPU_OK;
}
static void sharing(rk_npu_iommu_domain* domain,rk_npu_iommu_domain* other) {
    const int K=513,N=193;
    std::vector<int8_t> b(K*N,3), b2(K*N,-2);
    rk_npu_i4_weight_config wc{K,N,256};
    W w(rk_npu_i4i4i32_weights_create(domain,&wc,b.data()),rk_npu_i4i4i32_weights_free);
    W w2(rk_npu_i4i4i32_weights_create(domain,&wc,b2.data()),rk_npu_i4i4i32_weights_free);
    W foreign(rk_npu_i4i4i32_weights_create(other,&wc,b.data()),rk_npu_i4i4i32_weights_free);
    CHECK(w && w2 && foreign);
    for (int M:{1,5,131}) {
        rk_npu_i4_config cfg; rk_npu_i4_config_init(&cfg,M,N,K);
        cfg.k_tile=256;cfg.m_tile=M==1?1:32;cfg.n_tile=64;cfg.npu_core_mask=M==131?7:1;
        P p(rk_npu_i4_workspace_create(domain,&cfg),rk_npu_i4_workspace_free); CHECK(p);
        std::vector<int8_t> a(M*K);
        for (int m=0;m<M;++m) for (int k=0;k<K;++k) a[m*K+k]=generated(m,k);
        std::vector<int32_t> c(M*N);
        Producer state{K,-1,p.get(),w.get(),c.data()};
        CHECK(!rk_npu_i4i4i32_run_with_producer(p.get(),w.get(),produce,&state,c.data(),nullptr));
        compare(M,N,K,a,b,c.data());
        CHECK(!rk_npu_i4i4i32_run(p.get(),w2.get(),a.data(),c.data(),nullptr));
        compare(M,N,K,a,b2,c.data());
        CHECK(rk_npu_i4i4i32_run(p.get(),foreign.get(),a.data(),c.data(),nullptr)==RK_NPU_ERR_DOMAIN);
        state.fail_k=256; // failure while an earlier wave may still execute
        CHECK(rk_npu_i4i4i32_run_with_producer(p.get(),w.get(),produce,&state,c.data(),nullptr)==RK_NPU_ERR_PARAM);
        state.fail_k=-1;
        CHECK(!rk_npu_i4i4i32_run_with_producer(p.get(),w.get(),produce,&state,c.data(),nullptr));
        compare(M,N,K,a,b,c.data());
        a.back()=8;
        CHECK(rk_npu_i4i4i32_run(p.get(),w.get(),a.data(),c.data(),nullptr)==RK_NPU_ERR_PARAM);
        a.back()=generated(M-1,K-1);
        CHECK(!rk_npu_i4i4i32_run(p.get(),w.get(),a.data(),c.data(),nullptr));
        compare(M,N,K,a,b,c.data());
    }
    b[0]=8; CHECK(!rk_npu_i4i4i32_weights_create(domain,&wc,b.data()));
    std::puts("PASS weight sharing/rebinding, native producer, BUSY/domain errors, pipeline error recovery");
}

static void i8_canary(rk_npu_iommu_domain* domain) {
    rk_npu_matmul_strategy s{};
    s.op_kind=RK_NPU_MATMUL_I8I8I32;s.M=3;s.K=64;s.N=64;
    s.k_tile=64;s.n_tile=64;s.wave_count=1;s.n_groups=1;s.npu_core_mask=1;s.cpu_threads=1;
    s.cpu_core_mask=1ull<<sched_getcpu();
    s.a_layout=RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    rk_npu_matmul_weight_config wc{64,64,64};
    std::vector<int8_t> a(3*64,2),b(64*64,-3); std::vector<int32_t> c(3*64);
    auto* w=rk_npu_i8i8i32_weights_create(domain,&wc,b.data());
    auto* p=rk_npu_matmul_workspace_create(domain,&s); CHECK(w && p);
    const int rc=rk_npu_i8i8i32_run(p,w,a.data(),c.data());
    rk_npu_matmul_workspace_free(p);rk_npu_i8i8i32_weights_free(w);
    CHECK(!rc); CHECK(std::all_of(c.begin(),c.end(),[](int x){return x==-384;}));
    std::puts("PASS existing INT8 canary after INT4");
}

int main(int argc,char** argv) {
    auto* ctx=rk_npu_open(nullptr); if (!ctx) return 2;
    auto* domain=rk_npu_iommu_domain_create(ctx,0);
    auto* other=rk_npu_iommu_domain_create(ctx,1);
    int result=0;
    try {
        CHECK(domain && other);
        ordinary(domain,2,480,64,128,480,1024,1,0);
        if (argc<2 || std::strcmp(argv[1],"--canary")) {
            for (int pipe:{0,1}) {
                ordinary(domain,1,1,1,128,480,1024,1,pipe);
                ordinary(domain,3,33,65,2,32,64,1,pipe);
                ordinary(domain,129,513,193,32,256,64,1,pipe);
                ordinary(domain,5,2048,128,4,480,64,1,pipe,true);
            }
            for (int mask:{2,4,3,7}) {
                ordinary(domain,7,511,257,3,480,64,mask,1,true);
                ordinary(domain,129,1025,257,64,256,128,mask,1);
            }
            ordinary(domain,2,64,4161,128,480,4096,1,1);
            // Cross the parallel-reduction threshold, compare every output
            // including unaligned row strides and N tails, not only samples.
            ordinary(domain,129,513,513,64,256,128,7,0);
            ordinary(domain,129,513,513,64,256,128,7,1);
            sharing(domain,other);
            i8_canary(domain);
            ordinary(domain,3,511,65,2,480,64,1,1,true);
        }
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); result=1; }
    rk_npu_iommu_domain_free(other);rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);
    return result;
}

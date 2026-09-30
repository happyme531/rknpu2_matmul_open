#include "rk_npu_quant_matmul.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace bits = rknpu2_matmul_open::bits;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void mode(int m) {
    setenv("RK_NPU_I8_NPU_REDUCE", m ? "1" : "0", 1);
    setenv("RK_NPU_W8A8_NPU_DEQUANT", m == 2 ? "1" : "0", 1);
}
int main(int argc, char** argv) {
    try {
        require(argc == 7 || (argc == 8 && std::string(argv[7]) == "--scale-boundaries"),
                "usage: test_i8_dpu_chain M K N Ktile Ntile mask [--scale-boundaries]");
        const int M=std::stoi(argv[1]), K=std::stoi(argv[2]), N=std::stoi(argv[3]);
        const int kt=std::stoi(argv[4]), nt=std::stoi(argv[5]), mask=std::stoi(argv[6]);
        require(M>0 && K>0 && N>0 && kt>0 && nt>0,"positive shape");
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);
        require(bool(ctx), "open");
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)>
            domain(rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free);
        require(bool(domain), "domain");
        std::vector<int8_t> a(size_t(M)*K,127), b(size_t(K)*N);
        for(int m=0;m<M;++m) a[size_t(m)*K]=126-m%3;
        for(int k=0;k<K;++k) for(int n=0;n<N;++n)
            b[size_t(k)*N+n]=k==0 ? 126-n%7 : (n%2 ? -128 : 127);
        const int recipes[][2]={{0,0},{1,0},{2,0},{3,0},{0,1},{2,1},{3,2}};
        rk_npu_matmul_weight_config wc{K,N,kt};
        auto* iw=rk_npu_i8i8i32_weights_create(domain.get(),&wc,b.data());
        require(iw,"i8 weights");
        std::vector<float> sw(N),sa(M),af(a.size());
        for(int n=0;n<N;++n) sw[n]=.001003f*(1+n%5);
        for(int m=0;m<M;++m) sa[m]=.010007f*(1+m%3);
        if(argc == 8) {
            const float scales[]={0x1p-24f,0x1p-14f,1.f,65504.f,.0137f};
            for(int m=0;m<M;++m)sa[m]=scales[m%5];
            for(int n=0;n<N;++n)sw[n]=scales[n%5];
        }
        auto* fw=rk_npu_f32i8f32_weights_create(domain.get(),&wc,b.data(),sw.data());
        require(fw,"f32 weights");
        auto sw_alt=sw;
        // Keep the half-subnormal boundary representable in both weight sets.
        for(auto& scale:sw_alt) scale=scale>0x1p-24f ? scale*.5f : scale*2.f;
        auto* fw_alt=rk_npu_f32i8f32_weights_create(domain.get(),&wc,b.data(),sw_alt.data());
        auto sw_bad=sw_alt;sw_bad[N-1]=1e-30f;
        auto* fw_bad=rk_npu_f32i8f32_weights_create(domain.get(),&wc,b.data(),sw_bad.data());
        require(fw_alt && fw_bad,"scale-switch weights");
        int passed=0;
        for(const auto& recipe:recipes) {
            rk_npu_matmul_strategy s{};
            s.M=M;s.K=K;s.N=N;s.k_tile=kt;s.n_tile=nt;
            s.a_layout=rk_npu_matmul_a_layout(recipe[0]);s.c_layout=rk_npu_matmul_c_layout(recipe[1]);
            s.npu_core_mask=mask;s.cpu_core_mask=0x10;s.cpu_threads=1;
            s.wave_count=(K+kt-1)/kt;s.n_groups=((N+31)/32*32+nt-1)/nt;
            s.op_kind=RK_NPU_MATMUL_I8I8I32;
            mode(1);
            rk_npu_matmul_workspace_requirements mem{};
            if(rk_npu_matmul_workspace_memory_query(&s,&mem)) continue;
            auto* ip=rk_npu_matmul_workspace_create(domain.get(),&s);
            require(ip,"i8 workspace");
            s.op_kind=RK_NPU_MATMUL_F32I8F32_STATIC;
            mode(2);
            auto* fp=rk_npu_matmul_workspace_create(domain.get(),&s);
            require(fp,"f32 workspace");
            // Creating CPU workspaces after NPU ones also tests that changing
            // the environment does not mutate an already prepared workspace.
            mode(0);
            auto* cp=rk_npu_matmul_workspace_create(domain.get(),&s);
            require(cp,"cpu workspace");
            std::vector<int32_t> out(size_t(M)*N+16,0x12345678);
            std::vector<float> f(out.size(),12345.f),cpu(out.size());
            for(int pass=0;pass<4;++pass) {
                auto* weights=pass==1 ? fw_alt : fw;
                const auto& weight_scale=pass==1 ? sw_alt : sw;
                auto input=a;
                if(pass==1) for(auto& x:input)x=-x;
                if(pass==2) std::fill(input.begin(),input.end(),0);
                for(int m=0;m<M;++m)for(int k=0;k<K;++k)
                    af[size_t(m)*K+k]=float(input[size_t(m)*K+k])*sa[m];
                require(rk_npu_i8i8i32_run(ip,iw,input.data(),out.data())==0,"raw chain run");
                require(rk_npu_f32i8f32_run_static(fp,weights,af.data(),sa.data(),f.data())==0,"dequant chain run");
                require(rk_npu_f32i8f32_run_static(cp,weights,af.data(),sa.data(),cpu.data())==0,"cpu run");
                for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
                    int64_t sum=0;
                    for(int k=0;k<K;++k)sum+=int(input[size_t(m)*K+k])*int(b[size_t(k)*N+n]);
                    const auto i=size_t(m)*N+n;
                    require(out[i]==sum,"exact INT32 >2^24 oracle");
                    const float as=bits::half_to_float(bits::float_to_half(sa[m]));
                    const float ws=bits::half_to_float(bits::float_to_half(weight_scale[n]));
                    const float expected=(float(sum)*as)*ws;
                    if(f[i]!=expected) {
                        std::fprintf(stderr,"mismatch m=%d n=%d got=%.9g expected=%.9g sum=%lld\n",
                                     m,n,f[i],expected,(long long)sum);
                        throw std::runtime_error("FP16 coefficient oracle");
                    }
                    require(cpu[i]==(float(sum)*sa[m])*weight_scale[n],"CPU default unchanged");
                }
                for(size_t i=size_t(M)*N;i<out.size();++i)
                    require(out[i]==0x12345678 && f[i]==12345.f,"output guard");
            }
            // Invalid half coefficients return before submit, then recover on
            // the same workspace with valid scales (no hidden CPU fallback).
            auto bad=sa;bad[0]=1e-30f;
            require(rk_npu_f32i8f32_run_static(fp,fw,af.data(),bad.data(),f.data())==RK_NPU_ERR_PARAM,
                    "unrepresentable scale rejected");
            require(rk_npu_f32i8f32_run_static(fp,fw,af.data(),sa.data(),f.data())==0,"valid retry");
            const auto good=f;
            require(rk_npu_f32i8f32_run_static(fp,fw_bad,af.data(),sa.data(),f.data())==RK_NPU_ERR_PARAM,
                    "invalid channel scale rejected after partial coefficient update");
            for(int repeat=0;repeat<2;++repeat) {
                require(rk_npu_f32i8f32_run_static(fp,fw,af.data(),sa.data(),f.data())==0,
                        "channel coefficient recovery and cache hit");
                require(f==good,"channel coefficient cache contents");
            }
            rk_npu_matmul_workspace_free(cp);rk_npu_matmul_workspace_free(fp);rk_npu_matmul_workspace_free(ip);
            std::printf("PASS M=%d K=%d N=%d kt=%d nt=%d mask=%d A=%d C=%d\n",M,K,N,kt,nt,mask,recipe[0],recipe[1]);
            std::fflush(stdout);
            ++passed;
        }
        require(passed>0,"no supported layout exercised");
        rk_npu_f32i8f32_weights_free(fw_bad);rk_npu_f32i8f32_weights_free(fw_alt);
        rk_npu_f32i8f32_weights_free(fw);rk_npu_i8i8i32_weights_free(iw);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

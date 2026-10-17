#include "rk_npu_gdn.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { if(!(x)) throw std::runtime_error(std::string(#x)+":"+std::to_string(__LINE__)); } while(0)
namespace {
constexpr int HK=16, HV=32, D=128, CW=8192, SS=HV*D*D;
using Workspace=std::unique_ptr<rk_npu_gdn_workspace,decltype(&rk_npu_gdn_workspace_free)>;
struct Input {
    int T;
    std::vector<float> q,k,v,g,beta,state;
    explicit Input(int t):T(t),q(t*HK*D),k(q.size()),v(t*HV*D),g(t*HV),beta(g.size()),state(SS) {
        auto fill=[](std::vector<float>& v,int factor,float scale) {
            for(size_t i=0;i<v.size();++i) v[i]=float(int((i*factor+11)%127)-63)*scale;
        };
        fill(q,17,.001f); fill(k,19,.001f); fill(v,23,.01f); fill(state,29,.0001f);
        for(int i=0;i<T*HV;++i) { g[i]=-.01f*float(i%13+1); beta[i]=.2f+.01f*float(i%61); }
        if(T>2) g[2*HV+3]=-INFINITY;
    }
};
struct Reference { std::vector<float> output,state; };
Reference reference(const Input& x,bool interleaved) {
    std::vector<double> state(x.state.begin(),x.state.end());
    Reference result{std::vector<float>(size_t(x.T)*HV*D),std::vector<float>(SS)};
    for(int h=0;h<HV;++h) {
        const int kh=interleaved?h%HK:h/2;
        for(int t=0;t<x.T;++t) for(int v=0;v<D;++v) {
            double* row=state.data()+(h*D+v)*D;
            const float* key=x.k.data()+(t*HK+kh)*D;
            const float* query=x.q.data()+(t*HK+kh)*D;
            double prediction=0;
            const double decay=std::exp(double(x.g[t*HV+h]));
            for(int k=0;k<D;++k) { row[k]*=decay; prediction+=row[k]*key[k]; }
            const double delta=(x.v[(t*HV+h)*D+v]-prediction)*x.beta[t*HV+h];
            double output=0;
            for(int k=0;k<D;++k) { row[k]+=delta*key[k]; output+=row[k]*query[k]; }
            result.output[(t*HV+h)*D+v]=float(output/std::sqrt(double(D)));
        }
    }
    std::copy(state.begin(),state.end(),result.state.begin());
    return result;
}
void close(const std::vector<float>& actual,const std::vector<float>& expected,double tolerance) {
    CHECK(actual.size()==expected.size());
    double diff=0,norm=0;
    for(size_t i=0;i<actual.size();++i) {
        CHECK(std::isfinite(actual[i]));
        const double d=double(actual[i])-expected[i]; diff+=d*d; norm+=double(expected[i])*expected[i];
    }
    const double l2=std::sqrt(diff/std::max(norm,1e-30));
    if(l2>tolerance) std::fprintf(stderr,"relative_l2=%g tolerance=%g\n",l2,tolerance);
    CHECK(l2<=tolerance);
}
Workspace prepare(rk_npu_iommu_domain* domain,const rk_npu_gdn_config& config) {
    rk_npu_gdn_workspace* raw=nullptr;
    CHECK(rk_npu_gdn_prepare(domain,&config,&raw)==RK_NPU_OK);
    return Workspace(raw,rk_npu_gdn_workspace_free);
}
void metadata() {
    rk_npu_gdn_config c; rk_npu_gdn_config_init(&c,128);
    rk_npu_gdn_sizes s{};
    CHECK(rk_npu_gdn_query(&c,&s)==0);
    CHECK(s.state_bytes==2097152 && s.projected_bytes==128*8192*4 && s.conv_history_bytes==8192*3*4);
    c.tokens=0; CHECK(rk_npu_gdn_query(&c,&s)==RK_NPU_ERR_PARAM);
    c.tokens=128; c.key_dim=64; CHECK(rk_npu_gdn_query(&c,&s)==RK_NPU_ERR_PARAM);
    rk_npu_gdn_workspace* w=reinterpret_cast<rk_npu_gdn_workspace*>(1);
    CHECK(rk_npu_gdn_prepare(nullptr,&c,&w)==RK_NPU_ERR_PARAM && w==nullptr);
    rk_npu_gdn_config_init(&c,2048);
    c.precision=RK_NPU_GDN_COMPENSATED_TF32; c.chunk_size=16;
    CHECK(rk_npu_gdn_query(&c,&s)==RK_NPU_ERR_PARAM);
    rk_npu_gdn_workspace_free(nullptr);
    std::puts("PASS metadata and C ABI shape contract");
}
void recurrent(rk_npu_iommu_domain* domain,int tokens,int chunk,uint32_t mask,int threads,int precision,bool interleaved) {
    rk_npu_gdn_config c; rk_npu_gdn_config_init(&c,tokens);
    c.chunk_size=chunk; c.core_mask=mask; c.cpu_threads=threads;
    c.precision=static_cast<rk_npu_gdn_precision>(precision);
    c.head_mapping=interleaved?RK_NPU_GDN_HEAD_INTERLEAVED:RK_NPU_GDN_HEAD_GROUPED;
    auto w=prepare(domain,c);
    Input x(tokens);
    std::vector<float> output(x.v.size()),state(SS),again(output.size()),inplace=x.state;
    const auto expected=reference(x,interleaved);
    rk_npu_gdn_timings timing{};
    auto run=[&](float* o,const float* s,float* final) {
        return rk_npu_gdn_run(w.get(),x.q.data(),x.k.data(),x.v.data(),x.g.data(),x.beta.data(),s,o,final,&timing);
    };
    CHECK(run(output.data(),x.state.data(),state.data())==0);
    close(output,expected.output,precision?1e-4:4e-3);
    close(state,expected.state,precision?1e-4:4e-3);
    CHECK(timing.hardware_us>0 && timing.total_us>=timing.hardware_us);
    CHECK(run(again.data(),inplace.data(),inplace.data())==0);
    CHECK(output==again && state==inplace);
    const auto saved=inplace;
    x.g[0]=1.f;
    CHECK(run(again.data(),inplace.data(),inplace.data())==RK_NPU_ERR_PARAM && inplace==saved);
    std::printf("PASS recurrence T=%d C=%d mask=%u cpu=%d precision=%d mapping=%d replay/inplace/oracle\n",
        tokens,chunk,mask,threads,precision,int(interleaved));
}
void fused(rk_npu_iommu_domain* domain,int precision) {
    constexpr int T=7;
    rk_npu_gdn_config c; rk_npu_gdn_config_init(&c,T);
    c.precision=static_cast<rk_npu_gdn_precision>(precision);
    auto w=prepare(domain,c);
    Input x(T);
    std::vector<float> projected(T*CW),weights(CW*4),history(CW*3),out(x.v.size()),state(SS),final_history(history.size());
    for(size_t i=0;i<projected.size();++i) projected[i]=float(int((i*17)%127)-63)*.02f;
    for(size_t i=0;i<weights.size();++i) weights[i]=float(int((i*19)%13)-6)*.0625f;
    for(size_t i=0;i<history.size();++i) history[i]=float(int((i*23)%127)-63)*.02f;
    for(int t=0;t<T;++t) {
        std::vector<float> row(CW);
        for(int channel=0;channel<CW;++channel) {
            double value=0;
            for(int tap=0;tap<4;++tap) {
                const float a=t+tap<3?history[channel*3+t+tap]:projected[(t+tap-3)*CW+channel];
                value+=double(a)*weights[channel*4+tap];
            }
            row[channel]=float(value/(1+std::exp(-value)));
        }
        for(int h=0;h<HK;++h) for(int kind=0;kind<2;++kind) {
            const float* source=row.data()+kind*HK*D+h*D;
            double sum=1e-6;
            for(int k=0;k<D;++k) sum+=double(source[k])*source[k];
            float* dest=(kind?x.k:x.q).data()+(t*HK+h)*D;
            for(int k=0;k<D;++k) dest[k]=float(source[k]/std::sqrt(sum));
        }
        std::copy(row.begin()+2*HK*D,row.end(),x.v.begin()+t*HV*D);
    }
    const auto expected=reference(x,false);
    auto run=[&](float* state_in,float* state_out,float* hist_in,float* hist_out) {
        return rk_npu_gdn_run_projected(w.get(),projected.data(),weights.data(),hist_in,x.g.data(),x.beta.data(),
            state_in,out.data(),state_out,hist_out,nullptr);
    };
    CHECK(run(x.state.data(),state.data(),history.data(),final_history.data())==0);
    close(out,expected.output,precision?1e-4:4e-3); close(state,expected.state,precision?1e-4:4e-3);
    for(int channel=0;channel<CW;++channel) for(int j=0;j<3;++j)
        CHECK(final_history[channel*3+j]==projected[(T-3+j)*CW+channel]);
    auto inplace=x.state,old_history=history;
    CHECK(run(inplace.data(),inplace.data(),old_history.data(),old_history.data())==0);
    CHECK(inplace==state && old_history==final_history);
    x.g[0]=1.f; const auto saved_state=inplace,saved_history=old_history;
    CHECK(run(inplace.data(),inplace.data(),old_history.data(),old_history.data())==RK_NPU_ERR_PARAM);
    CHECK(inplace==saved_state && old_history==saved_history);
    std::printf("PASS fused Conv/SiLU/L2 precision=%d oracle/history/inplace\n",precision);
}
void busy(rk_npu_iommu_domain* domain) {
    rk_npu_gdn_config c; rk_npu_gdn_config_init(&c,128);
    auto w=prepare(domain,c); Input x(128);
    std::atomic<int> ready{0}; std::atomic<bool> start{false};
    int rc[8]{}; std::vector<std::thread> callers;
    for(int i=0;i<8;++i) callers.emplace_back([&,i] {
        std::vector<float> out(x.v.size()),state(SS);
        ++ready; while(!start.load()) std::this_thread::yield();
        rc[i]=rk_npu_gdn_run(w.get(),x.q.data(),x.k.data(),x.v.data(),x.g.data(),x.beta.data(),x.state.data(),out.data(),state.data(),nullptr);
    });
    while(ready.load()!=8) std::this_thread::yield();
    start=true; for(auto& t:callers) t.join();
    const int successes=std::count(std::begin(rc),std::end(rc),RK_NPU_OK);
    const int rejected=std::count(std::begin(rc),std::end(rc),RK_NPU_ERR_BUSY);
    CHECK(successes>=1 && rejected>=1 && successes+rejected==8);
    std::puts("PASS concurrent workspace use returns BUSY");
}
}
int main(int argc,char** argv) try {
    metadata();
    if(argc==1) return 0;
    CHECK(argc==2 && std::string(argv[1])=="--device");
    std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close); CHECK(ctx);
    std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free); CHECK(domain);
    for(int precision:{0,1}) {
        for(uint32_t mask:{1u,2u,4u,3u,7u}) recurrent(domain.get(),7,64,mask,4,precision,false);
        recurrent(domain.get(),65,64,7,1,precision,true);
        recurrent(domain.get(),129,32,7,4,precision,false);
        recurrent(domain.get(),2048,64,7,4,precision,false);
        fused(domain.get(),precision);
    }
    recurrent(domain.get(),7,16,7,2,1,false);
    recurrent(domain.get(),7,128,7,3,1,true);
    busy(domain.get());
    auto extra=rk_npu_iommu_domain_create(ctx.get(),1);
    if(extra) {
        rk_npu_gdn_config c; rk_npu_gdn_config_init(&c,1);
        auto w=prepare(extra,c); rk_npu_iommu_domain_free(extra);
        Input x(1); std::vector<float> output(x.v.size()),state(SS);
        CHECK(rk_npu_gdn_run(w.get(),x.q.data(),x.k.data(),x.v.data(),x.g.data(),x.beta.data(),x.state.data(),output.data(),state.data(),nullptr)==0);
        close(state,reference(x,false).state,4e-3);
        std::puts("PASS retained nonzero IOMMU domain");
    } else std::puts("SKIP nonzero IOMMU domain unavailable");
    std::puts("PASS GDN device integration");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }

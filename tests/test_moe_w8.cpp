#include "rk_npu_moe_w8.h"
#include "../src/rk_npu_moe_w4_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void check(int rc){if(rc)throw std::runtime_error(rk_npu_strerror(rc));}
struct Expert {
    std::vector<int8_t> g,d;
    std::vector<float> gs,ds;
};
static float quantize(const float* x,int n,int8_t* q) {
    float maximum=0;for(int i=0;i<n;++i)maximum=std::max(maximum,std::abs(x[i]));
    const float scale=maximum?maximum/127.f:1.f,inv=1.f/scale;
    for(int i=0;i<n;++i)q[i]=int8_t(std::max(-127L,std::min(127L,std::lrint(x[i]*inv))));
    return scale;
}
static std::vector<float> oracle(int m,int h,int n,int top,int shared,
    const std::vector<Expert>& experts,const std::vector<int8_t>& input,const std::vector<float>& scales,
    const std::vector<int32_t>& ids,const std::vector<float>& routes) {
    std::vector<float> out(size_t(m)*h,0),g(2*n),hidden(n);std::vector<int8_t> q(n);
    for(int row=0;row<m;++row)for(int slot=0;slot<top+shared;++slot) {
        const auto& e=experts[slot<top?ids[row*top+slot]:experts.size()-1];
        for(int col=0;col<2*n;++col) {
            int64_t acc=0;for(int k=0;k<h;++k)acc+=int(input[size_t(row)*h+k])*int(e.g[size_t(k)*2*n+col]);
            g[col]=(float(acc)*scales[row])*e.gs[col];
        }
        for(int col=0;col<n;++col)hidden[col]=(g[col]/(1.f+std::exp(-g[col])))*g[n+col];
        const float scale=quantize(hidden.data(),n,q.data());
        for(int col=0;col<h;++col) {
            int64_t acc=0;for(int k=0;k<n;++k)acc+=int(q[k])*int(e.d[size_t(k)*h+col]);
            const float value=(float(acc)*scale)*e.ds[col];
            auto& target=out[size_t(row)*h+col];target=std::fma(value,slot<top?routes[row*top+slot]:1.f,target);
        }
    }
    return out;
}
static double relative(const std::vector<float>& got,const std::vector<float>& expected) {
    double d=0,n=0;require(got.size()==expected.size(),"output length");
    for(size_t i=0;i<got.size();++i){require(std::isfinite(got[i]),"nonfinite output");double x=double(got[i])-expected[i];d+=x*x;n+=double(expected[i])*expected[i];}
    return n?std::sqrt(d/n):(d?INFINITY:0);
}
int main(int argc,char** argv) try {
    const bool npu=argc>1 && std::string(argv[1])=="--npu-middle";
    const bool w4=argc>1 && std::string(argv[1])=="--w4";
    const auto create_weights=w4?rk_npu_moe_w4_weights_create:rk_npu_moe_w8_weights_create;
    const auto create_workspace=w4?rk_npu_moe_w4_workspace_create:rk_npu_moe_w8_workspace_create;
    auto* ctx=rk_npu_open(nullptr);require(ctx,"open NPU");
    auto* domain=rk_npu_iommu_domain_create(ctx,15);require(domain,"create domain");
    const int H=64,N=npu?512:(w4?64:32),E=3,TOP=2,MAX=256;
    rk_npu_moe_w8_config config;rk_npu_moe_w8_config_init(&config,H,N,E,1,TOP,MAX);
    config.middle=npu?RK_NPU_MOE_MIDDLE_NPU_LUT:RK_NPU_MOE_MIDDLE_CPU;
    std::vector<Expert> a(E+1),b(E+1);std::vector<rk_npu_moe_w8_expert> av(E+1),bv(E+1);
    for(int e=0;e<=E;++e)for(int variant=0;variant<2;++variant) {
        auto& x=(variant?b:a)[e];x.g.resize(size_t(H)*2*N);x.d.resize(size_t(N)*H);
        x.gs.assign(2*N,.0625f);x.ds.assign(H,.015625f);
        for(size_t i=0;i<x.g.size();++i)x.g[i]=int8_t(int((i*17+i/19+e*11+variant*3)%7)-3);
        for(size_t i=0;i<x.d.size();++i)x.d[i]=int8_t(int((i*13+i/7+e*5+variant*2)%7)-3);
        (variant?bv:av)[e]={x.g.data(),x.gs.data(),x.d.data(),x.ds.data()};
    }
    auto* wa=create_weights(domain,&config.weights,av.data());
    auto* wb=create_weights(domain,&config.weights,bv.data());
    require(wa&&wb,"create weights");
    for(int native=0;native<(w4?1:2);++native) {
        config.native_input=native;
        auto* ws=create_workspace(domain,&config);require(ws,"create workspace");
        for(int rows:{1,5,128,256,1}) {
            std::vector<int8_t> x(size_t(rows)*H);std::vector<float> scales(rows,.03125f),out(x.size(),0);
            std::vector<int32_t> ids(size_t(rows)*TOP);std::vector<float> routes(ids.size());
            for(size_t i=0;i<x.size();++i)x[i]=int8_t(int((i*7+i/5)%(w4?256:255))-(w4?128:127));
            for(int row=0;row<rows;++row)for(int slot=0;slot<TOP;++slot) {
                ids[row*TOP+slot]=(row+slot)%E;routes[row*TOP+slot]=slot?.375f:.625f;
            }
            // Duplicate indices, zero coefficients and empty experts are valid.
            ids[0]=ids[1]=0;routes[1]=0;
            for(int variant:{0,1,0}) {
                auto* w=variant?wb:wa;const auto reference=oracle(rows,H,N,TOP,1,variant?b:a,x,scales,ids,routes);
                rk_npu_moe_w8_timings timing{};
                check(rk_npu_moe_w8_run_quantized(ws,w,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),&timing));
                const double error=relative(out,reference);
                require(error<(npu?.02:1e-6),"oracle mismatch");
                auto first=out;
                check(rk_npu_moe_w8_run_quantized(ws,w,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),nullptr));
                require(out==first,"repeat mismatch");
                std::printf("native=%d rows=%d variant=%d relL2=%.8g\n",native,rows,variant,error);
            }
            const auto saved=out;ids[0]=E;
            require(rk_npu_moe_w8_run_quantized(ws,wa,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),nullptr)==RK_NPU_ERR_PARAM,"reject route");
            require(out==saved,"invalid call wrote output");ids[0]=0;
            scales[0]=0;
            require(rk_npu_moe_w8_run_quantized(ws,wa,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),nullptr)==RK_NPU_ERR_PARAM,"reject scale");
            scales[0]=1;
            std::vector<float> f(x.size(),0);
            std::vector<int8_t> fq(f.size());std::vector<float> fs(rows);
            const float anchors[]={127,-127,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};
            for(int row=0;row<rows;++row) {
                for(int col=0;col<H;++col)f[size_t(row)*H+col]=col<8?anchors[col]:float((row*13+col*7)%63-31)/8;
                fs[row]=quantize(f.data()+size_t(row)*H,H,fq.data()+size_t(row)*H);
            }
            check(rk_npu_moe_w8_run_quantized(ws,wa,rows,fq.data(),fs.data(),ids.data(),routes.data(),out.data(),nullptr));
            const auto even=out;
            check(rk_npu_moe_w8_run_f32(ws,wa,rows,f.data(),ids.data(),routes.data(),out.data(),nullptr));
            require(out==even,"FP32/nearest-even input differs from prequantized entry");
            std::fill(f.begin(),f.end(),0.f);
            check(rk_npu_moe_w8_run_f32(ws,wa,rows,f.data(),ids.data(),routes.data(),out.data(),nullptr));
            require(std::all_of(out.begin(),out.end(),[](float value){return value==0;}),"zero output");
            f[0]=std::numeric_limits<float>::quiet_NaN();
            require(rk_npu_moe_w8_run_f32(ws,wa,rows,f.data(),ids.data(),routes.data(),out.data(),nullptr)==RK_NPU_ERR_PARAM,"reject NaN");
            f[0]=0;
            require(rk_npu_moe_w8_run_f32(ws,wa,rows,f.data(),ids.data(),routes.data(),f.data(),nullptr)==RK_NPU_ERR_PARAM,"reject alias");
        }
        rk_npu_moe_w8_workspace_free(ws);
    }
    // Fewer active jobs than requested NPU cores, no shared expert.
    auto single=config;single.weights.routed_experts=1;single.weights.shared_expert=0;single.top_k=1;
    auto* one=create_weights(domain,&single.weights,av.data());
    auto* ws=create_workspace(domain,&single);require(one&&ws,"single expert handles");
    auto* other_domain=rk_npu_iommu_domain_create(ctx,14);require(other_domain,"other domain");
    auto* foreign=create_weights(other_domain,&single.weights,av.data());require(foreign,"foreign weights");
    for(int rows:{1,128,256}) {
        std::vector<int8_t> x(size_t(rows)*H,1);std::vector<float> scales(rows,.03125f),routes(rows,1),out(x.size());
        std::vector<int32_t> ids(rows,0);
        require(rk_npu_moe_w8_run_quantized(ws,foreign,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),nullptr)==RK_NPU_ERR_DOMAIN,"domain rejection");
        check(rk_npu_moe_w8_run_quantized(ws,one,rows,x.data(),scales.data(),ids.data(),routes.data(),out.data(),nullptr));
        auto reference=oracle(rows,H,N,1,0,{a[0]},x,scales,ids,routes);
        require(relative(out,reference)<(npu?.02:1e-6),"single expert oracle");
    }
    rk_npu_moe_w8_weights_free(foreign);rk_npu_iommu_domain_free(other_domain);
    rk_npu_moe_w8_workspace_free(ws);rk_npu_moe_w8_weights_free(one);
    rk_npu_moe_w8_weights_free(wb);rk_npu_moe_w8_weights_free(wa);
    rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);
    std::puts("MoE public API checks passed");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}

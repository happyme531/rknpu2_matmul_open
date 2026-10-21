#include "rk_npu_moe_w8.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool ok,const char* msg) {if(!ok)throw std::runtime_error(msg);}
static float quantize(const std::vector<float>& x,std::vector<int8_t>& q) {
    float max=0;for(float v:x)max=std::max(max,std::abs(v));
    const float scale=max?max/127.f:1.f;
    q.resize(x.size());for(size_t i=0;i<x.size();++i)q[i]=int8_t(std::clamp(std::lrint(x[i]/scale),-127L,127L));
    return scale;
}
int main() try {
    constexpr int H=2048,F=512,E=3,K=2;
    auto* ctx=rk_npu_open(nullptr);require(ctx,"open NPU");
    auto* domain=rk_npu_iommu_domain_create(ctx,15);require(domain,"create domain");
    std::vector<std::vector<int8_t>> gate(E),down(E);
    std::vector<std::vector<float>> gs(E),ds(E);
    std::vector<rk_npu_moe_w8_expert> experts(E);
    for(int e=0;e<E;++e) {
        gate[e].resize(H*2*F);down[e].resize(F*H);gs[e].resize(2*F);ds[e].resize(H);
        for(size_t i=0;i<gate[e].size();++i)gate[e][i]=int8_t(int((i*7+i/23+e*5)%15)-7);
        for(size_t i=0;i<down[e].size();++i)down[e][i]=int8_t(int((i*3+i/11+e*7)%15)-7);
        for(int i=0;i<2*F;++i)gs[e][i]=.001f*float(i%13+1);
        for(int i=0;i<H;++i)ds[e][i]=.001f*float(i%7+1);
        experts[e]={gate[e].data(),gs[e].data(),down[e].data(),ds[e].data()};
    }
    rk_npu_moe_w8_config cfg;rk_npu_moe_w8_config_init(&cfg,H,F,E,0,K,128);
    auto* weights=rk_npu_moe_quant_weights_create(domain,&cfg.weights,experts.data());
    auto* work=rk_npu_moe_quant_workspace_create(domain,&cfg);
    require(weights&&work,"create quantized FFN");
    for(int rows:{1,5,128,1}) {
        std::vector<float> input(rows*H),output(rows*H),routes(rows*K);
        std::vector<int32_t> ids(rows*K);
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        for(int r=0;r<rows;++r)for(int k=0;k<K;++k){ids[r*K+k]=(r+k)%E;routes[r*K+k]=k?.375f:.625f;}
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),ids.data(),routes.data(),output.data(),nullptr)==0,"run quantized FFN");
        double delta=0,energy=0;
        std::vector<int> samples{0};if(rows>1)samples.push_back(rows/2);if(rows>2)samples.push_back(rows-1);
        for(int row:samples) {
            std::vector<float> expected(H,0);
            for(int k=0;k<K;++k) {
                int e=ids[row*K+k];std::vector<float> x(H),g(2*F),hidden(F);std::vector<int8_t> q,hq;
                for(int i=0;i<H;++i)x[i]=input[row*H+i];
                float scale=quantize(x,q);
                for(int n=0;n<2*F;++n) {
                    int64_t acc=0;for(int i=0;i<H;++i)acc+=int(q[i])*int(gate[e][i*2*F+n]);
                    g[n]=(float(acc)*scale)*gs[e][n];
                }
                for(int n=0;n<F;++n)hidden[n]=(g[n]/(1.f+std::exp(-g[n])))*g[F+n];
                scale=quantize(hidden,hq);
                for(int n=0;n<H;++n) {
                    int64_t acc=0;for(int i=0;i<F;++i)acc+=int(hq[i])*int(down[e][i*H+n]);
                    float y=(float(acc)*scale)*ds[e][n];expected[n]=std::fma(y,routes[row*K+k],expected[n]);
                }
            }
            for(int n=0;n<H;++n){require(std::isfinite(output[row*H+n]),"nonfinite output");double d=output[row*H+n]-expected[n];delta+=d*d;energy+=double(expected[n])*expected[n];}
        }
        double error=std::sqrt(delta/std::max(energy,1e-30));
        std::printf("W4 rows=%d sampled_rows=%zu relL2=%.9g\n",rows,samples.size(),error);
        require(error<2e-5,"integer oracle mismatch");
    }
    auto one=cfg;one.top_k=1;
    auto* slots=rk_npu_moe_quant_slots_create(domain,&one.weights);
    auto* slot_work=rk_npu_moe_quant_workspace_create(domain,&one);
    require(slots&&slot_work,"create physical expert slot");
    const size_t bytes=rk_npu_moe_quant_expert_blob_bytes(weights);
    const size_t stride=(bytes+4095)/4096*4096;
    std::vector<uint8_t> blob(stride),roundtrip(stride);
    for(int expert:{0,2,1,0}) {
        const int physical=(expert+1)%E;
        require(rk_npu_moe_quant_export_expert(weights,expert,blob.data(),blob.size())==0,"export packed expert");
        require(rk_npu_moe_quant_import_expert(slots,physical,blob.data(),blob.size())==0,"import packed expert");
        require(rk_npu_moe_quant_export_expert(slots,physical,roundtrip.data(),roundtrip.size())==0,"export imported slot");
        require(blob==roundtrip,"packed expert byte roundtrip");
        const int rows=5;std::vector<float> input(rows*H),actual(rows*H),expected(rows*H);
        std::vector<int32_t> source_ids(rows*K,expert),slot_ids(rows,physical);
        std::vector<float> source_routes(rows*K,0),slot_routes(rows,1);
        for(int r=0;r<rows;++r)source_routes[r*K]=1;
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),source_ids.data(),source_routes.data(),expected.data(),nullptr)==0,"run resident expert");
        require(rk_npu_moe_w8_run_f32(slot_work,slots,rows,input.data(),slot_ids.data(),slot_routes.data(),actual.data(),nullptr)==0,"run replaced slot");
        require(actual==expected,"slot replacement changed expert arithmetic");
    }
    {
        const int rows=5;std::vector<float> input(rows*H),actual(rows*H),expected(rows*H);
        std::vector<int32_t> source_ids(rows*K),slot_ids(rows);
        std::vector<float> source_routes(rows*K,0),slot_routes(rows,1);
        for(int r=0;r<rows;++r){source_ids[r*K]=source_ids[r*K+1]=r%E;source_routes[r*K]=1;slot_ids[r]=(r%E+1)%E;}
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),source_ids.data(),source_routes.data(),expected.data(),nullptr)==0,"run resident permutation");
        require(rk_npu_moe_w8_run_f32(slot_work,slots,rows,input.data(),slot_ids.data(),slot_routes.data(),actual.data(),nullptr)==0,"run physical slot permutation");
        require(actual==expected,"physical expert slot permutation changed output");
        std::printf("W4 slot permutation checks passed\n");
    }
    require(rk_npu_moe_quant_import_expert(slots,0,blob.data(),bytes-1)==RK_NPU_ERR_PARAM,"reject short packed record");
    rk_npu_moe_w8_workspace_free(slot_work);rk_npu_moe_w8_weights_free(slots);
    rk_npu_moe_w8_workspace_free(work);rk_npu_moe_w8_weights_free(weights);
    rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);
    std::puts("Quantized MoE checks passed");return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}

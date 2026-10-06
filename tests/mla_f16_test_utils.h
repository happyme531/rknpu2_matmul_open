#pragma once
#include "rk_npu_mla_f16.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace mla_test {
namespace bits=rknpu2_matmul_open::bits;
constexpr int H=16,D=128,C=512,R=64,QK=192;
inline void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
inline void check(int rc){if(rc)throw std::runtime_error(rk_npu_strerror(rc));}
template<class T,void(*Free)(T*)>using Handle=std::unique_ptr<T,decltype(Free)>;
using Weights=Handle<rk_npu_mla_f16_weights,rk_npu_mla_f16_weights_free>;
using Cache=Handle<rk_npu_mla_f16_cache,rk_npu_mla_f16_cache_free>;
using Work=Handle<rk_npu_mla_f16_workspace,rk_npu_mla_f16_workspace_free>;
struct Device{
    rk_npu_ctx* ctx=nullptr;rk_npu_iommu_domain* domain=nullptr;
    Device(){ctx=rk_npu_open(nullptr);domain=ctx?rk_npu_iommu_domain_create(ctx,14):nullptr;require(ctx&&domain,"open/domain");}
    ~Device(){rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);}
};
struct BoardLock{
    int fd;
    explicit BoardLock(const char* p):fd(open(p,O_CREAT|O_RDWR,0666)){require(fd>=0 && !flock(fd,LOCK_EX|LOCK_NB),"NPU lock busy");}
    ~BoardLock(){flock(fd,LOCK_UN);close(fd);}
};
inline Weights weights(Device& d,const uint16_t* data){rk_npu_mla_f16_weights* p=nullptr;check(rk_npu_mla_f16_weights_create(d.domain,data,&p));return Weights(p,rk_npu_mla_f16_weights_free);}
inline Cache cache(Device& d,const rk_npu_mla_f16_config& c,rk_npu_mla_f16_weights* w){rk_npu_mla_f16_cache* p=nullptr;check(rk_npu_mla_f16_cache_create(d.domain,&c,w,&p));return Cache(p,rk_npu_mla_f16_cache_free);}
inline Work work(Device& d,const rk_npu_mla_f16_config& c,rk_npu_mla_f16_weights* w){rk_npu_mla_f16_workspace* p=nullptr;check(rk_npu_mla_f16_prepare(d.domain,&c,w,&p));return Work(p,rk_npu_mla_f16_workspace_free);}
inline std::vector<uint16_t> random_data(size_t n,unsigned seed,float sigma){
    std::mt19937 rng(seed);std::normal_distribution<float> dist(0,sigma);std::vector<uint16_t> v(n);
    for(auto& x:v)x=bits::float_to_half(dist(rng));return v;
}
inline std::vector<float> widen(const std::vector<uint16_t>& x){std::vector<float> y(x.size());for(size_t i=0;i<x.size();++i)y[i]=bits::half_to_float(x[i]);return y;}
inline float dot(const float* a,const float* b,int count){
#if defined(__aarch64__)
    auto x=vdupq_n_f32(0),y=x;for(int j=0;j<count;j+=8){x=vfmaq_f32(x,vld1q_f32(a+j),vld1q_f32(b+j));y=vfmaq_f32(y,vld1q_f32(a+j+4),vld1q_f32(b+j+4));}return vaddvq_f32(vaddq_f32(x,y));
#else
    float x=0;for(int j=0;j<count;++j)x+=a[j]*b[j];return x;
#endif
}
inline void axpy(float* a,const float* b,float p,int count){
#if defined(__aarch64__)
    for(int j=0;j<count;j+=4)vst1q_f32(a+j,vfmaq_n_f32(vld1q_f32(a+j),vld1q_f32(b+j),p));
#else
    for(int j=0;j<count;++j)a[j]+=b[j]*p;
#endif
}
struct Fixture{
    std::vector<uint16_t> w,latent,rope,q;
    std::vector<float> wf,cf,kf,qf,gates;
    Fixture(int length,int rows):w(random_data(size_t(H)*2*D*C,20261003,.045f)),
        latent(random_data(size_t(length)*C,123,.7f)),rope(random_data(size_t(length)*R,321,.7f)),
        q(random_data(size_t(rows)*H*QK,999,1.1f)),wf(widen(w)),cf(widen(latent)),kf(widen(rope)),qf(widen(q)),gates(size_t(rows)*H){
        const float logits[]={-80,-2,0,2,80};for(size_t i=0;i<gates.size();++i)gates[i]=logits[i%5];
    }
    // Independent FP32 exact-softmax oracle, using the algebraic MLA identity.
    // It does not round intermediate projections or use the NPU exp recipe.
    std::vector<float> reference(int length,int rows,int start,rk_npu_attention_f16_mask_mode mode,
        const rk_npu_attention_f16_boolean_mask* mask=nullptr,bool gated=true,int query_offset=0)const{
        std::vector<float> out(size_t(rows)*H*D),qabs(C),scores(length),average(C);
        for(int r=0;r<rows;++r)for(int h=0;h<H;++h){
            std::fill(qabs.begin(),qabs.end(),0);std::fill(average.begin(),average.end(),0);
            const auto* query=qf.data()+(size_t(query_offset+r)*H+h)*QK;
            for(int d=0;d<D;++d)axpy(qabs.data(),wf.data()+(size_t(h)*2*D+d)*C,query[d],C);
            float maximum=-INFINITY;std::vector<int> valid;
            for(int j=0;j<length;++j){
                bool keep=mode!=RK_NPU_ATTENTION_F16_CAUSAL || j<=start+r;
                if(mask){int hi=mask->heads==1?0:h,ri=mask->query_rows==1?0:mask->query_offset+r,ki=mask->key_length==1?0:j;
                    keep=mask->data[uint64_t(hi)*mask->head_stride+uint64_t(ri)*mask->query_stride+uint64_t(ki)*mask->key_stride]!=0;}
                if(!keep)continue;
                scores[j]=(dot(qabs.data(),cf.data()+size_t(j)*C,C)+dot(query+D,kf.data()+size_t(j)*R,R))/std::sqrt(float(QK));
                maximum=std::max(maximum,scores[j]);valid.push_back(j);
            }
            if(valid.empty())continue;float z=0;
            for(int j:valid){const float e=std::exp(scores[j]-maximum);z+=e;axpy(average.data(),cf.data()+size_t(j)*C,e,C);}
            const float gate=gated?1/(1+std::exp(-gates[size_t(query_offset+r)*H+h])):1;
            for(int d=0;d<D;++d)out[(size_t(r)*H+h)*D+d]=dot(average.data(),wf.data()+(size_t(h)*2*D+D+d)*C,C)/z*gate;
        }return out;
    }
};
inline double error(const std::vector<float>& a,const std::vector<float>& b){
    require(a.size()==b.size(),"output size");double diff=0,norm=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]),"nonfinite output");double d=a[i]-b[i];diff+=d*d;norm+=double(b[i])*b[i];}
    return std::sqrt(diff/std::max(norm,1e-30));
}
struct GuardOutput{
    std::vector<float> data;size_t count;
    explicit GuardOutput(size_t n):data(n+16,-98765.125f),count(n){}
    float* ptr(){return data.data()+8;}
    std::vector<float> values()const{
        for(int i=0;i<8;++i)require(data[i]==-98765.125f && data[count+8+i]==-98765.125f,"output guard");
        return {data.begin()+8,data.end()-8};
    }
};
} // namespace

#include "rk_npu_mla_f16.h"
#include "rk_npu_attention_f16_internal.h"
#include "rk_npu_matmul_f16.h"
#include <cmath>
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#define MLA_NEON 1
#endif

namespace rknpu2_matmul_open::mla {
using namespace attention;
constexpr int H=16,C=512,D=128,R=64,QK=192;
int validate(const rk_npu_mla_f16_config& c){
    const auto core=c.core_mask?c.core_mask:1;
    if((c.mode!=RK_NPU_MLA_F16_EXPANDED && c.mode!=RK_NPU_MLA_F16_ABSORBED) ||
       c.max_query_rows<1 || c.max_query_rows>(c.mode==RK_NPU_MLA_F16_EXPANDED?128:32) || c.initial_capacity<32 ||
       c.initial_capacity%32 || c.max_capacity<c.initial_capacity ||
       c.max_capacity>32768 || c.max_capacity%32 ||
       (c.kv_tile!=256 && c.kv_tile!=512 && c.kv_tile!=1024 && c.kv_tile!=2048 && c.kv_tile!=4096) ||
       (core!=1 && core!=2 && core!=3 && core!=4 && core!=7) ||
       (c.flags!=RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL && c.flags!=RK_NPU_MLA_F16_STABLE_SOFTMAX) || !std::isfinite(c.exp_shift) ||
       (c.mask_mode!=RK_NPU_ATTENTION_F16_CAUSAL && c.mask_mode!=RK_NPU_ATTENTION_F16_NO_MASK &&
        c.mask_mode!=RK_NPU_ATTENTION_F16_BOOLEAN_MASK))return RK_NPU_ERR_PARAM;
    return 0;
}
int group(const rk_npu_mla_f16_config& c,int rows=32){
    if(c.mode==RK_NPU_MLA_F16_ABSORBED && rows==1)
        return ceil_div(H,__builtin_popcount(c.core_mask?c.core_mask:1));
    return c.mode==RK_NPU_MLA_F16_ABSORBED?(rows<=16?4:2):1;
}
rk_npu_attention_f16_config engine_config(const rk_npu_mla_f16_config& c,int rows=32){
    const int groups=c.mode==RK_NPU_MLA_F16_ABSORBED && rows==1?
        __builtin_popcount(c.core_mask?c.core_mask:1):H/group(c,rows);
    return {H,groups,QK,c.max_query_rows,c.kv_tile,c.initial_capacity,c.max_capacity,
        c.core_mask,c.timeout_ms,c.flags,c.exp_shift,c.mask_mode};
}
Shape engine_shape(const rk_npu_mla_f16_config& c,int rows=32){
    return {c.mode==RK_NPU_MLA_F16_ABSORBED?C+R:QK,
        c.mode==RK_NPU_MLA_F16_ABSORBED?C:D,group(c,rows),c.mode==RK_NPU_MLA_F16_ABSORBED,
        1/std::sqrt(float(QK)),c.mode==RK_NPU_MLA_F16_ABSORBED && rows==1,
        c.mode==RK_NPU_MLA_F16_ABSORBED && rows==1?2:4};
}
template<class F>int api(F&& f)noexcept{
    try{f();return 0;}catch(Error e){return e.code;}catch(const std::bad_alloc&){return RK_NPU_ERR_NOMEM;}
    catch(...){return RK_NPU_ERR_PARAM;}
}
rk_npu_matmul_f16_config projection_config(int m,int n,int k,uint32_t core=1,uint32_t timeout=2000){
    rk_npu_matmul_f16_config c{};rk_npu_matmul_f16_config_init(&c,m,n,k,RK_NPU_FUSE_NONE);
    c.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;c.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
    c.core_mask=core;c.timeout_ms=timeout;return c;
}
struct Weights {
    Lease lease;
    std::atomic<unsigned> references{1};
    Buffer expand,absorb,value;
    virtual ~Weights()=default;
    explicit Weights(rk_npu_iommu_domain* d,const uint16_t* b):lease(d){
        std::vector<uint16_t> transposed(size_t(H)*C*2*D),wk(size_t(H)*D*C),wv(size_t(H)*C*D);
        for(int h=0;h<H;++h)for(int c=0;c<C;++c)for(int j=0;j<2*D;++j)
            transposed[(size_t(h)*C+c)*2*D+j]=b[(size_t(h)*2*D+j)*C+c];
        for(int h=0;h<H;++h){
            std::copy_n(b+size_t(h)*2*D*C,size_t(D)*C,wk.data()+size_t(h)*D*C);
            for(int c=0;c<C;++c)for(int j=0;j<D;++j)
                wv[(size_t(h)*C+c)*D+j]=b[(size_t(h)*2*D+D+j)*C+c];
        }
        auto pack=[&](Buffer& out,int n,int k,const uint16_t* data){
            auto cfg=projection_config(4,n,k);rk_npu_matmul_sizes s{};
            check(rk_npu_matmul_f16_batch_query(H,&cfg,&s));out.alloc(d,s.weight_bytes);
            check(rk_npu_matmul_f16_batch_pack_b(H,&cfg,data,&out.mem));out.sync(RK_NPU_SYNC_TO_DEVICE);
        };
        pack(expand,2*D,C,transposed.data());pack(absorb,C,D,wk.data());pack(value,D,C,wv.data());
    }
    void retain(){references.fetch_add(1,std::memory_order_relaxed);}
    void release(){if(references.fetch_sub(1,std::memory_order_acq_rel)==1)delete this;}
};
struct WeightRef {
    Weights* p;
    explicit WeightRef(Weights* w):p(w){if(p)p->retain();}
    ~WeightRef(){if(p)p->release();}
};
struct Projection {
    rk_npu_iommu_domain* domain;
    rk_npu_matmul_f16_config cfg;
    rk_npu_matmul_f16_batch_plan* plan=nullptr;
    rk_npu_matmul_f16_plan* dense_plan=nullptr;
    bool shared_input=false;
    Buffer a,out;
    bool poisoned=false;
    Projection(rk_npu_iommu_domain* d,int m,int n,int k,uint32_t core,uint32_t timeout,bool shared=false)
        :domain(d),cfg(projection_config(m,shared?H*n:n,k,core,timeout)),shared_input(shared){
        if(shared)cfg.n_tile=n;
        rk_npu_matmul_sizes s{};check(rk_npu_matmul_f16_batch_query(shared?1:H,&cfg,&s));
        a.alloc(d,s.input_bytes);out.alloc(d,s.output_bytes);
        if(shared){dense_plan=rk_npu_matmul_f16_prepare(d,&cfg);if(!dense_plan)throw Error{RK_NPU_ERR_NOMEM};}
        else {plan=rk_npu_matmul_f16_batch_prepare(d,H,&cfg);if(!plan)throw Error{RK_NPU_ERR_NOMEM};}
    }
    ~Projection(){rk_npu_matmul_f16_batch_plan_free(plan);rk_npu_matmul_f16_plan_free(dense_plan);}
    void run(const uint16_t* input,Buffer& weight){
        if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
        check(rk_npu_matmul_f16_batch_pack_a(shared_input?1:H,&cfg,input,&a.mem));a.sync(RK_NPU_SYNC_TO_DEVICE);
        try{check(shared_input?rk_npu_matmul_f16_run(domain->ctx,dense_plan,&a.mem,&weight.mem,nullptr,&out.mem):
            rk_npu_matmul_f16_batch_run(domain->ctx,plan,&a.mem,&weight.mem,&out.mem));out.sync(RK_NPU_SYNC_FROM_DEVICE);}
        catch(...){poisoned=true;throw;}
    }
    uint16_t at(int h,int row,int col)const{
        return block(h,row,col)[col%8];
    }
    const uint16_t* block(int h,int row,int col)const{
        const int per_head=shared_input?cfg.N/H:align_up(cfg.N,32);
        return out.data()+((size_t(h)*(per_head/8)+col/8)*cfg.M+row)*8;
    }
    void copy_row(int h,int row,int first,int count,uint16_t* dst)const{
        for(int j=0;j<count;j+=8)std::memcpy(dst+j,block(h,row,first+j),16);
    }
};
struct Cache {
    WeightRef weights;
    rk_npu_mla_f16_config cfg;
    attention::Cache native;
    std::atomic<bool> busy{false};
    std::unique_ptr<Projection> expansion;
    std::vector<uint16_t> a,k,v;
    Cache(rk_npu_iommu_domain* d,const rk_npu_mla_f16_config& c,Weights* w)
        :weights(w),cfg(c),native(d,engine_config(c),engine_shape(c)){}
    void inputs(const uint16_t* latent,const uint16_t* rope,int count){
        if(count<0 || (count && (!latent || !rope || !weights.p)))throw Error{RK_NPU_ERR_PARAM};
    }
    void expand(const uint16_t* latent,const uint16_t* rope,int count){
        k.resize(size_t(H)*count*QK);v.resize(size_t(H)*count*D);
        for(int first=0;first<count;first+=128){
            const int rows=std::min(128,count-first),m=align_up(rows,4);
            if(!expansion || expansion->cfg.M!=m)
                expansion=std::make_unique<Projection>(native.lease.domain,m,2*D,C,cfg.core_mask,cfg.timeout_ms,true);
            a.assign(size_t(m)*C,0);
            std::copy_n(latent+size_t(first)*C,size_t(rows)*C,a.data());
            expansion->run(a.data(),weights.p->expand);
            for(int h=0;h<H;++h)for(int r=0;r<rows;++r){
                auto* key=k.data()+(size_t(h)*count+first+r)*QK;
                expansion->copy_row(h,r,0,D,key);
                expansion->copy_row(h,r,D,D,v.data()+(size_t(h)*count+first+r)*D);
                std::copy_n(rope+size_t(first+r)*R,R,key+D);
            }
        }
    }
    void concat(const uint16_t* latent,const uint16_t* rope,int count){
        k.resize(size_t(count)*(C+R));
        for(int r=0;r<count;++r){std::copy_n(latent+size_t(r)*C,C,k.data()+size_t(r)*(C+R));
            std::copy_n(rope+size_t(r)*R,R,k.data()+size_t(r)*(C+R)+C);}
    }
    void append(const uint16_t* latent,const uint16_t* rope,int count){
        inputs(latent,rope,count);if(native.poisoned)throw Error{RK_NPU_ERR_SUBMIT};
        if(count>cfg.max_capacity-native.length)throw Error{RK_NPU_ERR_NOMEM};
        if(!count)return;
        try{
            if(cfg.mode==RK_NPU_MLA_F16_ABSORBED){
                concat(latent,rope,count);
                native.append(k.data(),latent,count);return;
            }
            expand(latent,rope,count);native.append(k.data(),v.data(),count);
        }catch(...){native.poisoned=true;throw;}
    }
    void load(const uint16_t* latent,const uint16_t* rope,int count){
        inputs(latent,rope,count);if(count>cfg.max_capacity)throw Error{RK_NPU_ERR_PARAM};
        native.reserve(count);
        if(expansion && expansion->poisoned)expansion.reset();
        try{
            if(cfg.mode==RK_NPU_MLA_F16_ABSORBED){concat(latent,rope,count);native.load(k.data(),latent,count);}
            else {expand(latent,rope,count);native.load(k.data(),v.data(),count);}
            // Large compact load staging is temporary, rather than a second
            // persistent expanded cache. Small decode staging stays reusable.
            if(count>128){std::vector<uint16_t>().swap(k);std::vector<uint16_t>().swap(v);}
        }catch(...){native.poisoned=true;throw;}
    }
};
struct Workspace {
    WeightRef weights;
    rk_npu_mla_f16_config cfg;
    attention::Workspace engine;
    std::unique_ptr<attention::Workspace> grouped;
    std::unique_ptr<attention::Workspace> decode;
    std::atomic<bool> busy{false};
    bool poisoned=false;
    std::unique_ptr<Projection> absorb,value;
    std::vector<uint16_t> input,q,latent;
    std::vector<float> attended;
    Workspace(rk_npu_iommu_domain* d,const rk_npu_mla_f16_config& c,Weights* w)
        :weights(w),cfg(c),engine(d,engine_config(c),engine_shape(c)){
        engine.full_causal_prefill=true;
        if(c.mode==RK_NPU_MLA_F16_ABSORBED)
            grouped=std::make_unique<attention::Workspace>(d,engine_config(c,16),engine_shape(c,16));
        if(c.mode==RK_NPU_MLA_F16_ABSORBED)
            decode=std::make_unique<attention::Workspace>(d,engine_config(c,1),engine_shape(c,1));
        if(grouped)grouped->full_causal_prefill=true;
    }
    void validate_call(Cache& c,const uint16_t* queries,int rows,int start,float* output,
        const rk_npu_attention_f16_boolean_mask* mask,int extra=0){
        compatible(engine,c.native);
        if(weights.p!=c.weights.p)throw Error{RK_NPU_ERR_PARAM};
        if(!queries || !output || rows<1 || rows>cfg.max_query_rows || start<0 ||
           c.native.length+extra<1 || (cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL &&
           start>c.native.length+extra-rows))throw Error{RK_NPU_ERR_PARAM};
        validate_boolean_mask(engine.cfg,mask,rows,c.native.length+extra);
        if(poisoned || engine.poisoned || (grouped && grouped->poisoned) ||
           (decode && decode->poisoned) || c.native.poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    }
    rk_npu_mla_f16_timings run(Cache& c,const uint16_t* queries,const float* gates,
        int rows,int start,float* output,const rk_npu_attention_f16_boolean_mask* mask){
        validate_call(c,queries,rows,start,output,mask);rk_npu_mla_f16_timings t{};auto begin=Clock::now();
        try{
            const bool compressed=cfg.mode==RK_NPU_MLA_F16_ABSORBED;
            const int m=align_up(rows,4),qd=compressed?C+R:QK,vd=compressed?C:D;
            q.resize(size_t(H)*rows*qd);attended.resize(size_t(H)*rows*vd);
            if(compressed){
                if(!absorb || absorb->cfg.M!=m){
                    absorb=std::make_unique<Projection>(engine.lease.domain,m,C,D,cfg.core_mask,cfg.timeout_ms);
                    value=std::make_unique<Projection>(engine.lease.domain,m,D,C,cfg.core_mask,cfg.timeout_ms);
                }
                input.assign(size_t(H)*m*D,0);
                for(int h=0;h<H;++h)for(int r=0;r<rows;++r)
                    std::copy_n(queries+(size_t(r)*H+h)*QK,D,input.data()+(size_t(h)*m+r)*D);
                absorb->run(input.data(),weights.p->absorb);
                for(int h=0;h<H;++h)for(int r=0;r<rows;++r){
                    absorb->copy_row(h,r,0,C,q.data()+(size_t(h)*rows+r)*qd);
                    std::copy_n(queries+(size_t(r)*H+h)*QK+D,R,q.data()+(size_t(h)*rows+r)*qd+C);
                }
            }else for(int h=0;h<H;++h)for(int r=0;r<rows;++r)
                std::copy_n(queries+(size_t(r)*H+h)*QK,QK,q.data()+(size_t(h)*rows+r)*QK);
            auto& selected=compressed && rows==1?*decode:compressed && rows<=16?*grouped:engine;
            auto projected=Clock::now();t.attention=selected.run(c.native,q.data(),rows,start,attended.data(),mask);
            auto computed=Clock::now();
            if(compressed){
                latent.assign(size_t(H)*m*C,0);
                for(int h=0;h<H;++h)for(int r=0;r<rows;++r)for(int j=0;j<C;j+=8){
                    auto* dst=latent.data()+(size_t(h)*m+r)*C+j;
                    const auto* src=attended.data()+(size_t(h)*rows+r)*C+j;
#ifdef MLA_NEON
                    vst1q_u16(dst,vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vld1q_f32(src)),vcvt_f16_f32(vld1q_f32(src+4)))));
#else
                    for(int lane=0;lane<8;++lane)dst[lane]=half(src[lane]);
#endif
                }
                value->run(latent.data(),weights.p->value);
            }
            auto valued=Clock::now();
            for(int r=0;r<rows;++r)for(int h=0;h<H;++h){
                float gate=1;
                if(gates){const float x=gates[size_t(r)*H+h];const float e=std::exp(-std::abs(x));gate=x>=0?1/(1+e):e/(1+e);}
                for(int j=0;j<D;j+=8){auto* dst=output+(size_t(r)*H+h)*D+j;
#ifdef MLA_NEON
                    float32x4_t lo,hi;
                    if(compressed){const auto x=vreinterpretq_f16_u16(vld1q_u16(value->block(h,r,j)));lo=vcvt_f32_f16(vget_low_f16(x));hi=vcvt_f32_f16(vget_high_f16(x));}
                    else {const auto* src=attended.data()+(size_t(h)*rows+r)*D+j;lo=vld1q_f32(src);hi=vld1q_f32(src+4);}
                    vst1q_f32(dst,vmulq_n_f32(lo,gate));vst1q_f32(dst+4,vmulq_n_f32(hi,gate));
#else
                    for(int lane=0;lane<8;++lane)dst[lane]=gate*(compressed?widen(value->at(h,r,j+lane)):attended[(size_t(h)*rows+r)*D+j+lane]);
#endif
                }
            }
            auto done=Clock::now();t.query_projection_us=elapsed(begin,projected);t.attention_us=elapsed(projected,computed);
            t.value_projection_us=elapsed(computed,valued);t.gate_output_us=elapsed(valued,done);t.total_us=elapsed(begin,done);return t;
        }catch(...){poisoned=true;throw;}
    }
};
void domain(rk_npu_iommu_domain* d,Weights* w){
    if(!d || !w || d->ctx!=w->lease.domain->ctx)throw Error{RK_NPU_ERR_PARAM};
    if(d->id!=w->lease.domain->id)throw Error{RK_NPU_ERR_DOMAIN};
}
} // namespace
struct rk_npu_mla_f16_weights : rknpu2_matmul_open::mla::Weights {using Weights::Weights;};
struct rk_npu_mla_f16_cache : rknpu2_matmul_open::mla::Cache {using Cache::Cache;};
struct rk_npu_mla_f16_workspace : rknpu2_matmul_open::mla::Workspace {using Workspace::Workspace;};
using namespace rknpu2_matmul_open::mla;
extern "C" {
void rk_npu_mla_f16_config_init(rk_npu_mla_f16_config* c){
    if(c)*c={RK_NPU_MLA_F16_EXPANDED,32,512,128,32768,1,2000,0,4,RK_NPU_ATTENTION_F16_CAUSAL};
}
int rk_npu_mla_f16_query(const rk_npu_mla_f16_config* c,int capacity,int rows,int length,rk_npu_mla_f16_sizes* out){
    if(!c || !out || validate(*c))return RK_NPU_ERR_PARAM;
    rk_npu_attention_f16_sizes s{};const int rc=query_engine(engine_config(*c,rows),engine_shape(*c,rows),capacity,rows,length,s);
    if(rc)return rc;*out={s.key_bytes,s.value_bytes,s.tail_bytes,s.score_bytes,s.mask_bytes,s.partial_bytes,
        size_t(H)*rows*D*4,s.compute_length,s.physical_rows,s.tasks};return 0;
}
int rk_npu_mla_f16_weights_create(rk_npu_iommu_domain* d,const uint16_t* b,rk_npu_mla_f16_weights** out){
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;if(!d || !b)return RK_NPU_ERR_PARAM;
    return api([&]{*out=new rk_npu_mla_f16_weights(d,b);});
}
void rk_npu_mla_f16_weights_free(rk_npu_mla_f16_weights* w){if(w)w->release();}
int rk_npu_mla_f16_cache_create(rk_npu_iommu_domain* d,const rk_npu_mla_f16_config* c,rk_npu_mla_f16_weights* w,rk_npu_mla_f16_cache** out){
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;if(!c || validate(*c))return RK_NPU_ERR_PARAM;
    return api([&]{if(w)domain(d,w);else if(!d || c->mode!=RK_NPU_MLA_F16_EXPANDED)throw Error{RK_NPU_ERR_PARAM};
        *out=new rk_npu_mla_f16_cache(d,*c,w);});
}
void rk_npu_mla_f16_cache_free(rk_npu_mla_f16_cache* c){delete c;}
int rk_npu_mla_f16_cache_get_info(rk_npu_mla_f16_cache* c,rk_npu_mla_f16_cache_info* out){
    if(!c || !out)return RK_NPU_ERR_PARAM;return api([&]{Lock l(c->busy);auto& n=c->native;
        *out={n.length,n.capacity,n.cfg.max_capacity,n.key.mem.size,n.value.mem.size,n.tail.mem.size};});
}
int rk_npu_mla_f16_cache_reserve(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* c,int capacity){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->native.lease.domain);Lock l(c->busy);c->native.reserve(capacity);});
}
int rk_npu_mla_f16_cache_load(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* c,const uint16_t* latent,const uint16_t* rope,int count){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->native.lease.domain);Lock l(c->busy);c->load(latent,rope,count);});
}
int rk_npu_mla_f16_cache_append(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* c,const uint16_t* latent,const uint16_t* rope,int count){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->native.lease.domain);Lock l(c->busy);c->append(latent,rope,count);});
}
int rk_npu_mla_f16_cache_load_expanded(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* c,const uint16_t* k,const uint16_t* v,int count){
    if(!c || c->cfg.mode!=RK_NPU_MLA_F16_EXPANDED)return RK_NPU_ERR_PARAM;
    return api([&]{context(ctx,c->native.lease.domain);Lock l(c->busy);c->native.load(k,v,count);});
}
int rk_npu_mla_f16_cache_append_expanded(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* c,const uint16_t* k,const uint16_t* v,int count){
    if(!c || c->cfg.mode!=RK_NPU_MLA_F16_EXPANDED)return RK_NPU_ERR_PARAM;
    return api([&]{context(ctx,c->native.lease.domain);Lock l(c->busy);c->native.append(k,v,count);});
}
int rk_npu_mla_f16_prepare(rk_npu_iommu_domain* d,const rk_npu_mla_f16_config* c,rk_npu_mla_f16_weights* w,rk_npu_mla_f16_workspace** out){
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;if(!c || validate(*c))return RK_NPU_ERR_PARAM;
    return api([&]{if(w)domain(d,w);else if(!d || c->mode!=RK_NPU_MLA_F16_EXPANDED)throw Error{RK_NPU_ERR_PARAM};
        *out=new rk_npu_mla_f16_workspace(d,*c,w);});
}
void rk_npu_mla_f16_workspace_free(rk_npu_mla_f16_workspace* w){delete w;}
int rk_npu_mla_f16_run(rk_npu_ctx* ctx,rk_npu_mla_f16_workspace* w,rk_npu_mla_f16_cache* c,const uint16_t* q,const float* gates,
    int rows,int start,float* out,rk_npu_mla_f16_timings* t,const rk_npu_attention_f16_boolean_mask* mask){
    if(!w || !c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,w->engine.lease.domain);Lock a(w->busy),b(c->busy);
        const auto result=w->run(*c,q,gates,rows,start,out,mask);if(t)*t=result;});
}
int rk_npu_mla_f16_decode_step(rk_npu_ctx* ctx,rk_npu_mla_f16_workspace* w,rk_npu_mla_f16_cache* c,const uint16_t* q,const float* gates,
    const uint16_t* latent,const uint16_t* rope,float* out,rk_npu_mla_f16_timings* t,const rk_npu_attention_f16_boolean_mask* mask){
    if(!w || !c || !latent || !rope)return RK_NPU_ERR_PARAM;
    return api([&]{context(ctx,w->engine.lease.domain);Lock a(w->busy),b(c->busy);const int start=c->native.length;
        w->validate_call(*c,q,1,start,out,mask,1);if(start>=c->cfg.max_capacity)throw Error{RK_NPU_ERR_NOMEM};
        auto begin=Clock::now();c->append(latent,rope,1);auto appended=Clock::now();
        try{auto result=w->run(*c,q,gates,1,start,out,mask);result.kv_update_us=elapsed(begin,appended);
            result.total_us=elapsed(begin,Clock::now());if(t)*t=result;}catch(...){c->native.poisoned=true;throw;}
    });
}
} // extern C

#include "rk_npu_attention_f16.h"
#include "rk_npu_quant_matmul.h"
#include "rk_npu_matmul_f16.h"
#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_attention_f16_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif
namespace bits=rknpu2_matmul_open::bits;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
void check(int rc){if(rc)throw std::runtime_error(rk_npu_strerror(rc));}
using Cache=std::unique_ptr<rk_npu_attention_f16_cache,decltype(&rk_npu_attention_f16_cache_free)>;
using Work=std::unique_ptr<rk_npu_attention_f16_workspace,decltype(&rk_npu_attention_f16_workspace_free)>;
struct Device{
    rk_npu_ctx* ctx;rk_npu_iommu_domain* domain;
    Device():ctx(rk_npu_open(nullptr)),domain(ctx?rk_npu_iommu_domain_create(ctx,14):nullptr){require(ctx&&domain,"open/domain");}
    ~Device(){rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);}
};
struct Lock{int fd;explicit Lock(const char* p):fd(open(p,O_CREAT|O_RDWR,0666)){require(fd>=0 && !flock(fd,LOCK_EX|LOCK_NB),"NPU lock busy");}~Lock(){flock(fd,LOCK_UN);close(fd);}};
Cache cache(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c){rk_npu_attention_f16_cache* p=nullptr;check(rk_npu_attention_f16_cache_create(d,&c,&p));return Cache(p,rk_npu_attention_f16_cache_free);}
Work work(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c){rk_npu_attention_f16_workspace* p=nullptr;check(rk_npu_attention_f16_prepare(d,&c,&p));return Work(p,rk_npu_attention_f16_workspace_free);}
std::vector<uint16_t> random_data(size_t n,int seed,float sigma=.7f){std::mt19937 rng(seed);std::normal_distribution<float> dist(0,sigma);std::vector<uint16_t> v(n);for(auto& x:v)x=bits::float_to_half(dist(rng));return v;}
std::vector<uint16_t> prefix(const std::vector<uint16_t>& full,int stride,int n){std::vector<uint16_t> out(size_t(8)*n*128);if(n)for(int h=0;h<8;++h)std::copy_n(full.data()+size_t(h)*stride*128,size_t(n)*128,out.data()+size_t(h)*n*128);return out;}
std::vector<float> reference(const std::vector<uint16_t>& q,const std::vector<uint16_t>& k,const std::vector<uint16_t>& v,
    int rows,int length,int start,rk_npu_attention_f16_mask_mode mode,const rk_npu_attention_f16_boolean_mask* mask,int heads=32){
    std::vector<float> out(size_t(heads)*rows*128),scores(length),qf(128);
    for(int h=0;h<heads;++h)for(int r=0;r<rows;++r){
        for(int c=0;c<128;++c)qf[c]=bits::half_to_float(q[(size_t(h)*rows+r)*128+c])/std::sqrt(128.f);
        float mx=-INFINITY;std::vector<int> valid;
        for(int j=0;j<length;++j){
            bool keep=mode==RK_NPU_ATTENTION_F16_CAUSAL?j<=start+r:true;
            if(mask){int hi=mask->heads==1?0:h,ri=mask->query_rows==1?0:mask->query_offset+r,ki=mask->key_length==1?0:j;
                keep=mask->data[uint64_t(hi)*mask->head_stride+uint64_t(ri)*mask->query_stride+uint64_t(ki)*mask->key_stride]!=0;}
            if(!keep)continue;float sum=0;const auto* src=k.data()+(size_t(h/4)*length+j)*128;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            auto a=vdupq_n_f32(0),b=a;
            for(int c=0;c<128;c+=8){auto x=vreinterpretq_f16_u16(vld1q_u16(src+c));a=vfmaq_f32(a,vld1q_f32(qf.data()+c),vcvt_f32_f16(vget_low_f16(x)));b=vfmaq_f32(b,vld1q_f32(qf.data()+c+4),vcvt_f32_f16(vget_high_f16(x)));}
            sum=vaddvq_f32(vaddq_f32(a,b));
#else
            for(int c=0;c<128;++c)sum+=qf[c]*bits::half_to_float(src[c]);
#endif
            scores[j]=sum;mx=std::max(mx,sum);valid.push_back(j);
        }
        if(valid.empty())continue;float z=0;auto* dst=out.data()+(size_t(h)*rows+r)*128;
        for(int j:valid){float e=std::exp(scores[j]-mx);z+=e;const auto* src=v.data()+(size_t(h/4)*length+j)*128;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            for(int c=0;c<128;c+=8){auto x=vreinterpretq_f16_u16(vld1q_u16(src+c));vst1q_f32(dst+c,vfmaq_n_f32(vld1q_f32(dst+c),vcvt_f32_f16(vget_low_f16(x)),e));vst1q_f32(dst+c+4,vfmaq_n_f32(vld1q_f32(dst+c+4),vcvt_f32_f16(vget_high_f16(x)),e));}
#else
            for(int c=0;c<128;++c)dst[c]+=e*bits::half_to_float(src[c]);
#endif
        }
        for(int c=0;c<128;++c)dst[c]/=z;
    }return out;
}
double error(const std::vector<float>& a,const std::vector<float>& b){double d=0,n=0;require(a.size()==b.size(),"length mismatch");for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]),"nonfinite output");double x=a[i]-b[i];d+=x*x;n+=double(b[i])*b[i];}return std::sqrt(d/std::max(n,1e-30));}
struct RestoreEnv {
    const char* key;
    bool present=std::getenv(key)!=nullptr;
    std::string saved=present?std::getenv(key):"";
    ~RestoreEnv(){if(present)setenv(key,saved.c_str(),1);else unsetenv(key);}
};
void dirty_policy_cpu(){
    using rknpu2_matmul_open::attention::experimental_dirty_sync_policy;
    RestoreEnv groups{"RK_NPU_ATTN_DIRTY_SYNC_GROUPS"},after{"RK_NPU_ATTN_DIRTY_SYNC_AFTER"};
    require(!unsetenv(groups.key) && !unsetenv(after.key),"clear dirty sync controls");
    require(!experimental_dirty_sync_policy().for_length(32768),"dirty sync default off");
    for(const char* value:{"1","2","4","span"}){
        require(!setenv(groups.key,value,1),"set groups");auto p=experimental_dirty_sync_policy();
        require(p.for_length(1)==(value[0]=='s'?32:value[0]-'0'),"legacy always-on groups");
    }
    require(!unsetenv(groups.key) && !setenv(after.key,"8192",1),"set threshold only");
    auto captured=experimental_dirty_sync_policy();
    require(captured.groups==1 && !captured.for_length(8191) && !captured.for_length(8192) && captured.for_length(8193)==1,"strict logical length threshold");
    require(!setenv(groups.key,"span",1) && !setenv(after.key,"32",1),"change future policy");
    auto p=experimental_dirty_sync_policy();require(!p.for_length(32) && p.for_length(33)==32,"threshold and span");
    require(captured.after==8192 && captured.groups==1,"policy captures environment once");
    require(!setenv(groups.key,"0",1),"explicit disable");require(!experimental_dirty_sync_policy().for_length(32768),"zero overrides threshold");
    for(const char* value:{"", "-1", "+1", " 1", "1 ", "1x", "0x20", "1.5", "2147483648", "999999999999999999999999"}){
        require(!setenv(after.key,value,1) && !setenv(groups.key,"1",1),"set invalid threshold");
        require(!experimental_dirty_sync_policy().for_length(32768),"invalid threshold fails closed");
    }
    require(!unsetenv(groups.key) && !setenv(after.key,"0",1),"zero threshold");
    p=experimental_dirty_sync_policy();require(!p.for_length(0) && p.for_length(1)==1,"zero threshold means nonempty cache");
    require(!setenv(after.key,"2147483647",1),"largest threshold");require(!experimental_dirty_sync_policy().for_length(32768),"threshold beyond capacity");
    require(!setenv(after.key,"32",1),"valid threshold");
    for(const char* value:{"", "invalid", "8", "01"}){
        require(!setenv(groups.key,value,1),"invalid groups");require(!experimental_dirty_sync_policy().for_length(32768),"invalid groups override threshold default");
    }
    std::puts("attention dirty sync environment/threshold CPU contract PASS");
}
void cpu(){
    dirty_policy_cpu();
    // Check every dirty stripe (including denominator), base offsets, bounds,
    // merging and K32 edges without a device. Widths cover compact tails too.
    using rknpu2_matmul_open::attention::dirty_value_ranges;
    for(int channels:{128,512})for(int width:{32,64,96,256,512,1024,2048,4096})
        for(int groups:{1,2,4,32})for(int first:{0,1,15,31,width/2,width-1})
            for(bool merged_denominator:{false,true})
            for(int end:{first+1,std::min(width,first+33),width}){
                const uint64_t base=4096,bytes=uint64_t(channels+32)*width*2;
                std::vector<uint8_t> covered(bytes);int calls=0;
                dirty_value_ranges(base,channels,width,first,end,groups,[&](uint64_t offset,uint64_t count){
                    require(offset>=base && offset-base+count<=bytes && offset%64==0 && count%64==0,"dirty sync range bounds/alignment");
                    std::fill_n(covered.begin()+offset-base,count,1);++calls;
                },merged_denominator);
                require(calls==(merged_denominator && groups>=channels/16?1:(channels/16+groups-1)/groups+(end+31)/32-first/32),"dirty sync call count");
                for(int c=0;c<=channels;++c)for(int pos=first;pos<end;++pos){
                    const uint64_t offset=((uint64_t(c/16)*(width/32)+pos/32)*16+c%16)*32*2+pos%32*2;
                    require(covered[offset] && covered[offset+1],"dirty sync coverage");
                }
            }
    std::puts("attention dirty stripe CPU coverage PASS");
    rk_npu_attention_f16_config c;rk_npu_attention_f16_config_init(&c);rk_npu_attention_f16_sizes s{};
    require(rk_npu_attention_f16_query(&c,8192,1,4096,&s)==RK_NPU_ERR_PARAM,"explicit approximation opt-in");
    c.flags=RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL;
    check(rk_npu_attention_f16_query(&c,8192,1,129,&s));require(s.compute_length==160 && s.physical_rows==4,"active length vs capacity");
    c.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;check(rk_npu_attention_f16_query(&c,8192,5,129,&s));require(!s.mask_bytes && s.physical_rows==20,"no-mask sizing");
    for(int length:{4095,4096,4097,8193,16385,32768}){
        check(rk_npu_attention_f16_query(&c,32768,1,length,&s));const int span=(length+31)/32*32;
        require(s.tasks==8*((span+4095)/4096+(span+511)/512),"long QK/PV task sizing");
    }
    c.kv_heads=32;c.query_heads=128;c.kv_tile=256;
    require(rk_npu_attention_f16_query(&c,32768,1,32768,&s)==RK_NPU_ERR_PARAM,"long task-count limit");
    c.kv_heads=8;c.query_heads=32;c.kv_tile=512;
    for(int core:{1,2,3,4,7}){c.core_mask=core;check(rk_npu_attention_f16_query(&c,8192,32,4096,&s));}
    for(int core:{5,6,8}){c.core_mask=core;require(rk_npu_attention_f16_query(&c,8192,1,129,&s)==RK_NPU_ERR_PARAM,"unsupported core mask");}
    c.core_mask=7;c.kv_heads=2;c.query_heads=8;
    require(rk_npu_attention_f16_query(&c,8192,1,129,&s)==RK_NPU_ERR_PARAM,"one KV head per core");
    c.kv_heads=8;c.query_heads=32;
    c.core_mask=1;c.head_dim=64;require(rk_npu_attention_f16_query(&c,8192,1,129,&s)==RK_NPU_ERR_PARAM,"shape gate");
    rk_npu_attention_f16_workspace* ws=reinterpret_cast<rk_npu_attention_f16_workspace*>(1);
    require(rk_npu_attention_f16_prepare(nullptr,&c,&ws)==RK_NPU_ERR_PARAM && !ws,"create failure output");
    std::puts("attention CPU contract PASS");
}
struct Linear{
    rk_npu_matmul_workspace* ws;rk_npu_i8i8i32_weights* weights;
    std::vector<int8_t> a;std::vector<int32_t> output;
    Linear(rk_npu_iommu_domain* d,int core):a(4*128,1),output(4*32){
        rk_npu_matmul_strategy s{};s.op_kind=RK_NPU_MATMUL_I8I8I32;s.M=4;s.N=32;s.K=128;s.k_tile=128;s.n_tile=32;
        s.a_layout=RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;s.wave_count=1;s.n_groups=1;s.npu_core_mask=core;s.cpu_core_mask=1ull<<4;s.cpu_threads=1;
        ws=rk_npu_matmul_workspace_create(d,&s);rk_npu_matmul_weight_config cfg{128,32,128};std::vector<int8_t> b(128*32,1);
        weights=rk_npu_i8i8i32_weights_create(d,&cfg,b.data());require(ws&&weights,"linear prepare");
    }
    ~Linear(){rk_npu_matmul_workspace_free(ws);rk_npu_i8i8i32_weights_free(weights);}
    void run(){check(rk_npu_i8i8i32_run(ws,weights,a.data(),output.data()));require(std::all_of(output.begin(),output.end(),[](int32_t x){return x==128;}),"linear mixed-state mismatch");}
};
struct Linears{
    std::vector<std::unique_ptr<Linear>> per_core;
    Linears(rk_npu_iommu_domain* d,int mask){for(int core:{1,2,4})if(mask&core)per_core.emplace_back(new Linear(d,core));}
    void run(){for(auto& x:per_core)x->run();}
};
void mixed_canary(int mask,int width){
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;Linears linear(dev.domain,mask);
    linear.run();linear.run();std::puts("CHECK standalone repeated INT8 PASS");std::fflush(stdout);
    for(auto shape:{std::make_pair(128,32),std::make_pair(128,width),std::make_pair(512,160)}){
        const int k=shape.first,n=shape.second;
        for(int core:{1,2,4})if(mask&core){
            rk_npu_matmul_f16_config cfg;rk_npu_matmul_f16_config_init(&cfg,4,n,k,RK_NPU_FUSE_NONE);
            cfg.core_mask=core;cfg.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;cfg.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
            rk_npu_matmul_sizes s{};check(rk_npu_matmul_f16_query(&cfg,&s));rk_npu_mem a{},b{},out{};
            check(rk_npu_mem_alloc(dev.domain,s.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&a));check(rk_npu_mem_alloc(dev.domain,s.weight_bytes,RK_NPU_MEM_DATA_DEFAULT,&b));check(rk_npu_mem_alloc(dev.domain,s.output_bytes,RK_NPU_MEM_DATA_DEFAULT,&out));
            std::vector<uint16_t> av(size_t(cfg.M)*cfg.K,0x3000),bv(size_t(cfg.K)*cfg.N,0x3400),ov(size_t(cfg.M)*cfg.N);
            check(rk_npu_matmul_f16_pack_a(&cfg,av.data(),&a));check(rk_npu_matmul_f16_pack_b(&cfg,bv.data(),&b));
            check(rk_npu_mem_sync(dev.ctx,&a,RK_NPU_SYNC_TO_DEVICE));check(rk_npu_mem_sync(dev.ctx,&b,RK_NPU_SYNC_TO_DEVICE));
            auto* plan=rk_npu_matmul_f16_prepare(dev.domain,&cfg);require(plan,"FP16 canary prepare");
            check(rk_npu_matmul_f16_run(dev.ctx,plan,&a,&b,nullptr,&out));check(rk_npu_mem_sync(dev.ctx,&out,RK_NPU_SYNC_FROM_DEVICE));
            check(rk_npu_matmul_f16_unpack_d(&cfg,&out,ov.data()));require(std::all_of(ov.begin(),ov.end(),[&](uint16_t x){return x==bits::float_to_half(k/32.f);}),"FP16 canary output");
            rk_npu_matmul_f16_plan_free(plan);rk_npu_mem_free(dev.ctx,&out);rk_npu_mem_free(dev.ctx,&b);rk_npu_mem_free(dev.ctx,&a);
        }
        std::printf("BEGIN INT8 after ordinary native FP16 K=%d N=%d\n",k,n);std::fflush(stdout);linear.run();linear.run();
        std::printf("CHECK INT8 after ordinary native FP16 K=%d N=%d PASS\n",k,n);std::fflush(stdout);
    }
}
void attention_mixed_canary(int core,int length,bool baseline){
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;Linears linear(dev.domain,core);
    linear.run();std::puts("CHECK INT8 before cache PASS");std::fflush(stdout);
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);cfg.flags=1;cfg.core_mask=core;cfg.max_capacity=32768;
    auto c=cache(dev.domain,cfg);auto w=work(dev.domain,cfg);
    auto k=random_data(size_t(8)*length*128,910),v=random_data(k.size(),911,.5f),q=random_data(32*128,921);
    check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),k.data(),v.data(),length));
    linear.run();std::puts("CHECK INT8 after cache load PASS");std::fflush(stdout);
    std::vector<float> out(q.size());check(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),q.data(),1,length-1,out.data(),nullptr,nullptr));
    std::printf("CHECK GQA core=%d L=%d PASS; BEGIN INT8\n",core,length);std::fflush(stdout);linear.run();
    std::puts("CHECK INT8 after GQA PASS");std::fflush(stdout);
    if(baseline){cfg.core_mask=1;auto single=work(dev.domain,cfg);check(rk_npu_attention_f16_run(dev.ctx,single.get(),c.get(),q.data(),1,length-1,out.data(),nullptr,nullptr));
        std::puts("CHECK single-core GQA baseline PASS; BEGIN INT8");std::fflush(stdout);linear.run();std::puts("CHECK INT8 after single-core baseline PASS");}
}
void board(int core){
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);cfg.flags=1;cfg.core_mask=core;cfg.max_capacity=8192;
    auto a=cache(dev.domain,cfg),b=cache(dev.domain,cfg);auto causal=work(dev.domain,cfg);
    auto none_cfg=cfg;none_cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;auto none=work(dev.domain,none_cfg);
    auto bool_cfg=cfg;bool_cfg.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;auto boolean=work(dev.domain,bool_cfg);Linears linear(dev.domain,core);
    auto single_cfg=cfg;single_cfg.core_mask=1;auto single_causal=work(dev.domain,single_cfg);
    single_cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;auto single_none=work(dev.domain,single_cfg);
    single_cfg.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;auto single_boolean=work(dev.domain,single_cfg);
    for(int length:{33,129,4096,4097}){
        auto k=random_data(size_t(8)*length*128,10+length),v=random_data(k.size(),20+length,.5f);
        check(rk_npu_attention_f16_cache_load(dev.ctx,a.get(),k.data(),v.data(),length));
        for(int rows:std::vector<int>{1,5,32}){
            int start=length<200?0:length-rows;auto q=random_data(size_t(32)*rows*128,rows+length);
            for(auto pair:{std::make_pair(causal.get(),cfg.mask_mode),std::make_pair(none.get(),none_cfg.mask_mode)}){
                std::vector<float> out(q.size());rk_npu_attention_f16_timings t{};linear.run();
                check(rk_npu_attention_f16_run(dev.ctx,pair.first,a.get(),q.data(),rows,start,out.data(),&t,nullptr));
                auto expected=reference(q,k,v,rows,length,start,pair.second,nullptr);double e=error(out,expected);require(e<.15,"attention reference mismatch");
                require(t.compute_length==((pair.second==RK_NPU_ATTENTION_F16_CAUSAL?start+rows:length)+31)/32*32,"compute prefix");
                if(core==3 || core==7){std::vector<float> baseline(q.size());
                    auto* sw=pair.second==RK_NPU_ATTENTION_F16_CAUSAL?single_causal.get():single_none.get();
                    check(rk_npu_attention_f16_run(dev.ctx,sw,a.get(),q.data(),rows,start,baseline.data(),nullptr,nullptr));
                    require(out==baseline,"single/multicore bitexact");}
                auto saved=out;linear.run();check(rk_npu_attention_f16_run(dev.ctx,pair.first,a.get(),q.data(),rows,start,out.data(),nullptr,nullptr));require(out==saved,"linear/attention stability");
                check(rk_npu_attention_f16_cache_reserve(dev.ctx,a.get(),8192));
                check(rk_npu_attention_f16_run(dev.ctx,pair.first,a.get(),q.data(),rows,start,out.data(),nullptr,nullptr));require(out==saved,"reserve moved KV incorrectly");
                std::printf("core=%d length=%d rows=%d mode=%d reference=%.8f compute=%d PASS\n",core,length,rows,int(pair.second),e,t.compute_length);
            }
        }
    }
    const int length=129,rows=5;auto k=random_data(size_t(8)*length*128,500),v=random_data(k.size(),501,.5f),q=random_data(32*rows*128,502);
    check(rk_npu_attention_f16_cache_load(dev.ctx,a.get(),k.data(),v.data(),length));
    std::vector<uint8_t> data(size_t(32)*7*length*2,0x55);
    for(int h=0;h<32;++h)for(int r=0;r<7;++r)for(int j=0;j<length;++j)data[((size_t(h)*7+r)*length+j)*2]=uint8_t(r!=1 && ((j+h+r)%3!=0));
    rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,data.data(),data.size(),32,7,length);
    mask.head_stride*=2;mask.query_stride*=2;mask.key_stride=2;mask.query_offset=1;mask.version=1;
    auto verify_bool=[&](rk_npu_attention_f16_boolean_mask* m){std::vector<float> out(q.size());check(rk_npu_attention_f16_run(dev.ctx,boolean.get(),a.get(),q.data(),rows,17,out.data(),nullptr,m));require(error(out,reference(q,k,v,rows,length,17,RK_NPU_ATTENTION_F16_BOOLEAN_MASK,m))<.15,"boolean mask semantics");
        if(core==3 || core==7){std::vector<float> baseline(q.size());check(rk_npu_attention_f16_run(dev.ctx,single_boolean.get(),a.get(),q.data(),rows,17,baseline.data(),nullptr,m));require(out==baseline,"boolean single/multicore bitexact");}return out;};
    auto first=verify_bool(&mask);for(int h=0;h<32;++h)for(int c=0;c<128;++c)require(first[(size_t(h)*rows)*128+c]==0,"all-hidden row must be zero");
    require(verify_bool(&mask)==first,"boolean version reuse");data[size_t(length)*2]=1;++mask.version;require(verify_bool(&mask)!=first,"boolean version update");
    std::vector<uint8_t> token(length);for(int j=0;j<length;++j)token[j]=j%3!=1;
    rk_npu_attention_f16_boolean_mask_init(&mask,token.data(),token.size(),1,1,length);verify_bool(&mask);
    uint8_t hidden=0;rk_npu_attention_f16_boolean_mask_init(&mask,&hidden,1,1,1,1);auto zero=verify_bool(&mask);require(std::all_of(zero.begin(),zero.end(),[](float x){return x==0;}),"broadcast all-hidden");
    mask.bytes=0;std::vector<float> sentinel(q.size(),123);require(rk_npu_attention_f16_run(dev.ctx,boolean.get(),a.get(),q.data(),rows,17,sentinel.data(),nullptr,&mask)==RK_NPU_ERR_PARAM,"mask bounds");require(std::all_of(sentinel.begin(),sentinel.end(),[](float x){return x==123;}),"invalid mask changed output");
    std::puts("strided/broadcast/versioned boolean masks PASS");
    auto neg=v;for(auto& x:neg)x^=0x8000;check(rk_npu_attention_f16_cache_load(dev.ctx,b.get(),k.data(),neg.data(),length));
    auto q1=random_data(32*128,700);std::vector<float> oa(q1.size()),ob(q1.size()),again(q1.size());
    check(rk_npu_attention_f16_run(dev.ctx,causal.get(),a.get(),q1.data(),1,128,oa.data(),nullptr,nullptr));
    check(rk_npu_attention_f16_run(dev.ctx,causal.get(),b.get(),q1.data(),1,128,ob.data(),nullptr,nullptr));
    check(rk_npu_attention_f16_run(dev.ctx,causal.get(),a.get(),q1.data(),1,128,again.data(),nullptr,nullptr));require(oa==again,"shared workspace A/B/A");for(size_t i=0;i<oa.size();++i)require(std::abs(oa[i]+ob[i])<1e-7,"cache rebinding");
    auto foreign_domain=rk_npu_iommu_domain_create(dev.ctx,13);require(foreign_domain,"foreign domain");auto foreign=cache(foreign_domain,cfg);rk_npu_iommu_domain_free(foreign_domain);
    require(rk_npu_attention_f16_run(dev.ctx,causal.get(),foreign.get(),q1.data(),1,0,again.data(),nullptr,nullptr)==RK_NPU_ERR_DOMAIN,"domain rejection");
    const int total=161;auto allk=random_data(size_t(8)*total*128,800),allv=random_data(allk.size(),801,.5f);
    auto small_cfg=cfg;small_cfg.initial_capacity=32;auto growing=cache(dev.domain,small_cfg);
    auto pk=prefix(allk,total,31),pv=prefix(allv,total,31);check(rk_npu_attention_f16_cache_load(dev.ctx,growing.get(),pk.data(),pv.data(),31));
    std::vector<uint16_t> onek(8*128),onev(8*128);int growths=0;
    for(int n=32;n<=total;++n){
        for(int h=0;h<8;++h){std::copy_n(allk.data()+(size_t(h)*total+n-1)*128,128,onek.data()+h*128);std::copy_n(allv.data()+(size_t(h)*total+n-1)*128,128,onev.data()+h*128);}
        rk_npu_attention_f16_timings t{};check(rk_npu_attention_f16_decode_step(dev.ctx,causal.get(),growing.get(),q1.data(),onek.data(),onev.data(),again.data(),&t,nullptr));
        auto kk=prefix(allk,total,n),vv=prefix(allv,total,n);require(error(again,reference(q1,kk,vv,1,n,n-1,RK_NPU_ATTENTION_F16_CAUSAL,nullptr))<.15,"growing cache reference");
        growths+=t.reserve_us>50;require(t.compute_length==((n+31)/32*32),"growth active length");
    }
    require(growths>0,"exercise automatic reserve");linear.run();
    for(int heads:std::vector<int>{__builtin_popcount(unsigned(core)),8,32}){
        auto shape=none_cfg;shape.kv_heads=heads;shape.query_heads=heads*4;
        auto c=cache(dev.domain,shape);auto w=work(dev.domain,shape);
        auto kk=random_data(size_t(heads)*13*128,900),vv=random_data(kk.size(),901,.5f),qq=random_data(size_t(heads)*4*32*128,902);
        std::vector<float> oo(qq.size());check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),kk.data(),vv.data(),13));
        check(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),qq.data(),32,0,oo.data(),nullptr,nullptr));
        require(error(oo,reference(qq,kk,vv,32,13,0,RK_NPU_ATTENTION_F16_NO_MASK,nullptr,heads*4))<.15,"head counts / cross attention");
    }
    std::puts("query length > KV / head count checks PASS");
    std::printf("core=%d shared-context/shared-workspace/reserve/append/INT8-interleave PASS\n",core);
}
void long_board(int core,int tile,bool int8_canary=true){
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);
    cfg.flags=1;cfg.core_mask=core;cfg.kv_tile=tile;cfg.max_capacity=32768;
    auto c=cache(dev.domain,cfg);Linears linear(dev.domain,core);
    if(int8_canary){linear.run();linear.run();std::puts("CHECK repeated INT8 before long GQA PASS");std::fflush(stdout);}
    else std::puts("META diagnostic long GQA without INT8 interleave");
    const int total=32768;auto allk=random_data(size_t(8)*total*128,910),allv=random_data(allk.size(),911,.5f);
    for(auto mode:{RK_NPU_ATTENTION_F16_CAUSAL,RK_NPU_ATTENTION_F16_NO_MASK,RK_NPU_ATTENTION_F16_BOOLEAN_MASK}){
        cfg.mask_mode=mode;auto w=work(dev.domain,cfg);auto single_cfg=cfg;single_cfg.core_mask=1;auto single=work(dev.domain,single_cfg);
        for(int length:{8191,8192,8193,16383,16384,16385,32767,32768}){
            auto k=prefix(allk,total,length),v=prefix(allv,total,length);
            check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),k.data(),v.data(),length));
            for(int rows:{1,5,32}){
                auto q=random_data(size_t(32)*rows*128,920+rows);const int start=length-rows;
                // A hidden first query and visible positions in every QK chunk
                // catch mask-base aliasing, including the final partial chunk.
                std::vector<uint8_t> data(size_t(32)*(rows+1)*length*2,0);
                for(int h=0;h<32;++h)for(int r=0;r<rows;++r)for(int j=0;j<length;++j)
                    data[((size_t(h)*(rows+1)+r+1)*length+j)*2]=uint8_t((rows==1 || r) && (j+h+r)%3!=0);
                rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,data.data(),data.size(),32,rows+1,length);
                mask.head_stride*=2;mask.query_stride*=2;mask.key_stride=2;mask.query_offset=1;mask.version=1;
                auto* mp=mode==RK_NPU_ATTENTION_F16_BOOLEAN_MASK?&mask:nullptr;
                std::vector<float> guarded(q.size()+16,-98765.125f),baseline(q.size());auto* out=guarded.data()+8;
                rk_npu_attention_f16_timings t{};
                if(int8_canary){std::printf("BEGIN long GQA core=%d tile=%d L=%d Q=%d mode=%d INT8 canary\n",core,tile,length,rows,int(mode));std::fflush(stdout);linear.run();}
                check(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),q.data(),rows,start,out,&t,mp));
                std::vector<float> saved(out,out+q.size());
                double e=error(saved,reference(q,k,v,rows,length,start,mode,mp));require(e<.15,"long GQA exact-softmax oracle");
                for(int i=0;i<8;++i)require(guarded[i]==-98765.125f && guarded[q.size()+8+i]==-98765.125f,"long GQA output guard");
                rk_npu_attention_f16_sizes sizes{};check(rk_npu_attention_f16_query(&cfg,32768,rows,length,&sizes));
                require(t.tasks==sizes.tasks,"long GQA task sizing");
                check(rk_npu_attention_f16_run(dev.ctx,single.get(),c.get(),q.data(),rows,start,baseline.data(),nullptr,mp));
                require(saved==baseline,"long GQA single/multicore bitexact");
                if(mp && rows>1)for(int h=0;h<32;++h)for(int j=0;j<128;++j)require(saved[size_t(h)*rows*128+j]==0,"long GQA hidden row");
                std::printf("CHECK long GQA core=%d tile=%d L=%d Q=%d mode=%d rel_l2=%.8f tasks=%d PASS\n",core,tile,length,rows,int(mode),e,t.tasks);std::fflush(stdout);
            }
        }
    }
    cfg.mask_mode=RK_NPU_ATTENTION_F16_CAUSAL;auto w=work(dev.domain,cfg);auto q=random_data(32*128,940);
    for(int boundary:{8192,16384}){
        auto growing_cfg=cfg;growing_cfg.initial_capacity=boundary;auto growing=cache(dev.domain,growing_cfg);
        auto k=prefix(allk,total,boundary-1),v=prefix(allv,total,boundary-1);
        check(rk_npu_attention_f16_cache_load(dev.ctx,growing.get(),k.data(),v.data(),boundary-1));
        for(int length=boundary;length<=boundary+33;++length){
            auto one_k=prefix(allk,total,0),one_v=one_k;one_k.resize(8*128);one_v.resize(8*128);
            for(int h=0;h<8;++h){std::copy_n(allk.data()+(size_t(h)*total+length-1)*128,128,one_k.data()+h*128);std::copy_n(allv.data()+(size_t(h)*total+length-1)*128,128,one_v.data()+h*128);}
            std::vector<float> out(q.size());check(rk_npu_attention_f16_decode_step(dev.ctx,w.get(),growing.get(),q.data(),one_k.data(),one_v.data(),out.data(),nullptr,nullptr));
            k=prefix(allk,total,length);v=prefix(allv,total,length);
            require(error(out,reference(q,k,v,1,length,length-1,cfg.mask_mode,nullptr))<.15,"long GQA growing/rebound cache");
        }
        std::printf("CHECK long GQA core=%d tile=%d growth=%d..%d PASS\n",core,tile,boundary,boundary+33);std::fflush(stdout);
    }
    if(int8_canary)linear.run();std::puts("CHECK long GQA board PASS");
}
void dirty_sync_board(int core,int tile){
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;Linears linear(dev.domain,core);
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);cfg.flags=1;cfg.core_mask=core;
    cfg.kv_tile=tile;cfg.initial_capacity=32;cfg.max_capacity=32768;
    auto ws=work(dev.domain,cfg);
    RestoreEnv groups{"RK_NPU_ATTN_DIRTY_SYNC_GROUPS"},after{"RK_NPU_ATTN_DIRTY_SYNC_AFTER"};
    require(!unsetenv(after.key),"clear inherited threshold");
    std::vector<Cache> variants;const char* policies[]={"0","1","2","4","span"};
    std::vector<rknpu2_matmul_open::attention::DirtySyncPolicy> expected;
    for(const char* policy:policies){require(!setenv(groups.key,policy,1),"set dirty policy");variants.push_back(cache(dev.domain,cfg));expected.push_back({policy[0]=='s'?32:policy[0]-'0',-1});}
    for(auto policy:std::vector<rknpu2_matmul_open::attention::DirtySyncPolicy>{{1,32},{32,tile},{2,tile+32}}){
        require(!setenv(after.key,std::to_string(policy.after).c_str(),1),"set delayed threshold");
        if(policy.groups==1)require(!unsetenv(groups.key),"threshold-only default");
        else require(!setenv(groups.key,policy.groups==32?"span":"2",1),"delayed explicit groups");
        variants.push_back(cache(dev.domain,cfg));expected.push_back(policy);
    }
    const std::vector<int> counts{1,14,1,33,tile-66,1,31,33,tile+19};
    const int total=2*tile+84;
    auto allk=random_data(size_t(8)*total*128,1201),allv=random_data(allk.size(),1202,.5f);
    auto q1=random_data(32*128,1203),q5=random_data(32*5*128,1204);
    auto chunk=[&](const std::vector<uint16_t>& src,int first,int count){
        std::vector<uint16_t> out(size_t(8)*count*128);
        for(int h=0;h<8;++h)std::copy_n(src.data()+(size_t(h)*total+first)*128,size_t(count)*128,out.data()+size_t(h)*count*128);
        return out;
    };
    auto verify=[&](int length){
        auto k=prefix(allk,total,length),v=prefix(allv,total,length);
        // End on the final query so the next append can use incremental tail.
        for(int rows:{1,5,1}){
            const int start=rows==1?length-1:std::max(0,length/2-5);const auto& q=rows==1?q1:q5;
            const auto ref=reference(q,k,v,rows,length,start,cfg.mask_mode,nullptr);
            std::vector<float> baseline;
            for(size_t index=0;index<variants.size();++index){
                const auto& policy=variants[index]->dirty_sync;const auto& wanted=expected[index];
                require(policy.groups==wanted.groups && policy.after==wanted.after && policy.for_length(length)==(length>wanted.after?wanted.groups:0),"cache policy capture/threshold transition");
                std::vector<float> out(q.size()+16,-98765.125f);
                check(rk_npu_attention_f16_run(dev.ctx,ws.get(),variants[index].get(),q.data(),rows,start,out.data()+8,nullptr,nullptr));
                for(int i=0;i<8;++i)require(out[i]==-98765.125f && out[q.size()+8+i]==-98765.125f,"dirty sync output guard");
                std::vector<float> values(out.begin()+8,out.end()-8);require(error(values,ref)<.15,"dirty sync exact-softmax oracle");
                if(!index)baseline=values;else require(values==baseline,"dirty sync full-output bitexact");
            }
        }
        linear.run();
    };
    for(int cycle=0;cycle<2;++cycle){
        if(cycle)for(auto& x:allv)x^=0x8000; // Reload after old native/tail data.
        int length=17;auto k=prefix(allk,total,length),v=prefix(allv,total,length);
        for(auto& c:variants)check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),k.data(),v.data(),length));
        verify(length);
        for(int count:counts){
            k=chunk(allk,length,count);v=chunk(allv,length,count);
            for(auto& c:variants)check(rk_npu_attention_f16_cache_append(dev.ctx,c.get(),k.data(),v.data(),count));
            length+=count;verify(length);
            std::printf("CHECK dirty sync core=%d tile=%d cycle=%d append=%d L=%d policies=0/1/2/4/span+after32/span_after_tile/2_after_tile32 guards/oracle/bitexact/threshold/INT8 PASS\n",core,tile,cycle,count,length);std::fflush(stdout);
        }
        require(length==total,"dirty test final length");
        for(auto& c:variants)check(rk_npu_attention_f16_cache_reserve(dev.ctx,c.get(),32768));
        verify(length);
    }
    std::puts("CHECK dirty sync multi-token/tile/K32/reserve/reload/earlier-query board PASS");
}
int main(int argc,char** argv)try{cpu();if(argc>1 && std::string(argv[1])=="--mixed-canary")mixed_canary(argc>2?std::stoi(argv[2]):7,argc>3?std::stoi(argv[3]):4096);
else if(argc>1 && std::string(argv[1])=="--dirty-sync-regression")dirty_sync_board(argc>2?std::stoi(argv[2]):7,argc>3?std::stoi(argv[3]):512);
else if(argc>1 && std::string(argv[1])=="--attention-mixed")attention_mixed_canary(argc>2?std::stoi(argv[2]):7,argc>3?std::stoi(argv[3]):8191,argc>4);
else if(argc>1 && std::string(argv[1])=="--board"){
    const int core=argc>2?std::stoi(argv[2]):1;
    if(argc>3 && std::string(argv[3])=="--long")long_board(core,argc>4?std::stoi(argv[4]):512,!(argc>5 && std::string(argv[5])=="--no-int8"));else board(core);
}return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

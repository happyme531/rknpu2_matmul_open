#include "rk_npu_attention_f16.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
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
namespace bits=rknpu2_matmul_open::bits;
void check(int rc){if(rc)throw std::runtime_error(rk_npu_strerror(rc));}
void require(bool yes,const char* s){if(!yes)throw std::runtime_error(s);}
struct Device{rk_npu_ctx* ctx=rk_npu_open(nullptr);rk_npu_iommu_domain* domain=ctx?rk_npu_iommu_domain_create(ctx,14):nullptr;Device(){require(ctx&&domain,"device/domain");}~Device(){rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);}};
struct Lock{int fd;explicit Lock(const char* p):fd(open(p,O_CREAT|O_RDWR,0666)){require(fd>=0 && !flock(fd,LOCK_EX|LOCK_NB),"NPU lock busy");}~Lock(){flock(fd,LOCK_UN);close(fd);}};
struct Variant{
    std::string name;
    std::unique_ptr<rk_npu_attention_f16_cache,decltype(&rk_npu_attention_f16_cache_free)> cache{nullptr,rk_npu_attention_f16_cache_free};
    std::unique_ptr<rk_npu_attention_f16_workspace,decltype(&rk_npu_attention_f16_workspace_free)> work{nullptr,rk_npu_attention_f16_workspace_free};
    std::vector<float> output;
    rk_npu_attention_f16_boolean_mask mask{};
    bool boolean=false;
};
struct Sample{std::string variant;int iteration,length;rk_npu_attention_f16_timings t;};
int main(int argc,char** argv)try{
    int length=4096,capacity=8192,loops=50,warmup=5,tile=512,core=1;bool step=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];require(i+1<argc,"missing argument");
        if(a=="--length")length=std::stoi(argv[++i]);else if(a=="--capacity")capacity=std::stoi(argv[++i]);
        else if(a=="--loops")loops=std::stoi(argv[++i]);else if(a=="--warmup")warmup=std::stoi(argv[++i]);
        else if(a=="--key-tile")tile=std::stoi(argv[++i]);else if(a=="--core-mask")core=std::stoi(argv[++i]);
        else if(a=="--scope")step=std::string(argv[++i])=="step";else throw std::runtime_error("unknown option");}
    require(length>0 && loops>0 && warmup>=0 && capacity>=length+(step?loops+warmup:0),"invalid benchmark sizes");
    Lock lock1("/tmp/rknpu_lowlevel_submit.lock"),lock2("/tmp/rk3588_npu_submit.lock");Device dev;
    std::mt19937 rng(20261003);std::normal_distribution<float> qk(0,.7),vv(0,.5);
    std::vector<uint16_t> k(size_t(8)*length*128),v(k.size()),q(32*128),nk(8*128),nv(nk.size());
    for(auto& x:k)x=bits::float_to_half(qk(rng));for(auto& x:v)x=bits::float_to_half(vv(rng));
    for(auto& x:q)x=bits::float_to_half(qk(rng));for(auto& x:nk)x=bits::float_to_half(qk(rng));for(auto& x:nv)x=bits::float_to_half(vv(rng));
    std::vector<uint8_t> visible(capacity,1);std::vector<Variant> variants;
    for(auto name:{"causal","none","boolean_static","boolean_refresh"}){
        Variant x;x.name=name;x.boolean=x.name.find("boolean")==0;x.output.resize(q.size());
        rk_npu_attention_f16_config c;rk_npu_attention_f16_config_init(&c);c.flags=1;c.core_mask=core;c.kv_tile=tile;
        c.initial_capacity=capacity;c.max_capacity=capacity;c.max_query_rows=1;
        c.mask_mode=x.boolean?RK_NPU_ATTENTION_F16_BOOLEAN_MASK:x.name=="none"?RK_NPU_ATTENTION_F16_NO_MASK:RK_NPU_ATTENTION_F16_CAUSAL;
        rk_npu_attention_f16_cache* cache=nullptr;rk_npu_attention_f16_workspace* w=nullptr;
        check(rk_npu_attention_f16_cache_create(dev.domain,&c,&cache));x.cache.reset(cache);
        check(rk_npu_attention_f16_prepare(dev.domain,&c,&w));x.work.reset(w);
        check(rk_npu_attention_f16_cache_load(dev.ctx,cache,k.data(),v.data(),length));
        rk_npu_attention_f16_boolean_mask_init(&x.mask,visible.data(),visible.size(),1,1,capacity);
        x.mask.version=x.name=="boolean_static"?1:0;
        check(rk_npu_attention_f16_run(dev.ctx,w,cache,q.data(),1,length-1,x.output.data(),nullptr,x.boolean?&x.mask:nullptr));
        variants.push_back(std::move(x));
    }
    for(auto& x:variants)require(x.output==variants[0].output,"all-visible mask vs no-mask mismatch");
    std::vector<Sample> samples;
    for(int i=0;i<warmup+loops;++i){
        for(size_t j=0;j<variants.size();++j){auto& x=variants[i%2?variants.size()-1-j:j];rk_npu_attention_f16_timings t{};
            if(step)check(rk_npu_attention_f16_decode_step(dev.ctx,x.work.get(),x.cache.get(),q.data(),nk.data(),nv.data(),x.output.data(),&t,x.boolean?&x.mask:nullptr));
            else check(rk_npu_attention_f16_run(dev.ctx,x.work.get(),x.cache.get(),q.data(),1,length-1,x.output.data(),&t,x.boolean?&x.mask:nullptr));
            if(i>=warmup)samples.push_back({x.name,i-warmup,length+(step?i+1:0),t});
        }
        for(auto& x:variants)require(x.output==variants[0].output,"repeated mode comparison mismatch");
    }
    std::printf("CHECK all_visible_modes_bitexact=PASS capacity=%d scope=%s core_mask=%d\n",capacity,step?"step":"run",core);
    std::puts("variant,iteration,length,compute_length,tasks,reserve_us,kv_update_us,prepare_us,tail_stage_us,mask_update_us,q_pack_us,submit_us,hardware_us,output_sync_us,finish_us,total_us");
    for(auto& x:samples){auto& t=x.t;std::printf("%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",x.variant.c_str(),x.iteration,x.length,t.compute_length,t.tasks,t.reserve_us,t.kv_update_us,t.prepare_us,t.tail_stage_us,t.mask_update_us,t.query_pack_us,t.submit_us,t.hardware_us,t.output_sync_us,t.finish_us,t.total_us);}
    std::puts("RESULT PASS");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

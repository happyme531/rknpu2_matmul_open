#include "rk_npu_attention_f16.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

namespace bits=rknpu2_matmul_open::bits;
using Clock=std::chrono::steady_clock;
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
void check(int rc){if(rc)throw std::runtime_error(rk_npu_strerror(rc));}
struct Lock{
    int fd;
    explicit Lock(const char* name):fd(open(name,O_CREAT|O_RDWR,0666)){require(fd>=0 && !flock(fd,LOCK_EX|LOCK_NB),"NPU lock busy");}
    ~Lock(){flock(fd,LOCK_UN);close(fd);}
};
struct Device{
    rk_npu_ctx* ctx=rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain=ctx?rk_npu_iommu_domain_create(ctx,14):nullptr;
    Device(){require(ctx&&domain,"device/domain");}
    ~Device(){rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);}
};
using Cache=std::unique_ptr<rk_npu_attention_f16_cache,decltype(&rk_npu_attention_f16_cache_free)>;
using Work=std::unique_ptr<rk_npu_attention_f16_workspace,decltype(&rk_npu_attention_f16_workspace_free)>;
Cache make_cache(Device& d,const rk_npu_attention_f16_config& cfg){
    rk_npu_attention_f16_cache* p=nullptr;check(rk_npu_attention_f16_cache_create(d.domain,&cfg,&p));
    return Cache(p,rk_npu_attention_f16_cache_free);
}
struct Variant{
    int core;
    Cache own{nullptr,rk_npu_attention_f16_cache_free};
    rk_npu_attention_f16_cache* cache=nullptr;
    Work workspace{nullptr,rk_npu_attention_f16_workspace_free};
    std::vector<float> output;
};
struct Sample{int core,iteration,length;rk_npu_attention_f16_timings t;};
int main(int argc,char** argv)try{
    int length=4096,rows=1,tile=512,capacity=8192,loops=25,warmup=3;
    std::string scope="run",mode="causal",masks="1,3,7";
    for(int i=1;i<argc;++i){require(i+1<argc,"missing option value");std::string option=argv[i],value=argv[++i];
        if(option=="--length")length=std::stoi(value);else if(option=="--query-rows")rows=std::stoi(value);
        else if(option=="--key-tile")tile=std::stoi(value);else if(option=="--capacity")capacity=std::stoi(value);
        else if(option=="--loops")loops=std::stoi(value);else if(option=="--warmup")warmup=std::stoi(value);
        else if(option=="--scope")scope=value;else if(option=="--mask")mode=value;
        else if(option=="--core-masks")masks=value;else throw std::runtime_error("unknown option");
    }
    const bool step=scope=="step",fresh=scope=="fresh";
    require((scope=="run" || step || fresh) && (mode=="causal" || mode=="none" || mode=="boolean"),"invalid scope/mask");
    require(length>=rows && rows>=1 && rows<=32 && loops>0 && warmup>=0 &&
            capacity>=length+(step?loops+warmup:0) && (!step || rows==1),"invalid dimensions");
    std::vector<int> cores;
    for(size_t begin=0;begin<masks.size();){auto end=masks.find(',',begin);cores.push_back(std::stoi(masks.substr(begin,end-begin)));if(end==std::string::npos)break;begin=end+1;}
    require(!cores.empty() && cores[0]==1,"first variant must be core1 reference");
    Lock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    std::mt19937 rng(20261003);std::normal_distribution<float> qk(0,.7f),vv(0,.5f);
    std::vector<uint16_t> k(size_t(8)*length*128),v(k.size()),q(size_t(32)*rows*128),nk(8*128),nv(nk.size());
    for(auto& x:k)x=bits::float_to_half(qk(rng));for(auto& x:v)x=bits::float_to_half(vv(rng));
    for(auto& x:q)x=bits::float_to_half(qk(rng));for(auto& x:nk)x=bits::float_to_half(qk(rng));for(auto& x:nv)x=bits::float_to_half(vv(rng));
    std::vector<uint8_t> visible(capacity,1);rk_npu_attention_f16_boolean_mask boolean{};
    rk_npu_attention_f16_boolean_mask_init(&boolean,visible.data(),visible.size(),1,1,capacity);boolean.version=1;
    const auto* mask=mode=="boolean"?&boolean:nullptr;
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);cfg.flags=1;cfg.kv_tile=tile;
    cfg.max_query_rows=rows;cfg.initial_capacity=cfg.max_capacity=capacity;
    cfg.mask_mode=mode=="none"?RK_NPU_ATTENTION_F16_NO_MASK:mode=="boolean"?RK_NPU_ATTENTION_F16_BOOLEAN_MASK:RK_NPU_ATTENTION_F16_CAUSAL;
    auto shared=make_cache(dev,cfg);check(rk_npu_attention_f16_cache_load(dev.ctx,shared.get(),k.data(),v.data(),length));
    std::vector<Variant> variants;
    for(int core:cores){Variant x;x.core=core;x.output.resize(q.size()+16,12345.f);cfg.core_mask=core;
        if(step){x.own=make_cache(dev,cfg);check(rk_npu_attention_f16_cache_load(dev.ctx,x.own.get(),k.data(),v.data(),length));x.cache=x.own.get();}
        else x.cache=shared.get();
        rk_npu_attention_f16_workspace* w=nullptr;check(rk_npu_attention_f16_prepare(dev.domain,&cfg,&w));x.workspace.reset(w);
        check(rk_npu_attention_f16_run(dev.ctx,w,x.cache,q.data(),rows,length-rows,x.output.data()+8,nullptr,mask));
        variants.push_back(std::move(x));
    }
    auto validate=[&]{for(auto& x:variants){require(x.output==variants.front().output,"single/multicore output mismatch");
        require(std::all_of(x.output.begin(),x.output.begin()+8,[](float f){return f==12345.f;}) &&
                std::all_of(x.output.end()-8,x.output.end(),[](float f){return f==12345.f;}),"output guard overwritten");}};
    validate();const auto stable=variants.front().output;std::vector<Sample> samples;
    for(int i=0;i<loops+warmup;++i){
        if(step){for(auto& x:nk)x=bits::float_to_half(qk(rng));for(auto& x:nv)x=bits::float_to_half(vv(rng));}
        // Rotate and reverse order to avoid assigning the same thermal/cache
        // position to a particular core count. Comparisons are outside timers.
        for(size_t j=0;j<variants.size();++j){size_t at=(j+size_t(i))%variants.size();if(i%2)at=variants.size()-1-at;
            auto& x=variants[at];rk_npu_attention_f16_timings t{};
            auto begin=Clock::now();
            if(fresh)check(rk_npu_attention_f16_cache_load(dev.ctx,x.cache,k.data(),v.data(),length));
            auto loaded=Clock::now();
            if(step)check(rk_npu_attention_f16_decode_step(dev.ctx,x.workspace.get(),x.cache,q.data(),nk.data(),nv.data(),x.output.data()+8,&t,mask));
            else check(rk_npu_attention_f16_run(dev.ctx,x.workspace.get(),x.cache,q.data(),rows,length-rows,x.output.data()+8,&t,mask));
            if(fresh){t.kv_update_us=std::chrono::duration<double,std::micro>(loaded-begin).count();t.total_us=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();}
            if(i>=warmup)samples.push_back({x.core,i-warmup,length+(step?i+1:0),t});
        }
        validate();if(!step)require(variants.front().output==stable,"unstable repeated output");
    }
    std::printf("CHECK single_multicore_bitexact=PASS output_guards=PASS length=%d rows=%d tile=%d scope=%s mask=%s cores=%s\n",length,rows,tile,scope.c_str(),mode.c_str(),masks.c_str());
    std::puts("core_mask,iteration,length,compute_length,tasks,reserve_us,kv_update_us,prepare_us,tail_stage_us,mask_update_us,q_pack_us,submit_us,hardware_us,output_sync_us,finish_us,total_us");
    for(auto& x:samples){auto& t=x.t;std::printf("%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",x.core,x.iteration,x.length,t.compute_length,t.tasks,t.reserve_us,t.kv_update_us,t.prepare_us,t.tail_stage_us,t.mask_update_us,t.query_pack_us,t.submit_us,t.hardware_us,t.output_sync_us,t.finish_us,t.total_us);}
    std::puts("RESULT PASS");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

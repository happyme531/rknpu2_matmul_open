#include "mla_f16_test_utils.h"
#include <chrono>
using namespace mla_test;
using Clock=std::chrono::steady_clock;
void accumulate(rk_npu_mla_f16_timings& t,const rk_npu_mla_f16_timings& x){
    for(auto p:{&rk_npu_mla_f16_timings::query_projection_us,&rk_npu_mla_f16_timings::attention_us,
        &rk_npu_mla_f16_timings::value_projection_us,&rk_npu_mla_f16_timings::gate_output_us})t.*p+=x.*p;
    for(auto p:{&rk_npu_attention_f16_timings::prepare_us,&rk_npu_attention_f16_timings::tail_stage_us,
        &rk_npu_attention_f16_timings::mask_update_us,&rk_npu_attention_f16_timings::query_pack_us,
        &rk_npu_attention_f16_timings::submit_us,&rk_npu_attention_f16_timings::hardware_us,
        &rk_npu_attention_f16_timings::output_sync_us,&rk_npu_attention_f16_timings::finish_us})t.attention.*p+=x.attention.*p;
    t.attention.compute_length=x.attention.compute_length;t.attention.tasks+=x.attention.tasks;
}
struct Variant{
    int mode,core;
    Cache owned{nullptr,rk_npu_mla_f16_cache_free};
    rk_npu_mla_f16_cache* cache=nullptr;
    Work ws{nullptr,rk_npu_mla_f16_workspace_free};
    GuardOutput output;
    Variant(int md,int cr,int rows):mode(md),core(cr),output(size_t(rows)*H*D){}
};
int main(int argc,char** argv){try{
    int length=4096,rows=32,tile=512,loops=25,warm=3,capacity=8192;std::string scope="run",mask="causal";bool gated=true;
    for(int i=1;i<argc;++i){std::string arg=argv[i];require(i+1<argc,"missing option value");const char* val=argv[++i];
        if(arg=="--length")length=std::stoi(val);else if(arg=="--query-rows")rows=std::stoi(val);
        else if(arg=="--capacity")capacity=std::stoi(val);
        else if(arg=="--key-tile")tile=std::stoi(val);else if(arg=="--loops")loops=std::stoi(val);
        else if(arg=="--warmup")warm=std::stoi(val);else if(arg=="--scope")scope=val;
        else if(arg=="--mask")mask=val;else if(arg=="--gate")gated=std::stoi(val)!=0;else throw std::runtime_error("unknown option");
    }
    require(length>=rows && rows>0 && rows<=32 && loops>0 && warm>=0,"invalid geometry");
    require(scope=="run" || scope=="fresh" || scope=="step" || scope=="prefill","invalid scope");
    require(mask=="causal" || mask=="none" || mask=="boolean","invalid mask");
    if(scope=="step")require(rows==1 && warm==0,"step requires Q1 and warmup0");
    if(scope=="prefill")require(length%rows==0 && mask=="causal","prefill requires whole causal chunks");
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    Fixture fixture(length+(scope=="step"?loops:0),scope=="prefill"?length:rows);auto weights=mla_test::weights(dev,fixture.w.data());
    rk_npu_mla_f16_config cfg;rk_npu_mla_f16_config_init(&cfg);cfg.flags=1;cfg.kv_tile=tile;
    cfg.initial_capacity=capacity;cfg.max_capacity=capacity;require(length+(scope=="step"?loops:0)<=capacity,"capacity");
    cfg.mask_mode=mask=="causal"?RK_NPU_ATTENTION_F16_CAUSAL:mask=="none"?RK_NPU_ATTENTION_F16_NO_MASK:RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
    std::vector<uint8_t> bytes(capacity,1);rk_npu_attention_f16_boolean_mask boolean;
    rk_npu_attention_f16_boolean_mask_init(&boolean,bytes.data(),bytes.size(),1,1,capacity);boolean.version=1;
    const auto* bm=mask=="boolean"?&boolean:nullptr;const auto* gate=gated?fixture.gates.data():nullptr;
    std::vector<std::unique_ptr<Variant>> variants;std::vector<Cache> shared;
    for(int mode=0;mode<2;++mode){cfg.mode=static_cast<rk_npu_mla_f16_mode>(mode);cfg.core_mask=1;
        if(scope=="run"){shared.push_back(cache(dev,cfg,weights.get()));check(rk_npu_mla_f16_cache_load(dev.ctx,shared.back().get(),fixture.latent.data(),fixture.rope.data(),length));}
        for(int core:{1,3,7}){cfg.core_mask=core;auto v=std::make_unique<Variant>(mode,core,scope=="prefill"?length:rows);v->ws=work(dev,cfg,weights.get());
            if(scope=="run")v->cache=shared.back().get();else {v->owned=cache(dev,cfg,weights.get());v->cache=v->owned.get();
                check(rk_npu_mla_f16_cache_load(dev.ctx,v->cache,fixture.latent.data(),fixture.rope.data(),length));}
            rk_npu_mla_f16_cache_info info{};check(rk_npu_mla_f16_cache_get_info(v->cache,&info));
            std::printf("META mode=%d core=%d capacity=%d key_bytes=%llu value_bytes=%llu\n",mode,core,info.capacity,(unsigned long long)info.key_bytes,(unsigned long long)info.value_bytes);
            variants.push_back(std::move(v));
        }
    }
    auto oracle=fixture.reference(length,rows,length-rows,cfg.mask_mode,bm,gated);
    std::vector<int> spot_starts;std::vector<std::vector<float>> spot_oracles;
    if(scope=="prefill")for(int start:{0,length/2/rows*rows,length-rows}){
        spot_starts.push_back(start);spot_oracles.push_back(fixture.reference(length,rows,start,cfg.mask_mode,nullptr,gated,start));
    }
    struct Sample{int mode,core,iteration,length;rk_npu_mla_f16_timings t;double total;};std::vector<Sample> samples;
    std::vector<float> stable[2];double maximum_error[2]{};
    for(int iteration=-warm;iteration<loops;++iteration){
        const int active=scope=="step"?length+iteration+1:length;
        const bool reference_check=scope!="step"?iteration==-warm:iteration==0 || active==4097 || active==4129 || iteration==loops-1;
        if(scope=="step" && reference_check)oracle=fixture.reference(active,rows,active-rows,cfg.mask_mode,bm,gated);
        std::vector<float> per_mode[2];
        for(size_t slot=0;slot<variants.size();++slot){
            size_t index=(slot+size_t(iteration+warm))%variants.size();if((iteration+warm)&1)index=variants.size()-1-index;
            auto& v=*variants[index];rk_npu_mla_f16_timings t{};auto begin=Clock::now();
            if(scope=="fresh" || scope=="prefill"){
                check(rk_npu_mla_f16_cache_load(dev.ctx,v.cache,fixture.latent.data(),fixture.rope.data(),length));
                t.kv_update_us=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
            }
            const double load_us=t.kv_update_us;
            if(scope=="step")check(rk_npu_mla_f16_decode_step(dev.ctx,v.ws.get(),v.cache,fixture.q.data(),gate,
                fixture.latent.data()+size_t(active-1)*C,fixture.rope.data()+size_t(active-1)*R,v.output.ptr(),&t,bm));
            else if(scope=="prefill")for(int start=0;start<length;start+=rows){rk_npu_mla_f16_timings chunk{};
                check(rk_npu_mla_f16_run(dev.ctx,v.ws.get(),v.cache,fixture.q.data()+size_t(start)*H*QK,
                    gate?gate+size_t(start)*H:nullptr,rows,start,v.output.ptr()+size_t(start)*H*D,&chunk,nullptr));accumulate(t,chunk);}
            else check(rk_npu_mla_f16_run(dev.ctx,v.ws.get(),v.cache,fixture.q.data(),gate,rows,active-rows,v.output.ptr(),&t,bm));
            if(scope=="fresh")t.kv_update_us=load_us;
            const double total=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();auto values=v.output.values();
            if(reference_check){
                if(scope=="prefill")for(size_t i=0;i<spot_starts.size();++i){auto first=values.begin()+size_t(spot_starts[i])*H*D;
                    double e=error(std::vector<float>(first,first+size_t(rows)*H*D),spot_oracles[i]);maximum_error[v.mode]=std::max(maximum_error[v.mode],e);require(e<.08,"full-prefill spot oracle");}
                else {double e=error(values,oracle);maximum_error[v.mode]=std::max(maximum_error[v.mode],e);require(e<.08,"exact-softmax oracle");}
            }
            if(per_mode[v.mode].empty())per_mode[v.mode]=values;else require(per_mode[v.mode]==values,"single/multi bitexact");
            if(scope!="step"){if(stable[v.mode].empty())stable[v.mode]=values;else require(stable[v.mode]==values,"repeated output stable");}
            if(iteration>=0)samples.push_back({v.mode,v.core,iteration,active,t,total});
        }
    }
    std::printf("CHECK exact_softmax_rel_l2 expanded=%.8f absorbed=%.8f PASS\n",maximum_error[0],maximum_error[1]);
    std::puts("CHECK all_outputs guards/single_multi_bitexact/repeat_or_growth PASS");
    std::puts("mode,core_mask,iteration,length,compute_length,tasks,kv_update_us,query_projection_us,attention_us,value_projection_us,gate_output_us,prepare_us,tail_stage_us,mask_update_us,query_pack_us,submit_us,hardware_us,output_sync_us,finish_us,total_us");
    for(const auto& s:samples){const auto& t=s.t;const auto& a=t.attention;
        std::printf("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            s.mode,s.core,s.iteration,s.length,a.compute_length,a.tasks,t.kv_update_us,t.query_projection_us,t.attention_us,t.value_projection_us,t.gate_output_us,
            a.prepare_us,a.tail_stage_us,a.mask_update_us,a.query_pack_us,a.submit_us,a.hardware_us,a.output_sync_us,a.finish_us,s.total);
    }return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MLA BENCH FAIL: %s\n",e.what());return 1;}}

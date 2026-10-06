#include "mla_f16_test_utils.h"
#include <chrono>
#include <dlfcn.h>
using namespace mla_test;
using Clock=std::chrono::steady_clock;

// Each version owns its context and every opaque handle. No private structs
// cross the DSO boundary. DEEPBIND prevents the baseline calling new internals.
#define API_FUNCTIONS(X) \
    X(rk_npu_open) X(rk_npu_close) X(rk_npu_iommu_domain_create) X(rk_npu_iommu_domain_free) \
    X(rk_npu_mla_f16_weights_create) X(rk_npu_mla_f16_weights_free) \
    X(rk_npu_mla_f16_cache_create) X(rk_npu_mla_f16_cache_free) X(rk_npu_mla_f16_cache_get_info) \
    X(rk_npu_mla_f16_cache_load) X(rk_npu_mla_f16_cache_append) \
    X(rk_npu_mla_f16_prepare) X(rk_npu_mla_f16_workspace_free) \
    X(rk_npu_mla_f16_run) X(rk_npu_mla_f16_decode_step)
struct Api {
    void* handle=nullptr;
#define FIELD(f) decltype(&::f) f=nullptr;
    API_FUNCTIONS(FIELD)
#undef FIELD
    explicit Api(const std::string& file){
        handle=dlopen(file.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
        if(!handle)throw std::runtime_error(dlerror());
#define LOAD(f) f=reinterpret_cast<decltype(f)>(dlsym(handle,#f));require(f!=nullptr,"missing version API");
        API_FUNCTIONS(LOAD)
#undef LOAD
    }
    ~Api(){if(handle)dlclose(handle);}
};
struct Backend {
    Api api;
    bool old;
    rk_npu_ctx* ctx=nullptr;
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mla_f16_weights* weights=nullptr;
    Backend(const std::string& file,bool before,const Fixture& f):api(file),old(before){
        ctx=api.rk_npu_open(nullptr);domain=ctx?api.rk_npu_iommu_domain_create(ctx,14):nullptr;
        require(ctx && domain,"backend open/domain");check(api.rk_npu_mla_f16_weights_create(domain,f.w.data(),&weights));
    }
    ~Backend(){api.rk_npu_mla_f16_weights_free(weights);api.rk_npu_iommu_domain_free(domain);api.rk_npu_close(ctx);}
};
void add(rk_npu_mla_f16_timings& t,const rk_npu_mla_f16_timings& x){
    for(auto p:{&rk_npu_mla_f16_timings::query_projection_us,&rk_npu_mla_f16_timings::attention_us,
        &rk_npu_mla_f16_timings::value_projection_us,&rk_npu_mla_f16_timings::gate_output_us})t.*p+=x.*p;
    for(auto p:{&rk_npu_attention_f16_timings::prepare_us,&rk_npu_attention_f16_timings::tail_stage_us,
        &rk_npu_attention_f16_timings::mask_update_us,&rk_npu_attention_f16_timings::query_pack_us,
        &rk_npu_attention_f16_timings::submit_us,&rk_npu_attention_f16_timings::hardware_us,
        &rk_npu_attention_f16_timings::output_sync_us,&rk_npu_attention_f16_timings::finish_us})t.attention.*p+=x.attention.*p;
    t.attention.compute_length=x.attention.compute_length;t.attention.tasks+=x.attention.tasks;
}
struct Variant {
    Backend& backend;
    int mode,chunk;
    std::string name;
    rk_npu_mla_f16_config cfg{};
    rk_npu_mla_f16_cache* cache=nullptr;
    rk_npu_mla_f16_workspace* work=nullptr;
    GuardOutput output;
    double max_diff=0,max_reference=0;
    bool all_bitexact=true;
    Variant(Backend& b,int md,int q,int core,int count,uint32_t flags,int capacity):backend(b),mode(md),chunk(q),
        name(std::string(b.old?"old":"new")+"_q"+std::to_string(q)),output(size_t(count)*H*D){
        rk_npu_mla_f16_config_init(&cfg);cfg.flags=flags;cfg.mode=static_cast<rk_npu_mla_f16_mode>(md);
        cfg.core_mask=core;cfg.max_query_rows=b.old && flags==1?32:md==0?128:32;
        cfg.initial_capacity=capacity;cfg.max_capacity=capacity;
        check(backend.api.rk_npu_mla_f16_prepare(b.domain,&cfg,b.weights,&work));
    }
    ~Variant(){backend.api.rk_npu_mla_f16_workspace_free(work);backend.api.rk_npu_mla_f16_cache_free(cache);}
    void create(){check(backend.api.rk_npu_mla_f16_cache_create(backend.domain,&cfg,backend.weights,&cache));}
    void reset_cache(){backend.api.rk_npu_mla_f16_cache_free(cache);cache=nullptr;create();}
    void reset_work(){backend.api.rk_npu_mla_f16_workspace_free(work);work=nullptr;
        check(backend.api.rk_npu_mla_f16_prepare(backend.domain,&cfg,backend.weights,&work));}
};
int main(int argc,char** argv){try{
    std::string old_file,new_file,scope="run",modes="both";int length=4096,core=7,queries=128,loops=15,warm=3,initial=8192;bool cold=false;
    uint32_t flags=RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL;
    for(int i=1;i<argc;++i){std::string arg=argv[i];require(i+1<argc,"missing value");std::string v=argv[++i];
        if(arg=="--old")old_file=v;else if(arg=="--new")new_file=v;else if(arg=="--scope")scope=v;
        else if(arg=="--length")length=std::stoi(v);else if(arg=="--core")core=std::stoi(v);
        else if(arg=="--queries")queries=std::stoi(v);else if(arg=="--loops")loops=std::stoi(v);
        else if(arg=="--warmup")warm=std::stoi(v);else if(arg=="--modes")modes=v;
        else if(arg=="--flags")flags=std::stoul(v);
        else if(arg=="--initial-capacity")initial=std::stoi(v);else if(arg=="--cold-workspace")cold=std::stoi(v)!=0;else throw std::runtime_error("unknown option");
    }
    require(!old_file.empty() && !new_file.empty(),"two DSO paths required");
    require(scope=="run" || scope=="fresh" || scope=="prefill" || scope=="stream" || scope=="step","scope");
    require(modes=="both" || modes=="expanded" || modes=="absorbed","modes");
    require(core==1 || core==3 || core==7,"core");
    const bool stable_softmax=flags==RK_NPU_MLA_F16_STABLE_SOFTMAX;
    require(stable_softmax || flags==RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL,"flags");
    require(loops>0 && warm>=0 && queries>0 && queries<=128 && length>=queries && length<=(stable_softmax?32768:4096),"geometry");
    if(scope=="step")require(queries==1 && warm==0,"step Q1/warmup0");
    if(scope=="prefill" || scope=="stream")require(length%128==0,"whole prefill batch128");
    const int output_rows=(scope=="prefill" || scope=="stream")?length:queries;
    const int fixture_length=length+(scope=="step"?loops:0);
    const int capacity=std::max(8192,(fixture_length+31)/32*32);
    require(capacity<=32768,"capacity");
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");
    Fixture f(fixture_length,output_rows);Backend old(old_file,true,f),current(new_file,false,f);
    std::vector<std::unique_ptr<Variant>> variants;
    for(int md=0;md<2;++md){if((modes=="expanded" && md) || (modes=="absorbed" && !md))continue;
        variants.emplace_back(new Variant(old,md,std::min(md==0 && stable_softmax?128:32,queries),core,output_rows,flags,capacity));
        for(int q:stable_softmax?std::vector<int>{std::min(md==0?128:32,queries)}:
                queries==1?std::vector<int>{1}:md==0?std::vector<int>{32,64,128}:std::vector<int>{32}){
            if(q>queries && scope!="prefill" && scope!="stream")continue;
            variants.emplace_back(new Variant(current,md,std::min(q,queries),core,output_rows,flags,capacity));
        }
    }
    for(auto& vp:variants){auto& v=*vp;v.cfg.initial_capacity=scope=="stream"?initial:capacity;v.create();
        if(scope!="stream")check(v.backend.api.rk_npu_mla_f16_cache_load(v.backend.ctx,v.cache,f.latent.data(),f.rope.data(),length));}
    struct Sample{std::string variant;int mode,iteration,length;double total;rk_npu_mla_f16_timings t;};std::vector<Sample> samples;
    std::vector<std::pair<int,std::vector<float>>> references;
    if(scope=="prefill" || scope=="stream")for(int start:{0,length/2/32*32,length-32})
        references.push_back({start,f.reference(length,32,start,RK_NPU_ATTENTION_F16_CAUSAL,nullptr,true,start)});
    else references.push_back({0,f.reference(length,queries,length-queries,RK_NPU_ATTENTION_F16_CAUSAL)});
    std::vector<float> stable[2];
    for(int iteration=-warm;iteration<loops;++iteration){
        const int active=scope=="step"?length+iteration+1:length;
        const bool verify=iteration==-warm || (scope=="step" && (active==4097 || active==4129 || iteration==loops-1));
        if(scope=="step" && verify)references={{0,f.reference(active,1,active-1,RK_NPU_ATTENTION_F16_CAUSAL)}};
        for(size_t slot=0;slot<variants.size();++slot){
            // Rotation followed by reversal cancels for two variants. The
            // stable before/after pair must actually alternate its order.
            size_t index=stable_softmax?slot:(slot+size_t(iteration+warm))%variants.size();
            if((iteration+warm)&1)index=variants.size()-1-index;
            auto& v=*variants[index];auto& api=v.backend.api;auto* ctx=v.backend.ctx;
            if(scope=="stream")v.reset_cache(); // Initial allocation excluded; growth below is timed.
            if(cold)v.reset_work(); // No graph/buffers allocated by prepare; first-use allocation is timed.
            rk_npu_mla_f16_timings total{};auto begin=Clock::now();
            if(scope=="fresh" || scope=="prefill"){
                check(api.rk_npu_mla_f16_cache_load(ctx,v.cache,f.latent.data(),f.rope.data(),length));
                total.kv_update_us=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
            }
            if(scope=="step")check(api.rk_npu_mla_f16_decode_step(ctx,v.work,v.cache,f.q.data(),f.gates.data(),
                f.latent.data()+size_t(active-1)*C,f.rope.data()+size_t(active-1)*R,v.output.ptr(),&total,nullptr));
            else if(scope=="stream")for(int first=0;first<length;first+=128){auto before=Clock::now();
                check(api.rk_npu_mla_f16_cache_append(ctx,v.cache,f.latent.data()+size_t(first)*C,f.rope.data()+size_t(first)*R,128));
                total.kv_update_us+=std::chrono::duration<double,std::micro>(Clock::now()-before).count();
                for(int r=0;r<128;r+=v.chunk){const int n=std::min(v.chunk,128-r);rk_npu_mla_f16_timings t{};
                    check(api.rk_npu_mla_f16_run(ctx,v.work,v.cache,f.q.data()+size_t(first+r)*H*QK,f.gates.data()+size_t(first+r)*H,
                        n,first+r,v.output.ptr()+size_t(first+r)*H*D,&t,nullptr));add(total,t);}
            }
            else {const int count=scope=="prefill"?length:queries;const int start=scope=="prefill"?0:length-count;
                for(int first=0;first<count;first+=v.chunk){const int n=std::min(v.chunk,count-first);rk_npu_mla_f16_timings t{};
                    check(api.rk_npu_mla_f16_run(ctx,v.work,v.cache,f.q.data()+size_t(first)*H*QK,f.gates.data()+size_t(first)*H,
                        n,start+first,v.output.ptr()+size_t(first)*H*D,&t,nullptr));add(total,t);}
            }
            double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();auto values=v.output.values();
            if(verify)for(const auto& ref:references){const auto first=values.begin()+size_t(ref.first)*H*D;
                const std::vector<float> block(first,first+ref.second.size());double e=error(block,ref.second);require(e<(stable_softmax?.006:.08),"exact-softmax oracle");v.max_reference=std::max(v.max_reference,e);}
            if(iteration>=0)samples.push_back({v.name,v.mode,iteration,active,elapsed,total});
        }
        std::vector<float> baseline[2];for(auto& v:variants)if(v->backend.old)baseline[v->mode]=v->output.values();
        for(auto& vp:variants){auto& v=*vp;auto values=v.output.values();
            const auto& expected=baseline[v.mode];require(expected.size()==values.size(),"baseline shape");
            v.all_bitexact=v.all_bitexact && std::memcmp(values.data(),expected.data(),values.size()*4)==0;
            double e=error(values,expected);v.max_diff=std::max(v.max_diff,e);require(e<.003,"old/new full-output mismatch");
            if(scope!="step"){if(stable[v.mode].empty())stable[v.mode]=expected;else require(stable[v.mode]==expected,"baseline repeat stability");}
        }
    }
    for(const auto& v:variants)std::printf("CHECK variant=%s mode=%d bitexact=%d old_new_rel_l2=%.9f exact_reference=%.9f PASS\n",
        v->name.c_str(),v->mode,int(v->all_bitexact),v->max_diff,v->max_reference);
    std::puts("CHECK all_outputs guards/old_new_oracle/repeat_or_growth PASS");
    std::puts("variant,mode,iteration,length,compute_length,tasks,kv_update_us,query_projection_us,attention_us,value_projection_us,gate_output_us,prepare_us,tail_stage_us,mask_update_us,query_pack_us,submit_us,hardware_us,output_sync_us,finish_us,total_us");
    for(const auto& s:samples){const auto& t=s.t;const auto& a=t.attention;
        std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            s.variant.c_str(),s.mode,s.iteration,s.length,a.compute_length,a.tasks,t.kv_update_us,t.query_projection_us,t.attention_us,t.value_projection_us,t.gate_output_us,
            a.prepare_us,a.tail_stage_us,a.mask_update_us,a.query_pack_us,a.submit_us,a.hardware_us,a.output_sync_us,a.finish_us,s.total);
    }return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MLA COMPARE FAIL: %s\n",e.what());return 1;}}

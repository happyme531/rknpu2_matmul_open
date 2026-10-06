#include "mla_f16_test_utils.h"
#include <chrono>
#include <dlfcn.h>
#include <cstdlib>
#include <sstream>
using namespace mla_test;
using Clock=std::chrono::steady_clock;

// Every DSO owns its context and opaque handles. DEEPBIND keeps calls within
// the selected version; no private structs or cache handles cross versions.
#define FUNCTIONS(X) \
    X(rk_npu_open) X(rk_npu_close) X(rk_npu_iommu_domain_create) X(rk_npu_iommu_domain_free) \
    X(rk_npu_mla_f16_weights_create) X(rk_npu_mla_f16_weights_free) \
    X(rk_npu_mla_f16_cache_create) X(rk_npu_mla_f16_cache_free) X(rk_npu_mla_f16_cache_load) \
    X(rk_npu_mla_f16_prepare) X(rk_npu_mla_f16_workspace_free) X(rk_npu_mla_f16_run) X(rk_npu_mla_f16_decode_step) \
    X(rk_npu_attention_f16_cache_create) X(rk_npu_attention_f16_cache_free) X(rk_npu_attention_f16_cache_load) \
    X(rk_npu_attention_f16_prepare) X(rk_npu_attention_f16_workspace_free) X(rk_npu_attention_f16_run) X(rk_npu_attention_f16_decode_step)
struct Api {
    void* library;
#define FIELD(f) decltype(&::f) f=nullptr;
    FUNCTIONS(FIELD)
#undef FIELD
    explicit Api(const std::string& path):library(dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND)){
        if(!library)throw std::runtime_error(dlerror());
#define LOAD(f) f=reinterpret_cast<decltype(f)>(dlsym(library,#f));require(f,"missing DSO API");
        FUNCTIONS(LOAD)
#undef LOAD
    }
    ~Api(){dlclose(library);}
};
struct Backend {
    Api api;rk_npu_ctx* ctx;rk_npu_iommu_domain* domain;rk_npu_mla_f16_weights* weights=nullptr;
    Backend(const std::string& path,const Fixture& f,bool gqa):api(path),ctx(api.rk_npu_open(nullptr)),domain(ctx?api.rk_npu_iommu_domain_create(ctx,14):nullptr){
        require(ctx && domain,"backend context/domain");if(!gqa)check(api.rk_npu_mla_f16_weights_create(domain,f.w.data(),&weights));
    }
    ~Backend(){api.rk_npu_mla_f16_weights_free(weights);api.rk_npu_iommu_domain_free(domain);api.rk_npu_close(ctx);}
};
struct GqaFixture {
    int length;std::vector<uint16_t> q,k,v;std::vector<float> qf,kf,vf;
    explicit GqaFixture(int n):length(n),q(mla_test::random_data(32*128,501,.7f)),k(mla_test::random_data(size_t(8)*n*128,502,.7f)),v(mla_test::random_data(k.size(),503,.5f)),qf(widen(q)),kf(widen(k)),vf(widen(v)){}
    std::vector<uint16_t> prefix(const std::vector<uint16_t>& src,int n)const{
        std::vector<uint16_t> dst(size_t(8)*n*128);for(int h=0;h<8;++h)std::copy_n(src.data()+size_t(h)*length*128,size_t(n)*128,dst.data()+size_t(h)*n*128);return dst;
    }
    std::vector<float> reference(int n)const{
        std::vector<float> out(32*128),scores(n);
        for(int h=0;h<32;++h){float mx=-INFINITY,z=0;auto* dst=out.data()+h*128;
            for(int j=0;j<n;++j){scores[j]=dot(qf.data()+h*128,kf.data()+(size_t(h/4)*length+j)*128,128)/std::sqrt(128.f);mx=std::max(mx,scores[j]);}
            for(int j=0;j<n;++j){float e=std::exp(scores[j]-mx);z+=e;axpy(dst,vf.data()+(size_t(h/4)*length+j)*128,e,128);}
            for(int j=0;j<128;++j)dst[j]/=z;
        }return out;
    }
};
struct Variant {
    Backend& b;std::string name;bool gqa;int tile;
    rk_npu_attention_f16_cache* gc=nullptr;rk_npu_attention_f16_workspace* gw=nullptr;
    rk_npu_mla_f16_cache* mc=nullptr;rk_npu_mla_f16_workspace* mw=nullptr;
    GuardOutput output;double max_reference=0,max_diff=0;bool bitexact=true;
    std::vector<uint16_t> token_key=std::vector<uint16_t>(8*128),token_value=std::vector<uint16_t>(8*128);
    Variant(Backend& backend,const std::string& label,const std::string& kernel,int t,int core,int capacity,int length,const Fixture& f,const GqaFixture& g,int dirty_groups=0,const std::string& dirty_after="")
        :b(backend),name(label),gqa(kernel=="gqa"),tile(t),output(gqa?32*128:H*D){
        struct RestoreEnv {
            const char* key="RK_NPU_ATTN_DIRTY_SYNC_GROUPS";
            bool present=std::getenv(key)!=nullptr;
            std::string saved=present?std::getenv(key):"";
            ~RestoreEnv(){if(present)setenv(key,saved.c_str(),1);else unsetenv(key);}
        } restore,threshold{"RK_NPU_ATTN_DIRTY_SYNC_AFTER"};
        require((dirty_after.empty()?unsetenv(threshold.key):setenv(threshold.key,dirty_after.c_str(),1))==0,"set or clear dirty threshold");
        require((dirty_groups<0?unsetenv(restore.key):setenv(restore.key,dirty_groups==32?"span":std::to_string(dirty_groups).c_str(),1))==0,"set dirty sync experiment");
        if(gqa){rk_npu_attention_f16_config c;rk_npu_attention_f16_config_init(&c);c.flags=1;c.core_mask=core;c.kv_tile=t;c.initial_capacity=c.max_capacity=capacity;
            check(b.api.rk_npu_attention_f16_cache_create(b.domain,&c,&gc));check(b.api.rk_npu_attention_f16_prepare(b.domain,&c,&gw));
            auto k=g.prefix(g.k,length),v=g.prefix(g.v,length);check(b.api.rk_npu_attention_f16_cache_load(b.ctx,gc,k.data(),v.data(),length));
        }else{rk_npu_mla_f16_config c;rk_npu_mla_f16_config_init(&c);c.flags=1;c.core_mask=core;c.kv_tile=t;c.initial_capacity=c.max_capacity=capacity;
            c.mode=kernel=="expanded"?RK_NPU_MLA_F16_EXPANDED:RK_NPU_MLA_F16_ABSORBED;
            check(b.api.rk_npu_mla_f16_cache_create(b.domain,&c,b.weights,&mc));check(b.api.rk_npu_mla_f16_prepare(b.domain,&c,b.weights,&mw));
            check(b.api.rk_npu_mla_f16_cache_load(b.ctx,mc,f.latent.data(),f.rope.data(),length));
        }
    }
    ~Variant(){b.api.rk_npu_attention_f16_workspace_free(gw);b.api.rk_npu_attention_f16_cache_free(gc);b.api.rk_npu_mla_f16_workspace_free(mw);b.api.rk_npu_mla_f16_cache_free(mc);}
    void inputs(int length,const GqaFixture& g){
        if(gqa)for(int h=0;h<8;++h){
            std::copy_n(g.k.data()+(size_t(h)*g.length+length-1)*128,128,token_key.data()+h*128);
            std::copy_n(g.v.data()+(size_t(h)*g.length+length-1)*128,128,token_value.data()+h*128);
        }
    }
    rk_npu_attention_f16_timings run(bool step,int length,const Fixture& f,const GqaFixture& g){
        rk_npu_attention_f16_timings t{};
        if(gqa){check(step?b.api.rk_npu_attention_f16_decode_step(b.ctx,gw,gc,g.q.data(),token_key.data(),token_value.data(),output.ptr(),&t,nullptr):
                b.api.rk_npu_attention_f16_run(b.ctx,gw,gc,g.q.data(),1,length-1,output.ptr(),&t,nullptr));
        }else{rk_npu_mla_f16_timings mt{};check(step?b.api.rk_npu_mla_f16_decode_step(b.ctx,mw,mc,f.q.data(),f.gates.data(),f.latent.data()+size_t(length-1)*C,f.rope.data()+size_t(length-1)*R,output.ptr(),&mt,nullptr):
                b.api.rk_npu_mla_f16_run(b.ctx,mw,mc,f.q.data(),f.gates.data(),1,length-1,output.ptr(),&mt,nullptr));t=mt.attention;t.kv_update_us=mt.kv_update_us;
        }return t;
    }
};
int main(int argc,char** argv)try{
    std::string before,after,kernel="absorbed",scope="run",dirty_policies="0,1,2,4,span",dirty_after;int length=4096,loops=25,warmup=3,core=7,base_tile=512;bool tiles=true,dirty_sync=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];require(i+1<argc,"missing option value");std::string v=argv[++i];
        if(a=="--before")before=v;else if(a=="--after")after=v;else if(a=="--kernel")kernel=v;else if(a=="--scope")scope=v;
        else if(a=="--length")length=std::stoi(v);else if(a=="--loops")loops=std::stoi(v);else if(a=="--warmup")warmup=std::stoi(v);
        else if(a=="--core")core=std::stoi(v);else if(a=="--tiles")tiles=std::stoi(v)!=0;else if(a=="--tile")base_tile=std::stoi(v);
        else if(a=="--dirty-sync")dirty_sync=std::stoi(v)!=0;else if(a=="--dirty-policies")dirty_policies=v;else if(a=="--dirty-after")dirty_after=v;else throw std::runtime_error("unknown option");
    }
    const bool step=scope=="step",gqa=kernel=="gqa";const int total=length+(step?loops:0),capacity=(total+31)/32*32;
    require(!before.empty() && !after.empty() && (gqa || kernel=="expanded" || kernel=="absorbed"),"DSOs/kernel");
    require((scope=="run" || step) && length>0 && total<=32768 && loops>0 && warmup>=0 && (!step || warmup==0),"geometry/scope");
    if(!dirty_after.empty())require(dirty_sync && std::all_of(dirty_after.begin(),dirty_after.end(),[](char c){return c>='0' && c<='9';}) && std::stoll(dirty_after)<=2147483647,"dirty threshold tokens");
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Fixture f(total,1);GqaFixture g(total);
    Backend old(before,f,gqa),current(after,f,gqa);std::vector<std::unique_ptr<Variant>> variants;
    std::vector<int> policies{0};
    if(dirty_sync){policies.clear();std::istringstream parts(dirty_policies);std::string part;
        while(std::getline(parts,part,',')){require(part=="0" || part=="1" || part=="2" || part=="4" || part=="span","dirty sync policy");policies.push_back(part=="span"?32:std::stoi(part));}
        require(!policies.empty(),"empty dirty policies");}
    variants.emplace_back(new Variant(old,"before"+std::to_string(base_tile),kernel,base_tile,core,capacity,length,f,g));
    for(int tile:tiles?std::vector<int>{512,256,1024,2048,4096}:std::vector<int>{base_tile}){
        for(int groups:policies)
            variants.emplace_back(new Variant(current,"after"+std::to_string(tile)+(groups==32?"_dirtyspan":groups?"_dirty"+std::to_string(groups):""),kernel,tile,core,capacity,length,f,g,groups));
        if(!dirty_after.empty())variants.emplace_back(new Variant(current,"after"+std::to_string(tile)+"_dirtyafter"+dirty_after,kernel,tile,core,capacity,length,f,g,-1,dirty_after));
    }
    struct Sample{std::string variant;int iteration,length,tile;double total;rk_npu_attention_f16_timings t;};std::vector<Sample> samples;
    std::vector<float> reference,stable;
    for(int i=-warmup;i<loops;++i){const int active=length+(step?i+1:0);bool verify=i==-warmup || (step && (active==8193 || active==16385 || i==loops-1));
        if(verify)reference=gqa?g.reference(active):f.reference(active,1,active-1,RK_NPU_ATTENTION_F16_CAUSAL);
        // Reverse whole rotation cycles for stripe experiments, so an even
        // variant count does not pin each backend to even/odd positions.
        const size_t round=size_t(i+warmup);
        for(size_t j=0;j<variants.size();++j){size_t index=(j+round)%variants.size();if(dirty_sync?(round/variants.size())&1:round&1)index=variants.size()-1-index;auto& v=*variants[index];
            if(step)v.inputs(active,g);
            auto start=Clock::now();auto t=v.run(step,active,f,g);double elapsed=std::chrono::duration<double,std::micro>(Clock::now()-start).count();auto values=v.output.values();
            for(float x:values)require(std::isfinite(x),"nonfinite output");
            if(verify){double e=error(values,reference);v.max_reference=std::max(v.max_reference,e);require(e<(gqa?.15:.08),"exact-softmax reference");}
            if(i>=0)samples.push_back({v.name,i,active,v.tile,elapsed,t});
        }
        const auto baseline=variants[0]->output.values();if(!step){if(stable.empty())stable=baseline;else require(stable==baseline,"baseline repeat");}
        for(auto& v:variants){auto values=v->output.values();double e=error(values,baseline);v->max_diff=std::max(v->max_diff,e);v->bitexact=v->bitexact && values==baseline;
            require(e<.01,"tile/version full-output difference");if(v->tile==base_tile)require(values==baseline,"same-tile version bitexact");}
    }
    for(auto& v:variants)std::printf("CHECK variant=%s bitexact=%d max_diff=%.9f exact_reference=%.9f guards/finite/repeat_or_growth PASS\n",v->name.c_str(),int(v->bitexact),v->max_diff,v->max_reference);
    std::printf("META kernel=%s scope=%s core=%d capacity=%d\n",kernel.c_str(),scope.c_str(),core,capacity);
    std::puts("variant,iteration,length,tile,total_us,submit_us,hardware_us,prepare_us,mask_update_us,finish_us,kv_update_us,tasks,tail_stage_us,query_pack_us,output_sync_us,reserve_us");
    for(auto& s:samples)std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f\n",s.variant.c_str(),s.iteration,s.length,s.tile,s.total,s.t.submit_us,s.t.hardware_us,s.t.prepare_us,s.t.mask_update_us,s.t.finish_us,s.t.kv_update_us,s.t.tasks,s.t.tail_stage_us,s.t.query_pack_us,s.t.output_sync_us,s.t.reserve_us);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"DECODE COMPARE FAIL: %s\n",e.what());return 1;}

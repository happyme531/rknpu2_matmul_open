#include "../src/rk_npu_attention_f16_internal.h"
#include "mla_f16_test_utils.h"
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdarg>
#include <mutex>
#include <sys/syscall.h>
#include <thread>

namespace att=rknpu2_matmul_open::attention;
using mla_test::require;
using mla_test::check;
namespace {
// Test-only syscall interposition: reject a request before it reaches the
// driver, or pause a submit before issuing it. No failing NPU task is created.
std::atomic<unsigned long> failure_request{0};
std::atomic<int> failure_at{0},request_count{0};
std::atomic<int> failure_span{1},failure_errno{EIO};
std::atomic<bool> reject_next_submit{false};
std::atomic<bool> reject_wc{false};
std::atomic<int> wc_rejections{0};
// Only the caller thread of the serial fallback test modifies staged scores.
thread_local att::Buffer* nonfinite_score=nullptr;
std::mutex pause_mutex;
std::condition_variable pause_cv;
bool pause_submit=false,submit_entered=false,release_submit=false;
}
extern "C" int ioctl(int fd,unsigned long request,...)noexcept{
    va_list args;va_start(args,request);void* argument=va_arg(args,void*);va_end(args);
    if(request==IOCTL_MEM_CREATE && reject_wc.load() &&
        (static_cast<const rknpu_mem_create*>(argument)->flags & RK_NPU_MEM_WRITE_COMBINE)){
        ++wc_rejections;
        errno=EINVAL;return -1;
    }
    if(request==failure_request.load()){
        const int index=request_count.fetch_add(1)+1;
        const int first=failure_at.load();
        if(first>0 && index>=first && index-first<failure_span.load()){
            errno=failure_errno.load();return -1;
        }
    }
    if(request==IOCTL_SUBMIT){
        if(reject_next_submit.exchange(false)){errno=EIO;return -1;}
        std::unique_lock<std::mutex> lock(pause_mutex);
        if(pause_submit){
            submit_entered=true;pause_cv.notify_all();
            pause_cv.wait(lock,[]{return release_submit;});pause_submit=false;
        }
    }
    const int rc=int(syscall(SYS_ioctl,fd,request,argument));
    if(!rc && request==IOCTL_MEM_SYNC && nonfinite_score){
        const auto& sync=*static_cast<const rknpu_mem_sync*>(argument);
        if((sync.flags & RK_NPU_SYNC_FROM_DEVICE) && sync.obj_addr==nonfinite_score->mem.obj_addr){
            nonfinite_score->data()[0]=0x7c00;
            nonfinite_score=nullptr;
        }
    }
    return rc;
}
namespace {
struct Env {
    const char* key;bool present;std::string value;
    explicit Env(const char* k):key(k),present(std::getenv(k)),value(present?std::getenv(k):""){}
    ~Env(){if(present)setenv(key,value.c_str(),1);else unsetenv(key);}
};
using Cache=std::unique_ptr<rk_npu_attention_f16_cache,decltype(&rk_npu_attention_f16_cache_free)>;
using Work=std::unique_ptr<rk_npu_attention_f16_workspace,decltype(&rk_npu_attention_f16_workspace_free)>;
struct Idle {
    mla_test::Device& device;
    rk_npu_mem mem{};
    int reclaims=0;
    explicit Idle(mla_test::Device& dev):device(dev){
        check(rk_npu_mem_alloc(dev.domain,4096,RK_NPU_MEM_DATA_DEFAULT,&mem));
        rk_npu_iommu_domain_set_reclaim(dev.domain,[](void* ptr){
            auto& idle=*static_cast<Idle*>(ptr);
            if(idle.mem.handle && rk_npu_mem_free(idle.device.ctx,&idle.mem)==RK_NPU_OK)++idle.reclaims;
        },this);
    }
    ~Idle(){
        rk_npu_iommu_domain_set_reclaim(device.domain,nullptr,nullptr);
        if(mem.handle)rk_npu_mem_free(device.ctx,&mem);
    }
};
Cache cache(mla_test::Device& d,const rk_npu_attention_f16_config& c){
    rk_npu_attention_f16_cache* raw=nullptr;check(rk_npu_attention_f16_cache_create(d.domain,&c,&raw));
    return Cache(raw,rk_npu_attention_f16_cache_free);
}
Work work(mla_test::Device& d,const rk_npu_attention_f16_config& c,bool enabled){
    Env env("RK_NPU_ATTN_PIPELINE");if(enabled)unsetenv(env.key);else setenv(env.key,"0",1);
    rk_npu_attention_f16_workspace* raw=nullptr;check(rk_npu_attention_f16_prepare(d.domain,&c,&raw));
    return Work(raw,rk_npu_attention_f16_workspace_free);
}
void cpu(){
    Env env("RK_NPU_ATTN_PIPELINE");unsetenv(env.key);
    require(att::pipeline_policy(),"pipeline default enabled");
    rk_npu_attention_f16_config c;rk_npu_attention_f16_config_init(&c);
    c.flags=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;c.max_query_rows=128;c.core_mask=7;
    rk_npu_attention_f16_sizes on{},off{};
    check(rk_npu_attention_f16_query(&c,4096,128,4096,&on));
    setenv(env.key,"0",1);require(!att::pipeline_policy(),"pipeline disable token");
    check(rk_npu_attention_f16_query(&c,4096,128,4096,&off));
    require(on.query_bytes==off.query_bytes*2 && on.score_bytes==off.score_bytes*2 &&
        on.partial_bytes==off.partial_bytes*2 && on.tail_bytes==off.tail_bytes*2,"aggregate two-slot sizing");
    require(on.key_bytes==off.key_bytes && on.value_bytes==off.value_bytes &&
        on.output_bytes==off.output_bytes && on.tasks==off.tasks,"shared KV/output and unchanged GQA work");
    require(off.tail_bytes==uint64_t(c.kv_heads)*(c.head_dim+32)*(c.kv_tile-32)*2,
        "earlier causal CPU segments need a tail even at a full final tile");
    for(int rows:{1,4,32}){
        setenv(env.key,"0",1);check(rk_npu_attention_f16_query(&c,4096,rows,4096,&off));
        unsetenv(env.key);check(rk_npu_attention_f16_query(&c,4096,rows,4096,&on));
        require(on.query_bytes==off.query_bytes && on.score_bytes==off.score_bytes,"single-block sizing unchanged");
    }
    c.query_heads=c.kv_heads=16;c.head_dim=64;c.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;
    unsetenv(env.key);check(rk_npu_attention_f16_query(&c,1024,128,1024,&on));
    setenv(env.key,"0",1);check(rk_npu_attention_f16_query(&c,1024,128,1024,&off));
    require(on.physical_rows==64 && off.physical_rows==128,"pipeline/serial MHA row policy");
    require(on.score_bytes==off.score_bytes && on.query_bytes==off.query_bytes && on.tasks==off.tasks*2,
        "two half-sized MHA slots retain major buffer size");
    c.flags=RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL;
    check(rk_npu_attention_f16_query(&c,1024,128,1024,&off));
    setenv(env.key,"1",1);check(rk_npu_attention_f16_query(&c,1024,128,1024,&on));
    require(on.score_bytes==off.score_bytes && on.physical_rows==off.physical_rows,"fixed-shift remains serial");
    std::puts("Attention pipeline CPU policy/sizing PASS");
}
struct Fixture {
    rk_npu_attention_f16_config cfg{};
    int length,rows,start;
    std::vector<uint16_t> k,v,q;
    Fixture(int n=512,int r=128,int p=0,int hq=32,int hkv=8,int d=128):length(n),rows(r),start(p),
        k(mla_test::random_data(size_t(hkv)*n*d,17,.5f)),v(mla_test::random_data(k.size(),19,.5f)),
        q(mla_test::random_data(size_t(hq)*r*d,23,.5f)){
        rk_npu_attention_f16_config_init(&cfg);cfg.query_heads=hq;cfg.kv_heads=hkv;cfg.head_dim=d;
        cfg.max_query_rows=128;cfg.flags=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;cfg.core_mask=7;
        cfg.initial_capacity=32;cfg.max_capacity=align_up(n+128,32);
    }
    void load(mla_test::Device& dev,rk_npu_attention_f16_cache* c,const std::vector<uint16_t>& values){
        check(rk_npu_attention_f16_cache_load(dev.ctx,c,k.data(),values.data(),length));
    }
    std::vector<float> run(mla_test::Device& dev,rk_npu_attention_f16_workspace* w,
        rk_npu_attention_f16_cache* c,rk_npu_attention_f16_timings* timings=nullptr,
        const rk_npu_attention_f16_boolean_mask* mask=nullptr){
        mla_test::GuardOutput output(q.size());
        check(rk_npu_attention_f16_run(dev.ctx,w,c,q.data(),rows,start,output.ptr(),timings,mask));
        return output.values();
    }
};
void same(const std::vector<float>& a,const std::vector<float>& b){
    require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4),"pipeline/serial byte mismatch");
}
void compare_case(mla_test::Device& dev,Fixture& f,bool expect_pipeline=true){
    auto c=cache(dev,f.cfg),other=cache(dev,f.cfg);
    auto serial=work(dev,f.cfg,false),pipeline=work(dev,f.cfg,true);
    auto second=mla_test::random_data(f.v.size(),29,.4f);
    f.load(dev,c.get(),f.v);f.load(dev,other.get(),second);
    require(c->epoch==other->epoch,"same-epoch layer fixture");
    // Change the process policy after construction: existing handles retain it.
    Env env("RK_NPU_ATTN_PIPELINE");setenv(env.key,"0",1);
    for(auto* selected:{c.get(),other.get(),c.get()}){
        check(rk_npu_attention_f16_prepare_query(dev.ctx,pipeline.get(),selected,f.rows,f.start));
        rk_npu_attention_f16_timings t{};
        auto expected=f.run(dev,serial.get(),selected),actual=f.run(dev,pipeline.get(),selected,&t);same(expected,actual);
        require(bool(pipeline->pipeline)==expect_pipeline && !serial->pipeline,"captured policy and multi-block eligibility");
        require(t.total_us>0 && t.hardware_us>0 && t.tasks>0,"complete-call timings retained");
        std::vector<uint16_t> out(f.q.size()+16,0x35a5);
        check(rk_npu_attention_f16_run_half(dev.ctx,pipeline.get(),selected,f.q.data(),f.rows,f.start,out.data()+8,nullptr,nullptr));
        for(size_t i=0;i<expected.size();++i)require(out[i+8]==att::half(expected[i]),"pipeline FP16 output");
        for(int i=0;i<8;++i)require(out[i]==0x35a5 && out[out.size()-1-i]==0x35a5,"pipeline half guard");
    }
    check(rk_npu_attention_f16_cache_reserve(dev.ctx,c.get(),f.cfg.max_capacity));
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
}
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
// Feed native QK output directly so the exceptional encodings reach the CPU
// independently of how a particular MAC revision saturates its FP16 output.
void streamed_scores(mla_test::Device& dev,int kv_heads,uint32_t core_mask){
    Fixture f(256,32,3,4*kv_heads,kv_heads,64);
    f.cfg.kv_tile=256;f.cfg.core_mask=core_mask;
    Env lut("RK_NPU_ATTN_LUT_EXP");setenv(lut.key,"1",1);
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    Env mode("RK_NPU_ATTN_LUT_SUBMIT");setenv(mode.key,"pingpong",1);
    const auto shape=att::gqa_shape(f.cfg);
    att::Graph fast(dev.domain,f.cfg,32,f.length,*c,shape);
    require(fast.stream_scores && fast.tasks==fast.matmul_tasks,"streaming geometry must use only QK and PV");
    require(!fast.fallback_descriptors.mem.handle,"fallback commands should be unallocated");
    // Explicit PC mode runs the device exponent on the same geometry.
    setenv(mode.key,"pc",1);
    att::Graph reference(dev.domain,f.cfg,32,f.length,*c,shape);
    require(reference.lut && !reference.stream_scores,"DPU reference geometry");
    const size_t n=size_t(f.cfg.kv_heads)*fast.m*fast.span;
    auto scores=mla_test::random_data(n,53,7.f);
    scores[7]=0x7e00; // hidden in row zero
    uint32_t fallback_handle=0;
    for(uint16_t first : {0x4100,0x7c00,0x7e00,0x4100}){
        scores[0]=first;
        for(auto* g:{&reference,&fast}){
            std::copy(scores.begin(),scores.end(),g->score.data());
            const int executed=g->process_scores(f.start,nullptr,f.length);
            require(executed==(first==0x4100 && g==&fast?fast.matmul_tasks:reference.tasks),
                "executed task count");
            g->submit_pv(f.start);
        }
        require(fast.cpu_probabilities_ready==(first==0x4100),"exceptional row must use DPU");
        const auto handle=fast.fallback_descriptors.mem.handle;
        if(first==0x7c00){require(handle!=0,"fallback must create commands");fallback_handle=handle;}
        require(handle==fallback_handle,"fallback commands must be allocated only on first use");
        reference.exponent->range(0,n*2,RK_NPU_SYNC_FROM_DEVICE);
        require(!std::memcmp(reference.exponent->data(),fast.exponent->data(),n*2),"CPU/DPU exponent mismatch");
        require(reference.active==fast.active,"CPU/DPU active row mismatch");
    }
    std::printf("PASS native causal probabilities, nonfinite fallback and graph reuse core_mask=%u\n",core_mask);
}
void streamed_buffers(mla_test::Device& dev){
    Env lut("RK_NPU_ATTN_LUT_EXP");setenv(lut.key,"1",1);
    Env mode("RK_NPU_ATTN_LUT_SUBMIT");setenv(mode.key,"pingpong",1);
    Fixture f(512,65,0);
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto pipeline=work(dev,f.cfg,true);
    const auto expected=f.run(dev,pipeline.get(),c.get());
    require(pipeline->graph->cached_exponent.mem.handle && pipeline->graph->wc_exponent.mem.handle,
        "full and tail segments must retain both mappings");
    request_count=0;failure_at=0;failure_request=IOCTL_MEM_CREATE;
    const auto repeated=f.run(dev,pipeline.get(),c.get());
    failure_request=0;
    require(request_count.load()==0,"warm mixed geometry must reuse allocations");
    same(expected,repeated);

    auto fallback=work(dev,f.cfg,true);
    reject_wc=true;
    const auto actual=f.run(dev,fallback.get(),c.get());
    reject_wc=false;
    require(!fallback->graph->wc_exponent.mem.handle && !fallback->poisoned,
        "WC refusal must preserve the cached DPU path");
    same(expected,actual);
    same(expected,f.run(dev,fallback.get(),c.get()));
    std::puts("PASS streaming exponent buffer reuse and WC allocation fallback");
}
void serial_fallback_failures(mla_test::Device& dev){
    Env lut("RK_NPU_ATTN_LUT_EXP");setenv(lut.key,"1",1);
    Env mode("RK_NPU_ATTN_LUT_SUBMIT");setenv(mode.key,"pingpong",1);
    Fixture f(256,32,3);f.cfg.kv_tile=256;
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto prepared=[&]{
        auto w=work(dev,f.cfg,false);
        check(rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),f.rows,f.start));
        require(w->graph->stream_scores && !w->graph->fallback_tasks,"serial LUT fallback fixture");
        return w;
    };
    {
        auto w=prepared();nonfinite_score=&w->graph->score;
        const auto expected=f.run(dev,w.get(),c.get());
        require(!nonfinite_score && w->graph->fallback_tasks,"expected fallback after real QK");
        auto failed=prepared();std::vector<float> output(f.q.size());
        Idle idle(dev);
        request_count=0;failure_at=1;failure_span=2;failure_errno=ENOMEM;failure_request=IOCTL_MEM_CREATE;
        nonfinite_score=&failed->graph->score;
        const int rc=rk_npu_attention_f16_run(dev.ctx,failed.get(),c.get(),f.q.data(),
            f.rows,f.start,output.data(),nullptr,nullptr);
        failure_request=0;failure_span=1;failure_errno=EIO;
        require(!nonfinite_score && idle.reclaims==1 && request_count==2 &&
            rc==RK_NPU_ERR_IOCTL && !failed->poisoned,"LUT fallback allocation must remain recoverable");
        check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),nullptr,nullptr,0));f.load(dev,c.get(),f.v);
        nonfinite_score=&failed->graph->score;
        same(expected,f.run(dev,failed.get(),c.get()));
        require(!nonfinite_score && failed->graph->fallback_tasks,"same workspace completed LUT fallback");
        std::puts("PASS serial LUT fallback allocation and KV reset");
    }
    {
        const auto expected=f.run(dev,prepared().get(),c.get());
        auto w=work(dev,f.cfg,false);Idle idle(dev);
        wc_rejections=0;reject_wc=true;
        const auto actual=f.run(dev,w.get(),c.get());
        reject_wc=false;
        require(wc_rejections==1 && idle.reclaims==0 && idle.mem.handle &&
            !w->graph->wc_exponent.mem.handle,"WC EINVAL must fall back before reclaim");
        same(expected,actual);
        std::puts("PASS optional WC rejection attempted once and preserved idle GEM");
    }
}
#endif
void serial_graph_sync_failure(mla_test::Device& dev){
    Fixture f(512,32,480);
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto w=work(dev,f.cfg,false);
    // The aligned 512-token span has no staged tail; the first sync is
    // descriptor publication in the graph constructor.
    request_count=0;failure_at=1;failure_request=IOCTL_MEM_SYNC;
    const int rc=rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),f.rows,f.start);
    failure_request=0;
    require(rc==RK_NPU_ERR_IOCTL && w->poisoned && !w->graph,"serial graph sync must poison");
    check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),nullptr,nullptr,0));f.load(dev,c.get(),f.v);
    std::vector<float> output(f.q.size());
    require(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,
        output.data(),nullptr,nullptr)==RK_NPU_ERR_SUBMIT,"KV reset must preserve serial device protection");
    f.run(dev,work(dev,f.cfg,false).get(),c.get());
    std::puts("PASS serial graph sync failure stays protected after KV reset");
}
void failures(mla_test::Device& dev){
    Fixture f(4096,128,3968);auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto serial=work(dev,f.cfg,false);const auto expected=f.run(dev,serial.get(),c.get());
    std::vector<std::pair<unsigned long,int>> cases;
    for(auto operation : {IOCTL_SUBMIT, IOCTL_MEM_SYNC}) {
        auto probe=work(dev,f.cfg,true);
        check(rk_npu_attention_f16_prepare_query(dev.ctx,probe.get(),c.get(),f.rows,f.start));
        request_count=0;failure_at=0;failure_request=operation;
        auto got=f.run(dev,probe.get(),c.get());
        failure_request=0;const int observed=request_count.load();
        require(observed>0,"operation missing");same(expected,got);
        if(operation==IOCTL_SUBMIT)for(int i=1;i<=observed;++i)cases.emplace_back(operation,i);
        else for(int i : {2,4,observed})cases.emplace_back(operation,i);
    }
    for(auto spec:cases){
        auto w=work(dev,f.cfg,true);
        check(rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),f.rows,f.start));
        request_count=0;failure_at=spec.second;failure_request=spec.first;
        std::vector<float> output(f.q.size());
        const int rc=rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,output.data(),nullptr,nullptr);
        failure_request=0;
        require(rc==(spec.first==IOCTL_SUBMIT?RK_NPU_ERR_SUBMIT:RK_NPU_ERR_IOCTL),"injected failure returned");
        f.load(dev,c.get(),f.v);
        require(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,output.data(),nullptr,nullptr)==RK_NPU_ERR_SUBMIT,"failed pipeline poisoned");
        rk_npu_attention_f16_cache_info info{};check(rk_npu_attention_f16_cache_get_info(c.get(),&info));
        w.reset();auto fresh=work(dev,f.cfg,true);same(expected,f.run(dev,fresh.get(),c.get()));
        std::printf("PASS pipeline drained failed request=%lu index=%d\n",spec.first,spec.second);
    }
}
void allocation_failures(mla_test::Device& dev){
    Env lut("RK_NPU_ATTN_LUT_EXP");setenv(lut.key,"0",1);
    Fixture f(512,128,384);auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto serial=work(dev,f.cfg,false);
    const auto expected=f.run(dev,serial.get(),c.get());
    int block_rows=0;rk_npu_attention_f16_sizes sizes{};
    check(att::query_gqa(f.cfg,c->capacity,f.rows,f.length,sizes,block_rows,true));
    require(block_rows<f.rows,"allocation fixture requires multiple segments");
    {
        auto first=work(dev,f.cfg,true);auto cold_cache=cache(dev,f.cfg);
        f.load(dev,cold_cache.get(),f.v);
        request_count=0;failure_at=0;failure_request=IOCTL_MEM_CREATE;
        check(rk_npu_attention_f16_prepare_query(dev.ctx,first.get(),cold_cache.get(),block_rows,f.start));
        failure_request=0;
    }
    // The same first geometry runs before launch; the next allocation is
    // for slot 1 while the first slot's QK has been handed to the worker.
    const int second_slot=request_count.load()+1;
    require(second_slot>1,"first slot did not allocate");
    for(const auto [prepare_only,count,submit_error,error] : {
        std::tuple{true,2,false,ENOMEM},std::tuple{false,1,false,ENOMEM},
        std::tuple{false,2,false,ENOMEM},std::tuple{false,2,true,ENOMEM},
        std::tuple{false,1,false,EINVAL}}){
        auto w=work(dev,f.cfg,true);
        std::vector<float> output(f.q.size());
        Idle idle(dev);
        request_count=0;failure_at=second_slot;failure_span=count;failure_errno=error;
        reject_next_submit=submit_error;failure_request=IOCTL_MEM_CREATE;
        auto run=[&]{return rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),
            f.rows,f.start,output.data(),nullptr,nullptr);};
        const int rc=prepare_only?rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),
            f.rows,f.start):run();
        failure_request=0;failure_span=1;failure_errno=EIO;
        require(!reject_next_submit.exchange(false),"submit fault did not reach the worker");
        require(w->pipeline && idle.reclaims==1 && !idle.mem.handle &&
            request_count.load()>=second_slot+count-1,"slot 1 allocation did not reclaim/retry");
        const int expected_rc=submit_error?RK_NPU_ERR_SUBMIT:count==1?RK_NPU_OK:RK_NPU_ERR_IOCTL;
        require(rc==expected_rc,"allocation/submit error returned");
        if(count==1)same(expected,output);
        check(rk_npu_attention_f16_cache_load(dev.ctx,c.get(),nullptr,nullptr,0));
        f.load(dev,c.get(),f.v);
        if(submit_error)require(run()==RK_NPU_ERR_SUBMIT,"worker failure must still poison the workspace");
        else same(expected,f.run(dev,w.get(),c.get()));
        std::printf("PASS pipeline allocation prepare=%d failures=%d errno=%d submit_failure=%d rc=%d\n",
            prepare_only,count,error,submit_error,rc);
    }
}
void busy(mla_test::Device& dev){
    Fixture f(512,128,384);auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto w=work(dev,f.cfg,true),serial=work(dev,f.cfg,false);auto expected=f.run(dev,serial.get(),c.get());
    check(rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),f.rows,f.start));
    {std::lock_guard<std::mutex> lock(pause_mutex);pause_submit=true;submit_entered=release_submit=false;}
    std::vector<float> actual(f.q.size());int rc=-1;
    std::thread caller([&]{rc=rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,actual.data(),nullptr,nullptr);});
    {std::unique_lock<std::mutex> lock(pause_mutex);pause_cv.wait(lock,[]{return submit_entered;});}
    std::vector<float> untouched(f.q.size(),123);
    const int competing=rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,untouched.data(),nullptr,nullptr);
    const int update=rk_npu_attention_f16_cache_append(dev.ctx,c.get(),nullptr,nullptr,0);
    {std::lock_guard<std::mutex> lock(pause_mutex);release_submit=true;pause_cv.notify_all();}
    caller.join();check(rc);same(expected,actual);
    require(competing==RK_NPU_ERR_BUSY && update==RK_NPU_ERR_BUSY,"pipeline holds workspace and cache locks");
    require(std::all_of(untouched.begin(),untouched.end(),[](float x){return x==123;}),"busy call changed output");
    std::puts("PASS pipeline concurrent BUSY and lifetime");
}
void mutations(mla_test::Device& dev){
    for(bool causal:{true,false}){
    Fixture f(641,128,513);if(!causal)f.cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto serial=work(dev,f.cfg,false),pipeline=work(dev,f.cfg,true);
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    auto k=mla_test::random_data(size_t(f.cfg.kv_heads)*f.cfg.head_dim,41,.5f);
    auto v=mla_test::random_data(k.size(),43,.5f);
    if(!causal)std::fill(v.begin(),v.end(),att::half(65000));
    check(rk_npu_attention_f16_cache_append(dev.ctx,c.get(),k.data(),v.data(),1));++f.start;
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    check(rk_npu_attention_f16_cache_truncate(c.get(),f.length));--f.start;
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    auto replacement=mla_test::random_data(f.v.size(),47,.4f);f.load(dev,c.get(),replacement);
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    }
    // Reload a shorter sequence without changing align32 tail geometry. Its
    // discarded high V must not survive as a LUT carrier in a slot-owned tail.
    Fixture f(130,128,0);f.cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;
    for(int h=0;h<f.cfg.kv_heads;++h)std::fill_n(f.v.data()+(size_t(h)*130+129)*f.cfg.head_dim,
        f.cfg.head_dim,att::half(65000));
    auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto serial=work(dev,f.cfg,false),pipeline=work(dev,f.cfg,true);
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    std::vector<uint16_t> k(size_t(f.cfg.kv_heads)*129*f.cfg.head_dim),v(k.size());
    for(int h=0;h<f.cfg.kv_heads;++h){
        std::copy_n(f.k.data()+size_t(h)*130*f.cfg.head_dim,129*f.cfg.head_dim,k.data()+size_t(h)*129*f.cfg.head_dim);
        std::copy_n(f.v.data()+size_t(h)*130*f.cfg.head_dim,129*f.cfg.head_dim,v.data()+size_t(h)*129*f.cfg.head_dim);
    }
    f.k=std::move(k);f.v=std::move(v);f.length=129;f.load(dev,c.get(),f.v);
    same(f.run(dev,serial.get(),c.get()),f.run(dev,pipeline.get(),c.get()));
    std::puts("PASS pipeline incremental tail, truncate and reload");
}
void boolean(mla_test::Device& dev){
    Env lut("RK_NPU_ATTN_LUT_EXP");
    for(bool use_lut:{true,false}){
        setenv(lut.key,use_lut?"1":"0",1);
        Fixture f(513,128,0);f.cfg.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
        auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
        auto serial=work(dev,f.cfg,false),pipeline=work(dev,f.cfg,true);
        const int stride=f.length*2+17;
        std::vector<uint8_t> data(size_t(f.rows+3)*stride);
        for(int r=0;r<f.rows+3;++r)for(int k=0;k<f.length;++k)
            data[size_t(r)*stride+k*2]=(r%19 && k%7)?1:0;
        rk_npu_attention_f16_boolean_mask mask{};
        rk_npu_attention_f16_boolean_mask_init(&mask,data.data(),data.size(),1,f.rows+3,f.length);
        mask.query_stride=stride;mask.key_stride=2;mask.query_offset=3;
        auto expected=f.run(dev,serial.get(),c.get(),nullptr,&mask);
        same(expected,f.run(dev,pipeline.get(),c.get(),nullptr,&mask));
        const auto bytes=mask.bytes;mask.bytes=1;std::vector<float> output(f.q.size(),123);
        require(rk_npu_attention_f16_run(dev.ctx,pipeline.get(),c.get(),f.q.data(),f.rows,f.start,
            output.data(),nullptr,&mask)==RK_NPU_ERR_PARAM,"pipeline mask validated before writes");
        require(std::all_of(output.begin(),output.end(),[](float x){return x==123;}),"invalid pipeline mask changed output");
        mask.bytes=bytes;same(expected,f.run(dev,pipeline.get(),c.get(),nullptr,&mask));
        std::printf("PASS pipeline strided Boolean mask / hidden rows lut=%d\n",int(use_lut));
    }
}
void board(){
    mla_test::BoardLock lock("/tmp/rknpu_lowlevel_submit.lock"),legacy("/tmp/rk3588_npu_submit.lock");
    mla_test::Device dev;
    for(int rows:{33,65,128}){Fixture f(512,rows,0);compare_case(dev,f);}
    for(int start:{0,128,256}){
        Fixture f(512,128,start,16,4,32);
        for(int h=0;h<4;++h)for(int pos=384;pos<512;++pos)
            std::fill_n(f.v.data()+(size_t(h)*512+pos)*32,32,att::half(65000));
        compare_case(dev,f);
    }
    {Fixture f(1024,128,480);compare_case(dev,f);}
    {Fixture f(4096,128,3968,16,4,256);compare_case(dev,f);}
    {Fixture f(1024,128,0,16,16,64);f.cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;compare_case(dev,f);}
    {Fixture f(4096,1,4095);compare_case(dev,f,false);}
    {Fixture f(4096,4,4092);compare_case(dev,f,false);}
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    streamed_scores(dev,1,1);streamed_scores(dev,2,3);streamed_scores(dev,4,7);
    streamed_buffers(dev);serial_fallback_failures(dev);
#else
    std::puts("SKIP streaming probabilities and buffer reuse: FP16 NEON unavailable");
#endif
    mutations(dev);boolean(dev);serial_graph_sync_failure(dev);failures(dev);allocation_failures(dev);busy(dev);
    std::puts("Attention pipeline board contract PASS");
}
}
int main(int argc,char** argv)try{
    cpu();if(argc>1 && std::string(argv[1])=="--board")board();return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

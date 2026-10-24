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
std::mutex pause_mutex;
std::condition_variable pause_cv;
bool pause_submit=false,submit_entered=false,release_submit=false;
}
extern "C" int ioctl(int fd,unsigned long request,...)noexcept{
    va_list args;va_start(args,request);void* argument=va_arg(args,void*);va_end(args);
    if(request==failure_request.load() && request_count.fetch_add(1)+1==failure_at.load()){
        errno=EIO;return -1;
    }
    if(request==IOCTL_SUBMIT){
        std::unique_lock<std::mutex> lock(pause_mutex);
        if(pause_submit){
            submit_entered=true;pause_cv.notify_all();
            pause_cv.wait(lock,[]{return release_submit;});pause_submit=false;
        }
    }
    return int(syscall(SYS_ioctl,fd,request,argument));
}
namespace {
struct Env {
    const char* key;bool present;std::string value;
    explicit Env(const char* k):key(k),present(std::getenv(k)),value(present?std::getenv(k):""){}
    ~Env(){if(present)setenv(key,value.c_str(),1);else unsetenv(key);}
};
using Cache=std::unique_ptr<rk_npu_attention_f16_cache,decltype(&rk_npu_attention_f16_cache_free)>;
using Work=std::unique_ptr<rk_npu_attention_f16_workspace,decltype(&rk_npu_attention_f16_workspace_free)>;
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
void failures(mla_test::Device& dev){
    Fixture f(4096,128,3968);auto c=cache(dev,f.cfg);f.load(dev,c.get(),f.v);
    auto serial=work(dev,f.cfg,false);const auto expected=f.run(dev,serial.get(),c.get());
    for(auto spec:{std::make_pair(IOCTL_SUBMIT,1),{IOCTL_SUBMIT,3},{IOCTL_SUBMIT,6},
                   {IOCTL_SUBMIT,10},{IOCTL_MEM_SYNC,2},{IOCTL_MEM_SYNC,4}}){
        auto w=work(dev,f.cfg,true);
        check(rk_npu_attention_f16_prepare_query(dev.ctx,w.get(),c.get(),f.rows,f.start));
        request_count=0;failure_at=spec.second;failure_request=spec.first;
        std::vector<float> output(f.q.size());
        const int rc=rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,output.data(),nullptr,nullptr);
        failure_request=0;
        require(rc==(spec.first==IOCTL_SUBMIT?RK_NPU_ERR_SUBMIT:RK_NPU_ERR_IOCTL),"injected failure returned");
        require(rk_npu_attention_f16_run(dev.ctx,w.get(),c.get(),f.q.data(),f.rows,f.start,output.data(),nullptr,nullptr)==RK_NPU_ERR_SUBMIT,"failed pipeline poisoned");
        rk_npu_attention_f16_cache_info info{};check(rk_npu_attention_f16_cache_get_info(c.get(),&info));
        w.reset();auto fresh=work(dev,f.cfg,true);same(expected,f.run(dev,fresh.get(),c.get()));
        std::printf("PASS pipeline drained failed request=%lu index=%d\n",spec.first,spec.second);
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
    mutations(dev);boolean(dev);failures(dev);busy(dev);
    std::puts("Attention pipeline board contract PASS");
}
}
int main(int argc,char** argv)try{
    cpu();if(argc>1 && std::string(argv[1])=="--board")board();return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}

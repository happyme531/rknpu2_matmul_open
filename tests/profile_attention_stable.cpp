// Component diagnostic only. Complete-call comparisons use compare_mla_f16.
#include "rk_npu_attention_f16_internal.h"
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <random>
#include <sys/file.h>
#include <unistd.h>
using namespace rknpu2_matmul_open::attention;
struct BoardLock {
    int fd;
    explicit BoardLock(const char* path):fd(open(path,O_CREAT|O_RDWR,0666)){
        if(fd<0 || flock(fd,LOCK_EX|LOCK_NB))throw std::runtime_error("NPU lock busy");
    }
    ~BoardLock(){flock(fd,LOCK_UN);close(fd);}
};
int main(int argc,char** argv)try{
    const int length=argc>1?std::stoi(argv[1]):4096,rows=argc>2?std::stoi(argv[2]):128;
    const int loops=argc>3?std::stoi(argv[3]):10;
    if(length<32 || length>32768 || length%32 || rows<1 || rows>128 || loops<1)
        throw std::runtime_error("length32..32768 aligned32, rows1..128, loops>0");
    BoardLock first("/tmp/rknpu_lowlevel_submit.lock"),second("/tmp/rk3588_npu_submit.lock");
    auto* ctx=rk_npu_open(nullptr);
    auto* domain=ctx?rk_npu_iommu_domain_create(ctx,14):nullptr;
    if(!domain)throw std::runtime_error("open/domain");
    {
        rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);
        cfg.query_heads=cfg.kv_heads=16;cfg.max_query_rows=128;cfg.core_mask=7;
        cfg.initial_capacity=cfg.max_capacity=length;cfg.flags=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
        cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;
        Shape shape;shape.key_dim=192;shape.value_dim=128;shape.group=1;shape.scale=1/std::sqrt(192.f);
        Cache cache(domain,cfg,shape);
        std::mt19937 random(42);std::normal_distribution<float> distribution(0,.7);
        std::vector<uint16_t> k(size_t(16)*length*192),v(size_t(16)*length*128),q(size_t(16)*rows*192);
        for(auto* data:{&k,&v,&q})for(auto& x:*data)x=half(distribution(random));
        cache.load(k.data(),v.data(),length);cache.stage_tail(length);
        Graph graph(domain,cfg,rows,length,cache,shape);std::vector<float> output(size_t(16)*rows*128);
        std::puts("iteration,qk_us,read_sync_us,softmax_us,write_sync_us,pv_us,output_sync_us,finish_us,hardware_us,total_us");
        for(int iteration=-3;iteration<loops;++iteration){
            graph.pack_q(q.data());auto begin=Clock::now();
            double hardware=graph.submit();auto qk=Clock::now();
            graph.score.range(0,size_t(16)*graph.m*length*2,RK_NPU_SYNC_FROM_DEVICE);auto read=Clock::now();
            graph.stable_softmax(0,nullptr,length);auto softmax=Clock::now();
            graph.score.range(0,size_t(16)*graph.m*length*2);auto write=Clock::now();
            hardware+=graph.submit(1);auto pv=Clock::now();
            graph.partial.range(0,size_t(16)*graph.blocks*160*graph.m*2,RK_NPU_SYNC_FROM_DEVICE);auto sync=Clock::now();
            graph.finish(output.data());auto finish=Clock::now();
            for(float x:output)if(!std::isfinite(x))throw std::runtime_error("nonfinite output");
            if(iteration>=0)std::printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",iteration,
                elapsed(begin,qk),elapsed(qk,read),elapsed(read,softmax),elapsed(softmax,write),elapsed(write,pv),
                elapsed(pv,sync),elapsed(sync,finish),hardware,elapsed(begin,finish));
        }
    }
    rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);return 0;
}catch(const Error& e){std::fprintf(stderr,"PROFILE FAIL: %s\n",rk_npu_strerror(e.code));return 1;}
catch(const std::exception& e){std::fprintf(stderr,"PROFILE FAIL: %s\n",e.what());return 1;}

// CPU-only regression for ARM64-safe command memory and DMA visibility.
// Separate CPU/device copies make a missing cache flush observable at submit.
#include "../src/rk_npu_attention_f16_internal.h"
#include <cstdarg>
#include <cstdio>
#include <map>

using namespace rknpu2_matmul_open::attention;
namespace {
struct Memory {std::vector<unsigned char> cpu,device;uint32_t flags;};
std::map<uint32_t,Memory> memory;
uint32_t serial=0;
int flushes=0,submits=0;
Graph* active=nullptr;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void visible(const Buffer& buffer){
    const auto& m=memory.at(buffer.mem.handle);
    require(m.flags&RK_NPU_MEM_CACHEABLE,"command memory must support ordinary CPU writes");
    require(m.flags&RK_NPU_MEM_NON_CONTIGUOUS,"command memory must stay page-backed");
    require(m.cpu==m.device,"submit observed stale CPU command/descriptor writes");
}
uint32_t reg(const Graph& graph,const rknpu_task& task,uint16_t target,uint16_t address){
    const auto* words=static_cast<const uint64_t*>(graph.regs.mem.vaddr)+
        (task.regcmd_addr-graph.regs.mem.dma_addr)/8;
    for(uint32_t i=0;i<task.regcfg_amount;++i)
        if(uint16_t(words[i]>>48)==target && uint16_t(words[i])==address)return uint32_t(words[i]>>16);
    throw std::runtime_error("missing LUT register");
}
void lut_geometry(rk_npu_iommu_domain& domain){
    rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);
    cfg.initial_capacity=cfg.max_capacity=4096;cfg.query_heads=4;cfg.kv_heads=1;
    cfg.flags=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    Cache cache(&domain,cfg,gqa_shape(cfg));cache.lut_exp=true;
    for(int rows:{1,31,32})for(int span:{32,64,512,4096}){
        cache.length=span;cache.stage_tail(span);
        Graph graph(&domain,cfg,rows,span,cache,gqa_shape(cfg));
        require(graph.lut,"LUT geometry fixture must use device exp");
        const int chunk=graph.m==128?65536:4096;
        const int setup=graph.lut_split?1:0;
        require(graph.tasks==graph.matmul_tasks+ceil_div(graph.m*span,chunk)+setup,"LUT task count");
        const auto* tasks=static_cast<const rknpu_task*>(graph.descriptors.mem.vaddr);
        uint64_t values=0;
        for(int i=0;i<graph.tasks;++i)if(tasks[i].op_idx==4){
            const auto& task=tasks[i];
            const unsigned w=reg(graph,task,T_DPU,0x4030),h=reg(graph,task,T_DPU,0x4034);
            const unsigned count=(w+1)*(h+1)*8;
            require(w<8192 && h<8192 && count<=unsigned(chunk),"bounded LUT cube");
            require(reg(graph,task,T_DPU,0x403c)==0x70007,"eight-channel LUT cube");
            require(reg(graph,task,T_DPU,0x405c)==((h<<16)|w),"LUT WDMA must retain both dimensions");
            require(reg(graph,task,T_RDMA,0x500c)==w && reg(graph,task,T_RDMA,0x5010)==h,
                "LUT RDMA must match DPU geometry");
            require(reg(graph,task,T_DPU,0x4024)==count*2 && reg(graph,task,T_DPU,0x40c0)==count*2,
                "LUT byte strides must cover the full cube");
            require(count>4096?h>0:h==0,"large LUT tiles require a two-dimensional cube");
            values+=count;
        }
        require(values==uint64_t(graph.m)*span+setup*8,"LUT tiles must cover every score exactly once");
        visible(graph.regs);visible(graph.descriptors);
    }
}
}
extern "C" int __wrap_rk_npu_mem_alloc(rk_npu_iommu_domain* d,uint64_t bytes,uint32_t flags,rk_npu_mem* out){
    const auto handle=++serial;
    auto& m=memory.emplace(handle,Memory{std::vector<unsigned char>(bytes),std::vector<unsigned char>(bytes),flags}).first->second;
    *out={};out->handle=handle;out->obj_addr=handle;out->vaddr=m.cpu.data();
    out->size=out->capacity=bytes;out->dma_addr=uint64_t(handle)*0x100000;
    out->ctx_id=d->ctx->id;out->iommu_domain_id=d->id;
    if(flags&RK_NPU_MEM_CACHEABLE)out->flags|=RK_NPU_MEM_F_CACHEABLE;
    return RK_NPU_OK;
}
extern "C" int __wrap_rk_npu_mem_free(rk_npu_ctx*,rk_npu_mem* mem){
    memory.erase(mem->handle);*mem={};return RK_NPU_OK;
}
extern "C" int __wrap_rk_npu_mem_sync(rk_npu_ctx*,const rk_npu_mem* mem,rk_npu_sync_dir dir){
    require(dir==RK_NPU_SYNC_TO_DEVICE,"unexpected sync direction");
    auto& m=memory.at(uint32_t(mem->obj_addr));
    std::copy_n(m.cpu.begin()+mem->offset,mem->size,m.device.begin()+mem->offset);
    ++flushes;return RK_NPU_OK;
}
extern "C" int __wrap_ioctl(int,unsigned long request,...){
    require(request==IOCTL_SUBMIT && active,"unexpected device ioctl");
    visible(active->regs);visible(active->descriptors);
    require(memory.at(active->descriptors.mem.handle).flags&RK_NPU_MEM_KERNEL_MAPPING,"driver needs task kernel mapping");
    ++submits;return 0;
}
int main(){
    try {
        rk_npu_ctx ctx{};ctx.id=1;
        rk_npu_iommu_domain domain{};domain.ctx=&ctx;domain.refs.store(1);
        rk_npu_attention_f16_config cfg;rk_npu_attention_f16_config_init(&cfg);
        cfg.initial_capacity=cfg.max_capacity=128;cfg.query_heads=4;cfg.kv_heads=1;
        cfg.head_dim=128;cfg.kv_tile=32;cfg.flags=0;
        {
            Cache cache(&domain,cfg,gqa_shape(cfg));cache.length=32;
            Graph graph(&domain,cfg,4,32,cache,gqa_shape(cfg));
            active=&graph;graph.submit();
            const auto first=memory.at(graph.regs.mem.handle).device;
            cache.key.mem.dma_addr+=4096;++cache.generation;
            graph.bind(cache);graph.submit();
            require(first!=memory.at(graph.regs.mem.handle).device,"rebind did not change command addresses");
            const int before=flushes;graph.bind(cache);graph.submit();
            require(before==flushes,"unchanged commands should not be flushed again");
            Graph grown(&domain,cfg,8,64,cache,gqa_shape(cfg),&graph);
            active=&grown;grown.submit();active=nullptr;
        }
        lut_geometry(domain);
        require(memory.empty() && submits==4,"mock memory leaked or missing submits");
        std::puts("PASS: page-backed CPU-safe commands, initial flush, rebind flush, graph reuse");
        std::puts("PASS: flat/2D LUT command geometry, task counts and score coverage");
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
      catch(const Error& e){std::fprintf(stderr,"FAIL: NPU error %d\n",e.code);return 1;}
}

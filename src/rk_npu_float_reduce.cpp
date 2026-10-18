#include "rk_npu_float_reduce.h"
#include "rk_npu_internal.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

namespace rknpu2_matmul_open::detail {
namespace {
// DPU/RDMA FP16 Add recipe shared with the validated Add+RMSNorm first task.
// Tensor view is [channels/atom, width, atom], with 8 FP16 elements.
// Flattening does not change elementwise Add.
std::vector<uint64_t> add_regs(int width, int channels, uint64_t a, uint64_t b, uint64_t c) {
    std::vector<uint64_t> v;
    auto d = [&](int reg, uint32_t value) { v.push_back(E(T_DPU, reg, value)); };
    auto r = [&](int reg, uint32_t value) { v.push_back(E(T_RDMA, reg, value)); };
    d(0x4004,14); r(0x5004,14); d(0x400c,0x1e5); d(0x4010,0x48000002);
    d(0x4014,0); d(0x4020,uint32_t(c)); d(0x4024,width*16);
    d(0x4030,width-1); d(0x4034,0); d(0x4038,0);
    d(0x403c,((channels-1)<<16)|(channels-1));
    d(0x4040,0x53); d(0x4044,0); d(0x4048,0); d(0x404c,0);
    d(0x4050,2); d(0x4054,0); d(0x4058,channels-1); d(0x405c,width-1);
    d(0x4060,0x53); d(0x4064,0); d(0x4068,0); d(0x406c,0);
    d(0x4070,(width>1?0x10000000:0)|0x8202c0); d(0x4074,0); d(0x4078,1); d(0x407c,0); d(0x4080,0);
    d(0x4084,0x10001); d(0x4088,0);
    for (int reg=0x4090;reg<=0x40ac;reg+=4) d(reg,0);
    d(0x40c0,width*16); d(0x40c4,0);
    for (int reg=0x4100;reg<=0x412c;reg+=4) d(reg,0);
    r(0x500c,width-1); r(0x5010,0); r(0x5014,channels-1); r(0x5018,uint32_t(a));
    r(0x501c,0); r(0x5020,0); r(0x5028,0); r(0x502c,0);
    r(0x5034,(width>1?0x40000000:0)|8); r(0x5038,uint32_t(b)); r(0x5040,width*16); r(0x5044,0x17849);
    r(0x5048,0); r(0x504c,0); r(0x5064,0); r(0x5068,0x1010101); r(0x506c,0);
    return v;
}
}
struct FloatReduce::Impl {
    struct Level {
        rk_npu_mem regs{}, tasks{};
        uint32_t starts[3]{}, counts[3]{};
        uint32_t mask=1;
        int count=0;
    };
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mem scratch{}, zero{}, result{};
    std::vector<std::unique_ptr<Level>> levels;
    uint32_t timeout=500;
    ~Impl() {
        if (!domain) return;
        for (auto& level:levels) {
            rk_npu_mem_free(domain->ctx,&level->regs);
            rk_npu_mem_free(domain->ctx,&level->tasks);
        }
        rk_npu_mem_free(domain->ctx,&scratch); rk_npu_mem_free(domain->ctx,&zero);
        release_domain(domain);
    }
};
FloatReduce::FloatReduce()=default;
FloatReduce::~FloatReduce()=default;
int FloatReduce::prepare(rk_npu_iommu_domain* domain,const rk_npu_mem& partials,
    int count,uint64_t slab,uint32_t mask,uint32_t timeout) {
    if (p_ || !domain || !domain->ctx || count<2 || count>32 || !slab || slab%64 ||
        (mask!=1 && mask!=2 && mask!=4 && mask!=3 && mask!=7) ||
        partials.ctx_id!=domain->ctx->id || partials.iommu_domain_id!=domain->id ||
        slab>UINT32_MAX/uint64_t(count) || partials.size<slab*count ||
        partials.dma_addr>(1ull<<32)-slab*count) return RK_NPU_ERR_PARAM;
    try {
        auto p=std::make_unique<Impl>(); p->domain=domain; retain_domain(domain);
        p->timeout=timeout?timeout:500;
        int rc=rk_npu_mem_alloc(domain,slab*ceil_div(count,2),RK_NPU_MEM_DATA_DEFAULT,&p->scratch);
        if (!rc) rc=rk_npu_mem_alloc(domain,slab,RK_NPU_MEM_DATA_DEFAULT,&p->zero);
        if (rc) return rc;
        std::memset(p->zero.vaddr,0,slab);
        rc=rk_npu_mem_sync(domain->ctx,&p->zero,RK_NPU_SYNC_TO_DEVICE);
        if (rc) return rc;
        const rk_npu_mem *input=&partials,*output=&p->scratch;
        for (int live=count;live>1;live=ceil_div(live,2)) {
            auto level=std::make_unique<Impl::Level>();
            std::vector<std::vector<uint64_t>> bodies;
            std::vector<int> bases;
            int offset=0;
            for (int pair=0;pair<ceil_div(live,2);++pair) {
                uint64_t at=0;
                while (at<slab) {
                    constexpr int bytes=2;
                    const uint64_t elements=(slab-at)/bytes;
                    const int width=elements>=128*32 ? 128:1;
                    const int channels=width==128?std::min<uint64_t>(4096,elements/128/32*32):std::min<uint64_t>(4096,elements);
                    const auto a=input->dma_addr+uint64_t(pair*2)*slab+at;
                    const auto b=pair*2+1<live?input->dma_addr+uint64_t(pair*2+1)*slab+at:p->zero.dma_addr+at;
                    const auto c=output->dma_addr+uint64_t(pair)*slab+at;
                    bases.push_back(offset); bodies.push_back(add_regs(width,channels,a,b,c));
                    offset+=align_up(int(bodies.back().size())+4,2);
                    at+=uint64_t(width)*channels*bytes;
                }
            }
            level->count=int(bodies.size());level->mask=mask;
            while (__builtin_popcount(level->mask)>level->count)
                level->mask=level->mask==7?3:1;
            rc=rk_npu_mem_alloc(domain,uint64_t(offset)*8,RK_NPU_MEM_NON_CACHEABLE,&level->regs);
            if (!rc) rc=rk_npu_mem_alloc(domain,uint64_t(level->count)*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING,&level->tasks);
            p->levels.push_back(std::move(level));auto& l=*p->levels.back();
            if (rc) return rc;
            std::memset(l.tasks.vaddr,0,l.tasks.size);
            const int cores=__builtin_popcount(l.mask);int start=0;
            for (int core=0;core<cores;++core) {
                const int n=l.count/cores+int(core<l.count%cores);
                l.starts[core]=start;l.counts[core]=n;
                std::vector<std::vector<uint64_t>> group(bodies.begin()+start,bodies.begin()+start+n);
                std::vector<int> positions(bases.begin()+start,bases.begin()+start+n);
                write_chain(static_cast<uint64_t*>(l.regs.vaddr),static_cast<rknpu_task*>(l.tasks.vaddr)+start,
                    l.regs.dma_addr,group,positions,{0x18,2,0x18});
                start+=n;
            }
            if(l.regs.flags&RK_NPU_MEM_F_CACHEABLE)
                rc=rk_npu_mem_sync(domain->ctx,&l.regs,RK_NPU_SYNC_TO_DEVICE);
            if(!rc && (l.tasks.flags&RK_NPU_MEM_F_CACHEABLE))
                rc=rk_npu_mem_sync(domain->ctx,&l.tasks,RK_NPU_SYNC_TO_DEVICE);
            if (rc) return rc;
            input=output;output=output==&p->scratch?&partials:&p->scratch;
        }
        rc=rk_npu_mem_view(input,0,slab,&p->result);
        if (rc) return rc;
        p_=std::move(p);return RK_NPU_OK;
    } catch (const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
}
int FloatReduce::run() {
    if (!p_) return RK_NPU_ERR_PARAM;
    for (const auto& l:p_->levels) {
        const int rc=do_submit_multicore(p_->domain->ctx->fd,l->tasks.obj_addr,l->count,l->mask,
            l->starts,l->counts,p_->domain->id,p_->timeout);
        if (rc) return rc;
    }
    return RK_NPU_OK;
}
const rk_npu_mem& FloatReduce::output() const { return p_->result; }
}

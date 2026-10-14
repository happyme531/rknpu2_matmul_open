#include "rk_npu_w4a8_chain.h"
#include "rk_npu_i4_regs.h"
#include "rk_npu_trace.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>

namespace rknpu2_matmul_open::detail {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
}
struct W4A8Chain::Impl {
    struct Wave {int k0,k,pk;uint64_t input_offset,weight_offset;};
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_i4_config cfg{};
    I4InputLayout layout=I4InputLayout::Native;
    DomainDataBuffer input,partial;
    rk_npu_mem commands{},tasks{};
    std::vector<I4Tile> tiles;
    std::vector<Wave> waves;
    std::vector<uint64_t> weight_offsets;
    uint32_t starts[3]{},counts[3]{};
    uint64_t partial_stride=0,bound_dma=UINT64_MAX;
    ~Impl() {
        if(!domain)return;
        rk_npu_mem_free(domain->ctx,&tasks);rk_npu_mem_free(domain->ctx,&commands);
        input.release(domain->ctx);partial.release(domain->ctx);release_domain(domain);
    }
    int bind(const I4Weights& weights) {
        const auto& wc=weights.config();const auto& mem=weights.memory();
        if(mem.ctx_id!=domain->ctx->id || mem.iommu_domain_id!=domain->id)return RK_NPU_ERR_DOMAIN;
        if(wc.K!=cfg.K || wc.N!=cfg.N || wc.k_tile!=cfg.k_tile)return RK_NPU_ERR_PARAM;
        if(mem.dma_addr==bound_dma)return RK_NPU_OK;
        auto* words=static_cast<uint64_t*>(commands.vaddr);
        for(size_t i=0;i<weight_offsets.size();++i) {
            const uint64_t address=mem.dma_addr+weight_offsets[i];
            if(address>UINT32_MAX || address%16)return RK_NPU_ERR_PARAM;
            words[i*I4_COMMAND_WORDS+I4_WEIGHT_WORD]=E(T_CNA,R_CNA_DCOMP_ADDR0,uint32_t(address));
        }
        bound_dma=mem.dma_addr;return RK_NPU_OK;
    }
};
W4A8Chain::W4A8Chain():impl_(new Impl){}
W4A8Chain::~W4A8Chain()=default;
int W4A8Chain::query(const rk_npu_i4_config& cfg,I4InputLayout layout,rk_npu_i4_memory_info& info) {
    int rc=query_i4(cfg,info,layout);if(rc)return rc;
    const uint64_t tasks=uint64_t(info.tasks_per_wave)*info.wave_count;
    const uint64_t input=uint64_t(cfg.M)*align_up(cfg.K,32)/2;
    const uint64_t stride=uint64_t(cfg.M)*align_up(cfg.N,64)*2;
    if(tasks>INT_MAX/I4_COMMAND_WORDS || input>UINT32_MAX ||
       stride>UINT32_MAX/info.wave_count)return RK_NPU_ERR_NOMEM;
    info.input_bytes=input;info.partial_bytes=stride*info.wave_count;
    return RK_NPU_OK;
}
int W4A8Chain::prepare(rk_npu_iommu_domain* domain,const rk_npu_i4_config& cfg,I4InputLayout layout) {
    rk_npu_i4_memory_info info{};int rc=query(cfg,layout,info);
    if(rc || !domain || impl_->domain)return rc?rc:RK_NPU_ERR_PARAM;
    auto& p=*impl_;p.domain=domain;retain_domain(domain);p.cfg=cfg;p.layout=layout;
    uint32_t original_starts[3]{},original_counts[3]{};
    p.tiles=i4_tiles(cfg,original_starts,original_counts,layout);
    p.partial_stride=uint64_t(cfg.M)*align_up(cfg.N,64)*2;
    uint64_t input_offset=0,weight_offset=0;
    for(int k0=0;k0<cfg.K;k0+=cfg.k_tile) {
        const int k=std::min(cfg.k_tile,cfg.K-k0),pk=align_up(k,32);
        p.waves.push_back({k0,k,pk,input_offset,weight_offset});
        input_offset+=uint64_t(cfg.M)*pk/2;
        weight_offset+=uint64_t(pk)*align_up(cfg.N,64)/2;
    }
    rc=p.input.alloc(domain,info.input_bytes);if(rc)return rc;
    rc=p.partial.alloc(domain,info.partial_bytes);if(rc)return rc;
    const int total=int(p.tiles.size()*p.waves.size());
    rc=rk_npu_mem_alloc(domain,uint64_t(total)*I4_COMMAND_WORDS*8,0,&p.commands);if(rc)return rc;
    rc=rk_npu_mem_alloc(domain,uint64_t(total)*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING,&p.tasks);if(rc)return rc;
    std::memset(p.tasks.vaddr,0,size_t(p.tasks.size));p.weight_offsets.resize(total);
    int task=0;
    for(int core=0;core<__builtin_popcount(cfg.npu_core_mask);++core) {
        p.starts[core]=task;p.counts[core]=uint32_t(original_counts[core]*p.waves.size());
        std::vector<std::vector<uint64_t>> bodies(p.counts[core]);std::vector<int> bases(p.counts[core]);
        int local=0;
        for(size_t w=0;w<p.waves.size();++w)for(uint32_t j=0;j<original_counts[core];++j) {
            const auto& t=p.tiles[original_starts[core]+j];const auto& v=p.waves[w];
            rc=make_i4_regs(t.m,v.pk,t.n,
                p.input.mem.dma_addr+v.input_offset+uint64_t(t.m0)*v.pk/2,
                0,p.partial.mem.dma_addr+w*p.partial_stride+t.output_offset,bodies[local],layout);
            if(rc)return rc;
            bases[local]=task*I4_COMMAND_WORDS;
            p.weight_offsets[task]=v.weight_offset+uint64_t(t.n0)*v.pk/2;
            ++task;++local;
        }
        write_chain(static_cast<uint64_t*>(p.commands.vaddr),
            static_cast<rknpu_task*>(p.tasks.vaddr)+p.starts[core],p.commands.dma_addr,bodies,bases,{13,0,13});
    }
    return RK_NPU_OK;
}
int W4A8Chain::run(const I4Weights& weights,const W4A8Input& input,const W4A8Reduction& reduction,
                  rk_npu_i4_timings& times) {
    if(!impl_->domain)return RK_NPU_ERR_PARAM;
    auto& p=*impl_;times={};const auto total=Clock::now();int rc=p.bind(weights);if(rc)return rc;
    RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain", "M", input.M, "K", p.cfg.K,
        "N", p.cfg.N, "waves", p.waves.size(), "k_tile", p.cfg.k_tile,
        "n_tile", p.cfg.n_tile, "npu_mask", p.cfg.npu_core_mask);
    auto begin=Clock::now();
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_input_sync_begin");
      rc=p.input.begin_cpu_write(); }
    times.sync_us+=elapsed(begin);if(rc)return rc;
    begin=Clock::now();
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_pack");
    std::memset(p.input.mem.vaddr,0,size_t(uint64_t(p.cfg.M)*align_up(p.cfg.K,32)/2));
    for(const auto& w:p.waves)for(int m0=0;!rc && m0<p.cfg.M;) {
        const int rows=i4_m_rows(p.cfg.M-m0,p.cfg.m_tile,p.layout);
        rk_npu_i4_input_tile tile{m0,rows,w.k0,w.k,w.pk,
            static_cast<uint8_t*>(p.input.mem.vaddr)+w.input_offset+uint64_t(m0)*w.pk/2,uint64_t(rows)*w.pk/2};
        rc=w4a8_pack(const_cast<W4A8Input*>(&input),&tile);m0+=rows;
    }
    }
    times.pack_us+=elapsed(begin);begin=Clock::now();int end;
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_input_sync_end");
      end=p.input.end_cpu_access(); }
    times.sync_us+=elapsed(begin);if(!rc)rc=end;if(rc)return rc;
    begin=Clock::now();
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_submit");
    rc=do_submit_multicore(p.domain->ctx->fd,p.tasks.obj_addr,int(p.weight_offsets.size()),
        p.cfg.npu_core_mask,p.starts,p.counts,p.domain->id,p.cfg.timeout_ms); }
    times.submit_us+=elapsed(begin);if(rc)return rc;
    begin=Clock::now();
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_output_sync_begin");
      rc=p.partial.begin_cpu_read(); }
    times.sync_us+=elapsed(begin);if(rc)return rc;
    begin=Clock::now();const auto* src=static_cast<const int16_t*>(p.partial.mem.vaddr);
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_reduce", "cpu_threads", p.cfg.cpu_threads);
    try {
        for(int w=0;!rc && w<int(p.waves.size());++w)
            rc=w4a8_reduce_tiles(reduction,p.cfg,p.tiles,w,int(p.waves.size()),src+uint64_t(w)*p.partial_stride/2);
    } catch(...) {rc=RK_NPU_ERR_PARAM;}
    }
    times.reduce_us+=elapsed(begin);begin=Clock::now();int finish;
    { RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.chain_output_sync_end");
      finish=p.partial.end_cpu_access(); }
    times.sync_us+=elapsed(begin);if(!rc)rc=finish;times.total_us=elapsed(total);return rc;
}
}

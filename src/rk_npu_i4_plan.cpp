#include "rk_npu_i4_plan.h"
#include "rk_npu_i4_cpu.h"
#include "rk_npu_i4_regs.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>

namespace rknpu2_matmul_open::detail {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::micro>(Clock::now()-t).count(); }
bool weight_valid(const rk_npu_i4_weight_config& c) {
    return c.K > 0 && c.K <= INT32_MAX/64 && c.N > 0 && c.N <= INT_MAX-63 &&
           c.k_tile >= 32 && c.k_tile <= I4_K_TILE_MAX && c.k_tile%32==0;
}
bool valid(const rk_npu_i4_config& c, I4InputLayout layout) {
    const int cores = __builtin_popcount(c.npu_core_mask);
    if (!weight_valid({c.K,c.N,c.k_tile}) || c.M<1 || c.m_tile<1) return false;
    if (layout != I4InputLayout::Native && layout != I4InputLayout::Panel8) return false;
    if (layout == I4InputLayout::Panel8 && (c.m_tile % 8 || c.m_tile > 512 ||
        c.m_tile < 8 || c.M % 2 || c.k_tile < 32 || c.k_tile > I4_K_TILE_MAX ||
        i4_panel_banks(i4_m_rows(c.M,c.m_tile,layout),align_up(std::min(c.K,c.k_tile),32)) > 8))
        return false;
    return weight_valid({c.K,c.N,c.k_tile}) && c.M > 0 && c.m_tile > 0 && c.m_tile <= (layout == I4InputLayout::Native ? 128 : 512) &&
           c.n_tile >= 64 && c.n_tile <= 4096 && c.n_tile%64==0 &&
           (c.npu_core_mask==1 || c.npu_core_mask==2 || c.npu_core_mask==4 ||
            c.npu_core_mask==3 || c.npu_core_mask==7) && align_up(c.N,64)/64 >= cores &&
           c.timeout_ms > 0 && (c.pipeline==0 || c.pipeline==1) && c.cpu_threads>=1 && c.cpu_threads<=4;
}

} // namespace
std::vector<I4Tile> i4_tiles(const rk_npu_i4_config& c, uint32_t* starts, uint32_t* counts,
                             I4InputLayout layout) {
    std::vector<I4Tile> tiles;
    const int cores=__builtin_popcount(c.npu_core_mask), blocks=align_up(c.N,64)/64;
    int n0=0;
    uint64_t out=0;
    for (int core=0;core<cores;++core) {
        starts[core]=uint32_t(tiles.size());
        const int end=n0+(blocks/cores+(core<blocks%cores))*64;
        for (;n0<end;n0+=std::min(c.n_tile,end-n0)) {
            const int n=std::min(c.n_tile,end-n0);
            for (int m0=0;m0<c.M;) {
                const int m=i4_m_rows(c.M-m0,c.m_tile,layout);
                tiles.push_back({m0,m,n0,n,std::max(0,std::min(n,c.N-n0)),out});
                out+=uint64_t(m)*n*2;
                m0+=m;
            }
        }
        counts[core]=uint32_t(tiles.size())-starts[core];
    }
    return tiles;
}

int query_i4_weights(const rk_npu_i4_weight_config& cfg, uint64_t& bytes) {
    if (!weight_valid(cfg)) return RK_NPU_ERR_PARAM;
    bytes=uint64_t(align_up(cfg.K,32))*align_up(cfg.N,64)/2;
    // A single immutable B arena must fit the domain's 32-bit IOVA space.
    if (bytes > UINT32_MAX) return RK_NPU_ERR_NOMEM;
    return RK_NPU_OK;
}

int query_i4(const rk_npu_i4_config& cfg, rk_npu_i4_memory_info& out, I4InputLayout layout) {
    if (!valid(cfg,layout)) return RK_NPU_ERR_PARAM;
    out={};
    int rc=query_i4_weights({cfg.K,cfg.N,cfg.k_tile},out.packed_weight_bytes);
    if (rc) return rc;
    out.wave_count=uint32_t(ceil_div(cfg.K,cfg.k_tile));
    const uint64_t slots=std::min(out.wave_count,2u);
    out.input_bytes=slots*uint64_t(cfg.M)*align_up(std::min(cfg.K,cfg.k_tile),32)/2;
    out.partial_bytes=slots*uint64_t(cfg.M)*align_up(cfg.N,64)*2;
    const int cores=__builtin_popcount(cfg.npu_core_mask), blocks=align_up(cfg.N,64)/64;
    uint64_t nt=0;
    for (int i=0;i<cores;++i) nt+=ceil_div((blocks/cores+(i<blocks%cores))*64,cfg.n_tile);
    const uint64_t remainder = cfg.M % cfg.m_tile;
    const uint64_t panels = layout == I4InputLayout::Native
        ? (uint64_t(cfg.M)+cfg.m_tile-1)/cfg.m_tile
        : uint64_t(cfg.M)/cfg.m_tile + (remainder>=8) + (remainder%8!=0);
    const uint64_t tasks=nt*panels;
    if (tasks > INT_MAX/I4_COMMAND_WORDS || out.input_bytes/slots > UINT32_MAX ||
        out.partial_bytes/slots > UINT32_MAX) return RK_NPU_ERR_NOMEM;
    out.tasks_per_wave=uint32_t(tasks);
    out.control_bytes=uint64_t(out.wave_count)*tasks*(I4_COMMAND_WORDS*8+sizeof(rknpu_task));
    return RK_NPU_OK;
}

I4Weights::~I4Weights() { if (arena_.domain) arena_.release(arena_.domain->ctx); }
int I4Weights::prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_weight_config& cfg, const int8_t* B,
                      bool allow_unproven) {
    uint64_t bytes=0;
    int rc=query_i4_weights(cfg,bytes);
    if (rc || !B || !domain || arena_.domain) return rc ? rc : RK_NPU_ERR_PARAM;
    if (cfg.k_tile>I4_K_TILE_UNIVERSAL && !allow_unproven) {
        I4WeightBound bound;
        rc=bound_i4_weights(cfg.K,cfg.N,B,cfg.k_tile,bound);
        if (rc) return rc;
    }
    cfg_=cfg;
    rc=arena_.alloc(domain,bytes);
    if (rc) return rc;
    rc=arena_.begin_cpu_write();
    uint64_t offset=0;
    for (int k0=0;rc==RK_NPU_OK && k0<cfg.K;) {
        const int k=std::min(cfg.k_tile,cfg.K-k0), pk=align_up(k,32), pn=align_up(cfg.N,64);
        rc=pack_i4_weights(cfg.K,cfg.N,k0,k,pk,pn,B,static_cast<uint8_t*>(arena_.mem.vaddr)+offset);
        offset+=uint64_t(pk)*pn/2;
        k0+=k;
    }
    const int end=arena_.end_cpu_access();
    return rc ? rc : end;
}

I4Plan::~I4Plan() {
    if (!domain_) return;
    for (auto& wave:waves_) {
        rk_npu_mem_free(domain_->ctx,&wave.tasks);
        rk_npu_mem_free(domain_->ctx,&wave.regcmd);
    }
    for (int i=0;i<2;++i) { input_[i].release(domain_->ctx); output_[i].release(domain_->ctx); }
    retained_output_.release(domain_->ctx);
    release_domain(domain_);
}

int I4Plan::prepare(rk_npu_iommu_domain* domain, const rk_npu_i4_config& cfg, bool retain_partials, I4InputLayout layout) {
    rk_npu_i4_memory_info info{};
    int rc=query_i4(cfg,info,layout);
    if (rc || !domain || !domain->ctx || domain_) return rc ? rc : RK_NPU_ERR_PARAM;
    domain_=domain; retain_domain(domain_); cfg_=cfg;
    retained_=retain_partials; layout_=layout;
    slots_=std::min(info.wave_count,2u);
    for (int i=0;i<slots_;++i) {
        rc=input_[i].alloc(domain_,info.input_bytes/slots_); if (rc) return rc;
        if (!retained_) { rc=output_[i].alloc(domain_,info.partial_bytes/slots_); if (rc) return rc; }
    }
    tiles_=i4_tiles(cfg,starts_,counts_,layout_);
    waves_.resize(info.wave_count);
    if (retained_) {
        uint64_t guard=256*1024;
        for (const auto& t:tiles_)
            guard=std::max(guard,uint64_t(t.m)*t.n*2*info.wave_count);
        // Spatial INT16 Conv may fetch the zero-weight padded C16 plane.
        // Reserve a mapped guard after the final tile, with no extra CPU copy per run.
        const uint64_t bytes=info.partial_bytes/slots_*info.wave_count+guard;
        if (bytes>UINT32_MAX) return RK_NPU_ERR_NOMEM;
        rc=retained_output_.alloc(domain_,bytes); if (rc) return rc;
        rc=retained_output_.begin_cpu_write(); if (rc) return rc;
        std::memset(retained_output_.mem.vaddr,0,size_t(bytes));
        rc=retained_output_.end_cpu_access(); if (rc) return rc;
    }
    uint64_t boff=0;
    for (int w=0;w<wave_count();++w) {
        auto& wave=waves_[w];
        wave.k0=w*cfg.k_tile; wave.k=std::min(cfg.k_tile,cfg.K-wave.k0); wave.padded_k=align_up(wave.k,32);
        wave.weight_offset=boff; boff+=uint64_t(wave.padded_k)*align_up(cfg.N,64)/2;
        const int slot=w%slots_;
        rc=rk_npu_mem_alloc(domain_,uint64_t(tiles_.size())*I4_COMMAND_WORDS*8,0,&wave.regcmd);
        if (rc) return rc;
        rc=rk_npu_mem_alloc(domain_,uint64_t(tiles_.size())*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING,&wave.tasks);
        if (rc) return rc;
        std::memset(wave.tasks.vaddr,0,size_t(wave.tasks.size));
        // Each selected core gets a separate PC chain; all M panels of its N
        // shard are submitted together. A and B panels are reused by address.
        for (int core=0;core<__builtin_popcount(cfg.npu_core_mask);++core) {
            std::vector<std::vector<uint64_t>> bodies(counts_[core]);
            std::vector<int> bases(counts_[core]);
            for (uint32_t j=0;j<counts_[core];++j) {
                const int ti=int(starts_[core]+j);
                const auto& tile=tiles_[ti];
                const uint64_t output_dma=retained_
                    ? retained_output_.mem.dma_addr+tile.output_offset*info.wave_count+uint64_t(w)*tile.m*tile.n*2
                    : output_[slot].mem.dma_addr+tile.output_offset;
                rc=make_i4_regs(tile.m,wave.padded_k,tile.n,
                    input_[slot].mem.dma_addr+uint64_t(tile.m0)*wave.padded_k/2,
                    0,output_dma,bodies[j],layout_);
                if (rc) return rc;
                bases[j]=ti*I4_COMMAND_WORDS;
            }
            write_chain(static_cast<uint64_t*>(wave.regcmd.vaddr),
                static_cast<rknpu_task*>(wave.tasks.vaddr)+starts_[core],wave.regcmd.dma_addr,
                bodies,bases,{13,0,13});
        }
    }
    return RK_NPU_OK;
}

int I4Plan::bind(const I4Weights& weights) {
    const auto& wc=weights.config(); const auto& mem=weights.memory();
    if (!domain_ || mem.ctx_id!=domain_->ctx->id || mem.iommu_domain_id!=domain_->id) return RK_NPU_ERR_DOMAIN;
    if (wc.K!=cfg_.K || wc.N!=cfg_.N || wc.k_tile!=cfg_.k_tile) return RK_NPU_ERR_PARAM;
    if (bound_weight_dma_==mem.dma_addr) return RK_NPU_OK;
    for (auto& wave:waves_) {
        auto* cmd=static_cast<uint64_t*>(wave.regcmd.vaddr);
        for (size_t ti=0;ti<tiles_.size();++ti) {
            const uint64_t dma=mem.dma_addr+wave.weight_offset+uint64_t(tiles_[ti].n0)*wave.padded_k/2;
            if (dma>UINT32_MAX || (dma&15)) return RK_NPU_ERR_PARAM;
            cmd[ti*I4_COMMAND_WORDS+I4_WEIGHT_WORD]=E(T_CNA,R_CNA_DCOMP_ADDR0,uint32_t(dma));
        }
    }
    bound_weight_dma_=mem.dma_addr;
    return RK_NPU_OK;
}

int I4Plan::pack(int w, rk_npu_i4_input_producer producer, void* user, rk_npu_i4_timings& times) {
    if (w<0 || w>=wave_count() || !producer) return RK_NPU_ERR_PARAM;
    const auto& wave=waves_[w]; auto& buffer=input_[w%slots_];
    auto start=Clock::now(); int rc=buffer.begin_cpu_write(); times.sync_us+=elapsed(start);
    if (rc) return rc;
    start=Clock::now();
    for (int m0=0;m0<cfg_.M;) {
        const int rows=i4_m_rows(cfg_.M-m0,cfg_.m_tile,layout_);
        rk_npu_i4_input_tile tile{m0,rows,wave.k0,wave.k,wave.padded_k,
            static_cast<uint8_t*>(buffer.mem.vaddr)+uint64_t(m0)*wave.padded_k/2,uint64_t(rows)*wave.padded_k/2};
        std::memset(tile.dst,0,size_t(tile.bytes));
        try { rc=producer(user,&tile); } catch (...) { rc=RK_NPU_ERR_PARAM; }
        if (rc) break;
        m0+=rows;
    }
    times.pack_us+=elapsed(start);
    start=Clock::now(); const int end=buffer.end_cpu_access(); times.sync_us+=elapsed(start);
    return rc ? rc : end;
}

int I4Plan::submit(int wave) {
    if (wave<0 || wave>=wave_count() || bound_weight_dma_==UINT64_MAX) return RK_NPU_ERR_PARAM;
    return do_submit_multicore(domain_->ctx->fd,waves_[wave].tasks.obj_addr,int(tiles_.size()),
                               cfg_.npu_core_mask,starts_,counts_,domain_->id,cfg_.timeout_ms);
}
int I4Plan::begin_partial(int w, const int16_t** partial) {
    if (retained_ || w<0 || w>=wave_count() || !partial) return RK_NPU_ERR_PARAM;
    auto& buffer=output_[w%slots_]; const int rc=buffer.begin_cpu_read();
    if (!rc) *partial=static_cast<const int16_t*>(buffer.mem.vaddr);
    return rc;
}
int I4Plan::end_partial(int w) {
    if (retained_ || w<0 || w>=wave_count()) return RK_NPU_ERR_PARAM;
    return output_[w%slots_].end_cpu_access();
}
namespace {
int consume_i32(void* user, const I4Plan& plan, int w, const int16_t* src) {
    auto* C=static_cast<int32_t*>(user);
    const auto& cfg=plan.config(); const auto& tiles=plan.tiles();
    const int threads=cfg.cpu_threads;
    const bool parallel=threads>1 && uint64_t(cfg.M)*cfg.N>=65536;
#pragma omp parallel for num_threads(threads) schedule(static) if(parallel)
    for (int i=0;i<int(tiles.size());++i) {
        const auto& tile=tiles[i];
        reduce_i4_partial(tile.m,tile.logical_n,tile.n,
            src+tile.output_offset/2,C+size_t(tile.m0)*cfg.N+tile.n0,cfg.N,w!=0);
    }
    return RK_NPU_OK;
}
}
int I4Plan::reduce(int w, int32_t* C, rk_npu_i4_timings& times) {
    return C ? consume(w,consume_i32,C,times) : RK_NPU_ERR_PARAM;
}
int I4Plan::consume(int w, I4PartialConsumer consumer, void* user, rk_npu_i4_timings& times) {
    if (!consumer) return RK_NPU_ERR_PARAM;
    const int16_t* src=nullptr;
    auto start=Clock::now(); int rc=begin_partial(w,&src); times.sync_us+=elapsed(start);
    if (rc) return rc;
    start=Clock::now();
    try { rc=consumer(user,*this,w,src); } catch (...) { rc=RK_NPU_ERR_PARAM; }
    times.reduce_us+=elapsed(start);
    start=Clock::now(); const int end=end_partial(w); times.sync_us+=elapsed(start);
    return rc ? rc : end;
}
}

#pragma once
#include "rk_npu_attention_f16.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace rknpu2_matmul_open::attention {
using Clock=std::chrono::steady_clock;
using Task=detail::RegisterTask;
// DPU 0x403c / RDMA 0x5014 have a hard 8192-channel limit. Use a smaller
// QK tile: an otherwise-correct native FP16 N=8192 task can leave a following
// INT8 task timing out; N=4096 passes the mixed-precision board regression.
// This is a measured safe recipe, not a claim about the undocumented cause.
// QK output channels are KV positions, independently of the PV K tile.
constexpr int QK_CHANNEL_TILE=4096;
struct Error {int code;};
inline void check(int rc){if(rc)throw Error{rc};}
inline double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
inline uint16_t half(float x){return bits::float_to_half(x);}
inline float widen(uint16_t x){return bits::half_to_float(x);}
int validate(const rk_npu_attention_f16_config& cfg);
// Private engine geometry. The public GQA contract keeps its original shape;
// MLA uses independent QK/V dimensions and optionally one shared latent KV.
struct Shape {
    int key_dim=128,value_dim=128,group=4;
    bool shared_kv=false;
    float scale=0.08838834764831844f;
    bool balanced_heads=false;
    int row_alignment=4;
    int heads(const rk_npu_attention_f16_config& c,int index)const{
        return balanced_heads?c.query_heads/c.kv_heads+(index<c.query_heads%c.kv_heads):group;
    }
    int first_head(const rk_npu_attention_f16_config& c,int index)const{
        return balanced_heads?index*(c.query_heads/c.kv_heads)+std::min(index,c.query_heads%c.kv_heads):index*group;
    }
};
int query_engine(const rk_npu_attention_f16_config& cfg,const Shape& shape,
    int capacity,int rows,int length,rk_npu_attention_f16_sizes& out);
void validate_boolean_mask(const rk_npu_attention_f16_config& cfg,
    const rk_npu_attention_f16_boolean_mask* mask,int rows,int length);
inline uint32_t core_mask(const rk_npu_attention_f16_config& c){return c.core_mask?c.core_mask:1;}
inline uint64_t next_identity(){static std::atomic<uint64_t> serial{1};return serial.fetch_add(1);}
struct Lease {
    rk_npu_iommu_domain* domain;
    explicit Lease(rk_npu_iommu_domain* d):domain(d){if(!d || !d->ctx)throw Error{RK_NPU_ERR_PARAM};detail::retain_domain(d);}
    ~Lease(){detail::release_domain(domain);}
};
struct Lock {
    std::atomic<bool>& busy;
    explicit Lock(std::atomic<bool>& b):busy(b){bool no=false;if(!b.compare_exchange_strong(no,true))throw Error{RK_NPU_ERR_BUSY};}
    ~Lock(){busy.store(false,std::memory_order_release);}
};
struct Buffer {
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mem mem{};
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(mem.handle)rk_npu_mem_free(domain->ctx,&mem);}
    void swap(Buffer& other){std::swap(domain,other.domain);std::swap(mem,other.mem);}
    void alloc(rk_npu_iommu_domain* d,uint64_t bytes,uint32_t flags=RK_NPU_MEM_DATA_DEFAULT){
        if(mem.handle || !bytes)throw Error{RK_NPU_ERR_PARAM};domain=d;
        check(rk_npu_mem_alloc(d,bytes,flags,&mem));
    }
    void ensure(rk_npu_iommu_domain* d,uint64_t bytes,uint32_t flags=RK_NPU_MEM_DATA_DEFAULT){
        if(mem.size>=bytes)return;
        Buffer next;next.alloc(d,bytes,flags);swap(next);
    }
    uint16_t* data(){return static_cast<uint16_t*>(mem.vaddr);}
    const uint16_t* data()const{return static_cast<const uint16_t*>(mem.vaddr);}
    void sync(rk_npu_sync_dir dir){check(rk_npu_mem_sync(domain->ctx,&mem,dir));}
    void range(uint64_t offset,uint64_t bytes,rk_npu_sync_dir dir=RK_NPU_SYNC_TO_DEVICE){
        if(!bytes)return;rk_npu_mem view{};check(rk_npu_mem_view(&mem,offset,bytes,&view));
        check(rk_npu_mem_sync(domain->ctx,&view,dir));
    }
    void zero(){std::memset(mem.vaddr,0,mem.size);}
};
// Native B has N16/K32 stripes: 16 channels x 32 tokens x FP16 = 1024
// bytes. Merge adjacent channel groups to trade cache-clean bytes for ioctls.
// The denominator occupies only the first 64-byte line of its N16 stripe.
template<class F>void dirty_value_ranges(uint64_t base,int channels,int width,
    int first,int end,int groups,F&& range,bool merge_denominator=false){
    const int begin=first/32,last=ceil_div(end,32),count=channels/16;
    if(merge_denominator && groups>=count){
        range(base+uint64_t(begin)*1024,
              uint64_t(count)*width*32+uint64_t(last-1-begin)*1024+64);
        return;
    }
    for(int group=0;group<count;group+=groups){
        const int merged=std::min(groups,count-group);
        range(base+(uint64_t(group)*width*16+begin*512)*2,
              (uint64_t(merged-1)*width*16+(last-begin)*512)*2);
    }
    for(int stripe=begin;stripe<last;++stripe)
        range(base+(uint64_t(count)*width*16+stripe*512)*2,64);
}
struct DirtySyncPolicy {
    int groups=0,after=-1;
    int for_length(int length)const{return length>after?groups:0;}
};
DirtySyncPolicy experimental_dirty_sync_policy()noexcept;
struct Cache {
    Lease lease;
    rk_npu_attention_f16_config cfg;
    Shape shape;
    // Read once at construction; zero preserves the verified full-range path.
    DirtySyncPolicy dirty_sync=experimental_dirty_sync_policy();
    uint64_t identity=next_identity();
    std::atomic<bool> busy{false};
    Buffer key,value,tail;
    int length=0,capacity=0,blocks=0;
    int tail_block=-1,tail_length=0;
    bool poisoned=false;
    uint64_t generation=0,epoch=0,tail_epoch=~uint64_t(0),tail_generation=0;
    int dirty_first=0,dirty_count=0;
    // Magnitude bits are ordered for finite FP16. Track V on updates so a
    // stable run can bound each FP16 PV partial without rescanning the cache.
    uint16_t value_absmax[32]{};
    explicit Cache(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c,Shape s={}):lease(d),cfg(c),shape(s){reserve(c.initial_capacity);}
    int h()const{return shape.shared_kv?1:cfg.kv_heads;}
    int n()const{return shape.value_dim+32;}
    size_t voffset(int hi,int block)const{return size_t(hi*blocks+block)*n()*cfg.kv_tile;}
    void reserve(int requested);
    void load(const uint16_t* k,const uint16_t* v,int count);
    void append(const uint16_t* k,const uint16_t* v,int count,double* reserve_time=nullptr);
    void stage_tail(int span);
    void write(int position,const uint16_t* k,const uint16_t* v,int source_stride,int source_position);
};
struct Binding {size_t word;int kind,head,block;};
struct Graph {
    rk_npu_iommu_domain* domain;
    rk_npu_attention_f16_config cfg;
    Shape shape;
    int rows,m,span,blocks,full,tail_length,tasks=0;
    uint32_t task_start[3]{},task_count[3]{}; // Logical selected-core order.
    uint32_t stage_start[2][3]{},stage_count[2][3]{};
    Buffer q,score,mask,partial,regs,descriptors;
    std::vector<float> sum;
    std::vector<float> softmax_scratch;
    std::vector<Binding> bindings;
    Cache* bound=nullptr;
    uint64_t bound_generation=~uint64_t(0),bound_tail_generation=~uint64_t(0);
    uint64_t bound_identity=0;
    int mask_start=-1;
    rk_npu_attention_f16_boolean_mask saved_mask{};
    explicit Graph(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c,int qr,int s,Cache& cache,Shape shape,Graph* reuse=nullptr);
    void bind(Cache& cache);
    void update_mask(int start,const rk_npu_attention_f16_boolean_mask* boolean,int length);
    void pack_q(const uint16_t* query);
    double submit(int stage=0);
    void stable_softmax(int start,const rk_npu_attention_f16_boolean_mask* mask,int length);
    void finish(float* output);
};
struct Workspace {
    Lease lease;
    rk_npu_attention_f16_config cfg;
    Shape shape;
    std::atomic<bool> busy{false};
    std::unique_ptr<Graph> graph;
    bool poisoned=false;
    bool full_causal_prefill=false; // MLA query chunks reuse one valid-KV graph.
    Workspace(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c,Shape s={}):lease(d),cfg(c),shape(s){}
    void geometry(Cache& cache,int rows,int start,rk_npu_attention_f16_timings& t);
    rk_npu_attention_f16_timings run(Cache& cache,const uint16_t* q,int rows,int start,float* output,
        const rk_npu_attention_f16_boolean_mask* mask);
};
inline void context(rk_npu_ctx* ctx,rk_npu_iommu_domain* d){if(!ctx || ctx!=d->ctx)throw Error{RK_NPU_ERR_PARAM};}
inline void compatible(const Workspace& w,const Cache& c){
    if(w.lease.domain->ctx!=c.lease.domain->ctx)throw Error{RK_NPU_ERR_PARAM};
    if(w.lease.domain->id!=c.lease.domain->id)throw Error{RK_NPU_ERR_DOMAIN};
    const bool shared=w.shape.shared_kv && c.shape.shared_kv;
    if(w.cfg.query_heads!=c.cfg.query_heads || (!shared && w.cfg.kv_heads!=c.cfg.kv_heads) ||
       w.cfg.head_dim!=c.cfg.head_dim || w.cfg.kv_tile!=c.cfg.kv_tile ||
       c.cfg.max_capacity>w.cfg.max_capacity ||
       w.shape.key_dim!=c.shape.key_dim || w.shape.value_dim!=c.shape.value_dim ||
       (!shared && w.shape.group!=c.shape.group) || w.shape.shared_kv!=c.shape.shared_kv ||
       w.shape.scale!=c.shape.scale)throw Error{RK_NPU_ERR_PARAM};
}
} // namespace
struct rk_npu_attention_f16_cache : rknpu2_matmul_open::attention::Cache {
    using Cache::Cache;
};
struct rk_npu_attention_f16_workspace : rknpu2_matmul_open::attention::Workspace {
    using Workspace::Workspace;
};

#include "rk_npu_trace.h"
#include "rk_npu_attention_f16_internal.h"
#include "rk_npu_matmul_f16.h"
#include "rk_npu_attention_f16_lut.h"
#include <cmath>
#include <sys/ioctl.h>
#include <omp.h>
#include <type_traits>
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#define ATTENTION_NEON 1
#endif

namespace rknpu2_matmul_open::attention {
namespace {
uint32_t value(uint64_t w){return uint32_t(w>>16);}
bool reg(uint64_t w,uint16_t t,uint16_t a){return uint16_t(w>>48)==t && uint16_t(w)==a;}
size_t index(const Task& t,uint16_t target,uint16_t address){
    for(size_t i=0;i<t.body.size();++i)if(reg(t.body[i],target,address))return i;
    throw Error{RK_NPU_ERR_PARAM};
}
void patch(Task& t,uint16_t target,uint16_t address,uint32_t v){
    for(auto& w:t.body)if(reg(w,target,address)){w=E(target,address,v);return;}
    t.body.push_back(E(target,address,v));
}
struct Matmul {
    std::vector<Task> build(rk_npu_iommu_domain* domain,const rk_npu_attention_f16_config& c,
        int batch,int m,int n,int k,Buffer& a,Buffer& b,Buffer& out,int n_tile=0){
        rk_npu_matmul_f16_config cfg{};rk_npu_matmul_f16_config_init(&cfg,m,n,k,RK_NPU_FUSE_NONE);
        cfg.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;cfg.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
        cfg.n_tile=n_tile;
        // Export independent register bodies; partition the mixed QK/PV graph
        // by KV head below, rather than splitting GEMM tasks across cores.
        cfg.core_mask=core_mask(c)&-core_mask(c);cfg.timeout_ms=c.timeout_ms;
        std::vector<Task> result;check(detail::emit_f16_batch_tasks(domain,batch,&cfg,&a.mem,&b.mem,&out.mem,result));return result;
    }
};

// Built-in DPU-RDMA programming rule, verified against a13 native Softmax
// captures for M4/8/16/32/128 and S128/512/4096/4128/4160/4608.
// The register sequence is generated from dimensions; no external tasks/model.
void mask_rdma(Task& t,int m,int span,uint32_t base){
    const std::pair<uint16_t,uint32_t> program[]={
        {0x5004,0xe},{0x500c,uint32_t(m-1)},{0x5010,0},{0x5014,uint32_t(span-1)},
        {0x5018,base},{0x501c,0},{0x5020,0},{0x5028,0},{0x502c,0},
        {0x5034,0x40000008},{0x5038,base+uint32_t(m*16)},
        {0x5040,uint32_t(m*16)},{0x5044,0x17d40},{0x5048,0},{0x504c,uint32_t(m*16)},
        {0x5064,0},{0x5068,0x01010101},{0x506c,uint32_t(m*16)}};
    for(auto pair:program)
        patch(t,T_RDMA,pair.first,pair.second);
}
int prefix_length(const rk_npu_attention_f16_config& c,const Cache& cache,int rows,int start){
    if(cache.poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    if(rows<1 || rows>c.max_query_rows || start<0 || cache.length<1 ||
       (c.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL && start>cache.length-rows))throw Error{RK_NPU_ERR_PARAM};
    return c.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL?start+rows:cache.length;
}
void validate_mask(const rk_npu_attention_f16_config& c,const rk_npu_attention_f16_boolean_mask* b,int rows,int length){
    if(c.mask_mode!=RK_NPU_ATTENTION_F16_BOOLEAN_MASK){if(b)throw Error{RK_NPU_ERR_PARAM};return;}
    if(!b || !b->data || b->heads<1 || (b->heads!=1 && b->heads!=c.query_heads) ||
       b->query_rows<1 || b->key_length<1 || b->query_offset<0 ||
       (b->query_rows!=1 && (b->query_offset>b->query_rows-rows)) ||
       (b->query_rows==1 && b->query_offset) || (b->key_length!=1 && b->key_length<length))
        throw Error{RK_NPU_ERR_PARAM};
    uint64_t end=0;
    const std::pair<uint64_t,uint64_t> bounds[]={
        {uint64_t(b->heads-1),b->head_stride},{uint64_t(b->query_rows-1),b->query_stride},
        {uint64_t(b->key_length-1),b->key_stride}};
    for(auto pair:bounds){
        if(pair.first && pair.second>(UINT64_MAX-end)/pair.first)throw Error{RK_NPU_ERR_PARAM};
        end+=pair.first*pair.second;
    }
    if(end>=b->bytes)throw Error{RK_NPU_ERR_PARAM};
}
bool same_mask(const rk_npu_attention_f16_boolean_mask& a,const rk_npu_attention_f16_boolean_mask& b){
    return a.version && a.data==b.data && a.version==b.version && a.bytes==b.bytes &&
        a.heads==b.heads && a.query_rows==b.query_rows && a.key_length==b.key_length &&
        a.query_offset==b.query_offset && a.head_stride==b.head_stride &&
        a.query_stride==b.query_stride && a.key_stride==b.key_stride;
}
} // namespace

void validate_boolean_mask(const rk_npu_attention_f16_config& c,
    const rk_npu_attention_f16_boolean_mask* mask,int rows,int length){
    validate_mask(c,mask,rows,length);
}

Graph::Graph(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c,int qr,int s,Cache& cache,Shape sh,Graph* reuse,const uint16_t* bounds,Tail* scratch)
    :domain(d),cfg(c),shape(sh),rows(qr),m(align_up(shape.group*qr,shape.row_alignment)),span(s),blocks(ceil_div(s,c.kv_tile)),full(s/c.kv_tile),tail_length(s%c.kv_tile){
    const char* mode=std::getenv("RK_NPU_ATTN_LUT_SUBMIT");
    lut_split=mode && !std::strcmp(mode,"split");
    if(!mode || lut_split || !std::strcmp(mode,"pingpong"))lut_flags|=RKNPU_JOB_PINGPONG;
    rk_npu_attention_f16_sizes sizes{};check(query_engine(c,shape,cache.capacity,qr,s,sizes));
    matmul_tasks=sizes.tasks;lut=can_lut(cache,bounds);
    if(reuse){q.swap(reuse->q);score.swap(reuse->score);mask.swap(reuse->mask);partial.swap(reuse->partial);
        regs.swap(reuse->regs);descriptors.swap(reuse->descriptors);sum.swap(reuse->sum);
        softmax_scratch.swap(reuse->softmax_scratch);exponent.swap(reuse->exponent);}
    auto ensure=[&](Buffer& buffer,uint64_t bytes,uint32_t flags=RK_NPU_MEM_DATA_DEFAULT){
        buffer.ensure(d,std::max(bytes,buffer.mem.size*2),flags);
    };
    // All physical QK/PV rows and aligned output channels are overwritten by
    // each submit. Keep untouched output pages device-owned, rather than
    // touching/cleaning them from the CPU before every geometry change.
    if(q.mem.size<sizes.query_bytes)ensure(q,sizes.query_bytes);
    if(score.mem.size<sizes.score_bytes)ensure(score,sizes.score_bytes);
    if(lut && exponent.mem.size<sizes.score_bytes)ensure(exponent,sizes.score_bytes);
    if(lut)active.resize(size_t(c.kv_heads)*m);
    if(partial.mem.size<sizes.partial_bytes)ensure(partial,sizes.partial_bytes);
    if(sizes.mask_bytes && mask.mem.size<sizes.mask_bytes)ensure(mask,sizes.mask_bytes);
    sum.resize(size_t(m)*(shape.value_dim+8));
    // CPU exp and causal LUT prefix sums share scratch. Retain its size across
    // graph changes so switching paths does not reinitialize unused elements.
    size_t scratch_size=0;
    if(lut && c.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL)
        scratch_size=size_t(rows)*shape.value_dim;
    else if(!lut && c.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX && span>1024)
        scratch_size=size_t(4)*4*span;
    if(softmax_scratch.size()<scratch_size)softmax_scratch.resize(scratch_size);
    Matmul qk;auto qt=qk.build(d,c,1,m,span,shape.key_dim,q,cache.key,score,std::min(span,QK_CHANNEL_TILE));
    if(qt.size()!=size_t(ceil_div(span,QK_CHANNEL_TILE)))throw Error{RK_NPU_ERR_PARAM};
    const float bias=15300.6240234375f-1477.f*c.exp_shift;uint32_t bias_bits;std::memcpy(&bias_bits,&bias,4);
    std::vector<Task> all;
    std::vector<std::tuple<size_t,int,int,int>> relocs;
    for(int hi=0;hi<c.kv_heads;++hi)for(const auto& body:qt){
        auto t=body;
        const size_t offset=value(t.body[index(t,T_DPU,0x4020)])-score.mem.dma_addr;
        const int first=offset/(m*2),channels=std::min(QK_CHANNEL_TILE,span-first);
        patch(t,T_CNA,0x1070,q.mem.dma_addr+size_t(hi)*m*shape.key_dim*2);
        patch(t,T_DPU,0x4020,score.mem.dma_addr+size_t(hi)*m*span*2+offset);
        if(c.flags==RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL){
            patch(t,T_DPU,0x4010,0x28000002);patch(t,T_DPU,0x4040,0x42);
            patch(t,T_DPU,0x4048,0x65c50000);patch(t,T_DPU,0x4060,0x20010);patch(t,T_DPU,0x4064,bias_bits);
            // Preserve MAC flying regroup BS_OW_CFG=0x126.
            if(c.mask_mode==RK_NPU_ATTENTION_F16_NO_MASK)patch(t,T_DPU,0x4070,0x383);
            else {patch(t,T_DPU,0x4070,0x108003c4);mask_rdma(t,m,channels,mask.mem.dma_addr+(size_t(hi)*span+first)*m*2);t.op_idx=1;t.enable_mask=0x1d;}
        }
        relocs.emplace_back(all.size(),0,hi,first);all.push_back(std::move(t));
    }
    auto pv=[&](int per_head,int length,bool is_tail){
        Matmul mm;Buffer& weights=is_tail?static_cast<Buffer&>(scratch?*scratch:cache.tail):cache.value;
        Buffer& probabilities=lut?exponent:score;
        auto pt=mm.build(d,c,per_head,m,shape.value_dim+1,length,probabilities,weights,partial);
        const size_t astride=size_t(m)*length*2,ostride=size_t(shape.value_dim+32)*m*2;
        for(int hi=0;hi<c.kv_heads;++hi)for(const auto& body:pt){
            Task t=body;
            // LUT configuration is shared across DPU register groups. A
            // prefetched PV must not clear it while exponent tasks run.
            // PV's EW bypass already disables LUT evaluation for its output.
            if(lut)t.body.erase(std::remove_if(t.body.begin(),t.body.end(),[](uint64_t w){
                return uint16_t(w>>48)==T_DPU && uint16_t(w)>=0x4100 && uint16_t(w)<=0x412c;
            }),t.body.end());
            const size_t a=value(t.body[index(t,T_CNA,0x1070)])-probabilities.mem.dma_addr;
            const size_t out=value(t.body[index(t,T_DPU,0x4020)])-partial.mem.dma_addr;
            const int batch=a/astride,block=is_tail?blocks-1:batch;
            if(batch>=per_head || out/ostride!=size_t(batch))throw Error{RK_NPU_ERR_PARAM};
            patch(t,T_CNA,0x1070,probabilities.mem.dma_addr+(size_t(hi)*span+block*c.kv_tile)*m*2+a%astride);
            patch(t,T_DPU,0x4020,partial.mem.dma_addr+size_t(hi*blocks+block)*ostride+out%ostride);
            relocs.emplace_back(all.size(),is_tail?2:1,hi,block);all.push_back(std::move(t));
        }
    };
    if(lut){
        const int cores=__builtin_popcount(core_mask(c));
        for(int core=0,first=0;core<cores;++core){
            const int heads=c.kv_heads/cores+(core<c.kv_heads%cores);
            if(lut_split){
                Task t;t.body=lut_exp_regs(8,score.mem.dma_addr+size_t(first)*m*span*2,
                    exponent.mem.dma_addr+size_t(first)*m*span*2,true);
                t.op_idx=4;t.enable_mask=0x18;
                relocs.emplace_back(all.size(),4,first,0);all.push_back(std::move(t));
            }
            for(int hi=first;hi<first+heads;++hi)for(int at=0;at<m*span;at+=4096){
                Task t;
                const size_t offset=(size_t(hi)*m*span+at)*2;
                t.body=lut_exp_regs(std::min(4096,m*span-at),score.mem.dma_addr+offset,
                    exponent.mem.dma_addr+offset,!lut_split && hi==first && at==0);
                t.op_idx=4;t.enable_mask=0x18;
                relocs.emplace_back(all.size(),3,hi,0);all.push_back(std::move(t));
            }
            first+=heads;
        }
    }
    if(full)pv(full,c.kv_tile,false);
    if(tail_length)pv(1,tail_length,true);
    tasks=int(all.size());if(tasks<1 || tasks>4095)throw Error{RK_NPU_ERR_PARAM};
    const int cores=__builtin_popcount(core_mask(c));
    const bool stable=c.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    const bool causal_lut=lut && c.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL;
    const int stages=stable?((causal_lut || lut_split)?3:2):1;
    if(cores==1 && !stable)task_count[0]=tasks;
    else {
        // Each core owns contiguous, disjoint KV heads and their complete
        // QK -> fused exp/mask -> PV chains. No cross-core dependency/barrier.
        std::vector<Task> ordered;ordered.reserve(all.size());
        std::vector<std::tuple<size_t,int,int,int>> bindings_in_order;
        for(int stage=0;stage<stages;++stage){
        int first_head=0;
        for(int core=0;core<cores;++core){
            const int heads=c.kv_heads/cores+(core<c.kv_heads%cores);
            const auto begin=ordered.size();
            for(auto [task,kind,hi,block]:relocs)if(hi>=first_head && hi<first_head+heads &&
                (!stable || (stage==0?kind==0:causal_lut?(stage==1?kind>=3:kind==1 || kind==2):stage==2?kind==4:kind!=0 && kind!=4))){
                bindings_in_order.emplace_back(ordered.size(),kind,hi,block);
                ordered.push_back(std::move(all[task]));
            }
            if(stable){stage_start[stage][core]=begin;stage_count[stage][core]=ordered.size()-begin;}
            else {task_start[core]=begin;task_count[core]=ordered.size()-begin;}
            first_head+=heads;
        }
        }
        if(ordered.size()!=all.size())throw Error{RK_NPU_ERR_PARAM};
        all.swap(ordered);relocs.swap(bindings_in_order);
    }
    std::vector<size_t> offsets;size_t words=0;
    for(auto& t:all){offsets.push_back(words);words+=align_up(int(t.body.size())+4,2);}
    if(regs.mem.size<words*8)ensure(regs,words*8,RK_NPU_MEM_REGCMD_PAGED);
    if(descriptors.mem.size<size_t(tasks)*sizeof(rknpu_task))ensure(descriptors,size_t(tasks)*sizeof(rknpu_task),RK_NPU_MEM_TASK_PAGED);
    auto* cmd=static_cast<uint64_t*>(regs.mem.vaddr);auto* desc=static_cast<rknpu_task*>(descriptors.mem.vaddr);
    std::memset(desc,0,size_t(tasks)*sizeof(*desc));
    for(int i=0;i<tasks;++i){
        auto& t=all[i];std::copy(t.body.begin(),t.body.end(),cmd+offsets[i]);auto* end=cmd+offsets[i]+t.body.size();
        bool next=i+1<tasks;
        for(int core=0;core<cores;++core){
            if(!stable && uint32_t(i+1)==task_start[core]+task_count[core])next=false;
            if(stable)for(int stage=0;stage<stages;++stage)
                if(uint32_t(i+1)==stage_start[stage][core]+stage_count[stage][core])next=false;
        }
        end[0]=next?E(T_PC_REG,R_PC_BASE_ADDRESS,regs.mem.dma_addr+offsets[i+1]*8):E(T_NOP,0,0);
        end[1]=E(T_PC_REG,R_PC_REGISTER_AMOUNTS,next?ceil_div(int(all[i+1].body.size()),2)+1:0);
        end[2]=E(T_VERSION,0,0);end[3]=E(T_PC,R_OPERATION_ENABLE,t.enable_mask);
        if(t.body.size()%2)end[4]=E(T_NOP,0,0);
        desc[i].op_idx=t.op_idx;desc[i].enable_mask=t.enable_mask;desc[i].int_mask=0x300;desc[i].int_clear=0x1ffff;
        desc[i].regcfg_amount=t.body.size();desc[i].regcmd_addr=regs.mem.dma_addr+offsets[i]*8;
    }
    for(auto [task,kind,hi,block]:relocs)if(kind<3)
        bindings.push_back({offsets[task]+index(all[task],T_CNA,0x1110),kind,hi,block});
    descriptors.sync(RK_NPU_SYNC_TO_DEVICE);
    bind(cache,bounds,scratch);
}
void Graph::bind(Cache& c,const uint16_t* bounds,Tail* scratch){
    std::copy_n(bounds?bounds:c.value_absmax,c.h(),value_absmax);
    const auto& tail=scratch?*scratch:c.tail;
    if(bound_identity==c.identity && bound_generation==c.generation &&
       bound_tail_generation==tail.generation && bound_tail_address==tail.mem.dma_addr)return;
    auto* cmd=static_cast<uint64_t*>(regs.mem.vaddr);
    for(auto b:bindings){
        const int head=shape.shared_kv?0:b.head;
        uint64_t addr=b.kind==0?c.key.mem.dma_addr+(size_t(head)*c.capacity+b.block)*shape.key_dim*2:
            b.kind==1?c.value.mem.dma_addr+c.voffset(head,b.block)*2:
                      tail.mem.dma_addr+size_t(head)*(shape.value_dim+32)*tail_length*2;
        if(addr>UINT32_MAX)throw Error{RK_NPU_ERR_DOMAIN};cmd[b.word]=E(T_CNA,0x1110,uint32_t(addr));
    }
    regs.sync(RK_NPU_SYNC_TO_DEVICE);
    bound=&c;bound_identity=c.identity;bound_generation=c.generation;
    bound_tail_generation=tail.generation;bound_tail_address=tail.mem.dma_addr;
}
void Graph::update_mask(int start,const rk_npu_attention_f16_boolean_mask* b,int length){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.mask_update", "query_start", start,
        "kv_length", length, "mask_mode", int(cfg.mask_mode));
    if(cfg.mask_mode==RK_NPU_ATTENTION_F16_NO_MASK)return;
    const bool causal=cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL;
    if(causal && mask_start==start)return;
    if(!causal && same_mask(saved_mask,*b) && mask_start==length)return;
#ifdef ATTENTION_NEON
    const int16_t lanes_data[]={0,1,2,3,4,5,6,7};const auto lanes=vld1q_s16(lanes_data);
#endif
    auto causal_word=[&](uint16_t* dst,int end,int column){
#ifdef ATTENTION_NEON
        vst1q_u16(dst,vandq_u16(vcleq_s16(lanes,vdupq_n_s16(int16_t(end-column))),vdupq_n_u16(0x3c00)));
#else
        for(int lane=0;lane<8;++lane)dst[lane]=column+lane<=end?0x3c00:0;
#endif
    };
    if(causal){
        const bool band=rows>1 && mask_start>=0 && start>mask_start && start-mask_start<=rows;
        const bool single=rows==1 && mask_start>=0 && start==mask_start+1;
        const int lo=band?(mask_start+1)/8*8:single?start/8*8:((start+1)/8*8);
        const int hi_pos=band?align_up(start+rows,8):single?lo+8:align_up(start+rows,8);
        for(int h=0;h<cfg.kv_heads;++h){
            const int real_rows=shape.heads(cfg,h)*rows;auto* base=mask.data()+size_t(h)*span*m;
            int ends[128];for(int r=0;r<m;++r)ends[r]=r<real_rows?start+r%rows:-1;
            if(!band && !single){
                std::fill_n(base,size_t(lo)*m,uint16_t(0x3c00));
                std::fill_n(base+size_t(hi_pos)*m,size_t(span-hi_pos)*m,uint16_t(0));
                if(real_rows<m)for(int j=0;j<lo;j+=8)for(int r=real_rows;r<m;++r)
                    std::fill_n(base+(size_t(j/8)*m+r)*8,8,uint16_t(0));
            }
            for(int j=lo;j<hi_pos;j+=8)for(int r=0;r<m;++r)
                causal_word(base+(size_t(j/8)*m+r)*8,ends[r],j);
            if(band || single)mask.range((size_t(h)*span+lo)*m*2,size_t(hi_pos-lo)*m*2);
        }
        if(!band && !single)mask.range(0,size_t(cfg.kv_heads)*span*m*2);
        mask_start=start;return;
    }
    const bool incremental=rows==1 && same_mask(saved_mask,*b) && length==mask_start+1;
    const int first=incremental?length-1:0,last=incremental?first+1:span;
    for(int hi=0;hi<cfg.kv_heads;++hi){
        const int real_rows=shape.heads(cfg,hi)*rows,head_first=shape.first_head(cfg,hi);
        const uint8_t* sources[128]{};
        for(int r=0;r<real_rows;++r){const int h=b->heads==1?0:head_first+r/rows,qr=b->query_rows==1?0:b->query_offset+r%rows;
            sources[r]=b->data+uint64_t(h)*b->head_stride+uint64_t(qr)*b->query_stride;}
        if(incremental){
            for(int r=0;r<m;++r){bool keep=r<real_rows && sources[r][b->key_length==1?0:uint64_t(first)*b->key_stride]!=0;
                mask.data()[((size_t(hi)*(span/8)+first/8)*m+r)*8+first%8]=keep?0x3c00:0;}
            mask.range((size_t(hi)*span+first/8*8)*m*2,size_t(m)*8*2);continue;
        }
        for(int j=0;j<last;j+=8)for(int r=0;r<m;++r){
            auto* dst=mask.data()+((size_t(hi)*(span/8)+j/8)*m+r)*8;
#ifdef ATTENTION_NEON
            if(r<real_rows && b->key_length!=1 && b->key_stride==1 && j+8<=length){
                const auto x=vmovl_u8(vld1_u8(sources[r]+j));
                vst1q_u16(dst,vandq_u16(vcgtq_u16(x,vdupq_n_u16(0)),vdupq_n_u16(0x3c00)));continue;
            }
#endif
            for(int lane=0;lane<8;++lane){const int pos=j+lane;
                const bool keep=r<real_rows && pos<length && sources[r][b->key_length==1?0:uint64_t(pos)*b->key_stride]!=0;
                dst[lane]=keep?0x3c00:0;}
        }
    }
    if(!incremental)mask.range(0,size_t(cfg.kv_heads)*span*m*2);
    mask_start=length;saved_mask=*b;
}
void Graph::pack_q(const uint16_t* query,int tensor_rows,int tensor_offset){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.query_pack", "query_rows", rows);
#ifdef ATTENTION_NEON
    const auto scale=vdupq_n_f16(float16_t(shape.scale));
#endif
    const int query_stride=tensor_rows?tensor_rows:rows;
#pragma omp parallel for num_threads(4) if(rows>1 && m>16) schedule(static)
    for(int hi=0;hi<cfg.kv_heads;++hi){
    const int real_rows=shape.heads(cfg,hi)*rows,first=shape.first_head(cfg,hi);
    for(int kb=0;kb<shape.key_dim/8;++kb)for(int r=0;r<m;++r){
        auto* dst=q.data()+((size_t(hi)*(shape.key_dim/8)+kb)*m+r)*8;
        if(r>=real_rows){std::fill_n(dst,8,0);continue;}
        const auto* src=query+((size_t(first)+r/rows)*query_stride+tensor_offset+r%rows)*shape.key_dim+kb*8;
#ifdef ATTENTION_NEON
        vst1q_u16(dst,vreinterpretq_u16_f16(vmulq_f16(vreinterpretq_f16_u16(vld1q_u16(src)),scale)));
#else
        for(int x=0;x<8;++x)dst[x]=half(widen(src[x])*widen(half(shape.scale)));
#endif
    }
    }
    q.range(0,size_t(cfg.kv_heads)*m*shape.key_dim*2);
}
double Graph::submit(int stage){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.graph_submit", "stage", stage,
        "query_rows", rows, "physical_M", m, "compute_length", span, "tasks", tasks,
        "npu_mask", core_mask(cfg));
    rknpu_submit s{};s.flags=lut && stage!=0?(stage==2?RKNPU_JOB_PC:lut_flags):RKNPU_JOB_PC|RKNPU_JOB_PINGPONG;s.timeout=cfg.timeout_ms?cfg.timeout_ms:2000;
    s.task_number=tasks;s.task_obj_addr=descriptors.mem.obj_addr;s.iommu_domain_id=domain->id;s.core_mask=core_mask(cfg);s.fence_fd=-1;
    // Match the existing board-validated driver ABI: dual uses slots 0/1,
    // triple uses 2/3/4, single uses the physical core's slot.
    const int cores=__builtin_popcount(s.core_mask);int logical=0;
    for(int physical=0;physical<3;++physical)if(s.core_mask&(1u<<physical)){
        const int slot=cores==3?physical+2:cores==2?logical:physical;
        s.subcore_task[slot]=cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX?
            rknpu_subcore_task{stage_start[stage][logical],stage_count[stage][logical]}:
            rknpu_subcore_task{task_start[logical],task_count[logical]};++logical;
    }
    if(ioctl(domain->ctx->fd,IOCTL_SUBMIT,&s)<0)throw Error{RK_NPU_ERR_SUBMIT};
    return s.hw_elapse_time/1000.;
}
void Graph::sync_scores(rk_npu_sync_dir direction){
    if(direction==RK_NPU_SYNC_FROM_DEVICE){
        RK_NPU_TRACE_EVENT("rknpu.detail", "attention.score_sync_from_device");
        score.range(0,size_t(cfg.kv_heads)*m*span*2,direction);
    }else{
        RK_NPU_TRACE_EVENT("rknpu.detail", "attention.score_sync_to_device");
        score.range(0,size_t(cfg.kv_heads)*m*span*2,direction);
    }
}
void Graph::process_scores(int start,const rk_npu_attention_f16_boolean_mask* mask,int length){
    stable_softmax(start,mask,length);sync_scores(RK_NPU_SYNC_TO_DEVICE);
}
double Graph::submit_pv(int start){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.pv_submit");
    if(lut && cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL){
        double hardware=submit(1);mask_start=start;
        // Masked LUT scores still emit a positive carrier. Remove future
        // carriers before FP16 PV so they cannot change its rounding.
        const int first=(start+1)/8*8;
        const size_t bytes=size_t(span-first)*m*2;
        for(int hi=0;hi<cfg.kv_heads;++hi){
            const size_t offset=(size_t(hi)*span+first)*m*2;
            exponent.range(offset,bytes,RK_NPU_SYNC_FROM_DEVICE);
            for(int r=0;r<shape.heads(cfg,hi)*rows;++r){
                const int end=start+r%rows+1;
                for(int j=end;j<span;++j)
                    exponent.data()[size_t(hi)*span*m+(size_t(j/8)*m+r)*8+j%8]=0;
            }
            exponent.range(offset,bytes);
        }
        return hardware+submit(2);
    }
    double hardware=lut && lut_split?submit(2):0;
    return hardware+submit(1);
}
void Graph::sync_output(){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.output_sync");
    partial.range(0,size_t(cfg.kv_heads)*blocks*(shape.value_dim+32)*m*2,RK_NPU_SYNC_FROM_DEVICE);
}
void Graph::stable_softmax(int start,const rk_npu_attention_f16_boolean_mask* mask_data,int length){
    if(lut){center_scores(start,mask_data,length);return;}
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.stable_softmax", "query_rows", rows,
        "kv_length", length, "compute_length", span);
    // Raw QK and PV use the same native block-8 tensor. Long sequences scan
    // Up to 128 adjacent rows together keep native score reads contiguous.
    // Small tensors keep row scans; normalized scratch uses four-row groups
    // and is retained by the workspace.
    // When tile*max(abs(V)) <= 32752, exp(score-max) gives finite FP16 PV
    // partials with a factor-of-two margin. PV already computes [E@V,E@1]
    // and finish() divides in FP32; omit CPU normalization in that case.
    // Larger V keeps normalized probabilities and the previous range bound.
    // Short tensors use degree-6 FP32 exp; long tensors split a degree-5
    // polynomial into independent terms. x<=-80 contributes less than 2e-35.
#ifdef ATTENTION_NEON
    auto exp4=[](float32x4_t x){
        x=vmaxq_f32(x,vdupq_n_f32(-80.f));
        const auto nf=vrndmq_f32(vmlaq_n_f32(vdupq_n_f32(.5f),x,1.4426950408889634f));
        auto r=vsubq_f32(x,vmulq_n_f32(nf,.693145751953125f));
        r=vsubq_f32(r,vmulq_n_f32(nf,1.428606765330187e-6f));
        auto p=vdupq_n_f32(1.f/720.f);
        p=vfmaq_f32(vdupq_n_f32(1.f/120.f),p,r);
        p=vfmaq_f32(vdupq_n_f32(1.f/24.f),p,r);
        p=vfmaq_f32(vdupq_n_f32(1.f/6.f),p,r);
        p=vfmaq_f32(vdupq_n_f32(.5f),p,r);
        p=vfmaq_f32(vdupq_n_f32(1.f),p,r);
        p=vfmaq_f32(vdupq_n_f32(1.f),p,r);
        const auto power=vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(nf),vdupq_n_s32(127)),23));
        return vmulq_f32(p,power);
    };
    // Adapted from ggml's ARM exp approximation (LICENSES/ggml-MIT.txt).
    // score-max is nonpositive; clamping to -80 keeps the exponent normal.
    auto long_exp4=[](float32x4_t x){
        x=vmaxq_f32(x,vdupq_n_f32(-80.f));
        const auto z=vfmaq_f32(vdupq_n_f32(0x1.8p23f),x,vdupq_n_f32(0x1.715476p+0f));
        const auto n=vsubq_f32(z,vdupq_n_f32(0x1.8p23f));
        const auto b=vfmsq_f32(vfmsq_f32(x,n,vdupq_n_f32(0x1.62e4p-1f)),n,vdupq_n_f32(0x1.7f7d1cp-20f));
        const auto e=vshlq_n_u32(vreinterpretq_u32_f32(z),23);
        const auto k=vreinterpretq_f32_u32(vaddq_u32(e,vreinterpretq_u32_f32(vdupq_n_f32(1.f))));
        const auto u=vmulq_f32(b,b);
        const auto j=vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f),b),
            vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f),vdupq_n_f32(0x1.555e66p-3f),b),
                      vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f),vdupq_n_f32(0x1.0e4020p-7f),b),u),u);
        return vfmaq_f32(k,j,k);
    };
#endif
    // Finite Q/K/V and representable FP16 QK scores are caller preconditions.
    // Keep the -INFINITY sentinel for padding and fully hidden rows only.
    // Keep the original row-wise normalized algorithm for small tensors,
    // including its cache-resident per-head exponent buffer.
    if(span<=1024){
#pragma omp parallel for num_threads(4)
    for(int hi=0;hi<cfg.kv_heads;++hi){
        std::vector<float> exponents(span);
        const bool half_probability=value_absmax[shape.shared_kv?0:hi]>0x77ff; // FP16 32752
        const int real_rows=shape.heads(cfg,hi)*rows;
        for(int r=0;r<m;++r){
            const int end=r>=real_rows?0:cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL?
                std::min(length,start+r%rows+1):length;
            auto visible=[&](int j){
                if(!mask_data)return true;
                const int head=shape.first_head(cfg,hi)+r/rows;
                const int h=mask_data->heads==1?0:head;
                const int q=mask_data->query_rows==1?0:mask_data->query_offset+r%rows;
                const int k=mask_data->key_length==1?0:j;
                return mask_data->data[uint64_t(h)*mask_data->head_stride+uint64_t(q)*mask_data->query_stride+uint64_t(k)*mask_data->key_stride]!=0;
            };
            auto at=[&](int j){return score.data()+(size_t(hi)*span*m+(size_t(j/8)*m+r)*8);};
            float maximum=-INFINITY;
            int j=0;
#ifdef ATTENTION_NEON
            if(!mask_data){
                auto mx=vdupq_n_f32(-INFINITY);
                for(;j+8<=end;j+=8){auto x=vreinterpretq_f16_u16(vld1q_u16(at(j)));
                    mx=vmaxq_f32(mx,vcvt_f32_f16(vget_low_f16(x)));mx=vmaxq_f32(mx,vcvt_f32_f16(vget_high_f16(x)));}
                maximum=vmaxvq_f32(mx);
            }
#endif
            for(;j<end;++j)if(visible(j))maximum=std::max(maximum,widen(at(j)[j%8]));
            float denominator=0;
            j=0;
            if(maximum!=-INFINITY){
#ifdef ATTENTION_NEON
                if(!mask_data){
                    auto total=vdupq_n_f32(0),mx=vdupq_n_f32(maximum);
                    for(;j+8<=end;j+=8){auto x=vreinterpretq_f16_u16(vld1q_u16(at(j)));
                        const auto low=exp4(vsubq_f32(vcvt_f32_f16(vget_low_f16(x)),mx));
                        const auto high=exp4(vsubq_f32(vcvt_f32_f16(vget_high_f16(x)),mx));
                        vst1q_f32(exponents.data()+j,low);vst1q_f32(exponents.data()+j+4,high);
                        total=vaddq_f32(total,vaddq_f32(low,high));}
                    denominator=vaddvq_f32(total);
                }
#endif
                for(;j<end;++j){const float p=visible(j)?std::exp(widen(at(j)[j%8])-maximum):0;
                    exponents[j]=p;denominator+=p;}
            }
            float inverse=denominator?1.f/denominator:0;
            if(half_probability)inverse*=.5f;
            for(j=0;j<span;j+=8){
                auto* dst=at(j);
#ifdef ATTENTION_NEON
                if(j+8<=end && inverse){
                    const auto low=vmulq_n_f32(vld1q_f32(exponents.data()+j),inverse);
                    const auto high=vmulq_n_f32(vld1q_f32(exponents.data()+j+4),inverse);
                    vst1q_u16(dst,vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high))));continue;
                }
#endif
                for(int lane=0;lane<8;++lane)dst[lane]=j+lane<end && inverse?half(exponents[j+lane]*inverse):0;
            }
        }
    }
        return;
    }
    // Compile separate scan sizes so small query batches retain their original
    // row scheduling and worker stack footprint.
    auto scan_rows=[&](auto block_tag
#ifdef ATTENTION_NEON
        ,auto exp4
#endif
        ){
    constexpr int max_row_block=decltype(block_tag)::value,scratch_rows=4;
    const int row_block=m>=32?std::min(m,max_row_block):scratch_rows;
    const int row_blocks=ceil_div(m,row_block);
    const int threads=size_t(cfg.query_heads)*rows*length>=4096?4:1;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int task=0;task<cfg.kv_heads*row_blocks;++task){
        const int hi=task/row_blocks,base=(task%row_blocks)*row_block;
        const int limit=std::min(m,base+row_block);
        const int real_rows=shape.heads(cfg,hi)*rows;
        const int cache_head=shape.shared_kv?0:hi;
        auto process=[&](auto delay_tag,auto mask_tag){
        constexpr bool delayed=decltype(delay_tag)::value,masked=decltype(mask_tag)::value;
        // Near FP16's upper V limit, rounded probabilities can sum slightly
        // above one. A common 1/2 gain also cancels in finish(), keeping the
        // normalized fallback's numerator safely away from FP16 overflow.
        const float probability_gain=widen(value_absmax[cache_head])>32752.f?.5f:1.f;
        const int process_rows=delayed?(max_row_block>32 || span>=2048?row_block:1):scratch_rows;
        auto* scratch=delayed?nullptr:softmax_scratch.data()+size_t(omp_get_thread_num())*scratch_rows*span;
        for(int first=base;first<limit;first+=process_rows){
        const int count=std::min(process_rows,limit-first);
        int ends[max_row_block];
        float maximum[max_row_block],denominator[max_row_block],inverse[max_row_block];
        for(int r=0;r<count;++r){
            const int row=first+r;
            ends[r]=row>=real_rows?0:cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL?
                std::min(length,start+row%rows+1):length;
            maximum[r]=-INFINITY;
            denominator[r]=inverse[r]=0;
        }
        auto visible=[&](int r,int j){
            if constexpr(!masked)return true;
            const int head=shape.first_head(cfg,hi)+(first+r)/rows;
            const int h=mask_data->heads==1?0:head;
            const int q=mask_data->query_rows==1?0:mask_data->query_offset+(first+r)%rows;
            const int k=mask_data->key_length==1?0:j;
            return mask_data->data[uint64_t(h)*mask_data->head_stride+uint64_t(q)*mask_data->query_stride+uint64_t(k)*mask_data->key_stride]!=0;
        };
        auto at=[&](int r,int j){return score.data()+(size_t(hi)*span*m+(size_t(j/8)*m+first+r)*8);};
        const int last=*std::max_element(ends,ends+count);
#ifdef ATTENTION_NEON
        float16x8_t maxima[max_row_block];
        for(int r=0;r<count;++r)maxima[r]=vdupq_n_f16(-INFINITY);
#endif
        for(int j=0;j<last;j+=8)for(int r=0;r<count;++r){
#ifdef ATTENTION_NEON
            if(!masked && j+8<=ends[r]){
                maxima[r]=vmaxq_f16(maxima[r],vreinterpretq_f16_u16(vld1q_u16(at(r,j))));continue;
            }
#endif
            for(int lane=0;lane<8 && j+lane<ends[r];++lane)if(visible(r,j+lane))
                maximum[r]=std::max(maximum[r],widen(at(r,j)[lane]));
        }
#ifdef ATTENTION_NEON
        float32x4_t totals[max_row_block];
        for(int r=0;r<count;++r){
            const float mx=vmaxvq_f16(maxima[r]);
            maximum[r]=std::max(maximum[r],mx);totals[r]=vdupq_n_f32(0);
        }
#endif
        for(int j=0;j<span;j+=8)for(int r=0;r<count;++r){
            auto* dst=at(r,j);
            if(j>=ends[r] || maximum[r]==-INFINITY){
                std::fill_n(dst,8,uint16_t(0));continue;
            }
#ifdef ATTENTION_NEON
            if(!masked && j+8<=ends[r]){
                const auto x=vreinterpretq_f16_u16(vld1q_u16(dst));
                const auto mx=vdupq_n_f32(maximum[r]);
                const auto low=exp4(vsubq_f32(vcvt_f32_f16(vget_low_f16(x)),mx));
                const auto high=exp4(vsubq_f32(vcvt_f32_f16(vget_high_f16(x)),mx));
                if(delayed)vst1q_u16(dst,vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high))));
                else {
                    auto* tmp=scratch+size_t(r)*span+j;
                    vst1q_f32(tmp,low);vst1q_f32(tmp+4,high);
                    totals[r]=vaddq_f32(totals[r],vaddq_f32(low,high));
                }
                continue;
            }
#endif
            for(int lane=0;lane<8;++lane){
                const float p=j+lane<ends[r] && visible(r,j+lane)?std::exp(widen(dst[lane])-maximum[r]):0;
                if(delayed)dst[lane]=half(p);
                else {scratch[size_t(r)*span+j+lane]=p;denominator[r]+=p;}
            }
        }
        if(delayed)continue;
        for(int r=0;r<count;++r){
#ifdef ATTENTION_NEON
            denominator[r]+=vaddvq_f32(totals[r]);
#endif
            inverse[r]=denominator[r]?probability_gain/denominator[r]:0;
        }
        for(int j=0;j<span;j+=8)for(int r=0;r<count;++r){
            if(j>=ends[r] || !inverse[r])continue; // Already zeroed above.
            auto* dst=at(r,j);
#ifdef ATTENTION_NEON
            if(j+8<=ends[r]){
                const auto low=vmulq_n_f32(vld1q_f32(scratch+size_t(r)*span+j),inverse[r]);
                const auto high=vmulq_n_f32(vld1q_f32(scratch+size_t(r)*span+j+4),inverse[r]);
                vst1q_u16(dst,vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high))));continue;
            }
#endif
            for(int lane=0;lane<8;++lane)dst[lane]=j+lane<ends[r]?half(scratch[size_t(r)*span+j+lane]*inverse[r]):0;
        }
        }
        };
        // Specialize the hot loops so normalized scratch/denominator work
        // and boolean-mask address calculations do not spill into exp-only.
        const bool delayed=widen(value_absmax[cache_head])<=32752.f/cfg.kv_tile;
        if(delayed){
            if(mask_data)process(std::true_type{},std::true_type{});
            else process(std::true_type{},std::false_type{});
        }else {
            if(mask_data)process(std::false_type{},std::true_type{});
            else process(std::false_type{},std::false_type{});
        }
    }
    };
    auto scan=[&](auto tag){scan_rows(tag
#ifdef ATTENTION_NEON
        ,long_exp4
#endif
        );};
    if(m>32)scan(std::integral_constant<int,128>{});
    else scan(std::integral_constant<int,32>{});
}
void Graph::center_scores(int start,const rk_npu_attention_f16_boolean_mask* mask_data,int length){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.center_scores", "compute_length", span);
    const int row_block=std::min(m,128),row_blocks=ceil_div(m,row_block);
    const int threads=size_t(cfg.query_heads)*rows*length>=4096?4:1;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int task=0;task<cfg.kv_heads*row_blocks;++task){
        const int hi=task/row_blocks,first=task%row_blocks*row_block;
        const int count=std::min(row_block,m-first),real_rows=shape.heads(cfg,hi)*rows;
        int ends[128];float maxima[128];
        auto visible=[&](int r,int j){
            if(!mask_data)return true;
            const int head=shape.first_head(cfg,hi)+(first+r)/rows;
            const int h=mask_data->heads==1?0:head;
            const int q=mask_data->query_rows==1?0:mask_data->query_offset+(first+r)%rows;
            const int k=mask_data->key_length==1?0:j;
            return mask_data->data[uint64_t(h)*mask_data->head_stride+uint64_t(q)*mask_data->query_stride+uint64_t(k)*mask_data->key_stride]!=0;
        };
        auto at=[&](int r,int j){return score.data()+(size_t(hi)*span*m+(size_t(j/8)*m+first+r)*8);};
#ifdef ATTENTION_NEON
        float16x8_t mx[128];
#endif
        for(int r=0;r<count;++r){
            ends[r]=first+r>=real_rows?0:cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL?
                std::min(length,start+(first+r)%rows+1):length;
            maxima[r]=-INFINITY;
#ifdef ATTENTION_NEON
            mx[r]=vdupq_n_f16(-INFINITY);
#endif
        }
        const int last=*std::max_element(ends,ends+count);
        for(int j=0;j<last;j+=8)for(int r=0;r<count;++r){
#ifdef ATTENTION_NEON
            if(!mask_data && j+8<=ends[r]){mx[r]=vmaxq_f16(mx[r],vreinterpretq_f16_u16(vld1q_u16(at(r,j))));continue;}
#endif
            for(int lane=0;lane<8 && j+lane<ends[r];++lane)if(visible(r,j+lane))
                maxima[r]=std::max(maxima[r],widen(at(r,j)[lane]));
        }
        for(int r=0;r<count;++r){
#ifdef ATTENTION_NEON
            maxima[r]=std::max(maxima[r],float(vmaxvq_f16(mx[r])));
#endif
            active[size_t(hi)*m+first+r]=maxima[r]!=-INFINITY;
        }
        for(int j=0;j<span;j+=8)for(int r=0;r<count;++r){
            auto* dst=at(r,j);
            if(j>=ends[r] || maxima[r]==-INFINITY){std::fill_n(dst,8,half(-16));continue;}
#ifdef ATTENTION_NEON
            if(!mask_data && j+8<=ends[r]){
                const auto x=vreinterpretq_f16_u16(vld1q_u16(dst));
                // The maximum is an FP16 score and the result is stored as
                // FP16. Subtract directly in that format, then clamp to -16.
                vst1q_u16(dst,vreinterpretq_u16_f16(vmaxq_f16(
                    vsubq_f16(x,vdupq_n_f16(float16_t(maxima[r]))),vdupq_n_f16(-16))));continue;
            }
#endif
            for(int lane=0;lane<8;++lane)dst[lane]=half(j+lane<ends[r] && visible(r,j+lane)?
                std::max(-16.f,widen(dst[lane])-maxima[r]):-16.f);
        }
    }
}
void Graph::finish(void* output,int tensor_rows,int tensor_offset,bool half_output){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.output_finish", "query_rows", rows);
    const size_t stride=size_t(shape.value_dim+32)*m,count=sum.size();
    const int output_stride=tensor_rows?tensor_rows:rows;
    const bool causal_lut=lut && cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL;
    for(int hi=0;hi<cfg.kv_heads;++hi){
        for(int block=0;block<blocks;++block){
            const auto* src=partial.data()+size_t(hi*blocks+block)*stride;
#ifdef ATTENTION_NEON
            for(size_t i=0;i<count;i+=8){
                auto x=vreinterpretq_f16_u16(vld1q_u16(src+i));auto lo=vcvt_f32_f16(vget_low_f16(x)),high=vcvt_f32_f16(vget_high_f16(x));
                if(block){lo=vaddq_f32(lo,vld1q_f32(sum.data()+i));high=vaddq_f32(high,vld1q_f32(sum.data()+i+4));}
                vst1q_f32(sum.data()+i,lo);vst1q_f32(sum.data()+i+4,high);
            }
#else
            for(size_t i=0;i<count;++i)sum[i]=(block?sum[i]:0)+widen(src[i]);
#endif
        }
        if(causal_lut){
            const int h=shape.shared_kv?0:hi,d=shape.value_dim,first=mask_start/32*32;
            const auto* prefix=bound->value_prefix.data()+(size_t(mask_start/32)*bound->h()+h)*d;
            std::copy_n(prefix,d,softmax_scratch.data());
            auto add=[&](float* dst,int pos){
                const int inner=pos%cfg.kv_tile;
                const auto* v=bound->value.data()+bound->voffset(h,pos/cfg.kv_tile);
                for(int c=0;c<d;++c){
                    const uint16_t raw=v[((size_t(c/16)*(cfg.kv_tile/32)+inner/32)*16+c%16)*32+inner%32];
#ifdef ATTENTION_NEON
                    __fp16 value;std::memcpy(&value,&raw,sizeof(value));dst[c]+=float(value);
#else
                    dst[c]+=widen(raw);
#endif
                }
            };
            // A checkpoint leaves at most 31 old tokens. Continue the exact
            // FP32 token order used by Cache::add_value_sum; no subtract-back.
            for(int pos=first;pos<=mask_start;++pos)add(softmax_scratch.data(),pos);
            for(int row=1;row<rows;++row){
                auto* dst=softmax_scratch.data()+size_t(row)*d;
                std::copy_n(dst-d,d,dst);add(dst,mask_start+row);
            }
        }
        for(int r=0;r<shape.heads(cfg,hi)*rows;++r){
            constexpr float delta=1.f/2048;
            const int length=causal_lut?mask_start+r%rows+1:std::min(span,bound->length);
            const float z=sum[size_t(shape.value_dim/8)*m*8+r*8]-(lut?delta*length:0);
            const float inverse=z==0 || (lut && !active[size_t(hi)*m+r])?0:1.f/z;
            const float* correction=causal_lut?softmax_scratch.data()+size_t(r%rows)*shape.value_dim:
                lut?bound->value_sum(shape.shared_kv?0:hi,span):nullptr;
            for(int cb=0;cb<shape.value_dim/8;++cb){
                const auto* src=sum.data()+size_t(cb)*m*8+r*8;
                const size_t target=((size_t(shape.first_head(cfg,hi))+r/rows)*output_stride+tensor_offset+r%rows)*shape.value_dim+cb*8;
#ifdef ATTENTION_NEON
                auto lo=vld1q_f32(src),high=vld1q_f32(src+4);
                if(lut){lo=vsubq_f32(lo,vmulq_n_f32(vld1q_f32(correction+cb*8),delta));
                    high=vsubq_f32(high,vmulq_n_f32(vld1q_f32(correction+cb*8+4),delta));}
                lo=vmulq_n_f32(lo,inverse);high=vmulq_n_f32(high,inverse);
                if(half_output)vst1q_u16(static_cast<uint16_t*>(output)+target,
                    vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(high))));
                else {auto* dst=static_cast<float*>(output)+target;vst1q_f32(dst,lo);vst1q_f32(dst+4,high);}
#else
                for(int j=0;j<8;++j){const float value=(src[j]-(lut?delta*correction[cb*8+j]:0))*inverse;
                    if(half_output)static_cast<uint16_t*>(output)[target+j]=half(value);
                    else static_cast<float*>(output)[target+j]=value;}
#endif
            }
        }
    }
}
void Workspace::geometry(Cache& cache,int rows,int start,rk_npu_attention_f16_timings& t,int query_end,Tail* scratch){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.geometry", "query_rows", rows,
        "query_start", start, "kv_length", cache.length, "kv_tile", cfg.kv_tile);
    const auto begin=Clock::now();
    compatible(*this,cache);if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    const int prefix=prefix_length(cfg,cache,rows,start);
    uint16_t bounds[32]{};
    if(!full_causal_prefill && cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX && cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL)
        cache.softmax_bounds(prefix,bounds);
    else std::copy_n(cache.value_absmax,cache.h(),bounds);
    const bool causal_prefill=full_causal_prefill && rows>1 && cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL;
    // Stable softmax has no persistent mask RDMA to update. Reuse buffers,
    // but trim preloaded future KV out of QK/PV and CPU cache sync. Rounding
    // to a full PV tile keeps one graph for several query chunks and avoids
    // repacking a changing partial tail whenever capacity allows it.
    int span=align_up(causal_prefill && cfg.flags==RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL?cache.length:prefix,32);
    const bool reuse_lut_span=rows>1 && !shape.shared_kv &&
        cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX &&
        cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL && cache.use_lut(bounds);
    if((causal_prefill || reuse_lut_span) && cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX){
        // GQA query() sizes scratch for this request's visible prefix, not
        // all preloaded KV. Reuse across its internal segments within that
        // bound; MLA's existing full-prefill path has a separate size query.
        const int limit=causal_prefill?cache.length:(query_end?query_end:prefix);
        span=std::min(align_up(span,cfg.kv_tile),align_up(limit,32));
    }
    rk_npu_attention_f16_sizes sizes{};int rc=query_engine(cfg,shape,cache.capacity,rows,span,sizes);
    // Extending a LUT graph must not cross a geometry/descriptor boundary
    // and silently select CPU exp. Its numerical mode is chosen per prefix
    // by causal_rows(), independently of any preloaded future KV.
    if(reuse_lut_span && !causal_prefill && (rc ||
       !lut_tasks_fit(cfg,sizes.physical_rows,sizes.compute_length,sizes.tasks,true))){
        span=align_up(prefix,32);rc=query_engine(cfg,shape,cache.capacity,rows,span,sizes);
    }
    check(rc);
    auto graph_cfg=cfg;
    // A final decode query sees every valid KV. Stable softmax zeroes padding
    // probabilities; fixed-shift caches retain zero V/denominator padding.
    // An earlier query into a preloaded cache still needs the causal mask.
    if(rows==1 && cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL && prefix==cache.length)
        graph_cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK;
    auto a=Clock::now();cache.stage_tail(span,scratch);auto b=Clock::now();
    { RK_NPU_TRACE_EVENT("rknpu.detail", "attention.graph_prepare", "compute_length", span);
    if(!graph || graph->rows!=rows || graph->span!=span || graph->cfg.mask_mode!=graph_cfg.mask_mode || graph->lut!=graph->can_lut(cache,bounds)){
        try{auto next=std::make_unique<Graph>(lease.domain,graph_cfg,rows,span,cache,shape,graph.get(),bounds,scratch);graph.swap(next);}
        catch(...){poisoned=true;graph.reset();throw;}
    }else graph->bind(cache,bounds,scratch);
    }
    t.tail_stage_us=elapsed(a,b);t.prepare_us=elapsed(begin,a)+elapsed(b,Clock::now());t.compute_length=span;t.tasks=graph->tasks;
}
rk_npu_attention_f16_timings Workspace::run(Cache& cache,const uint16_t* query,int rows,int start,void* output,
    const rk_npu_attention_f16_boolean_mask* mask,int tensor_rows,int tensor_offset,bool half_output){
    RK_NPU_TRACE_EVENT("rknpu", "attention.run", "query_rows", rows, "query_start", start,
        "query_heads", cfg.query_heads, "kv_heads", cfg.kv_heads,
        "head_dim", cfg.head_dim, "kv_length", cache.length,
        "kv_tile", cfg.kv_tile, "npu_mask", core_mask(cfg), "mask_mode", int(cfg.mask_mode),
        "stable_softmax", cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX);
    if(!query || !output)throw Error{RK_NPU_ERR_PARAM};
    validate_mask(cfg,mask,rows,cache.length);auto begin=Clock::now();rk_npu_attention_f16_timings t{};
    geometry(cache,rows,start,t,tensor_rows?start+tensor_rows-tensor_offset:0);auto prepared=Clock::now();
    const bool stable=cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    if(!stable)graph->update_mask(start,mask,cache.length);auto masked=Clock::now();
    graph->pack_q(query,tensor_rows,tensor_offset);auto packed=Clock::now();
    try{
        if(stable) {
            RK_NPU_TRACE_EVENT("rknpu.detail", "attention.qk_submit");
            t.hardware_us=graph->submit();
        } else {
            RK_NPU_TRACE_EVENT("rknpu.detail", "attention.fused_submit");
            t.hardware_us=graph->submit();
        }
        if(stable){
            graph->sync_scores(RK_NPU_SYNC_FROM_DEVICE);
            graph->process_scores(start,mask,cache.length);
            t.hardware_us+=graph->submit_pv(start);
        }
    }catch(...){poisoned=true;throw;}
    auto computed=Clock::now();
    graph->sync_output();
    auto synced=Clock::now();
    graph->finish(output,tensor_rows,tensor_offset,half_output);auto done=Clock::now();t.mask_update_us=elapsed(prepared,masked);t.query_pack_us=elapsed(masked,packed);
    t.submit_us=elapsed(packed,computed);t.output_sync_us=elapsed(computed,synced);t.finish_us=elapsed(synced,done);t.total_us=elapsed(begin,done);return t;
}

int query_engine(const rk_npu_attention_f16_config& config,const Shape& shape,
    int capacity,int rows,int length,rk_npu_attention_f16_sizes& out){
    const auto* c=&config;
    if(capacity<32 || capacity%32 || capacity>c->max_capacity || rows<1 ||
       rows>c->max_query_rows || length<1 || length>capacity)return RK_NPU_ERR_PARAM;
    rk_npu_attention_f16_sizes result{};const int s=align_up(length,32),m=align_up(shape.group*rows,shape.row_alignment),n=shape.value_dim+32,h=c->kv_heads,physical=shape.shared_kv?1:h;
    result.key_bytes=size_t(physical)*capacity*shape.key_dim*2;result.value_bytes=size_t(physical)*ceil_div(capacity,c->kv_tile)*n*c->kv_tile*2;
    result.tail_bytes=size_t(physical)*n*(s%c->kv_tile)*2;result.query_bytes=size_t(h)*m*shape.key_dim*2;
    result.score_bytes=size_t(h)*m*s*2;
    // Stable masking is performed while the CPU reads raw scores; no native
    // mask tensor or DPU mask RDMA is used by either NPU submit.
    result.mask_bytes=c->flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX || c->mask_mode==RK_NPU_ATTENTION_F16_NO_MASK?0:result.score_bytes;
    result.partial_bytes=size_t(h)*ceil_div(s,c->kv_tile)*n*m*2;result.output_bytes=size_t(c->query_heads)*rows*shape.value_dim*4;
    result.compute_length=s;result.physical_rows=m;result.active_core_mask=core_mask(*c);
    rk_npu_matmul_f16_config f{};rk_npu_matmul_f16_config_init(&f,m,s,shape.key_dim,RK_NPU_FUSE_NONE);
    f.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;f.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;f.core_mask=core_mask(*c)&-core_mask(*c);
    f.n_tile=std::min(s,QK_CHANNEL_TILE);
    rk_npu_matmul_sizes sizes{};int rc=rk_npu_matmul_f16_batch_query(h,&f,&sizes);if(rc)return rc;
    // Each graph binds whole grouped QK row tiles. The GQA entrypoint splits
    // logical query rows internally when native FP16 needs another M tile.
    if(sizes.num_tasks!=h*ceil_div(s,QK_CHANNEL_TILE))return RK_NPU_ERR_PARAM;
    result.tasks=sizes.num_tasks;
    for(int k=0;k<2;++k){const int count=k?s%c->kv_tile:s/c->kv_tile;
        if(!count)continue;f.N=shape.value_dim+1;f.K=k?count:c->kv_tile;f.n_tile=0;
        rc=rk_npu_matmul_f16_batch_query(k?h:h*count,&f,&sizes);if(rc)return rc;result.tasks+=sizes.num_tasks;
    }
    if(result.tasks>4095)return RK_NPU_ERR_PARAM;out=result;return 0;
}

int query_gqa(const rk_npu_attention_f16_config& cfg,int capacity,int rows,int length,
    rk_npu_attention_f16_sizes& out,int& block_rows,bool pipeline){
    if(rows<1 || rows>cfg.max_query_rows)return RK_NPU_ERR_PARAM;
    const auto shape=gqa_shape(cfg);
    const bool wide=cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX &&
        cfg.mask_mode==RK_NPU_ATTENTION_F16_NO_MASK && shape.group==1;
    // MHA has no grouped-query expansion. Keep larger native query tiles
    // when both CBUF geometry and the LUT descriptor budget allow them.
    block_rows=std::min(rows,wide?(pipeline?64:128):32);
    int rc=query_engine(cfg,shape,capacity,block_rows,length,out);
    // Include optional per-core LUT setup tasks even in pingpong mode, so
    // changing submit policy cannot turn a wider tile into CPU exp.
    while(block_rows>1 && (rc || (wide && block_rows>32 &&
          !lut_tasks_fit(cfg,out.physical_rows,out.compute_length,out.tasks,true)))){
        block_rows=(block_rows+1)/2;
        rc=query_engine(cfg,shape,capacity,block_rows,length,out);
    }
    if(rc)return rc;
    if(cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL && rows>1){
        // Numerical-mode splits can use an earlier, larger partial tail even
        // when the final public prefix ends exactly at a full PV tile.
        const int first=align_up(std::max(1,length-rows+1),32);
        const int tail=first/cfg.kv_tile==out.compute_length/cfg.kv_tile?
            out.compute_length%cfg.kv_tile:cfg.kv_tile-32;
        out.tail_bytes=uint64_t(cfg.kv_heads)*(cfg.head_dim+32)*tail*2;
    }
    if(pipeline && cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX && rows>block_rows){
        out.query_bytes*=2;out.score_bytes*=2;out.mask_bytes*=2;
        out.partial_bytes*=2;out.tail_bytes*=2;
    }
    out.output_bytes=uint64_t(cfg.query_heads)*rows*cfg.head_dim*sizeof(float);
    out.tasks*=ceil_div(rows,block_rows);
    return RK_NPU_OK;
}
void add_timings(rk_npu_attention_f16_timings& sum,const rk_npu_attention_f16_timings& part){
    for(auto field:{&rk_npu_attention_f16_timings::reserve_us,&rk_npu_attention_f16_timings::kv_update_us,
        &rk_npu_attention_f16_timings::prepare_us,&rk_npu_attention_f16_timings::tail_stage_us,
        &rk_npu_attention_f16_timings::mask_update_us,&rk_npu_attention_f16_timings::query_pack_us,
        &rk_npu_attention_f16_timings::submit_us,&rk_npu_attention_f16_timings::hardware_us,
        &rk_npu_attention_f16_timings::output_sync_us,&rk_npu_attention_f16_timings::finish_us})sum.*field+=part.*field;
    sum.compute_length=std::max(sum.compute_length,part.compute_length);sum.tasks+=part.tasks;
}
int Workspace::causal_rows(const Cache& cache,int rows,int start)const{
    if(rows==1 || cfg.flags!=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX || cfg.mask_mode!=RK_NPU_ATTENTION_F16_CAUSAL)
        return rows;
    struct Mode {
        bool lut=false,long_sequence=false;
        uint32_t delayed=0,half_probability=0;
        bool operator==(const Mode& b)const{
            return lut==b.lut && long_sequence==b.long_sequence && delayed==b.delayed && half_probability==b.half_probability;
        }
    };
    const char* policy=std::getenv("RK_NPU_ATTN_LUT_SUBMIT");
    const bool split=policy && !std::strcmp(policy,"split");
    auto mode=[&](int count,int prefix,const uint16_t* bounds){
        rk_npu_attention_f16_sizes sizes{};
        check(query_engine(cfg,shape,cache.capacity,count,align_up(prefix,32),sizes));
        Mode result;
        result.lut=cache.use_lut(bounds) && lut_tasks_fit(cfg,sizes.physical_rows,sizes.compute_length,sizes.tasks,split);
        if(result.lut)return result;
        result.long_sequence=sizes.compute_length>1024;
        for(int hi=0;hi<cache.h();++hi){
            const float bound=widen(bounds[hi]);
            if(result.long_sequence && bound<=32752.f/cfg.kv_tile)result.delayed|=1u<<hi;
            if(bound>32752.f)result.half_probability|=1u<<hi;
        }
        return result;
    };
    std::vector<std::array<uint16_t,32>> bounds(rows);
    std::vector<Mode> ar(rows);
    std::array<uint16_t,32> previous{};
    cache.softmax_bounds(start,previous.data());
    for(int r=0;r<rows;++r){
        bounds[r]=previous;
        cache.extend_softmax_bounds(start+r,1,bounds[r].data());
        ar[r]=mode(1,start+r+1,bounds[r].data());
        previous=bounds[r];
    }
    // Batch only rows that keep the same exp/FP16-normalization algorithm as
    // their individual AR prefixes. A batch crossing 1024 or a V/descriptor
    // threshold is split here, without exposing native policy to callers.
    for(int count=rows;count>1;--count){
        const auto batch=mode(count,start+count,bounds[count-1].data());
        if(std::all_of(ar.begin(),ar.begin()+count,[&](const Mode& one){return one==batch;}))return count;
    }
    return 1;
}
void Workspace::prepare_gqa(Cache& cache,int rows,int start){
    compatible(*this,cache);if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    const int length=prefix_length(cfg,cache,rows,start);
    int block_rows;rk_npu_attention_f16_sizes sizes{};
    check(query_gqa(cfg,cache.capacity,rows,length,sizes,block_rows,pipeline_enabled));
    if(pipeline_enabled && cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX && rows>block_rows){
        prepare_pipeline(cache,rows,start,block_rows);return;
    }
    for(int first=0;first<rows;){
        const int count=causal_rows(cache,std::min(block_rows,rows-first),start+first);
        rk_npu_attention_f16_timings part{};geometry(cache,count,start+first,part,length);first+=count;
    }
}
rk_npu_attention_f16_timings Workspace::run_gqa(Cache& cache,const uint16_t* query,int rows,int start,void* output,
    const rk_npu_attention_f16_boolean_mask* mask,bool half_output){
    if(rows==1)return run(cache,query,rows,start,output,mask,0,0,half_output);
    if(!query || !output)throw Error{RK_NPU_ERR_PARAM};
    compatible(*this,cache);if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    validate_mask(cfg,mask,rows,cache.length);const auto begin=Clock::now();
    const int length=prefix_length(cfg,cache,rows,start);int block_rows;rk_npu_attention_f16_sizes sizes{};
    check(query_gqa(cfg,cache.capacity,rows,length,sizes,block_rows,pipeline_enabled));
    if(pipeline_enabled && cfg.flags==RK_NPU_ATTENTION_F16_STABLE_SOFTMAX && rows>block_rows){
        const auto planned=Clock::now();
        auto t=run_pipeline(cache,query,rows,start,output,mask,half_output,block_rows);
        t.prepare_us+=elapsed(begin,planned);
        t.total_us=elapsed(begin,Clock::now());return t;
    }
    const int first_count=causal_rows(cache,block_rows,start);
    const double planning_us=elapsed(begin,Clock::now());
    if(first_count==rows){auto t=run(cache,query,rows,start,output,mask,0,0,half_output);
        t.prepare_us+=planning_us;t.total_us=elapsed(begin,Clock::now());return t;}
    rk_npu_attention_f16_timings total{};
    total.prepare_us=planning_us;
    for(int first=0;first<rows;){
        const auto planning=Clock::now();
        const int count=first?causal_rows(cache,std::min(block_rows,rows-first),start+first):first_count;
        total.prepare_us+=elapsed(planning,Clock::now());
        auto view=mask?*mask:rk_npu_attention_f16_boolean_mask{};
        if(mask && view.query_rows!=1)view.query_offset+=first;
        add_timings(total,run(cache,query,count,start+first,output,mask?&view:nullptr,rows,first,half_output));
        first+=count;
    }
    total.total_us=elapsed(begin,Clock::now());return total;
}

template<class F>int api(F&& f)noexcept{try{f();return RK_NPU_OK;}catch(Error e){return e.code;}catch(const std::bad_alloc&){return RK_NPU_ERR_NOMEM;}catch(...){return RK_NPU_ERR_PARAM;}}
} // namespace

using namespace rknpu2_matmul_open::attention;
extern "C" {
void rk_npu_attention_f16_config_init(rk_npu_attention_f16_config* c){
    if(!c)return;*c={32,8,128,32,512,128,32768,1,2000,0,4.f,RK_NPU_ATTENTION_F16_CAUSAL,0.f};
}
void rk_npu_attention_f16_boolean_mask_init(rk_npu_attention_f16_boolean_mask* b,const uint8_t* data,uint64_t bytes,int h,int q,int k){
    if(!b)return;*b={};b->data=data;b->bytes=bytes;b->heads=h;b->query_rows=q;b->key_length=k;
    if(q>0 && k>0){b->head_stride=uint64_t(q)*k;b->query_stride=k;b->key_stride=1;}
}
int rk_npu_attention_f16_query(const rk_npu_attention_f16_config* c,int capacity,int rows,int length,rk_npu_attention_f16_sizes* out){
    if(!c || !out || validate(*c))return RK_NPU_ERR_PARAM;
    int block_rows;return query_gqa(*c,capacity,rows,length,*out,block_rows,pipeline_policy());
}
int rk_npu_attention_f16_cache_create(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config* c,rk_npu_attention_f16_cache** out){
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;if(!d || !c || validate(*c))return RK_NPU_ERR_PARAM;
    return api([&]{*out=new rk_npu_attention_f16_cache(d,*c,gqa_shape(*c));});
}
void rk_npu_attention_f16_cache_free(rk_npu_attention_f16_cache* c){delete c;}
int rk_npu_attention_f16_cache_get_info(rk_npu_attention_f16_cache* c,rk_npu_attention_f16_cache_info* out){
    if(!c || !out)return RK_NPU_ERR_PARAM;return api([&]{Lock lock(c->busy);*out={c->length,c->capacity,c->cfg.max_capacity,c->key.mem.size,c->value.mem.size,c->tail.mem.size,c->generation};});
}
int rk_npu_attention_f16_cache_reserve(rk_npu_ctx* ctx,rk_npu_attention_f16_cache* c,int capacity){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->lease.domain);Lock lock(c->busy);c->reserve(capacity);});
}
int rk_npu_attention_f16_cache_load(rk_npu_ctx* ctx,rk_npu_attention_f16_cache* c,const uint16_t* k,const uint16_t* v,int length){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->lease.domain);Lock lock(c->busy);c->load(k,v,length);});
}
int rk_npu_attention_f16_cache_append(rk_npu_ctx* ctx,rk_npu_attention_f16_cache* c,const uint16_t* k,const uint16_t* v,int count){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,c->lease.domain);Lock lock(c->busy);c->append(k,v,count);});
}
int rk_npu_attention_f16_cache_truncate(rk_npu_attention_f16_cache* c,int length){
    if(!c)return RK_NPU_ERR_PARAM;return api([&]{Lock lock(c->busy);c->truncate(length);});
}
int rk_npu_attention_f16_prepare(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config* c,rk_npu_attention_f16_workspace** out){
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;if(!d || !c || validate(*c))return RK_NPU_ERR_PARAM;
    return api([&]{*out=new rk_npu_attention_f16_workspace(d,*c,gqa_shape(*c));});
}
void rk_npu_attention_f16_workspace_free(rk_npu_attention_f16_workspace* w){delete w;}
int rk_npu_attention_f16_prepare_query(rk_npu_ctx* ctx,rk_npu_attention_f16_workspace* w,rk_npu_attention_f16_cache* c,int rows,int start){
    if(!w || !c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,w->lease.domain);Lock a(w->busy),b(c->busy);w->prepare_gqa(*c,rows,start);});
}
int rk_npu_attention_f16_run(rk_npu_ctx* ctx,rk_npu_attention_f16_workspace* w,rk_npu_attention_f16_cache* c,
    const uint16_t* q,int rows,int start,float* output,rk_npu_attention_f16_timings* timings,const rk_npu_attention_f16_boolean_mask* mask){
    if(!w || !c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,w->lease.domain);Lock a(w->busy),b(c->busy);
        const auto t=w->run_gqa(*c,q,rows,start,output,mask);if(timings)*timings=t;});
}
int rk_npu_attention_f16_run_half(rk_npu_ctx* ctx,rk_npu_attention_f16_workspace* w,rk_npu_attention_f16_cache* c,
    const uint16_t* q,int rows,int start,uint16_t* output,rk_npu_attention_f16_timings* timings,const rk_npu_attention_f16_boolean_mask* mask){
    if(!w || !c)return RK_NPU_ERR_PARAM;return api([&]{context(ctx,w->lease.domain);Lock a(w->busy),b(c->busy);
        const auto t=w->run_gqa(*c,q,rows,start,output,mask,true);if(timings)*timings=t;});
}
int rk_npu_attention_f16_decode_step(rk_npu_ctx* ctx,rk_npu_attention_f16_workspace* w,rk_npu_attention_f16_cache* c,
    const uint16_t* q,const uint16_t* k,const uint16_t* v,float* output,rk_npu_attention_f16_timings* timings,const rk_npu_attention_f16_boolean_mask* mask){
    if(!w || !c || !q || !k || !v || !output)return RK_NPU_ERR_PARAM;
    return api([&]{context(ctx,w->lease.domain);Lock a(w->busy),b(c->busy);compatible(*w,*c);
        if(w->poisoned || c->poisoned)throw Error{RK_NPU_ERR_SUBMIT};
        if(c->length>=c->cfg.max_capacity)throw Error{RK_NPU_ERR_NOMEM};
        validate_mask(w->cfg,mask,1,c->length+1);const int start=c->length;const auto begin=Clock::now();double reserve_us=0;
        c->append(k,v,1,&reserve_us);const auto appended=Clock::now();
        try{auto t=w->run(*c,q,1,start,output,mask);t.reserve_us=reserve_us;t.kv_update_us=elapsed(begin,appended);
            t.total_us=elapsed(begin,Clock::now());if(timings)*timings=t;}catch(...){c->poisoned=true;throw;}
    });
}
} // extern C

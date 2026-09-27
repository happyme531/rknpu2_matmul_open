#include "rk_npu_moe_w8.h"
#include "rk_npu_matmul.h"
#include "rk_npu_internal.h"
#include "rk_npu_moe_regs.h"
#include "rk_npu_moe_w4_internal.h"
#include "rk_npu_i4_cpu.h"
#include "rk_npu_i4_regs.h"
#include "rk_npu_i4_bounds.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <vector>
#include <omp.h>
#include <sys/ioctl.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::moe {
constexpr int I=512; // The optional DPU recipe currently supports this width.
using Clock=std::chrono::steady_clock;
static double us(Clock::time_point begin) {
    return std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
}
struct Failure { int code; };
static void check(int code) { if(code)throw Failure{code}; }
struct Device {
    rk_npu_ctx* ctx;
    rk_npu_iommu_domain* domain;
    explicit Device(rk_npu_iommu_domain* source):ctx(source->ctx),
        domain(rk_npu_iommu_domain_create(ctx,source->id)) {
        if(!domain)throw Failure{RK_NPU_ERR_NOMEM};
    }
    ~Device(){rk_npu_iommu_domain_free(domain);}
};
struct Buffer {
    rk_npu_mem mem{};
    rk_npu_ctx* ctx=nullptr;
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(ctx)rk_npu_mem_free(ctx,&mem);}
    void alloc(Device& d,size_t bytes,uint32_t flags=RK_NPU_MEM_DATA_DEFAULT) {
        if(ctx)throw std::runtime_error("duplicate buffer alloc");
        check(rk_npu_mem_alloc(d.domain,bytes,flags,&mem));ctx=d.ctx;
    }
    rk_npu_mem view(size_t offset,size_t bytes) const {
        rk_npu_mem v{};check(rk_npu_mem_view(&mem,offset,bytes,&v));return v;
    }
    void sync(size_t bytes,rk_npu_sync_dir direction) {
        auto v=view(0,bytes);check(rk_npu_mem_sync(ctx,&v,direction));
    }
};
struct MiddleItem {
    int core,rows;
    uint64_t input,coeff,output;
    float scale;
    bool gate;
};
struct Commands {
    Buffer cmd,task;
    uint32_t start[3]{},count[3]{};
    size_t capacity=0,slot_qwords=0,copy_body_size=0,copy_slot_qwords=0;
    std::array<size_t,7> copy_patch{};
    std::array<uint64_t,7> copy_tags{};
    std::vector<size_t> previous_ends;
    struct MiddleTemplate {
        size_t body=0;
        std::array<size_t,5> offsets{};
        std::array<uint64_t,5> tags{};
    };
    std::array<MiddleTemplate,2> middle_templates;
    void alloc(Device& d,size_t max_tasks,size_t words=80) {
        cmd.alloc(d,max_tasks*words*8,RK_NPU_MEM_NON_CACHEABLE);
        task.alloc(d,max_tasks*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING);
        capacity=max_tasks;slot_qwords=words;
    }
    void build(const std::vector<rknpu2_matmul_open::detail::I8GroupItem>& items) {
        if(items.empty()){std::fill_n(start,3,0);std::fill_n(count,3,0);return;}
        check(rknpu2_matmul_open::detail::build_i8_group(items,&cmd.mem,&task.mem,start,count));
    }
    void build_i4(const std::vector<rknpu2_matmul_open::detail::I8GroupItem>& items,const Commands* i8=nullptr) {
        size_t index=0;
        std::memset(task.mem.vaddr,0,task.mem.size);
        for(int core=0;core<3;++core) {
            start[core]=uint32_t(index);
            std::vector<std::vector<uint64_t>> bodies;
            std::vector<int> bases;
            for(const auto& item:items)if(item.core==core) {
                if(index>=capacity)throw Failure{RK_NPU_ERR_NOMEM};
                bodies.emplace_back();const auto& c=*item.cfg;
                check(rknpu2_matmul_open::detail::make_i4_regs(c.M*2,c.K,c.N,item.input->dma_addr,
                    item.weight->dma_addr,item.output->dma_addr,bodies.back()));
                bases.push_back(int(index++*slot_qwords));
            }
            // Both INT4 and INT8 GEMMs enable CNA/CORE/DPU (13). Each body
            // programs its precision. Keep DPU-RDMA gather as a separate job.
            if(i8)for(uint32_t j=0;j<i8->count[core];++j) {
                if(index>=capacity)throw Failure{RK_NPU_ERR_NOMEM};
                const auto& t=static_cast<const rknpu_task*>(i8->task.mem.vaddr)[i8->start[core]+j];
                const auto* body=static_cast<const uint64_t*>(i8->cmd.mem.vaddr)+(t.regcmd_addr-i8->cmd.mem.dma_addr)/8;
                if(t.regcfg_amount+4>slot_qwords)throw Failure{RK_NPU_ERR_PARAM};
                bodies.emplace_back(body,body+t.regcfg_amount);bases.push_back(int(index++*slot_qwords));
            }
            count[core]=uint32_t(index)-start[core];
            if(count[core])rknpu2_matmul_open::detail::write_chain(static_cast<uint64_t*>(cmd.mem.vaddr),
                static_cast<rknpu_task*>(task.mem.vaddr)+start[core],cmd.mem.dma_addr,
                bodies,bases,{13,0,13});
        }
    }
    void prepare_copies() {
        if(copy_body_size)return;
        const auto body=copy_i8_body({0,0,32,1});
        copy_body_size=body.size();copy_slot_qwords=align_up(int(body.size())+4,2);
        if(capacity*copy_slot_qwords*8>cmd.mem.size || capacity*sizeof(rknpu_task)>task.mem.size)
            throw std::runtime_error("copy template arena overflow");
        const std::array<uint32_t,7> addresses{0x4020,0x4024,0x403c,0x4058,0x40c0,0x5014,0x5018};
        for(size_t i=0;i<addresses.size();++i) {
            auto found=std::find_if(body.begin(),body.end(),[&](uint64_t word){return (word&0xffff)==addresses[i];});
            if(found==body.end())throw std::runtime_error("missing copy template field");
            copy_patch[i]=size_t(found-body.begin());
            copy_tags[i]=*found&0xffff00000000ffffull;
        }
        auto* commands=static_cast<uint64_t*>(cmd.mem.vaddr);
        auto* descriptors=static_cast<rknpu_task*>(task.mem.vaddr);
        std::memset(descriptors,0,capacity*sizeof(rknpu_task));
        for(size_t i=0;i<capacity;++i) {
            auto* slot=commands+i*copy_slot_qwords;
            std::copy(body.begin(),body.end(),slot);
            slot[copy_body_size]=::E(T_PC_REG,R_PC_BASE_ADDRESS,uint32_t(cmd.mem.dma_addr+(i+1)*copy_slot_qwords*8));
            slot[copy_body_size+1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,ceil_div(int(copy_body_size),2)+1);
            slot[copy_body_size+2]=::E(T_VERSION,0,0);
            slot[copy_body_size+3]=::E(T_PC,R_OPERATION_ENABLE,24);
            if(copy_slot_qwords>copy_body_size+4)slot[copy_body_size+4]=0;
            auto& descriptor=descriptors[i];
            descriptor.regcmd_addr=cmd.mem.dma_addr+i*copy_slot_qwords*8;
            descriptor.regcfg_amount=uint32_t(copy_body_size);
            descriptor.enable_mask=24;descriptor.int_mask=0x300;descriptor.int_clear=0x1ffff;
        }
    }
    size_t build_gather(const std::vector<rknpu2_matmul_open::detail::I8GroupItem>& items,
                        const std::vector<std::vector<I8Copy>>& copies) {
        if(items.size()!=copies.size())throw std::runtime_error("copy/job count mismatch");
        prepare_copies();
        // A separate copy submission completes gathering before GEMM starts.
        // Each physical core owns disjoint outputs.
        auto* commands=static_cast<uint64_t*>(cmd.mem.vaddr);
        // Restore old core ends before rewriting this call's three boundaries.
        for(size_t end:previous_ends) {
            auto* tail=commands+end*copy_slot_qwords+copy_body_size;
            tail[0]=::E(T_PC_REG,R_PC_BASE_ADDRESS,uint32_t(cmd.mem.dma_addr+(end+1)*copy_slot_qwords*8));
            tail[1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,ceil_div(int(copy_body_size),2)+1);
        }
        previous_ends.clear();
        size_t total=0;
        for(int core=0;core<3;++core) {
            start[core]=uint32_t(total);
            for(size_t i=0;i<items.size();++i)if(items[i].core==core) {
                for(const auto& copy:copies[i]) {
                    validate_i8_copy(copy);
                    if(total>=capacity)throw std::runtime_error("copy count exceeds arena");
                    auto* slot=commands+total*copy_slot_qwords;
                    const uint32_t channels=uint32_t(copy.channels-1),stride=uint32_t(copy.destination_rows*16);
                    const std::array<uint32_t,7> values{uint32_t(copy.destination),stride,(channels<<16)|channels,
                        channels,stride,channels,uint32_t(copy.source)};
                    for(size_t field=0;field<copy_patch.size();++field) {
                        slot[copy_patch[field]]=copy_tags[field]|(uint64_t(values[field])<<16);
                    }
                    ++total;
                }
            }
            count[core]=uint32_t(total)-start[core];
            if(count[core]) {
                const size_t end=total-1;previous_ends.push_back(end);
                auto* tail=commands+end*copy_slot_qwords+copy_body_size;
                tail[0]=::E(T_NOP,0,0);tail[1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,0);
            }
        }
        return total;
    }
    void prepare_middle() {
        if(middle_templates[0].body)return;
        if(capacity%2)throw std::runtime_error("middle tasks require gate/up pairs");
        std::array<std::vector<uint64_t>,2> bodies;
        for(int kind=0;kind<2;++kind) {
            bodies[kind]=middle_body(1,I,0,0,0,1.f,kind==0);
            auto& t=middle_templates[kind];t.body=bodies[kind].size();
            if(t.body+4>slot_qwords)throw std::runtime_error("middle register slot overflow");
            const uint32_t addresses[]={0x4020,0x4048,0x5018,kind==0?0x502cu:0x5038u,0x504c};
            for(size_t f=0;f<t.offsets.size();++f) {
                auto it=std::find_if(bodies[kind].begin(),bodies[kind].end(),[&](uint64_t w){return (w&65535)==addresses[f];});
                if(it==bodies[kind].end())throw std::runtime_error("missing middle patch field");
                t.offsets[f]=size_t(it-bodies[kind].begin());t.tags[f]=*it&0xffff00000000ffffull;
            }
        }
        auto* commands=static_cast<uint64_t*>(cmd.mem.vaddr);
        auto* descriptors=static_cast<rknpu_task*>(task.mem.vaddr);
        for(size_t index=0;index<capacity;++index) {
            const auto& t=middle_templates[index%2];
            auto* slot=commands+index*slot_qwords;
            std::copy(bodies[index%2].begin(),bodies[index%2].end(),slot);
            slot[t.body]=::E(T_PC_REG,R_PC_BASE_ADDRESS,uint32_t(cmd.mem.dma_addr+(index+1)*slot_qwords*8));
            slot[t.body+1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,uint32_t(ceil_div(int(middle_templates[(index+1)%2].body),2)+1));
            slot[t.body+2]=::E(T_VERSION,0,0);slot[t.body+3]=::E(T_PC,R_OPERATION_ENABLE,24);
            auto& descriptor=descriptors[index];descriptor={};
            descriptor.regcmd_addr=cmd.mem.dma_addr+index*slot_qwords*8;
            descriptor.regcfg_amount=uint32_t(t.body);
            descriptor.enable_mask=24;descriptor.int_mask=0x300;descriptor.int_clear=0x1ffff;
        }
    }
    size_t build_middle(const std::vector<MiddleItem>& items,int cores) {
        if(items.size()>capacity)throw std::runtime_error("middle task arena overflow");
        prepare_middle();
        auto* commands=static_cast<uint64_t*>(cmd.mem.vaddr);
        for(size_t end:previous_ends) {
            auto* tail=commands+end*slot_qwords+middle_templates[end%2].body;
            tail[0]=::E(T_PC_REG,R_PC_BASE_ADDRESS,uint32_t(cmd.mem.dma_addr+(end+1)*slot_qwords*8));
            tail[1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,uint32_t(ceil_div(int(middle_templates[(end+1)%2].body),2)+1));
        }
        previous_ends.clear();
        size_t total=0;
        for(int core=0;core<cores;++core) {
            start[core]=uint32_t(total);
            for(const auto& item:items)if(item.core==core) {
                if(item.gate!=(total%2==0) || item.rows<1 || item.rows>128 ||
                   (item.input|item.coeff|item.output)>UINT32_MAX ||
                   ((item.input|item.coeff|item.output)&15))throw std::runtime_error("invalid middle item");
                const auto& t=middle_templates[total%2];
                const uint32_t values[]={uint32_t(item.output),uint32_t(half_bits(item.scale))<<16,
                    uint32_t(item.input),uint32_t(item.coeff),uint32_t(item.rows-1)*16};
                auto* slot=commands+total*slot_qwords;
                for(size_t f=0;f<t.offsets.size();++f)slot[t.offsets[f]]=t.tags[f]|(uint64_t(values[f])<<16);
                ++total;
            }
            count[core]=uint32_t(total)-start[core];
            if(count[core]) {
                const size_t end=total-1;previous_ends.push_back(end);
                auto* tail=commands+end*slot_qwords+middle_templates[end%2].body;
                tail[0]=::E(T_NOP,0,0);tail[1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,0);
            }
        }
        for(int core=cores;core<3;++core)start[core]=count[core]=0;
        return total;
    }
    void build_lut(int cores,uint64_t input,uint64_t output) {
        if(cores>int(capacity))throw std::runtime_error("LUT arena overflow");
        auto* commands=static_cast<uint64_t*>(cmd.mem.vaddr);
        auto* descriptors=static_cast<rknpu_task*>(task.mem.vaddr);
        for(int core=0;core<cores;++core) {
            auto body=middle_lut_load(input,output+size_t(core)*32);
            if(body.size()+4>slot_qwords)throw std::runtime_error("LUT register slot overflow");
            start[core]=uint32_t(core);count[core]=1;
            auto* slot=commands+size_t(core)*slot_qwords;
            std::copy(body.begin(),body.end(),slot);
            slot[body.size()]=::E(T_NOP,0,0);
            slot[body.size()+1]=::E(T_PC_REG,R_PC_REGISTER_AMOUNTS,0);
            slot[body.size()+2]=::E(T_VERSION,0,0);
            slot[body.size()+3]=::E(T_PC,R_OPERATION_ENABLE,24);
            auto& descriptor=descriptors[core];descriptor={};
            descriptor.regcmd_addr=cmd.mem.dma_addr+size_t(core)*slot_qwords*8;
            descriptor.regcfg_amount=uint32_t(body.size());
            descriptor.enable_mask=24;descriptor.int_mask=0x300;descriptor.int_clear=0x1ffff;
        }
        for(int core=cores;core<3;++core)start[core]=count[core]=0;
    }
    size_t run(Device& d,int cores,const char* stage="commands",uint32_t timeout=1000) {
        const int total=int(count[0]+count[1]+count[2]);
        if(!total)return 0;
        while(cores>1 && !count[cores-1])--cores;
        for(int core=0;core<cores;++core)if(!count[core])throw Failure{RK_NPU_ERR_PARAM};
        // Homogeneous copy or GEMM chain. Never issue a userspace reset.
        rknpu_submit request{};
        request.flags=RKNPU_JOB_PC|RKNPU_JOB_PINGPONG;
        request.timeout=timeout;request.task_number=uint32_t(total);
        request.task_obj_addr=task.mem.obj_addr;request.core_mask=(1u<<cores)-1;
        request.fence_fd=-1;request.iommu_domain_id=rk_npu_iommu_domain_id(d.domain);
        for(int core=0;core<cores;++core)
            request.subcore_task[cores==3?core+2:core]={start[core],count[core]};
        if(std::getenv("RK_NPU_MOE_TRACE"))std::cerr<<"submit "<<stage<<" cores="<<cores<<" tasks="<<total<<std::endl;
        if(ioctl(d.ctx->fd,IOCTL_SUBMIT,&request)<0)
            throw Failure{RK_NPU_ERR_SUBMIT};
        if(std::getenv("RK_NPU_MOE_TRACE"))std::cerr<<"done "<<stage<<std::endl;
        return size_t(total);
    }
};

static bool valid(const rk_npu_moe_w8_weight_config& c) {
    return c.hidden>=64 && c.hidden<=1536 && c.hidden%32==0 &&
        c.intermediate>=32 && c.intermediate<=512 && c.intermediate%32==0 &&
        c.routed_experts>=1 && c.routed_experts<=128 &&
        (c.shared_expert==0 || c.shared_expert==1);
}
static bool same(const rk_npu_moe_w8_weight_config& a,const rk_npu_moe_w8_weight_config& b) {
    return a.hidden==b.hidden && a.intermediate==b.intermediate &&
        a.routed_experts==b.routed_experts && a.shared_expert==b.shared_expert;
}
static bool valid(const rk_npu_moe_w8_config& c) {
    return c.struct_size==sizeof(c) && valid(c.weights) && c.max_rows>=1 && c.max_rows<=256 &&
        c.top_k>=1 && c.top_k<=std::min(c.weights.routed_experts,8) &&
        c.npu_cores>=1 && c.npu_cores<=3 && c.cpu_threads>=1 && c.cpu_threads<=4 &&
        (c.native_input==0 || c.native_input==1) && c.timeout_ms>=1 && c.timeout_ms<=60000 &&
        (c.middle==RK_NPU_MOE_MIDDLE_CPU ||
         (c.middle==RK_NPU_MOE_MIDDLE_NPU_LUT && c.weights.intermediate==512));
}
template<class F> static void workers(int count,const F& function) {
#pragma omp parallel for num_threads(count) schedule(static)
    for(int worker=0;worker<count;++worker)function(worker);
}
// No exception can cross an OpenMP region. Zero marks invalid/nonfinite input.
static float quantize(const float* input,int size,int8_t* output) {
    float maximum=0;
    for(int i=0;i<size;++i) {
        if(!std::isfinite(input[i]))return 0;
        maximum=std::max(maximum,std::abs(input[i]));
    }
    const float scale=maximum?maximum/127.f:1.f, inv=1.f/scale;
    if(!(scale>0) || !std::isfinite(inv))return 0;
    int i=0;
#if defined(__aarch64__)
    const auto inverse=vdupq_n_f32(inv);
    const auto lo=vdupq_n_s32(-127),hi=vdupq_n_s32(127);
    for(;i+16<=size;i+=16) {
        auto convert=[&](int off){return vmaxq_s32(lo,vminq_s32(hi,
            vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input+i+off),inverse))));};
        vst1q_s8(output+i,vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(convert(0)),vqmovn_s32(convert(4)))),
                                     vqmovn_s16(vcombine_s16(vqmovn_s32(convert(8)),vqmovn_s32(convert(12))))));
    }
#endif
    for(;i<size;++i) {
        const float x=std::max(-127.f,std::min(127.f,input[i]*inv));
        const float floor=std::floor(x),fraction=x-floor;
        int value=int(floor);
        if(fraction>.5f || (fraction==.5f && (value&1)))++value;
        output[i]=int8_t(value);
    }
    return scale;
}
// Native INT4 is high-nibble-first. Each K32 high/low row occupies 16 bytes.
static void split32(const int8_t* q,uint8_t* high,uint8_t* low) {
#if defined(__aarch64__)
    for(int j=0;j<32;j+=16) {
        const auto v=vreinterpretq_u8_s8(vld1q_s8(q+j));
        const auto h=vshrq_n_u8(v,4),l=veorq_u8(vandq_u8(v,vdupq_n_u8(15)),vdupq_n_u8(8));
        auto pack=[](uint8x16_t x){return vorr_u8(vshl_n_u8(vget_low_u8(vuzp1q_u8(x,x)),4),vget_low_u8(vuzp2q_u8(x,x)));};
        vst1_u8(high+j/2,pack(h));vst1_u8(low+j/2,pack(l));
    }
#else
    for(int j=0;j<32;j+=2) {
        const uint8_t a=uint8_t(q[j]),b=uint8_t(q[j+1]);
        high[j/2]=(a&240)|(b>>4);low[j/2]=uint8_t(((a&15)^8)<<4)|((b&15)^8);
    }
#endif
}
static void pack_w4(int k,int n,const int8_t* src,rk_npu_mem& dst,std::vector<int32_t>& correction) {
    rknpu2_matmul_open::detail::I4WeightBound bound;
    check(rknpu2_matmul_open::detail::bound_i4_weights(k,n,src,k,bound));
    correction.assign(n,0);
    for(int row=0;row<k;++row)for(int col=0;col<n;++col)correction[col]+=8*src[size_t(row)*n+col];
    check(rknpu2_matmul_open::detail::pack_i4_weights(k,n,0,k,k,n,src,static_cast<uint8_t*>(dst.vaddr)));
}
struct Weights {
    Device device;
    rk_npu_moe_w8_weight_config config;
    Buffer gate,down,coeff;
    std::vector<rk_npu_mem> gw,dw;
    std::vector<std::vector<float>> gs,ds;
    bool lut_scales=true,w4=false;
    std::vector<std::vector<int32_t>> gcor,dcor;
    Weights(rk_npu_iommu_domain* domain,const rk_npu_moe_w8_weight_config& c,
            const rk_npu_moe_w8_expert* experts,bool mixed=false):device(domain),config(c),w4(mixed) {
        const int H=c.hidden,N=c.intermediate,count=c.routed_experts+c.shared_expert;
        const size_t units=mixed?size_t(c.routed_experts)+2*c.shared_expert:size_t(count)*2;
        gate.alloc(device,units*H*N);down.alloc(device,units*N*H/2);
        gcor.resize(count);dcor.resize(count);
        gw.resize(count);dw.resize(count);gs.resize(count);ds.resize(count);
        if(N==512 && !mixed) {
            coeff.alloc(device,size_t(count)*6144);
            std::memset(coeff.mem.vaddr,0,coeff.mem.size);
        } else lut_scales=false;
        for(int e=0;e<count;++e) {
            const auto& x=experts[e];
            if(!x.gate_up || !x.down || !x.gate_up_scales || !x.down_scales)throw Failure{RK_NPU_ERR_PARAM};
            gs[e].assign(x.gate_up_scales,x.gate_up_scales+2*N);
            ds[e].assign(x.down_scales,x.down_scales+H);
            for(const auto* scales:{&gs[e],&ds[e]})for(float s:*scales)
                if(!(s>0) || !std::isfinite(s))throw Failure{RK_NPU_ERR_PARAM};
            const bool i4=mixed && e<c.routed_experts;
            const size_t offset=mixed?size_t(e):size_t(e)*2;
            gw[e]=gate.view(offset*H*N,size_t(H)*N*(i4?1:2));
            dw[e]=down.view(offset*N*H/2,size_t(N)*H/(i4?2:1));
            rk_npu_matmul_i8_config gc,dc;
            rk_npu_matmul_i8_config_init(&gc,1,2*N,H);rk_npu_matmul_i8_config_init(&dc,1,H,N);
            if(i4) {
                pack_w4(H,2*N,x.gate_up,gw[e],gcor[e]);pack_w4(N,H,x.down,dw[e],dcor[e]);
            } else {
                check(rknpu2_matmul_open::detail::pack_i8_b(&gc,x.gate_up,&gw[e]));check(rknpu2_matmul_open::detail::pack_i8_b(&dc,x.down,&dw[e]));
            }
            if(N==512 && !mixed) {
                auto* bytes=static_cast<uint8_t*>(coeff.mem.vaddr)+size_t(e)*6144;
                auto* g=reinterpret_cast<uint16_t*>(bytes);auto* u=reinterpret_cast<uint32_t*>(bytes+4096);
                for(int col=0;col<N;++col)try {
                    g[size_t(col/4)*8+col%4]=half_bits(gs[e][col]*middle_lut_index_scale);
                    u[col]=half_bits(gs[e][N+col]);
                } catch(const std::runtime_error&) {lut_scales=false;}
            }
        }
        gate.sync(gate.mem.size,RK_NPU_SYNC_TO_DEVICE);down.sync(down.mem.size,RK_NPU_SYNC_TO_DEVICE);
        if(N==512 && !mixed)coeff.sync(coeff.mem.size,RK_NPU_SYNC_TO_DEVICE);
    }
    uint64_t bytes() const{return gate.mem.size+down.mem.size+coeff.mem.size;}
};
struct Assignment {int token,slot;};
struct Job {
    int expert,rows,offset,core;
    bool w4=false;
    rk_npu_matmul_i8_config gcfg{},dcfg{};
    rk_npu_mem ga{},gc{},da{},dc{};
};
struct OutputRow {const int32_t* data;const float* scales;int rows,row;float scale;const int32_t* correction=nullptr;};
static int32_t integer_at(const int32_t* data,int rows,int row,int n,const int32_t* correction) {
    if(!correction)return data[size_t(n/4)*rows*4+row*4+n%4];
    const auto* p=reinterpret_cast<const int16_t*>(data)+size_t(n/8)*rows*16+row*16+n%8;
    return 16*int32_t(p[0])+p[8]+correction[n];
}
#if defined(__aarch64__)
static int32x4_t integer4(const int32_t* data,int rows,int row,int n,const int32_t* correction) {
    if(!correction)return vld1q_s32(data+size_t(n/4)*rows*4+row*4);
    const auto* p=reinterpret_cast<const int16_t*>(data)+size_t(n/8)*rows*16+row*16+n%8;
    return vaddw_s16(vmlal_n_s16(vld1q_s32(correction+n),vld1_s16(p),16),vld1_s16(p+8));
}
#endif
struct Workspace {
    Device device;
    rk_npu_moe_w8_config config;
    Buffer input,gate_a,gate_c,down_a,down_c,middle_gate,middle_up,lut_input,lut_output;
    Commands gather_cmd,gate_cmd,down_cmd,middle_cmd,lut_cmd,gate4_cmd,down4_cmd;
    Buffer input4;
    bool w4=false,mixed_chain=false;
    std::vector<float> input_scales,hidden_scales;
    std::vector<std::vector<Assignment>> groups;
    std::vector<Assignment> ordered;
    std::vector<Job> jobs;
    std::vector<OutputRow> output_rows;
    std::atomic_flag busy=ATOMIC_FLAG_INIT;
    Workspace(rk_npu_iommu_domain* domain,const rk_npu_moe_w8_config& c,bool mixed=false):device(domain),config(c),w4(mixed) {
        const int H=c.weights.hidden,N=c.weights.intermediate;
        const size_t max=size_t(c.max_rows)*(c.top_k+c.weights.shared_expert);
        input.alloc(device,size_t(c.max_rows)*H);
        gate_a.alloc(device,max*H);gate_c.alloc(device,max*2*N*4);
        down_a.alloc(device,max*N);down_c.alloc(device,max*H*4);
        gate_a.sync(gate_a.mem.size,RK_NPU_SYNC_TO_DEVICE);
        gather_cmd.alloc(device,max*(mixed?2:1));gather_cmd.prepare_copies();
        if(mixed) {
            const char* option=std::getenv("RK_NPU_MOE_W4_MIXED");
            if(option && std::string(option)!="0" && std::string(option)!="1")throw Failure{RK_NPU_ERR_PARAM};
            mixed_chain=!option || std::string(option)=="1";
            input4.alloc(device,size_t(c.max_rows+1)*H);
            gate4_cmd.alloc(device,max/32+512,rknpu2_matmul_open::detail::I4_COMMAND_WORDS);
            down4_cmd.alloc(device,max/32+512,rknpu2_matmul_open::detail::I4_COMMAND_WORDS);
        }
        gate_cmd.alloc(device,max/32+512);down_cmd.alloc(device,max/32+512);
        input_scales.resize(c.max_rows);hidden_scales.resize(max);output_rows.resize(max);
        groups.resize(c.weights.routed_experts+c.weights.shared_expert);
        for(auto& g:groups)g.reserve(max);
        ordered.reserve(max);jobs.reserve(max);
        if(c.middle==RK_NPU_MOE_MIDDLE_NPU_LUT) {
            middle_gate.alloc(device,max*N*4);middle_up.alloc(device,max*N*4);
            lut_input.alloc(device,32);lut_output.alloc(device,size_t(c.npu_cores)*32);
            std::memset(lut_input.mem.vaddr,0,32);lut_input.sync(32,RK_NPU_SYNC_TO_DEVICE);
            middle_cmd.alloc(device,max*2,72);middle_cmd.prepare_middle();
            lut_cmd.alloc(device,c.npu_cores,1200);
            lut_cmd.build_lut(c.npu_cores,lut_input.mem.dma_addr,lut_output.mem.dma_addr);
        }
    }
    uint64_t bytes() const {
        uint64_t sum=0;
        for(auto* b:{&input4,&input,&gate_a,&gate_c,&down_a,&down_c,&middle_gate,&middle_up,&lut_input,&lut_output})sum+=b->mem.size;
        for(auto* c:{&gather_cmd,&gate_cmd,&down_cmd,&middle_cmd,&lut_cmd,&gate4_cmd,&down4_cmd})sum+=c->cmd.mem.size+c->task.mem.size;
        return sum;
    }
    void run(const Weights& weights,int rows,const void* source,bool f32,const float* scales,
             const int32_t* ids,const float* route_weights,float* output,rk_npu_moe_w8_timings& t) {
        const auto begin=Clock::now();auto at=begin;
        const int H=config.weights.hidden,N=config.weights.intermediate,R=config.weights.routed_experts;
        const int top=config.top_k,slots=top+config.weights.shared_expert,cores=config.npu_cores,threads=config.cpu_threads;
        const bool native=config.native_input,npu=config.middle==RK_NPU_MOE_MIDDLE_NPU_LUT;
        if(w4!=weights.w4 || !same(config.weights,weights.config))throw Failure{RK_NPU_ERR_PARAM};
        if(device.ctx->id!=weights.device.ctx->id || device.domain->id!=weights.device.domain->id)
            throw Failure{RK_NPU_ERR_DOMAIN};
        if(npu && !weights.lut_scales)throw Failure{RK_NPU_ERR_PARAM};
        for(int i=0;i<rows*top;++i)if(ids[i]<0 || ids[i]>=R || !std::isfinite(route_weights[i]) || route_weights[i]<0)
            throw Failure{RK_NPU_ERR_PARAM};
        auto* q=static_cast<int8_t*>(input.mem.vaddr);
        if(f32) {
            workers(threads,[&](int w){for(int r=w;r<rows;r+=threads)
                input_scales[r]=quantize(static_cast<const float*>(source)+size_t(r)*H,H,q+size_t(r)*H);});
        } else {
            std::memcpy(q,source,size_t(rows)*H);std::copy(scales,scales+rows,input_scales.begin());
        }
        for(int r=0;r<rows;++r)if(!(input_scales[r]>0) || !std::isfinite(input_scales[r]))throw Failure{RK_NPU_ERR_PARAM};
        if(w4)workers(threads,[&](int worker){for(int r=worker;r<rows;r+=threads) {
            auto* dst=static_cast<uint8_t*>(input4.mem.vaddr)+size_t(r)*H;
            for(int k=0;k<H;k+=32)split32(q+size_t(r)*H+k,dst+k/2,dst+H/2+k/2);
            if(rows==1)for(int k=0;k<H;k+=32)split32(q+k,dst+H+k,dst+H+k+16);
        }});
        t.input_us=us(at);at=Clock::now();
        for(auto& g:groups)g.clear();
        for(int r=0;r<rows;++r) {
            for(int slot=0;slot<top;++slot)groups[ids[r*top+slot]].push_back({r,slot});
            if(config.weights.shared_expert)groups[R].push_back({r,top});
        }
        jobs.clear();ordered.clear();const int tile=w4?64:(native?84:128);
        for(int e=0;e<int(groups.size());++e)for(size_t first=0;first<groups[e].size();first+=tile) {
            Job j{};j.expert=e;j.rows=int(std::min<size_t>(tile,groups[e].size()-first));j.offset=int(ordered.size());
            ordered.insert(ordered.end(),groups[e].begin()+first,groups[e].begin()+first+j.rows);jobs.push_back(j);
        }
        std::stable_sort(jobs.begin(),jobs.end(),[](const Job& a,const Job& b){return a.rows>b.rows;});
        std::array<size_t,3> cost{};
        for(auto& j:jobs) {
            j.w4=w4 && j.expert<R;
            if(w4 && !j.w4 && !mixed_chain)j.core=0;
            else {j.core=int(std::min_element(cost.begin(),cost.begin()+cores)-cost.begin());cost[j.core]+=j.rows+32;}
            rk_npu_matmul_i8_config_init(&j.gcfg,j.rows,2*N,H);rk_npu_matmul_i8_config_init(&j.dcfg,j.rows,H,N);
            for(auto* c:{&j.gcfg,&j.dcfg}) {
                c->a_layout=native?RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16:RK_NPU_I8_A_LAYOUT_NORMAL;
                c->c_layout=RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;c->out_dtype=RK_NPU_I8_OUT_INT32;
            }
            if(npu)j.gcfg.out_dtype=RK_NPU_I8_OUT_FP32;
            j.ga=gate_a.view(size_t(j.offset)*H,size_t(j.rows)*H);j.gc=gate_c.view(size_t(j.offset)*2*N*4,size_t(j.rows)*2*N*4);
            j.da=down_a.view(size_t(j.offset)*N,size_t(j.rows)*N);j.dc=down_c.view(size_t(j.offset)*H*4,size_t(j.rows)*H*4);
            if(w4 && rows==1 && j.rows==1)j.ga=j.w4?input4.view(H,H):input.view(0,H);
        }
        t.schedule_us=us(at);at=Clock::now();
        std::vector<rknpu2_matmul_open::detail::I8GroupItem> gi,di,g4,d4,all;std::vector<std::vector<I8Copy>> copies(jobs.size());
        std::vector<MiddleItem> middle;middle.reserve(npu?size_t(rows)*slots*2:0);
        for(size_t index=0;index<jobs.size();++index) {
            auto& j=jobs[index];
            all.push_back({&j.gcfg,&j.ga,&weights.gw[j.expert],&j.gc,j.core});
            (j.w4?g4:gi).push_back(all.back());
            (j.w4?d4:di).push_back({&j.dcfg,&j.da,&weights.dw[j.expert],&j.dc,j.core});
            for(int row=0;row<j.rows;) {
                if(w4 && rows==1 && j.rows==1)break;
                const int token=ordered[j.offset+row].token;int run=1;
                if(j.w4) {
                    const auto src=input4.mem.dma_addr+size_t(token)*H,dst=j.ga.dma_addr+size_t(row)*32;
                    copies[index].push_back({src,dst,H/2,2*j.rows});
                    copies[index].push_back({src+H/2,dst+16,H/2,2*j.rows});
                    ++row;continue;
                }
                while(!native && run<5 && row+run<j.rows && ordered[j.offset+row+run].token==token+run)++run;
                copies[index].push_back({input.mem.dma_addr+size_t(token)*H,j.ga.dma_addr+size_t(row)*(native?16:H),run*H,native?j.rows:1});
                row+=run;
            }
            if(npu)for(int row=0;row<j.rows;++row) {
                const float scale=input_scales[ordered[j.offset+row].token];
                const uint64_t base=weights.coeff.mem.dma_addr+size_t(j.expert)*6144,src=j.gc.dma_addr+size_t(row)*16,dest=uint64_t(j.offset+row)*N*4;
                middle.push_back({j.core,j.rows,src,base,middle_gate.mem.dma_addr+dest,scale,true});
                middle.push_back({j.core,j.rows,src+size_t(N)*j.rows*4,base+4096,middle_up.mem.dma_addr+dest,scale,false});
            }
        }
        t.gather_tasks=uint32_t(gather_cmd.build_gather(all,copies));gate_cmd.build(gi);down_cmd.build(di);
        if(w4){gate4_cmd.build_i4(g4,mixed_chain?&gate_cmd:nullptr);down4_cmd.build_i4(d4,mixed_chain?&down_cmd:nullptr);}
        if(npu)t.middle_tasks=uint32_t(middle_cmd.build_middle(middle,cores));
        t.build_us=us(at);at=Clock::now();input.sync(size_t(rows)*H,RK_NPU_SYNC_TO_DEVICE);
        if(w4)input4.sync(size_t(rows+(rows==1))*H,RK_NPU_SYNC_TO_DEVICE);
        t.sync_us+=us(at);at=Clock::now();gather_cmd.run(device,cores,"gather",config.timeout_ms);
        t.gather_us=us(at);at=Clock::now();if(!mixed_chain)t.gemm_tasks+=uint32_t(gate_cmd.run(device,cores,"gate",config.timeout_ms));
        if(w4)t.gemm_tasks+=uint32_t(gate4_cmd.run(device,cores,"gate4",config.timeout_ms));
        t.gate_us=us(at);at=Clock::now();
        if(npu) {
            lut_cmd.run(device,cores,"lut",config.timeout_ms);t.lut_load_us=us(at);at=Clock::now();
            middle_cmd.run(device,cores,"middle",config.timeout_ms);t.middle_npu_us=us(at);at=Clock::now();
            middle_gate.sync(size_t(rows)*slots*N*4,RK_NPU_SYNC_FROM_DEVICE);middle_up.sync(size_t(rows)*slots*N*4,RK_NPU_SYNC_FROM_DEVICE);
        } else gate_c.sync(size_t(rows)*slots*2*N*4,RK_NPU_SYNC_FROM_DEVICE);
        t.sync_us+=us(at);at=Clock::now();
        workers(threads,[&](int worker){
            alignas(64) float gate[512],up[512],hidden[512];alignas(64) int8_t hidden_q[512];
            for(size_t index=worker;index<jobs.size();index+=threads) {
                const auto& j=jobs[index];const auto* acc=static_cast<const int32_t*>(j.gc.vaddr);
                auto* target=static_cast<int8_t*>(j.da.vaddr);
                for(int row=0;row<j.rows;++row) {
                    const int offset=j.offset+row;const float sa=input_scales[ordered[offset].token];
                    if(npu) {
                        const auto* g=static_cast<const float*>(middle_gate.mem.vaddr)+size_t(offset)*N;
                        const auto* u=static_cast<const float*>(middle_up.mem.vaddr)+size_t(offset)*N;
                        for(int n=0;n<N;++n)hidden[n]=g[n]*u[n];
                    } else {
                        for(int n=0;n<N;n+=4) {
                            const auto* corr=j.w4?weights.gcor[j.expert].data():nullptr;
#if defined(__aarch64__)
                            vst1q_f32(gate+n,vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(integer4(acc,j.rows,row,n,corr)),sa),vld1q_f32(weights.gs[j.expert].data()+n)));
                            vst1q_f32(up+n,vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(integer4(acc,j.rows,row,n+N,corr)),sa),vld1q_f32(weights.gs[j.expert].data()+N+n)));
#else
                            for(int lane=0;lane<4;++lane) {gate[n+lane]=(float(integer_at(acc,j.rows,row,n+lane,corr))*sa)*weights.gs[j.expert][n+lane];up[n+lane]=(float(integer_at(acc,j.rows,row,n+N+lane,corr))*sa)*weights.gs[j.expert][N+n+lane];}
#endif
                        }
                        for(int n=0;n<N;++n)hidden[n]=(gate[n]/(1.f+std::exp(-gate[n])))*up[n];
                    }
                    float scale=quantize(hidden,N,hidden_q);
                    if(npu && scale>0) {
                        bool zero=scale==1.f && std::all_of(hidden,hidden+N,[](float x){return x==0.f;});
                        scale=zero?1.f:scale/middle_lut_scale;
                    }
                    hidden_scales[offset]=scale;
                    if(!scale)continue;
                    if(j.w4)for(int k=0;k<N;k+=32) {
                        auto* high=reinterpret_cast<uint8_t*>(target)+size_t(k/32)*j.rows*32+row*32;
                        split32(hidden_q+k,high,high+16);
                    }
                    else if(!native)std::memcpy(target+size_t(row)*N,hidden_q,N);
                    else for(int k=0;k<N;k+=16)std::memcpy(target+size_t(k/16)*j.rows*16+row*16,hidden_q+k,16);
                    const auto a=ordered[offset];
                    output_rows[size_t(a.token)*slots+a.slot]={static_cast<const int32_t*>(j.dc.vaddr),weights.ds[j.expert].data(),j.rows,row,scale,j.w4?weights.dcor[j.expert].data():nullptr};
                }
            }
        });
        for(int i=0;i<rows*slots;++i)if(!(hidden_scales[i]>0) || !std::isfinite(hidden_scales[i]))throw Failure{RK_NPU_ERR_PARAM};
        t.middle_cpu_us=us(at);at=Clock::now();down_a.sync(size_t(rows)*slots*N,RK_NPU_SYNC_TO_DEVICE);
        t.sync_us+=us(at);at=Clock::now();if(!mixed_chain)t.gemm_tasks+=uint32_t(down_cmd.run(device,cores,"down",config.timeout_ms));
        if(w4)t.gemm_tasks+=uint32_t(down4_cmd.run(device,cores,"down4",config.timeout_ms));
        t.down_us=us(at);at=Clock::now();down_c.sync(size_t(rows)*slots*H*4,RK_NPU_SYNC_FROM_DEVICE);
        t.sync_us+=us(at);at=Clock::now();
        workers(threads,[&](int worker){
            const int first=rows*worker/threads,last=rows*(worker+1)/threads;
            for(int t0=first;t0<last;t0+=4)for(int n0=0;n0<H;n0+=64)
            for(int token=t0;token<std::min(last,t0+4);++token)for(int n=n0;n<std::min(H,n0+64);n+=4) {
#if defined(__aarch64__)
                auto sum=vdupq_n_f32(0);
                for(int slot=0;slot<slots;++slot) {
                    const auto& r=output_rows[size_t(token)*slots+slot];
                    auto value=vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(integer4(r.data,r.rows,r.row,n,r.correction)),r.scale),vld1q_f32(r.scales+n));
                    sum=vfmaq_n_f32(sum,value,slot<top?route_weights[token*top+slot]:1.f);
                }
                vst1q_f32(output+size_t(token)*H+n,sum);
#else
                for(int lane=0;lane<4;++lane) {
                    float sum=0;
                    for(int slot=0;slot<slots;++slot) {
                        const auto& r=output_rows[size_t(token)*slots+slot];
                        const float value=(float(integer_at(r.data,r.rows,r.row,n+lane,r.correction))*r.scale)*r.scales[n+lane];
                        sum+=value*(slot<top?route_weights[token*top+slot]:1.f);
                    }
                    output[size_t(token)*H+n+lane]=sum;
                }
#endif
            }
        });
        t.combine_us=us(at);t.total_us=us(begin);
    }
};
} // namespace rknpu2_matmul_open::moe

struct rk_npu_moe_w8_weights:rknpu2_matmul_open::moe::Weights {using Weights::Weights;};
struct rk_npu_moe_w8_workspace:rknpu2_matmul_open::moe::Workspace {using Workspace::Workspace;};
extern "C" void rk_npu_moe_w8_config_init(rk_npu_moe_w8_config* c,int h,int i,int e,int shared,int top,int rows) {
    if(c)*c={sizeof(*c),{h,i,e,shared},rows,top,3,4,0,RK_NPU_MOE_MIDDLE_CPU,1000};
}
extern "C" rk_npu_moe_w8_weights* rk_npu_moe_w8_weights_create(rk_npu_iommu_domain* domain,
        const rk_npu_moe_w8_weight_config* config,const rk_npu_moe_w8_expert* experts) {
    if(!domain || !config || !experts || !rknpu2_matmul_open::moe::valid(*config))return nullptr;
    try{return new rk_npu_moe_w8_weights(domain,*config,experts);}catch(...){return nullptr;}
}
extern "C" void rk_npu_moe_w8_weights_free(rk_npu_moe_w8_weights* weights){delete weights;}
extern "C" rk_npu_moe_w8_workspace* rk_npu_moe_w8_workspace_create(rk_npu_iommu_domain* domain,
        const rk_npu_moe_w8_config* config) {
    if(!domain || !config || !rknpu2_matmul_open::moe::valid(*config))return nullptr;
    try{return new rk_npu_moe_w8_workspace(domain,*config);}catch(...){return nullptr;}
}
extern "C" void rk_npu_moe_w8_workspace_free(rk_npu_moe_w8_workspace* workspace){delete workspace;}
extern "C" uint64_t rk_npu_moe_w8_weights_device_bytes(const rk_npu_moe_w8_weights* w){return w?w->bytes():0;}
extern "C" uint64_t rk_npu_moe_w8_workspace_device_bytes(const rk_npu_moe_w8_workspace* w){return w?w->bytes():0;}

static bool overlap(const void* a,size_t na,const void* b,size_t nb) {
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
static int moe_run(rk_npu_moe_w8_workspace* ws,const rk_npu_moe_w8_weights* weights,int rows,
    const void* input,bool f32,const float* scales,const int32_t* ids,const float* routes,float* output,
    rk_npu_moe_w8_timings* timings) {
    if(timings)*timings={};
    if(!ws || !weights || !input || !ids || !routes || !output || (!f32 && !scales) ||
       rows<1 || rows>ws->config.max_rows)return RK_NPU_ERR_PARAM;
    const size_t values=size_t(rows)*ws->config.weights.hidden,slots=size_t(rows)*ws->config.top_k;
    if(overlap(input,values*(f32?4:1),output,values*4) || overlap(ids,slots*4,output,values*4) ||
       overlap(routes,slots*4,output,values*4) || (!f32 && overlap(scales,size_t(rows)*4,output,values*4)))return RK_NPU_ERR_PARAM;
    if(ws->busy.test_and_set(std::memory_order_acquire))return RK_NPU_ERR_BUSY;
    struct Unlock {std::atomic_flag& flag;~Unlock(){flag.clear(std::memory_order_release);}} unlock{ws->busy};
    rk_npu_moe_w8_timings result{};
    try {ws->run(*weights,rows,input,f32,scales,ids,routes,output,result);if(timings)*timings=result;return RK_NPU_OK;}
    catch(const rknpu2_matmul_open::moe::Failure& e){return e.code;}
    catch(const std::bad_alloc&){return RK_NPU_ERR_NOMEM;}
    catch(...){return RK_NPU_ERR_PARAM;}
}
extern "C" int rk_npu_moe_w8_run_quantized(rk_npu_moe_w8_workspace* ws,const rk_npu_moe_w8_weights* w,
    int rows,const int8_t* input,const float* scales,const int32_t* ids,const float* routes,float* out,rk_npu_moe_w8_timings* t) {
    return moe_run(ws,w,rows,input,false,scales,ids,routes,out,t);
}
extern "C" int rk_npu_moe_w8_run_f32(rk_npu_moe_w8_workspace* ws,const rk_npu_moe_w8_weights* w,
    int rows,const float* input,const int32_t* ids,const float* routes,float* out,rk_npu_moe_w8_timings* t) {
    return moe_run(ws,w,rows,input,true,nullptr,ids,routes,out,t);
}

extern "C" rk_npu_moe_w8_weights* rk_npu_moe_w4_weights_create(rk_npu_iommu_domain* domain,
    const rk_npu_moe_w8_weight_config* c,const rk_npu_moe_w8_expert* experts) {
    if(!domain || !c || !experts || !rknpu2_matmul_open::moe::valid(*c) || c->hidden%64 || c->intermediate%64)return nullptr;
    try{return new rk_npu_moe_w8_weights(domain,*c,experts,true);}catch(...){return nullptr;}
}
extern "C" rk_npu_moe_w8_workspace* rk_npu_moe_w4_workspace_create(rk_npu_iommu_domain* domain,
    const rk_npu_moe_w8_config* c) {
    if(!domain || !c || !rknpu2_matmul_open::moe::valid(*c) || c->weights.hidden%64 || c->weights.intermediate%64 ||
       c->middle!=RK_NPU_MOE_MIDDLE_CPU || c->native_input)return nullptr;
    try{return new rk_npu_moe_w8_workspace(domain,*c,true);}catch(...){return nullptr;}
}

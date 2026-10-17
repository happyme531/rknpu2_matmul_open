#include "rk_npu_gdn_internal.h"
#include "rk_npu_float_backend.h"
namespace rknpu2_matmul_open::gdn {
void Buffer::init(rk_npu_iommu_domain* d,size_t bytes,uint32_t flags) {
    domain=d;
    check(rk_npu_mem_alloc(d,bytes,flags,&mem));
    std::memset(mem.vaddr,0,mem.size);
    sync(RK_NPU_SYNC_TO_DEVICE);
}
void Chain::build(rk_npu_iommu_domain* d,
    const std::vector<std::vector<std::vector<uint64_t>>>& bodies,
    const std::vector<std::vector<int>>& enables,uint32_t mask,uint32_t timeout) {
    domain=d; core_mask=mask; timeout_ms=timeout;
    std::vector<int> offsets[3];
    int words=0;
    for(int core=0;core<3;++core) {
        starts[core]=total; counts[core]=bodies[core].size(); total+=counts[core];
        for(const auto& body:bodies[core]) {
            offsets[core].push_back(words);
            words+=align_up(int(body.size())+4,2);
        }
    }
    commands.init(d,size_t(words)*8,RK_NPU_MEM_NON_CACHEABLE);
    tasks.init(d,size_t(total)*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING);
    auto* cmd=static_cast<uint64_t*>(commands.mem.vaddr);
    auto* task=static_cast<rknpu_task*>(tasks.mem.vaddr);
    for(int core=0;core<3;++core) {
        detail::write_chain(cmd,task+starts[core],commands.mem.dma_addr,
            bodies[core],offsets[core],{13,0,13});
        for(size_t i=0;i<bodies[core].size();++i) {
            cmd[offsets[core][i]+bodies[core][i].size()+3]=E(T_PC,8,enables[core][i]);
            task[starts[core]+i].enable_mask=enables[core][i];
            task[starts[core]+i].op_idx=enables[core][i]==24?1:0;
        }
    }
}
double Chain::run() {
    int64_t ns=0;
    check(detail::do_submit_multicore(domain->ctx->fd,tasks.mem.obj_addr,total,
        core_mask,starts,counts,domain->id,timeout_ms,&ns));
    return ns/1000.;
}
std::vector<uint64_t> gemm(int m,int n,int k,uint64_t a,uint64_t b,uint64_t c) {
    std::vector<uint64_t> regs;
    detail::float_native_regs(regs,m,n,k,a,b,c,detail::FloatPrecision::TF32);
    return regs;
}
void GroupedChain::build(rk_npu_iommu_domain * domain,
               const std::vector<std::vector<std::vector<uint64_t>>> & bodies,
               const std::vector<std::vector<int>> & enables, const int * front_count,
               const std::vector<StateGroup> * tails, uint32_t mask, uint32_t timeout) {
        chain.build(domain, bodies, enables, mask, timeout);
        auto * task = static_cast<rknpu_task *>(chain.tasks.mem.vaddr);
        prototypes.assign(task, task + chain.total);
        for (int core = 0; core < 3; ++core) {
            for (int i = 0; i < front_count[core]; ++i) front[core].push_back(chain.starts[core] + i);
            groups[core] = tails[core];
            for (auto & g : groups[core]) {
                g.add += chain.starts[core]; g.single += chain.starts[core];
                g.high += chain.starts[core]; g.low += chain.starts[core];
            }
            selected[core].reserve(front_count[core] + groups[core].size() * 2);
        }
    }
double GroupedChain::run(const std::vector<int> & scale_update) {
        auto * cmd = static_cast<uint64_t *>(chain.commands.mem.vaddr);
        auto * task = static_cast<rknpu_task *>(chain.tasks.mem.vaddr);
        int written = 0;
        for (int core = 0; core < 3; ++core) {
            auto & plan = selected[core]; plan.clear();
            plan.insert(plan.end(), front[core].begin(), front[core].end());
            for (const auto & g : groups[core]) {
                int mode = 0;
                for (int h = g.first; h < g.first + g.count; ++h) mode = std::max(mode, scale_update[h]);
                if (mode == 2) { plan.push_back(g.high); plan.push_back(g.low); }
                else plan.push_back(mode == 1 ? g.single : g.add);
            }
            chain.starts[core] = written; chain.counts[core] = plan.size();
            for (size_t i = 0; i < plan.size(); ++i) {
                const auto & entry = prototypes[plan[i]];
                task[written++] = entry;
                const size_t tail = (entry.regcmd_addr - chain.commands.mem.dma_addr) / 8 + entry.regcfg_amount;
                if (i + 1 < plan.size()) {
                    const auto & next = prototypes[plan[i + 1]];
                    cmd[tail] = E(T_PC_REG, R_PC_BASE_ADDRESS, uint32_t(next.regcmd_addr & 0xfffffff0));
                    cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, (next.regcfg_amount + 1) / 2 + 1);
                } else {
                    cmd[tail] = E(T_NOP, 0, 0);
                    cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, 0);
                }
            }
        }
        chain.total = written;
        return chain.run();
    }
std::vector<uint64_t> dpu(int rows,int channels,uint64_t a,uint64_t b,uint64_t c,
                        bool subtract,uint64_t coeff,bool low) {
    std::vector<uint64_t> v;
    auto w=[&](int r,uint32_t val){v.push_back(E(T_DPU,r,val));};
    auto rd=[&](int r,uint32_t val){v.push_back(E(T_RDMA,r,val));};
    w(0x4004,14);w(0x400c,0x1e5);w(0x4010,0xb4000005);w(0x4014,0);
    w(0x4020,c);w(0x4024,rows*16);w(0x4030,rows-1);w(0x4034,0);w(0x4038,0);
    w(0x403c,((channels-1)<<16)|(channels-1));
    w(0x4040,low?0x42:0x53);w(0x4044,0);w(0x4048,low?uint32_t(hb::float_to_half(0x1p-16f))<<16:0);w(0x404c,0);
    w(0x4050,2);w(0x4054,0);w(0x4058,channels-1);w(0x405c,rows-1);
    w(0x4060,coeff?0x42:0x53);w(0x4064,0);w(0x4068,coeff?1:0);w(0x406c,0);
    w(0x4070,b?(subtract?0x10c403c0:0x10c203c0):0x383);w(0x4074,0);w(0x4078,1);w(0x407c,0);
    w(0x4080,0);w(0x4084,1);w(0x4088,0);for(int r=0x4090;r<=0x40ac;r+=4)w(r,0);
    w(0x40c0,rows*16);w(0x40c4,0);
    rd(0x5004,14);rd(0x500c,rows-1);rd(0x5010,0);rd(0x5014,channels-1);rd(0x5018,a);
    rd(0x501c,0);rd(0x5020,0);rd(0x5028,coeff?8:0);rd(0x502c,coeff);
    rd(0x5034,b?0x4000000c:1);rd(0x5038,b);rd(0x5040,rows*16);rd(0x5044,0x2f8a1);
    rd(0x5048,0);rd(0x504c,0);rd(0x5064,0);rd(0x5068,0x01010101);rd(0x506c,0);
    return v;
}

} // namespace rknpu2_matmul_open::gdn

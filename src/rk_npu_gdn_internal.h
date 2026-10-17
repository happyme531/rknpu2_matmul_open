#pragma once
#include "rk_npu_gdn.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rknpu2_matmul_open::gdn {
constexpr int HK=16, HV=32, D=128, SS=D*D, STATE_GROUP=4;
using Clock=std::chrono::steady_clock;
namespace hb = bits;
struct Error { int code; };
inline void check(int rc) { if (rc) throw Error{rc}; }
inline double micros(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::micro>(b-a).count();
}
inline size_t si(int v,int k) { return (size_t(k/4)*D+v)*4+k%4; }
inline size_t bi(int k,int n,int K) { return ((size_t(n/16)*(K/16)+k/16)*16+n%16)*16+k%16; }
inline size_t native_a(int row,int col,int rows) { return (size_t(col/4)*rows+row)*4+col%4; }
struct Lease {
    rk_npu_iommu_domain* domain;
    explicit Lease(rk_npu_iommu_domain* d):domain(d) { detail::retain_domain(d); }
    ~Lease() { detail::release_domain(domain); }
};
struct Buffer {
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mem mem{};
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer() { if(mem.handle) rk_npu_mem_free(domain->ctx,&mem); }
    void init(rk_npu_iommu_domain*,size_t,uint32_t flags=RK_NPU_MEM_DATA_DEFAULT);
    float* ptr() { return static_cast<float*>(mem.vaddr); }
    uint64_t addr(size_t floats=0) const { return mem.dma_addr+floats*4; }
    void sync(rk_npu_sync_dir dir) { if(mem.flags & RK_NPU_MEM_F_CACHEABLE) check(rk_npu_mem_sync(domain->ctx,&mem,dir)); }
};
class Workers {
    std::mutex mutex;
    std::condition_variable ready;
    std::function<void(int)> job;
    unsigned epoch=0;
    bool stop=false;
    int count_;
    std::atomic<int> done{0};
    std::vector<std::thread> threads;
public:
    explicit Workers(int count);
    ~Workers();
    int count() const { return count_; }
    void run(std::function<void(int)> fn);
};
struct Chain {
    rk_npu_iommu_domain* domain=nullptr;
    Buffer commands,tasks;
    int total=0;
    uint32_t starts[3]{},counts[3]{};
    uint32_t core_mask=7,timeout_ms=6000;
    void build(rk_npu_iommu_domain*,const std::vector<std::vector<std::vector<uint64_t>>>&,
               const std::vector<std::vector<int>>&,uint32_t,uint32_t);
    double run();
};
struct StateGroup { int first,count,add,single,high,low; };
struct GroupedChain {
    Chain chain;
    std::vector<rknpu_task> prototypes;
    std::vector<int> front[3],selected[3];
    std::vector<StateGroup> groups[3];
    void build(rk_npu_iommu_domain*,const std::vector<std::vector<std::vector<uint64_t>>>&,
               const std::vector<std::vector<int>>&,const int*,const std::vector<StateGroup>*,uint32_t,uint32_t);
    double run(const std::vector<int>&);
};
std::vector<uint64_t> gemm(int,int,int,uint64_t,uint64_t,uint64_t);
std::vector<uint64_t> dpu(int,int,uint64_t,uint64_t,uint64_t,bool subtract=false,uint64_t coeff=0,bool low=false);
struct Inputs {
    const float *q,*k,*v,*g,*beta,*state;
};
struct Projected {
    const float *input,*weights,*history;
    float *values,*final_history;
};
struct Times {
    double total=0,gram_pack=0,gram=0,coefficients=0,predict=0,solve=0,update=0,handoff=0,driver=0;
    double pack_compute=0,coefficients_compute=0,solve_compute=0,collect_compute=0;
    double cpu_user=0,cpu_system=0,gram_driver=0,predict_driver=0,update_driver=0;
};
class Engine {
public:
    virtual ~Engine()=default;
    virtual Times run(const Inputs&,const Projected*,float*,float*)=0;
};
std::unique_ptr<Engine> prepare_engine(rk_npu_iommu_domain*,const rk_npu_gdn_config&,Workers&);
} // namespace rknpu2_matmul_open::gdn

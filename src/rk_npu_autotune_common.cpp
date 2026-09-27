#include "rk_npu_autotune_common.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

namespace rknpu2_matmul_open::tune {
double median(std::vector<double> v) {
    if(v.empty()) return 0;
    std::sort(v.begin(),v.end()); const size_t m=v.size()/2;
    return v.size()&1?v[m]:(v[m-1]+v[m])*0.5;
}
int measure(const Options& o,const std::function<int(Sample&)>& run,
            const std::function<bool()>& validate,Score& result) {
    if(o.warmup<0 || o.loops<1 || o.repeats<1) return RK_NPU_ERR_PARAM;
    for(int i=0;i<o.warmup;++i) { Sample s; const int rc=run(s); if(rc) return rc; }
    std::vector<double> total,input,npu,sync,output;
    for(int r=0;r<o.repeats;++r) {
        Sample sum;
        for(int l=0;l<o.loops;++l) {
            Sample s; const int rc=run(s); if(rc) return rc;
            if(!validate()) return RK_NPU_ERR_PARAM;
            sum.input_us+=s.input_us; sum.npu_us+=s.npu_us; sum.sync_us+=s.sync_us;
            sum.output_us+=s.output_us; sum.total_us+=s.total_us;
        }
        const double inv=1.0/o.loops;
        total.push_back(sum.total_us*inv);input.push_back(sum.input_us*inv);
        npu.push_back(sum.npu_us*inv);sync.push_back(sum.sync_us*inv);output.push_back(sum.output_us*inv);
    }
    Score s;
    s.input_us=median(input);s.npu_us=median(npu);s.sync_us=median(sync);
    s.output_us=median(output);s.total_us=median(total);
    const auto mm=std::minmax_element(total.begin(),total.end());
    s.jitter_pct=s.total_us?(*mm.second-*mm.first)*100.0/s.total_us:0;
    s.robust_us=s.total_us+(*mm.second-*mm.first)*0.5;
    result=s;return RK_NPU_OK;
}
uint64_t process_cpu_mask() {
    cpu_set_t set; CPU_ZERO(&set);
    if(::sched_getaffinity(0,sizeof(set),&set)!=0) return 1;
    uint64_t mask=0;
    for(int c=0;c<64 && c<CPU_SETSIZE;++c) if(CPU_ISSET(c,&set)) mask|=1ull<<c;
    return mask?mask:1;
}
std::vector<int> preferred_cpus(uint64_t allowed) {
    std::vector<int> cpus;
    for(int c=4;c<=7;++c) if(allowed&(1ull<<c)) cpus.push_back(c);
    for(int c=0;c<64;++c) if((c<4 || c>7) && (allowed&(1ull<<c))) cpus.push_back(c);
    return cpus;
}
uint64_t first_cpu_mask(const std::vector<int>& cpus,int count) {
    uint64_t mask=0;
    for(int i=0;i<count && i<int(cpus.size());++i) mask|=1ull<<cpus[size_t(i)];
    return mask;
}
uint32_t fixed_npu_mask(uint32_t allowed,int N,int alignment) {
    if(N<1 || alignment<1) return 0;
    const int groups=int((int64_t(N)+alignment-1)/alignment);
    for(uint32_t mask:{7u,3u,1u,2u,4u})
        if((mask&allowed)==mask && __builtin_popcount(mask)<=groups) return mask;
    return 0;
}
void pin_openmp_team(uint64_t mask,int threads) {
    const auto cpus=preferred_cpus(mask);
    omp_set_dynamic(0); omp_set_num_threads(threads);
#pragma omp parallel num_threads(threads)
    {
        const int rank=omp_get_thread_num();
        if(rank<int(cpus.size())) {
            cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[size_t(rank)],&set);
            (void)::pthread_setaffinity_np(::pthread_self(),sizeof(set),&set);
        }
    }
}
bool set_thread_cpu_mask(uint64_t mask) {
    cpu_set_t set;CPU_ZERO(&set);
    for(int c=0;c<64 && c<CPU_SETSIZE;++c) if(mask&(1ull<<c)) CPU_SET(c,&set);
    return mask && ::pthread_setaffinity_np(::pthread_self(),sizeof(set),&set)==0;
}
uint64_t double_bits(double d) { uint64_t b;static_assert(sizeof(b)==sizeof(d));std::memcpy(&b,&d,sizeof(b));return b; }
double bits_double(uint64_t b) { double d;std::memcpy(&d,&b,sizeof(d));return d; }
void Hash::bytes(const void* p,size_t n) {
    const auto* b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i) {value^=b[i];value*=1099511628211ull;}
}
void Hash::mix(uint64_t v) {
    for(int i=0;i<8;++i) {const uint8_t b=uint8_t(v>>(i*8));bytes(&b,1);}
}
} // namespace rknpu2_matmul_open::tune

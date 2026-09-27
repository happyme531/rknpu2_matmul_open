#include "rk_npu_w4a8_tune_internal.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace rknpu2_matmul_open::w4_tune {
Data::Data(const rk_npu_w4a8_autotune_config& cfg):M(cfg.M),K(cfg.K),N(cfg.N),
    half(cfg.activation_type==RK_NPU_W4A8_F16),a(size_t(M)*K),c(size_t(M)*N) {
    if(half) {ah.resize(a.size());ch.resize(c.size());}
    uint32_t state=0x57344138;
    for(int m=0;m<M;++m) for(int k=0;k<K;++k) {
        state=state*1664525u+1013904223u;
        // Exactly representable in both input dtypes. Zero rows exercise MSD
        // correction cancellation without making the entire decode fixture zero.
        const float v=M>=3 && m==0?0.0f:float(int((state>>16)%255)-127)*0.03125f;
        const size_t i=size_t(m)*K+k;a[i]=v;
        if(half) ah[i]=rknpu2_matmul_open::bits::float_to_half(v);
    }
}
int Data::run(rk_npu_w4a8_workspace* ws,const rk_npu_w4a8_weights* w,rknpu2_matmul_open::tune::Sample& s) {
    rk_npu_w4a8_timings t{};
    const int rc=half?rk_npu_w4a8_run_f16(ws,w,ah.data(),ch.data(),&t):rk_npu_w4a8_run_f32(ws,w,a.data(),c.data(),&t);
    s={t.activation_scan_us+t.quant_pack_us,t.submit_us,t.sync_us,t.reduce_us+t.dequant_us,t.total_us};
    return rc;
}
void Data::capture() {if(half) reference_h=ch;else reference=c;}
bool Data::same() const {return half?ch==reference_h:c==reference;}
bool Data::verify_reference(const int8_t* B,const float* scales) const {
    const bool full=int64_t(M)*K<=16000000/int64_t(N);
    std::vector<int> columns;
    if(full) for(int n=0;n<N;++n) columns.push_back(n);
    else for(int i=0;i<17;++i) columns.push_back(int(int64_t(i)*(N-1)/16));
    std::vector<int8_t> q(K);
    for(int m=0;m<M;++m) {
        float maximum=0;
        for(int k=0;k<K;++k) maximum=std::max(maximum,std::fabs(a[size_t(m)*K+k]));
        const float scale=maximum==0?1:std::max(maximum/127.f,FLT_MIN),inv=1.f/scale;
        for(int k=0;k<K;++k) q[k]=int8_t(std::clamp(std::round(a[size_t(m)*K+k]*inv),-127.f,127.f));
        for(int n:columns) {
            int64_t sum=0;
            for(int k=0;k<K;++k) sum+=int(q[k])*int(B[size_t(k)*N+n]);
            const float expected=(float(sum)*scale)*scales[n];const size_t i=size_t(m)*N+n;
            if(half?reference_h[i]!=rknpu2_matmul_open::bits::float_to_half(expected):reference[i]!=expected) return false;
        }
    }
    return true;
}
}

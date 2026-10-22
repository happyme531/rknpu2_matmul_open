#pragma once
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail::w4swiglu {
#if defined(__aarch64__)
// NOTE: Degree-7 FP32 exp with branch-free exponent saturation to [-80,80].
// Gates below -80 produce signed zero, truncating the tiny negative tail.
// There is no FP64/scalar fallback; exact expf rounding and nonfinite semantics
// are not preserved. Saturation prevents exponent-bit reconstruction wrapping.
// Four Horner chains keep activation and optional maxabs in result registers.
template <bool Collect> inline uint32x4_t apply16(const float* gate,const float* up,float* out) {
    const auto g0=vld1q_f32(gate),g1=vld1q_f32(gate+4);
    const auto g2=vld1q_f32(gate+8),g3=vld1q_f32(gate+12);
    auto reduce=[](float32x4_t x,float32x4_t& n) {
        x=vmaxq_f32(vdupq_n_f32(-80.f),vminq_f32(vdupq_n_f32(80.f),x));
        n=vrndmq_f32(vfmaq_n_f32(vdupq_n_f32(.5f),x,1.4426950408889634f));
        auto r=vfmsq_n_f32(x,n,.693145751953125f);
        return vfmsq_n_f32(r,n,1.428606765330187e-6f);
    };
    float32x4_t n0,n1,n2,n3;
    const auto r0=reduce(vnegq_f32(g0),n0),r1=reduce(vnegq_f32(g1),n1);
    const auto r2=reduce(vnegq_f32(g2),n2),r3=reduce(vnegq_f32(g3),n3);
    auto p0=vdupq_n_f32(1.f/5040.f),p1=p0,p2=p0,p3=p0;
    auto step=[&](float coefficient) {
        const auto c=vdupq_n_f32(coefficient);
        p0=vfmaq_f32(c,p0,r0);p1=vfmaq_f32(c,p1,r1);
        p2=vfmaq_f32(c,p2,r2);p3=vfmaq_f32(c,p3,r3);
    };
    step(1.f/720.f);step(1.f/120.f);step(1.f/24.f);step(1.f/6.f);
    step(.5f);step(1.f);step(1.f);
    auto finish=[](float32x4_t p,float32x4_t n) {
        const auto power=vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n),vdupq_n_s32(127)),23));
        return vaddq_f32(vdupq_n_f32(1.f),vmulq_f32(p,power));
    };
    const auto d0=finish(p0,n0),d1=finish(p1,n1),d2=finish(p2,n2),d3=finish(p3,n3);
    auto activate=[](float32x4_t g,float32x4_t d) {
        return vbslq_f32(vcltq_f32(g,vdupq_n_f32(-80.f)),vdupq_n_f32(-0.f),vdivq_f32(g,d));
    };
    const auto y0=activate(g0,d0),y1=activate(g1,d1),y2=activate(g2,d2),y3=activate(g3,d3);
    auto maximum=vdupq_n_u32(0);
    auto store=[&](float32x4_t y,const float* u,float* dst) {
        const auto result=vmulq_f32(y,vld1q_f32(u));
        vst1q_f32(dst,result);
        if constexpr(Collect)
            maximum=vmaxq_u32(maximum,vandq_u32(vreinterpretq_u32_f32(result),vdupq_n_u32(0x7fffffff)));
    };
    store(y0,up,out);store(y1,up+4,out+4);store(y2,up+8,out+8);store(y3,up+12,out+12);
    return maximum;
}
#endif
}

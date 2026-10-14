#pragma once
#include <cmath>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail::swiglu {
inline float scalar(float gate, float up) {
    return (gate / (1.f + std::exp(-gate))) * up;
}
#if defined(__aarch64__)
// Round a degree-12 FP64 exponential back to FP32 before SwiGLU arithmetic.
inline float64x2_t exp_f64x2(float64x2_t x) {
    const auto nf = vrndmq_f64(vfmaq_n_f64(vdupq_n_f64(.5),x,1.4426950408889634074));
    auto r = vfmsq_n_f64(x,nf,.6931471805599453094);
    r = vfmsq_n_f64(r,nf,2.3190468138462995584e-17);
    auto p = vdupq_n_f64(1./479001600.);
    p = vfmaq_f64(vdupq_n_f64(1./39916800.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./3628800.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./362880.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./40320.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./5040.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./720.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./120.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./24.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1./6.),p,r);
    p = vfmaq_f64(vdupq_n_f64(.5),p,r);
    p = vfmaq_f64(vdupq_n_f64(1.),p,r);
    p = vfmaq_f64(vdupq_n_f64(1.),p,r);
    const auto power = vreinterpretq_f64_s64(vshlq_n_s64(vaddq_s64(vcvtq_s64_f64(nf),vdupq_n_s64(1023)),52));
    return vmulq_f64(p,power);
}
inline float32x4_t exp4(float32x4_t x) {
    return vcombine_f32(vcvt_f32_f64(exp_f64x2(vcvt_f64_f32(vget_low_f32(x)))),
                        vcvt_f32_f64(exp_f64x2(vcvt_f64_f32(vget_high_f32(x)))));
}
inline float32x4_t apply4(float32x4_t gate, float32x4_t up) {
    if (vminvq_u32(vcleq_f32(vabsq_f32(gate), vdupq_n_f32(80.f)))) {
        const auto divisor = vaddq_f32(vdupq_n_f32(1.f), exp4(vnegq_f32(gate)));
        return vmulq_f32(vdivq_f32(gate, divisor), up);
    }
    float g[4], u[4], output[4];
    vst1q_f32(g, gate);
    vst1q_f32(u, up);
    for (int i = 0; i < 4; ++i) output[i] = scalar(g[i], u[i]);
    return vld1q_f32(output);
}
// Interleave four independent FP64 Horner chains with the same per-lane
// arithmetic as apply4. Each coefficient serves eight output elements.
__attribute__((always_inline)) inline void apply8(float32x4_t g0, float32x4_t u0,
                                                float32x4_t g1, float32x4_t u1, float* out) {
    const auto valid = vandq_u32(vcleq_f32(vabsq_f32(g0), vdupq_n_f32(80.f)),
                                vcleq_f32(vabsq_f32(g1), vdupq_n_f32(80.f)));
    if (!vminvq_u32(valid)) {
        vst1q_f32(out, apply4(g0, u0));
        vst1q_f32(out + 4, apply4(g1, u1));
        return;
    }
    const auto x0 = vnegq_f32(g0), x1 = vnegq_f32(g1);
    auto reduce = [](float64x2_t x, float64x2_t& n) {
        n = vrndmq_f64(vfmaq_n_f64(vdupq_n_f64(.5), x, 1.4426950408889634074));
        auto r = vfmsq_n_f64(x, n, .6931471805599453094);
        return vfmsq_n_f64(r, n, 2.3190468138462995584e-17);
    };
    float64x2_t n0, n1, n2, n3;
    const auto r0 = reduce(vcvt_f64_f32(vget_low_f32(x0)), n0);
    const auto r1 = reduce(vcvt_f64_f32(vget_high_f32(x0)), n1);
    const auto r2 = reduce(vcvt_f64_f32(vget_low_f32(x1)), n2);
    const auto r3 = reduce(vcvt_f64_f32(vget_high_f32(x1)), n3);
    auto p0 = vdupq_n_f64(1./479001600.), p1 = p0, p2 = p0, p3 = p0;
    auto step = [&](double coefficient) {
        const auto c = vdupq_n_f64(coefficient);
        p0 = vfmaq_f64(c, p0, r0);
        p1 = vfmaq_f64(c, p1, r1);
        p2 = vfmaq_f64(c, p2, r2);
        p3 = vfmaq_f64(c, p3, r3);
    };
    step(1./39916800.);
    step(1./3628800.);
    step(1./362880.);
    step(1./40320.);
    step(1./5040.);
    step(1./720.);
    step(1./120.);
    step(1./24.);
    step(1./6.);
    step(.5);
    step(1.);
    step(1.);
    auto finish = [](float64x2_t p, float64x2_t n) {
        const auto power = vreinterpretq_f64_s64(
            vshlq_n_s64(vaddq_s64(vcvtq_s64_f64(n), vdupq_n_s64(1023)), 52));
        return vcvt_f32_f64(vmulq_f64(p, power));
    };
    const auto e0 = vcombine_f32(finish(p0, n0), finish(p1, n1));
    const auto e1 = vcombine_f32(finish(p2, n2), finish(p3, n3));
    vst1q_f32(out, vmulq_f32(vdivq_f32(g0, vaddq_f32(vdupq_n_f32(1.f), e0)), u0));
    vst1q_f32(out + 4, vmulq_f32(vdivq_f32(g1, vaddq_f32(vdupq_n_f32(1.f), e1)), u1));
}
#endif
} // namespace rknpu2_matmul_open::detail::swiglu

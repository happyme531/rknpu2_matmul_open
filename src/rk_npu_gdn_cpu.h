#pragma once
#include "rk_npu_gdn_internal.h"
#include <arm_neon.h>
#include <cmath>
namespace rknpu2_matmul_open::gdn {
constexpr int KW=HK*D, VW=HV*D, CW=2*KW+VW;
inline float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ggml_v_expf, llama.cpp a17e94d229fc7fed1567a799baebd11b771fe5af.
// Copyright (c) 2023-2026 The ggml authors, MIT license.
// Full license: ../LICENSES/ggml-MIT.txt.
inline float32x4_t exp4(float32x4_t x) {
    const auto r = vdupq_n_f32(bits_float(0x4b400000));
    const auto z = vfmaq_n_f32(r, x, bits_float(0x3fb8aa3b));
    const auto n = vsubq_f32(z, r);
    const auto b = vfmsq_n_f32(vfmsq_n_f32(x, n, bits_float(0x3f317200)), n, bits_float(0x35bfbe8e));
    const auto e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const auto k = vreinterpretq_f32_u32(vaddq_u32(e, vdupq_n_u32(0x3f800000)));
    const auto u = vmulq_f32(b, b);
    const auto j = vfmaq_f32(vmulq_n_f32(b, bits_float(0x3f7ffff6)),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(bits_float(0x3efffedb)), vdupq_n_f32(bits_float(0x3e2aaf33)), b),
                  vfmaq_f32(vdupq_n_f32(bits_float(0x3d2b9f17)), vdupq_n_f32(bits_float(0x3c072010)), b), u), u);
    return vfmaq_f32(k, j, k);
}

inline float32x4_t silu4(float32x4_t x) {
    if (vminvq_u32(vcleq_f32(vabsq_f32(x), vdupq_n_f32(80.f))) == UINT32_MAX)
        return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.f), exp4(vnegq_f32(x))));
    alignas(16) float values[4];
    vst1q_f32(values, x);
    for (float & v : values) {
        const float e = std::exp(v >= 0 ? -v : v);
        v *= v >= 0 ? 1.f / (1.f + e) : e / (1.f + e);
    }
    return vld1q_f32(values);
}

inline void normalize(float * row) {
    float sum = 0;
    for (int i = 0; i < D; ++i) sum += row[i] * row[i];
    const float scale = 1.f / std::sqrt(sum + 1e-6f);
    for (int i = 0; i < D; i += 4) vst1q_f32(row + i, vmulq_n_f32(vld1q_f32(row + i), scale));
}

inline void transpose(float32x4_t* r) {
    auto a=vtrnq_f32(r[0],r[1]),b=vtrnq_f32(r[2],r[3]);
    r[0]=vcombine_f32(vget_low_f32(a.val[0]),vget_low_f32(b.val[0]));
    r[1]=vcombine_f32(vget_low_f32(a.val[1]),vget_low_f32(b.val[1]));
    r[2]=vcombine_f32(vget_high_f32(a.val[0]),vget_high_f32(b.val[0]));
    r[3]=vcombine_f32(vget_high_f32(a.val[1]),vget_high_f32(b.val[1]));
}
inline void conv_silu(const Projected& p,int t,int channel,int count,float* out) {
        for (int c = channel; c < channel + count; c += 4) {
            const auto weight = vld4q_f32(p.weights + c * 4);
            float32x4x3_t old{};
            if (t < 3) old = vld3q_f32(p.history + c * 3);
            auto value = vdupq_n_f32(0.f);
            for (int k = 0; k < 4; ++k) {
                const auto input = t + k < 3 ? old.val[t + k]
                    : vld1q_f32(p.input + size_t(t + k - 3) * CW + c);
                value = vfmaq_f32(value, input, weight.val[k]);
            }
            vst1q_f32(out + c - channel, silu4(value));
        }
    }

template<bool split>
inline void store_native(float* main,float* low,size_t at,float32x4_t value) {
    vst1q_f32(main+at,value);
    if constexpr(split) {
        const auto high=vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(value),vdupq_n_u32(0xffffe000)));
        vst1q_f32(low+at,vsubq_f32(value,high));
    }
}
template<bool split,int C>
void produce_native(Workers& workers,int T,const Projected& p,float* a,float* al,float* b,float* bl) {
    const int blocks=(T+C-1)/C;
    const float qscale=1.f/std::sqrt(float(D));
    workers.run([&](int w) {
        for(int job=w;job<blocks*HK;job+=workers.count()) {
            const int block=job/HK,head=job%HK,base=block*C,valid=std::min(C,T-base);
            const size_t ai=size_t(job)*C*D,bi0=size_t(job)*D*2*C;
            alignas(64) float query[D],key[D];
            for(int t=0;t<valid;++t) {
                conv_silu(p,base+t,head*D,D,query);
                conv_silu(p,base+t,KW+head*D,D,key);
                normalize(query); normalize(key);
                for(int k=0;k<D;k+=4) {
                    auto kv=vld1q_f32(key+k),qv=vmulq_n_f32(vld1q_f32(query+k),qscale);
                    store_native<split>(a,al,ai+native_a(t,k,C),kv);
                    store_native<split>(b,bl,bi0+bi(k,t,D),kv);
                    store_native<split>(b,bl,bi0+bi(k,C+t,D),qv);
                }
                conv_silu(p,base+t,2*KW+head*2*D,2*D,
                    p.values+size_t(base+t)*VW+head*2*D);
            }
        }
    });
    const int keep=std::min(T,3);
    workers.run([&](int w) {
        for(int c=w;c<CW;c+=workers.count()) {
            for(int j=0;j<3-keep;++j) p.final_history[c*3+j]=p.history[c*3+j+keep];
            for(int j=0;j<keep;++j) p.final_history[c*3+3-keep+j]=p.input[size_t(T-keep+j)*CW+c];
        }
    });
}
} // namespace rknpu2_matmul_open::gdn

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

inline float32x4_t silu_normal(float32x4_t x) {
    return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.f), exp4(vnegq_f32(x))));
}

inline float32x4_t silu4(float32x4_t x) {
    if (vminvq_u32(vcleq_f32(vabsq_f32(x), vdupq_n_f32(80.f))) == UINT32_MAX)
        return silu_normal(x);
    alignas(16) float values[4];
    vst1q_f32(values, x);
    for (float & v : values) {
        const float e = std::exp(v >= 0 ? -v : v);
        v *= v >= 0 ? 1.f / (1.f + e) : e / (1.f + e);
    }
    return vld1q_f32(values);
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
                    : vld1q_f32(p.input + size_t(t + k - 3) * p.stride + c);
                value = vfmaq_f32(value, input, weight.val[k]);
            }
            vst1q_f32(out + (c - channel), silu4(value));
        }
    }

// Groups start on a four-token boundary. Only the first group reads history.
// Each vector covers four channels. The four windows use seven input vectors.
inline void conv_silu_four(const Projected& p, int t, int channel, int count,
                           float* out, int out_stride) {
    for (int c = channel; c < channel + count; c += 4) {
        float32x4_t x[7];
        if (t == 0) {
            const auto history = vld3q_f32(p.history + c * 3);
            for (int i = 0; i < 3; ++i) x[i] = history.val[i];
        } else {
            for (int i = 0; i < 3; ++i)
                x[i] = vld1q_f32(p.input + size_t(t - 3 + i) * p.stride + c);
        }
        for (int i = 0; i < 4; ++i)
            x[3 + i] = vld1q_f32(p.input + size_t(t + i) * p.stride + c);
        const auto weight = vld4q_f32(p.weights + c * 4);
        float32x4_t y[4];
        auto normal = vdupq_n_u32(UINT32_MAX);
        for (int row = 0; row < 4; ++row) {
            y[row] = vdupq_n_f32(0.f);
            for (int tap = 0; tap < 4; ++tap)
                y[row] = vfmaq_f32(y[row], x[row + tap], weight.val[tap]);
            normal = vandq_u32(normal, vcleq_f32(vabsq_f32(y[row]), vdupq_n_f32(80.f)));
        }
        const bool fast = vminvq_u32(normal) == UINT32_MAX;
        // Keep the original per-vector SiLU fallback when any channel is out of range.
        for (int row = 0; row < 4; ++row)
            vst1q_f32(out + row * out_stride + (c - channel), fast ? silu_normal(y[row]) : silu4(y[row]));
    }
}

template<int Rows>
inline void conv_rows(const Projected& p, int t, int channel, int count,
                      float* out, int out_stride) {
    if constexpr (Rows == 4) conv_silu_four(p, t, channel, count, out, out_stride);
    else conv_silu(p, t, channel, count, out);
}

// Each lane accumulates one token in the original channel order.
template<int Rows>
inline void normalization_scales(const float* rows, float* scales) {
    if constexpr (Rows == 4) {
        auto sum = vdupq_n_f32(0.f);
        for (int k = 0; k < D; k += 4) {
            float32x4_t x[4];
            for (int row = 0; row < 4; ++row) x[row] = vld1q_f32(rows + row * D + k);
            transpose(x);
            for (auto v : x) sum = vaddq_f32(sum, vmulq_f32(v, v));
        }
        vst1q_f32(scales, vdivq_f32(vdupq_n_f32(1.f), vsqrtq_f32(vaddq_f32(sum, vdupq_n_f32(1e-6f)))));
    } else {
        float sum = 0;
        for (int k = 0; k < D; ++k) sum += rows[k] * rows[k];
        scales[0] = 1.f / std::sqrt(sum + 1e-6f);
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
template<bool split, int C, int Rows>
inline void produce_rows(const Projected& p, int token, int t, int head,
                         float* a, float* al, float* b, float* bl, size_t ai, size_t bi0) {
    const float qscale = 1.f / std::sqrt(float(D));
    alignas(64) float query[Rows * D], key[Rows * D];
    conv_rows<Rows>(p, token, head * D, D, query, D);
    conv_rows<Rows>(p, token, KW + head * D, D, key, D);
    float qs[Rows], ks[Rows];
    normalization_scales<Rows>(query, qs);
    normalization_scales<Rows>(key, ks);
    for (int row = 0; row < Rows; ++row) {
        for (int k = 0; k < D; k += 4) {
            const auto kv = vmulq_n_f32(vld1q_f32(key + row * D + k), ks[row]);
            const auto qv = vmulq_n_f32(vmulq_n_f32(vld1q_f32(query + row * D + k), qs[row]), qscale);
            store_native<split>(a, al, ai + native_a(t + row, k, C), kv);
            store_native<split>(b, bl, bi0 + bi(k, t + row, D), kv);
            store_native<split>(b, bl, bi0 + bi(k, C + t + row, D), qv);
        }
    }
    conv_rows<Rows>(p, token, 2 * KW + head * 2 * D, 2 * D,
                    p.values + size_t(token) * VW + head * 2 * D, VW);
}

template<bool split,int C>
void produce_native(Workers& workers,int T,const Projected& p,float* a,float* al,float* b,float* bl) {
    const int blocks=(T+C-1)/C;
    workers.run([&](int w) {
        for(int job=w;job<blocks*HK;job+=workers.count()) {
            const int block=job/HK,head=job%HK,base=block*C,valid=std::min(C,T-base);
            const size_t ai=size_t(job)*C*D,bi0=size_t(job)*D*2*C;
            int t = 0;
            for (; t + 4 <= valid; t += 4)
                produce_rows<split,C,4>(p, base + t, t, head, a, al, b, bl, ai, bi0);
            for (; t < valid; ++t)
                produce_rows<split,C,1>(p, base + t, t, head, a, al, b, bl, ai, bi0);
        }
    });
    const int keep=std::min(T,3);
    workers.run([&](int w) {
        for(int c=w;c<CW;c+=workers.count()) {
            for(int j=0;j<3-keep;++j) p.final_history[c*3+j]=p.history[c*3+j+keep];
            for(int j=0;j<keep;++j) p.final_history[c*3+3-keep+j]=p.input[size_t(T-keep+j)*p.stride+c];
        }
    });
}
} // namespace rknpu2_matmul_open::gdn

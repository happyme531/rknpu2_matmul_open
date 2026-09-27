#ifndef RK_NPU_W4A8_REDUCE_NEON_H
#define RK_NPU_W4A8_REDUCE_NEON_H
#include "rk_npu_half_bits.h"
#include <arm_neon.h>

namespace rknpu2_matmul_open::detail::w4reduce {
// One native token/N8 pair, without metadata aliasing or tail branches.
inline void add8(const int16_t *p, int32_t *d, int32x4_t a, int32x4_t b) {
    const auto h = vld1q_s16(p), l = vld1q_s16(p + 8);
    a = vaddw_s16(vmlal_n_s16(a, vget_low_s16(h), 16), vget_low_s16(l));
    b = vaddw_s16(vmlal_n_s16(b, vget_high_s16(h), 16), vget_high_s16(l));
    vst1q_s32(d, a);
    vst1q_s32(d + 4, b);
}
template <bool First>
inline void accumulate(const int16_t *__restrict p, int32_t *__restrict d,
                       const int32_t *correction, int rows, int groups) {
    if constexpr (First) {
        for (int nb = 0; nb < groups; ++nb) {
            const auto a = vld1q_s32(correction + nb * 8), b = vld1q_s32(correction + nb * 8 + 4);
            int r = 0;
            for (; r + 4 <= rows; r += 4, p += 64, d += 32) {
                add8(p, d, a, b);
                add8(p + 16, d + 8, a, b);
                add8(p + 32, d + 16, a, b);
                add8(p + 48, d + 24, a, b);
            }
            for (; r < rows; ++r, p += 16, d += 8)
                add8(p, d, a, b);
        }
    } else {
        const int pairs = rows * groups;
        int i = 0;
        for (; i + 4 <= pairs; i += 4, p += 64, d += 32) {
            add8(p, d, vld1q_s32(d), vld1q_s32(d + 4));
            add8(p + 16, d + 8, vld1q_s32(d + 8), vld1q_s32(d + 12));
            add8(p + 32, d + 16, vld1q_s32(d + 16), vld1q_s32(d + 20));
            add8(p + 48, d + 24, vld1q_s32(d + 24), vld1q_s32(d + 28));
        }
        for (; i < pairs; ++i, p += 16, d += 8)
            add8(p, d, vld1q_s32(d), vld1q_s32(d + 4));
    }
}
template <bool Half>
inline void finish8(const int16_t *p, const int32_t *a, float32x4_t scale, float32x4_t w0,
                    float32x4_t w1, void *output, size_t offset) {
    const auto h = vld1q_s16(p), l = vld1q_s16(p + 8);
    const auto x = vaddw_s16(vmlal_n_s16(vld1q_s32(a), vget_low_s16(h), 16), vget_low_s16(l));
    const auto y = vaddw_s16(vmlal_n_s16(vld1q_s32(a + 4), vget_high_s16(h), 16), vget_high_s16(l));
    const auto f0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(x), scale), w0);
    const auto f1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(y), scale), w1);
    if constexpr (!Half) {
        // Streaming FP32 output: avoid allocating every strided destination
        // line in L1 while consuming native partials and accumulator lines.
        asm volatile("stnp %q0, %q1, [%2]"
                     :
                     : "w"(f0), "w"(f1), "r"(static_cast<float *>(output) + offset)
                     : "memory");
    } else {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        vst1q_u16(static_cast<uint16_t *>(output) + offset,
                  vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1))));
#else
        float v[8];
        vst1q_f32(v, f0);
        vst1q_f32(v + 4, f1);
        for (int i = 0; i < 8; ++i)
            static_cast<uint16_t *>(output)[offset + i] = rknpu2_matmul_open::bits::float_to_half(v[i]);
#endif
    }
}
template <bool First, bool Half>
inline void finish32(const int16_t *__restrict p, const int32_t *__restrict acc,
                     const int32_t *correction, const float *scales, const float *weight,
                     void *output, size_t out_offset, int N, int rows) {
    // Hold 32 channel scales across the token loop. Separate named vectors
    // avoid the stack-array spills seen with counted SIMD loops under GCC -O2.
    const auto w0 = vld1q_f32(weight), w1 = vld1q_f32(weight + 4);
    const auto w2 = vld1q_f32(weight + 8), w3 = vld1q_f32(weight + 12);
    const auto w4 = vld1q_f32(weight + 16), w5 = vld1q_f32(weight + 20);
    const auto w6 = vld1q_f32(weight + 24), w7 = vld1q_f32(weight + 28);
    for (int r = 0; r < rows; ++r, p += 16, acc += 8, out_offset += N) {
        const auto scale = vdupq_n_f32(scales[r]);
        const auto *a = First ? correction : acc;
        const int pitch = First ? 8 : rows * 8;
        finish8<Half>(p, a, scale, w0, w1, output, out_offset);
        finish8<Half>(p + rows * 16, a + pitch, scale, w2, w3, output, out_offset + 8);
        finish8<Half>(p + rows * 32, a + pitch * 2, scale, w4, w5, output, out_offset + 16);
        finish8<Half>(p + rows * 48, a + pitch * 3, scale, w6, w7, output, out_offset + 24);
    }
}
} // namespace rknpu2_matmul_open::detail::w4reduce
#endif

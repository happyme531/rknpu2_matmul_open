#include "rk_npu_cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline float max_abs_f32_neon(const float* row, int K) {
    float32x4_t vmax0 = vdupq_n_f32(0.0f);
    float32x4_t vmax1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(row + k)));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(row + k + 4)));
    }
    float max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    for (; k < K; ++k) max_abs = std::max(max_abs, std::fabs(row[k]));
    return max_abs;
}

static inline int8x16_t quant16_f32_s8_symmetric_neon(
    const float* src, float inv_scale) {
    const float32x4_t scale = vdupq_n_f32(inv_scale);
    int32x4_t q0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src), scale));
    int32x4_t q1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + 4), scale));
    int32x4_t q2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + 8), scale));
    int32x4_t q3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + 12), scale));
    const int32x4_t lo = vdupq_n_s32(-127);
    const int32x4_t hi = vdupq_n_s32(127);
    q0 = vmaxq_s32(lo, vminq_s32(q0, hi));
    q1 = vmaxq_s32(lo, vminq_s32(q1, hi));
    q2 = vmaxq_s32(lo, vminq_s32(q2, hi));
    q3 = vmaxq_s32(lo, vminq_s32(q3, hi));
    const int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    const int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    return vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1));
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define RKCPU_HAS_NEON_FP16 1
#else
#define RKCPU_HAS_NEON_FP16 0
#endif

namespace rknpu2_matmul_open::cpu {
namespace {

/* TODO(split-k-cpu-tuning): retune together with cpu_threads while the NPU is
 * active.  These crossovers are from isolated RK3588 big-core runs; shared-DRAM
 * contention may move them upward in the overlapped pipeline. */
constexpr uint64_t kSplitKReduceParallelElems = 1u << 13;
constexpr uint64_t kSplitKAccumParallelElems2 = 1u << 13;
constexpr uint64_t kSplitKAccumParallelElems4 = 1u << 15;

inline bool split_k_accum_use_omp(uint64_t elems) {
    const uint64_t threshold = omp_get_max_threads() <= 2
                             ? kSplitKAccumParallelElems2
                             : kSplitKAccumParallelElems4;
    return elems >= threshold;
}

template <typename T>
inline void zero_n(T* p, uint64_t n) {
    std::memset(p, 0, (size_t)n * sizeof(T));
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) {
        if (man == 0) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7c00u | (man >> 13) | 1u);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_man = man >> shift;
        const uint32_t round_bit = 1u << (shift - 1);
        if ((man & round_bit) && ((man & (round_bit - 1)) || (half_man & 1u)))
            ++half_man;
        return (uint16_t)(sign | half_man);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1000u) && ((man & 0x0fffu) || (half & 1u)))
        ++half;
    return (uint16_t)half;
}

inline int8_t quantize_s8_symmetric(float x, float scale) {
    if (!(scale > 0.0f)) return 0;
    float q = std::round(x * (1.0f / scale));
    q = std::max(-127.0f, std::min(127.0f, q));
    return (int8_t)q;
}

inline float apply_activation(float x, ActivationOp act) {
    switch (act) {
    case ActivationOp::None:
        return x;
    case ActivationOp::Relu:
        return x > 0.0f ? x : 0.0f;
    case ActivationOp::Silu:
        return x / (1.0f + std::exp(-x));
    case ActivationOp::GeluTanh: {
        constexpr float kAlpha = 0.7978845608028654f; /* sqrt(2/pi) */
        constexpr float kBeta = 0.044715f;
        const float x3 = x * x * x;
        return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x3)));
    }
    }
    return x;
}

static inline void copy_u32_1to4(uint32_t* dst, const uint32_t* src, int n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (n == 4) {
        vst1q_u32(dst, vld1q_u32(src));
        return;
    }
#endif
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

static inline void copy16_u8(uint8_t* dst, const uint8_t* src) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    vst1q_u8(dst, vld1q_u8(src));
#else
    std::memcpy(dst, src, 16);
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline bool can_dequant4_neon(ActivationOp act) {
    return act == ActivationOp::None || act == ActivationOp::Relu;
}

static inline float32x4_t dequant4_i32_f32_neon(const int32_t* sp, float as,
                                                const float* w_scale,
                                                const float* bias,
                                                int n0, ActivationOp act) {
    float32x4_t y = vcvtq_f32_s32(vld1q_s32(sp));
    y = vmulq_f32(y, vdupq_n_f32(as));
    if (w_scale) y = vmulq_f32(y, vld1q_f32(w_scale + n0));
    if (bias) y = vaddq_f32(y, vld1q_f32(bias + n0));
    if (act == ActivationOp::Relu) y = vmaxq_f32(y, vdupq_n_f32(0.0f));
    return y;
}

static inline float32x4_t dequant4_i32_f32_neon(int32x4_t acc, float as,
                                                const float* w_scale,
                                                const float* bias,
                                                int n0, ActivationOp act) {
    float32x4_t y = vcvtq_f32_s32(acc);
    y = vmulq_f32(y, vdupq_n_f32(as));
    if (w_scale) y = vmulq_f32(y, vld1q_f32(w_scale + n0));
    if (bias) y = vaddq_f32(y, vld1q_f32(bias + n0));
    if (act == ActivationOp::Relu) y = vmaxq_f32(y, vdupq_n_f32(0.0f));
    return y;
}

template <int Lane>
static inline float32x4_t dequant4_i32_f32_scaled_neon(
    const int32_t* sp, float32x4_t row_scales, float32x4_t weight_scale) {
    float32x4_t y = vcvtq_f32_s32(vld1q_s32(sp));
    y = vmulq_laneq_f32(y, row_scales, Lane);
    return vmulq_f32(y, weight_scale);
}

template <int Lane>
static inline void dequant_c_native_f32_1rx16_neon(
    const int32_t* sp0, const int32_t* sp1,
    const int32_t* sp2, const int32_t* sp3,
    float32x4_t row_scales,
    float32x4_t ws0, float32x4_t ws1,
    float32x4_t ws2, float32x4_t ws3,
    float* dst) {
    const float32x4x4_t out = {{
        dequant4_i32_f32_scaled_neon<Lane>(sp0, row_scales, ws0),
        dequant4_i32_f32_scaled_neon<Lane>(sp1, row_scales, ws1),
        dequant4_i32_f32_scaled_neon<Lane>(sp2, row_scales, ws2),
        dequant4_i32_f32_scaled_neon<Lane>(sp3, row_scales, ws3),
    }};
    vst1q_f32_x4(dst, out);
}

/* Native C is [N/4, M, 4], so the unit being transposed is one 128-bit
 * four-channel vector rather than one scalar lane.  Four native blocks form
 * one 16-column destination cache line; keep those weights resident while
 * walking eight adjacent source rows. */
static inline void dequant_c_native_f32_8rx16_neon(
    int N, int r0, int n0,
    const int32_t* sp0, const int32_t* sp1,
    const int32_t* sp2, const int32_t* sp3,
    const float* a_scale,
    float32x4_t ws0, float32x4_t ws1,
    float32x4_t ws2, float32x4_t ws3,
    float* dst_f32) {
    const float32x4_t as_lo = vld1q_f32(a_scale + r0);
    const float32x4_t as_hi = vld1q_f32(a_scale + r0 + 4);
    dequant_c_native_f32_1rx16_neon<0>(
        sp0 + 0, sp1 + 0, sp2 + 0, sp3 + 0,
        as_lo, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 0) * N + n0);
    dequant_c_native_f32_1rx16_neon<1>(
        sp0 + 4, sp1 + 4, sp2 + 4, sp3 + 4,
        as_lo, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 1) * N + n0);
    dequant_c_native_f32_1rx16_neon<2>(
        sp0 + 8, sp1 + 8, sp2 + 8, sp3 + 8,
        as_lo, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 2) * N + n0);
    dequant_c_native_f32_1rx16_neon<3>(
        sp0 + 12, sp1 + 12, sp2 + 12, sp3 + 12,
        as_lo, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 3) * N + n0);
    dequant_c_native_f32_1rx16_neon<0>(
        sp0 + 16, sp1 + 16, sp2 + 16, sp3 + 16,
        as_hi, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 4) * N + n0);
    dequant_c_native_f32_1rx16_neon<1>(
        sp0 + 20, sp1 + 20, sp2 + 20, sp3 + 20,
        as_hi, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 5) * N + n0);
    dequant_c_native_f32_1rx16_neon<2>(
        sp0 + 24, sp1 + 24, sp2 + 24, sp3 + 24,
        as_hi, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 6) * N + n0);
    dequant_c_native_f32_1rx16_neon<3>(
        sp0 + 28, sp1 + 28, sp2 + 28, sp3 + 28,
        as_hi, ws0, ws1, ws2, ws3,
        dst_f32 + (size_t)(r0 + 7) * N + n0);
}

static inline int32x4_t reduce4_i32_neon(int partial_count,
                                         const int32_t* const* src_native,
                                         size_t offset) {
    const int32x4_t p0 = vld1q_s32(src_native[0] + offset);
    if (partial_count == 2)
        return vaddq_s32(p0, vld1q_s32(src_native[1] + offset));
    if (partial_count == 4) {
        const int32x4_t p1 = vld1q_s32(src_native[1] + offset);
        const int32x4_t p2 = vld1q_s32(src_native[2] + offset);
        const int32x4_t p3 = vld1q_s32(src_native[3] + offset);
        return vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3));
    }
    int32x4_t acc = p0;
    for (int p = 1; p < partial_count; ++p)
        acc = vaddq_s32(acc, vld1q_s32(src_native[p] + offset));
    return acc;
}
#endif

static inline int32_t reduce1_i32_scalar(int partial_count,
                                         const int32_t* const* src_native,
                                         size_t offset) {
    /* Unsigned addition defines the same modulo-2^32 behavior as NEON vadd. */
    uint32_t acc = 0;
    for (int p = 0; p < partial_count; ++p)
        acc += (uint32_t)src_native[p][offset];
    int32_t out;
    std::memcpy(&out, &acc, sizeof(out));
    return out;
}

#if RKCPU_HAS_NEON_FP16
static inline float max_abs_f16_neon(const uint16_t* row, int K) {
    float32x4_t vmax0 = vdupq_n_f32(0.0f);
    float32x4_t vmax1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(row + k));
        const float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
        const float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(lo));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(hi));
    }
    float max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    for (; k < K; ++k)
        max_abs = std::max(max_abs, std::fabs(half_to_float(row[k])));
    return max_abs;
}

static inline int16x4_t f32_to_s16_symmetric_q_neon(float32x4_t v) {
    int32x4_t q = vcvtaq_s32_f32(v);
    q = vmaxq_s32(vdupq_n_s32(-127), vminq_s32(q, vdupq_n_s32(127)));
    return vqmovn_s32(q);
}

static inline int8x16_t quant16_f16_s8_symmetric_neon(const uint16_t* src, float inv_scale) {
    const float32x4_t scale = vdupq_n_f32(inv_scale);
    const float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(src));
    const float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(src + 8));

    const int16x8_t q0 = vcombine_s16(
        f32_to_s16_symmetric_q_neon(vmulq_f32(vcvt_f32_f16(vget_low_f16(h0)), scale)),
        f32_to_s16_symmetric_q_neon(vmulq_f32(vcvt_f32_f16(vget_high_f16(h0)), scale)));
    const int16x8_t q1 = vcombine_s16(
        f32_to_s16_symmetric_q_neon(vmulq_f32(vcvt_f32_f16(vget_low_f16(h1)), scale)),
        f32_to_s16_symmetric_q_neon(vmulq_f32(vcvt_f32_f16(vget_high_f16(h1)), scale)));

    return vcombine_s8(vqmovn_s16(q0), vqmovn_s16(q1));
}
#endif

static inline void transpose16bit_scalar(uint16_t* dst, const uint16_t* src, int w, int h,
                                         int src_stride, int dst_stride) {
    for (int i = 0; i < h; ++i) {
        uint16_t* d = dst + (size_t)i * dst_stride;
        const uint16_t* s = src + i;
        for (int j = 0; j < w; ++j)
            d[j] = s[(size_t)j * src_stride];
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline void transpose8x8_u8(uint8_t* dst, const uint8_t* src,
                                   int src_stride, int dst_stride) {
    uint8x8_t s0 = vld1_u8(src + (size_t)0 * src_stride);
    uint8x8_t s1 = vld1_u8(src + (size_t)1 * src_stride);
    uint8x8_t s2 = vld1_u8(src + (size_t)2 * src_stride);
    uint8x8_t s3 = vld1_u8(src + (size_t)3 * src_stride);
    uint8x8_t s4 = vld1_u8(src + (size_t)4 * src_stride);
    uint8x8_t s5 = vld1_u8(src + (size_t)5 * src_stride);
    uint8x8_t s6 = vld1_u8(src + (size_t)6 * src_stride);
    uint8x8_t s7 = vld1_u8(src + (size_t)7 * src_stride);

    uint8x8x2_t t0 = vtrn_u8(s0, s1);
    uint8x8x2_t t1 = vtrn_u8(s2, s3);
    uint8x8x2_t t2 = vtrn_u8(s4, s5);
    uint8x8x2_t t3 = vtrn_u8(s6, s7);

    uint16x4x2_t u0 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]),
                               vreinterpret_u16_u8(t1.val[0]));
    uint16x4x2_t u1 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]),
                               vreinterpret_u16_u8(t3.val[0]));
    uint16x4x2_t u2 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]),
                               vreinterpret_u16_u8(t1.val[1]));
    uint16x4x2_t u3 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]),
                               vreinterpret_u16_u8(t3.val[1]));

    uint32x2x2_t v0 = vtrn_u32(vreinterpret_u32_u16(u0.val[0]),
                               vreinterpret_u32_u16(u1.val[0]));
    uint32x2x2_t v1 = vtrn_u32(vreinterpret_u32_u16(u2.val[0]),
                               vreinterpret_u32_u16(u3.val[0]));
    uint32x2x2_t v2 = vtrn_u32(vreinterpret_u32_u16(u0.val[1]),
                               vreinterpret_u32_u16(u1.val[1]));
    uint32x2x2_t v3 = vtrn_u32(vreinterpret_u32_u16(u2.val[1]),
                               vreinterpret_u32_u16(u3.val[1]));

    vst1_u8(dst + (size_t)0 * dst_stride, vreinterpret_u8_u32(v0.val[0]));
    vst1_u8(dst + (size_t)1 * dst_stride, vreinterpret_u8_u32(v1.val[0]));
    vst1_u8(dst + (size_t)2 * dst_stride, vreinterpret_u8_u32(v2.val[0]));
    vst1_u8(dst + (size_t)3 * dst_stride, vreinterpret_u8_u32(v3.val[0]));
    vst1_u8(dst + (size_t)4 * dst_stride, vreinterpret_u8_u32(v0.val[1]));
    vst1_u8(dst + (size_t)5 * dst_stride, vreinterpret_u8_u32(v1.val[1]));
    vst1_u8(dst + (size_t)6 * dst_stride, vreinterpret_u8_u32(v2.val[1]));
    vst1_u8(dst + (size_t)7 * dst_stride, vreinterpret_u8_u32(v3.val[1]));
}

static inline void transpose8x8_u16(uint16_t* dst, const uint16_t* src,
                                    int src_stride, int dst_stride) {
    uint16x8_t s0 = vld1q_u16(src + (size_t)0 * src_stride);
    uint16x8_t s1 = vld1q_u16(src + (size_t)1 * src_stride);
    uint16x8_t s2 = vld1q_u16(src + (size_t)2 * src_stride);
    uint16x8_t s3 = vld1q_u16(src + (size_t)3 * src_stride);
    uint16x8_t s4 = vld1q_u16(src + (size_t)4 * src_stride);
    uint16x8_t s5 = vld1q_u16(src + (size_t)5 * src_stride);
    uint16x8_t s6 = vld1q_u16(src + (size_t)6 * src_stride);
    uint16x8_t s7 = vld1q_u16(src + (size_t)7 * src_stride);

    uint16x8x2_t t01 = vzipq_u16(s0, s1);
    uint16x8x2_t t23 = vzipq_u16(s2, s3);
    uint16x8x2_t t45 = vzipq_u16(s4, s5);
    uint16x8x2_t t67 = vzipq_u16(s6, s7);

    uint32x4x2_t u0 = vzipq_u32(vreinterpretq_u32_u16(t01.val[0]),
                                vreinterpretq_u32_u16(t23.val[0]));
    uint32x4x2_t u1 = vzipq_u32(vreinterpretq_u32_u16(t01.val[1]),
                                vreinterpretq_u32_u16(t23.val[1]));
    uint32x4x2_t u2 = vzipq_u32(vreinterpretq_u32_u16(t45.val[0]),
                                vreinterpretq_u32_u16(t67.val[0]));
    uint32x4x2_t u3 = vzipq_u32(vreinterpretq_u32_u16(t45.val[1]),
                                vreinterpretq_u32_u16(t67.val[1]));

    const uint64x2_t u00 = vreinterpretq_u64_u32(u0.val[0]);
    const uint64x2_t u01 = vreinterpretq_u64_u32(u0.val[1]);
    const uint64x2_t u10 = vreinterpretq_u64_u32(u1.val[0]);
    const uint64x2_t u11 = vreinterpretq_u64_u32(u1.val[1]);
    const uint64x2_t u20 = vreinterpretq_u64_u32(u2.val[0]);
    const uint64x2_t u21 = vreinterpretq_u64_u32(u2.val[1]);
    const uint64x2_t u30 = vreinterpretq_u64_u32(u3.val[0]);
    const uint64x2_t u31 = vreinterpretq_u64_u32(u3.val[1]);

    vst1q_u16(dst + (size_t)0 * dst_stride, vreinterpretq_u16_u64(vzip1q_u64(u00, u20)));
    vst1q_u16(dst + (size_t)1 * dst_stride, vreinterpretq_u16_u64(vzip2q_u64(u00, u20)));
    vst1q_u16(dst + (size_t)2 * dst_stride, vreinterpretq_u16_u64(vzip1q_u64(u01, u21)));
    vst1q_u16(dst + (size_t)3 * dst_stride, vreinterpretq_u16_u64(vzip2q_u64(u01, u21)));
    vst1q_u16(dst + (size_t)4 * dst_stride, vreinterpretq_u16_u64(vzip1q_u64(u10, u30)));
    vst1q_u16(dst + (size_t)5 * dst_stride, vreinterpretq_u16_u64(vzip2q_u64(u10, u30)));
    vst1q_u16(dst + (size_t)6 * dst_stride, vreinterpretq_u16_u64(vzip1q_u64(u11, u31)));
    vst1q_u16(dst + (size_t)7 * dst_stride, vreinterpretq_u16_u64(vzip2q_u64(u11, u31)));
}
#endif

static inline void transpose8bit_scalar(uint8_t* dst, const uint8_t* src, int w, int h,
                                        int src_stride, int dst_stride) {
    for (int i = 0; i < h; ++i) {
        uint8_t* d = dst + (size_t)i * dst_stride;
        const uint8_t* s = src + i;
        for (int j = 0; j < w; ++j)
            d[j] = s[(size_t)j * src_stride];
    }
}

static inline void transpose8bit_tile(uint8_t* dst, const uint8_t* src, int w, int h,
                                      int src_stride, int dst_stride) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int w8 = w / 8;
    const int h8 = h / 8;
    for (int y = 0; y < h8; ++y)
        for (int x = 0; x < w8; ++x)
            transpose8x8_u8(dst + (size_t)(y * 8) * dst_stride + x * 8,
                            src + y * 8 + (size_t)(x * 8) * src_stride,
                            src_stride, dst_stride);
    const int h_done = h8 * 8;
    const int w_done = w8 * 8;
    if (h_done < h)
        transpose8bit_scalar(dst + (size_t)h_done * dst_stride, src + h_done,
                             w, h - h_done, src_stride, dst_stride);
    if (w_done < w)
        transpose8bit_scalar(dst + w_done, src + (size_t)w_done * src_stride,
                             w - w_done, h_done, src_stride, dst_stride);
#else
    transpose8bit_scalar(dst, src, w, h, src_stride, dst_stride);
#endif
}

static inline void transpose8bit_32x32(uint8_t* dst, const uint8_t* src, int src_stride) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            transpose8x8_u8(dst + (size_t)(y * 8) * 32 + x * 8,
                            src + y * 8 + (size_t)(x * 8) * src_stride,
                            src_stride, 32);
#else
    transpose8bit_scalar(dst, src, 32, 32, src_stride, 32);
#endif
}

static inline void transpose16bit_tile(uint16_t* dst, const uint16_t* src, int w, int h,
                                       int src_stride, int dst_stride) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int w8 = w / 8;
    const int h8 = h / 8;
    for (int y = 0; y < h8; ++y)
        for (int x = 0; x < w8; ++x)
            transpose8x8_u16(dst + (size_t)(y * 8) * dst_stride + x * 8,
                             src + y * 8 + (size_t)(x * 8) * src_stride,
                             src_stride, dst_stride);
    const int h_done = h8 * 8;
    const int w_done = w8 * 8;
    if (h_done < h)
        transpose16bit_scalar(dst + (size_t)h_done * dst_stride, src + h_done,
                              w, h - h_done, src_stride, dst_stride);
    if (w_done < w)
        transpose16bit_scalar(dst + w_done, src + (size_t)w_done * src_stride,
                              w - w_done, h_done, src_stride, dst_stride);
#else
    transpose16bit_scalar(dst, src, w, h, src_stride, dst_stride);
#endif
}

static inline void transpose16bit_32x16(uint16_t* dst, const uint16_t* src, int src_stride) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 4; ++x)
            transpose8x8_u16(dst + (size_t)(y * 8) * 32 + x * 8,
                             src + y * 8 + (size_t)(x * 8) * src_stride,
                             src_stride, 32);
#else
    transpose16bit_scalar(dst, src, 32, 16, src_stride, 32);
#endif
}

} /* namespace */

void i8_pack_a_normal(int M, int K, int align_in, const int8_t* src, int8_t* dst) {
    if (K == align_in) {
        std::memcpy(dst, src, (size_t)M * K);
        return;
    }
    const uint64_t elems = (uint64_t)M * align_in;
    zero_n(dst, elems);
#pragma omp parallel for schedule(static) if(elems >= (1u << 15))
    for (int r = 0; r < M; ++r) {
        std::memcpy(dst + (size_t)r * align_in, src + (size_t)r * K, (size_t)K);
    }
}

void i8_pack_a_native_k16_m16(int M, int K, int align_in, const int8_t *src, int8_t *dst,
                              int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows))
        return;
    constexpr int subK = 16;
    const int kb_count = align_in / subK;
    const uint64_t elems = (uint64_t)kb_count * M * subK;
    if (K != align_in)
        zero_n(dst, elems);
#pragma omp parallel for collapse(2) schedule(static) if (elems >= (1u << 20))
    for (int panel_index = 0; panel_index < (panel_rows ? M / panel_rows : 1); ++panel_index) {
        for (int kb = 0; kb < kb_count; ++kb) {
            for (int r = panel_index * (panel_rows ? panel_rows : M);
                 r < (panel_index + 1) * (panel_rows ? panel_rows : M); ++r) {
                const int k0 = kb * subK;
                const int valid = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
                if (valid == subK)
                    copy16_u8((uint8_t *)dst +
                                  rknpu2_matmul_open::cpu::i8_a_block_offset(M, align_in, r, kb, panel_rows),
                              (const uint8_t *)src + (size_t)r * K + k0);
                else if (valid > 0)
                    std::memcpy(dst + rknpu2_matmul_open::cpu::i8_a_block_offset(M, align_in, r, kb, panel_rows),
                                src + (size_t)r * K + k0, (size_t)valid);
            }
        }
    }
}

void i8_pack_a_f16_quant_native_k16_m16(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        QuantMode mode,
                                        const float* static_scale,
                                        float* scale_out,
                                        int8_t* dst) {
    constexpr int subK = 16;
    const int kb_count = align_in / subK;
    const uint64_t elems = (uint64_t)kb_count * M * subK;
    if (K != align_in) zero_n(dst, elems);

#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src_fp16 + (size_t)r * K;
        float scale = 1.0f;
        if (mode == QuantMode::DynamicPerToken) {
#if RKCPU_HAS_NEON_FP16
            const float max_abs = max_abs_f16_neon(row, K);
#else
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float v = half_to_float(row[k]);
                max_abs = std::max(max_abs, std::fabs(v));
            }
#endif
            scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            if (scale_out) scale_out[r] = scale;
        } else if (mode == QuantMode::StaticPerToken) {
            scale = static_scale ? static_scale[r] : 1.0f;
            if (scale_out) scale_out[r] = scale;
        } else {
            scale = static_scale ? static_scale[0] : 1.0f;
            if (scale_out) scale_out[r] = scale;
        }

        if (!(scale > 0.0f)) {
            for (int kb = 0; kb < kb_count; ++kb) {
                const int k0 = kb * subK;
                const int valid = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
                if (valid > 0)
                    std::memset(dst + (size_t)kb * M * subK + (size_t)r * subK, 0, (size_t)valid);
            }
            continue;
        }

        const float inv_scale = 1.0f / scale;
        for (int kb = 0; kb < kb_count; ++kb) {
            const int k0 = kb * subK;
            const int valid = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
            int8_t* out = dst + (size_t)kb * M * subK + (size_t)r * subK;
#if RKCPU_HAS_NEON_FP16
            if (valid == subK) {
                vst1q_s8(out, quant16_f16_s8_symmetric_neon(row + k0, inv_scale));
                continue;
            }
#endif
            for (int kk = 0; kk < valid; ++kk)
                out[kk] = quantize_s8_symmetric(half_to_float(row[k0 + kk]), scale);
        }
    }
}

void i8_compute_dynamic_per_token_scale_f16(int M, int K,
                                             const uint16_t* src_fp16,
                                             int src_row_stride,
                                             float* scale_out) {
    if (M <= 0 || K <= 0 || !src_fp16 || src_row_stride < K || !scale_out)
        return;
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src_fp16 + (size_t)r * src_row_stride;
#if RKCPU_HAS_NEON_FP16
        const float max_abs = max_abs_f16_neon(row, K);
#else
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::fabs(half_to_float(row[k])));
#endif
        scale_out[r] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
}

void i8_pack_a_f16_quant_normal_strided(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst) {
    if (M <= 0 || K <= 0 || align_in < K || !src_fp16 ||
        src_row_stride < K || !per_token_scale || !dst)
        return;
    if (K != align_in) zero_n(dst, (uint64_t)M * align_in);
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src_fp16 + (size_t)r * src_row_stride;
        int8_t* out = dst + (size_t)r * align_in;
        const float scale = per_token_scale[r];
        if (!(scale > 0.0f)) {
            std::memset(out, 0, (size_t)K);
            continue;
        }
        const float inv_scale = 1.0f / scale;
        int k = 0;
#if RKCPU_HAS_NEON_FP16
        for (; k + 16 <= K; k += 16)
            vst1q_s8(out + k, quant16_f16_s8_symmetric_neon(row + k, inv_scale));
#endif
        for (; k < K; ++k)
            out[k] = quantize_s8_symmetric(half_to_float(row[k]), scale);
    }
}

void i8_pack_a_f16_quant_native_strided(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    /* Keep the same four-row scheduling idea as KleidiAI's p4 quant packers:
     * source rows retain streaming reads while one native K16 block is emitted
     * by four adjacent 16-byte stores. Our scale remains full-row/per-token. */
    constexpr int sub_k = 16;
    constexpr int row_block = 4;
    if (M <= 0 || K <= 0 || align_in < K || (align_in % sub_k) != 0 ||
        !src_fp16 || src_row_stride < K || !per_token_scale || !dst)
        return;

    const int kb_count = align_in / sub_k;
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r0 = 0; r0 < M; r0 += row_block) {
        const int rows = std::min(row_block, M - r0);
        float inv_scale[row_block]{};
#pragma GCC unroll 4
        for (int rr = 0; rr < rows; ++rr) {
            const float scale = per_token_scale[r0 + rr];
            inv_scale[rr] = scale > 0.0f ? 1.0f / scale : 0.0f;
        }
        for (int kb = 0; kb < kb_count; ++kb) {
            const int k0 = kb * sub_k;
            const int valid = std::min(sub_k, std::max(0, K - k0));
            int8_t* out = dst + rknpu2_matmul_open::cpu::i8_a_block_offset(M, align_in, r0, kb, panel_rows);
#pragma GCC unroll 4
            for (int rr = 0; rr < rows; ++rr, out += sub_k) {
                const uint16_t* row = src_fp16 +
                    (size_t)(r0 + rr) * src_row_stride;
                if (!(inv_scale[rr] > 0.0f)) {
                    std::memset(out, 0, sub_k);
                    continue;
                }
#if RKCPU_HAS_NEON_FP16
                if (valid == sub_k) {
                    vst1q_s8(out, quant16_f16_s8_symmetric_neon(
                        row + k0, inv_scale[rr]));
                    continue;
                }
#endif
                const float scale = per_token_scale[r0 + rr];
                for (int kk = 0; kk < valid; ++kk)
                    out[kk] = quantize_s8_symmetric(
                        half_to_float(row[k0 + kk]), scale);
                if (valid < sub_k)
                    std::memset(out + valid, 0, (size_t)(sub_k - valid));
            }
        }
    }
}

void i8_pack_a_f16_dynamic_normal_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const uint16_t* src_fp16,
                                        float* scale_out,
                                        int8_t* const* dst_slices) {
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_fp16 || !scale_out || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src_fp16 + (size_t)r * full_K;
#if RKCPU_HAS_NEON_FP16
        const float max_abs = max_abs_f16_neon(row, full_K);
#else
        float max_abs = 0.0f;
        for (int k = 0; k < full_K; ++k)
            max_abs = std::max(max_abs, std::fabs(half_to_float(row[k])));
#endif
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = 1.0f / scale;
        scale_out[r] = scale;

        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            const uint16_t* slice = row + slice_k0[s];
            int8_t* out = dst_slices[s] + (size_t)r * slice_align_in[s];
            int k = 0;
#if RKCPU_HAS_NEON_FP16
            for (; k + 16 <= K; k += 16)
                vst1q_s8(out + k, quant16_f16_s8_symmetric_neon(slice + k, inv_scale));
#endif
            for (; k < K; ++k)
                out[k] = quantize_s8_symmetric(half_to_float(slice[k]), scale);
            if (slice_align_in[s] > K)
                std::memset(out + K, 0, (size_t)(slice_align_in[s] - K));
        }
    }
}

void i8_pack_a_f16_dynamic_native_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const uint16_t* src_fp16,
                                        float* scale_out,
                                        int8_t* const* dst_slices, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    constexpr int sub_k = 16;
    constexpr int row_block = 4;
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_fp16 || !scale_out || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] ||
            (slice_align_in[s] % sub_k) != 0 || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r0 = 0; r0 < M; r0 += row_block) {
        const int rows = std::min(row_block, M - r0);
        float scale[row_block]{};
        float inv_scale[row_block]{};
#pragma GCC unroll 4
        for (int rr = 0; rr < rows; ++rr) {
            const uint16_t* row = src_fp16 + (size_t)(r0 + rr) * full_K;
#if RKCPU_HAS_NEON_FP16
            const float max_abs = max_abs_f16_neon(row, full_K);
#else
            float max_abs = 0.0f;
            for (int k = 0; k < full_K; ++k)
                max_abs = std::max(max_abs, std::fabs(half_to_float(row[k])));
#endif
            scale[rr] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            inv_scale[rr] = 1.0f / scale[rr];
            scale_out[r0 + rr] = scale[rr];
        }

        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            const int kb_count = slice_align_in[s] / sub_k;
            for (int kb = 0; kb < kb_count; ++kb) {
                const int local_k = kb * sub_k;
                const int valid = std::min(sub_k, std::max(0, K - local_k));
                int8_t* out = dst_slices[s] +
                    rknpu2_matmul_open::cpu::i8_a_block_offset(M, slice_align_in[s], r0, kb, panel_rows);
#pragma GCC unroll 4
                for (int rr = 0; rr < rows; ++rr, out += sub_k) {
                    const uint16_t* src = src_fp16 +
                        (size_t)(r0 + rr) * full_K + slice_k0[s] + local_k;
#if RKCPU_HAS_NEON_FP16
                    if (valid == sub_k) {
                        vst1q_s8(out, quant16_f16_s8_symmetric_neon(
                            src, inv_scale[rr]));
                        continue;
                    }
#endif
                    for (int kk = 0; kk < valid; ++kk)
                        out[kk] = quantize_s8_symmetric(
                            half_to_float(src[kk]), scale[rr]);
                    if (valid < sub_k)
                        std::memset(out + valid, 0, (size_t)(sub_k - valid));
                }
            }
        }
    }
}

void i8_compute_dynamic_per_token_scale_f32(int M, int K,
                                             const float* src_f32,
                                             int src_row_stride,
                                             float* scale_out) {
    if (M <= 0 || K <= 0 || !src_f32 || src_row_stride < K || !scale_out)
        return;
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const float* row = src_f32 + (size_t)r * src_row_stride;
#if defined(__aarch64__) && defined(__ARM_NEON)
        const float max_abs = max_abs_f32_neon(row, K);
#else
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::fabs(row[k]));
#endif
        scale_out[r] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
}

void i8_pack_a_f32_quant_normal_strided(int M, int K, int align_in,
                                        const float* src_f32,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst) {
    if (M <= 0 || K <= 0 || align_in < K || !src_f32 ||
        src_row_stride < K || !per_token_scale || !dst)
        return;
    if (K != align_in) zero_n(dst, (uint64_t)M * align_in);
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const float* row = src_f32 + (size_t)r * src_row_stride;
        int8_t* out = dst + (size_t)r * align_in;
        const float scale = per_token_scale[r];
        if (!(scale > 0.0f)) {
            std::memset(out, 0, (size_t)K);
            continue;
        }
        const float inv_scale = 1.0f / scale;
        int k = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        for (; k + 16 <= K; k += 16)
            vst1q_s8(out + k, quant16_f32_s8_symmetric_neon(row + k, inv_scale));
#endif
        for (; k < K; ++k)
            out[k] = quantize_s8_symmetric(row[k], scale);
    }
}

void i8_pack_a_f32_quant_native_strided(int M, int K, int align_in,
                                        const float* src_f32,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    constexpr int sub_k = 16;
    constexpr int row_block = 4;
    if (M <= 0 || K <= 0 || align_in < K || (align_in % sub_k) != 0 ||
        !src_f32 || src_row_stride < K || !per_token_scale || !dst)
        return;
    const int kb_count = align_in / sub_k;
#pragma omp parallel for schedule(static) if((uint64_t)M * K >= (1u << 14))
    for (int r0 = 0; r0 < M; r0 += row_block) {
        const int rows = std::min(row_block, M - r0);
        float inv_scale[row_block]{};
#pragma GCC unroll 4
        for (int rr = 0; rr < rows; ++rr) {
            const float scale = per_token_scale[r0 + rr];
            inv_scale[rr] = scale > 0.0f ? 1.0f / scale : 0.0f;
        }
        for (int kb = 0; kb < kb_count; ++kb) {
            const int k0 = kb * sub_k;
            const int valid = std::min(sub_k, std::max(0, K - k0));
            int8_t* out = dst + rknpu2_matmul_open::cpu::i8_a_block_offset(M, align_in, r0, kb, panel_rows);
#pragma GCC unroll 4
            for (int rr = 0; rr < rows; ++rr, out += sub_k) {
                const float* row = src_f32 +
                    (size_t)(r0 + rr) * src_row_stride;
                if (!(inv_scale[rr] > 0.0f)) {
                    std::memset(out, 0, sub_k);
                    continue;
                }
#if defined(__aarch64__) && defined(__ARM_NEON)
                if (valid == sub_k) {
                    vst1q_s8(out, quant16_f32_s8_symmetric_neon(
                        row + k0, inv_scale[rr]));
                    continue;
                }
#endif
                const float scale = per_token_scale[r0 + rr];
                for (int kk = 0; kk < valid; ++kk)
                    out[kk] = quantize_s8_symmetric(row[k0 + kk], scale);
                if (valid < sub_k)
                    std::memset(out + valid, 0, (size_t)(sub_k - valid));
            }
        }
    }
}

void i8_pack_a_f32_dynamic_normal_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const float* src_f32,
                                        float* scale_out,
                                        int8_t* const* dst_slices) {
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_f32 || !scale_out || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const float* row = src_f32 + (size_t)r * full_K;
#if defined(__aarch64__) && defined(__ARM_NEON)
        const float max_abs = max_abs_f32_neon(row, full_K);
#else
        float max_abs = 0.0f;
        for (int k = 0; k < full_K; ++k)
            max_abs = std::max(max_abs, std::fabs(row[k]));
#endif
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = 1.0f / scale;
        scale_out[r] = scale;

        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            const float* slice = row + slice_k0[s];
            int8_t* out = dst_slices[s] + (size_t)r * slice_align_in[s];
            int k = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            for (; k + 16 <= K; k += 16)
                vst1q_s8(out + k,
                         quant16_f32_s8_symmetric_neon(slice + k, inv_scale));
#endif
            for (; k < K; ++k)
                out[k] = quantize_s8_symmetric(slice[k], scale);
            if (slice_align_in[s] > K)
                std::memset(out + K, 0, (size_t)(slice_align_in[s] - K));
        }
    }
}

void i8_pack_a_f32_dynamic_native_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const float* src_f32,
                                        float* scale_out,
                                        int8_t* const* dst_slices, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    constexpr int sub_k = 16;
    constexpr int row_block = 4;
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_f32 || !scale_out || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] ||
            (slice_align_in[s] % sub_k) != 0 || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r0 = 0; r0 < M; r0 += row_block) {
        const int rows = std::min(row_block, M - r0);
        float scale[row_block]{};
        float inv_scale[row_block]{};
#pragma GCC unroll 4
        for (int rr = 0; rr < rows; ++rr) {
            const float* row = src_f32 + (size_t)(r0 + rr) * full_K;
#if defined(__aarch64__) && defined(__ARM_NEON)
            const float max_abs = max_abs_f32_neon(row, full_K);
#else
            float max_abs = 0.0f;
            for (int k = 0; k < full_K; ++k)
                max_abs = std::max(max_abs, std::fabs(row[k]));
#endif
            scale[rr] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            inv_scale[rr] = 1.0f / scale[rr];
            scale_out[r0 + rr] = scale[rr];
        }
        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            const int kb_count = slice_align_in[s] / sub_k;
            for (int kb = 0; kb < kb_count; ++kb) {
                const int local_k = kb * sub_k;
                const int valid = std::min(sub_k, std::max(0, K - local_k));
                int8_t* out = dst_slices[s] +
                    rknpu2_matmul_open::cpu::i8_a_block_offset(M, slice_align_in[s], r0, kb, panel_rows);
#pragma GCC unroll 4
                for (int rr = 0; rr < rows; ++rr, out += sub_k) {
                    const float* src = src_f32 +
                        (size_t)(r0 + rr) * full_K + slice_k0[s] + local_k;
#if defined(__aarch64__) && defined(__ARM_NEON)
                    if (valid == sub_k) {
                        vst1q_s8(out, quant16_f32_s8_symmetric_neon(
                            src, inv_scale[rr]));
                        continue;
                    }
#endif
                    for (int kk = 0; kk < valid; ++kk)
                        out[kk] = quantize_s8_symmetric(src[kk], scale[rr]);
                    if (valid < sub_k)
                        std::memset(out + valid, 0, (size_t)(sub_k - valid));
                }
            }
        }
    }
}

void i8_pack_a_i8_normal_split(int M, int full_K,
                               int slice_count,
                               const int* slice_k0,
                               const int* slice_k,
                               const int* slice_align_in,
                               const int8_t* src_i8,
                               int8_t* const* dst_slices) {
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_i8 || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const int8_t* row = src_i8 + (size_t)r * full_K;
        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            int8_t* out = dst_slices[s] + (size_t)r * slice_align_in[s];
            std::memcpy(out, row + slice_k0[s], (size_t)K);
            if (slice_align_in[s] > K)
                std::memset(out + K, 0, (size_t)(slice_align_in[s] - K));
        }
    }
}

void i8_pack_a_f16_static_normal_split(int M, int full_K,
                                       int slice_count,
                                       const int* slice_k0,
                                       const int* slice_k,
                                       const int* slice_align_in,
                                       const uint16_t* src_fp16,
                                       const float* per_token_scale,
                                       int8_t* const* dst_slices) {
    if (M <= 0 || full_K <= 0 || slice_count <= 0 || !slice_k0 || !slice_k ||
        !slice_align_in || !src_fp16 || !per_token_scale || !dst_slices)
        return;
    for (int s = 0; s < slice_count; ++s) {
        if (slice_k0[s] < 0 || slice_k[s] <= 0 ||
            slice_k0[s] + slice_k[s] > full_K ||
            slice_align_in[s] < slice_k[s] || !dst_slices[s])
            return;
    }

#pragma omp parallel for schedule(static) if((uint64_t)M * full_K >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src_fp16 + (size_t)r * full_K;
        const float scale = per_token_scale[r];
        const float inv_scale = 1.0f / scale;
        for (int s = 0; s < slice_count; ++s) {
            const int K = slice_k[s];
            const uint16_t* slice = row + slice_k0[s];
            int8_t* out = dst_slices[s] + (size_t)r * slice_align_in[s];
            int k = 0;
#if RKCPU_HAS_NEON_FP16
            for (; k + 16 <= K; k += 16)
                vst1q_s8(out + k, quant16_f16_s8_symmetric_neon(slice + k, inv_scale));
#endif
            for (; k < K; ++k)
                out[k] = quantize_s8_symmetric(half_to_float(slice[k]), scale);
            if (slice_align_in[s] > K)
                std::memset(out + K, 0, (size_t)(slice_align_in[s] - K));
        }
    }
}

void i8_pack_b_native_n32_k32(int K, int N, int align_in, int align_out,
                              const int8_t* src, int8_t* dst) {
    constexpr int subN = 32;
    constexpr int subK = 32;
    const int nb_count = align_out / subN;
    const int kb_count = align_in / subK;
    const int total_tiles = nb_count * kb_count;
    const bool full_tiles = (K == align_in && N == align_out);
    const bool use_omp = (uint64_t)align_out * align_in >= (512u << 10);
    if (!full_tiles) zero_n(dst, (uint64_t)align_out * align_in);
#pragma omp parallel for schedule(static) if(use_omp)
    for (int tile = 0; tile < total_tiles; ++tile) {
        const int nb = tile / kb_count;
        const int kb = tile - nb * kb_count;
        const int n0 = nb * subN;
        const int k0 = kb * subK;
        int8_t* tile_dst = dst + ((size_t)nb * kb_count + kb) * subN * subK;
        const int8_t* tile_src = src + (size_t)k0 * N + n0;
        if (full_tiles) {
            transpose8bit_32x32((uint8_t*)tile_dst, (const uint8_t*)tile_src, N);
            continue;
        }
        const int valid_n = (n0 + subN <= N) ? subN : (N > n0 ? N - n0 : 0);
        const int valid_k = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
        if (valid_n <= 0 || valid_k <= 0) continue;
        transpose8bit_tile((uint8_t*)tile_dst, (const uint8_t*)tile_src, valid_k, valid_n, N, subK);
    }
}

void i8_unpack_c_normal(int M, int N, int align_out, const void* src, void* dst) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 13))
    for (int r = 0; r < M; ++r) {
        std::memcpy(d + (size_t)r * N * 4, s + (size_t)r * align_out * 4, (size_t)N * 4);
    }
}

void i8_unpack_c_native_n4_m4(int M, int N, const void* src, void* dst, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    const uint32_t* s = (const uint32_t*)src;
    uint32_t* d = (uint32_t*)dst;
    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    /* Board microbench: row-block is faster at N=2048, column-block is faster
     * for the large native-C output case N=6144. */
    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 17))
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = (nb0 + nb_group <= nb_count) ? (nb0 + nb_group) : nb_count;
                for (int r = r0; r < r1; ++r) {
                    uint32_t* dp = d + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                        const uint32_t* sp = s + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
                        copy_u32_1to4(dp + (nb - nb0) * 4, sp, valid);
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 17))
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = (nb0 + nb_block <= nb_count) ? (nb0 + nb_block) : nb_count;
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb = nb0; nb < nb1; ++nb) {
                const uint32_t* sp = s + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r0, nb, panel_rows);
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                    uint32_t* dp = d + (size_t)r * N + n0;
                    copy_u32_1to4(dp, sp + (r - r0) * 4, valid);
                }
            }
        }
    }
}

void i8_unpack_c_native_i32_dequant_f32(int M, int N,
                                        const int32_t* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        float* dst_f32, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
#if defined(__aarch64__) && defined(__ARM_NEON)
    /* The quantized FP32 runner always reaches this common case.  Keep small M,
     * small N, bias/activation, and nullable-scale behavior on the established
     * kernels so decode and uncommon fused epilogues do not pay for this tile.
     *
     * TODO(c-native-dequant-tile): retune the 8x64 tile and M/N crossover with
     * the NPU concurrently active; these are isolated big-core defaults. */
    if (!panel_rows && M >= 32 && N >= 4096 && a_scale && w_scale && !bias &&
        act == ActivationOp::None) {
        constexpr int row_tile = 8;
        constexpr int nb_tile = 16; /* 64 output columns */
        const int full_rows = M / row_tile * row_tile;
        const int full_nb = N / 4;
        const int nb_outer_count = (full_nb + nb_tile - 1) / nb_tile;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
        for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
            const int nb0 = nb_outer * nb_tile;
            const int nb1 = std::min(nb0 + nb_tile, full_nb);
            int nb = nb0;
            for (; nb + 4 <= nb1; nb += 4) {
                const int n0 = nb * 4;
                const size_t native_nb_stride = (size_t)M * 4;
                const int32_t* const sp0 = src_native + (size_t)nb * native_nb_stride;
                const int32_t* const sp1 = sp0 + native_nb_stride;
                const int32_t* const sp2 = sp1 + native_nb_stride;
                const int32_t* const sp3 = sp2 + native_nb_stride;
                const float32x4_t ws0 = vld1q_f32(w_scale + n0 + 0);
                const float32x4_t ws1 = vld1q_f32(w_scale + n0 + 4);
                const float32x4_t ws2 = vld1q_f32(w_scale + n0 + 8);
                const float32x4_t ws3 = vld1q_f32(w_scale + n0 + 12);
                for (int r0 = 0; r0 < full_rows; r0 += row_tile) {
                    dequant_c_native_f32_8rx16_neon(
                        N, r0, n0,
                        sp0 + (size_t)r0 * 4,
                        sp1 + (size_t)r0 * 4,
                        sp2 + (size_t)r0 * 4,
                        sp3 + (size_t)r0 * 4,
                        a_scale, ws0, ws1, ws2, ws3, dst_f32);
                }
                for (int r = full_rows; r < M; ++r) {
                    dequant_c_native_f32_1rx16_neon<0>(
                        sp0 + (size_t)r * 4,
                        sp1 + (size_t)r * 4,
                        sp2 + (size_t)r * 4,
                        sp3 + (size_t)r * 4,
                        vdupq_n_f32(a_scale[r]), ws0, ws1, ws2, ws3,
                        dst_f32 + (size_t)r * N + n0);
                }
            }
            for (; nb < nb1; ++nb) {
                const int n0 = nb * 4;
                for (int r = 0; r < M; ++r) {
                    const int32_t* sp = src_native + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
                    vst1q_f32(dst_f32 + (size_t)r * N + n0,
                              dequant4_i32_f32_neon(
                                  sp, a_scale[r], w_scale, nullptr, n0,
                                  ActivationOp::None));
                }
            }
        }

        /* At most three columns remain.  Keeping this tail scalar avoids
         * padding, scratch storage, or a second model-sized output copy. */
        if (full_nb * 4 < N) {
            const int n0 = full_nb * 4;
            const int valid = N - n0;
            for (int r = 0; r < M; ++r) {
                const int32_t* sp = src_native + ((size_t)full_nb * M + r) * 4;
                float* dp = dst_f32 + (size_t)r * N + n0;
                for (int i = 0; i < valid; ++i)
                    dp[i] = (float)sp[i] * a_scale[r] * w_scale[n0 + i];
            }
        }
        return;
    }
#endif

    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = (nb0 + nb_group <= nb_count) ? (nb0 + nb_group) : nb_count;
                for (int r = r0; r < r1; ++r) {
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    float* dp = dst_f32 + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                        const int32_t* sp = src_native + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
#if defined(__aarch64__) && defined(__ARM_NEON)
                        if (valid == 4 && can_dequant4_neon(act)) {
                            vst1q_f32(dp + (nb - nb0) * 4,
                                      dequant4_i32_f32_neon(sp, as, w_scale, bias, n0, act));
                            continue;
                        }
#endif
                        for (int i = 0; i < valid; ++i) {
                            const int n = n0 + i;
                            float y = (float)sp[i] * as * (w_scale ? w_scale[n] : 1.0f);
                            if (bias) y += bias[n];
                            dp[(nb - nb0) * 4 + i] = apply_activation(y, act);
                        }
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = (nb0 + nb_block <= nb_count) ? (nb0 + nb_block) : nb_count;
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb = nb0; nb < nb1; ++nb) {
                const int32_t* sp0 = src_native + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r0, nb, panel_rows);
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    const int32_t* sp = sp0 + (r - r0) * 4;
                    float* dp = dst_f32 + (size_t)r * N + n0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                    if (valid == 4 && can_dequant4_neon(act)) {
                        vst1q_f32(dp, dequant4_i32_f32_neon(sp, as, w_scale, bias, n0, act));
                        continue;
                    }
#endif
                    for (int i = 0; i < valid; ++i) {
                        const int n = n0 + i;
                        float y = (float)sp[i] * as * (w_scale ? w_scale[n] : 1.0f);
                        if (bias) y += bias[n];
                        dp[i] = apply_activation(y, act);
                    }
                }
            }
        }
    }
}

void i8_unpack_c_native_i32_dequant_f16(int M, int N,
                                        const int32_t* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        uint16_t* dst_fp16, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = (nb0 + nb_group <= nb_count) ? (nb0 + nb_group) : nb_count;
                for (int r = r0; r < r1; ++r) {
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    uint16_t* dp = dst_fp16 + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                        const int32_t* sp = src_native + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
#if RKCPU_HAS_NEON_FP16
                        if (valid == 4 && can_dequant4_neon(act)) {
                            const float32x4_t y = dequant4_i32_f32_neon(sp, as, w_scale, bias, n0, act);
                            vst1_u16(dp + (nb - nb0) * 4, vreinterpret_u16_f16(vcvt_f16_f32(y)));
                            continue;
                        }
#endif
                        for (int i = 0; i < valid; ++i) {
                            const int n = n0 + i;
                            float y = (float)sp[i] * as * (w_scale ? w_scale[n] : 1.0f);
                            if (bias) y += bias[n];
                            dp[(nb - nb0) * 4 + i] = float_to_half(apply_activation(y, act));
                        }
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = (nb0 + nb_block <= nb_count) ? (nb0 + nb_block) : nb_count;
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = (r0 + row_block <= M) ? (r0 + row_block) : M;
            for (int nb = nb0; nb < nb1; ++nb) {
                const int32_t* sp0 = src_native + rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r0, nb, panel_rows);
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = (n0 + 4 <= N) ? 4 : (N - n0);
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    const int32_t* sp = sp0 + (r - r0) * 4;
                    uint16_t* dp = dst_fp16 + (size_t)r * N + n0;
#if RKCPU_HAS_NEON_FP16
                    if (valid == 4 && can_dequant4_neon(act)) {
                        const float32x4_t y = dequant4_i32_f32_neon(sp, as, w_scale, bias, n0, act);
                        vst1_u16(dp, vreinterpret_u16_f16(vcvt_f16_f32(y)));
                        continue;
                    }
#endif
                    for (int i = 0; i < valid; ++i) {
                        const int n = n0 + i;
                        float y = (float)sp[i] * as * (w_scale ? w_scale[n] : 1.0f);
                        if (bias) y += bias[n];
                        dp[i] = float_to_half(apply_activation(y, act));
                    }
                }
            }
        }
    }
}

void i8_accumulate_c_native_i32(int M, int N,
                                const int32_t* src_native,
                                int32_t* accum_native, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    if (M <= 0 || N <= 0 || !src_native || !accum_native) return;
    const size_t elems = (size_t)(panel_rows ? (N + 31) / 32 * 32 : (N + 3) / 4 * 4) * M;
#pragma omp parallel for schedule(static) if(split_k_accum_use_omp(elems))
    for (size_t i = 0; i < elems; i += 4) {
#if defined(__aarch64__) && defined(__ARM_NEON)
        vst1q_s32(accum_native + i,
                  vaddq_s32(vld1q_s32(accum_native + i), vld1q_s32(src_native + i)));
#else
        for (int lane = 0; lane < 4; ++lane) {
            const uint32_t sum = (uint32_t)accum_native[i + lane] +
                                 (uint32_t)src_native[i + lane];
            std::memcpy(accum_native + i + lane, &sum, sizeof(sum));
        }
#endif
    }
}

void i8_reduce_c_native_i32_dequant_f32(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        float* dst_f32, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    if (M <= 0 || N <= 0 || partial_count <= 0 || !src_native || !dst_f32) return;
    for (int p = 0; p < partial_count; ++p)
        if (!src_native[p]) return;
    if (partial_count == 1) {
        i8_unpack_c_native_i32_dequant_f32(
            M, N, src_native[0], a_scale, w_scale, bias, act, dst_f32, panel_rows);
        return;
    }

    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = std::min(nb0 + nb_group, nb_count);
                for (int r = r0; r < r1; ++r) {
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    float* dp = dst_f32 + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = std::min(4, N - n0);
                        const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
#if defined(__aarch64__) && defined(__ARM_NEON)
                        if (valid == 4 && can_dequant4_neon(act)) {
                            vst1q_f32(dp + (nb - nb0) * 4,
                                      dequant4_i32_f32_neon(
                                          reduce4_i32_neon(partial_count, src_native, offset),
                                          as, w_scale, bias, n0, act));
                            continue;
                        }
#endif
                        for (int i = 0; i < valid; ++i) {
                            const int n = n0 + i;
                            float y = (float)reduce1_i32_scalar(partial_count, src_native, offset + i)
                                    * as * (w_scale ? w_scale[n] : 1.0f);
                            if (bias) y += bias[n];
                            dp[(nb - nb0) * 4 + i] = apply_activation(y, act);
                        }
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = std::min(nb0 + nb_block, nb_count);
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb = nb0; nb < nb1; ++nb) {
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = std::min(4, N - n0);
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
                    float* dp = dst_f32 + (size_t)r * N + n0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                    if (valid == 4 && can_dequant4_neon(act)) {
                        vst1q_f32(dp, dequant4_i32_f32_neon(
                                          reduce4_i32_neon(partial_count, src_native, offset),
                                          as, w_scale, bias, n0, act));
                        continue;
                    }
#endif
                    for (int i = 0; i < valid; ++i) {
                        const int n = n0 + i;
                        float y = (float)reduce1_i32_scalar(partial_count, src_native, offset + i)
                                * as * (w_scale ? w_scale[n] : 1.0f);
                        if (bias) y += bias[n];
                        dp[i] = apply_activation(y, act);
                    }
                }
            }
        }
    }
}

void i8_reduce_c_native_i32_dequant_f16(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        uint16_t* dst_fp16, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    if (M <= 0 || N <= 0 || partial_count <= 0 || !src_native || !dst_fp16) return;
    for (int p = 0; p < partial_count; ++p)
        if (!src_native[p]) return;
    if (partial_count == 1) {
        i8_unpack_c_native_i32_dequant_f16(
            M, N, src_native[0], a_scale, w_scale, bias, act, dst_fp16, panel_rows);
        return;
    }
    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = std::min(nb0 + nb_group, nb_count);
                for (int r = r0; r < r1; ++r) {
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    uint16_t* dp = dst_fp16 + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = std::min(4, N - n0);
                        const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
#if RKCPU_HAS_NEON_FP16
                        if (valid == 4 && can_dequant4_neon(act)) {
                            const float32x4_t y = dequant4_i32_f32_neon(
                                reduce4_i32_neon(partial_count, src_native, offset),
                                as, w_scale, bias, n0, act);
                            vst1_u16(dp + (nb - nb0) * 4,
                                     vreinterpret_u16_f16(vcvt_f16_f32(y)));
                            continue;
                        }
#endif
                        for (int i = 0; i < valid; ++i) {
                            const int n = n0 + i;
                            float y = (float)reduce1_i32_scalar(partial_count, src_native, offset + i)
                                    * as * (w_scale ? w_scale[n] : 1.0f);
                            if (bias) y += bias[n];
                            dp[(nb - nb0) * 4 + i] = float_to_half(apply_activation(y, act));
                        }
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = std::min(nb0 + nb_block, nb_count);
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb = nb0; nb < nb1; ++nb) {
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = std::min(4, N - n0);
                    const float as = a_scale ? a_scale[r] : 1.0f;
                    const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
                    uint16_t* dp = dst_fp16 + (size_t)r * N + n0;
#if RKCPU_HAS_NEON_FP16
                    if (valid == 4 && can_dequant4_neon(act)) {
                        const float32x4_t y = dequant4_i32_f32_neon(
                            reduce4_i32_neon(partial_count, src_native, offset),
                            as, w_scale, bias, n0, act);
                        vst1_u16(dp, vreinterpret_u16_f16(vcvt_f16_f32(y)));
                        continue;
                    }
#endif
                    for (int i = 0; i < valid; ++i) {
                        const int n = n0 + i;
                        float y = (float)reduce1_i32_scalar(partial_count, src_native, offset + i)
                                * as * (w_scale ? w_scale[n] : 1.0f);
                        if (bias) y += bias[n];
                        dp[i] = float_to_half(apply_activation(y, act));
                    }
                }
            }
        }
    }
}

void i8_reduce_c_native_i32_to_rowmajor(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        int32_t* dst_i32, int panel_rows) {
    if (panel_rows && ((panel_rows != 8 && panel_rows != 16) || M % panel_rows)) return;
    if (M <= 0 || N <= 0 || partial_count <= 0 || !src_native || !dst_i32)
        return;
    for (int p = 0; p < partial_count; ++p)
        if (!src_native[p]) return;

    const int row_block = panel_rows ? panel_rows : 16;
    constexpr int nb_group = 8;
    const int nb_count = (N + 3) / 4;
    const int rb_count = (M + row_block - 1) / row_block;

    if (N < 4096) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb0 = 0; nb0 < nb_count; nb0 += nb_group) {
                const int nb1 = std::min(nb0 + nb_group, nb_count);
                for (int r = r0; r < r1; ++r) {
                    int32_t* dp = dst_i32 + (size_t)r * N + nb0 * 4;
                    for (int nb = nb0; nb < nb1; ++nb) {
                        const int n0 = nb * 4;
                        const int valid = std::min(4, N - n0);
                        const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
#if defined(__aarch64__) && defined(__ARM_NEON)
                        if (valid == 4) {
                            vst1q_s32(dp + (nb - nb0) * 4,
                                      reduce4_i32_neon(partial_count, src_native, offset));
                            continue;
                        }
#endif
                        for (int i = 0; i < valid; ++i)
                            dp[(nb - nb0) * 4 + i] =
                                reduce1_i32_scalar(partial_count, src_native, offset + i);
                    }
                }
            }
        }
        return;
    }

    constexpr int nb_block = 32;
    const int nb_outer_count = (nb_count + nb_block - 1) / nb_block;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= kSplitKReduceParallelElems)
    for (int nb_outer = 0; nb_outer < nb_outer_count; ++nb_outer) {
        const int nb0 = nb_outer * nb_block;
        const int nb1 = std::min(nb0 + nb_block, nb_count);
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int nb = nb0; nb < nb1; ++nb) {
                for (int r = r0; r < r1; ++r) {
                    const int n0 = nb * 4;
                    const int valid = std::min(4, N - n0);
                    const size_t offset = rknpu2_matmul_open::cpu::i8_c_block_offset(M, N, r, nb, panel_rows);
                    int32_t* dp = dst_i32 + (size_t)r * N + n0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                    if (valid == 4) {
                        vst1q_s32(dp, reduce4_i32_neon(partial_count, src_native, offset));
                        continue;
                    }
#endif
                    for (int i = 0; i < valid; ++i)
                        dp[i] = reduce1_i32_scalar(partial_count, src_native, offset + i);
                }
            }
        }
    }
}

void f16_pack_a_normal(int M, int K, int align_in, const uint16_t* src, uint16_t* dst) {
    if (K == align_in) {
        std::memcpy(dst, src, (size_t)M * K * sizeof(uint16_t));
        return;
    }
    const uint64_t elems = (uint64_t)M * align_in;
    zero_n(dst, elems);
#pragma omp parallel for schedule(static) if(elems >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        std::memcpy(dst + (size_t)r * align_in, src + (size_t)r * K, (size_t)K * sizeof(uint16_t));
    }
}

void f16_pack_a_normal_batch(int B, int M, int K, int align_in,
                             const uint16_t* src, uint16_t* dst) {
    if (B <= 0 || M <= 0 || K <= 0 || align_in < K || !src || !dst) return;
    const int64_t rows = (int64_t)B * M;
    const uint64_t elems = (uint64_t)rows * align_in;
#pragma omp parallel for schedule(static) if(elems >= (1u << 18))
    for (int64_t row = 0; row < rows; ++row) {
        uint16_t* out = dst + (size_t)row * align_in;
        std::memcpy(out, src + (size_t)row * K,
                    (size_t)K * sizeof(uint16_t));
        if (K < align_in)
            std::memset(out + K, 0,
                        (size_t)(align_in - K) * sizeof(uint16_t));
    }
}

void f16_pack_a_normal_split(int M, int K, int split_count, int part_k,
                             const uint16_t* src, uint16_t* dst) {
    if (M <= 0 || K <= 0 || split_count <= 0 || part_k <= 0 || !src || !dst ||
        (uint64_t)split_count * part_k < (uint64_t)K)
        return;
    const uint64_t elems = (uint64_t)M * K;
    const uint64_t split_stride = (uint64_t)M * part_k;

    /* One row-parallel team copies every K slice.  Besides avoiding one OMP
     * launch per slice, this consumes the compact source row in increasing K
     * order while it is hot.  Destinations remain split-major as required by
     * the batch register-command builder. */
#pragma omp parallel for schedule(static) if(elems >= (1u << 14))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * K;
        for (int split = 0; split < split_count; ++split) {
            const int k0 = split * part_k;
            const int valid = std::min(part_k, std::max(0, K - k0));
            uint16_t* out = dst + (uint64_t)split * split_stride +
                                  (size_t)r * part_k;
            if (valid > 0)
                std::memcpy(out, row + k0,
                            (size_t)valid * sizeof(uint16_t));
            if (valid < part_k)
                std::memset(out + valid, 0,
                            (size_t)(part_k - valid) * sizeof(uint16_t));
        }
    }
}

void f16_pack_a_native_k8_m8(int M, int K, int align_in,
                             const uint16_t* src, uint16_t* dst) {
    constexpr int subK = 8;
    const int kb_count = align_in / subK;
    const uint64_t elems = (uint64_t)kb_count * M * subK;
    if (K != align_in) zero_n(dst, elems);

    /* Keep a small row slab hot while permuting the outer M/K8 dimensions.
     * The old surface-major loop revisited each large source row once per K8
     * block, after it had fallen out of cache.  Eight rows match the 8x8 NEON
     * blocking used by Whisper_cpp_rknpu2's historical native-B kernel while
     * still writing two full cache lines per destination surface. */
    if (align_in >= 512 && M >= 8) {
        constexpr int row_block = 8;
        const int rb_count = (M + row_block - 1) / row_block;
#pragma omp parallel for schedule(static) if(elems >= (1u << 19))
        for (int rb = 0; rb < rb_count; ++rb) {
            const int r0 = rb * row_block;
            const int r1 = std::min(r0 + row_block, M);
            for (int kb = 0; kb < kb_count; ++kb) {
                const int k0 = kb * subK;
                const int valid = (k0 + subK <= K)
                                ? subK : (K > k0 ? K - k0 : 0);
                if (valid <= 0) continue;
                for (int r = r0; r < r1; ++r) {
                    uint16_t* out = dst + ((size_t)kb * M + r) * subK;
                    const uint16_t* in = src + (size_t)r * K + k0;
                    if (valid == subK)
                        copy16_u8((uint8_t*)out, (const uint8_t*)in);
                    else
                        std::memcpy(out, in,
                                    (size_t)valid * sizeof(uint16_t));
                }
            }
        }
        return;
    }
#pragma omp parallel for schedule(static) if(elems >= (1u << 19))
    for (int kb = 0; kb < kb_count; ++kb) {
        const int k0 = kb * subK;
        const int valid = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
        if (valid <= 0) continue;
        for (int r = 0; r < M; ++r) {
            uint16_t* out = dst + ((size_t)kb * M + r) * subK;
            const uint16_t* in = src + (size_t)r * K + k0;
            if (valid == subK)
                copy16_u8((uint8_t*)out, (const uint8_t*)in);
            else
                std::memcpy(out, in, (size_t)valid * sizeof(uint16_t));
        }
    }
}

void f16_pack_a_native_k8_m8_batch(int B, int M, int K, int align_in,
                                   const uint16_t* src, uint16_t* dst) {
    if (B <= 0 || M <= 0 || K <= 0 || align_in < K || !src || !dst) return;
    const uint64_t one_elems = (uint64_t)M * align_in;
#pragma omp parallel for schedule(static) if((uint64_t)B * one_elems >= (1u << 18))
    for (int b = 0; b < B; ++b)
        f16_pack_a_native_k8_m8(
            M, K, align_in, src + (uint64_t)b * M * K,
            dst + (uint64_t)b * one_elems);
}

void f16_pack_b_native_n16_k32(int K, int N, int align_in, int align_out,
                               const uint16_t* src, uint16_t* dst) {
    constexpr int subN = 16;
    constexpr int subK = 32;
    const int nb_count = align_out / subN;
    const int kb_count = align_in / subK;
    const int total_tiles = nb_count * kb_count;
    const bool full_tiles = (K == align_in && N == align_out);
    if (!full_tiles) zero_n(dst, (uint64_t)align_out * align_in);
#pragma omp parallel for schedule(static) if((uint64_t)align_out * align_in >= (1u << 14))
    for (int tile = 0; tile < total_tiles; ++tile) {
        const int nb = tile / kb_count;
        const int kb = tile - nb * kb_count;
        const int n0 = nb * subN;
        const int k0 = kb * subK;
        uint16_t* tile_dst = dst + ((size_t)nb * kb_count + kb) * subN * subK;
        const uint16_t* tile_src = src + (size_t)k0 * N + n0;
        if (full_tiles) {
            transpose16bit_32x16(tile_dst, tile_src, N);
            continue;
        }
        const int valid_n = (n0 + subN <= N) ? subN : (N > n0 ? N - n0 : 0);
        const int valid_k = (k0 + subK <= K) ? subK : (K > k0 ? K - k0 : 0);
        if (valid_n <= 0 || valid_k <= 0) continue;
        transpose16bit_tile(tile_dst, tile_src, valid_k, valid_n, N, subK);
    }
}

void f16_pack_b_native_n16_k32_batch(int B, int K, int N,
                                     int align_in, int align_out,
                                     const uint16_t* src, uint16_t* dst) {
    if (B <= 0 || K <= 0 || N <= 0 || align_in < K || align_out < N ||
        (align_in % 32) != 0 || (align_out % 16) != 0 || !src || !dst)
        return;
    constexpr int subN = 16;
    constexpr int subK = 32;
    const int nb_count = align_out / subN;
    const int kb_count = align_in / subK;
    const int tiles_per_batch = nb_count * kb_count;
    const int64_t total_tiles = (int64_t)B * tiles_per_batch;
    const uint64_t src_batch_stride = (uint64_t)K * N;
    const uint64_t dst_batch_stride = (uint64_t)align_out * align_in;
    const bool full_tiles = (K == align_in && N == align_out);
    if (!full_tiles) zero_n(dst, (uint64_t)B * dst_batch_stride);

    /* One team covers the complete batch.  Calling the single-matrix kernel B
     * times repeatedly entered OpenMP for attention's many small head weights;
     * flattening [batch, N16, K32] keeps exactly the same packed layout. */
#pragma omp parallel for schedule(static) if((uint64_t)B * dst_batch_stride >= (1u << 14))
    for (int64_t batch_tile = 0; batch_tile < total_tiles; ++batch_tile) {
        const int b = (int)(batch_tile / tiles_per_batch);
        const int tile = (int)(batch_tile - (int64_t)b * tiles_per_batch);
        const int nb = tile / kb_count;
        const int kb = tile - nb * kb_count;
        const int n0 = nb * subN;
        const int k0 = kb * subK;
        uint16_t* tile_dst = dst + (uint64_t)b * dst_batch_stride +
            (size_t)tile * subN * subK;
        const uint16_t* tile_src = src + (uint64_t)b * src_batch_stride +
            (size_t)k0 * N + n0;
        if (full_tiles) {
            transpose16bit_32x16(tile_dst, tile_src, N);
            continue;
        }
        const int valid_n = (n0 + subN <= N)
                          ? subN : (N > n0 ? N - n0 : 0);
        const int valid_k = (k0 + subK <= K)
                          ? subK : (K > k0 ? K - k0 : 0);
        if (valid_n <= 0 || valid_k <= 0) continue;
        transpose16bit_tile(tile_dst, tile_src, valid_k, valid_n, N, subK);
    }
}

void f16_pack_operand_half_surface(int M, int N, int align_out, int m_tile,
                                   const uint16_t* src, uint16_t* dst) {
    uint64_t total = 0;
    for (int start = 0; start < M; start += m_tile) {
        const int tile_m = (start + m_tile <= M) ? m_tile : (M - start);
        total += (uint64_t)(align_out / 16) * 24 * tile_m;
    }
    zero_n(dst, total);

    uint64_t block_off = 0;
    for (int start = 0; start < M; start += m_tile) {
        const int tile_m = (start + m_tile <= M) ? m_tile : (M - start);
#pragma omp parallel for collapse(2) schedule(static) if((uint64_t)tile_m * N >= (1u << 13))
        for (int r = 0; r < tile_m; ++r) {
            for (int c = 0; c < N; ++c) {
                const int b = c >> 3;
                const uint64_t idx = (uint64_t)((b + 1) / 2) * (16 * tile_m)
                                   + (uint64_t)(b / 2) * (8 * tile_m)
                                   + (uint64_t)r * 8 + (c & 7);
                dst[block_off + idx] = src[(size_t)(start + r) * N + c];
            }
        }
        block_off += (uint64_t)(align_out / 16) * 24 * tile_m;
    }
}

void f16_unpack_d(int M, int N, int align_out, const uint16_t* src, uint16_t* dst) {
    const int row_stride = align_out * 2;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * row_stride;
        uint16_t* out = dst + (size_t)r * N;
        int c = 0;
        for (; c + 16 <= N; c += 16)
            std::memcpy(out + c, row + (c / 16) * 32, 16 * sizeof(uint16_t));
        for (; c < N; ++c)
            out[c] = row[(c / 16) * 32 + (c % 16)];
    }
}

void f16_unpack_d_strided(int M, int N, int align_out,
                          const uint16_t* src,
                          int dst_row_stride, int dst_col_offset,
                          uint16_t* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst ||
        dst_row_stride < dst_col_offset + N || dst_col_offset < 0)
        return;
    const int src_row_stride = align_out * 2;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * src_row_stride;
        uint16_t* out = dst + (size_t)r * dst_row_stride + dst_col_offset;
        int c = 0;
        for (; c + 16 <= N; c += 16)
            std::memcpy(out + c, row + (c / 16) * 32,
                        16 * sizeof(uint16_t));
        for (; c < N; ++c)
            out[c] = row[(c / 16) * 32 + (c % 16)];
    }
}

void f16_unpack_d_f32(int M, int N, int align_out, const uint16_t* src, float* dst) {
    const int row_stride = align_out * 2;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * row_stride;
        float* out = dst + (size_t)r * N;
        for (int c = 0; c < N; ++c)
            out[c] = half_to_float(row[(c / 16) * 32 + (c % 16)]);
    }
}

void f16_unpack_d_f32_strided(int M, int N, int align_out,
                              const uint16_t* src,
                              int dst_row_stride, int dst_col_offset,
                              float* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst ||
        dst_row_stride < dst_col_offset + N || dst_col_offset < 0)
        return;
    const int src_row_stride = align_out * 2;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * src_row_stride;
        float* out = dst + (size_t)r * dst_row_stride + dst_col_offset;
        for (int c = 0; c < N; ++c)
            out[c] = half_to_float(row[(c / 16) * 32 + (c % 16)]);
    }
}

void f16_unpack_d_compact(int M, int N, int align_out,
                          const uint16_t* src, uint16_t* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst) return;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * N,
                    src + (size_t)r * align_out,
                    (size_t)N * sizeof(uint16_t));
}

void f16_unpack_d_compact_add_bias(int M, int N, int align_out,
                                   const uint16_t* src,
                                   const uint16_t* bias,
                                   uint16_t* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !bias || !dst) return;
    const uint64_t elems = (uint64_t)M * N;
#pragma omp parallel for schedule(static) if(elems >= (1u << 16))
    for (int r = 0; r < M; ++r) {
        const uint16_t* in = src + (size_t)r * align_out;
        uint16_t* out = dst + (size_t)r * N;
        int c = 0;
#if RKCPU_HAS_NEON_FP16
        for (; c + 8 <= N; c += 8) {
            const float16x8_t value =
                vreinterpretq_f16_u16(vld1q_u16(in + c));
            const float16x8_t offset =
                vreinterpretq_f16_u16(vld1q_u16(bias + c));
            vst1q_u16(out + c,
                      vreinterpretq_u16_f16(vaddq_f16(value, offset)));
        }
#endif
        for (; c < N; ++c)
            out[c] = float_to_half(half_to_float(in[c]) +
                                   half_to_float(bias[c]));
    }
}

void f16_unpack_d_compact_batch(int B, int M, int N, int align_out,
                                uint64_t src_batch_stride,
                                const uint16_t* src, uint16_t* dst) {
    if (B <= 0 || M <= 0 || N <= 0 || align_out < N || !src || !dst ||
        src_batch_stride < (uint64_t)M * align_out)
        return;
    const int64_t rows = (int64_t)B * M;
#pragma omp parallel for schedule(static) if((uint64_t)rows * N >= (1u << 18))
    for (int64_t br = 0; br < rows; ++br) {
        const int b = (int)(br / M);
        const int r = (int)(br - (int64_t)b * M);
        std::memcpy(dst + (size_t)br * N,
                    src + (uint64_t)b * src_batch_stride +
                          (size_t)r * align_out,
                    (size_t)N * sizeof(uint16_t));
    }
}

void f16_unpack_d_compact_strided(int M, int N, int align_out,
                                  const uint16_t* src,
                                  int dst_row_stride, int dst_col_offset,
                                  uint16_t* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst ||
        dst_col_offset < 0 || dst_row_stride < dst_col_offset + N)
        return;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * dst_row_stride + dst_col_offset,
                    src + (size_t)r * align_out,
                    (size_t)N * sizeof(uint16_t));
}

void f16_unpack_d_f32_compact(int M, int N, int align_out,
                              const uint16_t* src, float* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst) return;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            dst[(size_t)r * N + c] =
                half_to_float(src[(size_t)r * align_out + c]);
}

void f16_unpack_d_f32_compact_strided(int M, int N, int align_out,
                                      const uint16_t* src,
                                      int dst_row_stride, int dst_col_offset,
                                      float* dst) {
    if (M <= 0 || N <= 0 || align_out < N || !src || !dst ||
        dst_col_offset < 0 || dst_row_stride < dst_col_offset + N)
        return;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 19))
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            dst[(size_t)r * dst_row_stride + dst_col_offset + c] =
                half_to_float(src[(size_t)r * align_out + c]);
}

void f16_unpack_d_native_n8_m8(int M, int N,
                               const uint16_t* src, uint16_t* dst) {
    f16_unpack_d_native_n8_m8_batch(
        1, M, N, (uint64_t)M * N, src, dst);
}

void f16_unpack_d_native_n8_m8_batch(int B, int M, int N,
                                     uint64_t src_batch_stride,
                                     const uint16_t* src, uint16_t* dst) {
    if (B <= 0 || M <= 0 || N <= 0 || !src || !dst ||
        src_batch_stride < (uint64_t)M * N)
        return;
    constexpr int subN = 8;
    const int nb_count = (N + subN - 1) / subN;
    const int full_nb = N / subN;
    constexpr int row_block = 4;
    const int rb_per_batch = (M + row_block - 1) / row_block;
    const int64_t blocks = (int64_t)B * rb_per_batch;
#pragma omp parallel for schedule(static) if((uint64_t)B * M * N >= (1u << 18))
    for (int64_t block = 0; block < blocks; ++block) {
        const int b = (int)(block / rb_per_batch);
        const int rb = (int)(block - (int64_t)b * rb_per_batch);
        const int r0 = rb * row_block;
        const int rows = std::min(row_block, M - r0);
        const uint16_t* batch_src = src + (uint64_t)b * src_batch_stride;
        uint16_t* out = dst + ((size_t)b * M + r0) * N;
        int nb = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (rows == row_block) {
            for (; nb < full_nb; ++nb) {
                const uint16_t* in =
                    batch_src + ((size_t)nb * M + r0) * subN;
                const uint16x8x4_t value = vld1q_u16_x4(in);
                vst1q_u16(out + (size_t)0 * N + nb * subN,
                          value.val[0]);
                vst1q_u16(out + (size_t)1 * N + nb * subN,
                          value.val[1]);
                vst1q_u16(out + (size_t)2 * N + nb * subN,
                          value.val[2]);
                vst1q_u16(out + (size_t)3 * N + nb * subN,
                          value.val[3]);
            }
        }
#endif
        for (; nb < full_nb; ++nb) {
            for (int row = 0; row < rows; ++row) {
                const uint16_t* in = batch_src +
                    ((size_t)nb * M + r0 + row) * subN;
#if defined(__aarch64__) && defined(__ARM_NEON)
                vst1q_u16(out + (size_t)row * N + nb * subN,
                          vld1q_u16(in));
#else
                std::memcpy(out + (size_t)row * N + nb * subN,
                            in, subN * sizeof(uint16_t));
#endif
            }
        }
        for (; nb < nb_count; ++nb) {
            const int n0 = nb * subN;
            const int valid = std::min(subN, N - n0);
            for (int row = 0; row < rows; ++row)
                std::memcpy(out + (size_t)row * N + n0,
                            batch_src +
                                ((size_t)nb * M + r0 + row) * subN,
                            (size_t)valid * sizeof(uint16_t));
        }
    }
}

void f16_unpack_d_f32_native_n8_m8(int M, int N,
                                   const uint16_t* src, float* dst) {
    if (M <= 0 || N <= 0 || !src || !dst) return;
    constexpr int subN = 8;
    const int nb_count = (N + subN - 1) / subN;
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 18))
    for (int r = 0; r < M; ++r) {
        float* out = dst + (size_t)r * N;
        for (int nb = 0; nb < nb_count; ++nb) {
            const int n0 = nb * subN;
            const int valid = std::min(subN, N - n0);
            const uint16_t* in = src + ((size_t)nb * M + r) * subN;
            for (int i = 0; i < valid; ++i)
                out[n0 + i] = half_to_float(in[i]);
        }
    }
}

static void f16_reduce_d_compact_split_f16_impl(
    int M, int N, int align_out, int split_count,
    uint64_t src_split_stride, const uint16_t* src,
    const uint16_t* bias, uint16_t* dst) {
    if (M <= 0 || N <= 0 || align_out < N || split_count <= 0 || !src || !dst ||
        src_split_stride < (uint64_t)M * align_out)
        return;
    const uint64_t elems = (uint64_t)M * N;
#pragma omp parallel for schedule(static) if(elems >= (1u << 16))
    for (int r = 0; r < M; ++r) {
        uint16_t* out = dst + (size_t)r * N;
        int c = 0;
#if RKCPU_HAS_NEON_FP16
        for (; c + 8 <= N; c += 8) {
            const uint16_t* in = src + (size_t)r * align_out + c;
            const float16x8_t h0 =
                vreinterpretq_f16_u16(vld1q_u16(in));
            float32x4_t sum_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t sum_hi = vcvt_f32_f16(vget_high_f16(h0));
            for (int split = 1; split < split_count; ++split) {
                const float16x8_t hp = vreinterpretq_f16_u16(vld1q_u16(
                    in + (uint64_t)split * src_split_stride));
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(hp)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(hp)));
            }
            if (bias) {
                const float16x8_t hb =
                    vreinterpretq_f16_u16(vld1q_u16(bias + c));
                sum_lo = vaddq_f32(sum_lo,
                                   vcvt_f32_f16(vget_low_f16(hb)));
                sum_hi = vaddq_f32(sum_hi,
                                   vcvt_f32_f16(vget_high_f16(hb)));
            }
            const float16x8_t result = vcombine_f16(
                vcvt_f16_f32(sum_lo), vcvt_f16_f32(sum_hi));
            vst1q_u16(out + c, vreinterpretq_u16_f16(result));
        }
#endif
        for (; c < N; ++c) {
            float sum = 0.0f;
            const uint16_t* in = src + (size_t)r * align_out + c;
            for (int split = 0; split < split_count; ++split)
                sum += half_to_float(in[(uint64_t)split * src_split_stride]);
            if (bias) sum += half_to_float(bias[c]);
            out[c] = float_to_half(sum);
        }
    }
}

void f16_reduce_d_compact_split_f16(int M, int N, int align_out,
                                    int split_count,
                                    uint64_t src_split_stride,
                                    const uint16_t* src, uint16_t* dst) {
    f16_reduce_d_compact_split_f16_impl(
        M, N, align_out, split_count, src_split_stride,
        src, nullptr, dst);
}

void f16_reduce_d_compact_split_f16_add_bias(
    int M, int N, int align_out, int split_count,
    uint64_t src_split_stride, const uint16_t* src,
    const uint16_t* bias, uint16_t* dst) {
    if (!bias) return;
    f16_reduce_d_compact_split_f16_impl(
        M, N, align_out, split_count, src_split_stride,
        src, bias, dst);
}

void f16_reduce_d_compact_split_f32(int M, int N, int align_out,
                                    int split_count,
                                    uint64_t src_split_stride,
                                    const uint16_t* src, float* dst) {
    if (M <= 0 || N <= 0 || align_out < N || split_count <= 0 || !src || !dst ||
        src_split_stride < (uint64_t)M * align_out)
        return;
    const uint64_t elems = (uint64_t)M * N;
#pragma omp parallel for schedule(static) if(elems >= (1u << 16))
    for (int r = 0; r < M; ++r) {
        float* out = dst + (size_t)r * N;
        int c = 0;
#if RKCPU_HAS_NEON_FP16
        for (; c + 8 <= N; c += 8) {
            const uint16_t* in = src + (size_t)r * align_out + c;
            const float16x8_t h0 =
                vreinterpretq_f16_u16(vld1q_u16(in));
            float32x4_t sum_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t sum_hi = vcvt_f32_f16(vget_high_f16(h0));
            for (int split = 1; split < split_count; ++split) {
                const float16x8_t hp = vreinterpretq_f16_u16(vld1q_u16(
                    in + (uint64_t)split * src_split_stride));
                sum_lo = vaddq_f32(sum_lo, vcvt_f32_f16(vget_low_f16(hp)));
                sum_hi = vaddq_f32(sum_hi, vcvt_f32_f16(vget_high_f16(hp)));
            }
            vst1q_f32(out + c, sum_lo);
            vst1q_f32(out + c + 4, sum_hi);
        }
#endif
        for (; c < N; ++c) {
            float sum = 0.0f;
            const uint16_t* in = src + (size_t)r * align_out + c;
            for (int split = 0; split < split_count; ++split)
                sum += half_to_float(in[(uint64_t)split * src_split_stride]);
            out[c] = sum;
        }
    }
}

} /* namespace rknpu2_matmul_open::cpu */

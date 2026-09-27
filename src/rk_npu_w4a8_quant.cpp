#include "rk_npu_w4a8_cpu.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail {
namespace {
inline float value(float x) {
    return x;
}
inline float value(uint16_t x) {
    return rknpu2_matmul_open::bits::half_to_float(x);
}
#if defined(__aarch64__)
inline float32x4_t load4(const float *p) {
    return vld1q_f32(p);
}
inline float32x4_t load4(const uint16_t *p) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
#else
    const float v[4] = {value(p[0]), value(p[1]), value(p[2]), value(p[3])};
    return vld1q_f32(v);
#endif
}
template <class T> int8x16_t quant16(const T *p, float inv) {
    const auto scale = vdupq_n_f32(inv);
    const auto lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    // Keep all four groups in SIMD registers. GCC -O2 otherwise emits a
    // counted loop that spills the int32x4_t array to the stack for every 16 A.
    const auto quant4 = [&](const T* q) {
        return vqmovn_s32(vmaxq_s32(lo, vminq_s32(hi,
            vcvtaq_s32_f32(vmulq_f32(load4(q), scale)))));
    };
    const auto q0 = quant4(p), q1 = quant4(p + 4);
    const auto q2 = quant4(p + 8), q3 = quant4(p + 12);
    return vcombine_s8(vqmovn_s16(vcombine_s16(q0, q1)),
                       vqmovn_s16(vcombine_s16(q2, q3)));
}
void pack16(uint8x16_t nibbles, uint8_t *dst) {
    const auto even = vget_low_u8(vuzp1q_u8(nibbles, nibbles));
    const auto odd = vget_low_u8(vuzp2q_u8(nibbles, nibbles));
    vst1_u8(dst, vorr_u8(vshl_n_u8(even, 4), odd));
}
#endif

template <class T>
int scales_impl(const T *A, int M, int K, int threads, float *scales, float *inverse) {
    int bad = 0;
#pragma omp parallel for num_threads(threads)                                                      \
    reduction(| : bad) if (threads > 1 && int64_t(M) * K >= 8192)
    for (int m = 0; m < M; ++m) {
        const T *row = A + size_t(m) * K;
        float maximum = 0;
        int k = 0;
#if defined(__aarch64__)
        auto vmax = vdupq_n_f32(0);
        auto invalid = vdupq_n_u32(0);
        for (; k + 4 <= K; k += 4) {
            const auto x = load4(row + k);
            const auto bits = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x7fffffff));
            invalid = vorrq_u32(invalid, vcgeq_u32(bits, vdupq_n_u32(0x7f800000)));
            vmax = vmaxq_f32(vmax, vabsq_f32(x));
        }
        if (vmaxvq_u32(invalid))
            bad = 1;
        maximum = vmaxvq_f32(vmax);
#endif
        for (; k < K; ++k) {
            const float x = value(row[k]);
            if (!std::isfinite(x))
                bad = 1;
            maximum = std::max(maximum, std::fabs(x));
        }
        const float s = maximum == 0 ? 1.0f : std::max(maximum / 127.0f, FLT_MIN);
        scales[m] = s;
        inverse[m] = 1.0f / s;
    }
    return bad ? RK_NPU_ERR_PARAM : RK_NPU_OK;
}

template <class T> int pack_impl(const W4A8Input &input, const rk_npu_i4_input_tile &t) {
    const auto *A = static_cast<const T *>(input.A);
    // Four tokens per block keep source rows local while writing complete
    // 128-byte native blocks. Workers never share destination cache lines.
    // The executor has already zeroed the buffer; the standalone helper also
    // defines padding, without clearing the full buffer a second time.
    const int blocks = (t.rows + 7) / 8;
#pragma omp parallel for num_threads(input.threads) schedule(static) \
    if(input.threads > 1 && blocks > 1 && int64_t(t.rows) * t.padded_k >= 8192)
    for (int block = 0; block < blocks; ++block) {
      const int r0 = block * 8, rend = std::min(r0 + 8, t.rows);
      for (int kb = 0; kb < t.padded_k; kb += 32) {
        const int valid = std::min(32, std::max(0, t.k - kb));
        for (int r = r0; r < rend; r += 2) {
            const int m = (t.m0 + r) / 2;
            const T *src = A + size_t(m) * input.K + t.k0 + kb;
            auto *high = input.layout == I4InputLayout::Panel8 && t.rows >= 8
                ? t.dst + size_t(r / 8) * 4 * t.padded_k + (size_t(kb / 32) * 8 + r % 8) * 16
                : t.dst + (size_t(kb / 32) * t.rows + r) * 16;
            auto *low = high + 16;
            if (valid != 32) {
                std::memset(high, 0, 16);
                std::memset(low, 0, 16);
            }
            int j = 0;
#if defined(__aarch64__)
            for (; j + 16 <= valid; j += 16) {
                const auto q = vreinterpretq_u8_s8(quant16(src + j, input.inverse[m]));
                pack16(vshrq_n_u8(q, 4), high + j / 2);
                pack16(veorq_u8(vandq_u8(q, vdupq_n_u8(15)), vdupq_n_u8(8)), low + j / 2);
            }
#endif
            for (; j < valid; ++j) {
                const int code = int(std::max(
                    -127.0f, std::min(127.0f, std::round(value(src[j]) * input.inverse[m]))));
                const uint8_t q = uint8_t(code), h = q >> 4, l = (q & 15) ^ 8;
                if (j & 1) {
                    high[j / 2] |= h;
                    low[j / 2] |= l;
                } else {
                    high[j / 2] = h << 4;
                    low[j / 2] = l << 4;
                }
            }
        }
      }
    }
    return RK_NPU_OK;
}
} // namespace

int w4a8_scales(const void *A, bool half, int M, int K, int threads, float *scales,
                float *inverse) {
    if (!A || M < 1 || K < 1 || threads < 1 || threads > 4 || !scales || !inverse)
        return RK_NPU_ERR_PARAM;
    return half ? scales_impl(static_cast<const uint16_t *>(A), M, K, threads, scales, inverse)
                : scales_impl(static_cast<const float *>(A), M, K, threads, scales, inverse);
}
int w4a8_pack(void *user, const rk_npu_i4_input_tile *t) {
    if (!user || !t)
        return RK_NPU_ERR_PARAM;
    const auto &input = *static_cast<const W4A8Input *>(user);
    if ((input.layout != I4InputLayout::Native && input.layout != I4InputLayout::Panel8) ||
        (input.layout == I4InputLayout::Panel8 && t->rows >= 8 && t->rows % 8) || !input.A || !input.scales || !input.inverse || !t->dst || t->m0 < 0 || t->m0 % 2 ||
        t->rows < 2 || t->rows > (input.layout == I4InputLayout::Panel8 ? 512 : 128) || t->rows % 2 ||
        int64_t(t->m0) + t->rows > int64_t(input.M) * 2 || t->k0 < 0 || t->k < 1 ||
        int64_t(t->k0) + t->k > input.K || t->padded_k < t->k || t->padded_k > I4_K_TILE_MAX ||
        t->padded_k % 32 || t->bytes < uint64_t(t->rows) * t->padded_k / 2 ||
        input.threads < 1 || input.threads > 4)
        return RK_NPU_ERR_PARAM;
    return input.half ? pack_impl<uint16_t>(input, *t) : pack_impl<float>(input, *t);
}
} // namespace rknpu2_matmul_open::detail

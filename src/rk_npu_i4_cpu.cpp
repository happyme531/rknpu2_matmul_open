#include "rk_npu_i4_cpu.h"
#include "rk_npu_i4_bounds.h"
#include <algorithm>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
int pack_row(const int8_t* src, int count, uint8_t* dst) {
    int j = 0;
#if defined(__aarch64__)
    const uint8x16_t mask = vdupq_n_u8(15);
    for (; j + 16 <= count; j += 16) {
        const int8x16_t x = vld1q_s8(src + j);
        if (vminvq_s8(x) < -8 || vmaxvq_s8(x) > 7) return RK_NPU_ERR_PARAM;
        const uint8x16_t q = vandq_u8(vreinterpretq_u8_s8(x), mask);
        const uint8x8_t even = vget_low_u8(vuzp1q_u8(q, q));
        const uint8x8_t odd = vget_low_u8(vuzp2q_u8(q, q));
        vst1_u8(dst + j / 2, vorr_u8(vshl_n_u8(even, 4), odd));
    }
#endif
    for (; j < count; ++j) {
        const int v = src[j];
        if (v < -8 || v > 7) return RK_NPU_ERR_PARAM;
        if (j & 1) dst[j / 2] |= uint8_t(v) & 15;
        else dst[j / 2] = uint8_t((uint8_t(v) & 15) << 4);
    }
    return RK_NPU_OK;
}
}

extern "C" int rk_npu_i4_pack_a_tile(const rk_npu_i4_input_tile* t,
                                       const int8_t* A, int lda) {
    if (!t || !A || !t->dst || t->m0 < 0 || t->rows <= 0 || t->rows > 128 ||
        t->k0 < 0 || t->k <= 0 || t->padded_k < t->k || t->padded_k > rknpu2_matmul_open::detail::I4_K_TILE_MAX ||
        t->padded_k % 32 || int64_t(t->k0) + t->k > lda ||
        t->bytes < uint64_t(t->rows) * t->padded_k / 2) return RK_NPU_ERR_PARAM;
    std::memset(t->dst, 0, size_t(t->rows) * t->padded_k / 2);
    for (int kb = 0; kb < t->padded_k; kb += 32) {
        const int valid = std::min(32, std::max(0, t->k - kb));
        if (!valid) continue;
        for (int m = 0; m < t->rows; ++m) {
            const int rc = pack_row(A + (size_t(t->m0) + m) * lda + t->k0 + kb,
                valid, t->dst + (size_t(kb / 32) * t->rows + m) * 16);
            if (rc) return rc;
        }
    }
    return RK_NPU_OK;
}

namespace rknpu2_matmul_open::detail {
int pack_i4_weights(int K, int N, int k0, int k, int pk, int pn,
                    const int8_t* B, uint8_t* dst) {
    if (!B || !dst || K <= 0 || N <= 0 || k0 < 0 || k < 1 || k > pk ||
        int64_t(k0) + k > K || pk % 32 || pn % 64 || pn < N) return RK_NPU_ERR_PARAM;
    std::memset(dst, 0, size_t(pk) * pn / 2);
    for (int nb = 0; nb < pn; nb += 64)
        for (int kb = 0; kb < pk; kb += 32)
            for (int n = nb; n < std::min(nb + 64, N); ++n) {
                auto* row = dst + (((size_t(nb / 64) * (pk / 32) + kb / 32) * 64 + n - nb) * 16);
                for (int j = 0; j < std::min(32, k - kb); ++j) {
                    const int v = B[size_t(k0 + kb + j) * N + n];
                    if (v < -8 || v > 7) return RK_NPU_ERR_PARAM;
                    if (j & 1) row[j / 2] |= uint8_t(v) & 15;
                    else row[j / 2] = uint8_t((uint8_t(v) & 15) << 4);
                }
            }
    return RK_NPU_OK;
}

void reduce_i4_partial(int rows, int n, int /*padded_n*/, const int16_t* src,
                       int32_t* dst, int ldc, bool add) {
    // Keep a pair of compact output rows hot. Walking every row for each N8
    // atom makes power-of-two LLM row strides thrash the CPU cache sets.
    for (int m0 = 0; m0 < rows; m0 += 2) {
      for (int nb = 0; nb < n; nb += 8) {
        const int count = std::min(8, n - nb);
        for (int m = m0; m < std::min(m0 + 2, rows); ++m) {
            const auto* in = src + (size_t(nb / 8) * rows + m) * 8;
            auto* out = dst + size_t(m) * ldc + nb;
#if defined(__aarch64__)
            if (count == 8) {
                const int16x8_t v = vld1q_s16(in);
                int32x4_t lo = vmovl_s16(vget_low_s16(v));
                int32x4_t hi = vmovl_s16(vget_high_s16(v));
                if (add) { lo = vaddq_s32(lo, vld1q_s32(out)); hi = vaddq_s32(hi, vld1q_s32(out + 4)); }
                vst1q_s32(out, lo); vst1q_s32(out + 4, hi);
                continue;
            }
#endif
            for (int j = 0; j < count; ++j) out[j] = (add ? out[j] : 0) + int32_t(in[j]);
        }
      }
    }
}
}

#include "rk_npu_flatquant.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail {
namespace {
using Clock = std::chrono::steady_clock;
double us(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
float value(const void *A, bool half, size_t i) {
    return half ? rknpu2_matmul_open::bits::half_to_float(static_cast<const uint16_t *>(A)[i])
                : static_cast<const float *>(A)[i];
}
int code(float x, float s, float inverse) {
    return int(std::clamp(std::nearbyint(std::isfinite(inverse) ? x * inverse : x / s), -8.f, 7.f));
}
void put(uint8_t *q, int M, int m, int k, int v) {
    auto *dst = q + (size_t(k / 32) * M + m) * 16 + (k % 32) / 2;
    if (k & 1)
        *dst |= uint8_t(v) & 15;
    else
        *dst = uint8_t((uint8_t(v) & 15) << 4);
}
void scale(float lo, float hi, float negative, float positive, float &s, float &inverse) {
    float a = std::max(-lo * negative, hi * positive);
    s = lo == 0 && hi == 0 ? 1 : std::max(a / 7, FLT_MIN);
    inverse = 1 / s;
}
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline __attribute__((always_inline)) void pack16(float32x4_t a, float32x4_t b, float32x4_t c,
                                                  float32x4_t d, float inverse, uint8_t *out) {
    auto mul = vdupq_n_f32(inverse);
    auto low = vdupq_n_s32(-8), high = vdupq_n_s32(7);
    auto q = [&](float32x4_t x) {
        return vqmovn_s32(vmaxq_s32(low, vminq_s32(high, vcvtnq_s32_f32(vmulq_f32(x, mul)))));
    };
    auto v = vandq_u8(vreinterpretq_u8_s8(vcombine_s8(vqmovn_s16(vcombine_s16(q(a), q(b))),
                                                      vqmovn_s16(vcombine_s16(q(c), q(d))))),
                      vdupq_n_u8(15));
    vst1_u8(out, vorr_u8(vshl_n_u8(vget_low_u8(vuzp1q_u8(v, v)), 4), vget_low_u8(vuzp2q_u8(v, v))));
}
#endif
} // namespace

int flat_validate_input(const void *A, bool half, int M, int K, int threads, bool to_half) {
    if (!A || M < 1 || K < 1 || threads < 1 || threads > 4)
        return RK_NPU_ERR_PARAM;
    int bad = 0;
#pragma omp parallel for num_threads(threads)                                                      \
    reduction(| : bad) if (threads > 1 && int64_t(M) * K >= 8192)
    for (int m = 0; m < M; ++m) {
        int k = 0;
#if defined(__aarch64__)
        if (half) {
            const auto *row = static_cast<const uint16_t *>(A) + size_t(m) * K;
            for (; k + 8 <= K; k += 8)
                if (vmaxvq_u16(vceqq_u16(vandq_u16(vld1q_u16(row + k), vdupq_n_u16(0x7c00)),
                                         vdupq_n_u16(0x7c00))))
                    bad = 1;
        } else {
            const auto *row = static_cast<const float *>(A) + size_t(m) * K;
            const auto max = vdupq_n_f32(to_half ? 65504.f : FLT_MAX);
            for (; k + 4 <= K; k += 4)
                if (vminvq_u32(vcleq_f32(vabsq_f32(vld1q_f32(row + k)), max)) != UINT32_MAX)
                    bad = 1;
        }
#endif
        for (; k < K; ++k) {
            float x = value(A, half, size_t(m) * K + k);
            if (!std::isfinite(x) || (!half && to_half && std::fabs(x) > 65504.f))
                bad = 1;
        }
    }
    return bad ? RK_NPU_ERR_PARAM : RK_NPU_OK;
}

void flat_pack_input(const void *A, bool half, int K, int m0, const FlatShape &s,
                     const std::vector<FlatGroup> &groups, int threads, uint16_t *output) {
    const int rows = groups.back().m0 + groups.back().rows;
    std::memset(output, 0, size_t(rows) * s.lp * s.rp * 2);
#pragma omp parallel for num_threads(threads) if (threads > 1 && int64_t(rows) * K >= 8192)
    for (int gi = 0; gi < int(groups.size()); ++gi) {
        const auto &g = groups[gi];
        auto *dst = output + size_t(g.m0) * s.lp * s.rp;
        const int H = g.rows * s.lp;
        for (int lb = 0; lb < (s.L + 3) / 4; ++lb)
            for (int m = 0; m < g.rows; ++m)
                for (int r = 0; r < s.R; r += 8) {
                    int n = std::min(8, s.R - r);
                    auto *out = dst + (size_t(r / 8) * H + (lb * g.rows + m) * 4) * 8;
                    for (int ll = 0; ll < std::min(4, s.L - lb * 4); ++ll) {
                        size_t src = (size_t(m0 + g.m0 + m) * s.L + lb * 4 + ll) * s.R + r;
                        if (half) {
                            const auto *in = static_cast<const uint16_t *>(A) + src;
                            if (n == 8)
                                std::memcpy(out + ll * 8, in, 16);
                            else
                                std::memcpy(out + ll * 8, in, n * 2);
                        } else {
                            const auto *in = static_cast<const float *>(A) + src;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                            if (n == 8) {
                                vst1q_u16(out + ll * 8, vreinterpretq_u16_f16(vcombine_f16(
                                                            vcvt_f16_f32(vld1q_f32(in)),
                                                            vcvt_f16_f32(vld1q_f32(in + 4)))));
                                continue;
                            }
#endif
                            for (int j = 0; j < n; ++j)
                                out[ll * 8 + j] = rknpu2_matmul_open::bits::float_to_half(in[j]);
                        }
                    }
                }
    }
}

int flat_quantize_native(const uint16_t *data, const FlatShape &s,
                         const std::vector<FlatGroup> &groups, int m0, int M, int threads,
                         float negative, float positive, uint8_t *packed, float *scales,
                         float *inverse, FlatMetrics &metrics) {
    auto start = Clock::now();
    int bad = 0;
#pragma omp parallel for num_threads(threads) reduction(| : bad) if (threads > 1)
    for (int gi = 0; gi < int(groups.size()); ++gi) {
        const auto &g = groups[gi];
        const auto *base = data + size_t(g.m0) * s.lp * s.rp;
        for (int m = 0; m < g.rows; ++m) {
            float lo = 0, hi = 0;
            for (int lb = 0; lb < s.lp / 4; ++lb) {
                const auto *row = base + (size_t(lb) * g.rows + m) * s.rp * 4;
                int j = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                auto vlo = vdupq_n_f32(0), vhi = vdupq_n_f32(0);
                for (; j + 8 <= s.rp * 4; j += 8) {
                    const auto bits = vld1q_u16(row + j);
                    if (vmaxvq_u16(
                            vceqq_u16(vandq_u16(bits, vdupq_n_u16(0x7c00)), vdupq_n_u16(0x7c00))))
                        bad = 1;
                    const auto h = vreinterpretq_f16_u16(bits);
                    const auto a = vcvt_f32_f16(vget_low_f16(h)),
                               b = vcvt_f32_f16(vget_high_f16(h));
                    vlo = vminq_f32(vlo, vminq_f32(a, b));
                    vhi = vmaxq_f32(vhi, vmaxq_f32(a, b));
                }
                lo = std::min(lo, vminvq_f32(vlo));
                hi = std::max(hi, vmaxvq_f32(vhi));
#endif
                for (; j < s.rp * 4; ++j) {
                    float x = rknpu2_matmul_open::bits::half_to_float(row[j]);
                    if (!std::isfinite(x))
                        bad = 1;
                    lo = std::min(lo, x);
                    hi = std::max(hi, x);
                }
            }
            int token = m0 + g.m0 + m;
            scale(lo, hi, negative, positive, scales[token], inverse[token]);
        }
    }
    metrics.scan += us(start);
    if (bad)
        return RK_NPU_ERR_PARAM;
    start = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1)
    for (int gi = 0; gi < int(groups.size()); ++gi) {
        const auto &g = groups[gi];
        const auto *base = data + size_t(g.m0) * s.lp * s.rp;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if (s.R % 32 == 0) {
            for (int lb = 0; lb < (s.L + 3) / 4; ++lb)
                for (int r0 = 0; r0 < s.R; r0 += 32)
                    for (int mb = 0; mb < g.rows; mb += 4)
                        for (int m = mb; m < std::min(mb + 4, g.rows); ++m) {
                            int token = m0 + g.m0 + m;
                            for (int rr = 0; rr < 32; rr += 16) {
                                int r = r0 + rr;
                                const auto *p =
                                    base +
                                    ((size_t(lb) * g.rows + m) * (s.rp / 2) + r / 16 * 8) * 8;
                                auto a = vld4q_u32(reinterpret_cast<const uint32_t *>(p));
                                auto b = vld4q_u32(reinterpret_cast<const uint32_t *>(p + 32));
                                // Constant lane indices keep the deinterleaved vectors in
                                // registers.
                                auto emit = [&](int ll, uint32x4_t av, uint32x4_t bv) {
                                    if (lb * 4 + ll >= s.L)
                                        return;
                                    auto x = vreinterpretq_u16_u32(av),
                                         y = vreinterpretq_u16_u32(bv);
                                    auto lo = vreinterpretq_f16_u16(vuzp1q_u16(x, y)),
                                         hi = vreinterpretq_f16_u16(vuzp2q_u16(x, y));
                                    int k = (lb * 4 + ll) * s.R + r;
                                    auto *out =
                                        packed + (size_t(k / 32) * M + token) * 16 + (k % 32) / 2;
                                    pack16(vcvt_f32_f16(vget_low_f16(lo)),
                                           vcvt_f32_f16(vget_high_f16(lo)),
                                           vcvt_f32_f16(vget_low_f16(hi)),
                                           vcvt_f32_f16(vget_high_f16(hi)), inverse[token], out);
                                };
                                emit(0, a.val[0], b.val[0]);
                                emit(1, a.val[1], b.val[1]);
                                emit(2, a.val[2], b.val[2]);
                                emit(3, a.val[3], b.val[3]);
                            }
                        }
            continue;
        }
#endif
        for (int m = 0; m < g.rows; ++m) {
            int token = m0 + g.m0 + m;
            for (int l = 0; l < s.L; ++l)
                for (int r = 0; r < s.R; ++r) {
                    size_t i =
                        ((size_t(l / 4) * g.rows + m) * (s.rp / 2) + r / 16 * 8 + r % 8) * 8 +
                        (l % 4) * 2 + (r / 8) % 2;
                    put(packed, M, token, l * s.R + r,
                        code(rknpu2_matmul_open::bits::half_to_float(base[i]), scales[token], inverse[token]));
                }
        }
    }
    metrics.quant += us(start);
    return RK_NPU_OK;
}

int flat_quantize_identity(const void *A, bool half, int M, int K, int threads, float negative,
                           float positive, uint8_t *packed, float *scales, float *inverse,
                           FlatMetrics &metrics) {
    auto start = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1 && int64_t(M) * K >= 8192)
    for (int m = 0; m < M; ++m) {
        float lo = 0, hi = 0;
        for (int k = 0; k < K; ++k) {
            float x = value(A, half, size_t(m) * K + k);
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        scale(lo, hi, negative, positive, scales[m], inverse[m]);
    }
    metrics.scan += us(start);
    start = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1 && int64_t(M) * K >= 8192)
    for (int m = 0; m < M; ++m) {
        int k = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if (std::isfinite(inverse[m]))
            for (; k + 16 <= K; k += 16) {
                float32x4_t a, b, c, d;
                if (half) {
                    const auto *in = static_cast<const uint16_t *>(A) + size_t(m) * K + k;
                    auto x = vreinterpretq_f16_u16(vld1q_u16(in)),
                         y = vreinterpretq_f16_u16(vld1q_u16(in + 8));
                    a = vcvt_f32_f16(vget_low_f16(x));
                    b = vcvt_f32_f16(vget_high_f16(x));
                    c = vcvt_f32_f16(vget_low_f16(y));
                    d = vcvt_f32_f16(vget_high_f16(y));
                } else {
                    const auto *in = static_cast<const float *>(A) + size_t(m) * K + k;
                    a = vld1q_f32(in);
                    b = vld1q_f32(in + 4);
                    c = vld1q_f32(in + 8);
                    d = vld1q_f32(in + 12);
                }
                pack16(a, b, c, d, inverse[m],
                       packed + (size_t(k / 32) * M + m) * 16 + (k % 32) / 2);
            }
#endif
        for (; k < K; ++k)
            put(packed, M, m, k, code(value(A, half, size_t(m) * K + k), scales[m], inverse[m]));
    }
    metrics.quant += us(start);
    return RK_NPU_OK;
}
} // namespace rknpu2_matmul_open::detail

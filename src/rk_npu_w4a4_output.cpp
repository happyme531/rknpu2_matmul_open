#include "rk_npu_half_bits.h"
#include "rk_npu_w4a4_linear_cpu.h"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail {
namespace {
template <bool First> void accumulate(const int16_t *src, int32_t *dst, size_t count) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= count; i += 8) {
        auto h = vld1q_s16(src + i);
        auto lo = vmovl_s16(vget_low_s16(h)), hi = vmovl_s16(vget_high_s16(h));
        if constexpr (!First) {
            lo = vaddq_s32(lo, vld1q_s32(dst + i));
            hi = vaddq_s32(hi, vld1q_s32(dst + i + 4));
        }
        vst1q_s32(dst + i, lo);
        vst1q_s32(dst + i + 4, hi);
    }
#endif
    for (; i < count; ++i)
        dst[i] = int32_t(src[i]) + (First ? 0 : dst[i]);
}
template <bool Half> void store(float value, void *output, size_t i) {
    if constexpr (Half)
        static_cast<uint16_t *>(output)[i] = rknpu2_matmul_open::bits::float_to_half(value);
    else
        static_cast<float *>(output)[i] = value;
}
template <bool First, bool Half>
void finish(const W4A4Reduction &s, const rk_npu_i4_config &cfg, const I4Tile &t,
            const int16_t *partial, int begin, int end) {
    const auto *src = partial + t.output_offset / 2;
    const auto *acc = s.accumulator + t.output_offset / 2;
    // Finish one complete FP16 cache line before moving to the next row.
    for (int n0 = begin; n0 < end; n0 += 32)
        for (int r = 0; r < t.m; ++r) {
            int m = t.m0 + r;
            for (int nb = n0; nb < std::min(n0 + 32, t.logical_n); nb += 8) {
                int n = t.n0 + nb, count = std::min(8, t.logical_n - nb);
                size_t i = (size_t(nb / 8) * t.m + r) * 8, out = size_t(m) * cfg.N + n;
#if defined(__aarch64__)
                if (count == 8) {
                    auto value = vld1q_s16(src + i);
                    auto lo = vmovl_s16(vget_low_s16(value)), hi = vmovl_s16(vget_high_s16(value));
                    if constexpr (!First) {
                        lo = vaddq_s32(lo, vld1q_s32(acc + i));
                        hi = vaddq_s32(hi, vld1q_s32(acc + i + 4));
                    }
                    auto as = vdupq_n_f32(s.a_scale[m]);
                    auto a = vmulq_f32(vmulq_f32(vcvtq_f32_s32(lo), as), vld1q_f32(s.w_scale + n));
                    auto b =
                        vmulq_f32(vmulq_f32(vcvtq_f32_s32(hi), as), vld1q_f32(s.w_scale + n + 4));
                    if constexpr (Half) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        vst1q_u16(
                            static_cast<uint16_t *>(s.output) + out,
                            vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b))));
#else
                        float values[8];
                        vst1q_f32(values, a);
                        vst1q_f32(values + 4, b);
                        for (int j = 0; j < 8; ++j)
                            store<true>(values[j], s.output, out + j);
#endif
                    } else {
                        vst1q_f32(static_cast<float *>(s.output) + out, a);
                        vst1q_f32(static_cast<float *>(s.output) + out + 4, b);
                    }
                    continue;
                }
#endif
                for (int j = 0; j < count; ++j) {
                    int32_t v = int32_t(src[i + j]) + (First ? 0 : acc[i + j]);
                    store<Half>((float(v) * s.a_scale[m]) * s.w_scale[n + j], s.output, out + j);
                }
            }
        }
}
template <bool First, bool Half>
void finish_tiles(const W4A4Reduction &s, const rk_npu_i4_config &cfg,
                  const std::vector<I4Tile> &tiles, const int16_t *partial) {
    int64_t jobs = 0;
    for (const auto &t : tiles)
        jobs += int64_t((t.logical_n + 31) / 32) * t.m;
    bool parallel = cfg.cpu_threads > 1 && int64_t(cfg.M) * cfg.N >= 65536;
#pragma omp parallel num_threads(cfg.cpu_threads) if (parallel)
    {
        int id = 0, workers = 1;
#ifdef _OPENMP
        id = omp_get_thread_num();
        workers = omp_get_num_threads();
#endif
        int64_t begin = jobs * id / workers, end = jobs * (id + 1) / workers, offset = 0;
        for (const auto &t : tiles) {
            int64_t count = int64_t((t.logical_n + 31) / 32) * t.m;
            int nb = int((std::clamp(begin - offset, int64_t(0), count) + t.m - 1) / t.m) * 32;
            int ne = int((std::clamp(end - offset, int64_t(0), count) + t.m - 1) / t.m) * 32;
            if (nb < ne)
                finish<First, Half>(s, cfg, t, partial, nb, ne);
            offset += count;
        }
    }
}
} // namespace
int w4a4_reduce_tiles(const W4A4Reduction &s, const rk_npu_i4_config &cfg,
                      const std::vector<I4Tile> &tiles, int wave, int waves,
                      const int16_t *partial) {
    if (!s.accumulator || !s.a_scale || !s.w_scale || !s.output || !partial || wave < 0 ||
        wave >= waves || cfg.cpu_threads < 1 || cfg.cpu_threads > 4 || tiles.empty())
        return RK_NPU_ERR_PARAM;
    if (wave + 1 != waves) {
        const size_t count = size_t(cfg.M) * align_up(cfg.N, 64), blocks = (count + 255) / 256;
#pragma omp parallel for num_threads(cfg.cpu_threads) if (cfg.cpu_threads > 1 && count >= 65536)
        for (size_t b = 0; b < blocks; ++b) {
            size_t start = b * 256, n = std::min(size_t(256), count - start);
            if (wave == 0)
                accumulate<true>(partial + start, s.accumulator + start, n);
            else
                accumulate<false>(partial + start, s.accumulator + start, n);
        }
    } else if (s.half) {
        if (wave == 0)
            finish_tiles<true, true>(s, cfg, tiles, partial);
        else
            finish_tiles<false, true>(s, cfg, tiles, partial);
    } else {
        if (wave == 0)
            finish_tiles<true, false>(s, cfg, tiles, partial);
        else
            finish_tiles<false, false>(s, cfg, tiles, partial);
    }
    return RK_NPU_OK;
}
int w4a4_reduce(void *user, const I4Plan &plan, int wave, const int16_t *partial) {
    return user ? w4a4_reduce_tiles(*static_cast<W4A4Reduction *>(user), plan.config(),
                                    plan.tiles(), wave, plan.wave_count(), partial)
                : RK_NPU_ERR_PARAM;
}
} // namespace rknpu2_matmul_open::detail

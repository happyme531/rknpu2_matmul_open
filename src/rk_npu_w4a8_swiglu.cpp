#include "rk_npu_w4a8_cpu.h"
#include "rk_npu_swiglu_fp32_neon.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rknpu2_matmul_open::detail {
namespace {
template <bool First, bool Two, bool Collect>
void finish_swiglu(const W4A8Reduction &state, const W4A8SwiGLUPair &pair,
                   const int16_t *src, const int16_t *second, int F, int begin, int end,
                   uint32_t *maxima) {
    const auto *gate = src + pair.gate_partial, *up = src + pair.up_partial;
    const auto *ga = First ? nullptr : state.accumulator + pair.gate_accumulator;
    const auto *ua = First ? nullptr : state.accumulator + pair.up_accumulator;
    for (int r = begin; r < end; ++r) {
        const int m = pair.m0 + r;
        auto *out = static_cast<float *>(state.output) + size_t(m) * F + pair.n0;
#if defined(__aarch64__)
        // Bound register lifetimes across reconstruction and activation.
        alignas(16) float gates[32], ups[32];
#endif
        uint32_t maximum = 0;
        for (int nb = 0; nb < 32; nb += 8) {
            const size_t pi = (size_t(nb / 8) * pair.rows + r) * 16;
            const size_t ai = pi / 2;
            const auto *gc = First ? state.correction + pair.n0 + nb : ga + ai;
            const auto *uc = First ? state.correction + F + pair.n0 + nb : ua + ai;
#if defined(__aarch64__)
            const auto as = vdupq_n_f32(state.a_scale[m]);
            const auto gh = vld1q_s16(gate + pi), gl = vld1q_s16(gate + pi + 8);
            const auto uh = vld1q_s16(up + pi), ul = vld1q_s16(up + pi + 8);
            auto g0 = vaddw_s16(vmlal_n_s16(vld1q_s32(gc), vget_low_s16(gh), 16), vget_low_s16(gl));
            auto g1 = vaddw_s16(vmlal_n_s16(vld1q_s32(gc + 4), vget_high_s16(gh), 16), vget_high_s16(gl));
            auto u0 = vaddw_s16(vmlal_n_s16(vld1q_s32(uc), vget_low_s16(uh), 16), vget_low_s16(ul));
            auto u1 = vaddw_s16(vmlal_n_s16(vld1q_s32(uc + 4), vget_high_s16(uh), 16), vget_high_s16(ul));
            if constexpr (Two) {
                const auto gh1 = vld1q_s16(second + pair.gate_partial + pi);
                const auto gl1 = vld1q_s16(second + pair.gate_partial + pi + 8);
                const auto uh1 = vld1q_s16(second + pair.up_partial + pi);
                const auto ul1 = vld1q_s16(second + pair.up_partial + pi + 8);
                g0 = vaddw_s16(vmlal_n_s16(g0, vget_low_s16(gh1), 16), vget_low_s16(gl1));
                g1 = vaddw_s16(vmlal_n_s16(g1, vget_high_s16(gh1), 16), vget_high_s16(gl1));
                u0 = vaddw_s16(vmlal_n_s16(u0, vget_low_s16(uh1), 16), vget_low_s16(ul1));
                u1 = vaddw_s16(vmlal_n_s16(u1, vget_high_s16(uh1), 16), vget_high_s16(ul1));
            }
            const auto gscale = state.w_scale + pair.n0 + nb;
            const auto uscale = gscale + F;
            const auto gf0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(g0), as), vld1q_f32(gscale));
            const auto gf1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(g1), as), vld1q_f32(gscale + 4));
            const auto uf0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(u0), as), vld1q_f32(uscale));
            const auto uf1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(u1), as), vld1q_f32(uscale + 4));
            vst1q_f32(gates + nb, gf0);
            vst1q_f32(gates + nb + 4, gf1);
            vst1q_f32(ups + nb, uf0);
            vst1q_f32(ups + nb + 4, uf1);
#else
            for (int j = 0; j < 8; ++j) {
                int64_t gsum = int64_t(gc[j]) + 16 * int64_t(gate[pi + j]) + gate[pi + 8 + j];
                int64_t usum = int64_t(uc[j]) + 16 * int64_t(up[pi + j]) + up[pi + 8 + j];
                if constexpr (Two) {
                    gsum += 16 * int64_t(second[pair.gate_partial + pi + j]) + second[pair.gate_partial + pi + 8 + j];
                    usum += 16 * int64_t(second[pair.up_partial + pi + j]) + second[pair.up_partial + pi + 8 + j];
                }
                const int32_t g = int32_t(gsum), u = int32_t(usum);
                const float gf = (float(g) * state.a_scale[m]) * state.w_scale[pair.n0 + nb + j];
                const float uf = (float(u) * state.a_scale[m]) * state.w_scale[F + pair.n0 + nb + j];
                const float y = (gf / (1.f + std::exp(-gf))) * uf;
                out[nb + j] = y;
                if constexpr (Collect) {
                    uint32_t bits;
                    std::memcpy(&bits, &y, sizeof(bits));
                    maximum = std::max(maximum, bits & 0x7fffffffu);
                }
            }
#endif
        }
#if defined(__aarch64__)
        auto vmax = vdupq_n_u32(0);
        for (int nb = 0; nb < 32; nb += 16) {
            const auto current = swiglu_fp32::apply16<Collect>(gates + nb, ups + nb, out + nb);
            if constexpr (Collect) vmax = vmaxq_u32(vmax, current);
        }
        if constexpr (Collect) maximum = vmaxvq_u32(vmax);
#endif
        if constexpr (Collect) maxima[m] = std::max(maxima[m], maximum);
    }
}

template <bool First, bool Two, bool Collect>
void reduce_swiglu(const W4A8Reduction &state, const rk_npu_i4_config &cfg,
                   const int16_t *src, const int16_t *second = nullptr) {
    const auto &pairs = *state.swiglu_pairs;
    int64_t jobs = 0;
    for (const auto &pair : pairs) jobs += pair.rows;
    // Each work unit is one row/N32 pair, including partial final M panels.
#pragma omp parallel num_threads(cfg.cpu_threads) if (cfg.cpu_threads > 1)
    {
        int id = 0, workers = 1;
#ifdef _OPENMP
        id = omp_get_thread_num();
        workers = omp_get_num_threads();
#endif
        const int64_t begin = jobs * id / workers, end = jobs * (id + 1) / workers;
        int64_t offset = 0;
        for (const auto &pair : pairs) {
            const int first = int(std::clamp(begin - offset, int64_t(0), int64_t(pair.rows)));
            const int last = int(std::clamp(end - offset, int64_t(0), int64_t(pair.rows)));
            if (first < last)
                finish_swiglu<First, Two, Collect>(state, pair, src, second, cfg.N / 2, first, last,
                    Collect ? state.row_maxima + size_t(id) * state.row_maxima_stride : nullptr);
            offset += pair.rows;
        }
    }
}
} // namespace

int w4a8_reduce_swiglu(const W4A8Reduction &state, const rk_npu_i4_config &cfg,
                       const int16_t *src, bool first) {
    if (state.row_maxima) {
        if (first) reduce_swiglu<true, false, true>(state, cfg, src);
        else reduce_swiglu<false, false, true>(state, cfg, src);
    } else {
        if (first) reduce_swiglu<true, false, false>(state, cfg, src);
        else reduce_swiglu<false, false, false>(state, cfg, src);
    }
    return RK_NPU_OK;
}

int w4a8_reduce_two_tiles(const W4A8Reduction &state, const rk_npu_i4_config &cfg,
                          const int16_t *first, const int16_t *second) {
    if (state.row_maxima) reduce_swiglu<true, true, true>(state, cfg, first, second);
    else reduce_swiglu<true, true, false>(state, cfg, first, second);
    return RK_NPU_OK;
}

int w4a8_reduce_two(void *user, const I4Plan &plan, const int16_t *first, const int16_t *second) {
    return w4a8_reduce_two_tiles(*static_cast<W4A8Reduction *>(user), plan.config(), first, second);
}

} // namespace rknpu2_matmul_open::detail

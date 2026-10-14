#include "rk_npu_half_bits.h"
#include "rk_npu_w4a8_cpu.h"
#include "rk_npu_swiglu_neon.h"
#include "rk_npu_trace.h"
#include <stdexcept>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__)
#include "rk_npu_w4a8_reduce_neon.h"
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::detail {
namespace {
#if defined(__aarch64__)
template <bool Half> inline void store_output4(float32x4_t y, void *output, size_t offset) {
    if constexpr (!Half) {
        vst1q_f32(static_cast<float *>(output) + offset, y);
    } else {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        vst1_u16(static_cast<uint16_t *>(output) + offset, vreinterpret_u16_f16(vcvt_f16_f32(y)));
#else
        float values[4];
        vst1q_f32(values, y);
        for (int j = 0; j < 4; ++j)
            static_cast<uint16_t *>(output)[offset + j] = rknpu2_matmul_open::bits::float_to_half(values[j]);
#endif
    }
}
#endif

// One N8 group / token. Two adjacent INT16 rows become one INT32 row.
// The native accumulator has exactly the same byte footprint as the partial.
template <bool First, bool Final, bool Half>
inline void combine8(const W4A8Reduction &state, const I4Tile &t, int N, const int16_t *partial,
                     int32_t *acc, int nb, int r) {
    const int logical_m = t.m / 2;
    const size_t index = (size_t(nb / 8) * logical_m + r) * 8;
    const auto *high = partial + index * 2;
    const auto *low = high + 8;
    auto *destination = acc + index;
    const auto *initial = First ? state.correction + t.n0 + nb : destination;
    const int m = t.m0 / 2 + r, n = t.n0 + nb;
    const int count = std::min(8, t.logical_n - nb);
#if defined(__aarch64__)
    if (count == 8) {
        const auto h = vld1q_s16(high), l = vld1q_s16(low);
        auto v0 = vmlal_n_s16(vld1q_s32(initial), vget_low_s16(h), 16);
        auto v1 = vmlal_n_s16(vld1q_s32(initial + 4), vget_high_s16(h), 16);
        v0 = vaddw_s16(v0, vget_low_s16(l));
        v1 = vaddw_s16(v1, vget_high_s16(l));
        if constexpr (Final) {
            const auto as = vdupq_n_f32(state.a_scale[m]);
            const auto y0 =
                vmulq_f32(vmulq_f32(vcvtq_f32_s32(v0), as), vld1q_f32(state.w_scale + n));
            const auto y1 =
                vmulq_f32(vmulq_f32(vcvtq_f32_s32(v1), as), vld1q_f32(state.w_scale + n + 4));
            store_output4<Half>(y0, state.output, size_t(m) * N + n);
            store_output4<Half>(y1, state.output, size_t(m) * N + n + 4);
        } else {
            vst1q_s32(destination, v0);
            vst1q_s32(destination + 4, v1);
        }
        return;
    }
#endif
    for (int j = 0; j < count; ++j) {
        const int32_t v = int32_t(int64_t(initial[j]) + 16 * int64_t(high[j]) + low[j]);
        if constexpr (Final) {
            const float y = (float(v) * state.a_scale[m]) * state.w_scale[n + j];
            if constexpr (Half)
                static_cast<uint16_t *>(state.output)[size_t(m) * N + n + j] =
                    rknpu2_matmul_open::bits::float_to_half(y);
            else
                static_cast<float *>(state.output)[size_t(m) * N + n + j] = y;
        } else {
            destination[j] = v;
        }
    }
}

template <bool First, bool Final, bool Half>
void reduce_tile(const W4A8Reduction &state, const rk_npu_i4_config &cfg, const I4Tile &t,
                 const int16_t *src, int nbegin, int nend) {
    const int rows = t.m / 2;
    const auto *partial = src + t.output_offset / 2;
    auto *acc = state.accumulator + t.output_offset / 4;
    if constexpr (!Final) {
        // Contiguous reads and writes across all tokens of each N8 block.
        int nb = nbegin;
#if defined(__aarch64__)
        const int full_end = std::min(nend, t.logical_n / 8 * 8);
        if (nb < full_end) {
            const size_t index = size_t(nb / 8) * rows * 8;
            w4reduce::accumulate<First>(partial + index * 2, acc + index,
                                        state.correction + t.n0 + nb, rows, (full_end - nb) / 8);
            nb = full_end;
        }
#endif
        for (; nb < std::min(nend, t.logical_n); nb += 8)
            for (int r = 0; r < t.m / 2; ++r)
                combine8<First, false, Half>(state, t, cfg.N, partial, acc, nb, r);
    } else {
        // Transpose only once, completing at least one FP16 cache line
        // per row before moving on. Interleaving small stores across
        // row strides such as N=4096 otherwise thrashes the same L1 sets.
        int n0 = nbegin;
#if defined(__aarch64__)
        for (; n0 + 32 <= std::min(nend, t.logical_n); n0 += 32) {
            const size_t index = size_t(n0 / 8) * rows * 8;
            const int m = t.m0 / 2, n = t.n0 + n0;
            w4reduce::finish32<First, Half>(partial + index * 2, acc + index, state.correction + n,
                                            state.a_scale + m, state.w_scale + n, state.output,
                                            size_t(m) * cfg.N + n, cfg.N, rows);
        }
#endif
        for (; n0 < std::min(nend, t.logical_n); n0 += 32)
            for (int r = 0; r < t.m / 2; ++r)
                for (int nb = n0; nb < std::min(n0 + 32, t.logical_n); nb += 8)
                    combine8<First, true, Half>(state, t, cfg.N, partial, acc, nb, r);
    }
}

template <bool First, bool Final, bool Half>
void reduce_tiles(const W4A8Reduction &state, const rk_npu_i4_config &cfg,
                  const std::vector<I4Tile> &tiles, const int16_t *src) {
    const int threads = cfg.cpu_threads;
    const bool parallel = threads > 1 && int64_t(cfg.M / 2) * cfg.N >= 65536;
    if constexpr (!Final) {
        // Preserve native streaming ownership between intermediate waves.
#pragma omp parallel for num_threads(threads) schedule(static) if (parallel)
        for (int i = 0; i < int(tiles.size()); ++i)
            reduce_tile<First, false, Half>(state, cfg, tiles[i], src, 0, tiles[i].logical_n);
    } else {
        int64_t jobs = 0;
        for (const auto &t : tiles)
            jobs += int64_t((t.logical_n + 31) / 32) * (t.m / 2);
#pragma omp parallel num_threads(threads) if (parallel)
        {
            int id = 0, workers = 1;
#ifdef _OPENMP
            id = omp_get_thread_num();
            workers = omp_get_num_threads();
#endif
            const int64_t begin = jobs * id / workers, end = jobs * (id + 1) / workers;
            int64_t offset = 0;
            for (const auto &t : tiles) {
                const int rows = t.m / 2;
                const int64_t work = int64_t((t.logical_n + 31) / 32) * rows;
                const int nbegin =
                    int((std::clamp(begin - offset, int64_t(0), work) + rows - 1) / rows) * 32;
                const int nend =
                    int((std::clamp(end - offset, int64_t(0), work) + rows - 1) / rows) * 32;
                offset += work;
                if (nbegin < nend)
                    reduce_tile<First, true, Half>(state, cfg, t, src, nbegin, nend);
            }
        }
    }
}
template <bool First>
void finish_swiglu(const W4A8Reduction &state, const W4A8SwiGLUPair &pair,
                   const int16_t *src, int F, int begin, int end) {
    const auto *gate = src + pair.gate_partial, *up = src + pair.up_partial;
    const auto *ga = state.accumulator + pair.gate_accumulator;
    const auto *ua = state.accumulator + pair.up_accumulator;
    for (int r = begin; r < end; ++r) {
        const int m = pair.m0 + r;
        auto *out = static_cast<float *>(state.output) + size_t(m) * F + pair.n0;
#if defined(__aarch64__)
        // Bound register lifetimes across reconstruction and activation.
        alignas(16) float gates[32], ups[32];
#endif
        for (int nb = 0; nb < 32; nb += 8) {
            const size_t pi = (size_t(nb / 8) * pair.rows + r) * 16;
            const size_t ai = pi / 2;
            const auto *gc = First ? state.correction + pair.n0 + nb : ga + ai;
            const auto *uc = First ? state.correction + F + pair.n0 + nb : ua + ai;
#if defined(__aarch64__)
            const auto as = vdupq_n_f32(state.a_scale[m]);
            const auto gh = vld1q_s16(gate + pi), gl = vld1q_s16(gate + pi + 8);
            const auto uh = vld1q_s16(up + pi), ul = vld1q_s16(up + pi + 8);
            const auto g0 = vaddw_s16(vmlal_n_s16(vld1q_s32(gc), vget_low_s16(gh), 16), vget_low_s16(gl));
            const auto g1 = vaddw_s16(vmlal_n_s16(vld1q_s32(gc + 4), vget_high_s16(gh), 16), vget_high_s16(gl));
            const auto u0 = vaddw_s16(vmlal_n_s16(vld1q_s32(uc), vget_low_s16(uh), 16), vget_low_s16(ul));
            const auto u1 = vaddw_s16(vmlal_n_s16(vld1q_s32(uc + 4), vget_high_s16(uh), 16), vget_high_s16(ul));
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
                const int32_t g = int32_t(int64_t(gc[j]) + 16 * int64_t(gate[pi + j]) + gate[pi + 8 + j]);
                const int32_t u = int32_t(int64_t(uc[j]) + 16 * int64_t(up[pi + j]) + up[pi + 8 + j]);
                const float gf = (float(g) * state.a_scale[m]) * state.w_scale[pair.n0 + nb + j];
                const float uf = (float(u) * state.a_scale[m]) * state.w_scale[F + pair.n0 + nb + j];
                out[nb + j] = swiglu::scalar(gf, uf);
            }
#endif
        }
#if defined(__aarch64__)
        for (int nb = 0; nb < 32; nb += 8)
            swiglu::apply8(vld1q_f32(gates + nb), vld1q_f32(ups + nb),
                          vld1q_f32(gates + nb + 4), vld1q_f32(ups + nb + 4), out + nb);
#endif
    }
}

template <bool First>
void reduce_swiglu(const W4A8Reduction &state, const rk_npu_i4_config &cfg, const int16_t *src) {
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
            if (first < last) finish_swiglu<First>(state, pair, src, cfg.N / 2, first, last);
            offset += pair.rows;
        }
    }
}
} // namespace

std::vector<W4A8SwiGLUPair> w4a8_swiglu_pairs(int N, const std::vector<I4Tile> &tiles) {
    if (N <= 0 || N % 64) throw std::invalid_argument("SwiGLU requires N%64=0");
    const int F = N / 2;
    std::vector<W4A8SwiGLUPair> pairs;
    for (const auto &g : tiles) {
        for (int n = g.n0; n < std::min(F, g.n0 + g.logical_n); n += 32) {
            const I4Tile *up = nullptr;
            for (const auto &u : tiles)
                if (u.m0 == g.m0 && u.m == g.m && u.n0 <= n + F && n + F + 32 <= u.n0 + u.logical_n) {
                    up = &u;
                    break;
                }
            if (!up) throw std::invalid_argument("SwiGLU gate/up tile mismatch");
            const int rows = g.m / 2;
            const uint64_t go = uint64_t((n - g.n0) / 8) * rows * 16;
            const uint64_t uo = uint64_t((n + F - up->n0) / 8) * rows * 16;
            pairs.push_back({g.output_offset / 2 + go, up->output_offset / 2 + uo,
                             g.output_offset / 4 + go / 2, up->output_offset / 4 + uo / 2,
                             g.m0 / 2, rows, n});
        }
    }
    return pairs;
}

int w4a8_reduce_tiles(const W4A8Reduction &state, const rk_npu_i4_config &cfg,
                      const std::vector<I4Tile> &tiles, int wave, int waves, const int16_t *src) {
    if (!src || !state.accumulator || !state.correction || !state.a_scale || !state.w_scale ||
        !state.output || wave < 0 || wave >= waves || cfg.cpu_threads < 1 || cfg.cpu_threads > 4)
        return RK_NPU_ERR_PARAM;
    for (const auto &t : tiles)
        if (t.m0 % 2 || t.m <= 0 || t.m % 2)
            return RK_NPU_ERR_PARAM;
    if (wave + 1 == waves) {
        if (state.swiglu_pairs) {
            if (state.half || cfg.N % 64) return RK_NPU_ERR_PARAM;
            RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.swiglu_epilogue", "M", cfg.M / 2,
                "F", cfg.N / 2, "cpu_threads", cfg.cpu_threads);
            if (wave == 0) reduce_swiglu<true>(state, cfg, src);
            else reduce_swiglu<false>(state, cfg, src);
        } else if (state.half) {
            if (wave == 0)
                reduce_tiles<true, true, true>(state, cfg, tiles, src);
            else
                reduce_tiles<false, true, true>(state, cfg, tiles, src);
        } else {
            if (wave == 0)
                reduce_tiles<true, true, false>(state, cfg, tiles, src);
            else
                reduce_tiles<false, true, false>(state, cfg, tiles, src);
        }
    } else if (wave == 0)
        reduce_tiles<true, false, false>(state, cfg, tiles, src);
    else
        reduce_tiles<false, false, false>(state, cfg, tiles, src);
    return RK_NPU_OK;
}

int w4a8_reduce(void *user, const I4Plan &plan, int wave, const int16_t *src) {
    return user ? w4a8_reduce_tiles(*static_cast<W4A8Reduction *>(user), plan.config(),
                                    plan.tiles(), wave, plan.wave_count(), src)
                : RK_NPU_ERR_PARAM;
}

void w4a8_dequant(int M, int N, const int32_t *acc, const float *a_scale, const float *w_scale,
                  bool half, void *output, int threads) {
#pragma omp parallel for num_threads(threads)                                                      \
    schedule(static) if (threads > 1 && int64_t(M) * N >= 8192)
    for (int m = 0; m < M; ++m) {
        const auto *row = acc + size_t(m) * N;
        int n = 0;
#if defined(__aarch64__)
        const auto as = vdupq_n_f32(a_scale[m]);
        for (; n + 4 <= N; n += 4) {
            const auto y =
                vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(row + n)), as), vld1q_f32(w_scale + n));
            if (!half)
                vst1q_f32(static_cast<float *>(output) + size_t(m) * N + n, y);
            else {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                vst1_u16(static_cast<uint16_t *>(output) + size_t(m) * N + n,
                         vreinterpret_u16_f16(vcvt_f16_f32(y)));
#else
                float tmp[4];
                vst1q_f32(tmp, y);
                for (int j = 0; j < 4; ++j)
                    static_cast<uint16_t *>(output)[size_t(m) * N + n + j] =
                        rknpu2_matmul_open::bits::float_to_half(tmp[j]);
#endif
            }
        }
#endif
        for (; n < N; ++n) {
            const float y = (float(row[n]) * a_scale[m]) * w_scale[n];
            if (half)
                static_cast<uint16_t *>(output)[size_t(m) * N + n] = rknpu2_matmul_open::bits::float_to_half(y);
            else
                static_cast<float *>(output)[size_t(m) * N + n] = y;
        }
    }
}
} // namespace rknpu2_matmul_open::detail

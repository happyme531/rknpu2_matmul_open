// CPU-only oracle for native INT16 -> INT32 accumulation -> floating output.
#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_w4a8_cpu.h"
#include "rk_npu_w4a8.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)

static void one(int M, int N, int mt, int nt, int mask, int waves, int threads, bool half) {
    rk_npu_i4_config cfg{M * 2, N, 480 * waves, mt * 2, 480, nt, unsigned(mask), 500, 1, threads};
    std::vector<rknpu2_matmul_open::detail::I4Tile> tiles;
    const int cores = __builtin_popcount(unsigned(mask)), blocks = (N + 63) / 64;
    int n0 = 0;
    uint64_t bytes = 0;
    for (int core = 0; core < cores; ++core) {
        const int end = n0 + (blocks / cores + (core < blocks % cores)) * 64;
        for (; n0 < end; n0 += std::min(nt, end - n0)) {
            const int n = std::min(nt, end - n0);
            for (int m0 = 0; m0 < M; m0 += mt) {
                const int m = std::min(mt, M - m0);
                tiles.push_back({m0 * 2, m * 2, n0, n, std::max(0, std::min(n, N - n0)), bytes});
                bytes += uint64_t(m) * n * 4;
            }
        }
    }
    std::mt19937 rng(20260922 + M + N);
    std::vector<int16_t> partial(bytes / 2 + 32, 0x1234);
    std::vector<int32_t> acc(bytes / 4 + 32, 0x12345678), correction(N);
    std::vector<int64_t> reference(size_t(M) * N);
    std::vector<float> as(M), ws(N), output(size_t(M) * N + 32, -12345);
    std::vector<uint16_t> output_h(output.size(), 0x5555);
    for (int n = 0; n < N; ++n) {
        correction[n] = (1 << 24) + int(rng() % 65536) - 32768;
        ws[n] = .0003f + float(n % 17) * .00001f;
    }
    for (int m = 0; m < M; ++m) {
        as[m] = .001f + float(m % 13) * .00013f;
        for (int n = 0; n < N; ++n)
            reference[size_t(m) * N + n] = correction[n];
    }
    rknpu2_matmul_open::detail::W4A8Reduction state{acc.data() + 16,
                               correction.data(),
                               as.data(),
                               ws.data(),
                               half ? static_cast<void *>(output_h.data() + 16)
                                    : output.data() + 16,
                               half};
    for (int wave = 0; wave < waves; ++wave) {
        for (const auto &t : tiles)
            for (int nb = 0; nb < t.logical_n; nb += 8)
                for (int r = 0; r < t.m / 2; ++r)
                    for (int lane = 0; lane < std::min(8, t.logical_n - nb); ++lane) {
                        const size_t i =
                            t.output_offset / 2 + (size_t(nb / 8) * (t.m / 2) + r) * 16 + lane;
                        const int16_t h = int(rng() % 65536) - 32768,
                                      l = int(rng() % 65536) - 32768;
                        partial[i] = h;
                        partial[i + 8] = l;
                        reference[size_t(t.m0 / 2 + r) * N + t.n0 + nb + lane] +=
                            16 * int64_t(h) + l;
                    }
        CHECK(!rknpu2_matmul_open::detail::w4a8_reduce_tiles(state, cfg, tiles, wave, waves, partial.data()));
        if (wave + 1 < waves) {
            for (const auto &t : tiles)
                for (int nb = 0; nb < t.logical_n; nb += 8)
                    for (int r = 0; r < t.m / 2; ++r)
                        for (int lane = 0; lane < std::min(8, t.logical_n - nb); ++lane) {
                            const size_t i =
                                t.output_offset / 4 + (size_t(nb / 8) * (t.m / 2) + r) * 8 + lane;
                            CHECK(state.accumulator[i] ==
                                  reference[size_t(t.m0 / 2 + r) * N + t.n0 + nb + lane]);
                        }
        }
    }
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            const size_t i = size_t(m) * N + n;
            const float y = (float(int32_t(reference[i])) * as[m]) * ws[n];
            CHECK(half ? output_h[i + 16] == rknpu2_matmul_open::bits::float_to_half(y) : output[i + 16] == y);
        }
    for (int i = 0; i < 16; ++i) {
        CHECK(acc[i] == 0x12345678 && acc[acc.size() - 1 - i] == 0x12345678);
        CHECK(output[i] == -12345 && output[output.size() - 1 - i] == -12345);
        CHECK(output_h[i] == 0x5555 && output_h[output_h.size() - 1 - i] == 0x5555);
    }
    // Native N padding must not be touched by the CPU consumer.
    for (const auto &t : tiles)
        for (int n = t.logical_n; n < t.n; ++n)
            for (int r = 0; r < t.m / 2; ++r) {
                const size_t i = t.output_offset / 4 + (size_t(n / 8) * (t.m / 2) + r) * 8 + n % 8;
                CHECK(state.accumulator[i] == 0x12345678);
            }
}
int main() {
    try {
        for (int threads : {1, 2, 3, 4})
            for (int waves : {1, 2, 3, 5, 6})
                for (bool half : {false, true}) {
                    one(1, 1, 1, 64, 1, waves, threads, half);
                    one(3, 65, 2, 64, 3, waves, threads, half);
                    one(7, 193, 3, 128, 7, waves, threads, half);
                    one(65, 1025, 64, 512, 7, waves, threads, half);
                    one(128, 4096, 64, 1408, 7, waves, threads, half);
                }
        rk_npu_w4a8_config cfg;
        rk_npu_w4a8_config_init(&cfg, 128, 4096, 11008);
        rk_npu_w4a8_memory_info info{};
        CHECK(!rk_npu_w4a8_memory_query(&cfg, &info));
        CHECK(info.backend.partial_bytes == 2u * 128 * 4096 * 4);
        cfg.pipeline = 0;
        CHECK(!rk_npu_w4a8_memory_query(&cfg, &info));
        CHECK(info.backend.partial_bytes == 2u * 128 * 4096 * 4);
        std::puts("PASS W4A8 reduction: first/middle/final, f32/f16, threads 1..4, native "
                  "tails/guards, >2^24 sums, memory query");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

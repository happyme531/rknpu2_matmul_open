#include "rk_npu_w4a8.h"
#include "../src/rk_npu_w4a8_cpu.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
static int nibble(const uint8_t *p, size_t i) {
    int x = (p[i / 2] >> ((i & 1) ? 0 : 4)) & 15;
    return x >= 8 ? x - 16 : x;
}
static void packing(bool half) {
    const int M = 5, K = 533;
    std::vector<float> a(M * K), scales(M), inverse(M);
    std::vector<uint16_t> h(M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            a[m * K + k] = m == 0 ? 0 : std::ldexp(float(k % 255 - 127), m - 3);
            if (k == K - 1)
                a[m * K + k] =
                    m == 0 ? 0 : std::ldexp(254.f, m - 3); // full-K max beyond first tile
            h[m * K + k] = rknpu2_matmul_open::bits::float_to_half(a[m * K + k]);
            if (half)
                a[m * K + k] = rknpu2_matmul_open::bits::half_to_float(h[m * K + k]);
        }
    // Exact halfway cases under scale=1, including the vector and scalar tails.
    for (int k = 0; k < K; ++k)
        a[K + k] = float(k % 255 - 127);
    a[K + 1] = .5f;
    a[K + 2] = -.5f;
    a[K + 3] = 1.5f;
    a[K + 4] = -1.5f;
    a[2 * K - 2] = -.5f;
    for (int k = 0; k < K; ++k)
        h[K + k] = rknpu2_matmul_open::bits::float_to_half(a[K + k]);
    const void *source = half ? static_cast<const void *>(h.data()) : a.data();
    CHECK(!rknpu2_matmul_open::detail::w4a8_scales(source, half, M, K, 1, scales.data(), inverse.data()));
    rknpu2_matmul_open::detail::W4A8Input input{source, half, M, K, scales.data(), inverse.data()};
    for (int kt : {32, 256, 480})
        for (int k0 = 0; k0 < K; k0 += kt)
            for (int m0 = 0; m0 < M; m0 += 2) {
                int rows = std::min(2, M - m0) * 2, k = std::min(kt, K - k0),
                    pk = (k + 31) / 32 * 32;
                std::vector<uint8_t> packed(rows * pk / 2 + 16, 0xa5);
                rk_npu_i4_input_tile t{
                    m0 * 2, rows, k0, k, pk, packed.data(), uint64_t(rows * pk / 2)};
                CHECK(!rknpu2_matmul_open::detail::w4a8_pack(&input, &t));
                for (int m = 0; m < rows / 2; ++m)
                    for (int j = 0; j < pk; ++j) {
                        size_t hi = (size_t(j / 32) * rows + 2 * m) * 32 + j % 32;
                        const int hq = nibble(packed.data(), hi),
                                  lq = nibble(packed.data(), hi + 32);
                        if (j >= k) {
                            CHECK(hq == 0 && lq == 0);
                            continue;
                        }
                        float maximum = 0;
                        for (int q = 0; q < K; ++q)
                            maximum = std::max(maximum, std::fabs(a[(m0 + m) * K + q]));
                        const float scale = maximum == 0 ? 1 : std::max(maximum / 127.f, FLT_MIN);
                        const int ref = int(std::clamp(
                            std::round(a[(m0 + m) * K + k0 + j] * (1.f / scale)), -127.f, 127.f));
                        CHECK(16 * hq + lq + 8 == ref);
                    }
                CHECK(std::all_of(packed.end() - 16, packed.end(),
                                  [](uint8_t x) { return x == 0xa5; }));
            }
    CHECK(scales[0] == 1);
    a[0] = std::numeric_limits<float>::infinity();
    CHECK(rknpu2_matmul_open::detail::w4a8_scales(a.data(), false, M, K, 1, scales.data(), inverse.data()) ==
          RK_NPU_ERR_PARAM);
    a[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(rknpu2_matmul_open::detail::w4a8_scales(a.data(), false, M, K, 1, scales.data(), inverse.data()) ==
          RK_NPU_ERR_PARAM);
    h[0] = 0x7c00;
    CHECK(rknpu2_matmul_open::detail::w4a8_scales(h.data(), true, M, K, 1, scales.data(), inverse.data()) ==
          RK_NPU_ERR_PARAM);
    float tiny = std::numeric_limits<float>::denorm_min();
    CHECK(!rknpu2_matmul_open::detail::w4a8_scales(&tiny, false, 1, 1, 1, scales.data(), inverse.data()));
    CHECK(scales[0] == FLT_MIN && std::isfinite(inverse[0]));
}
static void parallel_packing(bool half) {
    const int M = 65, K = 4097;
    std::vector<float> a(size_t(M)*K), scale(M), inv(M);
    std::vector<uint16_t> h(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = float(int(i % 255) - 127) * .125f;
        h[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
    }
    const void* source = half ? static_cast<const void*>(h.data()) : a.data();
    CHECK(!rknpu2_matmul_open::detail::w4a8_scales(source, half, M, K, 4, scale.data(), inv.data()));
    rknpu2_matmul_open::detail::W4A8Input in{source, half, M, K, scale.data(), inv.data(), 1};
    for (int kt : {480, 992, 1472, 1984, 2048}) {
        for (int k0 = 0; k0 < K; k0 += kt) {
            const int k = std::min(kt,K-k0), pk = (k+31)/32*32;
            for (int m0 = 0; m0 < M; m0 += 64) {
                const int rows = 2*std::min(64,M-m0);
                const size_t bytes = size_t(rows)*pk/2;
                std::vector<uint8_t> serial(bytes+32,0xa5), parallel(bytes+32,0xa5);
                rk_npu_i4_input_tile t{2*m0,rows,k0,k,pk,serial.data(),bytes};
                in.threads = 1;
                CHECK(!rknpu2_matmul_open::detail::w4a8_pack(&in,&t));
                in.threads = 4;
                t.dst = parallel.data();
                CHECK(!rknpu2_matmul_open::detail::w4a8_pack(&in,&t));
                CHECK(serial == parallel);
                CHECK(std::all_of(parallel.end()-32,parallel.end(),[](uint8_t x){return x==0xa5;}));
            }
        }
    }
}
int main() {
    try {
        for (unsigned h = 0; h < 65536; ++h) {
            const uint16_t round = rknpu2_matmul_open::bits::float_to_half(rknpu2_matmul_open::bits::half_to_float(uint16_t(h)));
            if ((h & 0x7c00) == 0x7c00 && (h & 1023))
                CHECK((round & 0x7c00) == 0x7c00 && (round & 1023));
            else
                CHECK(round == h);
        }
        packing(false);
        packing(true);
        parallel_packing(false);
        parallel_packing(true);
        const int32_t acc[8] = {-100001, 0, 1, 2, 2049, 65536, 12345, -45678};
        const float as[1] = {.125f}, ws[8] = {1, 1, 1, 1, 1, 1, .003f, 2};
        float out[8];
        uint16_t half[8];
        rknpu2_matmul_open::detail::w4a8_dequant(1, 8, acc, as, ws, false, out, 1);
        rknpu2_matmul_open::detail::w4a8_dequant(1, 8, acc, as, ws, true, half, 1);
        for (int n = 0; n < 8; ++n) {
            CHECK(out[n] == (float(acc[n]) * as[0]) * ws[n]);
            CHECK(half[n] == rknpu2_matmul_open::bits::float_to_half(out[n]));
        }
        rk_npu_w4a8_config cfg;
        rk_npu_w4a8_config_init(&cfg, 3, 65, 513);
        rk_npu_w4a8_memory_info mem{};
        CHECK(!rk_npu_w4a8_memory_query(&cfg, &mem));
        CHECK(mem.weight_metadata_bytes == 65 * 8 && mem.cpu_scratch_bytes == 3 * 128 * 4 + 3 * 8);
        cfg.m_tile = 65;
        CHECK(rk_npu_w4a8_memory_query(&cfg, &mem) == RK_NPU_ERR_PARAM);
        cfg.m_tile = 64;
        cfg.K = INT32_MAX / 1024 + 1;
        CHECK(rk_npu_w4a8_memory_query(&cfg, &mem) == RK_NPU_ERR_PARAM);
        std::puts("PASS W4A8 CPU: f32/f16, MSD codes, ties, full-token scale, padding/guard, "
                  "finite checks, half conversion");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

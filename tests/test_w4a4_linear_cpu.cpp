#include "../src/rk_npu_flatquant.h"
#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_w4a4_linear_cpu.h"
#include "rk_npu_w4a4_linear.h"
#include <algorithm>
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

static void native_quant(int L, int R) {
    const int M = 37, K = L * R;
    rknpu2_matmul_open::detail::FlatShape shape;
    CHECK(!rknpu2_matmul_open::detail::flat_shape(L, R, shape));
    const int lp = shape.lp, rp = shape.rp;
    std::vector<rknpu2_matmul_open::detail::FlatGroup> groups{{0, 16}, {16, 16}, {32, 5}};
    std::vector<float> logical(size_t(M) * K);
    std::vector<uint16_t> native(size_t(M) * lp * rp), packed_input(native.size());
    for (int m = 0; m < M; ++m)
        for (int l = 0; l < L; ++l)
            for (int r = 0; r < R; ++r) {
                float v = m == 0 ? 0 : float((m * 73 + l * 11 + r * 3) % 113 - 56) / 16;
                logical[(size_t(m) * L + l) * R + r] = v;
                const auto &g = groups[m < 16 ? 0 : m < 32 ? 1 : 2];
                size_t index =
                    size_t(g.m0) * lp * rp +
                    (((size_t(l / 4) * g.rows + m - g.m0) * (rp / 2) + r / 16 * 8 + r % 8) * 8 +
                     (l % 4) * 2 + (r / 8) % 2);
                native[index] = rknpu2_matmul_open::bits::float_to_half(v);
            }
    std::vector<uint8_t> q(size_t(M) * ((K + 31) / 32) * 16, 0);
    std::vector<float> scales(M), inverse(M);
    rknpu2_matmul_open::detail::FlatMetrics metrics;
    CHECK(!rknpu2_matmul_open::detail::flat_quantize_native(native.data(), shape, groups, 0, M, 4, .75f, .5f, q.data(),
                                       scales.data(), inverse.data(), metrics));
    for (int m = 0; m < M; ++m) {
        float lo = 0, hi = 0;
        for (int k = 0; k < K; ++k) {
            float x = logical[size_t(m) * K + k];
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        float expected = std::max(-lo * .75f, hi * .5f) / 7;
        if (expected == 0)
            expected = 1;
        CHECK(scales[m] == expected);
        for (int k = 0; k < (K + 31) / 32 * 32; ++k) {
            int code = k < K
                           ? int(std::clamp(std::nearbyint(logical[size_t(m) * K + k] * inverse[m]),
                                            -8.f, 7.f))
                           : 0;
            CHECK(nibble(q.data(), (size_t(k / 32) * M + m) * 32 + k % 32) == code);
        }
    }
    rknpu2_matmul_open::detail::flat_pack_input(logical.data(), false, K, 0, shape, groups, 4, packed_input.data());
    for (const auto &g : groups)
        for (int m = 0; m < g.rows; ++m)
            for (int l = 0; l < lp; ++l)
                for (int r = 0; r < rp; ++r) {
                    const size_t index =
                        size_t(g.m0) * lp * rp +
                        ((size_t(r / 8) * (g.rows * lp) + (l / 4 * g.rows + m) * 4 + l % 4) * 8 +
                         r % 8);
                    const uint16_t expected =
                        l < L && r < R
                            ? rknpu2_matmul_open::bits::float_to_half(logical[(size_t(g.m0 + m) * L + l) * R + r])
                            : 0;
                    CHECK(packed_input[index] == expected);
                }
    // Copy arbitrary M/K panels from the full native INT4 staging buffer.
    std::vector<uint8_t> tile_data(3 * 16);
    rknpu2_matmul_open::detail::W4A4Input in{q.data(), M, K, 1};
    rk_npu_i4_input_tile tile{2, 3, 0, std::min(32, K), 32, tile_data.data(), tile_data.size()};
    CHECK(!rknpu2_matmul_open::detail::w4a4_pack(&in, &tile));
    for (int m = 0; m < 3; ++m)
        for (int k = 0; k < 32; ++k)
            CHECK(nibble(tile_data.data(), m * 32 + k) == nibble(q.data(), (m + 2) * 32 + k));
    native[0] = 0x7c00;
    CHECK(rknpu2_matmul_open::detail::flat_quantize_native(native.data(), shape, groups, 0, M, 4, .75f, .5f, q.data(),
                                      scales.data(), inverse.data(), metrics) == RK_NPU_ERR_PARAM);
}
static void transform_and_pack() {
    native_quant(2, 3);
    native_quant(5, 13);
    native_quant(40, 64);
    native_quant(76, 128);
    std::vector<uint16_t> a(32);
    for (int k = 0; k < 32; ++k)
        a[k] = rknpu2_matmul_open::bits::float_to_half(float(k - 16) / 2);
    std::vector<uint8_t> q(16);
    float scale = 0, inverse = 0;
    rknpu2_matmul_open::detail::FlatMetrics metrics;
    CHECK(!rknpu2_matmul_open::detail::flat_validate_input(a.data(), true, 1, 32, 1, false));
    CHECK(!rknpu2_matmul_open::detail::flat_quantize_identity(a.data(), true, 1, 32, 1, 7.f / 8, 7.f / 8, q.data(),
                                         &scale, &inverse, metrics));
    CHECK(scale == 1 && nibble(q.data(), 31) == 7 && nibble(q.data(), 1) == -8 &&
          nibble(q.data(), 29) == 6);
    a[0] = 0x7c00;
    CHECK(rknpu2_matmul_open::detail::flat_validate_input(a.data(), true, 1, 32, 1, true) == RK_NPU_ERR_PARAM);
    std::fill(a.begin(), a.end(), rknpu2_matmul_open::bits::float_to_half(1.f));
    const float tiny = std::numeric_limits<float>::denorm_min();
    CHECK(!rknpu2_matmul_open::detail::flat_quantize_identity(a.data(), true, 1, 32, 1, tiny, tiny, q.data(), &scale,
                                         &inverse, metrics));
    CHECK(scale == std::numeric_limits<float>::min() && std::isfinite(inverse));
    for (int i = 0; i < 32; ++i)
        CHECK(nibble(q.data(), i) == 7);
    float large = 70000;
    CHECK(rknpu2_matmul_open::detail::flat_validate_input(&large, false, 1, 1, 1, true) == RK_NPU_ERR_PARAM);
    CHECK(!rknpu2_matmul_open::detail::flat_validate_input(&large, false, 1, 1, 1, false));
}

static void fused_reduction() {
    rk_npu_i4_config cfg{2, 9, 64, 2, 32, 64, 1, 500, 1, 2};
    std::vector<rknpu2_matmul_open::detail::I4Tile> tiles{{0, 2, 0, 64, 9, 0}};
    std::vector<int16_t> p0(2 * 64), p1(2 * 64);
    for (int nb = 0; nb < 2; ++nb)
        for (int m = 0; m < 2; ++m)
            for (int j = 0; j < 8; ++j) {
                const size_t i = (size_t(nb) * 2 + m) * 8 + j;
                p0[i] = int16_t(100 * m + 10 * nb + j - 20);
                p1[i] = int16_t(-30 * m + 4 * nb - j + 7);
            }
    std::vector<int32_t> accumulator(2 * 64, 0x55555555);
    std::vector<float> out(18, -999), as{.25f, .5f}, ws(9);
    for (int n = 0; n < 9; ++n)
        ws[n] = .01f * (n + 1);
    rknpu2_matmul_open::detail::W4A4Reduction reduction{accumulator.data(), as.data(), ws.data(), out.data(), false};
    CHECK(!rknpu2_matmul_open::detail::w4a4_reduce_tiles(reduction, cfg, tiles, 0, 2, p0.data()));
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == -999; }));
    CHECK(!rknpu2_matmul_open::detail::w4a4_reduce_tiles(reduction, cfg, tiles, 1, 2, p1.data()));
    for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 9; ++n) {
            const size_t i = (size_t(n / 8) * 2 + m) * 8 + n % 8;
            const float expected = float(int(p0[i]) + int(p1[i])) * as[m] * ws[n];
            CHECK(out[m * 9 + n] == expected);
        }
}

static void public_validation() {
    rk_npu_w4a4_linear_config cfg;
    rk_npu_w4a4_linear_config_init(&cfg, 3, 65, 6);
    cfg.k_tile = 32;
    cfg.n_tile = 64;
    rk_npu_w4a4_transform tr;
    rk_npu_w4a4_transform_init(&tr);
    const float left[4] = {1, 0, 0, 1}, right[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    tr.left_dim = 2;
    tr.right_dim = 3;
    tr.left = left;
    tr.right = right;
    rk_npu_w4a4_linear_memory_info info{};
    CHECK(!rk_npu_w4a4_linear_memory_query(&cfg, &tr, &info));
    CHECK(info.weight_metadata_bytes == 65 * sizeof(float) + 2 * (32 * 32 + 4 * 16 * 16));
    CHECK(info.cpu_scratch_bytes == 3 * 16 + 2 * 3 * sizeof(float) + 3 * 128 * sizeof(int32_t));
    CHECK(info.transform_workspace_bytes > 0);
    tr.positive_clip_ratio = 0;
    CHECK(rk_npu_w4a4_linear_memory_query(&cfg, &tr, &info) == RK_NPU_ERR_PARAM);
    tr.positive_clip_ratio = 1;
    tr.right_dim = 4;
    CHECK(rk_npu_w4a4_linear_memory_query(&cfg, &tr, &info) == RK_NPU_ERR_PARAM);
}

int main() {
    try {
        transform_and_pack();
        fused_reduction();
        public_validation();
        std::puts("PASS native FlatQuant input/output layout, clipped INT4 pack, "
                  "fused final dequant");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

#include "rk_npu_w4a8.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
using W = std::unique_ptr<rk_npu_w4a8_weights, decltype(&rk_npu_w4a8_weights_free)>;
using P = std::unique_ptr<rk_npu_w4a8_workspace, decltype(&rk_npu_w4a8_workspace_free)>;
constexpr const char *ENV = "RK_NPU_W4A8_K_TILE_MULTIPLIER";

static void compare(rk_npu_iommu_domain *domain, int M, int N, int K, int kt, int mask, bool half) {
    std::mt19937 rng(123);
    std::vector<int8_t> b(size_t(K) * N);
    std::vector<float> a(size_t(M) * K), s(N, .01f), ref(size_t(M) * N), out(ref.size());
    std::vector<uint16_t> ah(a.size()), h(ref.size()), hr(h.size());
    // Small-magnitude weights certify even K2048, independent of A.
    for (auto &v : b)
        v = int(rng() % 5) - 2;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = float(int(rng() % 255) - 127);
        ah[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
    }
    auto make = [&](int tile) {
        rk_npu_i4_weight_config wc{K, N, tile};
        return W(rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()),
                 rk_npu_w4a8_weights_free);
    };
    rk_npu_w4a8_config cfg;
    rk_npu_w4a8_config_init(&cfg, M, N, K);
    cfg.npu_core_mask = mask;
    cfg.m_tile = 32;
    cfg.n_tile = 128;
    cfg.cpu_threads = M * N >= 65536 ? 4 : 1;
    auto base = make(480);
    CHECK(base);
    P bp(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free);
    CHECK(bp);
    CHECK(!(half ? rk_npu_w4a8_run_f16(bp.get(), base.get(), ah.data(), hr.data(), nullptr)
                 : rk_npu_w4a8_run_f32(bp.get(), base.get(), a.data(), ref.data(), nullptr)));
    auto w = make(kt);
    CHECK(w);
    cfg.k_tile = kt;
    P p(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free);
    CHECK(p);
    for (int repeat = 0; repeat < 3; ++repeat) {
        CHECK(!(half ? rk_npu_w4a8_run_f16(p.get(), w.get(), ah.data(), h.data(), nullptr)
                     : rk_npu_w4a8_run_f32(p.get(), w.get(), a.data(), out.data(), nullptr)));
        CHECK(half ? h == hr : out == ref);
    }
    std::printf("PASS expanded tile M=%d K=%d N=%d kt=%d mask=%d %s full exact\n", M, K, N, kt,
                mask, half ? "f16" : "f32");
    std::fflush(stdout);
}
static void environment(rk_npu_iommu_domain *domain) {
    const int K = 4096, N = 64;
    std::vector<int8_t> b(K * N, -8);
    std::vector<float> s(N, 1), a(K, 0), c(N);
    rk_npu_i4_weight_config wc{K, N, 0};
    for (const char *factor : {"1", "1.5"}) {
        setenv(ENV, factor, 1);
        W w(rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()), rk_npu_w4a8_weights_free);
        CHECK(w);
        rk_npu_w4a8_weight_info info{};
        CHECK(!rk_npu_w4a8_weights_query(w.get(), &info));
        CHECK(info.safe_k_tile == 480);
        CHECK(info.config.k_tile == (factor[1] ? 704 : 480));
        rk_npu_w4a8_config cfg;
        rk_npu_w4a8_config_init(&cfg, 1, N, K);
        cfg.k_tile = info.config.k_tile;
        P p(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free);
        CHECK(p);
        CHECK(!rk_npu_w4a8_run_f32(p.get(), w.get(), a.data(), c.data(), nullptr));
        if (!info.bound_relaxed)
            CHECK(std::all_of(c.begin(), c.end(), [](float x) { return x == 0; }));
        else {
            CHECK(c[0] != 0);
            std::printf("EXPECTED LOSS: multiplier=1.5 kt=704 zero A, W=-8 gives %.0f instead of "
                        "0; no fallback\n",
                        c[0]);
        }
    }
    for (const char *invalid : {"0", "-1", "nan", "inf", "1.5oops", ""}) {
        setenv(ENV, invalid, 1);
        CHECK(!rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()));
    }
    setenv(ENV, "1.5", 1);
    wc.k_tile = 704;
    CHECK(!rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data())); // explicit remains strict
    std::fill(b.begin(), b.end(), 1);
    wc.k_tile = 0;
    W capped(rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()), rk_npu_w4a8_weights_free);
    CHECK(capped);
    rk_npu_w4a8_weight_info info{};
    CHECK(!rk_npu_w4a8_weights_query(capped.get(), &info));
    CHECK(info.safe_k_tile == 2048 && info.config.k_tile == 2048 && !info.bound_relaxed);
    unsetenv(ENV);
    std::puts("PASS multiplier parsing, auto metadata, cap, strict explicit tiles");
}
int main(int argc, char **argv) {
    auto *ctx = rk_npu_open(nullptr);
    if (!ctx)
        return 2;
    auto *domain = rk_npu_iommu_domain_create(ctx, 0);
    int result = 0;
    try {
        CHECK(domain);
        compare(domain, 2, 64, 4096, 992, 1, false);
        if (argc < 2 || std::strcmp(argv[1], "--canary")) {
            for (int kt : {1472, 1984, 2048})
                compare(domain, 3, 129, 4097, kt, 1, false);
            for (int mask : {2, 4, 3, 7})
                compare(domain, 65, 1025, 4097, 2048, mask, true);
            environment(domain);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        result = 1;
    }
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return result;
}

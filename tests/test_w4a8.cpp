#include "rk_npu_w4a8.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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

static std::vector<float> reference(int M, int K, int N, const std::vector<float> &a,
                                    const std::vector<int8_t> &b, const std::vector<float> &ws) {
    std::vector<float> c(size_t(M) * N);
    std::vector<int8_t> q(K);
    for (int m = 0; m < M; ++m) {
        float maximum = 0;
        for (int k = 0; k < K; ++k)
            maximum = std::max(maximum, std::fabs(a[size_t(m) * K + k]));
        float s = maximum == 0 ? 1 : std::max(maximum / 127.f, FLT_MIN), inv = 1.f / s;
        for (int k = 0; k < K; ++k)
            q[k] = int8_t(std::clamp(std::round(a[size_t(m) * K + k] * inv), -127.f, 127.f));
        for (int n = 0; n < N; ++n) {
            int64_t sum = 0;
            for (int k = 0; k < K; ++k)
                sum += int(q[k]) * int(b[size_t(k) * N + n]);
            c[size_t(m) * N + n] = (float(sum) * s) * ws[n];
        }
    }
    return c;
}
static void check_run(rk_npu_w4a8_workspace *p, rk_npu_w4a8_weights *w, int M, int K, int N,
                      std::vector<float> &a, const std::vector<int8_t> &b,
                      const std::vector<float> &scales, bool half) {
    std::vector<uint16_t> ah(a.size());
    if (half)
        for (size_t i = 0; i < a.size(); ++i) {
            ah[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
            a[i] = rknpu2_matmul_open::bits::half_to_float(ah[i]);
        }
    const auto ref = reference(M, K, N, a, b, scales);
    std::vector<float> c(ref.size() + 32, -12345.0f);
    std::vector<uint16_t> h(ref.size() + 32, 0x5555);
    int rc = half ? rk_npu_w4a8_run_f16(p, w, ah.data(), h.data() + 16, nullptr)
                  : rk_npu_w4a8_run_f32(p, w, a.data(), c.data() + 16, nullptr);
    if (rc) {
        std::fprintf(stderr, "run failed %d\n", rc);
        throw std::runtime_error("NPU failure: stop");
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const bool ok = half ? h[i + 16] == rknpu2_matmul_open::bits::float_to_half(ref[i]) : c[i + 16] == ref[i];
        if (!ok) {
            std::fprintf(stderr, "index=%zu expected=%.9g got=%.9g half=%d\n", i, ref[i],
                         half ? rknpu2_matmul_open::bits::half_to_float(h[i + 16]) : c[i + 16], half);
            throw std::runtime_error("quantized reference mismatch");
        }
    }
    for (int i = 0; i < 16; ++i)
        CHECK(c[i] == -12345 && c[c.size() - 1 - i] == -12345 && h[i] == 0x5555 &&
              h[h.size() - 1 - i] == 0x5555);
}
static void one(rk_npu_iommu_domain *domain, int M, int K, int N, int mask, int pipeline,
                bool extreme = false, int mt = 3, int nt = 128, int repeats = 3) {
    std::mt19937 rng(1234);
    std::vector<int8_t> b(size_t(K) * N);
    std::vector<float> a(size_t(M) * K), s(N);
    for (int n = 0; n < N; ++n)
        s[n] = std::ldexp(1.f, -(n % 6 + 6));
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = extreme ? (i % N % 2 ? 7 : -8) : int(rng() % 16) - 8;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            a[size_t(m) * K + k] =
                extreme ? (m % 2 ? 127.f : -127.f)
                        : float(int(rng() % 2049) - 1024) * std::ldexp(1.f, m % 5 - 8);
            if (!extreme && m == 0)
                a[k] = 0; // zero row tests exact correction cancellation
        }
    rk_npu_w4a8_config cfg;
    rk_npu_w4a8_config_init(&cfg, M, N, K);
    cfg.npu_core_mask = mask;
    cfg.pipeline = pipeline;
    cfg.m_tile = mt;
    cfg.n_tile = nt;
    cfg.cpu_threads = M * N >= 65536 ? 4 : 1;
    rk_npu_i4_weight_config wc{K, N, cfg.k_tile};
    W w(rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()), rk_npu_w4a8_weights_free);
    CHECK(w);
    P p(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free);
    CHECK(p);
    // One workspace/weight object alternates f32/f16; inputs change each call.
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (!extreme && M > 1)
            a[K + repeat % K] += 0.03125f;
        check_run(p.get(), w.get(), M, K, N, a, b, s, false);
        check_run(p.get(), w.get(), M, K, N, a, b, s, true);
    }
    std::vector<float> c(M * N);
    a[0] = std::numeric_limits<float>::quiet_NaN();
    rk_npu_w4a8_timings times{};
    CHECK(rk_npu_w4a8_run_f32(p.get(), w.get(), a.data(), c.data(), &times) == RK_NPU_ERR_PARAM);
    CHECK(times.submit_us == 0);
    std::vector<uint16_t> bad(a.size(), 0);
    bad.back() = 0xfc00;
    CHECK(rk_npu_w4a8_run_f16(p.get(), w.get(), bad.data(), bad.data(), nullptr) ==
          RK_NPU_ERR_PARAM);
    std::vector<uint16_t> ch(c.size());
    CHECK(rk_npu_w4a8_run_f16(p.get(), w.get(), bad.data(), ch.data(), &times) == RK_NPU_ERR_PARAM);
    CHECK(times.submit_us == 0);
    a[0] = 0;
    check_run(p.get(), w.get(), M, K, N, a, b, s, false);
    std::printf("PASS W4A8 M=%d K=%d N=%d mask=%d pipe=%d extreme=%d f32/f16 full exact\n", M, K, N,
                mask, pipeline, extreme);
    std::fflush(stdout);
}
static void shared(rk_npu_iommu_domain *domain, rk_npu_iommu_domain *other) {
    const int K = 513, N = 193;
    std::vector<int8_t> b(K * N, 3), b2(K * N, -2);
    std::vector<float> s(N, .01f), s2(N, .003f);
    rk_npu_i4_weight_config wc{K, N, 256};
    W w(rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()), rk_npu_w4a8_weights_free);
    W w2(rk_npu_w4a8_weights_create(domain, &wc, b2.data(), s2.data()), rk_npu_w4a8_weights_free);
    W foreign(rk_npu_w4a8_weights_create(other, &wc, b.data(), s.data()), rk_npu_w4a8_weights_free);
    CHECK(w && w2 && foreign);
    for (int M : {1, 5, 65}) {
        rk_npu_w4a8_config cfg;
        rk_npu_w4a8_config_init(&cfg, M, N, K);
        cfg.k_tile = 256;
        cfg.m_tile = 16;
        cfg.n_tile = 64;
        cfg.npu_core_mask = 7;
        P p(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free);
        CHECK(p);
        std::vector<float> a(M * K), c(M * N);
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = float(int(i % 255) - 127);
        check_run(p.get(), w.get(), M, K, N, a, b, s, false);
        check_run(p.get(), w2.get(), M, K, N, a, b2, s2, true);
        check_run(p.get(), w.get(), M, K, N, a, b, s, true);
        CHECK(rk_npu_w4a8_run_f32(p.get(), foreign.get(), a.data(), c.data(), nullptr) ==
              RK_NPU_ERR_DOMAIN);
    }
    s[0] = 0;
    CHECK(!rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()));
    s[0] = .01f;
    s[1] = std::numeric_limits<float>::infinity();
    CHECK(!rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()));
    s[1] = .01f;
    b[0] = 8;
    CHECK(!rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data()));
    std::puts("PASS W4A8 shared weights, dtype/M reuse, rebinding, invalid metadata, cross-domain "
              "rejection");
}
int main(int argc, char **argv) {
    auto *ctx = rk_npu_open(nullptr);
    if (!ctx)
        return 2;
    auto *domain = rk_npu_iommu_domain_create(ctx, 0);
    auto *other = rk_npu_iommu_domain_create(ctx, 1);
    int result = 0;
    try {
        CHECK(domain && other);
        one(domain, 2, 480, 64, 1, 0, true);
        if (argc < 2 || std::strcmp(argv[1], "--canary")) {
            one(domain, 1, 1, 1, 1, 1);
            one(domain, 3, 33, 65, 1, 1);
            one(domain, 5, 1025, 129, 1, 0);
            one(domain, 5, 1025, 129, 1, 1, true);
            one(domain, 2, 20003, 65, 1, 1, true); // >2^24 integer sums, FP16 overflow
            for (int mask : {2, 4, 3, 7})
                one(domain, 7, 513, 257, mask, 1);
            one(domain, 65, 513, 1025, 7, 1, true); // parallel reduction threshold + M/N tails
            one(domain, 65, 1441, 1025, 7, 1, false, 64, 1408, 2); // production-sized M tile + K/M/N tails
            one(domain, 65, 1921, 1025, 7, 1, true, 64, 1408, 2); // five waves + exact correction
            shared(domain, other);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        result = 1;
    }
    rk_npu_iommu_domain_free(other);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return result;
}

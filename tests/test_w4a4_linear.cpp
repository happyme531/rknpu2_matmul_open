#include "../src/rk_npu_half_bits.h"
#include "rk_npu_w4a4_linear.h"
#include <algorithm>
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
using W = std::unique_ptr<rk_npu_w4a4_linear_weights, decltype(&rk_npu_w4a4_linear_weights_free)>;
using P =
    std::unique_ptr<rk_npu_w4a4_linear_workspace, decltype(&rk_npu_w4a4_linear_workspace_free)>;

static std::vector<float> transform(const std::vector<float> &a, int M, int L, int R,
                                    const std::vector<float> &left,
                                    const std::vector<float> &right) {
    const int K = L * R;
    std::vector<float> stage(size_t(M) * K), out(stage.size());
    for (int m = 0; m < M; ++m)
        for (int l = 0; l < L; ++l)
            for (int r = 0; r < R; ++r) {
                const float x =
                    rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(a[(size_t(m) * L + l) * R + r]));
                for (int j = 0; j < R; ++j)
                    stage[(size_t(m) * L + l) * R + j] +=
                        x * rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(right[r * R + j]));
            }
    for (auto &v : stage)
        v = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(v));
    for (int m = 0; m < M; ++m)
        for (int o = 0; o < L; ++o)
            for (int l = 0; l < L; ++l) {
                const float x = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(left[l * L + o]));
                for (int r = 0; r < R; ++r)
                    out[(size_t(m) * L + o) * R + r] += x * stage[(size_t(m) * L + l) * R + r];
            }
    for (auto &v : out)
        v = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(v));
    return out;
}

static std::vector<float> reference(const std::vector<float> &a, int M, int K, int N, int L, int R,
                                    const std::vector<float> &left, const std::vector<float> &right,
                                    float negative_clip, float positive_clip,
                                    const std::vector<int8_t> &b,
                                    const std::vector<float> &weight_scale) {
    const auto x = L ? transform(a, M, L, R, left, right) : a;
    std::vector<float> out(size_t(M) * N);
    std::vector<int8_t> q(K);
    for (int m = 0; m < M; ++m) {
        float xmin = 0, xmax = 0;
        for (int k = 0; k < K; ++k) {
            xmin = std::min(xmin, x[size_t(m) * K + k]);
            xmax = std::max(xmax, x[size_t(m) * K + k]);
        }
        const float maximum = std::max(-xmin * negative_clip, xmax * positive_clip);
        const float scale = maximum == 0 ? 1 : maximum / 7;
        for (int k = 0; k < K; ++k)
            q[k] = int8_t(
                std::clamp(std::nearbyint(x[size_t(m) * K + k] * (1.0f / scale)), -8.f, 7.f));
        for (int n = 0; n < N; ++n) {
            int64_t sum = 0;
            for (int k = 0; k < K; ++k)
                sum += int(q[k]) * int(b[size_t(k) * N + n]);
            out[size_t(m) * N + n] = (float(sum) * scale) * weight_scale[n];
        }
    }
    return out;
}

static void make_transform(int dim, std::vector<float> &matrix) {
    matrix.assign(size_t(dim) * dim, 0);
    for (int i = 0; i < dim; ++i) {
        matrix[size_t(i) * dim + i] = 1;
        matrix[size_t(i) * dim + (i + 1) % dim] = (i & 1) ? -.0625f : .125f;
    }
}

static void one(rk_npu_iommu_domain *domain, int M, int L, int R, int N, int k_tile, int mask,
                bool half, bool identity = false) {
    const int K = L * R;
    std::mt19937 rng(1000 + M + K + N);
    std::vector<float> left, right;
    make_transform(L, left);
    make_transform(R, right);
    std::vector<float> a(size_t(M) * K), ws(N);
    std::vector<int8_t> b(size_t(K) * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = float(int(rng() % 1025) - 512) / 128;
    if (M)
        std::fill(a.begin(), a.begin() + K, 0); // all-zero scale path
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = int8_t(int(rng() % 15) - 7);
    for (int n = 0; n < N; ++n)
        ws[n] = std::ldexp(float(n % 7 + 1), -10);
    const float negative_clip = .875f, positive_clip = .9375f;

    rk_npu_w4a4_transform tr;
    rk_npu_w4a4_transform_init(&tr);
    if (!identity) {
        tr.left_dim = L;
        tr.right_dim = R;
        tr.left = left.data();
        tr.right = right.data();
    }
    tr.negative_clip_ratio = negative_clip;
    tr.positive_clip_ratio = positive_clip;
    rk_npu_i4_weight_config wc{K, N, k_tile};
    W weights(rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), ws.data(), &tr),
              rk_npu_w4a4_linear_weights_free);
    CHECK(weights);
    rk_npu_w4a4_linear_weight_info wi{};
    CHECK(!rk_npu_w4a4_linear_weights_query(weights.get(), &wi));
    CHECK(wi.config.k_tile == k_tile && wi.left_dim == (identity ? 0 : L) &&
          wi.right_dim == (identity ? 0 : R));

    rk_npu_w4a4_linear_config cfg;
    rk_npu_w4a4_linear_config_init(&cfg, M, N, K);
    cfg.k_tile = k_tile;
    cfg.m_tile = std::min(M, 32);
    cfg.n_tile = N > 192 ? 128 : 64;
    cfg.npu_core_mask = mask;
    cfg.cpu_threads = M >= 32 ? 4 : 1;
    cfg.pipeline = 1;
    P workspace(rk_npu_w4a4_linear_workspace_create(domain, &cfg, weights.get()),
                rk_npu_w4a4_linear_workspace_free);
    CHECK(workspace);

    std::vector<uint16_t> ah(a.size()), ch(size_t(M) * N + 16, 0x5555);
    if (half)
        for (size_t i = 0; i < a.size(); ++i) {
            ah[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
            a[i] = rknpu2_matmul_open::bits::half_to_float(ah[i]);
        }
    const auto ref = reference(a, M, K, N, identity ? 0 : L, R, left, right, negative_clip,
                               positive_clip, b, ws);
    std::vector<float> c(size_t(M) * N + 16, -12345);
    rk_npu_w4a4_linear_timings timings{};
    const int rc = half ? rk_npu_w4a4_linear_run_f16(workspace.get(), weights.get(), ah.data(),
                                                     ch.data() + 8, &timings)
                        : rk_npu_w4a4_linear_run_f32(workspace.get(), weights.get(), a.data(),
                                                     c.data() + 8, &timings);
    if (rc) {
        std::fprintf(stderr, "submit failed rc=%d M=%d K=%d N=%d\n", rc, M, K, N);
        throw std::runtime_error("NPU failure: stop");
    }
    CHECK(timings.total_us > 0 && timings.transform_scan_us > 0 && timings.submit_us > 0);
    for (size_t i = 0; i < ref.size(); ++i) {
        const bool ok = half ? ch[i + 8] == rknpu2_matmul_open::bits::float_to_half(ref[i]) : c[i + 8] == ref[i];
        if (!ok) {
            std::fprintf(stderr, "mismatch i=%zu ref=%.9g got=%.9g half=%d\n", i, ref[i],
                         half ? rknpu2_matmul_open::bits::half_to_float(ch[i + 8]) : c[i + 8], half);
            throw std::runtime_error("W4A4 Linear output mismatch");
        }
    }
    for (int i = 0; i < 8; ++i)
        CHECK(c[i] == -12345 && c[c.size() - 1 - i] == -12345 && ch[i] == 0x5555 &&
              ch[ch.size() - 1 - i] == 0x5555);
    std::printf("PASS W4A4 Linear M=%d K=%d N=%d kt=%d mask=%d %s total=%.3f us "
                "transform=%.3f pack=%.3f reduce=%.3f\n",
                M, K, N, k_tile, mask, half ? "f16" : "f32", timings.total_us,
                timings.transform_scan_us, timings.quant_pack_us, timings.reduce_dequant_us);
}

static void reuse_and_recover(rk_npu_iommu_domain *domain) {
    const int M = 70, K = 6, N = 193;
    std::vector<float> left{1, 0, 0, 1}, other{2, 0, 0, 2}, right{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<int8_t> b(K * N, 1);
    std::vector<float> scales(N, .03125f), a(M * K), out(M * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = float(int(i % 17) - 8) / 8;
    rk_npu_w4a4_transform tr;
    rk_npu_w4a4_transform_init(&tr);
    tr.left_dim = 2;
    tr.right_dim = 3;
    tr.left = left.data();
    tr.right = right.data();
    rk_npu_i4_weight_config wc{K, N, 32};
    W first(rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), scales.data(), &tr),
            rk_npu_w4a4_linear_weights_free);
    tr.left = other.data();
    W second(rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), scales.data(), &tr),
             rk_npu_w4a4_linear_weights_free);
    CHECK(first && second);
    rk_npu_w4a4_linear_config cfg;
    rk_npu_w4a4_linear_config_init(&cfg, M, N, K);
    cfg.k_tile = 32;
    cfg.n_tile = 64;
    cfg.npu_core_mask = 7;
    cfg.cpu_threads = 4;
    P plan(rk_npu_w4a4_linear_workspace_create(domain, &cfg, first.get()),
           rk_npu_w4a4_linear_workspace_free);
    CHECK(plan);
    for (int i = 0; i < 4; ++i) {
        auto *weights = (i & 1) ? second.get() : first.get();
        const auto expected =
            reference(a, M, K, N, 2, 3, (i & 1) ? other : left, right, 1, 1, b, scales);
        CHECK(!rk_npu_w4a4_linear_run_f32(plan.get(), weights, a.data(), out.data(), nullptr));
        CHECK(out == expected);
        float saved = a.back();
        a.back() = std::numeric_limits<float>::quiet_NaN();
        CHECK(rk_npu_w4a4_linear_run_f32(plan.get(), weights, a.data(), out.data(), nullptr) ==
              RK_NPU_ERR_PARAM);
        a.back() = saved;
    }
    std::puts("PASS workspace weight rebinding and recovery after rejected input");
}

int main(int argc, char **argv) {
    auto *ctx = rk_npu_open(nullptr);
    if (!ctx)
        return 2;
    auto *domain = rk_npu_iommu_domain_create(ctx, 0);
    int result = 0;
    try {
        CHECK(domain);
        one(domain, 3, 2, 3, 65, 32, 1, false);
        if (argc > 1 && !std::strcmp(argv[1], "--canary")) {
            rk_npu_iommu_domain_free(domain);
            rk_npu_close(ctx);
            return 0;
        }
        one(domain, 5, 5, 13, 129, 32, 1, true); // three K waves + tails
        one(domain, 33, 19, 27, 257, 480, 7,
            false); // multicore prefill + M/N/K tails
        one(domain, 137, 5, 13, 193, 32, 7,
            true);                                  // three preprocessing blocks + a tail
        one(domain, 17, 40, 64, 257, 480, 7, true); // NEON native scan/quant path
        one(domain, 9, 5, 13, 193, 32, 7, false,
            true); // identity without dense CPU transform

        one(domain, 3, 1, 1, 65, 32, 1, true);
        one(domain, 3, 128, 128, 193, 480, 7, true);
        reuse_and_recover(domain);

        // Auto K selection and invalid metadata are checked before any submit.
        std::vector<int8_t> b(65 * 64, 1);
        std::vector<float> s(64, .01f), l(25), r(169);
        make_transform(5, l);
        make_transform(13, r);
        rk_npu_w4a4_transform tr;
        rk_npu_w4a4_transform_init(&tr);
        tr.left_dim = 5;
        tr.right_dim = 13;
        tr.left = l.data();
        tr.right = r.data();
        rk_npu_i4_weight_config wc{65, 64, 0};
        W automatic(rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), s.data(), &tr),
                    rk_npu_w4a4_linear_weights_free);
        CHECK(automatic);
        rk_npu_w4a4_linear_weight_info info{};
        CHECK(!rk_npu_w4a4_linear_weights_query(automatic.get(), &info));
        CHECK(info.config.k_tile == 96);
        s[0] = 0;
        CHECK(!rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), s.data(), &tr));
        s[0] = .01f;
        l[0] = std::numeric_limits<float>::quiet_NaN();
        CHECK(!rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), s.data(), &tr));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        result = 1;
    }
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return result;
}

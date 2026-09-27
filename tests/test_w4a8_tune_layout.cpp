#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_w4a8_tune_internal.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
using namespace rknpu2_matmul_open::w4_tune;
static void oracle(Data &d, const std::vector<int8_t> &b, const std::vector<float> &scales) {
    d.reference.resize(size_t(d.M) * d.N);
    d.reference_h.resize(d.reference.size());
    std::vector<int> q(d.K);
    for (int m = 0; m < d.M; ++m) {
        float maximum = 0;
        for (int k = 0; k < d.K; ++k)
            maximum = std::max(maximum, std::fabs(d.a[size_t(m) * d.K + k]));
        const float scale = maximum ? std::max(maximum / 127.f, FLT_MIN) : 1.f, inv = 1.f / scale;
        for (int k = 0; k < d.K; ++k)
            q[k] = int(std::clamp(std::round(d.a[size_t(m) * d.K + k] * inv), -127.f, 127.f));
        for (int n = 0; n < d.N; ++n) {
            int64_t sum = 0;
            for (int k = 0; k < d.K; ++k)
                sum += q[k] * b[size_t(k) * d.N + n];
            const size_t ix = size_t(m) * d.N + n;
            d.reference[ix] = (float(sum) * scale) * scales[n];
            d.reference_h[ix] = rknpu2_matmul_open::bits::float_to_half(d.reference[ix]);
        }
    }
}
static void forced(rk_npu_iommu_domain *domain, int M, int K, int kt) {
    rk_npu_w4a8_autotune_config c;
    rk_npu_w4a8_autotune_config_init(&c, M, 193, K);
    c.k_tile_multiplier = 1;
    c.required_k_tile = kt;
    c.allowed_cpu_core_mask = 0xf0;
    std::vector<int8_t> b(size_t(K) * c.N);
    std::vector<float> scales(c.N, .003f);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = int((i * 17 + i / c.N) % 3) - 1;
    Profile p;
    CHECK(!profile(c, b.data(), scales.data(), p));
    const auto r = resources(c);
    int count = 0;
    for (const auto &x : candidates(c, p, r, true)) {
        if (x.execution.input_layout != RK_NPU_W4A8_INPUT_PANEL8)
            continue;
        Strategy s{};
        s.base.config = x.config;
        s.base.activation_type = c.activation_type;
        s.execution = x.execution;
        s.base.cpu_core_mask = r.cpu;
        s.base.weights = p.info(kt);
        s.base.weight_fingerprint = p.fingerprint;
        s.base.tuning_revision = revision;
        CHECK(valid_strategy(s, c, p, r, true));
        Weight w(rk_npu_w4a8_weights_create_tuned_ex(domain, &s, b.data(), scales.data()),
                 rk_npu_w4a8_weights_free);
        Workspace ws(rk_npu_w4a8_workspace_create_tuned_ex(domain, &s), rk_npu_w4a8_workspace_free);
        CHECK(w && ws);
        for (auto dtype : {RK_NPU_W4A8_F32, RK_NPU_W4A8_F16}) {
            c.activation_type = dtype;
            Data data(c);
            oracle(data, b, scales);
            rknpu2_matmul_open::tune::Sample t;
            CHECK(!data.run(ws.get(), w.get(), t));
            CHECK(data.same());
        }
        c.activation_type = RK_NPU_W4A8_F32;
        ++count;
    }
    CHECK(count > 0);
    std::printf("PASS all %d panel8 candidates M=%d K=%d kt=%d full INT64 oracle f32/f16\n", count,
                M, K, kt);
    std::fflush(stdout);
}
static void single(rk_npu_ctx *ctx, rk_npu_iommu_domain *domain, const std::string &file) {
    rk_npu_w4a8_autotune_config c;
    rk_npu_w4a8_autotune_config_init(&c, 128, 193, 513);
    c.k_tile_multiplier = 1;
    c.required_k_tile = 256;
    c.allowed_cpu_core_mask = 0xf0;
    c.warmup = 0;
    c.loops = 1;
    c.repeats = 1;
    std::vector<int8_t> b(size_t(c.K) * c.N, 1);
    std::vector<float> scales(c.N, .004f);
    Strategy fast{}, stable{};
    int hit = -1;
    const auto affinity = rknpu2_matmul_open::tune::process_cpu_mask();
    CHECK(!rk_npu_w4a8_autotune_cached_ex(ctx, &c, b.data(), scales.data(), file.c_str(), 1, &fast,
                                          &stable, &hit) &&
          hit == 0);
    CHECK(rknpu2_matmul_open::tune::process_cpu_mask() == affinity);
    const auto original = fast;
    CHECK(!rk_npu_w4a8_autotune_cached_ex(ctx, &c, b.data(), scales.data(), file.c_str(), 0, &fast,
                                          &stable, &hit) &&
          hit == 1);
    CHECK(fast.execution.input_layout == original.execution.input_layout &&
          fast.base.total_us == original.base.total_us);
    for (const auto &s : {fast, stable}) {
        Weight w(rk_npu_w4a8_weights_create_tuned_ex(domain, &s, b.data(), scales.data()),
                 rk_npu_w4a8_weights_free);
        Workspace ws(rk_npu_w4a8_workspace_create_tuned_ex(domain, &s), rk_npu_w4a8_workspace_free);
        CHECK(w && ws);
        Data data(c);
        oracle(data, b, scales);
        rknpu2_matmul_open::tune::Sample t;
        CHECK(!data.run(ws.get(), w.get(), t) && data.same());
        auto bad = s;
        bad.execution.reduce_backend = RK_NPU_W4A8_REDUCE_NPU;
        CHECK(!rk_npu_w4a8_workspace_create_tuned_ex(domain, &bad));
        b[0] = 2;
        CHECK(!rk_npu_w4a8_weights_create_tuned_ex(domain, &s, b.data(), scales.data()));
        b[0] = 1;
    }
    std::puts("PASS extended tune/cache replay, factory validation, caller affinity");
}
static void family_test(rk_npu_ctx *ctx, rk_npu_iommu_domain *domain, const std::string &dir) {
    const int ms[] = {1, 33, 129};
    double frequencies[] = {1, 2, 1};
    rk_npu_w4a8_family_config c;
    rk_npu_w4a8_family_config_init(&c, 193, 513, 1, ms);
    c.frequencies = frequencies;
    c.base.k_tile_multiplier = 1;
    c.base.required_k_tile = 256;
    c.base.allowed_cpu_core_mask = 0xf0;
    c.base.warmup = 0;
    c.base.loops = 1;
    c.base.repeats = 1;
    std::vector<int8_t> b(513 * 193, 1);
    std::vector<float> scale(193, .01f);
    rk_npu_w4a8_family_summary f{}, s{};
    Strategy fs[3]{}, ss[3]{};
    int hits = 0, misses = 0;
    auto run = [&] {
        return rk_npu_w4a8_autotune_family_cached_ex(ctx, &c, b.data(), scale.data(), dir.c_str(),
                                                     0, &f, fs, &s, ss, &hits, &misses);
    };
    CHECK(!run() && hits == 0 && misses == 1);
    c.m_count = 3;
    CHECK(!run() && hits == 1 && misses == 2);
    CHECK(!run() && hits == 3 && misses == 0);
    frequencies[0] = 20;
    CHECK(!run() && hits == 3 && misses == 0);
    Weight w(rk_npu_w4a8_weights_create_tuned_ex(domain, &fs[0], b.data(), scale.data()),
             rk_npu_w4a8_weights_free);
    CHECK(w);
    for (int i = 0; i < 3; ++i)
        for (const auto &selected : {fs[i], ss[i]}) {
            CHECK(selected.base.config.k_tile == f.weights.config.k_tile);
            Workspace ws(rk_npu_w4a8_workspace_create_tuned_ex(domain, &selected),
                         rk_npu_w4a8_workspace_free);
            CHECK(ws);
            auto dc = c.base;
            dc.M = ms[i];
            Data data(dc);
            oracle(data, b, scale);
            rknpu2_matmul_open::tune::Sample t;
            CHECK(!data.run(ws.get(), w.get(), t) && data.same());
        }
    std::puts(
        "PASS extended family shared weight, mixed M/layout, partial cache reuse and frequencies");
}
int main(int argc, char **) {
    char dir[] = "/tmp/rk_w4_layout_board_XXXXXX";
    if (!mkdtemp(dir))
        return 1;
    try {
        CHECK(rknpu2_matmul_open::tune::set_thread_cpu_mask(0xf0));
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),
                                                                 rk_npu_close);
        CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(), 0), rk_npu_iommu_domain_free);
        CHECK(domain);
        forced(domain.get(), 32, 33, 32);
        if (argc == 1) {
            forced(domain.get(), 65, 993, 992);
            forced(domain.get(), 127, 2049, 2048);
            forced(domain.get(), 129, 2049, 2048);
            forced(domain.get(), 257, 993, 992);
            single(ctx.get(), domain.get(), std::string(dir) + "/single");
            family_test(ctx.get(), domain.get(), std::string(dir) + "/family");
        }
        std::filesystem::remove_all(dir);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        std::filesystem::remove_all(dir);
        return 1;
    }
}

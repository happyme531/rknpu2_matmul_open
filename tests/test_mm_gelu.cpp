#include "../src/rk_npu_half_bits.h"
#include "rk_npu_mm_gelu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace bits = rknpu2_matmul_open::bits;
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x))                                                                                            \
            throw std::runtime_error(std::string(#x) + " line " + std::to_string(__LINE__));                 \
    } while (0)
using Work = std::unique_ptr<rk_npu_mm_gelu_workspace, decltype(&rk_npu_mm_gelu_free)>;
using Weight = std::unique_ptr<rk_npu_mm_tuned_weights, decltype(&rk_npu_mm_tuned_weights_free)>;
struct Device {
    rk_npu_ctx *ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain *domain = ctx ? rk_npu_iommu_domain_create(ctx, 12) : nullptr;
    Device() { CHECK(ctx && domain); }
    ~Device() {
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
    }
};
double gelu(double x) { return .5 * x * (1. + std::tanh(.7978845608028654 * (x + .044715 * x * x * x))); }
void board(Device &d, int m, int n, int k, uint32_t mask, int nt) {
    rk_npu_mm_gelu_config cfg;
    rk_npu_mm_gelu_config_init(&cfg, m, n, k);
    cfg.core_mask = mask;
    cfg.n_tile = nt;
    rk_npu_mm_gelu_workspace *raw = nullptr;
    CHECK(!rk_npu_mm_gelu_prepare(d.domain, &cfg, &raw));
    Work work(raw, rk_npu_mm_gelu_free);
    std::vector<uint16_t> a(size_t(m) * k), b(size_t(k) * n);
    std::vector<float> bias(n), previous(size_t(m) * n);
    uint32_t rng = 0x31415926;
    auto random = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    for (auto &v : a)
        v = bits::float_to_half((int(random() % 127) - 63) / 32.f);
    Weight weights[2] = {Weight(nullptr, rk_npu_mm_tuned_weights_free),
                         Weight(nullptr, rk_npu_mm_tuned_weights_free)};
    std::vector<std::vector<uint16_t>> sources;
    for (int layer = 0; layer < 2; ++layer) {
        for (auto &v : b)
            v = bits::float_to_half((int(random() % 127) - 63) / 128.f);
        sources.push_back(b);
        rk_npu_mm_autotune_config c;
        rk_npu_mm_autotune_config_init(&c, m, n, k);
        c.allowed_npu_core_mask = mask;
        rk_npu_mm_tuned_weights *w = nullptr;
        CHECK(!rk_npu_mm_tuned_weights_create(d.domain, &c, b.data(), b.size() * 2, nullptr, &w));
        weights[layer].reset(w);
    }
    for (int layer : {0, 1, 0}) {
        for (int j = 0; j < n; ++j)
            bias[j] = (j - n / 2) / 64.f + layer * .125f;
        rk_npu_mm_gelu_view view{};
        CHECK(rk_npu_mm_gelu_run_weights(work.get(), a.data(), 0, weights[layer].get(), bias.data(), &view) ==
              RK_NPU_ERR_NOMEM);
        CHECK(!view.data);
        CHECK(!rk_npu_mm_gelu_run_weights(work.get(), a.data(), a.size() * 2, weights[layer].get(),
                                          bias.data(), &view));
        CHECK(view.rows == m && view.columns == n && view.storage_rows >= m && view.storage_columns >= n);
        double max_error = 0;
        for (int r = 0; r < m; ++r)
            for (int j = 0; j < n; ++j) {
                double sum = bias[j];
                for (int inner = 0; inner < k; ++inner)
                    sum += double(bits::half_to_float(a[size_t(r) * k + inner])) *
                           bits::half_to_float(sources[layer][size_t(inner) * n + j]);
                const size_t index = (r / 8) * 8 * view.storage_columns + (j / 4 * 8 + r % 8) * 4 + j % 4;
                CHECK(index < view.elements);
                const float value = view.data[index] * view.scale;
                const double error = std::abs(value - gelu(sum));
                max_error = std::max(max_error, error);
                if (!std::isfinite(value) || error > .003 + .003 * std::abs(gelu(sum))) {
                    std::fprintf(stderr, "r=%d n=%d sum=%.9g actual=%.9g expected=%.9g error=%.9g\n", r, j,
                                 sum, value, gelu(sum), error);
                    CHECK(false);
                }
                if (layer == 0) {
                    if (previous[size_t(r) * n + j] != 0)
                        CHECK(previous[size_t(r) * n + j] == value);
                    previous[size_t(r) * n + j] = value;
                }
            }
        std::printf("GELU M=%d N=%d K=%d mask=%u nt=%d layer=%d max_abs=%.8g PASS\n", m, n, k, mask, nt,
                    layer, max_error);
    }
}
void tails(Device &d) {
    const int m = 13, n = 35, k = 37;
    const float anchors[] = {-65504.f, -1000.f, -32.f, -4.f, -1.f,   -0.f,   0.f,
                             .001f,    1.f,     4.f,   16.f, 1000.f, 65504.f};
    std::vector<uint16_t> a(m * k), b(k * n);
    std::vector<float> bias(n);
    for (int r = 0; r < m; ++r)
        for (int i = 0; i < k; ++i)
            a[r * k + i] = bits::float_to_half(anchors[r]);
    for (int j = 0; j < n; ++j)
        b[(j % k) * n + j] = bits::float_to_half(1.f);
    rk_npu_mm_autotune_config wc;
    rk_npu_mm_autotune_config_init(&wc, m, n, k);
    rk_npu_mm_tuned_weights *wr = nullptr;
    CHECK(!rk_npu_mm_tuned_weights_create(d.domain, &wc, b.data(), b.size() * 2, nullptr, &wr));
    Weight weight(wr, rk_npu_mm_tuned_weights_free);
    rk_npu_mm_gelu_config cfg;
    rk_npu_mm_gelu_config_init(&cfg, m, n, k);
    rk_npu_mm_gelu_workspace *raw = nullptr;
    CHECK(!rk_npu_mm_gelu_prepare(d.domain, &cfg, &raw));
    Work work(raw, rk_npu_mm_gelu_free);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int j = 0; j < n; ++j)
            bias[j] = (j % 7 - 3) / 128.f;
        rk_npu_mm_gelu_view view{};
        CHECK(!rk_npu_mm_gelu_run_weights(work.get(), a.data(), a.size() * 2, weight.get(), bias.data(),
                                          &view));
        for (int r = 0; r < m; ++r)
            for (int j = 0; j < n; ++j) {
                const float x = bits::half_to_float(a[r * k]) + bias[j];
                const float value =
                    view.data[(r / 8) * 8 * view.storage_columns + (j / 4 * 8 + r % 8) * 4 + j % 4] *
                    view.scale;
                CHECK(std::isfinite(value) && std::abs(value - gelu(x)) < .002 + .00001 * std::abs(x));
            }
        std::reverse(a.begin(), a.end());
    }
    std::puts("GELU large tails / zero / padded rows PASS");
}
int main(int argc, char **argv) {
    try {
        const bool cpu = argc == 2 && !std::strcmp(argv[1], "--cpu");
        CHECK(cpu || (argc == 2 && !std::strcmp(argv[1], "--board")));
        rk_npu_mm_gelu_config c;
        rk_npu_mm_gelu_config_init(&c, 3, 35, 37);
        CHECK(c.M == 3 && c.N == 35 && c.K == 37 && c.core_mask == 7);
        rk_npu_mm_gelu_workspace *bad = reinterpret_cast<rk_npu_mm_gelu_workspace *>(1);
        CHECK(rk_npu_mm_gelu_prepare(nullptr, &c, &bad) == RK_NPU_ERR_PARAM && !bad);
        rk_npu_mm_gelu_view v{};
        v.data = reinterpret_cast<const float *>(1);
        CHECK(rk_npu_mm_gelu_run(nullptr, nullptr, 0, nullptr, nullptr, &v) == RK_NPU_ERR_PARAM && !v.data);
        if (cpu) {
            std::puts("GELU CPU contract PASS");
            return 0;
        }
        Device d;
        for (auto mask : {1u, 2u, 4u, 3u, 7u}) {
            board(d, 3, 35, 37, mask, 32);
            board(d, 129, 160, 96, mask, 0);
        }
        board(d, 260, 4096, 1024, 7, 1376);
        tails(d);
        std::puts("GELU board PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

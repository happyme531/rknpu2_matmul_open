#include "rk_npu_w4a8.h"
#include "../src/rk_npu_w4a8_cpu.h"
#include "../src/rk_npu_w4a8_swiglu_neon.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <time.h>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
namespace detail = rknpu2_matmul_open::detail;
using Weight = std::unique_ptr<rk_npu_w4a8_weights, decltype(&rk_npu_w4a8_weights_free)>;
using Workspace = std::unique_ptr<rk_npu_w4a8_workspace, decltype(&rk_npu_w4a8_workspace_free)>;
using Clock = std::chrono::steady_clock;

static void activation(const float *gu, float *out, int M, int F, int threads) {
#pragma omp parallel for num_threads(threads) collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) for (int block = 0; block < (F + 63) / 64; ++block) {
        const auto *g = gu + size_t(m) * 2 * F;
        auto *y = out + size_t(m) * F;
        const int end = std::min(F, block * 64 + 64);
        int n = block * 64;
#if defined(__aarch64__)
        for (; n + 16 <= end; n += 16)
            detail::w4swiglu::apply16<false>(g + n, g + F + n, y + n);
#endif
        for (; n < end; ++n) y[n] = (g[n] / (1.f + std::exp(-g[n]))) * g[F + n];
    }
}
static void compare(const float *a, const float *b, size_t n) {
    if (std::memcmp(a, b, n * sizeof(float))) {
        for (size_t i = 0; i < n; ++i) if (std::memcmp(a + i, b + i, sizeof(float))) {
            std::fprintf(stderr, "mismatch i=%zu expected=%a actual=%a\n", i, double(a[i]), double(b[i]));
            throw std::runtime_error("fused output differs from ordinary projection + shared SwiGLU");
        }
    }
}
static void scalar_error() {
    constexpr int F = 8192;
    std::vector<float> gu(2 * F), out(F);
    std::mt19937 rng(20261011);
    for (int n = 0; n < F; ++n) {
        gu[n] = float(int(rng() % 1600001) - 800000) * .0001f;
        gu[F + n] = float(int(rng() % 20001) - 10000) * .01f;
    }
    // Cover both ends of the FP32 exponential domain and large finite gates
    // that previously wrapped the constructed exponent without saturation.
    for (int n = 0; n < 32; ++n) gu[n] = n % 2 ? -80.f : 80.f;
    const float extremes[] = {-100.f, 100.f, -10000.f, 10000.f,
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    for (int i = 0; i < int(sizeof(extremes) / sizeof(*extremes)); ++i) {
        gu[32 + i] = extremes[i];
        gu[F + 32 + i] = .5f;
    }
    activation(gu.data(), out.data(), 1, F, 4);
    double largest = 0;
    for (int n = 0; n < F; ++n) {
        const float expected = (gu[n] / (1.f + std::exp(-gu[n]))) * gu[F + n];
        const double relative = std::fabs(double(out[n]) - expected) /
                                std::max(1e-30, std::fabs(double(expected)));
        largest = std::max(largest, relative);
        CHECK(relative <= 7e-7);
    }
    std::printf("PASS activation scalar maximum relative error %.9g\n", largest);
}
static void native_reduction(int M, int F, int nt, int waves, int threads, bool panel) {
    rk_npu_i4_config cfg{2 * M, 2 * F, waves * 480, panel ? 16 : 128, 480, nt, F < 96 ? 1u : 7u, 500, 0, threads};
    uint32_t starts[3]{}, counts[3]{};
    const auto tiles = detail::i4_tiles(cfg, starts, counts,
        panel ? detail::I4InputLayout::Panel8 : detail::I4InputLayout::Native);
    const auto pairs = detail::w4a8_swiglu_pairs(2 * F, tiles);
    const size_t elements = size_t(M) * 2 * F;
    std::vector<int16_t> partial(elements * 2);
    std::vector<int32_t> acc(elements + 32, 0x12345678), fused_acc(acc), correction(2 * F);
    std::vector<float> as(M), ws(2 * F), gu(elements), expected(size_t(M) * F);
    std::vector<float> out(expected.size() + 32, -12345.f);
    std::mt19937 rng(20261011 + M + F + waves);
    for (int n = 0; n < 2 * F; ++n) {
        correction[n] = (1 << 24) + int(rng() % 65536) - 32768;
        ws[n] = .000001f + float(n % 13) * .0000001f;
    }
    for (int m = 0; m < M; ++m) as[m] = .01f + float(m % 11) * .001f;
    detail::W4A8Reduction normal{acc.data() + 16, correction.data(), as.data(), ws.data(), gu.data(), false};
    detail::W4A8Reduction fused{fused_acc.data() + 16, correction.data(), as.data(), ws.data(), out.data() + 16, false, &pairs};
    const int stride = M + 16;
    std::vector<uint32_t> maxima(size_t(stride) * threads);
    fused.row_maxima = maxima.data();
    fused.row_maxima_stride = stride;
    std::vector<int16_t> first;
    for (int w = 0; w < waves; ++w) {
        for (auto &v : partial) v = int(rng() % 65536) - 32768;
        if (w == 0 && waves == 2) first = partial;
        CHECK(!detail::w4a8_reduce_tiles(normal, cfg, tiles, w, waves, partial.data()));
        CHECK(!detail::w4a8_reduce_tiles(fused, cfg, tiles, w, waves, partial.data()));
    }
    activation(gu.data(), expected.data(), M, F, threads);
    compare(expected.data(), out.data() + 16, expected.size());
    auto check_scales = [&] {
        std::vector<float> scales(M), inverse(M), ref_scales(M), ref_inverse(M);
        CHECK(!detail::w4a8_finish_scales(M, threads, maxima.data(), stride, scales.data(), inverse.data()));
        CHECK(!detail::w4a8_scales(out.data() + 16, false, M, F, threads, ref_scales.data(), ref_inverse.data()));
        compare(ref_scales.data(), scales.data(), M);
        compare(ref_inverse.data(), inverse.data(), M);
    };
    check_scales();
    if (waves == 2) {
        std::fill(maxima.begin(), maxima.end(), 0u);
        std::fill(fused_acc.begin(), fused_acc.end(), 0x12345678);
        CHECK(!detail::w4a8_reduce_two_tiles(fused, cfg, first.data(), partial.data()));
        compare(expected.data(), out.data() + 16, expected.size());
        check_scales();
        for (int32_t v : fused_acc) CHECK(v == 0x12345678);
    }
    for (int i = 0; i < 16; ++i) {
        CHECK(out[i] == -12345.f && out[out.size() - 1 - i] == -12345.f);
        CHECK(fused_acc[i] == 0x12345678 && fused_acc[fused_acc.size() - 1 - i] == 0x12345678);
    }
}
static Weight weight(rk_npu_iommu_domain *domain, int K, int F, int kt, int variant) {
    std::vector<int8_t> b(size_t(K) * 2 * F);
    std::vector<float> scale(2 * F);
    for (int k = 0; k < K; ++k) for (int n = 0; n < 2 * F; ++n)
        b[size_t(k) * 2 * F + n] = int((k * (17 + variant * 2) + n * (11 + variant * 2)) % 3) - 1;
    for (int n = 0; n < 2 * F; ++n) scale[n] = (.003f + float(n % 13) * .000125f) * (variant + 1);
    const rk_npu_i4_weight_config wc{K, 2 * F, kt};
    Weight result(rk_npu_w4a8_weights_create(domain, &wc, b.data(), scale.data()), rk_npu_w4a8_weights_free);
    CHECK(result); return result;
}
static Workspace workspace(rk_npu_iommu_domain *domain, int M, int K, int F, int kt,
                           int nt, int threads, int pipeline, bool panel, bool npu = false, int mt = 0) {
    rk_npu_w4a8_config cfg; rk_npu_w4a8_config_init(&cfg, M, 2 * F, K);
    cfg.k_tile = kt; cfg.m_tile = mt > 0 ? mt : panel ? 8 : std::min(M, 64); cfg.n_tile = nt;
    cfg.cpu_threads = threads; cfg.pipeline = pipeline; cfg.npu_core_mask = 7;
    rk_npu_w4a8_options options; rk_npu_w4a8_options_init(&options);
    options.input_layout = panel ? RK_NPU_W4A8_INPUT_PANEL8 : RK_NPU_W4A8_INPUT_NATIVE;
    options.reduce_backend = npu ? RK_NPU_W4A8_REDUCE_NPU : RK_NPU_W4A8_REDUCE_CPU;
    Workspace result(rk_npu_w4a8_workspace_create_ex(domain, &cfg, &options), rk_npu_w4a8_workspace_free);
    CHECK(result); return result;
}
static void hardware(rk_npu_iommu_domain *domain, int M, int F, int pipeline, int index) {
    const int K = index % 3 == 0 ? 479 : index % 3 == 1 ? 959 : 1057;
    const bool panel = index % 2;
    const int threads = 1 + index % 4, nt = index % 2 ? 2048 : 4096;
    auto w0 = weight(domain, K, F, 480, 0), w1 = weight(domain, K, F, 480, 1);
    auto ws = workspace(domain, M, K, F, 480, nt, threads, pipeline, panel);
    std::vector<float> a(size_t(M) * K), gu(size_t(M) * 2 * F), expected(size_t(M) * F);
    std::vector<float> out(expected.size() + 32, -12345.f);
    for (int trial = 0; trial < 3; ++trial) {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = trial == 1 ? 0.f : float(int(i * (17 + trial) % 255) - 127) * .125f;
        for (const auto *w : {w0.get(), w1.get(), w0.get()}) {
            CHECK(!rk_npu_w4a8_run_f32(ws.get(), w, a.data(), gu.data(), nullptr));
            activation(gu.data(), expected.data(), M, F, threads);
            rk_npu_w4a8_timings t{};
            CHECK(!rk_npu_w4a8_run_swiglu_f32(ws.get(), w, a.data(), out.data() + 16, &t));
            compare(expected.data(), out.data() + 16, expected.size());
            CHECK(t.reduce_us > 0 && t.submit_us > 0);
            for (int i = 0; i < 16; ++i) CHECK(out[i] == -12345.f && out[out.size() - 1 - i] == -12345.f);
        }
    }
    a[0] = std::numeric_limits<float>::quiet_NaN(); rk_npu_w4a8_timings t{};
    CHECK(rk_npu_w4a8_run_swiglu_f32(ws.get(), w0.get(), a.data(), out.data() + 16, &t) == RK_NPU_ERR_PARAM);
    CHECK(t.submit_us == 0);
    a[0] = 0;
    CHECK(!rk_npu_w4a8_run_swiglu_f32(ws.get(), w0.get(), a.data(), out.data() + 16, nullptr));
    CHECK(!rk_npu_w4a8_run_f32(ws.get(), w0.get(), a.data(), gu.data(), nullptr));
    activation(gu.data(), expected.data(), M, F, threads); compare(expected.data(), out.data() + 16, expected.size());
    std::printf("PASS fused M=%d K=%d F=%d nt=%d threads=%d pipeline=%d panel=%d waves=%d rebind/zero/repeat/guards\n",
                M, K, F, nt, threads, pipeline, int(panel), (K + 479) / 480);
}
static void reject(rk_npu_iommu_domain *domain) {
    std::vector<float> a(32 * 480, 1.f), out(32 * 256, -12345.f);
    auto w = weight(domain, 480, 256, 480, 0);
    auto ws = workspace(domain, 32, 480, 256, 480, 64, 1, 0, false, true);
    rk_npu_w4a8_timings t{};
    CHECK(rk_npu_w4a8_run_swiglu_f32(ws.get(), w.get(), a.data(), out.data(), &t) == RK_NPU_ERR_PARAM);
    CHECK(t.submit_us == 0 && std::all_of(out.begin(), out.end(), [](float x) { return x == -12345.f; }));
    rk_npu_w4a8_config cfg; rk_npu_w4a8_config_init(&cfg, 1, 257, 480);
    cfg.pipeline = 0; cfg.npu_core_mask = 7;
    Workspace bad(rk_npu_w4a8_workspace_create(domain, &cfg), rk_npu_w4a8_workspace_free); CHECK(bad);
    CHECK(rk_npu_w4a8_run_swiglu_f32(bad.get(), w.get(), a.data(), out.data(), &t) == RK_NPU_ERR_PARAM);
    CHECK(t.submit_us == 0);
    CHECK(rk_npu_w4a8_run_swiglu_f32(nullptr, w.get(), a.data(), out.data(), &t) == RK_NPU_ERR_PARAM);
    CHECK(rk_npu_w4a8_run_swiglu_f32(bad.get(), w.get(), a.data(), a.data(), &t) == RK_NPU_ERR_PARAM);
    // M<=16 keeps the documented effective CPU backend even if NPU was requested.
    auto small = workspace(domain, 1, 480, 256, 480, 64, 2, 0, false, true);
    CHECK(!rk_npu_w4a8_run_swiglu_f32(small.get(), w.get(), a.data(), out.data(), &t));
    std::puts("PASS unsupported N/effective NPU before submit; small-M effective CPU");
}
static double cpu_us() {
    timespec t{}; CHECK(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t));
    return t.tv_sec * 1e6 + t.tv_nsec * .001;
}
static void bench(rk_npu_iommu_domain *domain, int argc, char **argv) {
    CHECK(argc >= 9);
    const int M = std::atoi(argv[2]), K = std::atoi(argv[3]), F = std::atoi(argv[4]);
    const int threads = std::atoi(argv[5]), pipeline = std::atoi(argv[6]);
    const bool panel = std::atoi(argv[7]); const int nt = std::atoi(argv[8]);
    const int blocks = argc > 9 ? std::atoi(argv[9]) : 8, loops = argc > 10 ? std::atoi(argv[10]) : 20;
    const int kt = argc > 11 ? std::atoi(argv[11]) : 2048;
    const int mt = argc > 12 ? std::atoi(argv[12]) : 0;
    const int active_rows = argc > 13 ? std::atoi(argv[13]) : M;
    CHECK(M > 0 && K > 0 && F > 0 && blocks > 0 && loops > 0 && active_rows > 0 && active_rows <= M);
    auto w = weight(domain, K, F, kt, 0);
    auto ws = workspace(domain, M, K, F, kt, nt, threads, pipeline, panel, false, mt);
    std::vector<float> a(size_t(M) * K), gu(size_t(M) * 2 * F), reference(size_t(M) * F), out(reference.size());
    for (size_t i = 0; i < size_t(active_rows) * K; ++i) a[i] = float(int(i * 17 % 255) - 127) * .125f;
    CHECK(!rk_npu_w4a8_run_f32(ws.get(), w.get(), a.data(), gu.data(), nullptr));
    activation(gu.data(), reference.data(), active_rows, F, threads);
    CHECK(!rk_npu_w4a8_run_swiglu_f32(ws.get(), w.get(), a.data(), out.data(), nullptr));
    compare(reference.data(), out.data(), size_t(active_rows) * F);
    std::puts("block,order,backend,M,active_rows,K,F,threads,pipeline,panel,n_tile,k_tile,loops,wall_us,cpu_us,activation_scan_us,quant_pack_us,sync_us,submit_us,reduce_us,swiglu_us");
    for (int block = 0; block < blocks; ++block) for (int order = 0; order < 2; ++order) {
        const bool fused = (block % 2 == 0) ? order == 1 : order == 0;
        rk_npu_w4a8_timings sum{}; double act = 0;
        const auto begin = Clock::now(); const double cpu_begin = cpu_us();
        for (int loop = 0; loop < loops; ++loop) {
            rk_npu_w4a8_timings t{};
            if (fused) CHECK(!rk_npu_w4a8_run_swiglu_f32(ws.get(), w.get(), a.data(), out.data(), &t));
            else {
                CHECK(!rk_npu_w4a8_run_f32(ws.get(), w.get(), a.data(), gu.data(), &t));
                const auto start = Clock::now(); activation(gu.data(), out.data(), active_rows, F, threads);
                act += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            }
            sum.activation_scan_us += t.activation_scan_us; sum.quant_pack_us += t.quant_pack_us;
            sum.sync_us += t.sync_us; sum.submit_us += t.submit_us; sum.reduce_us += t.reduce_us;
        }
        const double cpu = (cpu_us() - cpu_begin) / loops;
        const double wall = std::chrono::duration<double, std::micro>(Clock::now() - begin).count() / loops;
        compare(reference.data(), out.data(), size_t(active_rows) * F);
        std::printf("%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            block, order, fused ? "fused" : "separate", M, active_rows, K, F, threads, pipeline, int(panel), nt, kt, loops,
            wall, cpu, sum.activation_scan_us / loops, sum.quant_pack_us / loops,
            sum.sync_us / loops, sum.submit_us / loops, sum.reduce_us / loops, act / loops);
    }
}
int main(int argc, char **argv) {
    try {
        if (argc > 1 && !std::strcmp(argv[1], "--cpu")) {
            scalar_error();
            for (int M : {1, 2, 4, 8, 64, 65, 128}) for (int F : {32, 96, 128, 9728})
                for (int threads : {1, 2, 3, 4}) for (int waves : {1, 2, 5})
                    native_reduction(M, F, threads % 2 ? 2048 : 4096, waves, threads, threads % 2);
            std::puts("PASS fused native reduction: full bitwise output, guards, >2^24, panel/tail, waves1/2/5 threads1..4");
            return 0;
        }
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr), rk_npu_close); CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(), 0), rk_npu_iommu_domain_free); CHECK(domain);
        if (argc > 1 && !std::strcmp(argv[1], "--bench")) { bench(domain.get(), argc, argv); return 0; }
        int index = 0;
        for (int M : {1, 2, 4, 8, 64, 65, 128}) for (int F : {128, 9728})
            for (int pipeline : {0, 1, 2}) hardware(domain.get(), M, F, pipeline, index++);
        for (int pipeline : {0, 1, 2}) hardware(domain.get(), 65, 96, pipeline, index++);
        reject(domain.get());
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}

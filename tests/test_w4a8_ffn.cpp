#include "rk_npu_w4a8.h"
#include "../src/rk_npu_w4a8_cpu.h"
#include "../src/rk_npu_i4_executor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
namespace detail = rknpu2_matmul_open::detail;
using Weight = std::unique_ptr<rk_npu_w4a8_weights, decltype(&rk_npu_w4a8_weights_free)>;
using Workspace = std::unique_ptr<rk_npu_w4a8_workspace, decltype(&rk_npu_w4a8_workspace_free)>;

static void scale_boundaries() {
    uint32_t maxima[2][5] = {{0, 1, 0x00800000, 0x3f800000, 0},
                              {0, 0, 0, 0x40000000, 0}};
    float scales[4], inverse[4];
    CHECK(!detail::w4a8_finish_scales(4, 2, maxima[0], 5, scales, inverse));
    CHECK(scales[0] == 1 && inverse[0] == 1);
    CHECK(scales[1] == std::numeric_limits<float>::min());
    CHECK(scales[2] == std::numeric_limits<float>::min());
    CHECK(scales[3] == 2.f / 127.f && inverse[3] == 1.f / scales[3]);
    for (uint32_t bad : {0x7f800000u, 0x7fc00001u}) {
        maxima[1][0] = bad;
        CHECK(detail::w4a8_finish_scales(4, 2, maxima[0], 5, scales, inverse) == RK_NPU_ERR_PARAM);
    }
    rk_npu_w4a8_ffn_timings t{};
    CHECK(rk_npu_w4a8_run_ffn_f32(nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, &t) == RK_NPU_ERR_PARAM);
    CHECK(t.total_us == 0 && t.gate_up.submit_us == 0 && t.down.submit_us == 0);
    std::puts("PASS FFN row scales: zero/subnormal/FLT_MIN/nonfinite and null API");
}

static Weight weight(rk_npu_iommu_domain *domain, int K, int N, bool overflow = false) {
    std::vector<int8_t> codes(size_t(K) * N);
    std::vector<float> scales(N);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n)
        codes[size_t(k) * N + n] = overflow ? 1 : int((k * 17 + n * 13) % 3) - 1;
    for (int n = 0; n < N; ++n)
        scales[n] = overflow ? std::numeric_limits<float>::max() : .002f + (n % 7) * .0001f;
    const rk_npu_i4_weight_config c{K, N, 480};
    Weight w(rk_npu_w4a8_weights_create(domain, &c, codes.data(), scales.data()), rk_npu_w4a8_weights_free);
    CHECK(w); return w;
}

static Workspace workspace(rk_npu_iommu_domain *domain, int M, int K, int N,
                            int pipeline, bool panel, int threads, bool npu = false) {
    rk_npu_w4a8_config c;
    rk_npu_w4a8_config_init(&c, M, N, K);
    c.k_tile = 480; c.n_tile = 128; c.m_tile = panel ? 8 : 32;
    c.cpu_threads = threads; c.pipeline = pipeline; c.npu_core_mask = 7;
    rk_npu_w4a8_options o; rk_npu_w4a8_options_init(&o);
    o.input_layout = panel ? RK_NPU_W4A8_INPUT_PANEL8 : RK_NPU_W4A8_INPUT_NATIVE;
    o.reduce_backend = npu ? RK_NPU_W4A8_REDUCE_NPU : RK_NPU_W4A8_REDUCE_CPU;
    Workspace ws(rk_npu_w4a8_workspace_create_ex(domain, &c, &o), rk_npu_w4a8_workspace_free);
    CHECK(ws); return ws;
}

static void one(rk_npu_iommu_domain *domain, int M, int K, int F, int pipeline,
                 bool panel, int threads, bool npu_down = false) {
    constexpr int N = 192;
    auto gu = weight(domain, K, 2 * F), dw = weight(domain, F, N);
    auto up = workspace(domain, M, K, 2 * F, pipeline, panel, threads);
    auto down = workspace(domain, M, F, N, npu_down ? 1 : (pipeline + 1) % 3, !panel, threads, npu_down);
    std::vector<float> input(size_t(M) * K), mid(size_t(M) * F), ref_mid(mid.size());
    std::vector<float> out(size_t(M) * N + 32, -12345.f), ref(size_t(M) * N);
    for (int pattern = 0; pattern < 4; ++pattern) {
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = pattern == 1 ? 0.f : float(int(i * 17 % 255) - 127) * (pattern == 2 ? .5f : pattern == 3 ? -.125f : .125f);
        CHECK(!rk_npu_w4a8_run_swiglu_f32(up.get(), gu.get(), input.data(), ref_mid.data(), nullptr));
        CHECK(!rk_npu_w4a8_run_f32(down.get(), dw.get(), ref_mid.data(), ref.data(), nullptr));
        rk_npu_w4a8_ffn_timings t{};
        CHECK(!rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), dw.get(),
            input.data(), mid.data(), out.data() + 16, &t));
        CHECK(!std::memcmp(mid.data(), ref_mid.data(), mid.size() * sizeof(float)));
        CHECK(!std::memcmp(out.data() + 16, ref.data(), ref.size() * sizeof(float)));
        CHECK(t.total_us > 0 && t.gate_up.submit_us > 0 && t.down.submit_us > 0);
        CHECK(t.down.activation_scan_us == 0);
        for (int i = 0; i < 16; ++i) CHECK(out[i] == -12345.f && out[out.size() - 1 - i] == -12345.f);
    }
    rk_npu_w4a8_ffn_timings t{};
    CHECK(rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), up.get(), gu.get(),
        input.data(), mid.data(), out.data() + 16, &t) == RK_NPU_ERR_PARAM);
    CHECK(t.gate_up.submit_us == 0 && t.down.submit_us == 0);
    CHECK(rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), gu.get(),
        input.data(), mid.data(), out.data() + 16, &t) == RK_NPU_ERR_PARAM);
    CHECK(t.gate_up.submit_us == 0 && t.down.submit_us == 0);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), dw.get(),
        input.data(), mid.data(), out.data() + 16, &t) == RK_NPU_ERR_PARAM);
    CHECK(t.gate_up.submit_us == 0 && t.down.submit_us == 0);
    input[0] = 0;
    CHECK(!rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), dw.get(),
        input.data(), mid.data(), out.data() + 16, nullptr));
    std::printf("PASS FFN M=%d K=%d F=%d pipeline=%d panel=%d threads=%d npu_down=%d: exact scale reuse/repeat/failure\n",
        M, K, F, pipeline, int(panel), threads, int(npu_down));
}

static void nonfinite_middle(rk_npu_iommu_domain *domain) {
    auto gu = weight(domain, 959, 256, true), dw = weight(domain, 128, 192);
    auto up = workspace(domain, 4, 959, 256, 1, true, 2);
    auto down = workspace(domain, 4, 128, 192, 1, false, 2);
    std::vector<float> input(4 * 959, 1.f), mid(4 * 128), out(4 * 192);
    rk_npu_w4a8_ffn_timings t{};
    CHECK(rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), dw.get(),
        input.data(), mid.data(), out.data(), &t) == RK_NPU_ERR_PARAM);
    CHECK(t.gate_up.submit_us > 0 && t.down.submit_us == 0);
    std::fill(input.begin(), input.end(), 0.f);
    CHECK(!rk_npu_w4a8_run_ffn_f32(up.get(), gu.get(), down.get(), dw.get(),
        input.data(), mid.data(), out.data(), nullptr));
    for (float y : out) CHECK(y == 0.f);
    std::puts("PASS nonfinite middle rejected before Down; same workspaces reusable");
}

static void consumer_failure(rk_npu_iommu_domain *domain) {
    const rk_npu_i4_config cfg{8, 192, 959, 8, 480, 64, 7, 500, 1, 2};
    std::vector<int8_t> codes(959 * 192, 1);
    detail::I4Weights weights;
    CHECK(!weights.prepare(domain, {959, 192, 480}, codes.data()));
    detail::I4Executor executor;
    CHECK(!executor.prepare(domain, cfg));
    std::vector<float> a(4 * 959, 1.f), scales(4, 1.f), inverse(4, 1.f);
    detail::W4A8Input input{a.data(), false, 4, 959, scales.data(), inverse.data(), 2};
    struct State { int mode, calls; };
    const auto consume = +[](void *user, const detail::I4Plan&, const int16_t *p0, const int16_t *p1) {
        auto &s = *static_cast<State *>(user);
        CHECK(p0 && p1 && p0 != p1);
        ++s.calls;
        if (s.mode == 1) throw std::runtime_error("test consumer failure");
        return s.mode == 0 ? RK_NPU_ERR_PARAM : RK_NPU_OK;
    };
    for (int mode : {0, 2, 1, 2}) {
        State state{mode, 0};
        const int rc = executor.run(weights, detail::w4a8_pack, &input, nullptr, nullptr,
            nullptr, &state, consume);
        CHECK(rc == (mode == 2 ? RK_NPU_OK : RK_NPU_ERR_PARAM));
        CHECK(state.calls == 1);
    }
    std::puts("PASS two-partial consumer return/exception failures and same-executor recovery");
}

int main(int argc, char **argv) {
    try {
        scale_boundaries();
        if (argc > 1 && !std::strcmp(argv[1], "--cpu")) return 0;
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr), rk_npu_close); CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(), 0), rk_npu_iommu_domain_free); CHECK(domain);
        for (int pipeline : {0, 1, 2}) {
            one(domain.get(), 1, 479, 96, pipeline, false, 1);
            one(domain.get(), 9, 959, 128, pipeline, true, 2);
            one(domain.get(), 65, 1057, 992, pipeline, true, 4);
        }
        nonfinite_middle(domain.get());
        consumer_failure(domain.get());
        one(domain.get(), 32, 959, 128, 1, true, 4, true);
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}

#include "rk_npu_matmul_f16.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t f2h(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float h2f(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

struct Case {
    int B, M, N, K, n_tile;
    uint32_t core_mask;
    const char* name;
    bool a_native = false;
    bool d_native = false;
};

int run_case(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
             const Case& c, int loops, std::mt19937& rng) {
    rk_npu_matmul_f16_config cfg{};
    rk_npu_matmul_f16_config_init(
        &cfg, c.M, c.N, c.K, RK_NPU_FUSE_NONE);
    cfg.n_tile = c.n_tile;
    cfg.core_mask = c.core_mask;
    cfg.a_layout = c.a_native ? RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8
                              : RK_NPU_F16_A_LAYOUT_NORMAL;
    cfg.d_layout = c.d_native ? RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8
                              : RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;

    rk_npu_matmul_sizes sz{};
    int rc = rk_npu_matmul_f16_batch_query(c.B, &cfg, &sz);
    if (rc != RK_NPU_OK) {
        std::printf("%-18s QUERY_FAIL %s\n", c.name, rk_npu_strerror(rc));
        return 1;
    }
    rk_npu_mem in{}, wt{}, out{};
    if (rk_npu_mem_alloc(domain, sz.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &in) ||
        rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_NON_CACHEABLE, &wt) ||
        rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_NON_CACHEABLE, &out)) {
        std::printf("%-18s ALLOC_FAIL\n", c.name);
        if (in.handle) rk_npu_mem_free(ctx, &in);
        if (wt.handle) rk_npu_mem_free(ctx, &wt);
        if (out.handle) rk_npu_mem_free(ctx, &out);
        return 1;
    }

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<uint16_t> A((size_t)c.B * c.M * c.K);
    std::vector<uint16_t> W((size_t)c.B * c.K * c.N);
    std::vector<uint16_t> D((size_t)c.B * c.M * c.N);
    std::vector<float> D32((size_t)c.M * c.N);
    for (auto& x : A) x = f2h(dist(rng));
    for (auto& x : W) x = f2h(dist(rng));

    rc = rk_npu_matmul_f16_batch_pack_a(c.B, &cfg, A.data(), &in);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_pack_b(c.B, &cfg, W.data(), &wt);
    rk_npu_matmul_f16_batch_plan* plan =
        rc == RK_NPU_OK
            ? rk_npu_matmul_f16_batch_prepare(domain, c.B, &cfg)
            : nullptr;
    if (!plan) rc = RK_NPU_ERR_NOMEM;
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_run(ctx, plan, &in, &wt, &out);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_unpack_d(c.B, &cfg, &out, D.data());
    if (rc == RK_NPU_OK) {
        rk_npu_matmul_sizes one{};
        rk_npu_mem first{};
        rc = rk_npu_matmul_f16_query(&cfg, &one);
        if (rc == RK_NPU_OK)
            rc = rk_npu_mem_view(&out, 0, one.output_bytes, &first);
        if (rc == RK_NPU_OK)
            rc = rk_npu_matmul_f16_unpack_d_f32(&cfg, &first, D32.data());
    }

    double max_abs = 0.0, max_ref = 1.0;
    if (rc == RK_NPU_OK) {
        for (int b = 0; b < c.B; ++b) {
            for (int m = 0; m < c.M; ++m) {
                for (int n = 0; n < c.N; ++n) {
                    double ref = 0.0;
                    for (int k = 0; k < c.K; ++k) {
                        ref += (double)h2f(A[((size_t)b * c.M + m) * c.K + k]) *
                               h2f(W[((size_t)b * c.K + k) * c.N + n]);
                    }
                    max_abs = std::max(max_abs, std::fabs(
                        (double)h2f(D[((size_t)b * c.M + m) * c.N + n]) - ref));
                    if (b == 0) {
                        max_abs = std::max(max_abs, std::fabs(
                            (double)D32[(size_t)m * c.N + n] - ref));
                    }
                    max_ref = std::max(max_ref, std::fabs(ref));
                }
            }
        }
    }
    const bool ok = rc == RK_NPU_OK && max_abs < 0.03 * max_ref + 0.05;

    double avg_us = 0.0;
    if (ok && loops > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < loops; ++i)
            rc = rk_npu_matmul_f16_batch_run(ctx, plan, &in, &wt, &out);
        const auto t1 = std::chrono::steady_clock::now();
        avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                 loops;
    }
    std::printf("%-18s B=%d M=%d K=%d N=%d nt=%d mask=0x%x ac=%d%d tasks=%d %s "
                "max_abs=%.5f avg_us=%.2f\n",
                c.name, c.B, c.M, c.K, c.N, c.n_tile, c.core_mask,
                c.a_native ? 1 : 0, c.d_native ? 1 : 0,
                sz.num_tasks, ok ? "PASS" : "FAIL", max_abs, avg_us);

    rk_npu_matmul_f16_batch_plan_free(plan);
    rk_npu_mem_free(ctx, &in);
    rk_npu_mem_free(ctx, &wt);
    rk_npu_mem_free(ctx, &out);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const int loops = argc > 1 ? std::max(0, std::atoi(argv[1])) : 10;
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    if (!ctx) return 2;
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) {
        rk_npu_close(ctx);
        return 2;
    }
    std::mt19937 rng(0x46313642u);
    const Case cases[] = {
        {3, 5, 160, 64, 64, 7, "n_gt_k_3core"},
        {4, 17, 64, 96, 64, 3, "m_tiled_2core"},
        {2, 9, 512, 128, 128, 7, "wide_n_3core"},
        {1, 7, 80, 33, 32, 1, "odd_single"},
        {2, 128, 64, 64, 0, 3, "native_full_m", true, true},
        {2, 123, 96, 1280, 64, 7, "native_split_mn", true, true},
        {1, 26, 64, 5120, 0, 3, "native_a_large_k", true, false},
        {2, 64, 160, 96, 64, 3, "native_d_only", false, true},
    };
    int failed = 0;
    for (const Case& c : cases)
        failed += run_case(ctx, domain, c, loops, rng);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    std::printf("%s\n", failed ? "SOME FAILED" : "ALL PASS");
    return failed ? 1 : 0;
}

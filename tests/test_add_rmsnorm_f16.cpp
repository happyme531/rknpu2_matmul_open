#include "rk_npu_add_rmsnorm_f16.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static uint16_t f16_bits(float value) {
    __fp16 h = (__fp16)value;
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static float f16_float(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(bits));
    return (float)h;
}

static size_t native_index(int M, int D, int m, int d) {
    return ((size_t)(d / 8) * M + m) * 8 + (d % 8);
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 1;
    const int D = 4096;
    const int loops = argc > 2 ? std::atoi(argv[2]) : 30;
    if ((M != 1 && M != 128) || loops <= 0) {
        std::fprintf(stderr, "require M=1/128 and loops > 0\n");
        return 1;
    }
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain = ctx ? rk_npu_iommu_domain_create(ctx, 0) : nullptr;
    if (!ctx || !domain) return 2;

    rk_npu_add_rmsnorm_f16_config cfg;
    rk_npu_add_rmsnorm_f16_config_init(&cfg, M, D);
    rk_npu_add_rmsnorm_f16_sizes sizes{};
    if (rk_npu_add_rmsnorm_f16_query(&cfg, &sizes) != RK_NPU_OK) return 3;
    std::vector<uint16_t> gamma(D);
    for (int d = 0; d < D; ++d)
        gamma[d] = f16_bits(0.75f + 0.5f * d / (D - 1));
    auto* plan = rk_npu_add_rmsnorm_f16_plan_create(domain, &cfg, gamma.data());
    if (!plan) return 4;

    rk_npu_mem x{}, residual{}, residual_out{}, norm_out{};
    auto alloc = [&](rk_npu_mem* mem) {
        return rk_npu_mem_alloc(domain, sizes.tensor_bytes,
                                RK_NPU_MEM_NON_CACHEABLE, mem);
    };
    if (alloc(&x) || alloc(&residual) || alloc(&residual_out) || alloc(&norm_out))
        return 5;
    auto* xp = static_cast<uint16_t*>(x.vaddr);
    auto* rp = static_cast<uint16_t*>(residual.vaddr);
    std::vector<uint16_t> x_row((size_t)M * D), r_row((size_t)M * D);
    std::mt19937 rng(20260827);
    std::normal_distribution<float> dx(0.0f, 0.8f), dr(0.0f, 0.35f);
    for (int m = 0; m < M; ++m) for (int d = 0; d < D; ++d) {
        const size_t row = (size_t)m * D + d;
        x_row[row] = f16_bits(std::clamp(dx(rng), -3.0f, 3.0f));
        r_row[row] = f16_bits(std::clamp(dr(rng), -1.5f, 1.5f));
        xp[native_index(M, D, m, d)] = x_row[row];
        rp[native_index(M, D, m, d)] = r_row[row];
    }

    int rc = rk_npu_add_rmsnorm_f16_run(
        ctx, plan, &x, &residual, &residual_out, &norm_out);
    if (rc != RK_NPU_OK) {
        std::printf("submit failed: %s\n", rk_npu_strerror(rc));
        return 6;
    }
    std::vector<double> samples;
    for (int i = 0; i < loops; ++i) {
        auto start = std::chrono::steady_clock::now();
        rc = rk_npu_add_rmsnorm_f16_run(
            ctx, plan, &x, &residual, &residual_out, &norm_out);
        auto end = std::chrono::steady_clock::now();
        if (rc != RK_NPU_OK) return 7;
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    auto* residual_got = static_cast<uint16_t*>(residual_out.vaddr);
    auto* norm_got = static_cast<uint16_t*>(norm_out.vaddr);
    size_t residual_mismatch = 0;
    double max_abs = 0.0, sq_error = 0.0, dot = 0.0, ga = 0.0, gb = 0.0;
    for (int m = 0; m < M; ++m) {
        std::vector<uint16_t> sum(D);
        double variance = 0.0;
        for (int d = 0; d < D; ++d) {
            const size_t row = (size_t)m * D + d;
            sum[d] = f16_bits(f16_float(x_row[row]) + f16_float(r_row[row]));
            const double v = f16_float(sum[d]);
            variance += v * v;
            if (residual_got[native_index(M, D, m, d)] != sum[d])
                ++residual_mismatch;
        }
        const double inv = 1.0 / std::sqrt(variance / D + cfg.eps);
        for (int d = 0; d < D; ++d) {
            const double ref = f16_float(f16_bits(
                (float)(f16_float(sum[d]) * inv * f16_float(gamma[d]))));
            const double got = f16_float(norm_got[native_index(M, D, m, d)]);
            const double err = std::fabs(got - ref);
            max_abs = std::max(max_abs, err);
            sq_error += err * err;
            dot += got * ref; ga += got * got; gb += ref * ref;
        }
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double rmse = std::sqrt(sq_error / ((size_t)M * D));
    const double cosine = dot / std::sqrt(ga * gb);
    std::printf(
        "ADD_RMSNORM_F16 M=%d D=%d tasks=%d residual_mismatch=%zu "
        "max_abs=%.8g rmse=%.8g cosine=%.10f median_us=%.2f\n",
        M, D, sizes.num_tasks, residual_mismatch, max_abs, rmse, cosine, median);
    const bool pass = residual_mismatch == 0 && max_abs <= 0.04 && cosine >= 0.99999;

    rk_npu_mem_free(ctx, &norm_out);
    rk_npu_mem_free(ctx, &residual_out);
    rk_npu_mem_free(ctx, &residual);
    rk_npu_mem_free(ctx, &x);
    rk_npu_add_rmsnorm_f16_plan_free(plan);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    std::puts(pass ? "RESULT PASS" : "RESULT FAIL");
    return pass ? 0 : 1;
}

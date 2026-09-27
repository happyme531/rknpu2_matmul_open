/* Isolated end-to-end latency of the public i8i8i32 API. */
#include "rk_npu_quant_matmul.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <sched.h>
#include <vector>

namespace {

uint64_t first_cpu() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0)
        for (int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &set)) return 1ull << cpu;
    return 1;
}

rk_npu_matmul_strategy strategy(int M, int N, int K, uint64_t cpu) {
    rk_npu_matmul_strategy s{};
    s.op_kind = RK_NPU_MATMUL_I8I8I32;
    s.M = M; s.N = N; s.K = K;
    s.k_tile = K; s.n_tile = ((N + 31) / 32) * 32;
    s.wave_count = 1; s.n_groups = 1;
    s.npu_core_mask = 1;
    s.cpu_core_mask = cpu; s.cpu_threads = 1;
    return s;
}

} /* namespace */

int main(int argc, char** argv) {
    rk_npu_ctx* ctx = rk_npu_open(argc > 1 ? argv[1] : nullptr);
    if (!ctx) return 1;
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-8, 7);
    struct Shape { int M, N, K; };
    const Shape shapes[] = {
        {4, 32, 64}, {128, 128, 128}, {32, 256, 256},
        {256, 256, 256}, {128, 384, 256}, {512, 256, 128},
    };
    std::printf("%-14s %12s %10s\n", "M,K,N", "total_us", "TOPS");
    for (const Shape& shape : shapes) {
        std::vector<int8_t> A((size_t)shape.M * shape.K);
        std::vector<int8_t> B((size_t)shape.K * shape.N);
        std::vector<int32_t> C((size_t)shape.M * shape.N);
        for (int8_t& x : A) x = (int8_t)dist(rng);
        for (int8_t& x : B) x = (int8_t)dist(rng);
        const auto s = strategy(shape.M, shape.N, shape.K, first_cpu());
        const rk_npu_matmul_weight_config wc{shape.K, shape.N, s.k_tile};
        rk_npu_matmul_workspace* workspace =
            rk_npu_matmul_workspace_create(domain, &s);
        rk_npu_i8i8i32_weights* weights =
            rk_npu_i8i8i32_weights_create(domain, &wc, B.data());
        if (!workspace || !weights) {
            rk_npu_i8i8i32_weights_free(weights);
            rk_npu_matmul_workspace_free(workspace);
            continue;
        }
        for (int i = 0; i < 3; ++i)
            rk_npu_i8i8i32_run(workspace, weights, A.data(), C.data());
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int loops = 100;
        for (int i = 0; i < loops; ++i)
            rk_npu_i8i8i32_run(workspace, weights, A.data(), C.data());
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count() / loops;
        const double tops = 2.0 * shape.M * shape.N * shape.K / us / 1.0e6;
        std::printf("%d,%d,%d %12.2f %10.3f\n",
                    shape.M, shape.K, shape.N, us, tops);
        rk_npu_i8i8i32_weights_free(weights);
        rk_npu_matmul_workspace_free(workspace);
    }
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return 0;
}

/* Public i8i8i32 single-GEMM API correctness demo. */
#include "rk_npu_quant_matmul.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

void reference(int M, int N, int K, const int8_t* A, const int8_t* B,
               std::vector<int32_t>& C) {
    C.assign((size_t)M * N, 0);
    for (int m = 0; m < M; ++m) {
        uint32_t* out = reinterpret_cast<uint32_t*>(C.data() + (size_t)m * N);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                out[n] += (uint32_t)(A[(size_t)m * K + k] *
                                     (int32_t)B[(size_t)k * N + n]);
    }
}

rk_npu_matmul_strategy strategy(int M, int N, int K, uint64_t cpu) {
    rk_npu_matmul_strategy s{};
    s.op_kind = RK_NPU_MATMUL_I8I8I32;
    s.M = M; s.N = N; s.K = K;
    s.k_tile = K;
    s.n_tile = ((N + 31) / 32) * 32;
    s.wave_count = 1;
    s.n_groups = 1;
    s.npu_core_mask = 1;
    s.cpu_core_mask = cpu;
    s.cpu_threads = 1;
    return s;
}

} /* namespace */

int main(int argc, char** argv) {
    rk_npu_ctx* ctx = rk_npu_open(argc > 1 ? argv[1] : nullptr);
    if (!ctx) return 1;
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-8, 8);
    struct Shape { int M, N, K; };
    const Shape shapes[] = {
        {2, 2, 1}, {8, 8, 8}, {9, 9, 9}, {16, 32, 32}, {17, 40, 33},
        {64, 64, 64}, {128, 48, 96}, {1, 200, 200}, {300, 48, 33},
        {256, 256, 256},
    };
    bool ok = true;
    for (const Shape& shape : shapes) {
        std::vector<int8_t> A((size_t)shape.M * shape.K);
        std::vector<int8_t> B((size_t)shape.K * shape.N);
        for (int8_t& x : A) x = (int8_t)dist(rng);
        for (int8_t& x : B) x = (int8_t)dist(rng);
        const auto s = strategy(shape.M, shape.N, shape.K, first_cpu());
        const rk_npu_matmul_weight_config wc{shape.K, shape.N, s.k_tile};
        rk_npu_matmul_workspace* workspace =
            rk_npu_matmul_workspace_create(domain, &s);
        rk_npu_i8i8i32_weights* weights =
            rk_npu_i8i8i32_weights_create(domain, &wc, B.data());
        std::vector<int32_t> got((size_t)shape.M * shape.N), ref;
        const int rc = workspace && weights
            ? rk_npu_i8i8i32_run(workspace, weights, A.data(), got.data())
            : RK_NPU_ERR_NOMEM;
        reference(shape.M, shape.N, shape.K, A.data(), B.data(), ref);
        const bool pass = rc == RK_NPU_OK && got == ref;
        std::printf("M=%d K=%d N=%d %s\n",
                    shape.M, shape.K, shape.N, pass ? "PASS" : "FAIL");
        ok &= pass;
        rk_npu_i8i8i32_weights_free(weights);
        rk_npu_matmul_workspace_free(workspace);
    }
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return ok ? 0 : 1;
}

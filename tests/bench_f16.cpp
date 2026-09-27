#include "rk_npu_matmul_f16.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

uint16_t f2h(float value) {
    __fp16 half = (__fp16)value;
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float h2f(uint16_t bits) {
    __fp16 half;
    std::memcpy(&half, &bits, sizeof(half));
    return (float)half;
}

double elapsed_us(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s M K N [loops=10] [n_tile=0] [core_mask=1] [batch=1] "
                 "[a_native=0] [d_native=0] [cached=0] [split_k=0]\n",
                 argv0);
}

bool parse_positive(const char* text, int* value) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 0);
    if (!text[0] || !end || *end || parsed <= 0 || parsed > INT32_MAX)
        return false;
    *value = (int)parsed;
    return true;
}

bool parse_nonnegative(const char* text, int* value) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 0);
    if (!text[0] || !end || *end || parsed < 0 || parsed > INT32_MAX)
        return false;
    *value = (int)parsed;
    return true;
}

struct OwnedMem {
    rk_npu_mem mem{};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 12) {
        print_usage(argv[0]);
        return 2;
    }

    int M = 0, K = 0, N = 0, loops = 10, n_tile = 0, batch = 1;
    int a_native = 0, d_native = 0, cached = 0, split_k = 0;
    int core_mask = 1;
    if (!parse_positive(argv[1], &M) || !parse_positive(argv[2], &K) ||
        !parse_positive(argv[3], &N) ||
        (argc > 4 && !parse_positive(argv[4], &loops)) ||
        (argc > 5 && !parse_nonnegative(argv[5], &n_tile)) ||
        (argc > 6 && !parse_positive(argv[6], &core_mask)) ||
        (argc > 7 && !parse_positive(argv[7], &batch)) ||
        (argc > 8 && !parse_nonnegative(argv[8], &a_native)) ||
        (argc > 9 && !parse_nonnegative(argv[9], &d_native)) ||
        (argc > 10 && !parse_nonnegative(argv[10], &cached)) ||
        (argc > 11 && !parse_nonnegative(argv[11], &split_k)) ||
        a_native > 1 || d_native > 1 || cached > 1 ||
        (split_k != 0 && split_k != 2 && split_k != 3) ||
        (split_k != 0 &&
         (batch != 1 || n_tile != 0 || a_native != 0 || d_native != 0))) {
        print_usage(argv[0]);
        return 2;
    }

    rk_npu_matmul_f16_config cfg{};
    rk_npu_matmul_f16_config_init(&cfg, M, N, K, RK_NPU_FUSE_NONE);
    cfg.n_tile = n_tile;
    cfg.core_mask = (uint32_t)core_mask;
    cfg.a_layout = a_native ? RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8
                            : RK_NPU_F16_A_LAYOUT_NORMAL;
    cfg.d_layout = d_native ? RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8
                            : RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;

    rk_npu_matmul_sizes sizes{};
    int rc = split_k
        ? rk_npu_matmul_f16_splitk_query(split_k, &cfg, &sizes)
        : rk_npu_matmul_f16_batch_query(batch, &cfg, &sizes);
    if (rc != RK_NPU_OK) {
        std::fprintf(stderr, "query failed: %s\n", rk_npu_strerror(rc));
        return 1;
    }

    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    if (!ctx) {
        std::fprintf(stderr, "open failed\n");
        return 1;
    }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) {
        std::fprintf(stderr, "domain create failed\n");
        rk_npu_close(ctx);
        return 1;
    }

    OwnedMem input{}, weight{}, output{};
    const uint32_t data_flags = cached
        ? RK_NPU_MEM_DATA_DEFAULT : RK_NPU_MEM_NON_CACHEABLE;
    if ((rc = rk_npu_mem_alloc(domain, sizes.input_bytes,
                               data_flags, &input.mem)) != RK_NPU_OK ||
        (rc = rk_npu_mem_alloc(domain, sizes.weight_bytes,
                               data_flags, &weight.mem)) != RK_NPU_OK ||
        (rc = rk_npu_mem_alloc(domain, sizes.output_bytes,
                               data_flags, &output.mem)) != RK_NPU_OK) {
        std::fprintf(stderr, "buffer allocation failed: %s\n", rk_npu_strerror(rc));
        if (input.mem.handle) rk_npu_mem_free(ctx, &input.mem);
        if (weight.mem.handle) rk_npu_mem_free(ctx, &weight.mem);
        if (output.mem.handle) rk_npu_mem_free(ctx, &output.mem);
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
        return 1;
    }

    std::vector<uint16_t> A((size_t)batch * M * K);
    std::vector<uint16_t> B((size_t)batch * K * N);
    std::vector<uint16_t> D((size_t)batch * M * N);
    std::mt19937 rng(0x46313650u + (uint32_t)M * 3u + (uint32_t)K * 5u +
                     (uint32_t)N * 7u + (uint32_t)batch * 11u);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    for (uint16_t& value : A) value = f2h(dist(rng));
    for (uint16_t& value : B) value = f2h(dist(rng));

    const auto pack_a_begin = std::chrono::steady_clock::now();
    rc = split_k
        ? rk_npu_matmul_f16_splitk_pack_a(
              split_k, &cfg, A.data(), &input.mem)
        : rk_npu_matmul_f16_batch_pack_a(
              batch, &cfg, A.data(), &input.mem);
    const auto pack_a_end = std::chrono::steady_clock::now();
    const auto pack_b_begin = pack_a_end;
    if (rc == RK_NPU_OK)
        rc = split_k
            ? rk_npu_matmul_f16_splitk_pack_b(
                  split_k, &cfg, B.data(), &weight.mem)
            : rk_npu_matmul_f16_batch_pack_b(
                  batch, &cfg, B.data(), &weight.mem);
    if (rc == RK_NPU_OK && cached)
        rc = rk_npu_mem_sync(ctx, &input.mem, RK_NPU_SYNC_TO_DEVICE);
    if (rc == RK_NPU_OK && cached)
        rc = rk_npu_mem_sync(ctx, &weight.mem, RK_NPU_SYNC_TO_DEVICE);
    const auto pack_b_end = std::chrono::steady_clock::now();

    const auto prepare_begin = pack_b_end;
    rk_npu_matmul_f16_batch_plan* plan =
        rc == RK_NPU_OK && !split_k
            ? rk_npu_matmul_f16_batch_prepare(domain, batch, &cfg)
            : nullptr;
    rk_npu_matmul_f16_splitk_plan* split_plan =
        rc == RK_NPU_OK && split_k
            ? rk_npu_matmul_f16_splitk_prepare(
                  domain, split_k, &cfg)
            : nullptr;
    const auto prepare_end = std::chrono::steady_clock::now();
    if ((!split_k && !plan) || (split_k && !split_plan))
        rc = RK_NPU_ERR_NOMEM;

    auto run = [&]() {
        return split_k
            ? rk_npu_matmul_f16_splitk_run(
                  ctx, split_plan, &input.mem, &weight.mem, &output.mem)
            : rk_npu_matmul_f16_batch_run(
                  ctx, plan, &input.mem, &weight.mem, &output.mem);
    };
    if (rc == RK_NPU_OK) rc = run();

    std::vector<double> samples;
    samples.reserve((size_t)loops);
    for (int loop = 0; rc == RK_NPU_OK && loop < loops; ++loop) {
        const auto begin = std::chrono::steady_clock::now();
        rc = run();
        const auto end = std::chrono::steady_clock::now();
        if (rc == RK_NPU_OK) samples.push_back(elapsed_us(begin, end));
    }
    const auto unpack_begin = std::chrono::steady_clock::now();
    if (rc == RK_NPU_OK && cached)
        rc = rk_npu_mem_sync(ctx, &output.mem, RK_NPU_SYNC_FROM_DEVICE);
    if (rc == RK_NPU_OK)
        rc = split_k
            ? rk_npu_matmul_f16_splitk_unpack_d(
                  split_k, &cfg, &output.mem, D.data())
            : rk_npu_matmul_f16_batch_unpack_d(
                  batch, &cfg, &output.mem, D.data());
    const auto unpack_end = std::chrono::steady_clock::now();

    double max_abs = 0.0;
    double max_ref = 1.0;
    double worst_reference = 0.0, worst_actual = 0.0;
    int worst_b = 0, worst_m = 0, worst_n = 0;
    constexpr int kChecks = 64;
    if (rc == RK_NPU_OK) {
        for (int check = 0; check < kChecks; ++check) {
            const int b = check % batch;
            const int m = (check * 37 + 3) % M;
            const int n = (check * 101 + 7) % N;
            double reference = 0.0;
            for (int k = 0; k < K; ++k) {
                reference += (double)h2f(A[((size_t)b * M + m) * K + k]) *
                             h2f(B[((size_t)b * K + k) * N + n]);
            }
            const double actual = h2f(D[((size_t)b * M + m) * N + n]);
            const double error = std::fabs(actual - reference);
            if (error > max_abs) {
                max_abs = error;
                worst_reference = reference;
                worst_actual = actual;
                worst_b = b;
                worst_m = m;
                worst_n = n;
            }
            max_ref = std::max(max_ref, std::fabs(reference));
        }
    }
    const bool correct = rc == RK_NPU_OK && max_abs < 0.03 * max_ref + 0.05;

    double average_us = 0.0, min_us = 0.0, max_us = 0.0;
    if (!samples.empty()) {
        average_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
                     samples.size();
        min_us = *std::min_element(samples.begin(), samples.end());
        max_us = *std::max_element(samples.begin(), samples.end());
    }
    const double tflops = average_us > 0.0
        ? (2.0 * batch * M * K * N) / (average_us * 1.0e6)
        : 0.0;

    std::printf("shape B=%d M=%d K=%d N=%d n_tile=%d core_mask=0x%x "
                "a_native=%d d_native=%d cached=%d split_k=%d\n",
                batch, M, K, N, n_tile, core_mask,
                a_native, d_native, cached, split_k);
    std::printf("buffers input=%llu weight=%llu output=%llu regcmd=%llu task=%llu tasks=%d\n",
                (unsigned long long)sizes.input_bytes,
                (unsigned long long)sizes.weight_bytes,
                (unsigned long long)sizes.output_bytes,
                (unsigned long long)sizes.regcmd_bytes,
                (unsigned long long)sizes.task_bytes, sizes.num_tasks);
    std::printf("host pack_a=%.2f us pack_b=%.2f us unpack_d=%.2f us prepare=%.2f us\n",
                elapsed_us(pack_a_begin, pack_a_end),
                elapsed_us(pack_b_begin, pack_b_end),
                elapsed_us(unpack_begin, unpack_end),
                elapsed_us(prepare_begin, prepare_end));
    std::printf("run loops=%zu avg=%.2f us min=%.2f us max=%.2f us %.4f TFLOP/s\n",
                samples.size(), average_us, min_us, max_us, tflops);
    std::printf("check %s max_abs=%.6f max_ref=%.6f worst=(%d,%d,%d) ref=%.6f actual=%.6f rc=%s\n",
                correct ? "PASS" : "FAIL", max_abs, max_ref,
                worst_b, worst_m, worst_n, worst_reference, worst_actual,
                rk_npu_strerror(rc));

    rk_npu_matmul_f16_batch_plan_free(plan);
    rk_npu_matmul_f16_splitk_plan_free(split_plan);
    if (input.mem.handle) rk_npu_mem_free(ctx, &input.mem);
    if (weight.mem.handle) rk_npu_mem_free(ctx, &weight.mem);
    if (output.mem.handle) rk_npu_mem_free(ctx, &output.mem);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return correct ? 0 : 1;
}

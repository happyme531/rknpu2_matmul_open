/*
 * bench_llm_i8 - small RK3588 W8A8 LLM matmul benchmark.
 *
 * The timed region is one complete public i8i8i32 call.  It includes compact
 * row-major A packing, NPU execution, split-K reduction, and compact row-major
 * C output, but excludes the one-time B packing/weight creation.
 *
 * M=1/4 are memory-bound decode-style cases and report logical INT8 weight
 * bandwidth (K*N bytes / wall time).  M=128 is a prefill-style case and
 * reports 2*M*K*N operations / wall time in GOPS.
 */
#include "rk_npu_quant_matmul.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <vector>

namespace {

constexpr int kMaxNTile = 3072;
constexpr int kKTile = 2048;

struct Args {
    const char* dev = nullptr;
    int loops = 10;
    int warmup = 2;
    uint32_t npu_mask = 7;
};

struct Projection {
    const char* name;
    int K;
    int N;
};

const Projection kProjections[] = {
    {"hidden",   4096,  4096},
    {"ffn_up",   4096, 11008},
    {"ffn_down", 11008, 4096},
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [--dev PATH] [--loops N] [--warmup N] [--npu-mask 1|2|3|4|7]\n",
        argv0);
}

bool parse_int(const char* text, int min_value, int* out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 0);
    if (errno || !end || *end || value < min_value || value > INT32_MAX)
        return false;
    *out = (int)value;
    return true;
}

bool parse_args(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto next = [&]() -> const char* {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if (option == "--dev") {
            args->dev = next();
            if (!args->dev) return false;
        } else if (option == "--loops") {
            if (!parse_int(next(), 1, &args->loops)) return false;
        } else if (option == "--warmup") {
            if (!parse_int(next(), 0, &args->warmup)) return false;
        } else if (option == "--npu-mask") {
            int mask = 0;
            if (!parse_int(next(), 1, &mask) ||
                !(mask == 1 || mask == 2 || mask == 3 || mask == 4 || mask == 7))
                return false;
            args->npu_mask = (uint32_t)mask;
        } else if (option == "-h" || option == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

int popcount(uint32_t value) {
    return __builtin_popcount(value);
}

uint64_t benchmark_cpu_mask() {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
        return 1;

    std::vector<int> order;
    for (int cpu = 4; cpu <= 7; ++cpu)
        if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &affinity)) order.push_back(cpu);
    for (int cpu = 0; cpu < CPU_SETSIZE && cpu < 64; ++cpu)
        if ((cpu < 4 || cpu > 7) && CPU_ISSET(cpu, &affinity)) order.push_back(cpu);

    uint64_t mask = 0;
    for (int i = 0; i < (int)order.size() && i < 4; ++i)
        mask |= 1ull << order[(size_t)i];
    return mask ? mask : 1;
}

int choose_n_tile(int N, uint32_t npu_mask) {
    const int aligned_n = align_up(N, 32);
    const int groups = std::max(popcount(npu_mask), ceil_div(aligned_n, kMaxNTile));
    return align_up(ceil_div(aligned_n, groups), 32);
}

rk_npu_matmul_strategy make_strategy(int M, int N, int K,
                                     uint32_t npu_mask, uint64_t cpu_mask) {
    rk_npu_matmul_strategy strategy{};
    strategy.op_kind = RK_NPU_MATMUL_I8I8I32;
    strategy.M = M;
    strategy.N = N;
    strategy.K = K;
    strategy.k_tile = std::min(K, kKTile);
    strategy.n_tile = choose_n_tile(N, npu_mask);
    strategy.a_layout = RK_NPU_MATMUL_A_LAYOUT_NORMAL;
    strategy.c_layout = RK_NPU_MATMUL_C_LAYOUT_NATIVE;
    strategy.wave_count = ceil_div(K, strategy.k_tile);
    strategy.n_groups = ceil_div(align_up(N, 32), strategy.n_tile);
    strategy.npu_core_mask = npu_mask;
    strategy.cpu_core_mask = cpu_mask;
    strategy.cpu_threads = __builtin_popcountll(cpu_mask);
    return strategy;
}

struct XorShift64 {
    uint64_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (uint32_t)(state >> 16);
    }
    int8_t next_i8() { return (int8_t)((int)(next() & 15u) - 8); }
};

void fill_i8(std::vector<int8_t>* values, uint64_t seed) {
    XorShift64 rng{seed};
    for (int8_t& value : *values) value = rng.next_i8();
}

bool verify_samples(int M, int N, int K, const int8_t* A, const int8_t* B,
                    const int32_t* C) {
    constexpr int kSamples = 16;
    for (int sample = 0; sample < kSamples; ++sample) {
        const int m = (sample * 37 + M - 1) % M;
        const int n = (sample * 1013 + N - 1) % N;
        uint32_t reference = 0;
        for (int k = 0; k < K; ++k) {
            const int32_t product = (int32_t)A[(size_t)m * K + k] *
                                    (int32_t)B[(size_t)k * N + n];
            reference += (uint32_t)product;
        }
        uint32_t actual = 0;
        std::memcpy(&actual, C + (size_t)m * N + n, sizeof(actual));
        if (actual != reference) {
            std::fprintf(stderr,
                "verify failed at M=%d K=%d N=%d [%d,%d]: got=%d expected=%d\n",
                M, K, N, m, n, C[(size_t)m * N + n], (int32_t)reference);
            return false;
        }
    }
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() & 1 ? values[middle]
                             : (values[middle - 1] + values[middle]) * 0.5;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }

    rk_npu_ctx* ctx = rk_npu_open(args.dev);
    if (!ctx) {
        std::fprintf(stderr, "rk_npu_open failed\n");
        return 1;
    }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) {
        std::fprintf(stderr, "rk_npu_iommu_domain_create failed\n");
        rk_npu_close(ctx);
        return 1;
    }

    const uint64_t cpu_mask = benchmark_cpu_mask();
    std::printf("RK3588 INT8 LLM matmul: median of %d runs after %d warmups, "
                "npu_mask=0x%x cpu_mask=0x%llx\n",
                args.loops, args.warmup, args.npu_mask,
                (unsigned long long)cpu_mask);
    std::printf("Timed call includes A pack + NPU + split-K reduction + C unpack; "
                "one-time B pack is excluded.\n");
    std::printf("%-9s %4s %6s %6s %9s  %10s %-11s %4s %4s   %s\n",
                "layer", "M", "K", "N", "median_us", "value", "metric",
                "Kt", "Nt", "check");

    bool ok = true;
    for (size_t projection_index = 0;
         projection_index < sizeof(kProjections) / sizeof(kProjections[0]) && ok;
         ++projection_index) {
        const Projection& projection = kProjections[projection_index];
        std::vector<int8_t> B((size_t)projection.K * projection.N);
        fill_i8(&B, 0x425f4c4c4dull ^ ((uint64_t)projection.K << 32) ^
                    (uint64_t)projection.N);
        const rk_npu_matmul_weight_config weight_config{
            projection.K, projection.N, std::min(projection.K, kKTile)};
        rk_npu_i8i8i32_weights* weights =
            rk_npu_i8i8i32_weights_create(domain, &weight_config, B.data());
        if (!weights) {
            std::fprintf(stderr, "weight create failed for K=%d N=%d\n",
                         projection.K, projection.N);
            ok = false;
            break;
        }
        for (int M : {1, 4, 128}) {
            const rk_npu_matmul_strategy strategy =
                make_strategy(M, projection.N, projection.K,
                              args.npu_mask, cpu_mask);
            rk_npu_matmul_workspace* workspace =
                rk_npu_matmul_workspace_create(domain, &strategy);
            if (!workspace) {
                std::fprintf(stderr, "workspace create failed for M=%d K=%d N=%d\n",
                             M, projection.K, projection.N);
                ok = false;
                break;
            }
            std::vector<int8_t> A((size_t)M * projection.K);
            std::vector<int32_t> C((size_t)M * projection.N);
            fill_i8(&A, 0x415f4c4c4dull ^ (uint64_t)M ^
                        ((uint64_t)projection.K << 16));
            int rc = RK_NPU_OK;
            for (int i = 0; i < args.warmup && rc == RK_NPU_OK; ++i)
                rc = rk_npu_i8i8i32_run(workspace, weights, A.data(), C.data());
            std::vector<double> samples;
            samples.reserve((size_t)args.loops);
            for (int i = 0; i < args.loops && rc == RK_NPU_OK; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                rc = rk_npu_i8i8i32_run(workspace, weights, A.data(), C.data());
                const auto end = std::chrono::steady_clock::now();
                if (rc == RK_NPU_OK)
                    samples.push_back(std::chrono::duration<double, std::micro>(
                        end - begin).count());
            }
            if (rc != RK_NPU_OK || samples.empty() ||
                !verify_samples(M, projection.N, projection.K,
                                A.data(), B.data(), C.data())) {
                if (rc != RK_NPU_OK)
                    std::fprintf(stderr, "run failed for M=%d K=%d N=%d: %s\n",
                                 M, projection.K, projection.N,
                                 rk_npu_strerror(rc));
                ok = false;
                rk_npu_matmul_workspace_free(workspace);
                break;
            }
            const double us = median(samples);
            const double metric = M < 128
                ? ((double)projection.K * projection.N) / us / 1000.0
                : (2.0 * M * projection.K * projection.N) / us / 1000.0;
            std::printf("%-9s %4d %6d %6d %9.2f  %10.2f %-11s %4d %4d   PASS\n",
                        projection.name, M, projection.K, projection.N, us,
                        metric, M < 128 ? "weight GB/s" : "GOPS",
                        strategy.k_tile, strategy.n_tile);
            rk_npu_matmul_workspace_free(workspace);
        }
        rk_npu_i8i8i32_weights_free(weights);
    }

    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return ok ? 0 : 1;
}

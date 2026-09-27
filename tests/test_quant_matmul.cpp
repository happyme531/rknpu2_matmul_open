#include "rk_npu_quant_matmul.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <random>
#include <sched.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

void remove_cache_directory(const char* path) {
    DIR* directory = ::opendir(path);
    if (!directory) return;
    while (dirent* entry = ::readdir(directory)) {
        if (std::strcmp(entry->d_name, ".") == 0 ||
            std::strcmp(entry->d_name, "..") == 0)
            continue;
        const std::string file = std::string(path) + "/" + entry->d_name;
        (void)::unlink(file.c_str());
    }
    ::closedir(directory);
    (void)::rmdir(path);
}

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1000u) && ((man & 0x0fffu) || (half & 1u))) ++half;
    return (uint16_t)half;
}

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint64_t allowed_cpu_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) return 1;
    uint64_t mask = 0;
    for (int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) mask |= 1ull << cpu;
    return mask ? mask : 1;
}

uint64_t take_cpus(uint64_t allowed, int count) {
    uint64_t mask = 0;
    for (int cpu = 4; cpu <= 7 && __builtin_popcountll(mask) < count; ++cpu)
        if (allowed & (1ull << cpu)) mask |= 1ull << cpu;
    for (int cpu = 0; cpu < 64 && __builtin_popcountll(mask) < count; ++cpu)
        if (allowed & (1ull << cpu)) mask |= 1ull << cpu;
    return mask;
}

int8_t quant(float x, float scale) {
    float q = std::round(x / scale);
    q = std::max(-127.0f, std::min(127.0f, q));
    return (int8_t)q;
}

void gemm_i8(int M, int N, int K, const int8_t* A, const int8_t* B,
             std::vector<int32_t>& C) {
    C.assign((size_t)M * N, 0);
    for (int m = 0; m < M; ++m) {
        uint32_t* out = reinterpret_cast<uint32_t*>(C.data() + (size_t)m * N);
        for (int k = 0; k < K; ++k) {
            const int32_t a = A[(size_t)m * K + k];
            for (int n = 0; n < N; ++n)
                out[n] += (uint32_t)(a * (int32_t)B[(size_t)k * N + n]);
        }
    }
}

void f16_reference(int M, int N, int K, const std::vector<uint16_t>& A,
                   const std::vector<int8_t>& B, const std::vector<float>& ws,
                   const float* static_scale, std::vector<uint16_t>& C) {
    std::vector<float> as((size_t)M);
    std::vector<int8_t> Aq((size_t)M * K);
    for (int m = 0; m < M; ++m) {
        float scale = static_scale ? static_scale[m] : 0.0f;
        if (!static_scale) {
            for (int k = 0; k < K; ++k)
                scale = std::max(scale, std::fabs(half_to_float(A[(size_t)m * K + k])));
            scale = scale > 0 ? scale / 127.0f : 1.0f;
        }
        as[(size_t)m] = scale;
        for (int k = 0; k < K; ++k)
            Aq[(size_t)m * K + k] = quant(half_to_float(A[(size_t)m * K + k]), scale);
    }
    std::vector<int32_t> acc;
    gemm_i8(M, N, K, Aq.data(), B.data(), acc);
    C.resize((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[(size_t)m * N + n] = float_to_half(
                (float)acc[(size_t)m * N + n] * as[(size_t)m] * ws[(size_t)n]);
}

void f32_reference(int M, int N, int K, const std::vector<float>& A,
                   const std::vector<int8_t>& B, const std::vector<float>& ws,
                   const float* static_scale, std::vector<float>& C) {
    std::vector<float> as((size_t)M);
    std::vector<int8_t> Aq((size_t)M * K);
    for (int m = 0; m < M; ++m) {
        float scale = static_scale ? static_scale[m] : 0.0f;
        if (!static_scale) {
            for (int k = 0; k < K; ++k)
                scale = std::max(scale, std::fabs(A[(size_t)m * K + k]));
            scale = scale > 0 ? scale / 127.0f : 1.0f;
        }
        as[(size_t)m] = scale;
        for (int k = 0; k < K; ++k)
            Aq[(size_t)m * K + k] = quant(A[(size_t)m * K + k], scale);
    }
    std::vector<int32_t> acc;
    gemm_i8(M, N, K, Aq.data(), B.data(), acc);
    C.resize((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[(size_t)m * N + n] =
                (float)acc[(size_t)m * N + n] * as[(size_t)m] * ws[(size_t)n];
}

bool close_f32(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        const float tol = 1e-5f + 2e-6f * std::fabs(ref[i]);
        if (std::fabs(got[i] - ref[i]) > tol) return false;
    }
    return true;
}

rk_npu_matmul_strategy strategy(rk_npu_matmul_op_kind kind,
                                int M, int N, int K, int kt, int nt,
                                uint32_t npu_mask, uint64_t cpu_mask,
                                rk_npu_matmul_a_layout a_layout =
                                    RK_NPU_MATMUL_A_LAYOUT_NORMAL) {
    rk_npu_matmul_strategy s{};
    s.op_kind = kind;
    s.M = M; s.N = N; s.K = K;
    s.k_tile = kt; s.n_tile = nt;
    s.a_layout = a_layout;
    s.wave_count = (K + kt - 1) / kt;
    s.n_groups = ((N + 31) / 32 * 32 + nt - 1) / nt;
    s.npu_core_mask = npu_mask;
    s.cpu_core_mask = cpu_mask;
    s.cpu_threads = __builtin_popcountll(cpu_mask);
    return s;
}

} /* namespace */

int main() {
    constexpr int M = 8, N = 96, K = 96;
    std::mt19937 rng(0x51554e54);
    std::uniform_int_distribution<int> qi(-8, 8);
    std::uniform_real_distribution<float> qf(-2.0f, 2.0f);
    std::vector<int8_t> A8((size_t)M * K), B((size_t)K * N), B2((size_t)K * N);
    std::vector<uint16_t> A16((size_t)M * K);
    std::vector<float> A32((size_t)M * K);
    std::vector<float> as((size_t)M), ws((size_t)N);
    for (int8_t& x : A8) x = (int8_t)qi(rng);
    for (int8_t& x : B) x = (int8_t)qi(rng);
    for (int8_t& x : B2) x = (int8_t)qi(rng);
    for (uint16_t& x : A16) x = float_to_half(qf(rng));
    for (float& x : A32) x = qf(rng);
    for (int m = 0; m < M; ++m) as[(size_t)m] = 0.012f + m * 0.0003f;
    for (int n = 0; n < N; ++n) ws[(size_t)n] = 0.005f + (n % 17) * 0.0002f;

    std::vector<int32_t> ref_i32, ref_i32_b2;
    gemm_i8(M, N, K, A8.data(), B.data(), ref_i32);
    gemm_i8(M, N, K, A8.data(), B2.data(), ref_i32_b2);
    std::vector<uint16_t> ref_dyn, ref_static;
    f16_reference(M, N, K, A16, B, ws, nullptr, ref_dyn);
    f16_reference(M, N, K, A16, B, ws, as.data(), ref_static);
    std::vector<float> ref_f32_dyn, ref_f32_static;
    f32_reference(M, N, K, A32, B, ws, nullptr, ref_f32_dyn);
    f32_reference(M, N, K, A32, B, ws, as.data(), ref_f32_static);

    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    if (!ctx) return 2;
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    rk_npu_iommu_domain* other_domain = rk_npu_iommu_domain_create(ctx, 1);
    rk_npu_iommu_domain* third_domain = rk_npu_iommu_domain_create(ctx, 2);
    if (!domain || !other_domain || !third_domain) {
        rk_npu_iommu_domain_free(third_domain);
        rk_npu_iommu_domain_free(other_domain);
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
        return 2;
    }
    const uint64_t allowed = allowed_cpu_mask();
    const uint64_t cpu_mask = take_cpus(
        allowed, std::min(2, __builtin_popcountll(allowed)));
    bool ok = true;

    for (auto s : {strategy(RK_NPU_MATMUL_I8I8I32, M, N, K, K, N, 1, cpu_mask),
                   strategy(RK_NPU_MATMUL_I8I8I32, M, N, K, 32, 32, 7, cpu_mask),
                   strategy(RK_NPU_MATMUL_I8I8I32, M, N, K, 32, 32, 7,
                            cpu_mask,
                            RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16)}) {
        rk_npu_matmul_weight_config wc{K, N, s.k_tile};
        rk_npu_matmul_workspace_requirements workspace_memory{};
        rk_npu_matmul_weight_requirements weight_memory{};
        ok &= rk_npu_matmul_workspace_memory_query(&s, &workspace_memory) ==
                  RK_NPU_OK &&
              rk_npu_i8i8i32_weights_memory_query(&wc, &weight_memory) ==
                  RK_NPU_OK &&
              workspace_memory.wave_count == (uint32_t)s.wave_count &&
              weight_memory.packed_b_bytes == (uint64_t)K * N &&
              weight_memory.scale_bytes == 0 &&
              workspace_memory.input_bytes ==
                  (uint64_t)std::min(s.wave_count, 2) * M *
                  ((std::min(s.k_tile, K) + 31) / 32 * 32) &&
              workspace_memory.output_bytes ==
                  (uint64_t)std::min(s.wave_count, 3) * M * N * sizeof(int32_t) &&
              workspace_memory.buffer_count ==
                  (uint32_t)(2 * s.wave_count + std::min(s.wave_count, 2) +
                             std::min(s.wave_count, 3)) &&
              weight_memory.buffer_count == 1;
        rk_npu_matmul_workspace* workspace =
            rk_npu_matmul_workspace_create(domain, &s);
        rk_npu_i8i8i32_weights* p0 =
            rk_npu_i8i8i32_weights_create(domain, &wc, B.data());
        rk_npu_i8i8i32_weights* p1 =
            rk_npu_i8i8i32_weights_create(domain, &wc, B2.data());
        std::vector<int32_t> got((size_t)M * N);
        int rc = p0 ? rk_npu_i8i8i32_run(
            workspace, p0, A8.data(), got.data()) : RK_NPU_ERR_NOMEM;
        ok &= rc == RK_NPU_OK && got == ref_i32;
        rc = p1 ? rk_npu_i8i8i32_run(
            workspace, p1, A8.data(), got.data()) : RK_NPU_ERR_NOMEM;
        ok &= rc == RK_NPU_OK && got == ref_i32_b2;
        rc = p0 ? rk_npu_i8i8i32_run(
            workspace, p0, A8.data(), got.data()) : RK_NPU_ERR_NOMEM;
        ok &= rc == RK_NPU_OK && got == ref_i32;
        if (s.wave_count > 1 && p0 && p1) {
            std::atomic<int> good{0}, busy{0}, bad{0};
            std::atomic<bool> go{false};
            std::vector<int32_t> concurrent0((size_t)M * N);
            std::vector<int32_t> concurrent1((size_t)M * N);
            auto hammer = [&](rk_npu_i8i8i32_weights* p,
                              std::vector<int32_t>& out) {
                while (!go.load(std::memory_order_acquire)) {}
                for (int i = 0; i < 32; ++i) {
                    const int run_rc = rk_npu_i8i8i32_run(
                        workspace, p, A8.data(), out.data());
                    if (run_rc == RK_NPU_OK) ++good;
                    else if (run_rc == RK_NPU_ERR_BUSY) ++busy;
                    else ++bad;
                }
            };
            std::thread t0(hammer, p0, std::ref(concurrent0));
            std::thread t1(hammer, p1, std::ref(concurrent1));
            go.store(true, std::memory_order_release);
            t0.join();
            t1.join();
            ok &= good.load() > 0 && busy.load() > 0 && bad.load() == 0;
        }
        /* The same packed B must work with a different exact M and task plan. */
        auto m1_s = strategy(
            RK_NPU_MATMUL_I8I8I32, 1, N, K, s.k_tile, N, 1, cpu_mask);
        rk_npu_matmul_workspace* m1_workspace =
            rk_npu_matmul_workspace_create(domain, &m1_s);
        std::vector<int32_t> got_m1((size_t)N);
        rc = rk_npu_i8i8i32_run(
            m1_workspace, p0, A8.data(), got_m1.data());
        ok &= rc == RK_NPU_OK &&
              std::equal(got_m1.begin(), got_m1.end(), ref_i32.begin());
        rk_npu_matmul_workspace_free(m1_workspace);

        if (s.k_tile == 32) {
            auto incompatible_s = strategy(
                RK_NPU_MATMUL_I8I8I32, M, N, K, K, N, 1, cpu_mask);
            rk_npu_matmul_workspace* incompatible_workspace =
                rk_npu_matmul_workspace_create(domain, &incompatible_s);
            ok &= rk_npu_i8i8i32_run(
                incompatible_workspace, p0, A8.data(), got.data()) ==
                RK_NPU_ERR_PARAM;
            rk_npu_matmul_workspace_free(incompatible_workspace);
        }
        rk_npu_i8i8i32_weights_free(p1);
        rk_npu_i8i8i32_weights_free(p0);
        rk_npu_matmul_workspace_free(workspace);
    }

    auto dynamic_s = strategy(
        RK_NPU_MATMUL_F16I8F16_DYNAMIC, M, N, K, 32, 32, 7, cpu_mask);
    rk_npu_matmul_weight_config f16_wc{K, N, dynamic_s.k_tile};
    rk_npu_matmul_weight_requirements dynamic_memory{};
    ok &= rk_npu_f16i8f16_weights_memory_query(&f16_wc, &dynamic_memory) ==
              RK_NPU_OK &&
          dynamic_memory.scale_bytes == (uint64_t)N * sizeof(float) &&
          dynamic_memory.buffer_count == 1;
    rk_npu_matmul_workspace* dynamic_workspace =
        rk_npu_matmul_workspace_create(domain, &dynamic_s);
    rk_npu_f16i8f16_weights* f16_weights =
        rk_npu_f16i8f16_weights_create(domain, &f16_wc, B.data(), ws.data());
    std::vector<uint16_t> got_dyn((size_t)M * N);
    int rc = f16_weights ? rk_npu_f16i8f16_run_dynamic(
        dynamic_workspace, f16_weights, A16.data(), got_dyn.data())
        : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && got_dyn == ref_dyn;

    auto static_s = dynamic_s;
    static_s.op_kind = RK_NPU_MATMUL_F16I8F16_STATIC;
    rk_npu_matmul_workspace* static_workspace =
        rk_npu_matmul_workspace_create(domain, &static_s);
    std::vector<uint16_t> got_static((size_t)M * N);
    rc = f16_weights ? rk_npu_f16i8f16_run_static(
        static_workspace, f16_weights, A16.data(), as.data(), got_static.data())
        : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && got_static == ref_static;
    std::vector<float> bad_scale = as;
    bad_scale[0] = 0.0f;
    ok &= rk_npu_f16i8f16_run_static(
        static_workspace, f16_weights, A16.data(), bad_scale.data(),
        got_static.data()) == RK_NPU_ERR_PARAM;

    auto native_dynamic_s = dynamic_s;
    native_dynamic_s.a_layout = RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    rk_npu_matmul_workspace* native_dynamic_workspace =
        rk_npu_matmul_workspace_create(domain, &native_dynamic_s);
    rc = f16_weights ? rk_npu_f16i8f16_run_dynamic(
        native_dynamic_workspace, f16_weights, A16.data(), got_dyn.data())
        : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && got_dyn == ref_dyn;
    auto native_static_s = static_s;
    native_static_s.a_layout = RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    rk_npu_matmul_workspace* native_static_workspace =
        rk_npu_matmul_workspace_create(domain, &native_static_s);
    rc = f16_weights ? rk_npu_f16i8f16_run_static(
        native_static_workspace, f16_weights, A16.data(), as.data(),
        got_static.data()) : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && got_static == ref_static;
    rk_npu_matmul_workspace_free(native_static_workspace);
    rk_npu_matmul_workspace_free(native_dynamic_workspace);
    rk_npu_f16i8f16_weights_free(f16_weights);
    rk_npu_matmul_workspace_free(static_workspace);
    rk_npu_matmul_workspace_free(dynamic_workspace);

    auto f32_dynamic_s = strategy(
        RK_NPU_MATMUL_F32I8F32_DYNAMIC, M, N, K, 32, 32, 7, cpu_mask);
    rk_npu_matmul_weight_config f32_wc{K, N, f32_dynamic_s.k_tile};
    rk_npu_matmul_weight_requirements f32_memory{};
    ok &= rk_npu_f32i8f32_weights_memory_query(&f32_wc, &f32_memory) ==
              RK_NPU_OK &&
          f32_memory.scale_bytes == (uint64_t)N * sizeof(float) &&
          f32_memory.buffer_count == 1;
    rk_npu_matmul_workspace* f32_dynamic_workspace =
        rk_npu_matmul_workspace_create(domain, &f32_dynamic_s);
    rk_npu_f32i8f32_weights* f32_weights =
        rk_npu_f32i8f32_weights_create(domain, &f32_wc, B.data(), ws.data());
    std::vector<float> got_f32_dyn((size_t)M * N);
    rc = f32_weights ? rk_npu_f32i8f32_run_dynamic(
        f32_dynamic_workspace, f32_weights, A32.data(), got_f32_dyn.data())
        : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && close_f32(got_f32_dyn, ref_f32_dyn);

    auto f32_static_s = f32_dynamic_s;
    f32_static_s.op_kind = RK_NPU_MATMUL_F32I8F32_STATIC;
    rk_npu_matmul_workspace* f32_static_workspace =
        rk_npu_matmul_workspace_create(domain, &f32_static_s);
    std::vector<float> got_f32_static((size_t)M * N);
    rc = f32_weights ? rk_npu_f32i8f32_run_static(
        f32_static_workspace, f32_weights, A32.data(), as.data(),
        got_f32_static.data()) : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && close_f32(got_f32_static, ref_f32_static);
    ok &= rk_npu_f32i8f32_run_static(
        f32_static_workspace, f32_weights, A32.data(), bad_scale.data(),
        got_f32_static.data()) == RK_NPU_ERR_PARAM;

    auto f32_native_dynamic_s = f32_dynamic_s;
    f32_native_dynamic_s.a_layout = RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    rk_npu_matmul_workspace* f32_native_dynamic_workspace =
        rk_npu_matmul_workspace_create(domain, &f32_native_dynamic_s);
    rc = f32_weights ? rk_npu_f32i8f32_run_dynamic(
        f32_native_dynamic_workspace, f32_weights, A32.data(),
        got_f32_dyn.data()) : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && close_f32(got_f32_dyn, ref_f32_dyn);
    auto f32_native_static_s = f32_static_s;
    f32_native_static_s.a_layout = RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;
    rk_npu_matmul_workspace* f32_native_static_workspace =
        rk_npu_matmul_workspace_create(domain, &f32_native_static_s);
    rc = f32_weights ? rk_npu_f32i8f32_run_static(
        f32_native_static_workspace, f32_weights, A32.data(), as.data(),
        got_f32_static.data()) : RK_NPU_ERR_NOMEM;
    ok &= rc == RK_NPU_OK && close_f32(got_f32_static, ref_f32_static);
    rk_npu_matmul_workspace_free(f32_native_static_workspace);
    rk_npu_matmul_workspace_free(f32_native_dynamic_workspace);
    rk_npu_f32i8f32_weights_free(f32_weights);
    rk_npu_matmul_workspace_free(f32_static_workspace);
    rk_npu_matmul_workspace_free(f32_dynamic_workspace);

    rk_npu_matmul_autotune_config cfg{};
    rk_npu_matmul_autotune_config_init(&cfg, M, N, K);
    cfg.allowed_cpu_core_mask = allowed;
    cfg.warmup = 0;
    cfg.loops = 2;
    cfg.repeats = 1;
    for (rk_npu_matmul_op_kind kind : {
             RK_NPU_MATMUL_I8I8I32,
             RK_NPU_MATMUL_F16I8F16_DYNAMIC,
             RK_NPU_MATMUL_F16I8F16_STATIC,
             RK_NPU_MATMUL_F32I8F32_DYNAMIC,
             RK_NPU_MATMUL_F32I8F32_STATIC}) {
        rk_npu_matmul_strategy fastest{}, stable{};
        int tune_rc = RK_NPU_ERR_PARAM;
        switch (kind) {
        case RK_NPU_MATMUL_I8I8I32:
            tune_rc = rk_npu_i8i8i32_autotune(ctx, &cfg, &fastest, &stable);
            break;
        case RK_NPU_MATMUL_F16I8F16_DYNAMIC:
            tune_rc = rk_npu_f16i8f16_dynamic_autotune(
                ctx, &cfg, &fastest, &stable);
            break;
        case RK_NPU_MATMUL_F16I8F16_STATIC:
            tune_rc = rk_npu_f16i8f16_static_autotune(
                ctx, &cfg, &fastest, &stable);
            break;
        case RK_NPU_MATMUL_F32I8F32_DYNAMIC:
            tune_rc = rk_npu_f32i8f32_dynamic_autotune(
                ctx, &cfg, &fastest, &stable);
            break;
        case RK_NPU_MATMUL_F32I8F32_STATIC:
            tune_rc = rk_npu_f32i8f32_static_autotune(
                ctx, &cfg, &fastest, &stable);
            break;
        }
        ok &= tune_rc == RK_NPU_OK && fastest.op_kind == kind && fastest.total_us > 0;
        const int fixed_threads = std::min(4, __builtin_popcountll(allowed));
        const uint64_t fixed_cpus = take_cpus(allowed, fixed_threads);
        ok &= fastest.npu_core_mask == 7 && stable.npu_core_mask == 7 &&
              fastest.cpu_threads == fixed_threads && stable.cpu_threads == fixed_threads &&
              fastest.cpu_core_mask == fixed_cpus && stable.cpu_core_mask == fixed_cpus;
        // Exercise the newly serialized C layout even if the tiny timing shape
        // happens to choose a legacy recipe. This is a value/cache round trip.
        if (tune_rc == RK_NPU_OK && kind == RK_NPU_MATMUL_I8I8I32) {
            char path[160];
            std::snprintf(path,sizeof(path),"/tmp/rk_panel_cache_%lld.tune",(long long)::getpid());
            auto panel_strategy = fastest;
            panel_strategy.a_layout = RK_NPU_MATMUL_A_LAYOUT_NORMAL;
            panel_strategy.c_layout = RK_NPU_MATMUL_C_LAYOUT_PANEL8;
            rk_npu_matmul_strategy loaded{}, loaded_stable{};
            ok &= rk_npu_matmul_autotune_cache_save(ctx,path,kind,&cfg,&panel_strategy,&panel_strategy)==RK_NPU_OK;
            ok &= rk_npu_matmul_autotune_cache_load(ctx,path,kind,&cfg,&loaded,&loaded_stable)==RK_NPU_OK;
            ok &= loaded.c_layout==RK_NPU_MATMUL_C_LAYOUT_PANEL8 &&
                  loaded_stable.c_layout==RK_NPU_MATMUL_C_LAYOUT_PANEL8 &&
                  loaded.total_us==panel_strategy.total_us;
            (void)::unlink(path);
        }
    }

    auto fixed_cfg = cfg;
    fixed_cfg.allowed_cpu_core_mask = take_cpus(allowed, 1);
    fixed_cfg.allowed_npu_core_mask = 4;
    rk_npu_matmul_strategy fixed_fast{}, fixed_stable{};
    ok &= rk_npu_i8i8i32_autotune(ctx,&fixed_cfg,&fixed_fast,&fixed_stable)==RK_NPU_OK;
    ok &= fixed_fast.npu_core_mask==4 && fixed_fast.cpu_threads==1 &&
          fixed_fast.cpu_core_mask==fixed_cfg.allowed_cpu_core_mask;
    fixed_cfg.N = 32;
    fixed_cfg.allowed_npu_core_mask = 7;
    ok &= rk_npu_i8i8i32_autotune(ctx,&fixed_cfg,&fixed_fast,&fixed_stable)==RK_NPU_OK;
    ok &= fixed_fast.npu_core_mask==1 && fixed_fast.cpu_threads==1;

    const int family_m[] = {1, 128};
    const double frequencies[] = {16.0, 1.0};
    rk_npu_matmul_family_autotune_config family_cfg{};
    rk_npu_matmul_family_autotune_config_init(
        &family_cfg, N, K, 2, family_m);
    family_cfg.frequencies = frequencies;
    family_cfg.allowed_cpu_core_mask = allowed;
    family_cfg.warmup = 0;
    family_cfg.loops = 1;
    family_cfg.repeats = 1;
    rk_npu_matmul_family_summary family_fast{}, family_stable{};
    rk_npu_matmul_strategy family_fast_s[2]{}, family_stable_s[2]{};
    char family_cache_dir[128];
    std::snprintf(family_cache_dir, sizeof(family_cache_dir),
                  "/tmp/rk_npu_matmul_family_cache_%lld",
                  (long long)::getpid());
    remove_cache_directory(family_cache_dir);
    int family_cache_hits = 0, family_cache_misses = 0;
    rk_npu_matmul_family_autotune_config one_m_family_cfg = family_cfg;
    one_m_family_cfg.m_count = 1;
    rk_npu_matmul_family_summary one_m_fast{}, one_m_stable{};
    rk_npu_matmul_strategy one_m_fast_s[1]{}, one_m_stable_s[1]{};
    rc = rk_npu_matmul_autotune_family_cached(
        ctx, &one_m_family_cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
        family_cache_dir, 1,
        &one_m_fast, one_m_fast_s, &one_m_stable, one_m_stable_s,
        &family_cache_hits, &family_cache_misses);
    ok &= rc == RK_NPU_OK && family_cache_hits == 0 &&
          family_cache_misses > 0;

    family_cache_hits = family_cache_misses = 0;
    rc = rk_npu_matmul_autotune_family_cached(
        ctx, &family_cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
        family_cache_dir, 0,
        &family_fast, family_fast_s, &family_stable, family_stable_s,
        &family_cache_hits, &family_cache_misses);
    ok &= rc == RK_NPU_OK && family_fast.weight_config.K == K &&
          family_fast.weight_config.N == N &&
          family_fast_s[0].k_tile == family_fast.weight_config.k_tile &&
          family_fast_s[1].k_tile == family_fast.weight_config.k_tile &&
          family_fast.weighted_total_us > 0 && family_cache_hits > 0 &&
          family_cache_misses > 0;

    const int cached_k_tile = family_fast.weight_config.k_tile;
    const double cached_total = family_fast.weighted_total_us;
    family_cache_hits = family_cache_misses = 0;
    rc = rk_npu_matmul_autotune_family_cached(
        ctx, &family_cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
        family_cache_dir, 0,
        &family_fast, family_fast_s, &family_stable, family_stable_s,
        &family_cache_hits, &family_cache_misses);
    ok &= rc == RK_NPU_OK && family_cache_hits > 0 &&
          family_cache_misses == 0 &&
          family_fast.weight_config.k_tile == cached_k_tile &&
          family_fast.weighted_total_us == cached_total;

    const double changed_frequencies[] = {1.0, 16.0};
    family_cfg.frequencies = changed_frequencies;
    family_cache_hits = family_cache_misses = 0;
    rc = rk_npu_matmul_autotune_family_cached(
        ctx, &family_cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
        family_cache_dir, 0,
        &family_fast, family_fast_s, &family_stable, family_stable_s,
        &family_cache_hits, &family_cache_misses);
    ok &= rc == RK_NPU_OK && family_cache_hits > 0 &&
          family_cache_misses == 0;
    family_cfg.frequencies = frequencies;
    remove_cache_directory(family_cache_dir);

    rc = rk_npu_f32i8f32_dynamic_autotune_family(
        ctx, &family_cfg, &family_fast, family_fast_s,
        &family_stable, family_stable_s);
    ok &= rc == RK_NPU_OK && family_fast.weight_config.K == K &&
          family_fast.weight_config.N == N &&
          family_fast_s[0].op_kind == RK_NPU_MATMUL_F32I8F32_DYNAMIC &&
          family_fast_s[1].k_tile == family_fast.weight_config.k_tile &&
          family_fast.weighted_total_us > 0;

    rk_npu_matmul_autotune_config constrained{};
    rk_npu_matmul_autotune_config_init(&constrained, 4, N, K);
    constrained.allowed_cpu_core_mask = allowed;
    constrained.warmup = 0;
    constrained.loops = 1;
    constrained.repeats = 1;
    constrained.required_k_tile = family_fast.weight_config.k_tile;
    rk_npu_matmul_strategy constrained_fast{};
    rc = rk_npu_f16i8f16_dynamic_autotune(
        ctx, &constrained, &constrained_fast, nullptr);
    ok &= rc == RK_NPU_OK &&
          constrained_fast.k_tile == family_fast.weight_config.k_tile;

    rk_npu_ctx* other_ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* cross_weight_domain = other_ctx
        ? rk_npu_iommu_domain_create(other_ctx, 0) : nullptr;
    rk_npu_matmul_weight_config cross_ctx_wc{K, N, 32};
    rk_npu_i8i8i32_weights* cross_ctx_weights = cross_weight_domain
        ? rk_npu_i8i8i32_weights_create(cross_weight_domain, &cross_ctx_wc, B.data())
        : nullptr;
    auto cross_ctx_s = strategy(
        RK_NPU_MATMUL_I8I8I32, M, N, K, 32, 32, 7, cpu_mask);
    rk_npu_matmul_workspace* cross_ctx_workspace =
        rk_npu_matmul_workspace_create(domain, &cross_ctx_s);
    std::vector<int32_t> cross_ctx_out((size_t)M * N);
    ok &= other_ctx && cross_ctx_weights && cross_ctx_workspace &&
          rk_npu_i8i8i32_run(
              cross_ctx_workspace, cross_ctx_weights, A8.data(),
              cross_ctx_out.data()) == RK_NPU_ERR_DOMAIN;
    rk_npu_matmul_workspace_free(cross_ctx_workspace);
    rk_npu_i8i8i32_weights_free(cross_ctx_weights);
    rk_npu_iommu_domain_free(cross_weight_domain);
    rk_npu_close(other_ctx);

    /* Same context and shape, but different address spaces, must be rejected
     * before any register patch or submit occurs. */
    rk_npu_i8i8i32_weights* domain1_weights =
        rk_npu_i8i8i32_weights_create(other_domain, &cross_ctx_wc, B.data());
    rk_npu_matmul_workspace* domain0_workspace =
        rk_npu_matmul_workspace_create(domain, &cross_ctx_s);
    ok &= domain1_weights && domain0_workspace &&
          rk_npu_i8i8i32_run(domain0_workspace, domain1_weights, A8.data(),
                              cross_ctx_out.data()) == RK_NPU_ERR_DOMAIN;
    rk_npu_matmul_workspace_free(domain0_workspace);
    rk_npu_i8i8i32_weights_free(domain1_weights);

    /* Keep three complete residency groups alive and alternate the global
     * device address space repeatedly.  Each result must remain exact after
     * switching away and back. */
    rk_npu_iommu_domain* domains[] = {domain, other_domain, third_domain};
    rk_npu_matmul_workspace* alternating_workspaces[3]{};
    rk_npu_i8i8i32_weights* alternating_weights[3]{};
    const auto alternating_s = strategy(
        RK_NPU_MATMUL_I8I8I32, M, N, K, K, N, 1, cpu_mask);
    const rk_npu_matmul_weight_config alternating_wc{K, N, K};
    for (int i = 0; i < 3; ++i) {
        alternating_workspaces[i] =
            rk_npu_matmul_workspace_create(domains[i], &alternating_s);
        alternating_weights[i] = rk_npu_i8i8i32_weights_create(
            domains[i], &alternating_wc, B.data());
        ok &= alternating_workspaces[i] && alternating_weights[i];
    }
    std::vector<int32_t> alternating_out((size_t)M * N);
    const int order[] = {0, 1, 2, 0};
    for (int repeat = 0; repeat < 25; ++repeat) {
        for (int index : order) {
            const int run_rc = rk_npu_i8i8i32_run(
                alternating_workspaces[index], alternating_weights[index],
                A8.data(), alternating_out.data());
            ok &= run_rc == RK_NPU_OK && alternating_out == ref_i32;
        }
    }
    for (int i = 2; i >= 0; --i) {
        rk_npu_i8i8i32_weights_free(alternating_weights[i]);
        rk_npu_matmul_workspace_free(alternating_workspaces[i]);
    }

    rk_npu_iommu_domain_free(third_domain);
    rk_npu_iommu_domain_free(other_domain);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    std::printf("test_quant_matmul: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#include "../src/rk_npu_dcomp.h"
#include "../src/rk_npu_kn_plan.h"
#include "rk_npu_matmul_f16.h"
#include "rk_npu_quant_matmul.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace {
void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
using Clock = std::chrono::steady_clock;
double us(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1000u) && ((man & 0x0fffu) || (half & 1u)))
        ++half;
    return (uint16_t)half;
}

uint64_t allowed_cpu_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0)
        return 1;
    uint64_t mask = 0;
    for (int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set))
            mask |= 1ull << cpu;
    return mask ? mask : 1;
}

uint64_t take_cpus(uint64_t allowed, int count) {
    uint64_t mask = 0;
    for (int cpu = 4; cpu <= 7 && __builtin_popcountll(mask) < count; ++cpu)
        if (allowed & (1ull << cpu))
            mask |= 1ull << cpu;
    for (int cpu = 0; cpu < 64 && __builtin_popcountll(mask) < count; ++cpu)
        if (allowed & (1ull << cpu))
            mask |= 1ull << cpu;
    return mask;
}

void gemm_i8(int M, int N, int K, const int8_t *A, const int8_t *B, std::vector<int32_t> &C) {
    C.assign((size_t)M * N, 0);
    for (int m = 0; m < M; ++m) {
        uint32_t *out = reinterpret_cast<uint32_t *>(C.data() + (size_t)m * N);
        for (int k = 0; k < K; ++k) {
            const int32_t a = A[(size_t)m * K + k];
            for (int n = 0; n < N; ++n)
                out[n] += (uint32_t)(a * (int32_t)B[(size_t)k * N + n]);
        }
    }
}

rk_npu_matmul_strategy strategy(rk_npu_matmul_op_kind kind, int M, int N, int K, int kt, int nt,
                                uint32_t npu_mask, uint64_t cpu_mask,
                                rk_npu_matmul_a_layout a_layout = RK_NPU_MATMUL_A_LAYOUT_NORMAL) {
    rk_npu_matmul_strategy s{};
    s.op_kind = kind;
    s.M = M;
    s.N = N;
    s.K = K;
    s.k_tile = kt;
    s.n_tile = nt;
    s.a_layout = a_layout;
    s.wave_count = (K + kt - 1) / kt;
    s.n_groups = ((N + 31) / 32 * 32 + nt - 1) / nt;
    s.npu_core_mask = npu_mask;
    s.cpu_core_mask = cpu_mask;
    s.cpu_threads = __builtin_popcountll(cpu_mask);
    return s;
}

template <class T> void load(const std::string &path, std::vector<T> &v) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T))), path.c_str());
}
template <class T> void save(const std::string &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    require(bool(f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T))),
            path.c_str());
}
void correctness(rk_npu_iommu_domain *d) {
    std::mt19937 rng(918);
    const uint64_t cpus = take_cpus(allowed_cpu_mask(), 2);
    for (auto shape : {std::vector<int>{1, 256, 256, 256},
                       {1, 32, 32, 32},
                       {4, 96, 96, 32},
                       {33, 70, 70, 32},
                       {129, 320, 160, 64},
                       {129, 320, 512, 256},
                       {4, 96, 32, 32},
                       {33, 320, 256, 256}}) {
        int M = shape[0], N = shape[1], K = shape[2], kt = shape[3];
        std::vector<int8_t> a(size_t(M) * K), b(size_t(K) * N);
        for (auto &x : a)
            x = int8_t(int(rng() % 256) - 128);
        for (size_t i = 0; i < b.size(); ++i) {
            if (K == 32 && N == 96)
                b[i] = 0;
            else if (K == 256 && N == 320 && i % N >= 160)
                b[i] = int8_t(int(rng() % 256) - 128);
            else
                b[i] = int8_t((K >= 256) ? int(rng() % 31) - 15 : int(rng() % 256) - 128);
        }
        std::vector<int32_t> ref, raw(size_t(M) * N), compressed(raw.size());
        gemm_i8(M, N, K, a.data(), b.data(), ref);
        rk_npu_matmul_weight_config wc{K, N, kt};
        auto *wr = rk_npu_i8i8i32_weights_create(d, &wc, b.data());
        auto *wz = rk_npu_i8i8i32_weights_create_compress(d, &wc, b.data());
        require(wr && wz, "create i8");
        // Source lifetime: exercise after overwriting the caller's input.
        std::fill(b.begin(), b.end(), 0);
        for (int native : {0, 1})
            for (int mask : {1, 2, 4, 3, 7}) {
                if (mask == 3 && N < 64)
                    continue;
                if (mask == 7 && N < 96)
                    continue;
                int nt = (mask == 1 || mask == 2 || mask == 4) ? ((N + 31) / 32 * 32) : 32;
                auto s = strategy(RK_NPU_MATMUL_I8I8I32, M, N, K, kt, nt, mask, cpus,
                                  native ? RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16
                                         : RK_NPU_MATMUL_A_LAYOUT_NORMAL);
                auto *ws = rk_npu_matmul_workspace_create(d, &s);
                require(ws, "workspace i8");
                for (int repeat = 0; repeat < 3; ++repeat) {
                    int rc = rk_npu_i8i8i32_run(ws, wz, a.data(), compressed.data());
                    if (rc)
                        std::fprintf(stderr, "i8 rc=%d M=%d N=%d K=%d native=%d mask=%d\n", rc, M,
                                     N, K, native, mask);
                    require(rc == 0, "run compressed i8");
                    require(rk_npu_i8i8i32_run(ws, wr, a.data(), raw.data()) == 0,
                            "run raw i8 after compressed");
                    require(raw == ref, "raw i8 CPU exact");
                    require(compressed == ref, "compressed i8 CPU exact");
                }
                rk_npu_matmul_workspace_free(ws);
            }
        rk_npu_i8i8i32_weights_free(wr);
        rk_npu_i8i8i32_weights_free(wz);
        std::printf("PASS i8 M=%d N=%d K=%d kt=%d codes exact, masks, layouts, transitions\n", M, N,
                    K, kt);
    }
    const int M = 8, N = 128, K = 256;
    rk_npu_matmul_weight_config wc{K, N, 128};
    std::vector<float> a(M * K), w(K * N), sc(N), as(M, 0.03f);
    std::vector<int8_t> b(K * N);
    std::vector<uint16_t> ah(M * K), hraw(M * N), hcomp(M * N);
    for (auto &x : a)
        x = float(int(rng() % 2001) - 1000) / 300;
    for (size_t i = 0; i < a.size(); ++i)
        ah[i] = float_to_half(a[i]);
    for (auto &x : b)
        x = int8_t(int(rng() % 31) - 15);
    for (int n = 0; n < N; ++n)
        sc[n] = 0.01f + n * 0.0001f;
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = float(int(rng() % 2001) - 1000) / 2000;
    auto *fr = rk_npu_f32i8f32_weights_create(d, &wc, b.data(), sc.data());
    auto *fz = rk_npu_f32i8f32_weights_create_compress(d, &wc, b.data(), sc.data());
    auto *hr = rk_npu_f16i8f16_weights_create(d, &wc, b.data(), sc.data());
    auto *hz = rk_npu_f16i8f16_weights_create_compress(d, &wc, b.data(), sc.data());
    require(fr && fz && hr && hz, "create scaled");
    for (int native : {0, 1})
        for (int dynamic : {0, 1}) {
            auto s = strategy(
                dynamic ? RK_NPU_MATMUL_F32I8F32_DYNAMIC : RK_NPU_MATMUL_F32I8F32_STATIC, M, N, K,
                128, 32, 7, cpus,
                native ? RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16 : RK_NPU_MATMUL_A_LAYOUT_NORMAL);
            auto *ws = rk_npu_matmul_workspace_create(d, &s);
            require(ws, "workspace f32");
            std::vector<float> raw(M * N), out(M * N);
            for (int rep = 0; rep < 3; ++rep) {
                require((dynamic ? rk_npu_f32i8f32_run_dynamic(ws, fz, a.data(), out.data())
                                 : rk_npu_f32i8f32_run_static(ws, fz, a.data(), as.data(),
                                                              out.data())) == 0,
                        "run f32 compressed");
                require((dynamic ? rk_npu_f32i8f32_run_dynamic(ws, fr, a.data(), raw.data())
                                 : rk_npu_f32i8f32_run_static(ws, fr, a.data(), as.data(),
                                                              raw.data())) == 0,
                        "run f32 raw");
                require(raw == out, "f32 bit exact");
            }
            rk_npu_matmul_workspace_free(ws);
            s.op_kind = dynamic ? RK_NPU_MATMUL_F16I8F16_DYNAMIC : RK_NPU_MATMUL_F16I8F16_STATIC;
            ws = rk_npu_matmul_workspace_create(d, &s);
            require(ws, "workspace f16");
            require((dynamic ? rk_npu_f16i8f16_run_dynamic(ws, hz, ah.data(), hcomp.data())
                             : rk_npu_f16i8f16_run_static(ws, hz, ah.data(), as.data(),
                                                          hcomp.data())) == 0,
                    "run f16 compressed");
            require((dynamic ? rk_npu_f16i8f16_run_dynamic(ws, hr, ah.data(), hraw.data())
                             : rk_npu_f16i8f16_run_static(ws, hr, ah.data(), as.data(),
                                                          hraw.data())) == 0,
                    "run f16 raw");
            require(hraw == hcomp, "f16 bit exact");
            rk_npu_matmul_workspace_free(ws);
        }
    rk_npu_f32i8f32_weights_free(fr);
    rk_npu_f32i8f32_weights_free(fz);
    rk_npu_f16i8f16_weights_free(hr);
    rk_npu_f16i8f16_weights_free(hz);
    std::vector<int8_t> q;
    std::vector<float> scales;
    require(rknpu2_matmul_open::detail::dcomp_quantize_f32(K, N, 128, w.data(), 6.5f, q, scales) == 0, "host RTN");
    fr = rk_npu_f32i8f32_weights_create(d, &wc, q.data(), scales.data());
    fz = rk_npu_f32i8f32_weights_create_from_f32_compress(d, &wc, w.data(), 6.5f);
    hr = rk_npu_f16i8f16_weights_create(d, &wc, q.data(), scales.data());
    hz = rk_npu_f16i8f16_weights_create_from_f32_compress(d, &wc, w.data(), 6.5f);
    require(fr && fz && hr && hz, "create from f32");
    auto s = strategy(RK_NPU_MATMUL_F32I8F32_DYNAMIC, M, N, K, 128, 64, 3, cpus);
    auto *ws = rk_npu_matmul_workspace_create(d, &s);
    require(ws, "workspace RTN");
    std::vector<float> raw(M * N), out(M * N);
    require(rk_npu_f32i8f32_run_dynamic(ws, fr, a.data(), raw.data()) == 0, "RTN raw");
    require(rk_npu_f32i8f32_run_dynamic(ws, fz, a.data(), out.data()) == 0, "RTN compressed");
    require(raw == out, "RTN f32 exact");
    rk_npu_matmul_workspace_free(ws);
    s.op_kind = RK_NPU_MATMUL_F16I8F16_DYNAMIC;
    ws = rk_npu_matmul_workspace_create(d, &s);
    require(ws, "workspace RTN f16");
    require(rk_npu_f16i8f16_run_dynamic(ws, hr, ah.data(), hraw.data()) == 0, "RTN half raw");
    require(rk_npu_f16i8f16_run_dynamic(ws, hz, ah.data(), hcomp.data()) == 0,
            "RTN half compressed");
    require(hraw == hcomp, "RTN f16 exact");
    rk_npu_matmul_workspace_free(ws);
    // Two workspaces simultaneously bind previously unseen N layouts on one
    // immutable weight handle. Each run has its own mutable workspace.
    auto *shared = rk_npu_f32i8f32_weights_create_compress(d, &wc, q.data(), scales.data());
    require(shared, "shared compressed weights");
    rk_npu_matmul_workspace *concurrent[2];
    for (int i = 0; i < 2; ++i) {
        auto cs = strategy(RK_NPU_MATMUL_F32I8F32_DYNAMIC, M, N, K, 128, i ? 64 : 32, 3, cpus);
        concurrent[i] = rk_npu_matmul_workspace_create(d, &cs);
        require(concurrent[i], "concurrent workspace");
    }
    std::atomic<bool> concurrent_ok{true};
    auto worker = [&](int i) {
        std::vector<float> result(M * N);
        for (int rep = 0; rep < 3; ++rep)
            if (rk_npu_f32i8f32_run_dynamic(concurrent[i], shared, a.data(), result.data()) != 0 ||
                result != raw)
                concurrent_ok = false;
    };
    std::thread t0(worker, 0), t1(worker, 1);
    t0.join();
    t1.join();
    require(concurrent_ok, "concurrent layout binding exact");
    // Exercise DCOMP -> fused FP16 -> DCOMP without an intervening raw/reset.
    for (auto op : {RK_NPU_FUSE_MUL, RK_NPU_FUSE_ADD}) {
        require(rk_npu_f32i8f32_run_dynamic(concurrent[1], shared, a.data(), out.data()) == 0,
                "before fused");
        rk_npu_matmul_f16_config fc;
        rk_npu_matmul_f16_config_init(&fc, 1, 32, 64, op);
        rk_npu_matmul_sizes size{};
        require(rk_npu_matmul_f16_query(&fc, &size) == 0, "fused query");
        rk_npu_mem ia{}, wb{}, operand{}, oc{};
        require(rk_npu_mem_alloc(d, size.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &ia) == 0,
                "fused A alloc");
        require(rk_npu_mem_alloc(d, size.weight_bytes, RK_NPU_MEM_NON_CACHEABLE, &wb) == 0,
                "fused B alloc");
        require(rk_npu_mem_alloc(d, size.operand_bytes, RK_NPU_MEM_NON_CACHEABLE, &operand) == 0,
                "fused operand alloc");
        require(rk_npu_mem_alloc(d, size.output_bytes, RK_NPU_MEM_NON_CACHEABLE, &oc) == 0,
                "fused C alloc");
        std::vector<uint16_t> fa(64, float_to_half(0.5f)), fb(64 * 32, float_to_half(0.5f)),
            fo(32, float_to_half(1));
        std::vector<float> fc_out(32);
        require(rk_npu_matmul_f16_pack_a(&fc, fa.data(), &ia) == 0, "fused pack A");
        require(rk_npu_matmul_f16_pack_b(&fc, fb.data(), &wb) == 0, "fused pack B");
        require(rk_npu_matmul_f16_pack_operand(&fc, fo.data(), &operand) == 0,
                "fused pack operand");
        auto *fp = rk_npu_matmul_f16_prepare(d, &fc);
        require(fp, "fused prepare");
        require(rk_npu_matmul_f16_run(d->ctx, fp, &ia, &wb, &operand, &oc) == 0,
                "fused after DCOMP");
        require(rk_npu_matmul_f16_unpack_d_f32(&fc, &oc, fc_out.data()) == 0, "fused unpack");
        for (float value : fc_out)
            require(value == (op == RK_NPU_FUSE_MUL ? 16.0f : 17.0f), "fused result exact");
        rk_npu_matmul_f16_plan_free(fp);
        for (auto *mem : {&ia, &wb, &operand, &oc})
            rk_npu_mem_free(d->ctx, mem);
        require(rk_npu_f32i8f32_run_dynamic(concurrent[1], shared, a.data(), out.data()) == 0 &&
                    out == raw,
                "DCOMP after fused");
    }
    for (auto *c : concurrent)
        rk_npu_matmul_workspace_free(c);
    rk_npu_f32i8f32_weights_free(shared);
    std::puts("PASS concurrent shared weights and DCOMP/fused-FP16 transitions");
    require(!rk_npu_f32i8f32_weights_create_from_f32_compress(d, &wc, w.data(), 0.1f),
            "reject unmet budget");
    scales[0] = 0;
    require(!rk_npu_f32i8f32_weights_create_compress(d, &wc, q.data(), scales.data()),
            "reject bad scale");
    rk_npu_f32i8f32_weights_free(fr);
    rk_npu_f32i8f32_weights_free(fz);
    rk_npu_f16i8f16_weights_free(hr);
    rk_npu_f16i8f16_weights_free(hz);
    std::puts("PASS f32/f16 dynamic/static, K waves, N splits, RTN constructors");
}
void bench(rk_npu_iommu_domain *d, const std::string &prefix, int M, int K, int N, int kt, int nt,
           int mask, int loops, float target = 0) {
    require(M > 0 && K > 0 && N > 0 && loops > 0, "bench dimensions");
    std::vector<int8_t> b(size_t(K) * N);
    std::vector<float> sc(N), a(size_t(M) * K), raw(size_t(M) * N), out(raw.size());
    load(prefix + ".codes.bin", b);
    load(prefix + ".scales.bin", sc);
    load(prefix + ".a.bin", a);
    rk_npu_matmul_weight_config wc{K, N, kt};
    auto t = Clock::now();
    auto *wr = rk_npu_f32i8f32_weights_create(d, &wc, b.data(), sc.data());
    double cr = us(t);
    std::vector<float> original;
    if (target > 0) {
        original.resize(size_t(K) * N);
        load(prefix + ".w.bin", original);
    }
    t = Clock::now();
    auto *wz =
        target > 0
            ? rk_npu_f32i8f32_weights_create_from_f32_compress(d, &wc, original.data(), target)
            : rk_npu_f32i8f32_weights_create_compress(d, &wc, b.data(), sc.data());
    double cz = us(t);
    require(wr && wz, "bench weights");
    auto s = strategy(RK_NPU_MATMUL_F32I8F32_DYNAMIC, M, N, K, kt, nt, mask,
                      take_cpus(allowed_cpu_mask(), 4));
    auto *ws = rk_npu_matmul_workspace_create(d, &s);
    require(ws, "bench workspace");
    t = Clock::now();
    require(rk_npu_f32i8f32_run_dynamic(ws, wr, a.data(), raw.data()) == 0, "bench raw first");
    double first_r = us(t);
    t = Clock::now();
    require(rk_npu_f32i8f32_run_dynamic(ws, wz, a.data(), out.data()) == 0,
            "bench compressed first");
    double first_z = us(t);
    require(raw == out, "bench exact");
    std::vector<double> tr, tz;
    for (int i = 0; i < loops + 4; ++i)
        for (int j = 0; j < 2; ++j) {
            bool compressed = ((i + j) & 1) != 0;
            t = Clock::now();
            require(rk_npu_f32i8f32_run_dynamic(ws, compressed ? wz : wr, a.data(),
                                                compressed ? out.data() : raw.data()) == 0,
                    "bench run");
            double elapsed = us(t);
            if (i >= 4)
                (compressed ? tz : tr).push_back(elapsed);
        }
    require(raw == out, "bench repeated exact");
    std::sort(tr.begin(), tr.end());
    std::sort(tz.begin(), tz.end());
    save(prefix + ".out.m" + std::to_string(M) + ".bin", out);
    std::printf("BENCH M=%d K=%d N=%d kt=%d nt=%d mask=%d raw_create_us=%.1f "
                "compress_create_us=%.1f raw_first_us=%.1f compress_first_us=%.1f raw_p50_us=%.2f "
                "compress_p50_us=%.2f speedup=%.3f exact=1\n",
                M, K, N, kt, nt, mask, cr, cz, first_r, first_z, tr[tr.size() / 2],
                tz[tz.size() / 2], tr[tr.size() / 2] / tz[tz.size() / 2]);
    rk_npu_matmul_workspace_free(ws);
    rk_npu_f32i8f32_weights_free(wr);
    rk_npu_f32i8f32_weights_free(wz);
}
} // namespace
int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto *ctx = rk_npu_open(nullptr);
    if (!ctx)
        return 2;
    auto *d = rk_npu_iommu_domain_create(ctx, 0);
    if (!d)
        return 2;
    int result = 0;
    try {
        if (argc == 1)
            correctness(d);
        else if (argc == 10 && std::string(argv[1]) == "--bench")
            bench(d, argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                  std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]), std::atoi(argv[9]));
        else if (argc == 11 && std::string(argv[1]) == "--bench-f32")
            bench(d, argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                  std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]), std::atoi(argv[9]),
                  std::strtof(argv[10], nullptr));
        else
            throw std::runtime_error("usage: test_compress [--bench prefix M K N kt nt mask loops] "
                                     "or [--bench-f32 ... bpw]");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        result = 1;
    }
    rk_npu_iommu_domain_free(d);
    rk_npu_close(ctx);
    return result;
}

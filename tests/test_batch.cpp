/*
 * test_batch.cpp - correctness/perf smoke test for batched int8 matmul.
 */
#include "test_common.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace rknpu2_matmul_open::test;

namespace {

struct Args {
    const char* dev = nullptr;
    int loops = 30;
    bool a_native = false;
    bool c_native = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (o == "--dev")   a.dev = next();
        else if (o == "--loops") a.loops = std::atoi(next());
        else if (o == "--a-native") a.a_native = true;
        else if (o == "--c-native") a.c_native = true;
        else { fprintf(stderr, "unknown option: %s\n", o.c_str()); std::exit(2); }
    }
    return a;
}

struct Case { int B, M, N, K; const char* name; };

bool verify_batch(int B, int M, int N, int K,
                  const std::vector<int8_t>& A,
                  const std::vector<int8_t>& W,
                  const std::vector<float>& C,
                  double& max_abs,
                  std::mt19937& rng) {
    bool ok = true;
    max_abs = 0.0;
    for (int b = 0; b < B; ++b) {
        const int8_t* Ab = A.data() + (size_t)b * M * K;
        const int8_t* Wb = W.data() + (size_t)b * K * N;
        const float*  Cb = C.data() + (size_t)b * M * N;
        Verify v = verify(M, N, K, Ab, Wb, Cb, 40000000ull, 4096, rng);
        ok = ok && v.ok;
        max_abs = std::max(max_abs, v.max_abs);
    }
    return ok;
}

int run_case(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
             const Case& c, const rk_npu_matmul_i8_config& cfg,
             int loops, std::mt19937& rng) {
    rk_npu_matmul_sizes one{}, sz{};
    int rc = rknpu2_matmul_open::detail::query_i8(&cfg, &one);
    if (rc != RK_NPU_OK) return 1;
    rc = rk_npu_matmul_i8_batch_query(c.B, &cfg, &sz);
    if (rc != RK_NPU_OK) return 1;

    bool size_ok = sz.input_bytes == one.input_bytes * (uint64_t)c.B &&
                   sz.weight_bytes == one.weight_bytes * (uint64_t)c.B &&
                   sz.output_bytes == one.output_bytes * (uint64_t)c.B &&
                   sz.num_tasks == one.num_tasks * c.B;
    if (!size_ok) {
        printf("%-18s BAD_QUERY\n", c.name);
        return 1;
    }

    rk_npu_mem in{}, wt{}, out{};
    if (rk_npu_mem_alloc(domain, sz.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &in) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_NON_CACHEABLE, &wt) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_NON_CACHEABLE, &out) != RK_NPU_OK) {
        printf("%-18s ALLOC_FAIL\n", c.name);
        if (in.vaddr) rk_npu_mem_free(ctx, &in);
        if (wt.vaddr) rk_npu_mem_free(ctx, &wt);
        if (out.vaddr) rk_npu_mem_free(ctx, &out);
        return 1;
    }

    std::vector<int8_t> A((size_t)c.B * c.M * c.K), W((size_t)c.B * c.K * c.N);
    std::vector<float> C((size_t)c.B * c.M * c.N);
    fill_i8(A, rng);
    fill_i8(W, rng);

    rk_npu_matmul_i8_batch_pack_a(c.B, &cfg, A.data(), &in);
    rk_npu_matmul_i8_batch_pack_b(c.B, &cfg, W.data(), &wt);
    rk_npu_matmul_i8_batch_plan* bp = rk_npu_matmul_i8_batch_prepare(domain, c.B, &cfg);
    rc = bp ? rk_npu_matmul_i8_batch_run(ctx, bp, &in, &wt, &out) : RK_NPU_ERR_NOMEM;
    if (rc == RK_NPU_OK) rk_npu_matmul_i8_batch_unpack_c(c.B, &cfg, &out, C.data());

    double max_abs = 0.0;
    bool ok = (rc == RK_NPU_OK) && verify_batch(c.B, c.M, c.N, c.K, A, W, C, max_abs, rng);

    double batch_us = 0.0, serial_us = 0.0;
    if (ok && loops > 0) {
        auto bt = timeit([&]{ rk_npu_matmul_i8_batch_run(ctx, bp, &in, &wt, &out); }, loops);
        batch_us = bt.first;

        rk_npu_matmul_i8_plan* sp = rknpu2_matmul_open::detail::prepare_i8(domain, &cfg);
        if (!sp) {
            printf("%-18s SINGLE_PREP_FAIL\n", c.name);
            ok = false;
        } else {
            std::vector<rk_npu_mem> inv(c.B), wtv(c.B), outv(c.B);
            for (int b = 0; b < c.B; ++b) {
                rk_npu_mem_view(&in,  (uint64_t)b * one.input_bytes,  one.input_bytes,  &inv[b]);
                rk_npu_mem_view(&wt,  (uint64_t)b * one.weight_bytes, one.weight_bytes, &wtv[b]);
                rk_npu_mem_view(&out, (uint64_t)b * one.output_bytes, one.output_bytes, &outv[b]);
            }
            auto st = timeit([&]{
                for (int b = 0; b < c.B; ++b)
                    rknpu2_matmul_open::detail::run_i8(ctx, sp, &inv[b], &wtv[b], &outv[b]);
            }, loops);
            serial_us = st.first;
            rknpu2_matmul_open::detail::free_i8_plan(sp);
        }
    }

    printf("%-18s B=%-2d M=%-4d K=%-4d N=%-4d tasks=%-4d %s max_abs=%.0f",
           c.name, c.B, c.M, c.K, c.N, sz.num_tasks, ok ? "PASS" : "FAIL", max_abs);
    if (ok && loops > 0) {
        printf(" batch_us=%.2f serial_us=%.2f speedup=%.2fx",
               batch_us, serial_us, serial_us / std::max(0.001, batch_us));
    }
    printf("\n");

    rk_npu_matmul_i8_batch_plan_free(bp);
    rk_npu_mem_free(ctx, &in);
    rk_npu_mem_free(ctx, &wt);
    rk_npu_mem_free(ctx, &out);
    return ok ? 0 : 1;
}

} /* namespace */

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    rk_npu_ctx* ctx = rk_npu_open(a.dev);
    if (!ctx) { fprintf(stderr, "rk_npu_open failed\n"); return 1; }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }

    std::mt19937 rng(123);
    const Case cases[] = {
        {1, 1,   16,  16,  "b1_equiv"},
        {2, 1,   16,  16,  "tiny_b2"},
        {4, 2,   64,  64,  "small_b4"},
        {8, 4,   64,  16,  "small_b8"},
        {3, 5,   80,  33,  "odd_b3"},
        {4, 3,   64,  512, "mtile1_b4"},
        {2, 900, 32,  384, "large_m_b2"},
    };

    int fail = 0;
    printf("== rk_npu_matmul batched single-submit test%s%s ==\n",
           a.a_native ? " [A=native]" : "",
           a.c_native ? " [C=native]" : "");
    for (const Case& c : cases) {
        rk_npu_matmul_i8_config cfg{};
        rk_npu_matmul_i8_config_init(&cfg, c.M, c.N, c.K);
        cfg.a_layout = a.a_native ? RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16
                                  : RK_NPU_I8_A_LAYOUT_NORMAL;
        cfg.c_layout = a.c_native ? RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4
                                  : RK_NPU_I8_C_LAYOUT_NORMAL_PADDED;
        fail += run_case(ctx, domain, c, cfg, a.loops, rng);
    }

    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    printf("%s\n", fail ? "SOME FAILED" : "ALL PASS");
    return fail ? 1 : 0;
}

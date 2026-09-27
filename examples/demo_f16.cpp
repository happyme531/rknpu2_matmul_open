/*
 * demo_f16 - exercise the fp16 fused matmul: D = A@B, A@B*C0, A@B+C0.
 * Compares the NPU result against a CPU fp32 reference (fp16 inputs).
 */
#include "rk_npu_matmul_f16.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

/* Self-contained helpers (no <cmath>/<random>) so the cross-built binary runs on
 * the board without pulling a newer glibc/libm than the device ships. */
template <typename T> static inline T fabs_(T x) { return x < 0 ? -x : x; }
template <typename T> static inline T max_(T a, T b) { return a > b ? a : b; }
struct LCG { uint64_t s; float next() { s = s*6364136223846793005ull + 1442695040888963407ull;
    return (float)((s >> 40) & 0xffff) / 32768.0f - 1.0f; } };   /* uniform in [-1,1) */

/* float -> IEEE half bit pattern (round-to-nearest-even, good enough for a demo) */
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = x & 0x7fffff;
    if (exp <= 0) return (uint16_t)sign;                 /* flush subnormals to 0 */
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00);   /* overflow -> inf */
    return (uint16_t)(sign | (exp << 10) | (man >> 13));
}
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) { bits = sign; }                       /* (demo ignores subnormals) */
    else if (exp == 0x1f) bits = sign | 0x7f800000 | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

static int run_case(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                    int M, int N, int K, rk_npu_fuse_op op, const char* name) {
    rk_npu_matmul_f16_config cfg{};
    rk_npu_matmul_f16_config_init(&cfg, M, N, K, op);
    LCG rng{ (uint64_t)(1234 + M*7 + N*13 + K + (int)op) };
    std::vector<uint16_t> A(M*K), B(K*N), C0(M*N), D(M*N);
    std::vector<float> Af(M*K), Bf(K*N), C0f(M*N);
    for (auto& i : A) i = f2h(rng.next() * 2.f);
    for (auto& i : B) i = f2h(rng.next() * 2.f);
    for (int i = 0; i < M*N; ++i) { float v = (float)(int)(rng.next() * 4.f); C0f[i] = v; C0[i] = f2h(v); }
    for (int i = 0; i < M*K; ++i) Af[i] = h2f(A[i]);
    for (int i = 0; i < K*N; ++i) Bf[i] = h2f(B[i]);

    rk_npu_matmul_sizes sz;
    int rc = rk_npu_matmul_f16_query(&cfg, &sz);
    if (rc) { printf("  %-22s query ERROR %s\n", name, rk_npu_strerror(rc)); return 1; }

    rk_npu_mem in{}, wt{}, op0{}, out{};
    if (rk_npu_mem_alloc(domain, sz.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &in) ||
        rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_NON_CACHEABLE, &wt) ||
        (op != RK_NPU_FUSE_NONE && rk_npu_mem_alloc(domain, sz.operand_bytes, RK_NPU_MEM_NON_CACHEABLE, &op0)) ||
        rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_NON_CACHEABLE, &out)) {
        printf("  %-22s alloc ERROR\n", name);
        rk_npu_mem_free(ctx, &in); rk_npu_mem_free(ctx, &wt);
        if (op0.vaddr) rk_npu_mem_free(ctx, &op0);
        rk_npu_mem_free(ctx, &out);
        return 1;
    }
    rk_npu_matmul_f16_plan* plan = rk_npu_matmul_f16_prepare(domain, &cfg);
    if (!plan) {
        printf("  %-22s prepare ERROR\n", name);
        rk_npu_mem_free(ctx, &in); rk_npu_mem_free(ctx, &wt);
        if (op0.vaddr) rk_npu_mem_free(ctx, &op0);
        rk_npu_mem_free(ctx, &out);
        return 1;
    }

    rk_npu_matmul_f16_pack_a(&cfg, A.data(), &in);
    rk_npu_matmul_f16_pack_b(&cfg, B.data(), &wt);
    if (op != RK_NPU_FUSE_NONE) rk_npu_matmul_f16_pack_operand(&cfg, C0.data(), &op0);
    rc = rk_npu_matmul_f16_run(ctx, plan, &in, &wt, op==RK_NPU_FUSE_NONE?nullptr:&op0, &out);
    if (rc) { printf("  %-22s ERROR %s\n", name, rk_npu_strerror(rc)); }
    else rk_npu_matmul_f16_unpack_d(&cfg, &out, D.data());

    rk_npu_matmul_f16_plan_free(plan);
    rk_npu_mem_free(ctx, &in); rk_npu_mem_free(ctx, &wt);
    if (op0.vaddr) rk_npu_mem_free(ctx, &op0);
    rk_npu_mem_free(ctx, &out);
    if (rc) return 1;

    double md = 0, scale = 1;
    for (int r = 0; r < M; ++r) for (int c = 0; c < N; ++c) {
        double acc = 0; for (int k = 0; k < K; ++k) acc += (double)Af[r*K+k]*Bf[k*N+c];
        double ref = op==RK_NPU_FUSE_MUL ? acc*C0f[r*N+c]
                   : op==RK_NPU_FUSE_ADD ? acc+C0f[r*N+c] : acc;
        md = max_(md, fabs_((double)h2f(D[r*N+c]) - ref));
        scale = max_(scale, fabs_(ref));
    }
    bool ok = md < 0.03 * scale + 0.05;
    printf("  %-22s %s  max_diff=%.4f (scale=%.1f)\n", name, ok?"PASS":"FAIL", md, scale);
    return ok ? 0 : 1;
}

/* Prepared fast-path: build a fixed-shape plan once, re-run with fresh data each iter. */
static int prepared_test(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                         int M, int N, int K, rk_npu_fuse_op op, const char* name) {
    rk_npu_matmul_f16_config cfg{};
    rk_npu_matmul_f16_config_init(&cfg, M, N, K, op);
    rk_npu_matmul_sizes sz;
    if (rk_npu_matmul_f16_query(&cfg, &sz)) return 1;
    rk_npu_mem in{}, wt{}, op0{}, out{};
    rk_npu_mem_alloc(domain, sz.input_bytes,  RK_NPU_MEM_NON_CACHEABLE,  &in);
    rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_NON_CACHEABLE,  &wt);
    if (sz.operand_bytes) rk_npu_mem_alloc(domain, sz.operand_bytes, RK_NPU_MEM_NON_CACHEABLE, &op0);
    rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_NON_CACHEABLE,  &out);

    rk_npu_matmul_f16_plan* plan = rk_npu_matmul_f16_prepare(domain, &cfg);
    int fail = 0;
    if (!plan) { printf("  %-22s prepare FAILED\n", name); fail = 1; }
    for (int it = 0; plan && it < 3; ++it) {
        LCG rng{ (uint64_t)(99 + it*31 + (int)op) };
        std::vector<uint16_t> A(M*K), B(K*N), C0(M*N), D(M*N);
        std::vector<float> Af(M*K), Bf(K*N), C0f(M*N);
        for (auto& i : A) i = f2h(rng.next()*2.f);
        for (auto& i : B) i = f2h(rng.next()*2.f);
        for (int i=0;i<M*N;++i){ float v=(float)(int)(rng.next()*4.f); C0f[i]=v; C0[i]=f2h(v); }
        for (int i=0;i<M*K;++i) Af[i]=h2f(A[i]);
        for (int i=0;i<K*N;++i) Bf[i]=h2f(B[i]);
        rk_npu_matmul_f16_pack_a(&cfg, A.data(), &in);
        rk_npu_matmul_f16_pack_b(&cfg, B.data(), &wt);
        if (op != RK_NPU_FUSE_NONE) rk_npu_matmul_f16_pack_operand(&cfg, C0.data(), &op0);
        int rc = rk_npu_matmul_f16_run(ctx, plan, &in, &wt,
                                        op==RK_NPU_FUSE_NONE?nullptr:&op0, &out);
        if (rc) { printf("  %-22s iter%d ERROR %s\n", name, it, rk_npu_strerror(rc)); fail=1; break; }
        rk_npu_matmul_f16_unpack_d(&cfg, &out, D.data());
        double md=0, scale=1;
        for (int r=0;r<M;++r) for (int c=0;c<N;++c){
            double acc=0; for(int k=0;k<K;++k) acc+=(double)Af[r*K+k]*Bf[k*N+c];
            double ref = op==RK_NPU_FUSE_MUL?acc*C0f[r*N+c]:op==RK_NPU_FUSE_ADD?acc+C0f[r*N+c]:acc;
            md=max_(md,fabs_((double)h2f(D[r*N+c])-ref)); scale=max_(scale,fabs_(ref));
        }
        bool ok = md < 0.03*scale + 0.05; fail += !ok;
        printf("  %-15s iter%d %s max_diff=%.4f\n", name, it, ok?"PASS":"FAIL", md);
    }
    rk_npu_matmul_f16_plan_free(plan);
    rk_npu_mem_free(ctx,&in); rk_npu_mem_free(ctx,&wt);
    if (op0.vaddr) rk_npu_mem_free(ctx,&op0);
    rk_npu_mem_free(ctx,&out);
    return fail;
}

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : nullptr;
    rk_npu_ctx* ctx = rk_npu_open(dev);
    if (!ctx) { printf("open failed\n"); return 1; }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }
    int shapes[][3] = {{4,32,32},{8,64,64},{3,48,40},{16,128,96},{5,80,33},{2,16,16}};
    int fail = 0;
    for (auto& s : shapes) {
        printf("[%dx%dx%d]\n", s[0], s[1], s[2]);
        fail += run_case(ctx, domain, s[0], s[1], s[2], RK_NPU_FUSE_NONE, "matmul (D=A@B)");
        fail += run_case(ctx, domain, s[0], s[1], s[2], RK_NPU_FUSE_MUL,  "fused mul (D=A@B*C0)");
        fail += run_case(ctx, domain, s[0], s[1], s[2], RK_NPU_FUSE_ADD,  "fused add (D=A@B+C0)");
    }
    printf("\n[prepared plan: internal regcmd/task, run repeatedly]\n");
    fail += prepared_test(ctx, domain, 8, 64, 64, RK_NPU_FUSE_NONE, "prep matmul");
    fail += prepared_test(ctx, domain, 8, 64, 64, RK_NPU_FUSE_MUL,  "prep mul");
    fail += prepared_test(ctx, domain, 8, 64, 64, RK_NPU_FUSE_ADD,  "prep add");
    fail += prepared_test(ctx, domain, 16, 128, 96, RK_NPU_FUSE_ADD, "prep add big");
    printf("\n%s\n", fail ? "SOME FAILED" : "ALL PASS");
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return fail ? 1 : 0;
}

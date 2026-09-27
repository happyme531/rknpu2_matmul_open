/*
 * test_mem_api.cpp - smoke test domain-aware native GEM and subrange views.
 */
#include "rk_npu_matmul.h"
#include "../src/rk_npu_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

static uint64_t align_up_u64(uint64_t x, uint64_t a) {
    return ((x + a - 1) / a) * a;
}

static void cpu_ref(int M, int N, int K, const int8_t* A, const int8_t* B, std::vector<float>& C) {
    C.assign((size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                C[(size_t)m * N + n] += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)k * N + n];
}

static int run_case(const char* dev, uint32_t domain_id) {
    rk_npu_ctx* ctx = rk_npu_open(dev);
    if (!ctx) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, domain_id);
    if (!domain) { rk_npu_close(ctx); return 1; }

    const int M = 8, N = 32, K = 32;
    rk_npu_matmul_i8_config cfg{};
    rk_npu_matmul_i8_config_init(&cfg, M, N, K);
    rk_npu_matmul_sizes sz{};
    if (rknpu2_matmul_open::detail::query_i8(&cfg, &sz) != RK_NPU_OK) return 1;

    const uint64_t page = 4096;
    uint64_t in_off = 0;
    uint64_t wt_off = align_up_u64(in_off + sz.input_bytes, page);
    uint64_t out_off = align_up_u64(wt_off + sz.weight_bytes, page);
    uint64_t total = align_up_u64(out_off + sz.output_bytes, page);

    rk_npu_mem base{};
    int rc = rk_npu_mem_alloc(domain, total, RK_NPU_MEM_DATA_DEFAULT, &base);
    if (rc != RK_NPU_OK) {
        fprintf(stderr, "domain %u alloc failed: %s\n",
                domain_id, rk_npu_strerror(rc));
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
        return 1;
    }

    rk_npu_mem in{}, wt{}, out{};
    rc = rk_npu_mem_view(&base, in_off, sz.input_bytes, &in);
    rc |= rk_npu_mem_view(&base, wt_off, sz.weight_bytes, &wt);
    rc |= rk_npu_mem_view(&base, out_off, sz.output_bytes, &out);
    if (rc != RK_NPU_OK) {
        fprintf(stderr, "view failed\n");
        return 1;
    }

    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)((int)(i % 17) - 8);
    for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)((int)(i % 13) - 6);
    rknpu2_matmul_open::detail::pack_i8_a(&cfg, A.data(), &in);
    rknpu2_matmul_open::detail::pack_i8_b(&cfg, B.data(), &wt);
    rk_npu_mem_sync(ctx, &in, RK_NPU_SYNC_TO_DEVICE);
    rk_npu_mem_sync(ctx, &wt, RK_NPU_SYNC_TO_DEVICE);

    rk_npu_matmul_i8_plan* plan = rknpu2_matmul_open::detail::prepare_i8(domain, &cfg);
    rc = plan ? rknpu2_matmul_open::detail::run_i8(ctx, plan, &in, &wt, &out) : RK_NPU_ERR_NOMEM;
    rknpu2_matmul_open::detail::free_i8_plan(plan);
    rk_npu_mem_sync(ctx, &out, RK_NPU_SYNC_FROM_DEVICE);
    if (rc != RK_NPU_OK) {
        fprintf(stderr, "run failed: %s\n", rk_npu_strerror(rc));
        return 1;
    }

    std::vector<float> C((size_t)M * N), ref;
    rknpu2_matmul_open::detail::unpack_i8_c(&cfg, &out, C.data());
    cpu_ref(M, N, K, A.data(), B.data(), ref);
    double maxd = 0.0;
    for (size_t i = 0; i < C.size(); ++i) maxd = std::max(maxd, std::fabs((double)C[i] - ref[i]));

    rk_npu_mem_free(ctx, &in);
    rk_npu_mem_free(ctx, &wt);
    rk_npu_mem_free(ctx, &out);
    rk_npu_mem_free(ctx, &base);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);

    printf("native GEM/view domain %u: %s (max_diff=%g)\n",
           domain_id, maxd == 0.0 ? "PASS" : "FAIL", maxd);
    return maxd == 0.0 ? 0 : 1;
}

static int run_import_case(const char* dev) {
    rk_npu_ctx* producer = rk_npu_open(dev);
    rk_npu_ctx* consumer = rk_npu_open(dev);
    if (!producer || !consumer) return 1;
    rk_npu_iommu_domain* producer_domain =
        rk_npu_iommu_domain_create(producer, 0);
    rk_npu_iommu_domain* consumer_domain =
        rk_npu_iommu_domain_create(consumer, 0);
    if (!producer_domain || !consumer_domain) return 1;

    const int M = 8, N = 32, K = 32;
    rk_npu_matmul_i8_config cfg{};
    rk_npu_matmul_i8_config_init(&cfg, M, N, K);
    rk_npu_matmul_sizes sz{};
    if (rknpu2_matmul_open::detail::query_i8(&cfg, &sz) != RK_NPU_OK) return 1;
    const uint64_t page = 4096;
    const uint64_t in_off = 0;
    const uint64_t wt_off = align_up_u64(sz.input_bytes, page);
    const uint64_t out_off = align_up_u64(wt_off + sz.weight_bytes, page);
    const uint64_t total = align_up_u64(out_off + sz.output_bytes, page);

    rk_npu_mem base{}, imported{};
    int rc = rk_npu_mem_alloc(producer_domain, total,
                              RK_NPU_MEM_DATA_DEFAULT, &base);
    int dmabuf_fd = -1;
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_export_dmabuf(producer, &base, &dmabuf_fd);
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_import_dmabuf(
            consumer_domain, dmabuf_fd, base.vaddr, total,
            RK_NPU_MEM_DATA_DEFAULT, &imported);
    if (dmabuf_fd >= 0) close(dmabuf_fd);
    if (rc != RK_NPU_OK) {
        fprintf(stderr, "dmabuf import failed: %s\n", rk_npu_strerror(rc));
        return 1;
    }

    rk_npu_mem in{}, wt{}, out{};
    rc = rk_npu_mem_view(&imported, in_off, sz.input_bytes, &in);
    rc |= rk_npu_mem_view(&imported, wt_off, sz.weight_bytes, &wt);
    rc |= rk_npu_mem_view(&imported, out_off, sz.output_bytes, &out);
    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (int8_t)((int)(i % 17) - 8);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (int8_t)((int)(i % 13) - 6);
    if (rc == RK_NPU_OK) rc = rknpu2_matmul_open::detail::pack_i8_a(&cfg, A.data(), &in);
    if (rc == RK_NPU_OK) rc = rknpu2_matmul_open::detail::pack_i8_b(&cfg, B.data(), &wt);
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_sync(consumer, &in, RK_NPU_SYNC_TO_DEVICE);
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_sync(consumer, &wt, RK_NPU_SYNC_TO_DEVICE);
    rk_npu_matmul_i8_plan* plan = rc == RK_NPU_OK
        ? rknpu2_matmul_open::detail::prepare_i8(consumer_domain, &cfg) : nullptr;
    if (rc == RK_NPU_OK)
        rc = plan ? rknpu2_matmul_open::detail::run_i8(consumer, plan, &in, &wt, &out)
                  : RK_NPU_ERR_NOMEM;
    rknpu2_matmul_open::detail::free_i8_plan(plan);
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_sync(consumer, &out, RK_NPU_SYNC_FROM_DEVICE);

    std::vector<float> C((size_t)M * N), ref;
    if (rc == RK_NPU_OK) rc = rknpu2_matmul_open::detail::unpack_i8_c(&cfg, &out, C.data());
    cpu_ref(M, N, K, A.data(), B.data(), ref);
    double maxd = 0.0;
    if (rc == RK_NPU_OK)
        for (size_t i = 0; i < C.size(); ++i)
            maxd = std::max(maxd, std::fabs((double)C[i] - ref[i]));

    rk_npu_mem_free(consumer, &in);
    rk_npu_mem_free(consumer, &wt);
    rk_npu_mem_free(consumer, &out);
    rk_npu_mem_free(consumer, &imported);
    rk_npu_mem_free(producer, &base);
    rk_npu_iommu_domain_free(consumer_domain);
    rk_npu_iommu_domain_free(producer_domain);
    rk_npu_close(consumer);
    rk_npu_close(producer);
    printf("dmabuf import/view domain 0: %s (max_diff=%g)\n",
           rc == RK_NPU_OK && maxd == 0.0 ? "PASS" : "FAIL", maxd);
    return rc == RK_NPU_OK && maxd == 0.0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : nullptr;
    for (uint32_t domain : {0u, 1u, 2u}) {
        const int rc = run_case(dev, domain);
        if (rc != 0) return rc;
    }
    return run_import_case(dev);
}

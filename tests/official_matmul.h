/*
 * official_matmul.h - optional benchmark wrapper for the vendor rknn_matmul API.
 * ============================================================================
 * Compiled into test_perf only when RK_WITH_OFFICIAL is defined (the official
 * headers + librknnrt.so live on the board, not in this repo).  Enable with:
 *
 *   cmake -B build -DRK_NPU_OFFICIAL=ON \
 *     -DRKNN_INCLUDE_DIR=.../runtime/Linux/librknn_api/include \
 *     -DRKNN_LIB=/usr/lib/librknnrt.so
 *
 * It runs the same int8 x int8 GEMM through the vendor runtime, times
 * rknn_matmul_run() the same way we time our prepared run(), and verifies the
 * vendor result against our CPU reference so the comparison is apples-to-apples.
 * Two output dtypes are selectable (out_int32):
 *   - false: RKNN_INT8_MM_INT8_TO_FLOAT32 -- the exact op our library does.
 *   - true : RKNN_INT8_MM_INT8_TO_INT32   -- the dtype the vendor's headline
 *            ~1.8 TOPS int8 number is quoted on.
 * B is native layout, A/C normal, and no quant params are set (default identity
 * scale -> output is the raw int32 sum, matching our int32->fp32 path).
 *
 * NOTE: the header here is rknnrt 2.3.2; link the *matching* 2.3.2 librknnrt.so
 * (e.g. rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so), not an older
 * /usr/lib/librknnrt.so -- a struct/enum mismatch otherwise yields
 * "unsupported matmul dtype: UNKNOW" or a crash.
 */
#ifndef RK_OFFICIAL_MATMUL_H
#define RK_OFFICIAL_MATMUL_H
#ifdef RK_WITH_OFFICIAL

#include "test_common.h"        /* timeit, verify, Verify */
#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <cstring>
#include <cstdint>
#include <random>

namespace rknpu2_matmul_open::test {

struct OfficialResult {
    bool   created  = false;  /* rknn_matmul_create succeeded   */
    bool   ran      = false;  /* at least one run returned 0    */
    bool   ok       = false;  /* vendor result matched our ref  */
    bool   verified = false;  /* result was actually checked    */
    double avg_us   = 0, min_us = 0, gops = 0;
};

/* Bench one shape on the vendor runtime.  A_rowmajor is M*K, B_rowmajor is K*N.
 *   out_int32  : INT8_TO_INT32 (vendor 1.8 TOPS dtype) vs INT8_TO_FLOAT32.
 *   ac_native  : AC_layout=NATIVE -- the vendor's real high-perf path (avoids the
 *                per-run A/C layout conversion that, at large N, costs ~3x).  This
 *                is how the vendor reaches ~1.8 TOPS at 128x1024x8192.  In native
 *                mode A/C are in the vendor's "perf layout", so the output can't be
 *                checked against our normal-layout reference -- timing only, the
 *                run is left UNVERIFIED (the matmul still does the same work, so the
 *                wall-clock is valid). */
inline OfficialResult bench_official(int M, int N, int K,
                                     const int8_t* A_rowmajor, const int8_t* B_rowmajor,
                                     uint64_t full_budget, int samples,
                                     std::mt19937& rng, int loops,
                                     bool out_int32, bool ac_native) {
    OfficialResult r;

    rknn_matmul_info info;  std::memset(&info, 0, sizeof(info));
    info.M = M; info.K = K; info.N = N;
    info.type      = out_int32 ? RKNN_INT8_MM_INT8_TO_INT32
                               : RKNN_INT8_MM_INT8_TO_FLOAT32;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;                          /* vendor high-perf path */
    info.AC_layout = ac_native ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;

    rknn_matmul_io_attr io;  std::memset(&io, 0, sizeof(io));
    rknn_matmul_ctx ctx = 0;
    if (rknn_matmul_create(&ctx, &info, &io) != 0) return r;
    r.created = true;

    /* No rknn_matmul_set_quant_params(): default (identity) scale -> fp32 C is the
     * raw int32 sum, matching our int32->fp32 output. */
    rknn_tensor_mem* A = rknn_create_mem(ctx, io.A.size);
    rknn_tensor_mem* B = rknn_create_mem(ctx, io.B.size);
    rknn_tensor_mem* C = rknn_create_mem(ctx, io.C.size);
    if (A && B && C) {
        std::memcpy(A->virt_addr, A_rowmajor, (size_t)M * K);
        rknn_B_normal_layout_to_native_layout((void*)B_rowmajor, B->virt_addr, K, N, &info);
        rknn_matmul_set_io_mem(ctx, A, &io.A);
        rknn_matmul_set_io_mem(ctx, B, &io.B);
        rknn_matmul_set_io_mem(ctx, C, &io.C);

        if (rknn_matmul_run(ctx) == 0) {
            r.ran = true;
            if (!ac_native) {
                /* normal layout: C is row-major, verify against our reference.
                 * int32 output is cast to float (same as our int32->fp32). */
                Verify v;
                if (out_int32) {
                    const int32_t* ci = (const int32_t*)C->virt_addr;
                    std::vector<float> Cf((size_t)M * N);
                    for (size_t i = 0; i < (size_t)M * N; ++i) Cf[i] = (float)ci[i];
                    v = verify(M, N, K, A_rowmajor, B_rowmajor, Cf.data(), full_budget, samples, rng);
                } else {
                    v = verify(M, N, K, A_rowmajor, B_rowmajor,
                               (const float*)C->virt_addr, full_budget, samples, rng);
                }
                r.ok = v.ok; r.verified = true;
            } else {
                /* native A/C layout: output is in the vendor perf layout, not
                 * comparable to our normal-layout reference -- timing only. */
                r.ok = true; r.verified = false;
            }
            auto pp = timeit([&]{ rknn_matmul_run(ctx); }, loops);
            r.avg_us = pp.first; r.min_us = pp.second;
            r.gops = 2.0 * M * N * K / (r.avg_us * 1e-6) / 1e9;
        }
    }

    if (A) rknn_destroy_mem(ctx, A);
    if (B) rknn_destroy_mem(ctx, B);
    if (C) rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    return r;
}

} /* namespace rknpu2_matmul_open::test */

#endif /* RK_WITH_OFFICIAL */
#endif /* RK_OFFICIAL_MATMUL_H */

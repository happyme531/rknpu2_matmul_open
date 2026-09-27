/*
 * test_common.h - shared scaffolding for the rk_npu_matmul test framework.
 * =======================================================================
 * Used by both test_correctness.cpp and test_perf.cpp.  All int8 x int8 -> fp32.
 *
 * What lives here:
 *   - the M / N / K dimension sweeps (segmented, deduplicated, sorted)
 *   - deterministic int8 random fill
 *   - an EXACT cpu reference: int8 MACs accumulated in int64 then cast ONCE to
 *     fp32 -- this matches the NPU, which accumulates in int32 and does a single
 *     int32->fp32 convert on chip.  (Accumulating in fp32 like the old demo
 *     rounds every step and diverges once a column sum exceeds 2^24.)
 *   - sampled verification: for big shapes we only reference a random subset of
 *     output positions (O(samples*K)) plus a full NaN/Inf scan, so verification
 *     never dominates the sweep.
 *   - a per-shape run wrapper that packs/runs and classifies the
 *     outcome into a Status, with a tunable resource guard so pathologically
 *     large shapes are skipped instead of stalling the device.
 *
 * Note on "valid" shapes: the private raw-query helper accepts any positive M,N,K; the
 * hardware limits are implicit.  N is padded to 32 and the DPU notch caps at 13
 * groups, so N beyond ~416 currently produces WRONG results (no N tiling yet);
 * large K forces one PC-chained task per row.  Mapping exactly where PASS turns
 * to FAIL is the whole point of the sweep.
 */
#ifndef RK_NPU_TEST_COMMON_H
#define RK_NPU_TEST_COMMON_H

#include "rk_npu_matmul.h"
#include "../src/rk_npu_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <algorithm>

namespace rknpu2_matmul_open::test {

/* ----------------------------------------------------------- dimension sweeps */

struct Seg { int lo, hi, step; };

/* Union of inclusive [lo,hi] stepped ranges, deduplicated and sorted ascending.
 * Overlapping segment endpoints (4,16,128,1024,...) collapse naturally. */
inline std::vector<int> sweep(std::initializer_list<Seg> segs) {
    std::set<int> s;
    for (const Seg& g : segs)
        for (int v = g.lo; v <= g.hi; v += g.step) s.insert(v);
    return std::vector<int>(s.begin(), s.end());
}

/* The task's requested ranges. N uses the same value set as K. */
inline std::vector<int> m_list() {
    return sweep({{1,4,1}, {4,16,4}, {16,128,16}, {128,1024,128}});
}
inline std::vector<int> k_list() {
    return sweep({{16,128,16}, {128,1024,128}, {1024,8192,512}});
}
inline std::vector<int> n_list() { return k_list(); }

/* Smaller exponential grids for quick capability/perf checks. */
inline std::vector<int> m_exp_list() {
    return {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
}
inline std::vector<int> k_exp_list() {
    return {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
}
inline std::vector<int> n_exp_list() { return k_exp_list(); }

/* Keep only the list entries inside [lo,hi] (for CLI sub-range selection). */
inline std::vector<int> clamp_list(const std::vector<int>& in, int lo, int hi) {
    std::vector<int> out;
    for (int v : in) if (v >= lo && v <= hi) out.push_back(v);
    return out;
}

/* --------------------------------------------------------------- data + ref */

inline void fill_i8(std::vector<int8_t>& v, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(-128, 127);
    for (auto& x : v) x = (int8_t)d(rng);
}

/* Exact reference for a single output element, int64 accumulate -> fp32. */
inline float ref_elem(int N, int K, const int8_t* A, const int8_t* B, int m, int n) {
    int64_t acc = 0;
    const int8_t* a = A + (size_t)m * K;
    for (int k = 0; k < K; ++k) acc += (int64_t)a[k] * (int64_t)B[(size_t)k * N + n];
    return (float)acc;
}

struct Verify {
    bool     ok = false;
    double   max_abs = 0;     /* worst |got - ref| over checked positions */
    double   max_rel = 0;     /* worst relative error                     */
    size_t   checked = 0;     /* how many output elements were referenced */
    bool     sampled = false; /* true if only a subset was referenced     */
    bool     bad_value = false; /* a NaN/Inf was seen in the output        */
};

/*
 * Verify C against the int8 reference.
 *   full when M*N*K <= full_budget, else sample `samples` random positions.
 * A position passes if got==ref exactly, or (for the huge sums where the NPU's
 * and the CPU's int->float rounding could differ by an ULP) the relative error
 * is < 1e-6.  Always scans every output for NaN/Inf.
 */
inline Verify verify(int M, int N, int K, const int8_t* A, const int8_t* B,
                     const float* C, uint64_t full_budget, int samples,
                     std::mt19937& rng) {
    Verify r;
    /* full NaN/Inf scan -- cheap, catches partial corruption sampling might miss */
    for (size_t i = 0; i < (size_t)M * N; ++i)
        if (!std::isfinite(C[i])) { r.bad_value = true; break; }

    auto one = [&](int m, int n) {
        float ref = ref_elem(N, K, A, B, m, n);
        float got = C[(size_t)m * N + n];
        double ad = std::fabs((double)got - (double)ref);
        double rd = ad / std::max(1.0, std::fabs((double)ref));
        r.max_abs = std::max(r.max_abs, ad);
        r.max_rel = std::max(r.max_rel, rd);
        ++r.checked;
    };

    uint64_t work = (uint64_t)M * N * K;
    if (work <= full_budget) {
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) one(m, n);
    } else {
        r.sampled = true;
        std::uniform_int_distribution<int> dm(0, M - 1), dn(0, N - 1);
        int s = std::min<uint64_t>(samples, (uint64_t)M * N);
        for (int i = 0; i < s; ++i) one(dm(rng), dn(rng));
    }
    r.ok = !r.bad_value && (r.max_abs == 0.0 || r.max_rel < 1e-6);
    return r;
}

/* ---------------------------------------------------------------- outcomes */

enum Status {
    ST_PASS,    /* ran and matched the reference                 */
    ST_WRONG,   /* ran but the result is wrong                   */
    ST_SUBMIT,  /* RK_NPU_ERR_SUBMIT (NPU job failed / timed out)*/
    ST_NOMEM,   /* RK_NPU_ERR_NOMEM                              */
    ST_PARAM,   /* RK_NPU_ERR_PARAM                              */
    ST_ALLOC,   /* a dmabuf allocation failed                    */
    ST_SKIP,    /* skipped by the resource guard (too large)     */
};

inline char status_char(Status s) {
    switch (s) {
        case ST_PASS:   return 'P';
        case ST_WRONG:  return 'X';
        case ST_SUBMIT: return 'T';   /* timeout / hw fail */
        case ST_NOMEM:  return 'M';
        case ST_PARAM:  return '!';
        case ST_ALLOC:  return 'A';
        case ST_SKIP:   return '.';
    }
    return '?';
}
inline const char* status_name(Status s) {
    switch (s) {
        case ST_PASS:   return "PASS";
        case ST_WRONG:  return "WRONG";
        case ST_SUBMIT: return "SUBMIT";
        case ST_NOMEM:  return "NOMEM";
        case ST_PARAM:  return "PARAM";
        case ST_ALLOC:  return "ALLOC";
        case ST_SKIP:   return "SKIP";
    }
    return "?";
}

/* Resource guard: shapes whose buffers/tiling exceed these are skipped. */
struct Guard {
    uint64_t max_weight_bytes = 16ull << 20;  /* packed B; pack cost ~ this   */
    uint64_t max_output_bytes = 32ull << 20;
    uint64_t max_input_bytes  = 16ull << 20;
    int      max_tasks        = 2048;         /* PC-chained tasks (M tiling)  */
};

/*
 * Caller-owned data dmabufs reused across the sweep. The plan allocates/frees
 * its own regcmd/task scratch per shape.
 */
struct Buffers {
    rk_npu_mem in{}, wt{}, out{};
    int max_tasks = 0;
    bool ok = false;
    int alloc(rk_npu_iommu_domain* domain, const rk_npu_matmul_sizes& sz) {
        int rc;
        if ((rc = rk_npu_mem_alloc(domain, sz.input_bytes,  RK_NPU_MEM_DEFAULT, &in)))  return rc;
        if ((rc = rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_DEFAULT, &wt)))  return rc;
        if ((rc = rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_DEFAULT, &out))) return rc;
        ok = true;
        return RK_NPU_OK;
    }
    /* Allocate once at the guard caps, sized to hold the largest runnable shape. */
    int alloc_caps(rk_npu_iommu_domain* domain, const Guard& g) {
        rk_npu_matmul_sizes sz{};
        sz.input_bytes  = g.max_input_bytes;
        sz.weight_bytes = g.max_weight_bytes;
        sz.output_bytes = g.max_output_bytes;
        int rc = alloc(domain, sz);
        if (rc == RK_NPU_OK) max_tasks = g.max_tasks;
        return rc;
    }
    /* Does this shape fit in the already-allocated buffers? */
    bool fits(const rk_npu_matmul_sizes& sz) const {
        return sz.input_bytes  <= in.size  && sz.weight_bytes <= wt.size &&
               sz.output_bytes <= out.size && sz.num_tasks <= max_tasks;
    }
    void free(rk_npu_ctx* ctx) {
        if (in.vaddr)  rk_npu_mem_free(ctx, &in);
        if (wt.vaddr)  rk_npu_mem_free(ctx, &wt);
        if (out.vaddr) rk_npu_mem_free(ctx, &out);
        in = wt = out = rk_npu_mem{};
        max_tasks = 0;
        ok = false;
    }
};

inline int sync_inputs_to_device(rk_npu_ctx* ctx, Buffers& b) {
    int rc = rk_npu_mem_sync(ctx, &b.in, RK_NPU_SYNC_TO_DEVICE);
    if (rc == RK_NPU_OK) rc = rk_npu_mem_sync(ctx, &b.wt, RK_NPU_SYNC_TO_DEVICE);
    return rc;
}

inline int sync_output_from_device(rk_npu_ctx* ctx, Buffers& b) {
    return rk_npu_mem_sync(ctx, &b.out, RK_NPU_SYNC_FROM_DEVICE);
}

inline Status rc_to_status(int rc) {
    switch (rc) {
        case RK_NPU_OK:         return ST_PASS;   /* caller refines to PASS/WRONG */
        case RK_NPU_ERR_SUBMIT: return ST_SUBMIT;
        case RK_NPU_ERR_NOMEM:  return ST_NOMEM;
        case RK_NPU_ERR_PARAM:  return ST_PARAM;
        default:                return ST_SUBMIT;
    }
}

/* Full result of probing one (M,N,K) shape for correctness. */
struct ShapeResult {
    Status status = ST_SKIP;
    Verify v{};
    rk_npu_matmul_sizes sz{};
};

/*
 * Pack, run and verify one shape into the caller's reusable
 * buffer set `b` (allocated once via alloc_caps).  Shapes that don't fit the
 * buffers return ST_SKIP without touching the device.  Does NOT allocate or free
 * data device memory.
 */
inline ShapeResult run_shape(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                             const rk_npu_matmul_i8_config& cfg, Buffers& b,
                             uint64_t full_budget, int samples, std::mt19937& rng) {
    ShapeResult r;
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    if (rknpu2_matmul_open::detail::query_i8(&cfg, &r.sz) != RK_NPU_OK) { r.status = ST_PARAM; return r; }
    if (!b.fits(r.sz)) { r.status = ST_SKIP; return r; }

    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    fill_i8(A, rng); fill_i8(B, rng);
    rknpu2_matmul_open::detail::pack_i8_a(&cfg, A.data(), &b.in);
    rknpu2_matmul_open::detail::pack_i8_b(&cfg, B.data(), &b.wt);
    if (sync_inputs_to_device(ctx, b) != RK_NPU_OK) { r.status = ST_SUBMIT; return r; }

    rk_npu_matmul_i8_plan* plan = rknpu2_matmul_open::detail::prepare_i8(domain, &cfg);
    if (!plan) { r.status = ST_ALLOC; return r; }
    int rc = rknpu2_matmul_open::detail::run_i8(ctx, plan, &b.in, &b.wt, &b.out);
    rknpu2_matmul_open::detail::free_i8_plan(plan);
    if (rc != RK_NPU_OK) { r.status = rc_to_status(rc); return r; }

    std::vector<float> C((size_t)M * N);
    if (sync_output_from_device(ctx, b) != RK_NPU_OK) { r.status = ST_SUBMIT; return r; }
    rknpu2_matmul_open::detail::unpack_i8_c(&cfg, &b.out, C.data());
    r.v = verify(M, N, K, A.data(), B.data(), C.data(), full_budget, samples, rng);
    r.status = r.v.ok ? ST_PASS : ST_WRONG;
    return r;
}

inline ShapeResult run_shape(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                             int M, int N, int K, Buffers& b,
                             uint64_t full_budget, int samples, std::mt19937& rng) {
    rk_npu_matmul_i8_config cfg{};
    rk_npu_matmul_i8_config_init(&cfg, M, N, K);
    return run_shape(ctx, domain, cfg, b, full_budget, samples, rng);
}

/* ------------------------------------------------------------------- timing */

using clk = std::chrono::steady_clock;
inline double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

/* Warmup then time `fn` `loops` times; returns {avg_us, min_us}. */
template <class F>
inline std::pair<double,double> timeit(F&& fn, int loops, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) fn();
    double tsum = 0, tmin = 1e30;
    for (int i = 0; i < loops; ++i) {
        auto t0 = clk::now();
        fn();
        double us = us_since(t0);
        tsum += us; tmin = std::min(tmin, us);
    }
    return {tsum / loops, tmin};
}

} /* namespace rknpu2_matmul_open::test */

#endif /* RK_NPU_TEST_COMMON_H */

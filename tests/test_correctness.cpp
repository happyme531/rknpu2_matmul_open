/*
 * test_correctness.cpp - exhaustive correctness sweep for rk_npu_matmul (int8).
 * ============================================================================
 * Runs the full cartesian product of the requested M / N / K sweeps, classifies
 * every shape against an exact int64 CPU reference, and prints a PASS/FAIL map
 * paginated by (M,K) along the N axis, plus a "largest passing N" capability
 * envelope.  Shapes that exceed the resource guard are skipped (not run); raise
 * the guard from the CLI to push the envelope further.
 *
 *   ./test_correctness [options]
 *     --dev PATH           DRM device (default /dev/dri/card1)
 *     --m LO:HI            restrict M sweep to [LO,HI]   (default full)
 *     --n LO:HI            restrict N sweep to [LO,HI]
 *     --k LO:HI            restrict K sweep to [LO,HI]
 *     --exp                use smaller exponential M/N/K lists
 *     --csv PATH           also write per-shape results as CSV
 *     --samples N          sampled-verify positions for big shapes (default 4096)
 *     --full-budget MACS   verify in full when M*N*K <= this   (default 4e7)
 *     --max-weight-mb MB   guard: skip if packed B exceeds this (default 16)
 *     --max-out-mb MB      guard: skip if output exceeds this   (default 32)
 *     --max-in-mb MB       guard: skip if input exceeds this    (default 16)
 *     --max-tasks N        guard: skip if M-tiling needs > N tasks (default 2048)
 *     --seed N             RNG seed (default 1234567)
 *     --a-native           pack/consume A as native layout (K/16,M,16)
 *     --c-native           write/unpack output C as native layout (N/4,M,4)
 *     --only-fail          only print grid rows that contain a non-PASS cell
 */
#include "test_common.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace rknpu2_matmul_open::test;

namespace {

struct Args {
    const char* dev = nullptr;
    const char* csv = nullptr;
    int   m_lo = 0, m_hi = 1 << 30;
    int   n_lo = 0, n_hi = 1 << 30;
    int   k_lo = 0, k_hi = 1 << 30;
    int   samples = 4096;
    uint64_t full_budget = 40000000ull;
    Guard guard;
    unsigned seed = 1234567;
    bool only_fail = false;
    bool exponential = false;
    bool a_native = false;
    bool c_native = false;
};

bool parse_range(const char* s, int& lo, int& hi) {
    const char* c = std::strchr(s, ':');
    if (!c) return false;
    lo = std::atoi(s); hi = std::atoi(c + 1);
    return true;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (o == "--dev")           a.dev = next();
        else if (o == "--csv")           a.csv = next();
        else if (o == "--m")             parse_range(next(), a.m_lo, a.m_hi);
        else if (o == "--n")             parse_range(next(), a.n_lo, a.n_hi);
        else if (o == "--k")             parse_range(next(), a.k_lo, a.k_hi);
        else if (o == "--samples")       a.samples = std::atoi(next());
        else if (o == "--full-budget")   a.full_budget = std::strtoull(next(), nullptr, 10);
        else if (o == "--max-weight-mb") a.guard.max_weight_bytes = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-out-mb")    a.guard.max_output_bytes = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-in-mb")     a.guard.max_input_bytes  = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-tasks")     a.guard.max_tasks = std::atoi(next());
        else if (o == "--seed")          a.seed = (unsigned)std::strtoul(next(), nullptr, 10);
        else if (o == "--a-native")      a.a_native = true;
        else if (o == "--c-native")      a.c_native = true;
        else if (o == "--only-fail")     a.only_fail = true;
        else if (o == "--exp" || o == "--exponential") a.exponential = true;
        else { fprintf(stderr, "unknown option: %s\n", o.c_str()); std::exit(2); }
    }
    return a;
}

} /* namespace */

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::vector<int> Ms = clamp_list(a.exponential ? m_exp_list() : m_list(), a.m_lo, a.m_hi);
    std::vector<int> Ns = clamp_list(a.exponential ? n_exp_list() : n_list(), a.n_lo, a.n_hi);
    std::vector<int> Ks = clamp_list(a.exponential ? k_exp_list() : k_list(), a.k_lo, a.k_hi);
    if (Ms.empty() || Ns.empty() || Ks.empty()) { fprintf(stderr, "empty sweep\n"); return 2; }
    const size_t total = Ms.size() * Ns.size() * Ks.size();

    rk_npu_ctx* ctx = rk_npu_open(a.dev);
    if (!ctx) { fprintf(stderr, "rk_npu_open failed (need access to /dev/dri/card1)\n"); return 1; }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }

    FILE* csv = nullptr;
    if (a.csv) {
        csv = std::fopen(a.csv, "w");
        if (!csv) { fprintf(stderr, "cannot open csv %s\n", a.csv); }
        else fprintf(csv, "M,N,K,status,max_abs,max_rel,checked,sampled,num_tasks\n");
    }

    printf("== rk_npu_matmul correctness sweep ==\n");
    printf("M(%zu) x N(%zu) x K(%zu) = %zu shapes%s%s%s\n",
           Ms.size(), Ns.size(), Ks.size(), total,
           a.exponential ? " [--exp]" : "",
           a.a_native ? " [A=native]" : "",
           a.c_native ? " [C=native]" : "");
    printf("guard: weight<=%lluMB out<=%lluMB in<=%lluMB tasks<=%d ; verify full<=%lluM MACs, else %d samples\n",
           (unsigned long long)(a.guard.max_weight_bytes >> 20),
           (unsigned long long)(a.guard.max_output_bytes >> 20),
           (unsigned long long)(a.guard.max_input_bytes  >> 20),
           a.guard.max_tasks, (unsigned long long)(a.full_budget / 1000000), a.samples);
    printf("legend: P=pass  X=WRONG  T=submit/timeout  M=nomem  A=alloc-fail  !=param  .=skip(guard)\n\n");

    printf("N axis (columns), left->right:\n  ");
    for (int N : Ns) printf("%d ", N);
    printf("\n\n");

    /* One reusable data buffer set sized to the guard caps. */
    Buffers bufs;
    if (bufs.alloc_caps(domain, a.guard) != RK_NPU_OK) {
        fprintf(stderr, "failed to allocate cap buffers (~%lluMB); lower --max-*-mb\n",
                (unsigned long long)((a.guard.max_input_bytes + a.guard.max_weight_bytes +
                                      a.guard.max_output_bytes) >> 20));
        bufs.free(ctx); rk_npu_iommu_domain_free(domain); rk_npu_close(ctx); return 1;
    }

    std::mt19937 rng(a.seed);
    long counts[7] = {0};
    /* envelope[mi][ki] = largest N that passed for that (M,K), or -1 */
    std::vector<std::vector<int>> envelope(Ms.size(), std::vector<int>(Ks.size(), -1));

    size_t done = 0;
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        int M = Ms[mi];
        printf("M=%d\n", M);
        for (size_t ki = 0; ki < Ks.size(); ++ki) {
            int K = Ks[ki];
            std::string row;
            bool any_fail = false;
            for (int N : Ns) {
                rk_npu_matmul_i8_config cfg{};
                rk_npu_matmul_i8_config_init(&cfg, M, N, K);
                cfg.a_layout = a.a_native ? RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16
                                          : RK_NPU_I8_A_LAYOUT_NORMAL;
                cfg.c_layout = a.c_native ? RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4
                                          : RK_NPU_I8_C_LAYOUT_NORMAL_PADDED;
                ShapeResult r = run_shape(ctx, domain, cfg, bufs,
                                          a.full_budget, a.samples, rng);
                counts[r.status]++;
                row += status_char(r.status);
                if (r.status == ST_PASS) envelope[mi][ki] = std::max(envelope[mi][ki], N);
                else if (r.status == ST_WRONG || r.status == ST_SUBMIT) any_fail = true;
                if (csv)
                    fprintf(csv, "%d,%d,%d,%s,%g,%g,%zu,%d,%d\n", M, N, K,
                            status_name(r.status), r.v.max_abs, r.v.max_rel,
                            r.v.checked, r.v.sampled ? 1 : 0, r.sz.num_tasks);
                ++done;
            }
            if (!a.only_fail || any_fail)
                printf("  K=%-5d | %s\n", K, row.c_str());
        }
        fflush(stdout);
        if (csv) fflush(csv);
    }

    /* ---- summary ---- */
    printf("\n-- summary (%zu shapes) --\n", total);
    printf("  PASS=%ld  WRONG=%ld  SUBMIT/timeout=%ld  NOMEM=%ld  ALLOC=%ld  PARAM=%ld  SKIP(guard)=%ld\n",
           counts[ST_PASS], counts[ST_WRONG], counts[ST_SUBMIT],
           counts[ST_NOMEM], counts[ST_ALLOC], counts[ST_PARAM], counts[ST_SKIP]);

    /* ---- capability envelope: largest passing N per (M,K) ---- */
    printf("\n-- largest passing N per (M,K)  ('-' = none passed, '.'=all skipped) --\n");
    printf("%6s", "M\\K");
    for (int K : Ks) printf(" %5d", K);
    printf("\n");
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        printf("%6d", Ms[mi]);
        for (size_t ki = 0; ki < Ks.size(); ++ki) {
            int e = envelope[mi][ki];
            if (e < 0) printf(" %5s", "-");
            else       printf(" %5d", e);
        }
        printf("\n");
    }

    if (csv) std::fclose(csv);
    bufs.free(ctx);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);

    long failed = counts[ST_WRONG] + counts[ST_SUBMIT] + counts[ST_NOMEM] + counts[ST_PARAM];
    printf("\n%s (%ld passed, %ld failed, %ld skipped)\n",
           failed == 0 ? "NO UNEXPECTED FAILURES" : "SEE FAILURES ABOVE",
           counts[ST_PASS], failed, counts[ST_SKIP] + counts[ST_ALLOC]);
    return 0;   /* sweep itself ran; FAIL cells are expected (no tiling yet) */
}

/*
 * test_perf.cpp - latency / throughput sweep for rk_npu_matmul (int8).
 * ===================================================================
 * For every shape it times the fixed-shape plan path, reports avg us and GOPS,
 * and verifies the result once.
 * Shapes that don't verify (or exceed the guard) are reported but not timed.
 *
 * Defaults to a COARSE grid so a perf run is quick; pass --full to sweep the
 * complete M/N/K cartesian (respecting the resource guard; only PASS shapes are
 * timed).  GOPS counts 2*M*N*K flops over the prepared-path average.
 *
 *   ./test_perf [options]
 *     --dev PATH        DRM device (default /dev/dri/card1)
 *     --full            sweep the full M/N/K lists (default: coarse subset)
 *     --exp             sweep smaller exponential M/N/K lists
 *     --m LO:HI         restrict M sweep        --n / --k likewise
 *     --loops N         timed iterations per shape (default 50)
 *     --official        also bench the vendor rknn_matmul and print our/off ratio
 *                       (needs -DRK_NPU_OFFICIAL=ON at configure; board only)
 *     --i32             benchmark both libraries in int8->int32 (the dtype the
 *                       vendor's ~1.8 TOPS int8 number is quoted on) instead of
 *                       the default int8->fp32
 *     --a-native        our raw path packs/consumes A as native (K/16,M,16)
 *     --c-native        our raw path writes/unpacks output C as native (N/4,M,4)
 *     --ac-native       vendor side uses AC_layout=NATIVE -- its real high-perf
 *                       path (channel-tiled A/C, no per-run layout conversion).
 *                       This is how the vendor reaches ~1.8 TOPS at large N; the
 *                       vendor output is then in native layout and left UNVERIFIED
 *                       (our library only does normal row-major output, so this is
 *                       NOT an apples-to-apples comparison -- it shows the native-
 *                       layout headroom we do not yet exploit)
 *     --csv PATH        write the perf table as CSV
 *     --max-weight-mb / --max-out-mb / --max-in-mb / --max-tasks   resource guard
 *     --seed N          RNG seed (default 99)
 */
#include "test_common.h"
#include "official_matmul.h"   /* vendor rknn_matmul bench (only active with RK_WITH_OFFICIAL) */

#include <cstdlib>
#include <cstring>
#include <string>

using namespace rknpu2_matmul_open::test;

#ifdef RK_WITH_OFFICIAL
constexpr bool kOfficialBuilt = true;
#else
constexpr bool kOfficialBuilt = false;
#endif

namespace {

struct Args {
    const char* dev = nullptr;
    const char* csv = nullptr;
    bool full = false;
    bool exponential = false;
    bool official = false;
    bool i32 = false;       /* benchmark BOTH libs in int8->int32 (vendor 1.8 TOPS dtype) */
    bool a_native = false;  /* our raw path uses native A input layout */
    bool c_native = false;  /* our raw path uses native output-C layout */
    bool ac_native = false; /* vendor side uses AC_layout=NATIVE (its real high-perf path) */
    int  m_lo = 0, m_hi = 1 << 30, n_lo = 0, n_hi = 1 << 30, k_lo = 0, k_hi = 1 << 30;
    int  loops = 50;
    Guard guard;
    unsigned seed = 99;
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
        else if (o == "--full")          a.full = true;
        else if (o == "--exp" || o == "--exponential") a.exponential = true;
        else if (o == "--official")      a.official = true;
        else if (o == "--i32")           a.i32 = true;
        else if (o == "--a-native")      a.a_native = true;
        else if (o == "--c-native")      a.c_native = true;
        else if (o == "--ac-native")     a.ac_native = true;
        else if (o == "--m")             parse_range(next(), a.m_lo, a.m_hi);
        else if (o == "--n")             parse_range(next(), a.n_lo, a.n_hi);
        else if (o == "--k")             parse_range(next(), a.k_lo, a.k_hi);
        else if (o == "--loops")         a.loops = std::atoi(next());
        else if (o == "--max-weight-mb") a.guard.max_weight_bytes = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-out-mb")    a.guard.max_output_bytes = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-in-mb")     a.guard.max_input_bytes  = (uint64_t)std::atoll(next()) << 20;
        else if (o == "--max-tasks")     a.guard.max_tasks = std::atoi(next());
        else if (o == "--seed")          a.seed = (unsigned)std::strtoul(next(), nullptr, 10);
        else { fprintf(stderr, "unknown option: %s\n", o.c_str()); std::exit(2); }
    }
    return a;
}

/* Coarse default grids: a representative spread across each requested range. */
std::vector<int> coarse(const std::vector<int>& full, const std::vector<int>& want) {
    std::vector<int> out;
    for (int v : want)
        if (std::find(full.begin(), full.end(), v) != full.end()) out.push_back(v);
    return out;
}

} /* namespace */

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::vector<int> Ms, Ns, Ks;
    if (a.exponential) {
        Ms = m_exp_list(); Ns = n_exp_list(); Ks = k_exp_list();
    } else if (a.full) {
        Ms = m_list(); Ns = n_list(); Ks = k_list();
    } else {
        Ms = coarse(m_list(), {1, 4, 16, 64, 256, 1024});
        Ks = coarse(k_list(), {16, 64, 256, 1024, 4096, 8192});
        Ns = coarse(n_list(), {16, 64, 256, 384});
    }
    Ms = clamp_list(Ms, a.m_lo, a.m_hi);
    Ns = clamp_list(Ns, a.n_lo, a.n_hi);
    Ks = clamp_list(Ks, a.k_lo, a.k_hi);
    if (Ms.empty() || Ns.empty() || Ks.empty()) { fprintf(stderr, "empty sweep\n"); return 2; }

    if (a.official && !kOfficialBuilt) {
        fprintf(stderr, "warning: --official requested but built without RK_NPU_OFFICIAL; "
                        "reconfigure with -DRK_NPU_OFFICIAL=ON. Ignoring.\n");
        a.official = false;
    }

    FILE* csv = nullptr;
    if (a.csv) {
        csv = std::fopen(a.csv, "w");
        if (csv) fprintf(csv, a.official ? "M,N,K,status,run_us,gops,num_tasks,off_us,off_gops,ratio\n"
                                         : "M,N,K,status,run_us,gops,num_tasks\n");
        else fprintf(stderr, "cannot open csv %s\n", a.csv);
    }

    printf("== rk_npu_matmul performance sweep ==\n");
    printf("M(%zu) x N(%zu) x K(%zu) = %zu shapes, %d timed loops each%s%s%s%s  dtype=int8->%s\n",
           Ms.size(), Ns.size(), Ks.size(), Ms.size()*Ns.size()*Ks.size(), a.loops,
           a.exponential ? " [--exp]" : (a.full ? " [--full]" : " [coarse; --full/--exp for more]"),
           a.a_native ? " [A=native]" : "",
           a.c_native ? " [C=native]" : "",
           a.official ? (a.ac_native ? " [+official, 2-phase, vendor AC=native]"
                                     : " [+official, 2-phase]") : "",
           a.i32 ? "int32" : "fp32");

    const uint64_t full_budget = 40000000ull;

    /* The vendor runtime and our raw-register path cannot be interleaved: after
     * librknnrt drives the NPU, our next direct submit reads stale state and goes
     * WRONG.  So when --official is set we run in two phases -- all of OUR shapes
     * first (device closed at the end), then all the VENDOR shapes -- and the
     * per-shape data is regenerated from a deterministic seed so both phases use
     * the same A/B for a shape. */
    auto shape_seed = [&](int M, int N, int K) -> unsigned {
        unsigned h = a.seed;
        h = h * 2654435761u + (unsigned)M;
        h = h * 2654435761u + (unsigned)N;
        h = h * 2654435761u + (unsigned)K;
        return h;
    };

    struct Rec {
        int M = 0, N = 0, K = 0, num_tasks = 0;
        bool fits = false, pass = false;
        const char* verdict = "SKIP";
        double our_us = 0, our_gops = 0;
        bool off_ran = false, off_ok = false, off_verified = false;
        const char* off_why = "-";
        double off_us = 0, off_gops = 0;
    };
    std::vector<Rec> recs;
    recs.reserve(Ms.size() * Ns.size() * Ks.size());

    /* ---- phase 1: our library (raw register path) ---- */
    rk_npu_ctx* ctx = rk_npu_open(a.dev);
    if (!ctx) { fprintf(stderr, "rk_npu_open failed (need access to /dev/dri/card1)\n"); return 1; }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }
    Buffers b;
    if (b.alloc_caps(domain, a.guard) != RK_NPU_OK) {
        fprintf(stderr, "failed to allocate cap buffers; lower --max-*-mb\n");
        b.free(ctx); rk_npu_iommu_domain_free(domain); rk_npu_close(ctx); return 1;
    }
    for (int M : Ms) for (int K : Ks) for (int N : Ns) {
        Rec r; r.M = M; r.N = N; r.K = K;
        rk_npu_matmul_i8_config cfg{};
        rk_npu_matmul_i8_config_init(&cfg, M, N, K);
        cfg.out_dtype = a.i32 ? RK_NPU_I8_OUT_INT32 : RK_NPU_I8_OUT_FP32;
        cfg.a_layout = a.a_native ? RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16
                                  : RK_NPU_I8_A_LAYOUT_NORMAL;
        cfg.c_layout = a.c_native ? RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4
                                  : RK_NPU_I8_C_LAYOUT_NORMAL_PADDED;
        rk_npu_matmul_sizes sz; rknpu2_matmul_open::detail::query_i8(&cfg, &sz);
        r.num_tasks = sz.num_tasks;
        r.fits = b.fits(sz);
        if (r.fits) {
            std::mt19937 rng(shape_seed(M, N, K));
            std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
            fill_i8(A, rng); fill_i8(B, rng);
            rknpu2_matmul_open::detail::pack_i8_a(&cfg, A.data(), &b.in);
            rknpu2_matmul_open::detail::pack_i8_b(&cfg, B.data(), &b.wt);
            sync_inputs_to_device(ctx, b);

            rk_npu_matmul_i8_plan* plan = rknpu2_matmul_open::detail::prepare_i8(domain, &cfg);
            int rc = plan ? rknpu2_matmul_open::detail::run_i8(ctx, plan, &b.in, &b.wt, &b.out) : RK_NPU_ERR_NOMEM;
            if (rc != RK_NPU_OK) { r.verdict = status_name(rc_to_status(rc)); }
            else {
                /* unpack_c does a raw 4-byte strided copy; in int32 mode those
                 * 4 bytes are an int32 sum, so reinterpret before verifying. */
                std::vector<float> C((size_t)M * N);
                sync_output_from_device(ctx, b);
                rknpu2_matmul_open::detail::unpack_i8_c(&cfg, &b.out, C.data());
                if (a.i32)
                    for (size_t i = 0; i < (size_t)M * N; ++i) {
                        int32_t raw; std::memcpy(&raw, &C[i], 4); C[i] = (float)raw;
                    }
                Verify v = verify(M, N, K, A.data(), B.data(), C.data(), full_budget, 4096, rng);
                r.pass = v.ok; r.verdict = v.ok ? "PASS" : "WRONG";
                if (v.ok) {
                    auto pp = timeit([&]{ rknpu2_matmul_open::detail::run_i8(ctx, plan, &b.in, &b.wt, &b.out); }, a.loops);
                    r.our_us = pp.first;
                    r.our_gops = 2.0 * M * N * K / (pp.first * 1e-6) / 1e9;
                }
            }
            rknpu2_matmul_open::detail::free_i8_plan(plan);
        }
        recs.push_back(r);
    }
    b.free(ctx);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);        /* release the NPU before the vendor runtime opens it */

    /* ---- phase 2: vendor rknn_matmul (only after our device handle is closed) ---- */
#ifdef RK_WITH_OFFICIAL
    if (a.official) {
        for (Rec& r : recs) {
            if (!r.fits || !r.pass) continue;
            std::mt19937 rng(shape_seed(r.M, r.N, r.K));
            std::vector<int8_t> A((size_t)r.M * r.K), B((size_t)r.K * r.N);
            fill_i8(A, rng); fill_i8(B, rng);
            OfficialResult orr = bench_official(r.M, r.N, r.K, A.data(), B.data(),
                                                full_budget, 4096, rng, a.loops, a.i32, a.ac_native);
            r.off_ran = orr.ran; r.off_ok = orr.ok; r.off_verified = orr.verified;
            r.off_us = orr.avg_us; r.off_gops = orr.gops;
            r.off_why = orr.created ? "runfail" : "crfail";
        }
    }
#endif

    /* ---- combined table + CSV ---- */
    if (a.official)
        printf("%-16s %10s %9s   %9s %9s %8s   %s\n",
               "M,K,N", "run_us", "GOPS", "off_us", "offGOPS", "our/off", "verify");
    else
        printf("%-16s %10s %9s   %s\n", "M,K,N", "run_us", "GOPS", "verify");

    for (const Rec& r : recs) {
        char shape[40]; snprintf(shape, sizeof(shape), "%d,%d,%d", r.M, r.K, r.N);
        if (!r.fits) {
            printf("%-16s %10s %9s   %s\n", shape, "-", "-", "SKIP(guard)");
            if (csv) fprintf(csv, a.official ? "%d,%d,%d,SKIP,0,0,%d,0,0,0\n" : "%d,%d,%d,SKIP,0,0,%d\n",
                             r.M, r.N, r.K, r.num_tasks);
        } else if (!r.pass) {
            printf("%-16s %10s %9s   %s\n", shape, "-", "-", r.verdict);
            if (csv) fprintf(csv, a.official ? "%d,%d,%d,%s,0,0,%d,0,0,0\n" : "%d,%d,%d,%s,0,0,%d\n",
                             r.M, r.N, r.K, r.verdict, r.num_tasks);
        } else if (a.official) {
            if (r.off_ran) {
                double ratio = r.off_gops > 0 ? r.our_gops / r.off_gops : 0.0;
                const char* tag = !r.off_verified ? " [off:native,unverified]"
                                : !r.off_ok       ? " [off!=ref]" : "";
                printf("%-16s %10.2f %9.1f   %9.2f %9.1f %7.2fx   %s%s\n",
                       shape, r.our_us, r.our_gops, r.off_us, r.off_gops, ratio, r.verdict, tag);
                if (csv) fprintf(csv, "%d,%d,%d,PASS,%.3f,%.2f,%d,%.3f,%.2f,%.3f\n",
                                 r.M, r.N, r.K, r.our_us, r.our_gops, r.num_tasks,
                                 r.off_us, r.off_gops, ratio);
            } else {
                printf("%-16s %10.2f %9.1f   %9s %9s %8s   %s\n",
                       shape, r.our_us, r.our_gops, "-", "-", r.off_why, r.verdict);
                if (csv) fprintf(csv, "%d,%d,%d,PASS,%.3f,%.2f,%d,0,0,0\n",
                                 r.M, r.N, r.K, r.our_us, r.our_gops, r.num_tasks);
            }
        } else {
            printf("%-16s %10.2f %9.1f   %s\n", shape, r.our_us, r.our_gops, r.verdict);
            if (csv) fprintf(csv, "%d,%d,%d,PASS,%.3f,%.2f,%d\n",
                             r.M, r.N, r.K, r.our_us, r.our_gops, r.num_tasks);
        }
    }

    if (csv) std::fclose(csv);
    return 0;
}

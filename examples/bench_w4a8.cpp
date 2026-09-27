/* Complete W4A8 calls, including dynamic quantization, all K waves and dequant.
 * Static weight packing/workspace construction excluded. Useful GOPS counts
 * logical W4A8 operations, not the two physical INT4 products. */
#include "rk_npu_w4a8.h"
#include "../src/rk_npu_half_bits.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using W = std::unique_ptr<rk_npu_w4a8_weights, decltype(&rk_npu_w4a8_weights_free)>;
using P = std::unique_ptr<rk_npu_w4a8_workspace, decltype(&rk_npu_w4a8_workspace_free)>;
static void check(int rc) {
    if (rc)
        throw std::runtime_error(std::string("stopped: ") + rk_npu_strerror(rc));
}
static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return (x[(x.size() - 1) / 2] + x[x.size() / 2]) / 2;
}
static std::string read(const char *p) {
    std::ifstream f(p);
    std::string s;
    f >> s;
    return s;
}
int main(int argc, char **argv) {
    try {
        rk_npu_w4a8_config base;
        rk_npu_w4a8_config_init(&base, 1, 4096, 4096);
        bool half = false, sweep = false, auto_kt = false;
        int loops = 15, warmup = 5;
        std::string csv;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto val = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value");
                return std::stoi(argv[++i]);
            };
            if (arg == "--shape") {
                base.M = val();
                base.K = val();
                base.N = val();
            } else if (arg == "--m-tile")
                base.m_tile = val();
            else if (arg == "--k-tile")
                base.k_tile = val();
            else if (arg == "--n-tile")
                base.n_tile = val();
            else if (arg == "--mask")
                base.npu_core_mask = val();
            else if (arg == "--cpu-threads")
                base.cpu_threads = val();
            else if (arg == "--loops")
                loops = val();
            else if (arg == "--warmup")
                warmup = val();
            else if (arg == "--serial")
                base.pipeline = 0;
            else if (arg == "--f16")
                half = true;
            else if (arg == "--sweep")
                sweep = true;
            else if (arg == "--auto-kt")
                auto_kt = true;
            else if (arg == "--csv" && i + 1 < argc)
                csv = argv[++i];
            else if (arg == "--help") {
                std::puts(
                    "bench_w4a8 [--shape M K N] [--f16] [--mask 1|2|4|3|7] [--cpu-threads 1..4]\n"
                    "            [--m-tile 1..64] [--k-tile 32..2048 | --auto-kt] [--n-tile "
                    "64..4096] "
                    "[--serial]\n"
                    "            [--sweep] [--warmup 5] [--loops 15] [--csv FILE]\n"
                    "Sweep K256/480, logical M16/32/64, N512/2048, serial/pipeline; fixed "
                    "resources. Auto mode uses RK_NPU_W4A8_K_TILE_MULTIPLIER.\n");
                return 0;
            } else
                throw std::runtime_error("unknown option: " + arg);
        }
        rk_npu_w4a8_memory_info info{};
        check(rk_npu_w4a8_memory_query(&base, &info));
        if (loops < 1 || warmup < 0)
            throw std::runtime_error("invalid repeats");
        std::ofstream file;
        if (!csv.empty()) {
            file.open(csv);
            if (!file)
                throw std::runtime_error("CSV open failed");
        }
        if (file.is_open())
            file << "dtype,M,K,N,mt,kt,nt,mask,pipeline,cpu_threads,scan_us,quant_pack_us,sync_us,"
                    "submit_us,reduce_us,dequant_us,total_us,useful_GOPS,sampled_exact\n";
        std::printf("W4A8 %s complete calls NPU=%s DDR=%s CPU_governor=%s warmup=%d loops=%d\n",
                    half ? "f16" : "f32", read("/sys/class/devfreq/fdab0000.npu/cur_freq").c_str(),
                    read("/sys/class/devfreq/dmc/cur_freq").c_str(),
                    read("/sys/devices/system/cpu/cpufreq/policy4/scaling_governor").c_str(),
                    warmup, loops);
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),
                                                                 rk_npu_close);
        if (!ctx)
            throw std::runtime_error("open NPU");
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(), 0), rk_npu_iommu_domain_free);
        if (!domain)
            throw std::runtime_error("domain");
        std::mt19937 rng(20260920);
        std::normal_distribution<float> normal;
        std::vector<float> a(size_t(base.M) * base.K), ws(base.N), c(size_t(base.M) * base.N);
        std::vector<uint16_t> ah(a.size()), ch(c.size());
        std::vector<int8_t> b(size_t(base.K) * base.N);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = normal(rng);
            ah[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
            if (half)
                a[i] = rknpu2_matmul_open::bits::half_to_float(ah[i]);
        }
        for (auto &v : b)
            v = int(rng() % 16) - 8;
        for (int n = 0; n < base.N; ++n)
            ws[n] = .005f + float(n % 17) * .0001f;
        std::set<int> columns;
        for (int j = 0; j < 17; ++j)
            columns.insert(int(int64_t(j) * (base.N - 1) / 16));
        std::vector<float> reference;
        reference.reserve(size_t(base.M) * columns.size());
        std::vector<int8_t> q(base.K);
        for (int m = 0; m < base.M; ++m) {
            float maximum = 0;
            for (int k = 0; k < base.K; ++k)
                maximum = std::max(maximum, std::fabs(a[size_t(m) * base.K + k]));
            const float s = maximum == 0 ? 1 : std::max(maximum / 127.f, FLT_MIN), inv = 1.f / s;
            for (int k = 0; k < base.K; ++k)
                q[k] =
                    int8_t(std::clamp(std::round(a[size_t(m) * base.K + k] * inv), -127.f, 127.f));
            for (int n : columns) {
                int64_t sum = 0;
                for (int k = 0; k < base.K; ++k)
                    sum += int(q[k]) * int(b[size_t(k) * base.N + n]);
                reference.push_back((float(sum) * s) * ws[n]);
            }
        }
        std::map<int, W> weights;
        if (auto_kt) {
            rk_npu_i4_weight_config wc{base.K, base.N, 0};
            W w(rk_npu_w4a8_weights_create(domain.get(), &wc, b.data(), ws.data()),
                rk_npu_w4a8_weights_free);
            if (!w)
                throw std::runtime_error("auto weight creation failed");
            rk_npu_w4a8_weight_info tiling{};
            check(rk_npu_w4a8_weights_query(w.get(), &tiling));
            base.k_tile = tiling.config.k_tile;
            std::printf("Ktile safe=%d multiplier=%g selected=%d bound_relaxed=%d\n",
                        tiling.safe_k_tile, tiling.multiplier, base.k_tile, tiling.bound_relaxed);
            weights.emplace(base.k_tile, std::move(w));
        }
        std::vector<rk_npu_w4a8_config> configs;
        std::set<std::tuple<int, int, int, int>> seen;
        auto add = [&](rk_npu_w4a8_config cfg) {
            cfg.m_tile = std::min(cfg.m_tile, cfg.M);
            if (seen.insert({cfg.m_tile, cfg.k_tile, cfg.n_tile, cfg.pipeline}).second)
                configs.push_back(cfg);
        };
        add(base);
        if (sweep)
            for (int kt : {256, 480, base.k_tile})
                for (int mt : {16, 32, 64})
                    for (int nt : {512, 2048})
                        for (int pipe : {0, 1}) {
                            auto cfg = base;
                            cfg.k_tile = kt;
                            cfg.m_tile = mt;
                            cfg.n_tile = nt;
                            cfg.pipeline = pipe;
                            add(cfg);
                        }
        double best = 1e300;
        rk_npu_w4a8_config winner{};
        for (const auto &cfg : configs) {
            check(rk_npu_w4a8_memory_query(&cfg, &info));
            if (!weights.count(cfg.k_tile)) {
                rk_npu_i4_weight_config wc{cfg.K, cfg.N, cfg.k_tile};
                W w(rk_npu_w4a8_weights_create(domain.get(), &wc, b.data(), ws.data()),
                    rk_npu_w4a8_weights_free);
                if (!w)
                    throw std::runtime_error("weight create");
                weights.emplace(cfg.k_tile, std::move(w));
            }
            P p(rk_npu_w4a8_workspace_create(domain.get(), &cfg), rk_npu_w4a8_workspace_free);
            if (!p)
                throw std::runtime_error("workspace create");
            auto *w = weights.at(cfg.k_tile).get();
            auto run = [&](rk_npu_w4a8_timings *t) {
                return half ? rk_npu_w4a8_run_f16(p.get(), w, ah.data(), ch.data(), t)
                            : rk_npu_w4a8_run_f32(p.get(), w, a.data(), c.data(), t);
            };
            for (int i = 0; i < warmup; ++i)
                check(run(nullptr));
            std::vector<double> scan, pack, sync, submit, reduce, dequant, total;
            for (int i = 0; i < loops; ++i) {
                rk_npu_w4a8_timings t{};
                check(run(&t));
                scan.push_back(t.activation_scan_us);
                pack.push_back(t.quant_pack_us);
                sync.push_back(t.sync_us);
                submit.push_back(t.submit_us);
                reduce.push_back(t.reduce_us);
                dequant.push_back(t.dequant_us);
                total.push_back(t.total_us);
            }
            size_t pos = 0;
            for (int m = 0; m < base.M; ++m)
                for (int n : columns) {
                    const float expected = reference[pos++];
                    if (half ? ch[size_t(m) * base.N + n] != rknpu2_matmul_open::bits::float_to_half(expected)
                             : c[size_t(m) * base.N + n] != expected)
                        throw std::runtime_error("sampled exact check failed; stop");
                }
            const double t = median(total), gops = 2.0 * cfg.M * cfg.K * cfg.N / (t * 1000);
            std::printf(
                "M=%d K=%d N=%d tile=%d/%d/%d mask=%u pipe=%d cpu=%d total=%.1fus scan=%.1f "
                "pack=%.1f sync=%.1f submit=%.1f reduce=%.1f dequant=%.1f %.2f useful GOPS exact\n",
                cfg.M, cfg.K, cfg.N, cfg.m_tile, cfg.k_tile, cfg.n_tile, cfg.npu_core_mask,
                cfg.pipeline, cfg.cpu_threads, t, median(scan), median(pack), median(sync),
                median(submit), median(reduce), median(dequant), gops);
            std::fflush(stdout);
            if (file.is_open()) {
                file << (half ? "f16" : "f32") << ',' << cfg.M << ',' << cfg.K << ',' << cfg.N
                     << ',' << cfg.m_tile << ',' << cfg.k_tile << ',' << cfg.n_tile << ','
                     << cfg.npu_core_mask << ',' << cfg.pipeline << ',' << cfg.cpu_threads << ','
                     << median(scan) << ',' << median(pack) << ',' << median(sync) << ','
                     << median(submit) << ',' << median(reduce) << ',' << median(dequant) << ','
                     << t << ',' << gops << ",1\n";
                file.flush();
            }
            if (t < best) {
                best = t;
                winner = cfg;
            }
        }
        std::printf(
            "BEST %.1fus logical M/K/N tiles=%d/%d/%d mask=%u pipeline=%d (%zu candidates)\n", best,
            winner.m_tile, winner.k_tile, winner.n_tile, winner.npu_core_mask, winner.pipeline,
            configs.size());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    }
}

#include "rk_npu_mm_tune.h"
#include "../src/rk_npu_internal.h"
#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_bfloat_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace bits = rknpu2_matmul_open::bits;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " line " + std::to_string(__LINE__)); } while (0)
using Work = std::unique_ptr<rk_npu_mm_tuned_workspace, decltype(&rk_npu_mm_tuned_workspace_free)>;
using Weight = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;

void cpu_tests() {
    CHECK(!unsetenv("RK_NPU_MM_K_TILING"));
    for(const auto& setting: {std::pair<const char*,int>{"",1},{"off",0},{"0",0},
                             {"fp16",1},{"on",1},{"1",1},{"fp32",-1},{"2",-1}}) {
        if(*setting.first)CHECK(!setenv("RK_NPU_MM_K_TILING",setting.first,1));
        else CHECK(!unsetenv("RK_NPU_MM_K_TILING"));
        rk_npu_mm_autotune_config policy;
        rk_npu_mm_autotune_config_init(&policy,3,32,64);
        CHECK(policy.allow_k_tiling==setting.second);
        rk_npu_mm_strategy fallback{};
        CHECK(rk_npu_mm_strategy_default(&policy,&fallback)==
              (setting.second<0?RK_NPU_ERR_PARAM:RK_NPU_OK));
        policy.dtype=RK_NPU_MM_BF16; // policy check only; no BF16 arithmetic
        CHECK(!rk_npu_mm_strategy_default(&policy,&fallback));
    }
    CHECK(!unsetenv("RK_NPU_MM_K_TILING"));
    rk_npu_mm_autotune_config c;
    rk_npu_mm_autotune_config_init(&c, 129, 160, 96);
    rk_npu_mm_strategy s{};
    CHECK(rk_npu_mm_strategy_default(&c, &s) == 0);
    CHECK(s.M == 129 && s.m_tile == 128 && s.options.split_k == 0);
    CHECK(s.options.a_layout == RK_NPU_MM_LAYOUT_NATIVE && s.options.c_layout == RK_NPU_MM_LAYOUT_NATIVE);
    c.dtype = RK_NPU_MM_BF16;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == 0 && s.dtype == RK_NPU_MM_BF16);
    c.dtype = RK_NPU_MM_F32;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == RK_NPU_ERR_PARAM);
    c.dtype = RK_NPU_MM_F16;
    c.with_bias = 1;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == 0);
    c.host_type = RK_NPU_MM_HOST_F32;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == 0);
    c.M = 0;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == RK_NPU_ERR_PARAM);
    c.M = 129; c.allowed_npu_core_mask = 5;
    CHECK(rk_npu_mm_strategy_default(&c, &s) == RK_NPU_ERR_PARAM);
    c.allowed_npu_core_mask = 7;
    CHECK(setenv("RK_NPU_MM_SPLIT_K", "invalid", 1) == 0);
    CHECK(rk_npu_mm_strategy_default(&c, &s) == 0 && s.options.split_k == 0);
    CHECK(unsetenv("RK_NPU_MM_SPLIT_K") == 0);
    rk_npu_ctx fake{};
    fake.driver_version = 900;
    CHECK(rk_npu_mm_autotune_cache_load(&fake, &c, "/nonexistent/mm-tune-test", &s, nullptr) == RK_NPU_ERR_CACHE_MISS);
    std::puts("MM tune CPU PASS");
}

struct Device {
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain = ctx ? rk_npu_iommu_domain_create(ctx, 12) : nullptr;
    Device() { CHECK(ctx && domain); }
    ~Device() { rk_npu_iommu_domain_free(domain); rk_npu_close(ctx); }
};
struct Data {
    rk_npu_mm_autotune_config c;
    std::vector<uint16_t> a, b, out;
    std::vector<float> af, bias, cf;
    explicit Data(rk_npu_mm_autotune_config config) : c(config), a(size_t(c.M) * c.K), b(size_t(c.K) * c.N),
        out(size_t(c.M) * c.N), af(a.size()), bias(c.N), cf(out.size()) {
        uint32_t rng = 0x31415926;
        auto random = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        for (size_t j = 0; j < a.size(); ++j) {
            af[j] = (int(random() % 127) - 63) / 256.f;
            a[j] = encode(af[j]);
        }
        for (auto& v : b) v = encode((int(random() % 127) - 63) / 512.f);
        for (auto& v : bias) v = (int(random() % 63) - 31) / 2048.f;
    }
    uint16_t encode(float x) const { return c.dtype == RK_NPU_MM_BF16 ? bits::float_to_bfloat(x) : bits::float_to_half(x); }
    float decode(uint16_t x) const { return c.dtype == RK_NPU_MM_BF16 ? bits::bfloat_to_float(x) : bits::half_to_float(x); }
    int run(rk_npu_mm_tuned_workspace* w, rk_npu_mm_packed_b* b) {
        return rk_npu_mm_tuned_run(w, c.host_type == RK_NPU_MM_HOST_F32 ? static_cast<void*>(af.data()) : a.data(),
            a.size() * (c.host_type == RK_NPU_MM_HOST_F32 ? 4 : 2), b, c.with_bias ? bias.data() : nullptr,
            c.host_type == RK_NPU_MM_HOST_F32 ? static_cast<void*>(cf.data()) : out.data(),
            out.size() * (c.host_type == RK_NPU_MM_HOST_F32 ? 4 : 2));
    }
    void verify(bool full) {
        const size_t count = full ? out.size() : std::min(out.size(), size_t(128));
        double max_error = 0;
        for (size_t j = 0; j < count; ++j) {
            const size_t index = full ? j : (j * 7919 + 37) % out.size();
            const int m = index / c.N, n = index % c.N;
            double ref = 0;
            for (int k = 0; k < c.K; ++k) ref += double(decode(a[size_t(m) * c.K + k])) * decode(b[size_t(k) * c.N + n]);
            if (c.with_bias) ref += bias[n];
            const float value = c.host_type == RK_NPU_MM_HOST_F32 ? cf[index] : decode(out[index]);
            const double err = std::abs(value - ref);
            max_error = std::max(max_error, err);
            CHECK(std::isfinite(value));
            CHECK(err <= (c.dtype == RK_NPU_MM_BF16 ? 0.015 + 0.02 * std::abs(ref) : 0.003 + 0.005 * std::abs(ref)));
        }
        std::printf("oracle M=%d K=%d N=%d dtype=%d host=%d checks=%zu max_abs=%.8g PASS\n", c.M, c.K, c.N, int(c.dtype), int(c.host_type), count, max_error);
    }
};
Work workspace(Device& d, const rk_npu_mm_strategy& s) {
    rk_npu_mm_tuned_workspace* raw = nullptr;
    CHECK(rk_npu_mm_tuned_workspace_create(d.domain, &s, &raw) == 0);
    return Work(raw, rk_npu_mm_tuned_workspace_free);
}
Weight weight(Device& d, const rk_npu_mm_strategy& s, const Data& data) {
    rk_npu_mm_packed_b* raw = nullptr;
    CHECK(rk_npu_mm_tuned_packed_b_create(d.domain, &s, data.b.data(), data.b.size() * 2, &raw) == 0);
    return Weight(raw, rk_npu_mm_packed_b_free);
}
void board_test(Device& d, int m, int k, int n, rk_npu_mm_dtype type,
                rk_npu_mm_host_type host, uint32_t mask, const std::string& cache) {
    rk_npu_mm_autotune_config c;
    rk_npu_mm_autotune_config_init(&c, m, n, k);
    c.dtype = type; c.host_type = host; c.with_bias = host == RK_NPU_MM_HOST_F32;
    c.allowed_npu_core_mask = mask; c.loops = 1; c.repeats = 2;
    rk_npu_mm_strategy base{}, fast{}, stable{}, loaded{};
    CHECK(rk_npu_mm_strategy_default(&c, &base) == 0);
    const auto affinity = [&]() { cpu_set_t set; CPU_ZERO(&set); CHECK(!pthread_getaffinity_np(pthread_self(), sizeof(set), &set)); return set; }();
    const int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
    int hit = -1;
    CHECK(rk_npu_mm_autotune_cached(d.domain, &c, cache.c_str(), 1, &fast, &stable, &hit) == 0 && hit == 0);
    cpu_set_t after; CPU_ZERO(&after);
    CHECK(!pthread_getaffinity_np(pthread_self(), sizeof(after), &after));
    CHECK(CPU_EQUAL(&affinity, &after) && omp_get_max_threads() == threads && omp_get_dynamic() == dynamic);
    CHECK(fast.options.allowed_npu_core_mask == mask && fast.options.split_k == 0);
    CHECK(rk_npu_mm_autotune_cache_load(d.ctx, &c, cache.c_str(), &loaded, nullptr) == 0);
    CHECK(loaded.total_us == fast.total_us && loaded.m_tile == fast.m_tile);
    CHECK(rk_npu_mm_autotune_cached(d.domain, &c, cache.c_str(), 0, &loaded, nullptr, &hit) == 0 && hit == 1);
    auto stale = c; stale.host_type = host == RK_NPU_MM_HOST_F32 ? RK_NPU_MM_HOST_STORAGE : RK_NPU_MM_HOST_F32; stale.with_bias = 0;
    CHECK(rk_npu_mm_autotune_cache_load(d.ctx, &stale, cache.c_str(), &loaded, nullptr) == RK_NPU_ERR_CACHE_MISS);
    stale = c; stale.M += 1;
    CHECK(rk_npu_mm_autotune_cache_load(d.ctx, &stale, cache.c_str(), &loaded, nullptr) == RK_NPU_ERR_CACHE_MISS);
    Data data(c);
    auto b = weight(d, fast.k_tile ? fast : base, data); // unsplit B is reusable across M/layout/N tiles
    auto w = workspace(d, fast);
    CHECK(data.run(w.get(), b.get()) == 0);
    data.verify(true);
    auto invalid = fast; invalid.options.split_k = 3;
    rk_npu_mm_tuned_workspace* rejected = nullptr;
    CHECK(rk_npu_mm_tuned_workspace_create(d.domain, &invalid, &rejected) == RK_NPU_ERR_PARAM);
    CHECK(!rejected);
    // Model weight loading is independent of logical M.
    // A selected K partition must already be resident before ordinary runs.
    auto weight_config = c; weight_config.M = 1;
    using Resident = std::unique_ptr<rk_npu_mm_tuned_weights, decltype(&rk_npu_mm_tuned_weights_free)>;
    rk_npu_mm_tuned_weights* raw_resident = nullptr;
    CHECK(!rk_npu_mm_tuned_weights_create(d.domain, &weight_config,
        data.b.data(), data.b.size()*2, cache.c_str(), &raw_resident));
    Resident resident(raw_resident, rk_npu_mm_tuned_weights_free);
    CHECK(!rk_npu_mm_tuned_weights_match(resident.get(), &fast));
    CHECK(!rk_npu_mm_tuned_weights_match(resident.get(), &base));
    CHECK(!rk_npu_mm_tuned_weights_create(d.domain, &weight_config,
        data.b.data(), data.b.size()*2, nullptr, &raw_resident));
    Resident fallback(raw_resident, rk_npu_mm_tuned_weights_free);
    CHECK(!rk_npu_mm_tuned_weights_match(fallback.get(), &base));
    CHECK(rk_npu_mm_tuned_weights_match(fallback.get(), &fast) ==
          (fast.k_tile ? RK_NPU_ERR_CACHE_MISS : RK_NPU_OK));
    std::fill(data.b.begin(), data.b.end(), 0); // host source is no longer needed
    const bool f32 = host == RK_NPU_MM_HOST_F32;
    const auto result_f32 = data.cf;
    const auto result_storage = data.out;
    CHECK(!rk_npu_mm_tuned_run_weights(w.get(),
        f32 ? static_cast<void*>(data.af.data()) : data.a.data(), data.a.size()*(f32 ? 4 : 2),
        resident.get(), c.with_bias ? data.bias.data() : nullptr,
        f32 ? static_cast<void*>(data.cf.data()) : data.out.data(), data.out.size()*(f32 ? 4 : 2)));
    CHECK(f32 ? data.cf == result_f32 : data.out == result_storage);
    std::printf("resident_weights M=%d K=%d N=%d Ktile=%d host_source_released=PASS output_bitwise_equal=PASS\n",
        m, k, n, fast.k_tile);
}
void k_tile_tests(Device& d) {
    for(auto shape:{std::vector<int>{17,1537,35},{129,4096,160}})
      for(auto host:{RK_NPU_MM_HOST_STORAGE,RK_NPU_MM_HOST_F32}) for(int biased:{0,1}) {
        if(host==RK_NPU_MM_HOST_F32 && !biased)continue;
        rk_npu_mm_autotune_config c;rk_npu_mm_autotune_config_init(&c,shape[0],shape[2],shape[1]);
        c.host_type=host;c.with_bias=biased;
        Data data(c);rk_npu_mm_strategy base;CHECK(!rk_npu_mm_strategy_default(&c,&base));
        auto full=weight(d,base,data);
        for(int tile:{256,512,1024,2048})if(tile<c.K)
          for(int cap:{0,86})for(int panel:{0,1})for(int mask:{1,7}) {
            auto s=base;s.m_tile=(c.M+15)/16*16;s.k_tile=tile;s.task_m_tile=cap;
            s.options.n_tile=32;s.options.allowed_npu_core_mask=mask;
            s.options.a_layout=panel?(tile<=1024?RK_NPU_MM_LAYOUT_PANEL8:RK_NPU_MM_LAYOUT_PANEL16):RK_NPU_MM_LAYOUT_NORMAL;
            s.options.c_layout=panel?s.options.a_layout:RK_NPU_MM_LAYOUT_NATIVE;
            auto w=workspace(d,s);auto b=weight(d,s,data);
            CHECK(data.run(w.get(),full.get())==RK_NPU_ERR_PARAM);
            CHECK(!data.run(w.get(),b.get()));data.verify(c.M<20);
            std::printf("K TILE PASS M=%d K=%d N=%d host=%d bias=%d tile=%d taskM=%d panel=%d mask=%d\n",c.M,c.K,c.N,int(host),biased,tile,cap,panel,mask);
          }
      }
    std::puts("FP16 K tiles / device reduce PASS");
}
void panel_fp16(Device& d) {
    for (auto shape : {std::vector<int>{32,96,35}, {32,97,35}, {32,1024,160}, {64,1536,160}, {32,4096,160}, {129,1024,160}}) {
        for (auto host : {RK_NPU_MM_HOST_STORAGE, RK_NPU_MM_HOST_F32}) {
            rk_npu_mm_autotune_config c;
            rk_npu_mm_autotune_config_init(&c, shape[0], shape[2], shape[1]);
            c.host_type = host; c.with_bias = host == RK_NPU_MM_HOST_F32;
            Data data(c);
            rk_npu_mm_strategy base;
            CHECK(rk_npu_mm_strategy_default(&c, &base) == 0);
            auto b = weight(d, base, data);
            for (auto layout : {std::vector<int>{0,0}, {0,1}, {1,0}, {1,1}, {2,1}, {3,1}, {0,2}, {2,2}, {3,3}}) {
                for (int cap : {0,16,32}) for (int nt : {0,32}) {
                    auto s = base;
                    s.m_tile = (c.M + 15) / 16 * 16;
                    s.task_m_tile = cap; s.options.n_tile = nt;
                    s.fused_io = host == RK_NPU_MM_HOST_F32;
                    s.options.a_layout = static_cast<rk_npu_mm_layout>(layout[0]);
                    s.options.c_layout = static_cast<rk_npu_mm_layout>(layout[1]);
                    rk_npu_mm_desc desc;
                    rk_npu_mm_desc_init(&desc, s.m_tile, c.N, c.K, 1);
                    rk_npu_mm_plan* probe = nullptr;
                    const int rc = rk_npu_mm_plan_create_tiled(&desc, &s.options, cap, &probe);
                    rk_npu_mm_plan_free(probe);
                    if (rc == RK_NPU_ERR_PARAM) continue;
                    CHECK(rc == 0);
                    auto w = workspace(d, s);
                    std::printf("panel probe M=%d K=%d N=%d host=%d A=%d C=%d cap=%d nt=%d\n",
                        c.M, c.K, c.N, int(host), layout[0], layout[1], cap, nt);
                    std::fflush(stdout);
                    CHECK(data.run(w.get(), b.get()) == 0);
                    data.verify(true);
                }
            }
        }
    }
    std::puts("FP16 panel/task/fused IO PASS");
}
int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 2 ? argv[1] : "";
        if (mode == "--help" || argc == 1) {
            std::puts("Usage: test_mm_tune --cpu|--fp16|--bf16|--panel-fp16|--k-tiles");
            return 0;
        }
        CHECK(argc == 2 && (mode == "--cpu" || mode == "--fp16" || mode == "--bf16" ||
                           mode == "--panel-fp16" || mode == "--k-tiles"));
        if (mode == "--cpu") { cpu_tests(); return 0; }
        Device d;
        if (mode == "--k-tiles") { k_tile_tests(d); return 0; }
        if (mode == "--panel-fp16") { panel_fp16(d); return 0; }
        const auto type = mode == "--bf16" ? RK_NPU_MM_BF16 : RK_NPU_MM_F16;
        char temp[] = "/tmp/rknpu-mm-tune-XXXXXX";
        CHECK(mkdtemp(temp));
        const std::string cache(temp);
        for (uint32_t mask : {1u, 2u, 4u, 3u, 7u})
            board_test(d, 3, 64, 35, type, RK_NPU_MM_HOST_STORAGE, mask, cache);
        board_test(d, 129, 96, 160, type, RK_NPU_MM_HOST_F32, 7, cache);
        if (type == RK_NPU_MM_F16)
            board_test(d, 17, 1537, 35, type, RK_NPU_MM_HOST_STORAGE, 7, cache);
        // Truncated cache files must be misses, leaving outputs untouched.
        for (const auto& entry : std::filesystem::directory_iterator(cache)) {
            std::ofstream truncated(entry.path()); truncated << "RKMM1\n";
        }
        rk_npu_mm_autotune_config c;
        rk_npu_mm_autotune_config_init(&c, 129, 160, 96); c.dtype = type; c.host_type = RK_NPU_MM_HOST_F32; c.with_bias = 1; c.loops = 1; c.repeats = 2;
        rk_npu_mm_strategy untouched{}; untouched.M = -7;
        CHECK(rk_npu_mm_autotune_cache_load(d.ctx, &c, cache.c_str(), &untouched, nullptr) == RK_NPU_ERR_CACHE_MISS && untouched.M == -7);
        std::filesystem::remove_all(cache);
        std::puts("MM tune board PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}

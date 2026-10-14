#include "../src/rk_npu_w4a8_tune_internal.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <cstdlib>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
using namespace rknpu2_matmul_open::w4_tune;
static void candidates_test() {
    Profile p;
    p.K = 11008;
    p.N = 4096;
    p.safe = 992;
    p.limit = 2048;
    p.multiplier = 2.1;
    p.safe_tiles = ~uint64_t(0);
    Resources r{7, 0xf0, 0xf0, 4};
    for (int M : {1, 2, 4, 8, 16, 17, 32, 33, 64, 65, 127, 128, 129, 255, 256, 257}) {
        rk_npu_w4a8_autotune_config c;
        rk_npu_w4a8_autotune_config_init(&c, M, p.N, p.K);
        const auto old = candidates(c, p, r), all = candidates(c, p, r, true);
        using Key = std::tuple<int, int, int, int, int, int>;
        std::set<Key> keys;
        bool panel = false, u128 = false, u256 = false;
        for (const auto &x : all) {
            const auto &cfg = x.config;
            const bool is_panel = x.execution.input_layout == RK_NPU_W4A8_INPUT_PANEL8;
            CHECK(keys.insert({int(x.execution.input_layout), cfg.m_tile, cfg.k_tile, cfg.n_tile,
                               cfg.cpu_threads, cfg.pipeline})
                      .second);
            CHECK(x.execution.reduce_backend == RK_NPU_W4A8_REDUCE_CPU);
            if (M <= small_cpu_batch_limit)
                CHECK((cfg.pipeline >= 0 && cfg.pipeline <= 2) &&
                      (cfg.cpu_threads == 1 || cfg.cpu_threads == r.threads));
            else
                CHECK(cfg.pipeline == 1 && cfg.cpu_threads == r.threads);
            rk_npu_w4a8_memory_info_ex mem{};
            CHECK(!rk_npu_w4a8_memory_query_ex(&cfg, &x.execution, &mem));
            if (is_panel) {
                panel = true;
                u128 |= cfg.m_tile == 128;
                u256 |= cfg.m_tile == 256;
                CHECK(cfg.m_tile % 4 == 0 && cfg.m_tile <= 256);
                CHECK(!(cfg.m_tile == 256 && cfg.k_tile == 2048));
            }
        }
        for (const auto &x : old)
            CHECK(keys.count({0, x.config.m_tile, x.config.k_tile, x.config.n_tile,
                              x.config.cpu_threads, x.config.pipeline}));
        if (M < 32 || (M <= 64 && M % 4))
            CHECK(!panel && all.size() == old.size());
        else
            CHECK(panel);
        if (M >= 128)
            CHECK(u128);
        if (M >= 256)
            CHECK(u256);
    }
}
static void small_batch_policy_cache(const std::string& path,int M) {
    rk_npu_ctx ctx{}; ctx.fd=-1; ctx.driver_version=0x908;
    rk_npu_w4a8_autotune_config c;
    rk_npu_w4a8_autotune_config_init(&c,M,193,1057);
    c.required_k_tile=480; c.k_tile_multiplier=1;
    c.allowed_cpu_core_mask=rknpu2_matmul_open::tune::first_cpu_mask(
        rknpu2_matmul_open::tune::preferred_cpus(rknpu2_matmul_open::tune::process_cpu_mask()),4);
    std::vector<int8_t> b(size_t(c.K)*c.N,1);
    std::vector<float> scale(c.N,.03125f);
    Profile p; CHECK(!profile(c,b.data(),scale.data(),p));
    const auto r=resources(c);
    // Check the full policy product and deduplication for one to four CPUs.
    for(int maximum=1;maximum<=4;++maximum) {
        Resources synthetic{7,((uint64_t(1)<<maximum)-1)<<4,0xf0,maximum};
        std::set<std::pair<int,int>> policies;
        for(const auto& x:candidates(c,p,synthetic,true))
            policies.emplace(x.config.cpu_threads,x.config.pipeline);
        std::set<std::pair<int,int>> expected{{1,0},{1,1},{1,2},{maximum,0},{maximum,1},{maximum,2}};
        CHECK(policies==expected);
    }
    for(bool extended:{false,true}) {
        const auto choices=candidates(c,p,r,extended);
        CHECK(!choices.empty());
        CHECK(choices.front().config.cpu_threads==r.threads && choices.front().config.pipeline==1);
        for(const auto& x:choices) {
            Strategy s{};s.base.config=x.config;s.execution=x.execution;
            s.base.activation_type=c.activation_type;s.base.cpu_core_mask=r.cpu;
            s.base.weights=p.info(x.config.k_tile);s.base.weight_fingerprint=p.fingerprint;
            s.base.tuning_revision=revision;s.base.total_us=1.25;s.base.robust_us=1.5;
            CHECK(valid_strategy(s,c,p,r,extended));
            Strategy f{},stable{};int hit=0;
            if(extended) {
                CHECK(!rk_npu_w4a8_autotune_cache_save_ex(&ctx,&c,b.data(),scale.data(),path.c_str(),&s,&s));
                CHECK(!rk_npu_w4a8_autotune_cached_ex(&ctx,&c,b.data(),scale.data(),path.c_str(),0,&f,&stable,&hit));
            } else {
                CHECK(!rk_npu_w4a8_autotune_cache_save(&ctx,&c,b.data(),scale.data(),path.c_str(),&s.base,&s.base));
                CHECK(!rk_npu_w4a8_autotune_cached(&ctx,&c,b.data(),scale.data(),path.c_str(),0,&f.base,&stable.base,&hit));
            }
            CHECK(hit==1 && f.base.config.cpu_threads==x.config.cpu_threads &&
                  f.base.config.pipeline==x.config.pipeline && f.base.cpu_core_mask==r.cpu);
            for(int mutation=0;mutation<3;++mutation) {
                auto bad=s;
                if(mutation==0) bad.base.config.cpu_threads=r.threads==2?3:2;
                if(mutation==1) bad.base.config.pipeline=3;
                if(mutation==2) bad.base.tuning_revision=revision-1;
                CHECK(!valid_strategy(bad,c,p,r,extended));
                CHECK(!save(path,key(&ctx,c,p,r,extended),bad,bad));
                CHECK(load(path,key(&ctx,c,p,r,extended),c,p,r,f,stable,extended)==RK_NPU_ERR_CACHE_MISS);
            }
        }
    }
}
static void cache_test(const std::string &path) {
    rk_npu_ctx ctx{};
    ctx.fd = -1;
    ctx.driver_version = 0x908;
    rk_npu_w4a8_autotune_config c;
    rk_npu_w4a8_autotune_config_init(&c, 128, 193, 2048);
    c.k_tile_multiplier = 1;
    c.required_k_tile = 2048;
    c.allowed_cpu_core_mask =
        rknpu2_matmul_open::tune::first_cpu_mask(rknpu2_matmul_open::tune::preferred_cpus(rknpu2_matmul_open::tune::process_cpu_mask()), 1);
    std::vector<int8_t> b(size_t(c.K) * c.N, 1);
    std::vector<float> scale(c.N, .125f);
    Profile p;
    CHECK(!profile(c, b.data(), scale.data(), p));
    const auto r = resources(c);
    Strategy s{};
    bool found = false;
    for (const auto &x : candidates(c, p, r, true))
        if (x.config.m_tile == 128) {
            s.base.config = x.config;
            s.execution = x.execution;
            found = true;
            break;
        }
    CHECK(found);
    auto &base = s.base;
    base.activation_type = c.activation_type;
    base.cpu_core_mask = r.cpu;
    base.weights = p.info(base.config.k_tile);
    base.weight_fingerprint = p.fingerprint;
    base.tuning_revision = revision;
    base.total_us = 1.25;
    base.robust_us = 1.5;
    CHECK(valid_strategy(s, c, p, r, true));
    CHECK(!valid_strategy(s, c, p, r, false));
    CHECK(key(&ctx, c, p, r) != key(&ctx, c, p, r, true));
    CHECK(!rk_npu_w4a8_autotune_cache_save_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), &s,
                                              &s));
    Strategy f{}, stable{};
    int hit = 0;
    // fd=-1 proves a cache hit performs no device allocation/submit.
    CHECK(!rk_npu_w4a8_autotune_cached_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), 0, &f,
                                          &stable, &hit) &&
          hit == 1);
    CHECK(f.execution.input_layout == RK_NPU_W4A8_INPUT_PANEL8 && f.base.config.m_tile == 128 &&
          f.base.total_us == 1.25);
    rk_npu_w4a8_strategy legacy{}, legacy2{};
    CHECK(rk_npu_w4a8_autotune_cache_load(&ctx, &c, b.data(), scale.data(), path.c_str(), &legacy,
                                          &legacy2) == RK_NPU_ERR_CACHE_MISS);
    for (int change = 0; change < 4; ++change) {
        auto bad = s;
        if (change == 0)
            bad.execution.input_layout = static_cast<rk_npu_w4a8_input_layout>(7);
        if (change == 1)
            bad.execution.reduce_backend = RK_NPU_W4A8_REDUCE_NPU;
        if (change == 2)
            bad.execution.struct_size = 0;
        if (change == 3)
            bad.base.tuning_revision = revision - 1;
        CHECK(rk_npu_w4a8_autotune_cache_save_ex(&ctx, &c, b.data(), scale.data(), path.c_str(),
                                                 &bad, &bad) == RK_NPU_ERR_PARAM);
        CHECK(!save(path, key(&ctx, c, p, r, true), bad, bad));
        CHECK(rk_npu_w4a8_autotune_cache_load_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), &f,
                                                 &stable) == RK_NPU_ERR_CACHE_MISS);
    }
    CHECK(!save(path, key(&ctx, c, p, r, true), s, s));
    {
        std::ifstream in(path);
        std::string magic, rest((std::istreambuf_iterator<char>(in)), {});
        rest.replace(0, rest.find('\n'), "rk_npu_w4a8_tuning_cache_v1");
        std::ofstream out(path);
        out << rest;
    }
    CHECK(rk_npu_w4a8_autotune_cache_load_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), &f,
                                             &stable) == RK_NPU_ERR_CACHE_MISS);
    // Native winners also retain extended-search identity.
    s.execution.input_layout = RK_NPU_W4A8_INPUT_NATIVE;
    s.base.config.m_tile = 64;
    CHECK(!rk_npu_w4a8_autotune_cache_save_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), &s,
                                              &s));
    CHECK(!rk_npu_w4a8_autotune_cache_load_ex(&ctx, &c, b.data(), scale.data(), path.c_str(), &f,
                                              &stable));
    CHECK(rk_npu_w4a8_autotune_cache_load(&ctx, &c, b.data(), scale.data(), path.c_str(), &legacy,
                                          &legacy2) == RK_NPU_ERR_CACHE_MISS);
}
static void environment_test(const std::string& path) {
    rk_npu_ctx ctx{};ctx.fd=-1;ctx.driver_version=0x908;
    rk_npu_w4a8_autotune_config c;rk_npu_w4a8_autotune_config_init(&c,4,193,1057);
    c.required_k_tile=480;c.k_tile_multiplier=1;
    std::vector<int8_t> b(size_t(c.K)*c.N,1);std::vector<float> scale(c.N,.01f);
    Profile p;CHECK(!profile(c,b.data(),scale.data(),p));
    CHECK(!setenv("RK_NPU_W4A8_CHAIN_MAX_M","8",1));const auto r=resources(c);
    const auto choices=candidates(c,p,r);CHECK(!choices.empty());
    Strategy s{};s.base.config=choices.front().config;s.execution=choices.front().execution;
    s.base.activation_type=c.activation_type;s.base.cpu_core_mask=r.cpu;s.base.weights=p.info(480);
    s.base.weight_fingerprint=p.fingerprint;s.base.tuning_revision=revision;
    CHECK(!rk_npu_w4a8_autotune_cache_save_ex(&ctx,&c,b.data(),scale.data(),path.c_str(),&s,&s));
    for(const char* value:{"0","2","4","16","64"}) {
        CHECK(!setenv("RK_NPU_W4A8_CHAIN_MAX_M",value,1));const auto next=resources(c);
        CHECK(key(&ctx,c,p,r)!=key(&ctx,c,p,next));Strategy a{},z{};
        CHECK(rk_npu_w4a8_autotune_cache_load_ex(&ctx,&c,b.data(),scale.data(),path.c_str(),&a,&z)==RK_NPU_ERR_CACHE_MISS);
        bool chain=false;for(const auto& candidate:candidates(c,p,next))chain|=candidate.config.pipeline==RK_NPU_W4A8_CHAIN;
        CHECK(chain==(c.M<=next.chain_max_m));
    }
    for(const char* value:{"","-1","65","eight","99999999999999999"}) {
        CHECK(!setenv("RK_NPU_W4A8_CHAIN_MAX_M",value,1));CHECK(chain_max_m()==-1);
        Strategy a{},z{};CHECK(rk_npu_w4a8_autotune_cache_load_ex(&ctx,&c,b.data(),scale.data(),path.c_str(),&a,&z)==RK_NPU_ERR_PARAM);
    }
    CHECK(!setenv("RK_NPU_W4A8_CHAIN_MAX_M","8",1));
}
int main() {
    char dir[] = "/tmp/rk_w4_layout_cpu_XXXXXX";
    if (!mkdtemp(dir))
        return 1;
    try {
        CHECK(!setenv("RK_NPU_W4A8_CHAIN_MAX_M","8",1));
        candidates_test();
        cache_test(std::string(dir) + "/cache");
        for(int M:{1,2,4,8})small_batch_policy_cache(std::string(dir)+"/small-"+std::to_string(M),M);
        environment_test(std::string(dir)+"/environment");
        std::filesystem::remove_all(dir);
        std::puts("PASS native/panel8 candidates, ubatch/CBUF/tails, strategy/cache layout and "
                  "revision, small-batch teams/chain, environment cache isolation, zero-device cache hit");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        std::filesystem::remove_all(dir);
        return 1;
    }
}

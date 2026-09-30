#include "rk_npu_quant_matmul.h"
#include "rk_npu_autotune_common.h"
#include "rk_npu_autotune_cache.h"

#include "rk_npu_cpu_kernels.h"
#include "rk_npu_internal.h"
#include "rk_npu_kn_plan.h"
#include "rk_npu_dcomp.h"
#include "rk_npu_moe_regs.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <locale>
#include <memory>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using rknpu2_matmul_open::tune::process_cpu_mask;
using rknpu2_matmul_open::tune::preferred_cpus;
using rknpu2_matmul_open::tune::first_cpu_mask;
using rknpu2_matmul_open::tune::fixed_npu_mask;
using rknpu2_matmul_open::tune::pin_openmp_team;
using rknpu2_matmul_open::tune::double_bits;
using rknpu2_matmul_open::tune::bits_double;

using Clock = std::chrono::steady_clock;
constexpr int N_TILE_SAFE_MAX = 3072;
constexpr uint64_t AUTOTUNE_SEED = 0x726b6e70755f6938ull;
constexpr uint32_t AUTOTUNE_CACHE_FORMAT_VERSION = 2;
/* TODO(tuning-cache-revision): bump whenever candidate generation, validation,
 * production pipeline behavior, or CPU kernels change strategy selection. */
constexpr uint32_t AUTOTUNE_CACHE_TUNING_REVISION = 5; // fused chain input packing and scale cache
constexpr const char* AUTOTUNE_CACHE_MAGIC = "rk_npu_matmul_tuning_cache_v2";

double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

int popcount64(uint64_t x) { return __builtin_popcountll(x); }

bool finite_positive(float x) {
    return std::isfinite(x) && x > 0.0f;
}

bool valid_scales(const float* scales, int count) {
    if (!scales || count <= 0) return false;
    for (int i = 0; i < count; ++i)
        if (!finite_positive(scales[i])) return false;
    return true;
}

bool valid_config(const rk_npu_matmul_autotune_config* cfg) {
    return cfg && cfg->M > 0 && cfg->N > 0 && cfg->K > 0 &&
           (cfg->allowed_npu_core_mask & ~7u) == 0 &&
           cfg->warmup >= 0 && cfg->loops > 0 && cfg->repeats > 0 &&
           cfg->timeout_ms > 0 && cfg->required_k_tile >= 0 &&
           cfg->required_k_tile <= cfg->K;
}

bool valid_family_config(const rk_npu_matmul_family_autotune_config* cfg) {
    if (!cfg || cfg->K <= 0 || cfg->N <= 0 || cfg->m_count <= 0 ||
        !cfg->m_values || (cfg->allowed_npu_core_mask & ~7u) != 0 ||
        cfg->warmup < 0 || cfg->loops <= 0 || cfg->repeats <= 0 ||
        cfg->timeout_ms == 0)
        return false;
    std::set<int> seen;
    for (int i = 0; i < cfg->m_count; ++i) {
        if (cfg->m_values[i] <= 0 || !seen.insert(cfg->m_values[i]).second)
            return false;
        if (cfg->frequencies &&
            (!std::isfinite(cfg->frequencies[i]) || cfg->frequencies[i] <= 0))
            return false;
    }
    return true;
}

bool supported_npu_mask(uint32_t mask) {
    return mask == 1u || mask == 2u || mask == 4u || mask == 3u || mask == 7u;
}

enum class HostDType { I8, I32, F16, F32 };
enum class QuantKind { None, Dynamic, Static };

struct OpTraits {
    HostDType input = HostDType::I8;
    HostDType output = HostDType::I32;
    QuantKind quant = QuantKind::None;
    bool valid = false;
};

OpTraits op_traits(rk_npu_matmul_op_kind kind) {
    switch (kind) {
    case RK_NPU_MATMUL_I8I8I32:
        return {HostDType::I8, HostDType::I32, QuantKind::None, true};
    case RK_NPU_MATMUL_F16I8F16_DYNAMIC:
        return {HostDType::F16, HostDType::F16, QuantKind::Dynamic, true};
    case RK_NPU_MATMUL_F16I8F16_STATIC:
        return {HostDType::F16, HostDType::F16, QuantKind::Static, true};
    case RK_NPU_MATMUL_F32I8F32_DYNAMIC:
        return {HostDType::F32, HostDType::F32, QuantKind::Dynamic, true};
    case RK_NPU_MATMUL_F32I8F32_STATIC:
        return {HostDType::F32, HostDType::F32, QuantKind::Static, true};
    }
    return {};
}

bool supported_op_kind(rk_npu_matmul_op_kind kind) {
    return op_traits(kind).valid;
}

// Read at workspace/tuner creation. Existing workspaces never change backend
// when an environment variable changes. No public strategy ABI change.
uint32_t execution_mode(rk_npu_matmul_op_kind kind) {
    auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    };
    const bool dequant = kind != RK_NPU_MATMUL_I8I8I32 &&
        enabled("RK_NPU_W8A8_NPU_DEQUANT");
    return (enabled("RK_NPU_I8_NPU_REDUCE") || dequant ? 1u : 0u) |
           (dequant ? 2u : 0u);
}

bool valid_weight_config(const rk_npu_matmul_weight_config* config) {
    return config && config->K > 0 && config->N > 0 &&
           config->k_tile > 0 && config->k_tile <= config->K;
}


bool valid_strategy(const rk_npu_matmul_strategy* s,
                    rk_npu_matmul_op_kind expected) {
    if (!s || !supported_op_kind(expected) || s->op_kind != expected ||
        s->M <= 0 || s->N <= 0 || s->K <= 0 ||
        s->k_tile <= 0 || s->k_tile > s->K || s->n_tile <= 0 ||
        s->n_tile > N_TILE_SAFE_MAX ||
        (s->a_layout < RK_NPU_MATMUL_A_LAYOUT_NORMAL ||
         s->a_layout > RK_NPU_MATMUL_A_LAYOUT_PANEL16) ||
        (s->c_layout < RK_NPU_MATMUL_C_LAYOUT_NATIVE ||
         s->c_layout > RK_NPU_MATMUL_C_LAYOUT_PANEL16) ||
        !supported_npu_mask(s->npu_core_mask) || s->cpu_core_mask == 0 ||
        s->cpu_threads <= 0 || s->cpu_threads != popcount64(s->cpu_core_mask))
        return false;
    if (s->cpu_core_mask & ~process_cpu_mask()) return false;
    if (s->a_layout >= RK_NPU_MATMUL_A_LAYOUT_PANEL8 ||
        s->c_layout != RK_NPU_MATMUL_C_LAYOUT_NATIVE) {
        rknpu2_matmul_open::detail::I8KnPlanConfig pcfg{};
        pcfg.M = s->M; pcfg.N = s->N; pcfg.K = s->K;
        pcfg.k_tile = s->k_tile; pcfg.n_tile = s->n_tile;
        pcfg.npu_core_mask = s->npu_core_mask;
        pcfg.a_layout = (rk_npu_matmul_i8_a_layout)s->a_layout;
        pcfg.c_layout = (rk_npu_matmul_i8_c_layout)((int)s->c_layout + 1);
        rknpu2_matmul_open::detail::I8KnMemoryInfo info{};
        if (rknpu2_matmul_open::detail::query_i8_kn_memory(pcfg, &info) != RK_NPU_OK) return false;
    }
    const int align_n = std::max(MIN_CHANNEL_TILE, align_up(s->N, MIN_CHANNEL_TILE));
    const int nt = std::min(align_n, align_up(s->n_tile, MIN_CHANNEL_TILE));
    return ceil_div(align_n, nt) >= __builtin_popcount(s->npu_core_mask);
}

struct AutotuneCacheKey {
    uint32_t format_version = AUTOTUNE_CACHE_FORMAT_VERSION;
    uint32_t tuning_revision = AUTOTUNE_CACHE_TUNING_REVISION;
    uint32_t driver_version = 0;
    int op_kind = 0;
    int M = 0;
    int N = 0;
    int K = 0;
    uint32_t allowed_npu_core_mask = 0;
    uint64_t allowed_cpu_core_mask = 0;
    int warmup = 0;
    int loops = 0;
    int repeats = 0;
    uint32_t timeout_ms = 0;
    int required_k_tile = 0;
};

bool make_cache_key(rk_npu_ctx* ctx,
                    const rk_npu_matmul_autotune_config* cfg,
                    rk_npu_matmul_op_kind kind,
                    AutotuneCacheKey& key) {
    if (!ctx || !valid_config(cfg) || !supported_op_kind(kind)) return false;
    key.driver_version = ctx->driver_version;
    // Serialize and hash the backend along with the tuning revision, including
    // explicit cache paths. CPU/reduce/dequant never reuse each other's scores.
    key.tuning_revision |= execution_mode(kind) << 16;
    key.op_kind = (int)kind;
    key.M = cfg->M;
    key.N = cfg->N;
    key.K = cfg->K;
    key.allowed_npu_core_mask = cfg->allowed_npu_core_mask
                              ? cfg->allowed_npu_core_mask : 7u;
    key.allowed_cpu_core_mask = (cfg->allowed_cpu_core_mask
                              ? cfg->allowed_cpu_core_mask : process_cpu_mask())
                              & process_cpu_mask();
    key.warmup = cfg->warmup;
    key.loops = cfg->loops;
    key.repeats = cfg->repeats;
    key.timeout_ms = cfg->timeout_ms;
    key.required_k_tile = cfg->required_k_tile;
    return key.allowed_cpu_core_mask != 0 && key.allowed_npu_core_mask != 0;
}

bool same_cache_key(const AutotuneCacheKey& a, const AutotuneCacheKey& b) {
    return a.format_version == b.format_version &&
           a.tuning_revision == b.tuning_revision &&
           a.driver_version == b.driver_version &&
           a.op_kind == b.op_kind && a.M == b.M && a.N == b.N && a.K == b.K &&
           a.allowed_npu_core_mask == b.allowed_npu_core_mask &&
           a.allowed_cpu_core_mask == b.allowed_cpu_core_mask &&
           a.warmup == b.warmup && a.loops == b.loops &&
           a.repeats == b.repeats && a.timeout_ms == b.timeout_ms &&
           a.required_k_tile == b.required_k_tile;
}

uint64_t cache_key_hash(const AutotuneCacheKey& key) {
    rknpu2_matmul_open::tune::Hash hash;
    auto mix = [&](uint64_t value) { hash.mix(value); };
    mix(key.format_version);
    mix(key.tuning_revision);
    mix(key.driver_version);
    mix((uint32_t)key.op_kind);
    mix((uint32_t)key.M);
    mix((uint32_t)key.N);
    mix((uint32_t)key.K);
    mix(key.allowed_npu_core_mask);
    mix(key.allowed_cpu_core_mask);
    mix((uint32_t)key.warmup);
    mix((uint32_t)key.loops);
    mix((uint32_t)key.repeats);
    mix(key.timeout_ms);
    mix((uint32_t)key.required_k_tile);
    return hash.value;
}

std::string default_cache_path(const AutotuneCacheKey& key) {
    return rknpu2_matmul_open::tune::cache_path(cache_key_hash(key), nullptr, "rk_npu_matmul");
}
std::string cache_path_in_directory(const char* directory, const AutotuneCacheKey& key) {
    if (!directory || !*directory) return {};
    return rknpu2_matmul_open::tune::cache_path(cache_key_hash(key), directory, "rk_npu_matmul");
}

void write_strategy(std::ostream& out, const char* label,
                    const rk_npu_matmul_strategy& s) {
    out << label << ' ' << (int)s.op_kind << ' ' << s.M << ' ' << s.N << ' '
        << s.K << ' ' << s.k_tile << ' ' << s.n_tile << ' ' << (int)s.a_layout
        << ' ' << (int)s.c_layout
        << ' ' << s.wave_count << ' ' << s.n_groups << ' ' << s.npu_core_mask
        << ' ' << s.cpu_core_mask << ' ' << s.cpu_threads << ' '
        << double_bits(s.input_us) << ' ' << double_bits(s.npu_us) << ' '
        << double_bits(s.sync_us) << ' ' << double_bits(s.output_us) << ' '
        << double_bits(s.total_us) << ' ' << double_bits(s.jitter_pct) << ' '
        << double_bits(s.robust_us) << '\n';
}

bool read_strategy(std::istream& in, const char* expected_label,
                   rk_npu_matmul_strategy& s) {
    std::string label;
    int op = 0, layout = 0, c_layout = 0;
    uint64_t input = 0, npu = 0, sync = 0, output = 0;
    uint64_t total = 0, jitter = 0, robust = 0;
    if (!(in >> label >> op >> s.M >> s.N >> s.K >> s.k_tile >> s.n_tile
          >> layout >> c_layout >> s.wave_count >> s.n_groups >> s.npu_core_mask
          >> s.cpu_core_mask >> s.cpu_threads >> input >> npu >> sync >> output
          >> total >> jitter >> robust) || label != expected_label)
        return false;
    s.op_kind = (rk_npu_matmul_op_kind)op;
    s.a_layout = (rk_npu_matmul_a_layout)layout;
    s.c_layout = (rk_npu_matmul_c_layout)c_layout;
    s.input_us = bits_double(input);
    s.npu_us = bits_double(npu);
    s.sync_us = bits_double(sync);
    s.output_us = bits_double(output);
    s.total_us = bits_double(total);
    s.jitter_pct = bits_double(jitter);
    s.robust_us = bits_double(robust);
    return true;
}

bool strategy_matches_cache(const rk_npu_matmul_strategy& s,
                            const AutotuneCacheKey& key) {
    if (!valid_strategy(&s, (rk_npu_matmul_op_kind)key.op_kind) ||
        s.M != key.M || s.N != key.N || s.K != key.K ||
        (s.npu_core_mask & ~key.allowed_npu_core_mask) != 0 ||
        (s.cpu_core_mask & ~key.allowed_cpu_core_mask) != 0 ||
        s.n_tile != align_up(s.n_tile, MIN_CHANNEL_TILE) ||
        (key.required_k_tile > 0 && s.k_tile != key.required_k_tile) ||
        s.wave_count != ceil_div(s.K, s.k_tile))
        return false;
    const int align_n = std::max(MIN_CHANNEL_TILE, align_up(s.N, MIN_CHANNEL_TILE));
    if (s.n_groups != ceil_div(align_n, s.n_tile)) return false;
    for (double value : {s.input_us, s.npu_us, s.sync_us, s.output_us,
                         s.total_us, s.jitter_pct, s.robust_us})
        if (!std::isfinite(value) || value < 0) return false;
    return true;
}

int load_cache_file(const std::string& path, const AutotuneCacheKey& expected,
                    rk_npu_matmul_strategy& fastest, rk_npu_matmul_strategy& stable) {
    return rknpu2_matmul_open::tune::load_cache(path, [&](std::istream& in) {
        std::string magic, label;
        AutotuneCacheKey stored{};
        if (!(in >> magic >> label >> stored.format_version >> stored.tuning_revision
              >> stored.driver_version >> stored.op_kind >> stored.M >> stored.N >> stored.K
              >> stored.allowed_npu_core_mask >> stored.allowed_cpu_core_mask >> stored.warmup
              >> stored.loops >> stored.repeats >> stored.timeout_ms >> stored.required_k_tile))
            return false;
        return magic == AUTOTUNE_CACHE_MAGIC && label == "key" && same_cache_key(stored, expected)
            && read_strategy(in, "fastest", fastest) && read_strategy(in, "stable", stable)
            && strategy_matches_cache(fastest, expected) && strategy_matches_cache(stable, expected);
    });
}
int save_cache_file(const std::string& path, const AutotuneCacheKey& key,
                    const rk_npu_matmul_strategy& fastest, const rk_npu_matmul_strategy& stable) {
    return rknpu2_matmul_open::tune::save_cache(path, [&](std::ostream& out) {
        out << AUTOTUNE_CACHE_MAGIC << '\n'
            << "key " << key.format_version << ' ' << key.tuning_revision << ' '
            << key.driver_version << ' ' << key.op_kind << ' ' << key.M << ' '
            << key.N << ' ' << key.K << ' ' << key.allowed_npu_core_mask << ' '
            << key.allowed_cpu_core_mask << ' ' << key.warmup << ' ' << key.loops
            << ' ' << key.repeats << ' ' << key.timeout_ms << ' '
            << key.required_k_tile << '\n';
        write_strategy(out, "fastest", fastest);
        write_strategy(out, "stable", stable);
    });
}

int choose_n_tile(int N, int target_groups) {
    const int align_n = std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE));
    int tile = align_up(ceil_div(align_n, target_groups), MIN_CHANNEL_TILE);
    if (ceil_div(align_n, tile) < target_groups) {
        tile = std::max(MIN_CHANNEL_TILE,
                        (align_n / target_groups / MIN_CHANNEL_TILE) * MIN_CHANNEL_TILE);
    }
    return tile;
}

int k_knee(int M) {
    const int rows = std::max(1, std::min(M, 128));
    return std::max(MIN_CHANNEL_TILE,
        (8 * CBUF_BANK_SIZE / rows / MIN_CHANNEL_TILE) * MIN_CHANNEL_TILE);
}

std::vector<int> k_candidates(int M, int K) {
    const int knee = k_knee(M);
    constexpr int numerators[] = {2, 3, 4, 5, 6, 8};
    std::set<int> values;
    for (int num : numerators) {
        int k = (knee * num + 2) / 4;
        k = std::max(MIN_CHANNEL_TILE,
                     (k / MIN_CHANNEL_TILE) * MIN_CHANNEL_TILE);
        if (k <= K) values.insert(k);
    }
    if (K <= 2 * knee) values.insert(K);
    if (values.empty()) values.insert(K);
    return std::vector<int>(values.begin(), values.end());
}

rknpu2_matmul_open::detail::I8KnPlanConfig plan_config(const rk_npu_matmul_strategy& s,
                                  uint32_t timeout_ms) {
    rknpu2_matmul_open::detail::I8KnPlanConfig cfg{};
    cfg.M = s.M;
    cfg.N = s.N;
    cfg.K = s.K;
    cfg.k_tile = s.k_tile;
    cfg.n_tile = s.n_tile;
    cfg.npu_core_mask = s.npu_core_mask;
    cfg.timeout_ms = timeout_ms;
    cfg.a_layout = (rk_npu_matmul_i8_a_layout)s.a_layout;
    cfg.c_layout = (rk_npu_matmul_i8_c_layout)((int)s.c_layout + 1);
    const uint32_t mode = execution_mode(s.op_kind);
    cfg.npu_reduce = (mode & 1u) != 0;
    cfg.npu_dequant = (mode & 2u) != 0;
    return cfg;
}

rknpu2_matmul_open::detail::I8KnWeightConfig weight_config(int K, int N, int k_tile) {
    rknpu2_matmul_open::detail::I8KnWeightConfig config{};
    config.K = K;
    config.N = N;
    config.k_tile = k_tile;
    return config;
}

rknpu2_matmul_open::detail::I8KnWeightConfig weight_config(const rk_npu_matmul_strategy& s) {
    return weight_config(s.K, s.N, s.k_tile);
}

rknpu2_matmul_open::detail::I8KnWeightConfig weight_config(
    const rk_npu_matmul_weight_config& config) {
    return weight_config(config.K, config.N, config.k_tile);
}

void fill_topology(rk_npu_matmul_strategy& s,
                   rk_npu_matmul_op_kind kind,
                   int M, int N, int K, int kt, int nt,
                   uint32_t npu_mask,
                   rk_npu_matmul_a_layout a_layout) {
    std::memset(&s, 0, sizeof(s));
    s.op_kind = kind;
    s.M = M;
    s.N = N;
    s.K = K;
    s.k_tile = kt;
    s.n_tile = nt;
    s.a_layout = a_layout;
    s.wave_count = ceil_div(K, kt);
    s.n_groups = ceil_div(std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE)), nt);
    s.npu_core_mask = npu_mask;
}

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if (man & 0x1000u) ++half;
    return (uint16_t)half;
}

struct DeterministicRng {
    uint64_t state = AUTOTUNE_SEED;
    uint32_t next_u32() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (uint32_t)(state >> 16);
    }
    int8_t next_i8() { return (int8_t)((int)(next_u32() % 255u) - 127); }
    float next_finite() {
        return ((int)(next_u32() % 6001u) - 3000) * 0.001f;
    }
};

struct TuneData {
    std::vector<int8_t> A_i8;
    std::vector<uint16_t> A_f16;
    std::vector<float> A_f32;
    std::vector<int8_t> B_i8;
    std::vector<float> a_scale;
    std::vector<float> w_scale;
};

TuneData make_tune_data(rk_npu_matmul_op_kind kind, int M, int N, int K) {
    TuneData d;
    DeterministicRng rng;
    d.B_i8.resize((size_t)K * N);
    for (int8_t& x : d.B_i8) x = rng.next_i8();
    const OpTraits traits = op_traits(kind);
    if (traits.quant == QuantKind::None) {
        d.A_i8.resize((size_t)M * K);
        for (int8_t& x : d.A_i8) x = rng.next_i8();
        return d;
    }
    if (traits.input == HostDType::F16) {
        d.A_f16.resize((size_t)M * K);
        for (uint16_t& x : d.A_f16) x = float_to_half(rng.next_finite());
    } else {
        d.A_f32.resize((size_t)M * K);
        for (float& x : d.A_f32) x = rng.next_finite();
    }
    d.w_scale.resize((size_t)N);
    for (int n = 0; n < N; ++n)
        d.w_scale[(size_t)n] = 0.002f + (float)(rng.next_u32() % 1801u) * 0.00001f;
    d.a_scale.resize((size_t)M);
    for (int m = 0; m < M; ++m)
        d.a_scale[(size_t)m] = 0.005f + (float)(rng.next_u32() % 301u) * 0.00005f;
    return d;
}

int prepare_for_kind(rknpu2_matmul_open::detail::I8KnPlan& plan, rk_npu_matmul_op_kind kind,
                     const TuneData& data) {
    const OpTraits traits = op_traits(kind);
    if (traits.quant != QuantKind::Dynamic) return RK_NPU_OK;
    if (traits.input == HostDType::F16)
        return plan.prepare_f16_dynamic(data.A_f16.data());
    if (traits.input == HostDType::F32)
        return plan.prepare_f32_dynamic(data.A_f32.data());
    return RK_NPU_OK;
}

int pack_wave_for_kind(rknpu2_matmul_open::detail::I8KnPlan& plan,
                       rk_npu_matmul_op_kind kind,
                       const TuneData& data, int wave) {
    const OpTraits traits = op_traits(kind);
    if (traits.quant == QuantKind::None)
        return plan.pack_wave_i8(wave, data.A_i8.data());
    if (traits.input == HostDType::F16)
        return traits.quant == QuantKind::Dynamic
            ? plan.pack_wave_f16_dynamic(wave, data.A_f16.data())
            : plan.pack_wave_f16_static(
                  wave, data.A_f16.data(), data.a_scale.data());
    if (traits.input == HostDType::F32)
        return traits.quant == QuantKind::Dynamic
            ? plan.pack_wave_f32_dynamic(wave, data.A_f32.data())
            : plan.pack_wave_f32_static(
                  wave, data.A_f32.data(), data.a_scale.data());
    return RK_NPU_ERR_PARAM;
}

int run_collect_exact(rknpu2_matmul_open::detail::I8KnPlan& plan,
                      const rknpu2_matmul_open::detail::I8KnWeights& weights,
                      rk_npu_matmul_op_kind kind, const TuneData& data,
                      std::vector<int32_t>& result) {
    int rc = prepare_for_kind(plan, kind, data);
    result.assign((size_t)plan.config().M * plan.config().N, 0);
    std::vector<int32_t> partial;
    for (int w = 0; rc == RK_NPU_OK && w < plan.wave_count(); ++w) {
        rc = pack_wave_for_kind(plan, kind, data, w);
        if (rc != RK_NPU_OK) break;
        rc = plan.run_wave(w, weights);
        if (rc == RK_NPU_OK) rc = plan.begin_wave_output_cpu_read(w);
        if (rc == RK_NPU_OK) rc = plan.unpack_partial_i32(w, partial);
        if (rc == RK_NPU_OK) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(result.data());
            const uint32_t* src = reinterpret_cast<const uint32_t*>(partial.data());
            for (size_t i = 0; i < result.size(); ++i) dst[i] += src[i];
        }
        const int end_rc = plan.end_wave_output_cpu_access(w);
        if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    }
    return rc;
}

bool equal_i32(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(int32_t)) == 0;
}

using RunMetrics = rknpu2_matmul_open::tune::Sample;

struct RunJob {
    const rknpu2_matmul_open::detail::I8KnWeights* weights = nullptr;
    const float* w_scale = nullptr;
    const int8_t* A_i8 = nullptr;
    const uint16_t* A_f16 = nullptr;
    const float* A_f32 = nullptr;
    const float* a_scale = nullptr;
    int32_t* C_i32 = nullptr;
    uint16_t* C_f16 = nullptr;
    float* C_f32 = nullptr;
};

/* Keep the blocking RKNPU submit on a persistent thread.  The execution
 * worker can then reduce a completed partial while the next wave is running. */
class NpuSubmitter {
public:
    explicit NpuSubmitter(rknpu2_matmul_open::detail::I8KnPlan* plan) : plan_(plan) {}
    ~NpuSubmitter() { stop(); }

    int start() {
        try {
            worker_ = std::thread(&NpuSubmitter::worker_main, this);
        } catch (const std::system_error&) {
            return RK_NPU_ERR_NOMEM;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return initialized_; });
        return RK_NPU_OK;
    }

    int submit(const rknpu2_matmul_open::detail::I8KnWeights* weights, int wave) {
        if (!weights) return RK_NPU_ERR_PARAM;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || pending_ || executing_ || !done_)
                return RK_NPU_ERR_BUSY;
            weights_ = weights;
            wave_ = wave;
            done_ = false;
            pending_ = true;
        }
        cv_.notify_all();
        return RK_NPU_OK;
    }

    int wait(double* npu_us) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return done_ || stopping_; });
        if (!done_) return RK_NPU_ERR_PARAM;
        if (npu_us) *npu_us = elapsed_us_;
        return rc_;
    }

private:
    void stop() {
        if (!worker_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    void worker_main() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            initialized_ = true;
        }
        cv_.notify_all();
        for (;;) {
            const rknpu2_matmul_open::detail::I8KnWeights* weights = nullptr;
            int wave = -1;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || pending_; });
                if (stopping_ && !pending_) break;
                weights = weights_;
                wave = wave_;
                pending_ = false;
                executing_ = true;
            }
            const auto t0 = Clock::now();
            const int rc = plan_->run_wave(wave, *weights);
            const double elapsed = us_since(t0);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                rc_ = rc;
                elapsed_us_ = elapsed;
                executing_ = false;
                done_ = true;
            }
            cv_.notify_all();
        }
    }

    rknpu2_matmul_open::detail::I8KnPlan* plan_ = nullptr;
    const rknpu2_matmul_open::detail::I8KnWeights* weights_ = nullptr;
    int wave_ = -1;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool initialized_ = false;
    bool stopping_ = false;
    bool pending_ = false;
    bool executing_ = false;
    bool done_ = true;
    int rc_ = RK_NPU_OK;
    double elapsed_us_ = 0;
};

class SingleRunner {
public:
    SingleRunner(rknpu2_matmul_open::detail::I8KnPlan* plan, rk_npu_matmul_op_kind kind,
                 uint64_t cpu_mask, int cpu_threads)
        : plan_(plan), kind_(kind), cpu_mask_(cpu_mask),
          cpu_threads_(cpu_threads), npu_(plan) {}

    ~SingleRunner() { stop(); }

    int start() {
        int rc = plan_->config().npu_reduce ? RK_NPU_OK : npu_.start();
        if (rc != RK_NPU_OK) return rc;
        try {
            worker_ = std::thread(&SingleRunner::worker_main, this);
        } catch (const std::system_error&) {
            return RK_NPU_ERR_NOMEM;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return initialized_; });
        return RK_NPU_OK;
    }

    int run(const RunJob& job, RunMetrics* metrics) {
        const auto total0 = Clock::now();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stopping_) return RK_NPU_ERR_PARAM;
            if (pending_ || executing_) return RK_NPU_ERR_BUSY;
            job_ = job;
            pending_ = true;
            done_ = false;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return done_; });
        RunMetrics out = metrics_;
        out.total_us = us_since(total0);
        const int rc = rc_;
        lock.unlock();
        if (metrics) *metrics = out;
        return rc;
    }

private:
    void stop() {
        if (!worker_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    void worker_main() {
        pin_openmp_team(cpu_mask_, cpu_threads_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            initialized_ = true;
        }
        cv_.notify_all();
        for (;;) {
            RunJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || pending_; });
                if (stopping_ && !pending_) break;
                job = job_;
                pending_ = false;
                executing_ = true;
            }
            RunMetrics metrics{};
            const int rc = execute(job, metrics);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_ = metrics;
                rc_ = rc;
                executing_ = false;
                done_ = true;
            }
            cv_.notify_all();
        }
    }

    int execute(const RunJob& job, RunMetrics& metrics) {
        const OpTraits traits = op_traits(kind_);
        if (!job.weights) return RK_NPU_ERR_PARAM;
        if (traits.quant == QuantKind::None) {
            if (!job.A_i8 || !job.C_i32) return RK_NPU_ERR_PARAM;
        } else {
            const bool valid_io = traits.input == HostDType::F16
                ? job.A_f16 && job.C_f16
                : job.A_f32 && job.C_f32;
            if (!valid_io || !job.w_scale)
                return RK_NPU_ERR_PARAM;
            if (traits.quant == QuantKind::Static && !job.a_scale)
                return RK_NPU_ERR_PARAM;
        }
        int rc = RK_NPU_OK;

        const int waves = plan_->wave_count();
        if (waves <= 0) return RK_NPU_ERR_PARAM;
        auto pack = [&](int wave) {
            const auto input0 = Clock::now();
            int one_rc = RK_NPU_ERR_PARAM;
            if (traits.quant == QuantKind::None)
                one_rc = plan_->pack_wave_i8(wave, job.A_i8);
            else if (traits.input == HostDType::F16)
                one_rc = traits.quant == QuantKind::Dynamic
                    ? (wave == 0
                        ? plan_->prepare_and_pack_wave0_f16_dynamic(job.A_f16)
                        : plan_->pack_wave_f16_dynamic(wave, job.A_f16))
                    : plan_->pack_wave_f16_static(
                          wave, job.A_f16, job.a_scale);
            else if (traits.input == HostDType::F32)
                one_rc = traits.quant == QuantKind::Dynamic
                    ? (wave == 0
                        ? plan_->prepare_and_pack_wave0_f32_dynamic(job.A_f32)
                        : plan_->pack_wave_f32_dynamic(wave, job.A_f32))
                    : plan_->pack_wave_f32_static(
                          wave, job.A_f32, job.a_scale);
            metrics.input_us += us_since(input0);
            return one_rc;
        };
        auto submit = [&](int wave) {
            return npu_.submit(job.weights, wave);
        };
        auto wait = [&]() {
            double one_us = 0;
            const int one_rc = npu_.wait(&one_us);
            metrics.npu_us += one_us;
            return one_rc;
        };
        auto write_output = [&](int partial_count,
                                const int32_t* const* partials) {
            if (traits.quant == QuantKind::None) {
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_to_rowmajor(
                    plan_->config().M, plan_->config().N, partial_count,
                    partials, job.C_i32, plan_->c_panel_width());
                return;
            }
            const float* scale = traits.quant == QuantKind::Dynamic
                               ? plan_->a_scale() : job.a_scale;
            if (traits.output == HostDType::F16) {
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
                    plan_->config().M, plan_->config().N, partial_count,
                    partials, scale, job.w_scale, nullptr,
                    rknpu2_matmul_open::cpu::ActivationOp::None, job.C_f16, plan_->c_panel_width());
            } else {
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(
                    plan_->config().M, plan_->config().N, partial_count,
                    partials, scale, job.w_scale, nullptr,
                    rknpu2_matmul_open::cpu::ActivationOp::None, job.C_f32, plan_->c_panel_width());
            }
        };

        if (plan_->config().npu_reduce) {
            if (job.weights->compressed()) return RK_NPU_ERR_PARAM;
            if (traits.quant == QuantKind::Dynamic) {
                const auto input0 = Clock::now();
                rc = traits.input == HostDType::F16
                    ? plan_->pack_chain_dynamic(job.A_f16)
                    : plan_->pack_chain_dynamic(job.A_f32);
                metrics.input_us += us_since(input0);
            } else {
                for (int w = 0; w < waves && rc == RK_NPU_OK; ++w) rc = pack(w);
            }
            if (rc != RK_NPU_OK) return rc;
            const float* scale = traits.quant == QuantKind::Dynamic
                               ? plan_->a_scale() : job.a_scale;
            const auto npu0 = Clock::now();
            rc = plan_->run_chain(*job.weights, scale, job.w_scale,
                                   plan_->config().npu_dequant);
            metrics.npu_us += us_since(npu0);
            if (rc != RK_NPU_OK) return rc;
            if (plan_->config().npu_dequant) {
                const auto output0 = Clock::now();
                rc = plan_->copy_dequant(traits.output == HostDType::F16
                    ? static_cast<void*>(job.C_f16) : static_cast<void*>(job.C_f32),
                    traits.output == HostDType::F16);
                metrics.output_us += us_since(output0); // includes final cache sync
                return rc;
            }
            const auto sync0 = Clock::now();
            rc = plan_->begin_wave_output_cpu_read(0);
            metrics.sync_us += us_since(sync0);
            if (rc != RK_NPU_OK) return rc;
            const int32_t* final[] = {plan_->partial_i32(0)};
            const auto output0 = Clock::now();
            write_output(1, final);
            metrics.output_us += us_since(output0);
            return plan_->end_wave_output_cpu_access(0);
        }

        rc = pack(0);
        if (rc == RK_NPU_OK) rc = submit(0);
        if (rc == RK_NPU_OK && waves > 1) rc = pack(1);
        const int first_wait_rc = wait();
        if (rc == RK_NPU_OK) rc = first_wait_rc;
        if (rc != RK_NPU_OK) return rc;
        auto sync0 = Clock::now();
        rc = waves > 2 ? plan_->begin_wave_output_cpu_readwrite(0)
                       : plan_->begin_wave_output_cpu_read(0);
        metrics.sync_us += us_since(sync0);
        if (rc != RK_NPU_OK) return rc;
        int32_t* accum = const_cast<int32_t*>(plan_->partial_i32(0));
        int pending_partial = -1;

        if (waves > 1) {
            rc = submit(1);
            if (rc == RK_NPU_OK && waves > 2) rc = pack(2);
            const int second_wait_rc = wait();
            if (rc == RK_NPU_OK) rc = second_wait_rc;
            if (rc == RK_NPU_OK) {
                sync0 = Clock::now();
                rc = plan_->begin_wave_output_cpu_read(1);
                metrics.sync_us += us_since(sync0);
                pending_partial = 1;
            }
        }
        if (rc == RK_NPU_OK && waves == 2) {
            const int32_t* pair[] = {
                plan_->partial_i32(0), plan_->partial_i32(1)};
            const auto output0 = Clock::now();
            write_output(2, pair);
            metrics.output_us += us_since(output0);
            sync0 = Clock::now();
            const int end1 = plan_->end_wave_output_cpu_access(1);
            const int end0 = plan_->end_wave_output_cpu_access(0);
            metrics.sync_us += us_since(sync0);
            return end1 != RK_NPU_OK ? end1 : end0;
        }
        for (int wave = 2; rc == RK_NPU_OK && wave < waves; ++wave) {
            rc = submit(wave);
            if (rc != RK_NPU_OK) break;
            if (wave + 1 < waves) {
                rc = pack(wave + 1);
                if (rc != RK_NPU_OK) {
                    (void)wait();
                    break;
                }
            }
            const auto reduce0 = Clock::now();
            rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(
                plan_->config().M, plan_->config().N,
                plan_->partial_i32(pending_partial), accum, plan_->c_panel_width());
            metrics.output_us += us_since(reduce0);
            sync0 = Clock::now();
            const int end_rc = plan_->end_wave_output_cpu_access(
                pending_partial);
            metrics.sync_us += us_since(sync0);
            pending_partial = -1;
            if (end_rc != RK_NPU_OK) {
                (void)wait();
                rc = end_rc;
                break;
            }
            rc = wait();
            if (rc == RK_NPU_OK) {
                sync0 = Clock::now();
                rc = plan_->begin_wave_output_cpu_read(wave);
                metrics.sync_us += us_since(sync0);
                pending_partial = wave;
            }
        }
        if (rc == RK_NPU_OK && pending_partial >= 0) {
            const auto reduce0 = Clock::now();
            rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(
                plan_->config().M, plan_->config().N,
                plan_->partial_i32(pending_partial), accum, plan_->c_panel_width());
            metrics.output_us += us_since(reduce0);
            sync0 = Clock::now();
            rc = plan_->end_wave_output_cpu_access(pending_partial);
            metrics.sync_us += us_since(sync0);
        }
        if (rc != RK_NPU_OK) {
            (void)plan_->end_wave_output_cpu_access(0);
            return rc;
        }

        const int32_t* final_partial[] = {accum};
        const auto output0 = Clock::now();
        write_output(1, final_partial);
        metrics.output_us += us_since(output0);

        const auto sync1 = Clock::now();
        const int end_rc = plan_->end_wave_output_cpu_access(0);
        metrics.sync_us += us_since(sync1);
        return end_rc;
    }

    rknpu2_matmul_open::detail::I8KnPlan* plan_ = nullptr;
    rk_npu_matmul_op_kind kind_ = RK_NPU_MATMUL_I8I8I32;
    uint64_t cpu_mask_ = 0;
    int cpu_threads_ = 0;
    NpuSubmitter npu_;

    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool initialized_ = false;
    bool stopping_ = false;
    bool pending_ = false;
    bool executing_ = false;
    bool done_ = false;
    RunJob job_{};
    RunMetrics metrics_{};
    int rc_ = RK_NPU_OK;
};

struct WorkspaceState {
    rknpu2_matmul_open::detail::I8KnPlan impl;
    rk_npu_matmul_strategy strategy{};
    std::unique_ptr<SingleRunner> runner;
};

int prepare_workspace(WorkspaceState& state, rk_npu_iommu_domain* domain,
                      const rk_npu_matmul_strategy* strategy) {
    if (!domain || !domain->ctx || !strategy ||
        !valid_strategy(strategy, strategy->op_kind))
        return RK_NPU_ERR_PARAM;
    state.strategy = *strategy;
    const int rc = state.impl.prepare(domain, plan_config(*strategy, 500));
    if (rc != RK_NPU_OK) return rc;
    state.runner.reset(new SingleRunner(
        &state.impl, strategy->op_kind,
        strategy->cpu_core_mask, strategy->cpu_threads));
    const int start_rc = state.runner->start();
    if (start_rc != RK_NPU_OK) state.runner.reset();
    return start_rc;
}

int run_tune_job(SingleRunner& runner, const rknpu2_matmul_open::detail::I8KnWeights& weights,
                 const TuneData& data, int32_t* C_i32, uint16_t* C_f16,
                 float* C_f32,
                 RunMetrics* metrics) {
    RunJob job{};
    job.weights = &weights;
    job.w_scale = data.w_scale.empty() ? nullptr : data.w_scale.data();
    job.A_i8 = data.A_i8.data();
    job.A_f16 = data.A_f16.data();
    job.A_f32 = data.A_f32.data();
    job.a_scale = data.a_scale.data();
    job.C_i32 = C_i32;
    job.C_f16 = C_f16;
    job.C_f32 = C_f32;
    return runner.run(job, metrics);
}

struct LayoutRecipe {
    rk_npu_matmul_a_layout a;
    rk_npu_matmul_c_layout c;
    const char *name;
};

// Layouts are complete recipes, not an independent A/C Cartesian search.
constexpr LayoutRecipe kLayoutRecipes[] = {
    {RK_NPU_MATMUL_A_LAYOUT_NORMAL, RK_NPU_MATMUL_C_LAYOUT_NATIVE, "normal"},
    {RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16, RK_NPU_MATMUL_C_LAYOUT_NATIVE, "native"},
    {RK_NPU_MATMUL_A_LAYOUT_PANEL8, RK_NPU_MATMUL_C_LAYOUT_NATIVE, "pn8"},
    {RK_NPU_MATMUL_A_LAYOUT_PANEL16, RK_NPU_MATMUL_C_LAYOUT_NATIVE, "pn16"},
    {RK_NPU_MATMUL_A_LAYOUT_NORMAL, RK_NPU_MATMUL_C_LAYOUT_PANEL8, "pc8"},
    {RK_NPU_MATMUL_A_LAYOUT_PANEL8, RK_NPU_MATMUL_C_LAYOUT_PANEL8, "p8"},
    {RK_NPU_MATMUL_A_LAYOUT_PANEL16, RK_NPU_MATMUL_C_LAYOUT_PANEL16, "p16"},
};

struct TuneOutput {
    std::vector<int32_t> i32;
    std::vector<uint16_t> f16;
    std::vector<float> f32;
    explicit TuneOutput(size_t size) : i32(size), f16(size), f32(size) {}
};

bool same_tune_output(rk_npu_matmul_op_kind kind, const TuneOutput &expected,
                      const TuneOutput &actual) {
    const auto traits = op_traits(kind);
    if (traits.output == HostDType::I32)
        return equal_i32(expected.i32, actual.i32);
    if (traits.output == HostDType::F16) {
        for (size_t i = 0; i < expected.f16.size(); ++i) {
            const uint16_t a = expected.f16[i], b = actual.f16[i];
            if (a != b && ((a ^ b) & 0x8000u || std::abs(int(a) - int(b)) > 1))
                return false;
        }
    } else {
        for (size_t i = 0; i < expected.f32.size(); ++i) {
            const float a = expected.f32[i], b = actual.f32[i];
            if (!std::isfinite(b) || std::fabs(a - b) > 1e-6f * std::max(1.0f, std::fabs(a)))
                return false;
        }
    }
    return true;
}

// Full-K 3840 at M64 is a recorded submit failure. Keep uncertain intermediate
// large K out of online tuning rather than using a failed submit as a probe.
// This is a conservative policy, not a complete silicon safety formula.
bool tune_k_partition_safe(int M, int K, int kt) {
    if (M <= 4)
        return true;
    for (int k0 = 0; k0 < K; k0 += kt) {
        const int k = align_up(std::min(kt, K - k0), MIN_CHANNEL_TILE);
        if (k > 3072 && k < 4096)
            return false;
    }
    return true;
}

int autotune_impl(rk_npu_ctx *ctx, const rk_npu_matmul_autotune_config *cfg,
                  rk_npu_matmul_op_kind kind, rk_npu_matmul_strategy *fastest,
                  rk_npu_matmul_strategy *stable) {
    if (!ctx || !valid_config(cfg) || (!fastest && !stable))
        return RK_NPU_ERR_PARAM;
    std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
        rk_npu_iommu_domain_create(ctx, 0), rk_npu_iommu_domain_free);
    if (!domain)
        return RK_NPU_ERR_DOMAIN;
    const uint32_t npu_mask =
        fixed_npu_mask(cfg->allowed_npu_core_mask ? cfg->allowed_npu_core_mask : 7u, cfg->N);
    uint64_t allowed_cpu =
        (cfg->allowed_cpu_core_mask ? cfg->allowed_cpu_core_mask : process_cpu_mask()) &
        process_cpu_mask();
    const auto cpu_order = preferred_cpus(allowed_cpu);
    const int cpu_count = std::min(4, (int)cpu_order.size());
    const uint64_t cpu_mask = first_cpu_mask(cpu_order, cpu_count);
    if (!npu_mask || !cpu_count)
        return RK_NPU_ERR_PARAM;
    const int cores = __builtin_popcount(npu_mask);
    const int align_n = std::max(MIN_CHANNEL_TILE, align_up(cfg->N, MIN_CHANNEL_TILE));
    const int safe_groups = std::max(1, ceil_div(align_n, N_TILE_SAFE_MAX));
    std::set<int> n_tiles;
    for (int groups : {cores, cores * 2, safe_groups, safe_groups * 2}) {
        const int nt = choose_n_tile(cfg->N, groups);
        if (nt <= N_TILE_SAFE_MAX && ceil_div(align_n, nt) >= cores)
            n_tiles.insert(nt);
    }
    if (n_tiles.empty())
        return RK_NPU_ERR_PARAM;
    const TuneData data = make_tune_data(kind, cfg->M, cfg->N, cfg->K);
    // Honor the caller's packed-weight partition, including the exact anchor.
    // Otherwise start from a <=2048 K wave instead of the unsafe full-K knee.
    const int anchor_kt =
        cfg->required_k_tile > 0 ? cfg->required_k_tile : std::min({cfg->K, k_knee(cfg->M), 2048});
    if (!tune_k_partition_safe(cfg->M, cfg->K, anchor_kt))
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_strategy anchor_s{};
    fill_topology(anchor_s, kind, cfg->M, cfg->N, cfg->K, anchor_kt, *n_tiles.rbegin(), npu_mask,
                  RK_NPU_MATMUL_A_LAYOUT_NORMAL);
    anchor_s.cpu_threads = cpu_count;
    anchor_s.cpu_core_mask = cpu_mask;
    rknpu2_matmul_open::detail::I8KnPlan anchor;
    rknpu2_matmul_open::detail::I8KnWeights anchor_weights;
    int rc = anchor.prepare(domain.get(), plan_config(anchor_s, cfg->timeout_ms));
    if (rc == RK_NPU_OK)
        rc = anchor_weights.prepare(domain.get(), weight_config(anchor_s), data.B_i8.data());
    std::vector<int32_t> exact_reference;
    if (rc == RK_NPU_OK)
        rc = run_collect_exact(anchor, anchor_weights, kind, data, exact_reference);
    if (rc != RK_NPU_OK)
        return rc;
    TuneOutput expected((size_t)cfg->M * cfg->N);
    {
        SingleRunner reference_runner(&anchor, kind, cpu_mask, cpu_count);
        rc = reference_runner.start();
        if (rc == RK_NPU_OK)
            rc = run_tune_job(reference_runner, anchor_weights, data, expected.i32.data(),
                              expected.f16.data(), expected.f32.data(), nullptr);
        if (rc != RK_NPU_OK)
            return rc;
    }
    if (kind == RK_NPU_MATMUL_I8I8I32 && !equal_i32(exact_reference, expected.i32))
        return RK_NPU_ERR_PARAM;
    if (anchor.config().npu_dequant) {
        // Independent CPU oracle for the explicitly selected FP16-scale
        // contract. Never validate DPU dequant solely against another DPU run.
        try {
            const float* a_scale = op_traits(kind).quant == QuantKind::Dynamic
                                 ? anchor.a_scale() : data.a_scale.data();
            for (int m = 0; m < cfg->M; ++m) {
                const float sa = rknpu2_matmul_open::moe::half_value(
                    rknpu2_matmul_open::moe::half_bits(a_scale[m]));
                for (int n = 0; n < cfg->N; ++n) {
                    const float sw = rknpu2_matmul_open::moe::half_value(
                        rknpu2_matmul_open::moe::half_bits(data.w_scale[n]));
                    const size_t i = size_t(m) * cfg->N + n;
                    const float value = (float(exact_reference[i]) * sa) * sw;
                    expected.f32[i] = value;
#if defined(__aarch64__)
                    const __fp16 half = value;
#else
                    const _Float16 half = value;
#endif
                    std::memcpy(&expected.f16[i], &half, 2);
                }
            }
        } catch (const std::exception&) { return RK_NPU_ERR_PARAM; }
    }
    if (cfg->verbose)
        std::fprintf(stderr, "autotune: fixed npu=0x%x cpu=0x%llx/%d anchor Kt=%d Nt=%d\n",
                     npu_mask, (unsigned long long)cpu_mask, cpu_count, anchor_kt, anchor_s.n_tile);
    auto k_tiles = cfg->required_k_tile > 0 ? std::vector<int>{cfg->required_k_tile}
                                            : k_candidates(cfg->M, cfg->K);
    if (std::find(k_tiles.begin(), k_tiles.end(), anchor_kt) == k_tiles.end())
        k_tiles.push_back(anchor_kt);
    bool have_result = false;
    rk_npu_matmul_strategy best_fast{}, best_stable{};
    for (int kt : k_tiles) {
        if (!tune_k_partition_safe(cfg->M, cfg->K, kt))
            continue;
        rknpu2_matmul_open::detail::I8KnWeights weights;
        rc = weights.prepare(domain.get(), weight_config(cfg->K, cfg->N, kt), data.B_i8.data());
        if (rc != RK_NPU_OK)
            continue;
        for (const auto &recipe : kLayoutRecipes) {
            // M1/M4 panels duplicate existing formats and showed no useful gain.
            if (cfg->M < 8 && (recipe.a >= RK_NPU_MATMUL_A_LAYOUT_PANEL8 ||
                               recipe.c != RK_NPU_MATMUL_C_LAYOUT_NATIVE))
                continue;
            for (int nt : n_tiles) {
                rk_npu_matmul_strategy s{};
                fill_topology(s, kind, cfg->M, cfg->N, cfg->K, kt, nt, npu_mask, recipe.a);
                s.c_layout = recipe.c;
                s.cpu_threads = cpu_count;
                s.cpu_core_mask = cpu_mask;
                const auto pcfg = plan_config(s, cfg->timeout_ms);
                rknpu2_matmul_open::detail::I8KnMemoryInfo memory{};
                // Validates every K wave, M tail and full-row DPU notch before allocation/submit.
                if (rknpu2_matmul_open::detail::query_i8_kn_memory(pcfg, &memory) != RK_NPU_OK)
                    continue;
                rknpu2_matmul_open::detail::I8KnPlan plan;
                rc = plan.prepare(domain.get(), pcfg);
                if (rc != RK_NPU_OK)
                    continue;
                bool valid = true;
                // Repeated prepared submits catch the previously observed first-run-only success.
                for (int trial = 0; trial < 2 && valid; ++trial) {
                    std::vector<int32_t> actual;
                    rc = run_collect_exact(plan, weights, kind, data, actual);
                    if (rc == RK_NPU_ERR_SUBMIT)
                        return rc;
                    valid = rc == RK_NPU_OK && equal_i32(exact_reference, actual);
                }
                if (!valid)
                    continue;
                SingleRunner runner(&plan, kind, cpu_mask, cpu_count);
                rc = runner.start();
                if (rc != RK_NPU_OK)
                    continue;
                TuneOutput output((size_t)cfg->M * cfg->N);
                rknpu2_matmul_open::tune::Score score;
                rc = rknpu2_matmul_open::tune::measure({cfg->warmup, cfg->loops, cfg->repeats},
                    [&](rknpu2_matmul_open::tune::Sample& one) {
                        return run_tune_job(runner, weights, data, output.i32.data(),
                                            output.f16.data(), output.f32.data(), &one);
                    }, [&] { return same_tune_output(kind, expected, output); }, score);
                if (rc == RK_NPU_ERR_SUBMIT) return rc;
                if (rc != RK_NPU_OK) {
                    if (cfg->verbose)
                        std::fprintf(stderr, "autotune: reject %s Kt=%d Nt=%d after pipeline validation\n",
                                     recipe.name, kt, nt);
                    continue;
                }
                s.input_us = score.input_us; s.npu_us = score.npu_us;
                s.sync_us = score.sync_us; s.output_us = score.output_us;
                s.total_us = score.total_us; s.jitter_pct = score.jitter_pct;
                s.robust_us = score.robust_us;
                if (cfg->verbose)
                    std::fprintf(stderr, "autotune: total layout=%s Kt=%d Nt=%d %.2f us\n",
                                 recipe.name, kt, nt, s.total_us);
                if (!have_result || s.total_us < best_fast.total_us)
                    best_fast = s;
                if (!have_result || s.robust_us < best_stable.robust_us)
                    best_stable = s;
                have_result = true;
            }
        }
    }
    if (!have_result)
        return RK_NPU_ERR_SUBMIT;
    if (fastest)
        *fastest = best_fast;
    if (stable)
        *stable = best_stable;
    return RK_NPU_OK;
}

int autotune_cached_impl(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_op_kind kind, const char* cache_path, int refresh,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable,
    int* cache_hit) {
    if (!ctx || !valid_config(cfg) || !supported_op_kind(kind) ||
        (!fastest && !stable) || (cache_path && !*cache_path))
        return RK_NPU_ERR_PARAM;
    if (cache_hit) *cache_hit = 0;
    AutotuneCacheKey key{};
    if (!make_cache_key(ctx, cfg, kind, key)) return RK_NPU_ERR_PARAM;
    const std::string path = cache_path ? cache_path : default_cache_path(key);
    rk_npu_matmul_strategy cached_fast{}, cached_stable{};
    if (!refresh && !path.empty() &&
        load_cache_file(path, key, cached_fast, cached_stable) == RK_NPU_OK) {
        if (fastest) *fastest = cached_fast;
        if (stable) *stable = cached_stable;
        if (cache_hit) *cache_hit = 1;
        if (cfg->verbose)
            std::fprintf(stderr, "autotune cache: hit %s\n", path.c_str());
        return RK_NPU_OK;
    }

    if (cfg->verbose)
        std::fprintf(stderr, "autotune cache: %s%s\n",
                     refresh ? "refresh " : "miss ",
                     path.empty() ? "(cache directory unavailable)" : path.c_str());
    const int rc = autotune_impl(ctx, cfg, kind, &cached_fast, &cached_stable);
    if (rc != RK_NPU_OK) return rc;
    if (fastest) *fastest = cached_fast;
    if (stable) *stable = cached_stable;
    if (!path.empty()) {
        const int save_rc = save_cache_file(path, key, cached_fast, cached_stable);
        if (save_rc != RK_NPU_OK && cfg->verbose)
            std::fprintf(stderr, "autotune cache: save failed %s: %s\n",
                         path.c_str(), rk_npu_strerror(save_rc));
    }
    return RK_NPU_OK;
}

int autotune_family_impl(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_op_kind kind,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies,
    bool use_cache = false, const char* cache_directory = nullptr,
    int refresh = 0, int* cache_hits = nullptr, int* cache_misses = nullptr) {
    const bool want_fast = fastest_summary && fastest_strategies;
    const bool want_stable = stable_summary && stable_strategies;
    if (!ctx || !valid_family_config(cfg) || !supported_op_kind(kind) ||
        ((fastest_summary == nullptr) != (fastest_strategies == nullptr)) ||
        ((stable_summary == nullptr) != (stable_strategies == nullptr)) ||
        (!want_fast && !want_stable) ||
        (use_cache && cache_directory && !*cache_directory))
        return RK_NPU_ERR_PARAM;
    if (cache_hits) *cache_hits = 0;
    if (cache_misses) *cache_misses = 0;
    int hit_count = 0;
    int miss_count = 0;

    std::set<int> k_tiles;
    for (int i = 0; i < cfg->m_count; ++i) {
        const std::vector<int> one = k_candidates(cfg->m_values[i], cfg->K);
        k_tiles.insert(one.begin(), one.end());
    }
    if (k_tiles.empty()) return RK_NPU_ERR_PARAM;

    double frequency_sum = 0;
    for (int i = 0; i < cfg->m_count; ++i)
        frequency_sum += cfg->frequencies ? cfg->frequencies[i] : 1.0;

    bool have_fast = false, have_stable = false;
    double best_fast_score = 0, best_stable_score = 0;
    std::vector<rk_npu_matmul_strategy> best_fast((size_t)cfg->m_count);
    std::vector<rk_npu_matmul_strategy> best_stable((size_t)cfg->m_count);

    for (int kt : k_tiles) {
        std::vector<rk_npu_matmul_strategy> family_fast((size_t)cfg->m_count);
        std::vector<rk_npu_matmul_strategy> family_stable((size_t)cfg->m_count);
        double fast_score = 0, stable_score = 0;
        int rc = RK_NPU_OK;
        for (int i = 0; i < cfg->m_count && rc == RK_NPU_OK; ++i) {
            rk_npu_matmul_autotune_config one{};
            one.M = cfg->m_values[i];
            one.N = cfg->N;
            one.K = cfg->K;
            one.allowed_npu_core_mask = cfg->allowed_npu_core_mask;
            one.allowed_cpu_core_mask = cfg->allowed_cpu_core_mask;
            one.warmup = cfg->warmup;
            one.loops = cfg->loops;
            one.repeats = cfg->repeats;
            one.timeout_ms = cfg->timeout_ms;
            one.verbose = cfg->verbose;
            one.required_k_tile = kt;
            if (use_cache) {
                std::string member_path;
                if (cache_directory) {
                    AutotuneCacheKey member_key{};
                    if (!make_cache_key(ctx, &one, kind, member_key)) {
                        rc = RK_NPU_ERR_PARAM;
                        break;
                    }
                    member_path = cache_path_in_directory(
                        cache_directory, member_key);
                }
                int one_hit = 0;
                rc = autotune_cached_impl(
                    ctx, &one, kind,
                    cache_directory ? member_path.c_str() : nullptr,
                    refresh, &family_fast[(size_t)i],
                    &family_stable[(size_t)i], &one_hit);
                if (rc == RK_NPU_OK) {
                    hit_count += one_hit;
                    miss_count += !one_hit;
                    if (cache_hits) *cache_hits = hit_count;
                    if (cache_misses) *cache_misses = miss_count;
                }
            } else {
                rc = autotune_impl(ctx, &one, kind,
                                   &family_fast[(size_t)i],
                                   &family_stable[(size_t)i]);
            }
            if (rc == RK_NPU_OK) {
                const double frequency = cfg->frequencies
                    ? cfg->frequencies[i] : 1.0;
                fast_score += frequency * family_fast[(size_t)i].total_us;
                stable_score += frequency * family_stable[(size_t)i].robust_us;
            }
        }
        if (rc == RK_NPU_ERR_SUBMIT) return rc;
        if (rc != RK_NPU_OK) continue;
        fast_score /= frequency_sum;
        stable_score /= frequency_sum;
        if (cfg->verbose)
            std::fprintf(stderr,
                         "autotune family: Kt=%d weighted=%.2f robust=%.2f us\n",
                         kt, fast_score, stable_score);
        if (!have_fast || fast_score < best_fast_score) {
            best_fast_score = fast_score;
            best_fast = family_fast;
            have_fast = true;
        }
        if (!have_stable || stable_score < best_stable_score) {
            best_stable_score = stable_score;
            best_stable = family_stable;
            have_stable = true;
        }
    }
    if ((want_fast && !have_fast) || (want_stable && !have_stable))
        return RK_NPU_ERR_SUBMIT;

    if (want_fast) {
        fastest_summary->weight_config = {
            cfg->K, cfg->N, best_fast[0].k_tile};
        fastest_summary->weighted_total_us = best_fast_score;
        double robust = 0;
        for (int i = 0; i < cfg->m_count; ++i) {
            fastest_strategies[i] = best_fast[(size_t)i];
            robust += (cfg->frequencies ? cfg->frequencies[i] : 1.0) *
                      best_fast[(size_t)i].robust_us;
        }
        fastest_summary->weighted_robust_us = robust / frequency_sum;
    }
    if (want_stable) {
        stable_summary->weight_config = {
            cfg->K, cfg->N, best_stable[0].k_tile};
        stable_summary->weighted_robust_us = best_stable_score;
        double total = 0;
        for (int i = 0; i < cfg->m_count; ++i) {
            stable_strategies[i] = best_stable[(size_t)i];
            total += (cfg->frequencies ? cfg->frequencies[i] : 1.0) *
                     best_stable[(size_t)i].total_us;
        }
        stable_summary->weighted_total_us = total / frequency_sum;
    }
    if (use_cache && cfg->verbose)
        std::fprintf(stderr, "autotune family cache: hits=%d misses=%d\n",
                     hit_count, miss_count);
    return RK_NPU_OK;
}

} /* anonymous namespace */

struct rk_npu_matmul_workspace {
    std::shared_ptr<WorkspaceState> state;
};
struct rk_npu_i8i8i32_weights {
    rknpu2_matmul_open::detail::I8KnWeights impl;
};
struct rk_npu_f16i8f16_weights {
    rknpu2_matmul_open::detail::I8KnWeights impl;
    std::vector<float> w_scale;
};
struct rk_npu_f32i8f32_weights {
    rknpu2_matmul_open::detail::I8KnWeights impl;
    std::vector<float> w_scale;
};

extern "C" void rk_npu_matmul_autotune_config_init(
    rk_npu_matmul_autotune_config* cfg, int M, int N, int K) {
    if (!cfg) return;
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->M = M;
    cfg->N = N;
    cfg->K = K;
    cfg->allowed_npu_core_mask = 7;
    cfg->warmup = 2;
    cfg->loops = 8;
    cfg->repeats = 3;
    cfg->timeout_ms = 500;
}

extern "C" void rk_npu_matmul_family_autotune_config_init(
    rk_npu_matmul_family_autotune_config* cfg, int N, int K,
    int m_count, const int* m_values) {
    if (!cfg) return;
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->K = K;
    cfg->N = N;
    cfg->m_count = m_count;
    cfg->m_values = m_values;
    cfg->allowed_npu_core_mask = 7;
    cfg->warmup = 2;
    cfg->loops = 8;
    cfg->repeats = 3;
    cfg->timeout_ms = 500;
}

extern "C" int rk_npu_matmul_autotune_cache_load(
    rk_npu_ctx* ctx, const char* cache_path,
    rk_npu_matmul_op_kind kind,
    const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest,
    rk_npu_matmul_strategy* stable) {
    if (!cache_path || !*cache_path || !fastest || !stable)
        return RK_NPU_ERR_PARAM;
    AutotuneCacheKey key{};
    if (!make_cache_key(ctx, cfg, kind, key)) return RK_NPU_ERR_PARAM;
    return load_cache_file(cache_path, key, *fastest, *stable);
}

extern "C" int rk_npu_matmul_autotune_cache_save(
    rk_npu_ctx* ctx, const char* cache_path,
    rk_npu_matmul_op_kind kind,
    const rk_npu_matmul_autotune_config* cfg,
    const rk_npu_matmul_strategy* fastest,
    const rk_npu_matmul_strategy* stable) {
    if (!cache_path || !*cache_path || !fastest || !stable)
        return RK_NPU_ERR_PARAM;
    AutotuneCacheKey key{};
    if (!make_cache_key(ctx, cfg, kind, key) ||
        !strategy_matches_cache(*fastest, key) ||
        !strategy_matches_cache(*stable, key))
        return RK_NPU_ERR_PARAM;
    return save_cache_file(cache_path, key, *fastest, *stable);
}

extern "C" int rk_npu_matmul_autotune_cached(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_op_kind kind, const char* cache_path, int refresh,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable,
    int* cache_hit) {
    return autotune_cached_impl(
        ctx, cfg, kind, cache_path, refresh,
        fastest, stable, cache_hit);
}

extern "C" int rk_npu_i8i8i32_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable) {
    return autotune_impl(ctx, cfg, RK_NPU_MATMUL_I8I8I32, fastest, stable);
}

extern "C" int rk_npu_f16i8f16_dynamic_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable) {
    return autotune_impl(ctx, cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC, fastest, stable);
}

extern "C" int rk_npu_f16i8f16_static_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable) {
    return autotune_impl(ctx, cfg, RK_NPU_MATMUL_F16I8F16_STATIC, fastest, stable);
}

extern "C" int rk_npu_f32i8f32_dynamic_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable) {
    return autotune_impl(ctx, cfg, RK_NPU_MATMUL_F32I8F32_DYNAMIC, fastest, stable);
}

extern "C" int rk_npu_f32i8f32_static_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable) {
    return autotune_impl(ctx, cfg, RK_NPU_MATMUL_F32I8F32_STATIC, fastest, stable);
}

extern "C" int rk_npu_i8i8i32_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies) {
    return autotune_family_impl(
        ctx, cfg, RK_NPU_MATMUL_I8I8I32,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies);
}

extern "C" int rk_npu_f16i8f16_dynamic_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies) {
    return autotune_family_impl(
        ctx, cfg, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies);
}

extern "C" int rk_npu_f16i8f16_static_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies) {
    return autotune_family_impl(
        ctx, cfg, RK_NPU_MATMUL_F16I8F16_STATIC,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies);
}

extern "C" int rk_npu_f32i8f32_dynamic_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies) {
    return autotune_family_impl(
        ctx, cfg, RK_NPU_MATMUL_F32I8F32_DYNAMIC,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies);
}

extern "C" int rk_npu_f32i8f32_static_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies) {
    return autotune_family_impl(
        ctx, cfg, RK_NPU_MATMUL_F32I8F32_STATIC,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies);
}

extern "C" int rk_npu_matmul_autotune_family_cached(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_op_kind kind, const char* cache_directory, int refresh,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies,
    int* cache_hits, int* cache_misses) {
    return autotune_family_impl(
        ctx, cfg, kind,
        fastest_summary, fastest_strategies,
        stable_summary, stable_strategies,
        true, cache_directory, refresh, cache_hits, cache_misses);
}

extern "C" int rk_npu_matmul_workspace_memory_query(
    const rk_npu_matmul_strategy* strategy,
    rk_npu_matmul_workspace_requirements* out) {
    if (!strategy || !out || !valid_strategy(strategy, strategy->op_kind))
        return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::detail::I8KnMemoryInfo info{};
    const int rc = rknpu2_matmul_open::detail::query_i8_kn_memory(plan_config(*strategy, 500), &info);
    if (rc != RK_NPU_OK) return rc;
    std::memset(out, 0, sizeof(*out));
    out->input_bytes = info.input_bytes;
    out->output_bytes = info.output_bytes;
    out->control_bytes = info.control_bytes;
    out->bytes = info.input_bytes + info.output_bytes + info.control_bytes +
                 (uint64_t)strategy->M * sizeof(float);
    out->wave_count = info.wave_count;
    out->buffer_count = info.workspace_buffer_count;
    return RK_NPU_OK;
}

namespace {
int weights_memory_query(const rk_npu_matmul_weight_config* config,
                         bool with_scale,
                         rk_npu_matmul_weight_requirements* out) {
    if (!valid_weight_config(config) || !out) return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::detail::I8KnMemoryInfo info{};
    const int rc = rknpu2_matmul_open::detail::query_i8_kn_weight_memory(
        weight_config(*config), &info);
    if (rc != RK_NPU_OK) return rc;
    std::memset(out, 0, sizeof(*out));
    out->packed_b_bytes = info.weight_bytes;
    out->scale_bytes = with_scale ? (uint64_t)config->N * sizeof(float) : 0;
    out->bytes = out->packed_b_bytes + out->scale_bytes;
    out->wave_count = info.wave_count;
    out->buffer_count = info.weight_buffer_count;
    return RK_NPU_OK;
}
} /* anonymous namespace */

extern "C" int rk_npu_i8i8i32_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out) {
    return weights_memory_query(config, false, out);
}

extern "C" int rk_npu_f16i8f16_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out) {
    return weights_memory_query(config, true, out);
}

extern "C" int rk_npu_f32i8f32_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out) {
    return weights_memory_query(config, true, out);
}

extern "C" rk_npu_matmul_workspace* rk_npu_matmul_workspace_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_strategy* strategy) {
    if (!domain || !domain->ctx || !strategy ||
        !valid_strategy(strategy, strategy->op_kind))
        return nullptr;
    rk_npu_matmul_workspace* handle = nullptr;
    try {
        handle = new rk_npu_matmul_workspace();
        handle->state = std::make_shared<WorkspaceState>();
    } catch (const std::bad_alloc&) {
        delete handle;
        return nullptr;
    }
    if (prepare_workspace(*handle->state, domain, strategy) != RK_NPU_OK) {
        delete handle;
        return nullptr;
    }
    return handle;
}

extern "C" void rk_npu_matmul_workspace_free(
    rk_npu_matmul_workspace* workspace) {
    delete workspace;
}

extern "C" rk_npu_i8i8i32_weights* rk_npu_i8i8i32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor) {
    if (!domain || !domain->ctx || !valid_weight_config(config) || !B_rowmajor)
        return nullptr;
    rk_npu_i8i8i32_weights* weights = new rk_npu_i8i8i32_weights();
    if (weights->impl.prepare(domain, weight_config(*config), B_rowmajor) !=
        RK_NPU_OK) {
        delete weights;
        return nullptr;
    }
    return weights;
}

extern "C" void rk_npu_i8i8i32_weights_free(
    rk_npu_i8i8i32_weights* weights) {
    delete weights;
}

extern "C" int rk_npu_i8i8i32_run(rk_npu_matmul_workspace* workspace,
                                    const rk_npu_i8i8i32_weights* weights,
                                    const int8_t* A_rowmajor,
                                    int32_t* C_rowmajor) {
    if (!workspace || !workspace->state || !weights || !A_rowmajor ||
        !C_rowmajor ||
        workspace->state->strategy.op_kind != RK_NPU_MATMUL_I8I8I32)
        return RK_NPU_ERR_PARAM;
    if (!weights->impl.same_domain(workspace->state->impl))
        return RK_NPU_ERR_DOMAIN;
    if (!weights->impl.compatible(workspace->state->impl))
        return RK_NPU_ERR_PARAM;
    RunJob job{};
    job.weights = &weights->impl;
    job.A_i8 = A_rowmajor;
    job.C_i32 = C_rowmajor;
    return workspace->state->runner->run(job, nullptr);
}

namespace {
int prepare_scaled_weights(rknpu2_matmul_open::detail::I8KnWeights& impl,
                           std::vector<float>& stored_scale,
                           rk_npu_iommu_domain* domain,
                           const rk_npu_matmul_weight_config* config,
                           const int8_t* B_rowmajor, const float* w_scale) {
    if (!domain || !domain->ctx || !valid_weight_config(config) || !B_rowmajor ||
        !valid_scales(w_scale, config->N))
        return RK_NPU_ERR_PARAM;
    const int rc = impl.prepare(domain, weight_config(*config), B_rowmajor);
    if (rc != RK_NPU_OK) return rc;
    stored_scale.assign(w_scale, w_scale + config->N);
    return RK_NPU_OK;
}
} /* anonymous namespace */

extern "C" rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale) {
    rk_npu_f16i8f16_weights* weights = new rk_npu_f16i8f16_weights();
    const int rc = prepare_scaled_weights(
        weights->impl, weights->w_scale, domain, config, B_rowmajor, w_scale);
    if (rc != RK_NPU_OK) {
        delete weights;
        return nullptr;
    }
    return weights;
}

extern "C" void rk_npu_f16i8f16_weights_free(
    rk_npu_f16i8f16_weights* weights) {
    delete weights;
}

extern "C" int rk_npu_f16i8f16_run_dynamic(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f16i8f16_weights* weights,
    const uint16_t* A_fp16_rowmajor,
    uint16_t* C_fp16_rowmajor) {
    if (!workspace || !workspace->state || !weights ||
        workspace->state->strategy.op_kind != RK_NPU_MATMUL_F16I8F16_DYNAMIC ||
        !A_fp16_rowmajor || !C_fp16_rowmajor)
        return RK_NPU_ERR_PARAM;
    if (!weights->impl.same_domain(workspace->state->impl))
        return RK_NPU_ERR_DOMAIN;
    if (!weights->impl.compatible(workspace->state->impl))
        return RK_NPU_ERR_PARAM;
    RunJob job{};
    job.weights = &weights->impl;
    job.w_scale = weights->w_scale.data();
    job.A_f16 = A_fp16_rowmajor;
    job.C_f16 = C_fp16_rowmajor;
    return workspace->state->runner->run(job, nullptr);
}

extern "C" int rk_npu_f16i8f16_run_static(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f16i8f16_weights* weights,
    const uint16_t* A_fp16_rowmajor,
    const float* a_scale, uint16_t* C_fp16_rowmajor) {
    if (!workspace || !workspace->state || !weights ||
        workspace->state->strategy.op_kind != RK_NPU_MATMUL_F16I8F16_STATIC ||
        !A_fp16_rowmajor || !C_fp16_rowmajor ||
        !valid_scales(a_scale, workspace->state->strategy.M))
        return RK_NPU_ERR_PARAM;
    if (!weights->impl.same_domain(workspace->state->impl))
        return RK_NPU_ERR_DOMAIN;
    if (!weights->impl.compatible(workspace->state->impl))
        return RK_NPU_ERR_PARAM;
    RunJob job{};
    job.weights = &weights->impl;
    job.w_scale = weights->w_scale.data();
    job.A_f16 = A_fp16_rowmajor;
    job.a_scale = a_scale;
    job.C_f16 = C_fp16_rowmajor;
    return workspace->state->runner->run(job, nullptr);
}

extern "C" rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale) {
    rk_npu_f32i8f32_weights* weights = new rk_npu_f32i8f32_weights();
    const int rc = prepare_scaled_weights(
        weights->impl, weights->w_scale, domain, config, B_rowmajor, w_scale);
    if (rc != RK_NPU_OK) {
        delete weights;
        return nullptr;
    }
    return weights;
}

extern "C" void rk_npu_f32i8f32_weights_free(
    rk_npu_f32i8f32_weights* weights) {
    delete weights;
}

extern "C" int rk_npu_f32i8f32_run_dynamic(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f32i8f32_weights* weights,
    const float* A_fp32_rowmajor,
    float* C_fp32_rowmajor) {
    if (!workspace || !workspace->state || !weights ||
        workspace->state->strategy.op_kind != RK_NPU_MATMUL_F32I8F32_DYNAMIC ||
        !A_fp32_rowmajor || !C_fp32_rowmajor)
        return RK_NPU_ERR_PARAM;
    if (!weights->impl.same_domain(workspace->state->impl))
        return RK_NPU_ERR_DOMAIN;
    if (!weights->impl.compatible(workspace->state->impl))
        return RK_NPU_ERR_PARAM;
    RunJob job{};
    job.weights = &weights->impl;
    job.w_scale = weights->w_scale.data();
    job.A_f32 = A_fp32_rowmajor;
    job.C_f32 = C_fp32_rowmajor;
    return workspace->state->runner->run(job, nullptr);
}

extern "C" int rk_npu_f32i8f32_run_static(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f32i8f32_weights* weights,
    const float* A_fp32_rowmajor,
    const float* a_scale,
    float* C_fp32_rowmajor) {
    if (!workspace || !workspace->state || !weights ||
        workspace->state->strategy.op_kind != RK_NPU_MATMUL_F32I8F32_STATIC ||
        !A_fp32_rowmajor || !C_fp32_rowmajor ||
        !valid_scales(a_scale, workspace->state->strategy.M))
        return RK_NPU_ERR_PARAM;
    if (!weights->impl.same_domain(workspace->state->impl))
        return RK_NPU_ERR_DOMAIN;
    if (!weights->impl.compatible(workspace->state->impl))
        return RK_NPU_ERR_PARAM;
    RunJob job{};
    job.weights = &weights->impl;
    job.w_scale = weights->w_scale.data();
    job.A_f32 = A_fp32_rowmajor;
    job.a_scale = a_scale;
    job.C_f32 = C_fp32_rowmajor;
    return workspace->state->runner->run(job, nullptr);
}

// Compression is an opt-in weight-construction choice; execution types and
// per-token activation quantization remain identical to the raw API.
extern "C" rk_npu_i8i8i32_weights* rk_npu_i8i8i32_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B) {
    if (!domain || !domain->ctx || !valid_weight_config(config) || !B) return nullptr;
    try {
        auto w = std::make_unique<rk_npu_i8i8i32_weights>();
        if (w->impl.prepare_compress(domain, weight_config(*config), B) != RK_NPU_OK)
            return nullptr;
        return w.release();
    } catch (const std::exception&) { return nullptr; }
}

namespace {
template<class Weights> Weights* create_scaled_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B, const float* scale) {
    if (!domain || !domain->ctx || !valid_weight_config(config) || !B ||
        !valid_scales(scale, config->N)) return nullptr;
    try {
        auto w = std::make_unique<Weights>();
        if (w->impl.prepare_compress(domain, weight_config(*config), B) != RK_NPU_OK)
            return nullptr;
        w->w_scale.assign(scale, scale + config->N);
        return w.release();
    } catch (const std::exception&) { return nullptr; }
}
template<class Weights> Weights* create_f32_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const float* B, float target) {
    if (!domain || !domain->ctx || !valid_weight_config(config) || !B) return nullptr;
    try {
        std::vector<int8_t> codes;
        std::vector<float> scales;
        if (rknpu2_matmul_open::detail::dcomp_quantize_f32(config->K, config->N, config->k_tile,
                                     B, target, codes, scales) != RK_NPU_OK) return nullptr;
        return create_scaled_compress<Weights>(domain, config, codes.data(), scales.data());
    } catch (const std::exception&) { return nullptr; }
}
}

extern "C" rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B, const float* scale) {
    return create_scaled_compress<rk_npu_f16i8f16_weights>(domain, config, B, scale);
}
extern "C" rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create_from_f32_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const float* B, float target_bpw) {
    return create_f32_compress<rk_npu_f16i8f16_weights>(domain, config, B, target_bpw);
}

extern "C" rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B, const float* scale) {
    return create_scaled_compress<rk_npu_f32i8f32_weights>(domain, config, B, scale);
}
extern "C" rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create_from_f32_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const float* B, float target_bpw) {
    return create_f32_compress<rk_npu_f32i8f32_weights>(domain, config, B, target_bpw);
}

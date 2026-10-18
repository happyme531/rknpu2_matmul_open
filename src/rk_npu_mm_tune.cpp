#include "rk_npu_mm_gelu.h"
#include "rk_npu_mm_tune.h"
#include "rk_npu_autotune_common.h"
#include "rk_npu_autotune_cache.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include "rk_npu_bfloat_bits.h"
#include "rk_npu_float_reduce.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <istream>
#include <memory>
#include <map>
#include <omp.h>
#include <ostream>
#include <pthread.h>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tune = rknpu2_matmul_open::tune;
namespace bits = rknpu2_matmul_open::bits;
namespace detail = rknpu2_matmul_open::detail;
namespace {
// Bump when candidate generation, layouts, execution or validation changes.
uint32_t revision_for(rk_npu_mm_dtype type) { return type == RK_NPU_MM_F16 ? 7 : 1; }
int environment_k_mode() {
    const char* value=std::getenv("RK_NPU_MM_K_TILING");
    if(!value || !*value || !std::strcmp(value,"fp16") || !std::strcmp(value,"on") || !std::strcmp(value,"1"))return 1;
    if(!std::strcmp(value,"off") || !std::strcmp(value,"0"))return 0;
    return -1;
}
using Clock = std::chrono::steady_clock;
using Plan = std::unique_ptr<rk_npu_mm_plan, decltype(&rk_npu_mm_plan_free)>;
using Work = std::unique_ptr<rk_npu_mm_workspace, decltype(&rk_npu_mm_workspace_free)>;
using Weight = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;
using TunedWork = std::unique_ptr<rk_npu_mm_tuned_workspace, decltype(&rk_npu_mm_tuned_workspace_free)>;
double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
uint16_t encode(float x, rk_npu_mm_dtype type) {
    return type == RK_NPU_MM_BF16 ? bits::float_to_bfloat(x) : bits::float_to_half(x);
}
float decode(uint16_t x, rk_npu_mm_dtype type) {
    return type == RK_NPU_MM_BF16 ? bits::bfloat_to_float(x) : bits::half_to_float(x);
}
void narrow(const float* src, uint16_t* dst, size_t count, rk_npu_mm_dtype type) {
    size_t j = 0;
#if defined(__aarch64__)
    if (type == RK_NPU_MM_F16) {
        for (; j + 8 <= count; j += 8) {
            const auto lo = vcvt_f16_f32(vld1q_f32(src + j));
            const auto hi = vcvt_f16_f32(vld1q_f32(src + j + 4));
            vst1q_u16(dst + j, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
        }
    }
#endif
    for (; j < count; ++j) dst[j] = encode(src[j], type);
}
void collect(const uint16_t* src, float* dst, size_t count, int n,
             const float* bias, rk_npu_mm_dtype type) {
    for (size_t row = 0; row < count; row += n) {
        int j = 0;
#if defined(__aarch64__)
        for (; j + 8 <= n; j += 8) {
            const auto raw = vld1q_u16(src + row + j);
            float32x4_t lo, hi;
            if (type == RK_NPU_MM_F16) {
                const auto h = vreinterpretq_f16_u16(raw);
                lo = vcvt_f32_f16(vget_low_f16(h));
                hi = vcvt_f32_f16(vget_high_f16(h));
            } else {
                lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
                hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw), 16));
            }
            if (bias) { lo = vaddq_f32(lo, vld1q_f32(bias + j)); hi = vaddq_f32(hi, vld1q_f32(bias + j + 4)); }
            vst1q_f32(dst + row + j, lo); vst1q_f32(dst + row + j + 4, hi);
        }
#endif
        for (; j < n; ++j) dst[row + j] = decode(src[row + j], type) + (bias ? bias[j] : 0.f);
    }
}
void storage_bias(const uint16_t* src,uint16_t* dst,size_t count,int n,
                  const float* bias,rk_npu_mm_dtype type) {
    for(size_t row=0;row<count;row+=n) {
        int j=0;
#if defined(__aarch64__)
        if(type==RK_NPU_MM_F16)for(;j+8<=n;j+=8) {
            const auto h=vreinterpretq_f16_u16(vld1q_u16(src+row+j));
            const auto lo=vaddq_f32(vcvt_f32_f16(vget_low_f16(h)),vld1q_f32(bias+j));
            const auto hi=vaddq_f32(vcvt_f32_f16(vget_high_f16(h)),vld1q_f32(bias+j+4));
            vst1q_u16(dst+row+j,vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(hi))));
        }
#endif
        for(;j<n;++j)dst[row+j]=encode(decode(src[row+j],type)+bias[j],type);
    }
}
int normalize(const rk_npu_mm_autotune_config* src, rk_npu_mm_autotune_config& c) {
    if (!src) return RK_NPU_ERR_PARAM;
    c = *src;
    if(c.dtype==RK_NPU_MM_BF16)c.allow_k_tiling=0;
    if (c.M < 1 || c.N < 1 || c.K < 1 || c.M > 65504 || c.N > 65504 || c.K > 65504 ||
        (c.dtype != RK_NPU_MM_F16 && c.dtype != RK_NPU_MM_BF16) ||
        (c.host_type != RK_NPU_MM_HOST_STORAGE && c.host_type != RK_NPU_MM_HOST_F32) ||
        (c.with_bias != 0 && c.with_bias != 1) ||
        (c.allow_k_tiling != 0 && c.allow_k_tiling != 1) ||
        c.warmup < 0 || c.loops < 1 || c.repeats < 1) return RK_NPU_ERR_PARAM;
    if (!c.allowed_npu_core_mask) c.allowed_npu_core_mask = 7;
    const auto mask = c.allowed_npu_core_mask;
    if (mask != 1 && mask != 2 && mask != 4 && mask != 3 && mask != 7) return RK_NPU_ERR_PARAM;
    const auto available = tune::process_cpu_mask();
    if (c.allowed_cpu_core_mask && (c.allowed_cpu_core_mask & available) != c.allowed_cpu_core_mask)
        return RK_NPU_ERR_PARAM;
    if (!c.allowed_cpu_core_mask)
        c.allowed_cpu_core_mask = tune::first_cpu_mask(tune::preferred_cpus(available), 4);
    if (__builtin_popcountll(c.allowed_cpu_core_mask) > 4) return RK_NPU_ERR_PARAM;
    return RK_NPU_OK;
}
int make_plan(const rk_npu_mm_strategy& s, Plan& p) {
    if (s.tuning_revision != revision_for(s.dtype) || s.M < 1 || s.M > 65504 ||
        s.N < 1 || s.K < 1 || s.m_tile < 1 || s.m_tile > 65504 ||
        (s.dtype != RK_NPU_MM_F16 && s.dtype != RK_NPU_MM_BF16) ||
        (s.host_type != RK_NPU_MM_HOST_STORAGE && s.host_type != RK_NPU_MM_HOST_F32) ||
        (s.with_bias != 0 && s.with_bias != 1) ||
        s.options.split_k != 0 || (s.fused_io != 0 && s.fused_io != 1) ||
        s.k_tile < 0 || (s.k_tile && s.dtype != RK_NPU_MM_F16) ||
        (s.fused_io && (s.dtype != RK_NPU_MM_F16 || s.host_type != RK_NPU_MM_HOST_F32 || s.k_tile)))
        return RK_NPU_ERR_PARAM;
    rk_npu_mm_desc d;
    rk_npu_mm_desc_init(&d, s.m_tile, s.N, s.K, 1);
    d.a_type = d.b_type = d.c_type = s.dtype;
    rk_npu_mm_plan* raw = nullptr;
    const int rc = rk_npu_mm_plan_create_partitioned(&d, &s.options, s.task_m_tile, s.k_tile, &raw);
    p.reset(raw);
    return rc;
}
rk_npu_mm_strategy baseline(const rk_npu_mm_autotune_config& c) {
    rk_npu_mm_strategy s{};
    s.M = c.M; s.N = c.N; s.K = c.K;
    s.dtype = c.dtype; s.host_type = c.host_type; s.with_bias = c.with_bias;
    s.m_tile = 128;
    rk_npu_mm_options_init(&s.options);
    s.options.a_layout = s.options.c_layout = RK_NPU_MM_LAYOUT_NATIVE;
    s.options.allowed_npu_core_mask = c.allowed_npu_core_mask;
    s.options.timeout_ms = c.timeout_ms;
    s.options.split_k = 0;
    s.cpu_core_mask = c.allowed_cpu_core_mask;
    s.cpu_threads = __builtin_popcountll(s.cpu_core_mask);
    s.tuning_revision = revision_for(c.dtype);
    return s;
}
std::vector<int> task_caps(const rk_npu_mm_strategy& s) {
    std::vector<int> caps{0,8,16,32,64,96,128,192,256};
    auto panel=[](rk_npu_mm_layout l) {return l==RK_NPU_MM_LAYOUT_PANEL8?8:l==RK_NPU_MM_LAYOUT_PANEL16?16:1;};
    const int alignment=std::max(panel(s.options.a_layout),panel(s.options.c_layout));
    const int cores=__builtin_popcount(s.options.allowed_npu_core_mask);
    for(int groups=1;groups<=4;++groups) {
        const int balanced=ceil_div(s.m_tile,cores*groups);
        caps.push_back(align_up(balanced,alignment));
        caps.push_back(std::max(alignment,balanced/alignment*alignment));
        if(alignment==1)caps.push_back(align_up(balanced,8));
    }
    std::sort(caps.begin(),caps.end());caps.erase(std::unique(caps.begin(),caps.end()),caps.end());
    return caps;
}
std::vector<int> k_tiles(int k) {
    std::vector<int> result;
    for(int tile:{256,512,1024,2048,align_up(ceil_div(k,2),32),align_up(ceil_div(k,3),32)})
        if(tile<k && ceil_div(k,tile)<=32)result.push_back(tile);
    std::sort(result.begin(),result.end());
    result.erase(std::unique(result.begin(),result.end()),result.end());
    return result;
}
std::vector<rk_npu_mm_strategy> candidates(const rk_npu_mm_autotune_config& c) {
    const auto base = baseline(c);
    std::vector<rk_npu_mm_strategy> result;
    // Always measure the production fallback first.
    result.push_back(base);
    std::vector<int> rows{c.M};
    const std::vector<int> m_values = c.dtype == RK_NPU_MM_F16
        ? std::vector<int>{16,32,64,96,128,192,256,384,512,768,1024}
        : std::vector<int>{32,64,128,256,512};
    for (int m : m_values)
        if (m < c.M && (c.dtype != RK_NPU_MM_F16 || m >= std::max(16, c.M / 8))) rows.push_back(m);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    const int an = align_up(c.N, 32);
    std::vector<int> tiles{0};
    const std::vector<int> n_values = c.dtype == RK_NPU_MM_F16
        ? std::vector<int>{128,256,512,1024,2048} : std::vector<int>{256,512,1024};
    for (int nt : n_values) if (nt < an) tiles.push_back(nt);
    const int balanced = align_up(ceil_div(an, __builtin_popcount(c.allowed_npu_core_mask)), 32);
    if (balanced < an) tiles.push_back(balanced);
    std::sort(tiles.begin(), tiles.end());
    tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
    std::vector<int> kt{0};
    if(c.dtype==RK_NPU_MM_F16 && c.allow_k_tiling && c.K>1024) {
        auto tiles=k_tiles(c.K);kt.insert(kt.end(),tiles.begin(),tiles.end());
    }
    for(int tile_k:kt) {
        const auto recipes = c.dtype == RK_NPU_MM_F16
            ? std::vector<std::pair<int,int>>{{0,0},{0,1},{1,0},{1,1},{2,1},{3,1},{0,2},{2,2},{3,3}}
            : std::vector<std::pair<int,int>>{{0,0},{0,1},{1,0},{1,1}};
        for (int m : rows) for (int nt : tiles) for (auto layout : recipes) {
            const int a = layout.first, d = layout.second;
            auto s = base;
            s.m_tile = m; s.options.n_tile = nt; s.k_tile=tile_k;
            s.fused_io = c.dtype == RK_NPU_MM_F16 && c.host_type == RK_NPU_MM_HOST_F32 && !tile_k;
            s.options.a_layout = static_cast<rk_npu_mm_layout>(a);
            s.options.c_layout = static_cast<rk_npu_mm_layout>(d);
            if (m == 128 && !nt && !tile_k && a == 1 && d == 1 && !s.fused_io) continue;
            Plan p(nullptr, rk_npu_mm_plan_free);
            if (!make_plan(s, p)) result.push_back(s);
        }
    }
    return result;
}
struct CpuScope {
    int threads, old_threads = omp_get_max_threads(), old_dynamic = omp_get_dynamic();
    std::vector<cpu_set_t> saved;
    explicit CpuScope(uint64_t mask) : threads(__builtin_popcountll(mask)), saved(threads) {
        omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
        {
            auto& set = saved[omp_get_thread_num()];
            CPU_ZERO(&set);
            pthread_getaffinity_np(pthread_self(), sizeof(set), &set);
        }
        tune::pin_openmp_team(mask, threads);
        tune::set_thread_cpu_mask(mask);
    }
    ~CpuScope() {
#pragma omp parallel num_threads(threads)
        {
            auto& set = saved[omp_get_thread_num()];
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        omp_set_num_threads(old_threads);
        omp_set_dynamic(old_dynamic);
    }
};
std::vector<uint64_t> key(rk_npu_ctx* ctx, const rk_npu_mm_autotune_config& c) {
    std::vector<uint64_t> result{revision_for(c.dtype), ctx->driver_version, uint64_t(c.M), uint64_t(c.N), uint64_t(c.K),
        uint64_t(c.dtype), uint64_t(c.host_type), uint64_t(c.with_bias),
        c.allowed_npu_core_mask, c.allowed_cpu_core_mask, 0, // fixed split policy
        uint64_t(c.warmup), uint64_t(c.loops), uint64_t(c.repeats), c.timeout_ms};
    if(c.dtype==RK_NPU_MM_F16) {
        result.push_back(c.allow_k_tiling);
    }
    return result;
}
std::string path_for(const std::vector<uint64_t>& k, const char* directory) {
    tune::Hash hash;
    for (auto v : k) hash.mix(v);
    return tune::cache_path(hash.value, directory, "rk_npu_mm");
}
std::vector<uint64_t> partition_key(rk_npu_ctx* ctx,rk_npu_mm_autotune_config c,int tile) {
    c.M=0;auto result=key(ctx,c);result.push_back(tile);return result;
}
int save_partition(rk_npu_ctx* ctx,const rk_npu_mm_autotune_config& c,const char* directory,int tile) {
    if(!tile || c.dtype!=RK_NPU_MM_F16)return RK_NPU_OK;
    const auto identity=partition_key(ctx,c,tile);
    return tune::save_cache(path_for(identity,directory),[&](std::ostream& out) {
        out<<"RKMMK7";for(auto v:identity)out<<' '<<v;out<<'\n';
    });
}
void write_strategy(std::ostream& out, const rk_npu_mm_strategy& s) {
    out << s.M << ' ' << s.N << ' ' << s.K << ' ' << int(s.dtype) << ' ' << int(s.host_type)
        << ' ' << s.with_bias << ' ' << s.m_tile;
    if (s.dtype == RK_NPU_MM_F16) out << ' ' << s.task_m_tile << ' ' << s.fused_io << ' ' << s.k_tile;
    out << ' ' << int(s.options.a_layout)
        << ' ' << int(s.options.c_layout) << ' ' << s.options.n_tile << ' ' << s.options.split_k
        << ' ' << s.options.allowed_npu_core_mask << ' ' << s.options.timeout_ms
        << ' ' << s.cpu_core_mask << ' ' << s.cpu_threads << ' ' << s.tuning_revision
        << ' ' << s.total_us << ' ' << s.jitter_pct << ' ' << s.robust_us
        << ' ' << s.workspace_us << ' ' << s.weight_us << ' ' << s.max_abs_error << '\n';
}
bool read_strategy(std::istream& in, const rk_npu_mm_autotune_config& c, rk_npu_mm_strategy& s) {
    s={};
    int dtype, host, a, d;
    rk_npu_mm_options_init(&s.options);
    if (!(in >> s.M >> s.N >> s.K >> dtype >> host >> s.with_bias >> s.m_tile)) return false;
    if (c.dtype == RK_NPU_MM_F16) {
        if(!(in>>s.task_m_tile>>s.fused_io>>s.k_tile))return false;
    }
    if (!(in >> a >> d
        >> s.options.n_tile >> s.options.split_k >> s.options.allowed_npu_core_mask
        >> s.options.timeout_ms >> s.cpu_core_mask >> s.cpu_threads >> s.tuning_revision
        >> s.total_us >> s.jitter_pct >> s.robust_us >> s.workspace_us >> s.weight_us
        >> s.max_abs_error)) return false;
    s.dtype = static_cast<rk_npu_mm_dtype>(dtype);
    s.host_type = static_cast<rk_npu_mm_host_type>(host);
    s.options.a_layout = static_cast<rk_npu_mm_layout>(a);
    s.options.c_layout = static_cast<rk_npu_mm_layout>(d);
    if(s.tuning_revision!=revision_for(c.dtype))return false;
    if (s.M != c.M || s.N != c.N || s.K != c.K || s.dtype != c.dtype || s.host_type != c.host_type ||
        s.with_bias != c.with_bias || s.options.allowed_npu_core_mask != c.allowed_npu_core_mask ||
        s.options.timeout_ms != c.timeout_ms || s.cpu_core_mask != c.allowed_cpu_core_mask ||
        s.cpu_threads != __builtin_popcountll(c.allowed_cpu_core_mask)) return false;
    for (double v : {s.total_us, s.jitter_pct, s.robust_us, s.workspace_us, s.weight_us, s.max_abs_error})
        if (!std::isfinite(v) || v < 0) return false;
    if (s.total_us <= 0 || s.robust_us < s.total_us) return false;
    Plan plan(nullptr, rk_npu_mm_plan_free);
    if (make_plan(s, plan)) return false;
    const auto all = candidates(c);
    const auto caps=task_caps(s);
    return std::any_of(all.begin(), all.end(), [&](const auto& q) {
        return s.m_tile == q.m_tile && s.options.n_tile == q.options.n_tile &&
            s.options.a_layout == q.options.a_layout && s.options.c_layout == q.options.c_layout &&
            s.options.split_k == q.options.split_k && s.tuning_revision == revision_for(c.dtype) &&
            s.k_tile==q.k_tile &&
            (s.task_m_tile == 0 || (c.dtype == RK_NPU_MM_F16 &&
             std::binary_search(caps.begin(),caps.end(),s.task_m_tile)));
    });
}
int load_profile(rk_npu_ctx* ctx,const rk_npu_mm_autotune_config& c,const char* directory,
    rk_npu_mm_strategy& fastest,rk_npu_mm_strategy& stable) {
    const auto identity=key(ctx,c);
    return tune::load_cache(path_for(identity,directory),[&](std::istream& in) {
        std::string magic;if(!(in>>magic) || magic!="RKMM1")return false;
        for(auto expected:identity) {uint64_t value;if(!(in>>value) || value!=expected)return false;}
        return read_strategy(in,c,fastest) && read_strategy(in,c,stable);
    });
}
} // namespace

struct rk_npu_mm_tuned_workspace {
    rk_npu_mm_strategy strategy{};
    Plan plan{nullptr, rk_npu_mm_plan_free};
    Work work{nullptr, rk_npu_mm_workspace_free};
    std::vector<uint16_t> a, c;
    Plan collect_plan{nullptr,rk_npu_mm_plan_free};
    rk_npu_ctx* ctx=nullptr;
    rk_npu_mem device_a{},partials{};
    std::unique_ptr<detail::FloatReduce> reduce;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    ~rk_npu_mm_tuned_workspace() {
        reduce.reset();
        if(ctx) {rk_npu_mem_free(ctx,&device_a);rk_npu_mem_free(ctx,&partials);}
    }
};

extern "C" void rk_npu_mm_autotune_config_init(rk_npu_mm_autotune_config* c, int m, int n, int k) {
    if (!c) return;
    *c = {};
    c->M = m; c->N = n; c->K = k;
    c->dtype = RK_NPU_MM_F16; c->host_type = RK_NPU_MM_HOST_STORAGE;
    c->allowed_npu_core_mask = 7;
    c->allow_k_tiling=environment_k_mode();
    c->warmup = 1; c->loops = 3; c->repeats = 3; c->timeout_ms = 500;
}
extern "C" int rk_npu_mm_strategy_default(const rk_npu_mm_autotune_config* src, rk_npu_mm_strategy* out) {
    if (!out) return RK_NPU_ERR_PARAM;
    rk_npu_mm_autotune_config c;
    const int rc = normalize(src, c);
    if (rc) return rc;
    auto s = baseline(c);
    Plan p(nullptr, rk_npu_mm_plan_free);
    const int result = make_plan(s, p);
    if (!result) *out = s;
    return result;
}
extern "C" int rk_npu_mm_tuned_packed_b_create(rk_npu_iommu_domain* domain,
    const rk_npu_mm_strategy* s, const uint16_t* b, uint64_t bytes, rk_npu_mm_packed_b** out) {
    if (!out) return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!s) return RK_NPU_ERR_PARAM;
    Plan p(nullptr, rk_npu_mm_plan_free);
    int rc = make_plan(*s, p);
    if (!rc) rc = rk_npu_mm_packed_b_create(domain, p.get(), b, bytes, out);
    return rc;
}
extern "C" int rk_npu_mm_tuned_workspace_create(rk_npu_iommu_domain* domain,
    const rk_npu_mm_strategy* s, rk_npu_mm_tuned_workspace** out) {
    if (!out) return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!s) return RK_NPU_ERR_PARAM;
    try {
        TunedWork w(new rk_npu_mm_tuned_workspace, rk_npu_mm_tuned_workspace_free);
        w->strategy = *s;
        int rc = make_plan(*s, w->plan);
        if (rc) return rc;
        rk_npu_mm_workspace* raw = nullptr;
        rc = rk_npu_mm_workspace_create(domain, w->plan.get(),
            s->k_tile?RK_NPU_MM_DEVICE_ONLY:RK_NPU_MM_HOST_PACKED_B,&raw);
        w->work.reset(raw);
        if (rc) return rc;
        if(s->k_tile) {
            w->ctx=domain->ctx;
            rk_npu_mm_info info{};rk_npu_mm_plan_get_info(w->plan.get(),&info);
            rc=rk_npu_mem_alloc(domain,info.device.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&w->device_a);
            if(rc)return rc;
            if(!rc)rc=rk_npu_mem_alloc(domain,info.device.output_bytes,RK_NPU_MEM_DATA_DEFAULT,&w->partials);
            if(rc)return rc;
            w->reduce=std::make_unique<detail::FloatReduce>();
            rc=w->reduce->prepare(domain,w->partials,info.split_count,
                info.device.output_bytes/info.split_count,s->options.allowed_npu_core_mask,s->options.timeout_ms);
            if(rc)return rc;
            auto desc=info.desc;desc.K=info.part_k;desc.lda=desc.K;desc.batch_stride_a=uint64_t(desc.M)*desc.K;
            desc.ldb=desc.N;desc.batch_stride_b=uint64_t(desc.K)*desc.N;
            auto options=s->options;options.split_k=0;
            rk_npu_mm_plan* collect=nullptr;
            rc=rk_npu_mm_plan_create(&desc,&options,&collect);w->collect_plan.reset(collect);
            if(rc)return rc;
        }
        if (!s->fused_io) {
            w->a.resize(size_t(s->m_tile) * s->K);
            w->c.resize(size_t(s->m_tile) * s->N);
        }
        *out = w.release();
        return RK_NPU_OK;
    } catch (const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
}
extern "C" void rk_npu_mm_tuned_workspace_free(rk_npu_mm_tuned_workspace* w) { delete w; }
extern "C" int rk_npu_mm_tuned_run(rk_npu_mm_tuned_workspace* w, const void* input, uint64_t ab,
    const rk_npu_mm_packed_b* weight, const float* bias, void* output, uint64_t cb) {
    if (!w || !input || !output || !weight) return RK_NPU_ERR_PARAM;
    if (w->busy.test_and_set(std::memory_order_acquire)) return RK_NPU_ERR_BUSY;
    struct Unlock { rk_npu_mm_tuned_workspace* w; ~Unlock() { w->busy.clear(std::memory_order_release); } } guard{w};
    const auto& s = w->strategy;
    const int bytes = s.host_type == RK_NPU_MM_HOST_F32 ? 4 : 2;
    const uint64_t na = uint64_t(s.M) * s.K * bytes, nc = uint64_t(s.M) * s.N * bytes;
    const auto ai = reinterpret_cast<uintptr_t>(input), ci = reinterpret_cast<uintptr_t>(output);
    if (ai % bytes || ci % bytes || ai > UINTPTR_MAX - na || ci > UINTPTR_MAX - nc ||
        (ai < ci + nc && ci < ai + na) || bool(bias) != bool(s.with_bias)) return RK_NPU_ERR_PARAM;
    if (ab < na || cb < nc) return RK_NPU_ERR_NOMEM;
    for (int first = 0; first < s.M; first += s.m_tile) {
        const int rows = std::min(s.m_tile, s.M - first);
        const size_t count = size_t(rows) * s.K;
        if (s.fused_io) {
            const int rc = rk_npu_mm_run_packed_b_f32(w->work.get(),
                static_cast<const float*>(input) + size_t(first) * s.K, weight, bias,
                static_cast<float*>(output) + size_t(first) * s.N, rows);
            if (rc) return rc;
            continue;
        }
        const uint16_t* a;
        if (s.host_type == RK_NPU_MM_HOST_F32) {
            narrow(static_cast<const float*>(input) + size_t(first) * s.K,
                   w->a.data(), count, s.dtype);
            a = w->a.data();
        } else {
            a = static_cast<const uint16_t*>(input) + size_t(first) * s.K;
            if (rows < s.m_tile) {
                std::memcpy(w->a.data(), a, count * 2);
                a = w->a.data();
            }
        }
        if (rows < s.m_tile) std::fill(w->a.begin() + count, w->a.end(), 0);
        int rc=0;
        if(s.k_tile) {
            rc=rk_npu_mm_pack_a(w->plan.get(),a,w->a.size()*2,&w->device_a);
            if(!rc)rc=rk_npu_mem_sync(w->ctx,&w->device_a,RK_NPU_SYNC_TO_DEVICE);
            if(!rc)rc=rk_npu_mm_run_device_packed_b(w->work.get(),&w->device_a,weight,&w->partials);
            if(!rc)rc=w->reduce->run();
            if(!rc)rc=rk_npu_mem_sync(w->ctx,&w->reduce->output(),RK_NPU_SYNC_FROM_DEVICE);
            if(!rc)rc=rk_npu_mm_unpack_c(w->collect_plan.get(),&w->reduce->output(),w->c.data(),w->c.size()*2);
        } else rc = rk_npu_mm_run_packed_b(w->work.get(), a, w->a.size() * 2,
            weight, w->c.data(), w->c.size() * 2);
        if (rc) return rc;
        const size_t outputs = size_t(rows) * s.N;
        if (s.host_type == RK_NPU_MM_HOST_F32) {
            collect(w->c.data(), static_cast<float*>(output) + size_t(first) * s.N,
                    outputs, s.N, bias, s.dtype);
        } else if(s.with_bias) {
            storage_bias(w->c.data(),static_cast<uint16_t*>(output)+size_t(first)*s.N,
                         outputs,s.N,bias,s.dtype);
        } else {
            std::memcpy(static_cast<uint16_t*>(output) + size_t(first) * s.N, w->c.data(), outputs * 2);
        }
    }
    return RK_NPU_OK;
}

static int autotune_impl(rk_npu_iommu_domain* domain,
    const rk_npu_mm_autotune_config* src, rk_npu_mm_strategy* fastest, rk_npu_mm_strategy* stable) {
    if (!domain || !domain->ctx || (!fastest && !stable)) return RK_NPU_ERR_PARAM;
    rk_npu_mm_autotune_config c;
    int rc = normalize(src, c);
    if (rc) return rc;
    try {
        CpuScope affinity(c.allowed_cpu_core_mask);
        // Eight independent per-row coefficients and nonperiodic B give an
        // O(8*M*N + K*N) full-output oracle, independent of device layouts.
        std::vector<float> coefficients(size_t(c.M) * 8), sums(size_t(c.N) * 8, 0.f);
        std::vector<uint16_t> a(size_t(c.M) * c.K), b(size_t(c.K) * c.N), out(size_t(c.M) * c.N);
        std::vector<float> af(a.size()), cf(out.size()), reference(out.size()), bias(c.N);
        uint32_t rng = 0x6d6d4631;
        auto random = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        for (auto& v : coefficients) v = decode(encode((int(random() % 127) - 63) / 256.f, c.dtype), c.dtype);
        for (int m = 0; m < c.M; ++m) for (int k = 0; k < c.K; ++k) {
            const size_t j = size_t(m) * c.K + k;
            af[j] = coefficients[size_t(m) * 8 + k % 8]; a[j] = encode(af[j], c.dtype);
        }
        for (int k = 0; k < c.K; ++k) for (int n = 0; n < c.N; ++n) {
            const auto v = encode((int(random() % 127) - 63) / 512.f, c.dtype);
            b[size_t(k) * c.N + n] = v; sums[size_t(k % 8) * c.N + n] += decode(v, c.dtype);
        }
        for (int n = 0; n < c.N; ++n) bias[n] = c.with_bias ? (int(random() % 63) - 31) / 1024.f : 0.f;
        for (int m = 0; m < c.M; ++m) for (int n = 0; n < c.N; ++n) {
            double value = 0;
            for (int t = 0; t < 8; ++t) value += double(coefficients[size_t(m) * 8 + t]) * sums[size_t(t) * c.N + n];
            reference[size_t(m) * c.N + n] = float(value) + bias[n];
        }
        rk_npu_mm_strategy best{}, robust{};
        bool found = false;
        struct SharedWeight {Weight value{nullptr,rk_npu_mm_packed_b_free};double us=0;};
        std::map<std::pair<int,int>,SharedWeight> common_weights;
        auto search = candidates(c);
        const size_t layout_count = search.size();
        std::vector<rk_npu_mm_strategy> per_layout;
        for (size_t candidate = 0; candidate < search.size() || (candidate == layout_count && c.dtype == RK_NPU_MM_F16); ++candidate) {
            if (candidate == layout_count && c.dtype == RK_NPU_MM_F16) {
                // Equal task counts can have different row/core balance, e.g.
                // 160+96 versus 128+128. Keep distinct legal row caps.
                search.push_back(baseline(c));
                for (const auto& seed : per_layout) {
                    for (int io : {seed.fused_io, 1 - seed.fused_io}) {
                        if (io && (c.host_type != RK_NPU_MM_HOST_F32 || seed.k_tile)) continue;
                        for (int cap : task_caps(seed)) {
                            auto trial = seed; trial.task_m_tile = cap; trial.fused_io = io;
                            Plan plan(nullptr, rk_npu_mm_plan_free);
                            if (make_plan(trial, plan)) continue;
                            search.push_back(trial);
                        }
                    }
                }
            }
            if (candidate >= search.size()) break;
            auto s = search[candidate];
            const bool final = c.dtype != RK_NPU_MM_F16 || candidate >= layout_count;
            const auto start = Clock::now();
            rk_npu_mm_tuned_workspace* raw = nullptr;
            rc = rk_npu_mm_tuned_workspace_create(domain, &s, &raw);
            TunedWork work(raw, rk_npu_mm_tuned_workspace_free);
            if (rc) return rc;
            const auto prepared = Clock::now();
            rk_npu_mm_packed_b* wb = nullptr;
            Weight weight(nullptr, rk_npu_mm_packed_b_free);
            const bool common = c.dtype == RK_NPU_MM_F16;
            rk_npu_mm_info wi{};rk_npu_mm_plan_get_info(work->plan.get(),&wi);
            const auto partition=std::make_pair(wi.split_count,wi.part_k);
            if (common && common_weights.count(partition)) {
                wb = common_weights.at(partition).value.get();
                s.weight_us = common_weights.at(partition).us;
            } else {
                rc = rk_npu_mm_tuned_packed_b_create(domain, &s, b.data(), b.size() * 2, &wb);
                if (rc) return rc;
                s.weight_us = us(prepared, Clock::now());
                if (common) {auto& entry=common_weights[partition];entry.value.reset(wb);entry.us=s.weight_us;}
                else weight.reset(wb);
            }
            s.workspace_us = us(start, prepared);
            double max_error = 0;
            bool valid = true;
            auto validate = [&]() {
                const float atol = c.dtype == RK_NPU_MM_BF16 ? 0.015f : 0.003f;
                const float rtol = c.dtype == RK_NPU_MM_BF16 ? 0.02f : 0.005f;
                for (size_t j = 0; j < out.size(); ++j) {
                    const float v = c.host_type == RK_NPU_MM_HOST_F32 ? cf[j] : decode(out[j], c.dtype);
                    const double error = std::abs(double(v) - reference[j]);
                    if (!std::isfinite(v) || error > atol + rtol * std::abs(reference[j])) valid = false;
                    max_error = std::max(max_error, error);
                }
                return valid;
            };
            int sample = 0;
            auto run = [&](tune::Sample& timing) {
                const auto begin = Clock::now();
                const int result = rk_npu_mm_tuned_run(work.get(),
                    c.host_type == RK_NPU_MM_HOST_F32 ? static_cast<void*>(af.data()) : a.data(),
                    a.size() * (c.host_type == RK_NPU_MM_HOST_F32 ? 4 : 2), wb,
                    c.with_bias ? bias.data() : nullptr,
                    c.host_type == RK_NPU_MM_HOST_F32 ? static_cast<void*>(cf.data()) : out.data(),
                    out.size() * (c.host_type == RK_NPU_MM_HOST_F32 ? 4 : 2));
                timing.total_us = us(begin, Clock::now());
                if (c.verbose >= 2) std::fprintf(stderr, "MM sample %d M=%d tile=%d Ntile=%d A=%d C=%d split=%d us=%.3f\n",
                    sample++, c.M, s.m_tile, s.options.n_tile, int(s.options.a_layout), int(s.options.c_layout), s.options.split_k, timing.total_us);
                return result;
            };
            tune::Score score;
            rc = tune::measure({c.warmup, final ? c.loops : 1, final ? c.repeats : 1}, run, validate, score);
            if (rc && valid) return rc; // device/submit failures stop the search
            if (!valid) {
                if (c.verbose) std::fprintf(stderr, "MM rejected tile=%d Ntile=%d A=%d C=%d split=%d max_abs=%.6g\n",
                    s.m_tile, s.options.n_tile, int(s.options.a_layout), int(s.options.c_layout), s.options.split_k, max_error);
                continue;
            }
            s.total_us = score.total_us; s.robust_us = score.robust_us;
            s.jitter_pct = score.jitter_pct; s.max_abs_error = max_error;
            if (c.verbose) std::fprintf(stderr, "MM phase=%s M=%d K=%d N=%d dtype=%d tile=%d Ntile=%d A=%d C=%d split=%d taskM=%d fused=%d Ktile=%d npu_reduce=%d total=%.3f robust=%.3f max_abs=%.6g\n",
                final ? "final" : "coarse", c.M, c.K, c.N, int(c.dtype), s.m_tile, s.options.n_tile, int(s.options.a_layout), int(s.options.c_layout),
                s.options.split_k, s.task_m_tile, s.fused_io,s.k_tile,int(s.k_tile>0),s.total_us, s.robust_us, s.max_abs_error);
            if (!final) {
                auto it = std::find_if(per_layout.begin(), per_layout.end(), [&](const auto& q) {
                    return q.options.a_layout == s.options.a_layout && q.options.c_layout == s.options.c_layout &&
                        q.k_tile==s.k_tile && q.options.split_k==s.options.split_k;
                });
                if (it == per_layout.end()) per_layout.push_back(s);
                else if (s.total_us < it->total_us) *it = s;
            }
            if (!final) continue;
            if (!found || s.total_us < best.total_us) best = s;
            if (!found || s.robust_us < robust.robust_us) robust = s;
            found = true;
        }
        if (!found) return RK_NPU_ERR_PARAM;
        if (fastest) *fastest = best;
        if (stable) *stable = robust;
        return RK_NPU_OK;
    } catch (const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
}
extern "C" int rk_npu_mm_autotune(rk_npu_iommu_domain* domain,
    const rk_npu_mm_autotune_config* c,rk_npu_mm_strategy* fastest,rk_npu_mm_strategy* stable) {
    return autotune_impl(domain,c,fastest,stable);
}
extern "C" int rk_npu_mm_autotune_cache_load(rk_npu_ctx* ctx,
    const rk_npu_mm_autotune_config* src, const char* directory,
    rk_npu_mm_strategy* fastest, rk_npu_mm_strategy* stable) {
    if (!ctx || (!fastest && !stable)) return RK_NPU_ERR_PARAM;
    rk_npu_mm_autotune_config c;
    int rc = normalize(src, c);
    if (rc) return rc;
    rk_npu_mm_strategy f{}, s{};
    rc=load_profile(ctx,c,directory,f,s);
    if (!rc) { if (fastest) *fastest = f; if (stable) *stable = s; }
    return rc;
}
extern "C" int rk_npu_mm_autotune_cached(rk_npu_iommu_domain* domain,
    const rk_npu_mm_autotune_config* src, const char* directory, int refresh,
    rk_npu_mm_strategy* fastest, rk_npu_mm_strategy* stable, int* hit) {
    if (hit) *hit = 0;
    if (!domain || !domain->ctx || (!fastest && !stable)) return RK_NPU_ERR_PARAM;
    rk_npu_mm_autotune_config c;
    int rc = normalize(src, c);
    if (rc) return rc;
    rk_npu_mm_strategy f{}, s{};
    if (!refresh) {
        rc = rk_npu_mm_autotune_cache_load(domain->ctx, &c, directory, &f, &s);
        if (!rc) { if (hit) *hit = 1; }
        else if (rc != RK_NPU_ERR_CACHE_MISS) return rc;
    }
    if (refresh || rc == RK_NPU_ERR_CACHE_MISS) {
        rc=autotune_impl(domain,&c,&f,&s);
        if (rc) return rc;
        const auto identity = key(domain->ctx, c);
        rc = tune::save_cache(path_for(identity, directory), [&](std::ostream& out) {
            out << "RKMM1";
            for (auto v : identity) out << ' ' << v;
            out << '\n' << std::setprecision(17);
            write_strategy(out, f); write_strategy(out, s);
        });
        if (rc) return rc;
    }
    rc=save_partition(domain->ctx,c,directory,f.k_tile);
    if(rc)return rc;
    if (fastest) *fastest = f;
    if (stable) *stable = s;
    return RK_NPU_OK;
}
static int load_weight_tiles(rk_npu_ctx* ctx,
    const rk_npu_mm_autotune_config* src,const char* directory,int* tiles,int capacity,int* count) {
    if(!ctx || !count || capacity<0 || (capacity && !tiles))return RK_NPU_ERR_PARAM;
    *count=0;rk_npu_mm_autotune_config c;const int rc=normalize(src,c);if(rc)return rc;
    if(c.dtype!=RK_NPU_MM_F16 || !c.allow_k_tiling)return RK_NPU_OK;
    for(int tile:k_tiles(c.K)) {
        const auto identity=partition_key(ctx,c,tile);
        const int status=tune::load_cache(path_for(identity,directory),[&](std::istream& in) {
            std::string magic;if(!(in>>magic) || magic!="RKMMK7")return false;
            for(auto v:identity) {uint64_t got;if(!(in>>got) || got!=v)return false;}
            return true;
        });
        if(status==RK_NPU_ERR_CACHE_MISS)continue;
        if(status)return status;
        if(*count<capacity)tiles[*count]=tile;
        ++*count;
    }
    return *count>capacity?RK_NPU_ERR_NOMEM:RK_NPU_OK;
}
struct rk_npu_mm_tuned_weights {
    int N=0,K=0;rk_npu_mm_dtype dtype=RK_NPU_MM_F16;
    std::map<std::pair<int,int>,Weight> variants;
};
namespace {
std::pair<int,int> weight_partition(const rk_npu_mm_strategy& s) {
    if(s.k_tile>0)return {ceil_div(s.K,s.k_tile),s.k_tile};
    return {1,s.K};
}
}
extern "C" int rk_npu_mm_tuned_weights_create(rk_npu_iommu_domain* domain,
    const rk_npu_mm_autotune_config* src,const uint16_t* b,uint64_t bytes,const char* directory,
    rk_npu_mm_tuned_weights** out) {
    if(!out)return RK_NPU_ERR_PARAM;*out=nullptr;
    if(!domain || !domain->ctx || !b)return RK_NPU_ERR_PARAM;
    rk_npu_mm_autotune_config c;int rc=normalize(src,c);if(rc)return rc;
    try {
        auto weights=std::make_unique<rk_npu_mm_tuned_weights>();
        weights->N=c.N;weights->K=c.K;weights->dtype=c.dtype;
        std::vector<rk_npu_mm_strategy> strategies{baseline(c)};
        if(directory) {
            int tiles[32]{},count=0;
            rc=load_weight_tiles(domain->ctx,&c,directory,tiles,32,&count);if(rc)return rc;
            for(int j=0;j<count;++j) {auto s=baseline(c);s.k_tile=tiles[j];strategies.push_back(s);}
        }
        for(const auto& s:strategies) {
            const auto partition=weight_partition(s);
            if(weights->variants.count(partition))continue;
            rk_npu_mm_packed_b* raw=nullptr;
            rc=rk_npu_mm_tuned_packed_b_create(domain,&s,b,bytes,&raw);if(rc)return rc;
            weights->variants.emplace(partition,Weight(raw,rk_npu_mm_packed_b_free));
        }
        *out=weights.release();return RK_NPU_OK;
    } catch(const std::bad_alloc&) {return RK_NPU_ERR_NOMEM;}
}
extern "C" void rk_npu_mm_tuned_weights_free(rk_npu_mm_tuned_weights* weights) {delete weights;}
extern "C" int rk_npu_mm_tuned_weights_match(const rk_npu_mm_tuned_weights* weights,const rk_npu_mm_strategy* s) {
    if(!weights || !s || s->options.split_k || s->k_tile<0 || s->K!=weights->K || s->N!=weights->N || s->dtype!=weights->dtype)
        return RK_NPU_ERR_PARAM;
    return weights->variants.count(weight_partition(*s))?RK_NPU_OK:RK_NPU_ERR_CACHE_MISS;
}
extern "C" int rk_npu_mm_tuned_run_weights(rk_npu_mm_tuned_workspace* work,
    const void* a,uint64_t ab,const rk_npu_mm_tuned_weights* weights,const float* bias,void* c,uint64_t cb) {
    if(!work)return RK_NPU_ERR_PARAM;
    const int rc=rk_npu_mm_tuned_weights_match(weights,&work->strategy);if(rc)return rc;
    const auto& b=weights->variants.at(weight_partition(work->strategy));
    return rk_npu_mm_tuned_run(work,a,ab,b.get(),bias,c,cb);
}

extern "C" int rk_npu_mm_gelu_run_weights(rk_npu_mm_gelu_workspace* work,
    const uint16_t* a, uint64_t bytes, const rk_npu_mm_tuned_weights* weights,
    const float* bias, rk_npu_mm_gelu_view* view) {
    if (view) *view = {};
    if (!weights || weights->dtype != RK_NPU_MM_F16) return RK_NPU_ERR_PARAM;
    const auto found = weights->variants.find({1, weights->K});
    if (found == weights->variants.end()) return RK_NPU_ERR_CACHE_MISS;
    return rk_npu_mm_gelu_run(work, a, bytes, found->second.get(), bias, view);
}

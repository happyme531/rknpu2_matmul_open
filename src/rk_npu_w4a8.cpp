#include "rk_npu_w4a8.h"
#include "rk_npu_w4a8_internal.h"
#include "rk_npu_w4a8_tune_internal.h"
#include "rk_npu_i4_executor.h"
#include "rk_npu_w4a8_cpu.h"
#include "rk_npu_w4a8_chain.h"
#include "rk_npu_w4a8_npu_reduce.h"
#include "rk_npu_trace.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <memory>
#include <vector>

struct rk_npu_w4a8_weights {
    rknpu2_matmul_open::detail::I4Weights packed;
    std::vector<float> scales;
    std::vector<int32_t> correction;
    rk_npu_w4a8_weight_info info{};
    rknpu2_matmul_open::w4_tune::Profile profile;
};
struct rk_npu_w4a8_workspace {
    rk_npu_w4a8_config config{};
    uint64_t ctx_id = 0;
    uint32_t domain_id = 0;
    std::vector<float> scales, inverse;
    std::vector<int32_t> accumulator;
    std::vector<uint32_t> row_maxima;
    std::vector<rknpu2_matmul_open::detail::W4A8SwiGLUPair> swiglu_pairs;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    rknpu2_matmul_open::detail::I4Executor executor;
    std::unique_ptr<rknpu2_matmul_open::detail::W4A8Chain> chain;
    rknpu2_matmul_open::detail::W4A8NpuReduce reducer;
    rknpu2_matmul_open::detail::I4InputLayout layout = rknpu2_matmul_open::detail::I4InputLayout::Native;
    bool npu_reduce = false;
};

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
bool valid_shape(int K, int N) { return K > 0 && K <= INT32_MAX / 1024 && N > 0; }
int options_config(const rk_npu_w4a8_options* opt, int M, rknpu2_matmul_open::detail::I4InputLayout& layout, bool& npu) {
    layout=rknpu2_matmul_open::detail::I4InputLayout::Native; npu=false;
    if(!opt) return RK_NPU_OK;
    if(opt->struct_size!=sizeof(*opt) ||
       (opt->input_layout!=RK_NPU_W4A8_INPUT_NATIVE && opt->input_layout!=RK_NPU_W4A8_INPUT_PANEL8) ||
       (opt->reduce_backend!=RK_NPU_W4A8_REDUCE_CPU && opt->reduce_backend!=RK_NPU_W4A8_REDUCE_NPU))
        return RK_NPU_ERR_PARAM;
    layout=opt->input_layout==RK_NPU_W4A8_INPUT_PANEL8 ? rknpu2_matmul_open::detail::I4InputLayout::Panel8 : rknpu2_matmul_open::detail::I4InputLayout::Native;
    npu=opt->reduce_backend==RK_NPU_W4A8_REDUCE_NPU && M>16;
    return RK_NPU_OK;
}
int backend_config(const rk_npu_w4a8_config &cfg, rk_npu_i4_config &out,
                   rknpu2_matmul_open::detail::I4InputLayout layout = rknpu2_matmul_open::detail::I4InputLayout::Native) {
    if (!valid_shape(cfg.K, cfg.N) || cfg.M < 1 || cfg.M > INT_MAX / 2 || cfg.m_tile < 1 ||
        cfg.m_tile > (layout == rknpu2_matmul_open::detail::I4InputLayout::Panel8 ? 256 : 64) ||
        cfg.pipeline<RK_NPU_W4A8_SERIAL || cfg.pipeline>RK_NPU_W4A8_CHAIN)
        return RK_NPU_ERR_PARAM;
    out = {cfg.M * 2,         cfg.N,          cfg.K,        cfg.m_tile * 2, cfg.k_tile, cfg.n_tile,
           cfg.npu_core_mask, cfg.timeout_ms, cfg.pipeline==RK_NPU_W4A8_CHAIN?0:cfg.pipeline, cfg.cpu_threads};
    return RK_NPU_OK;
}

class WorkspaceLock {
public:
    explicit WorkspaceLock(rk_npu_w4a8_workspace &ws)
        : busy_(ws.busy), held_(!busy_.test_and_set(std::memory_order_acquire)) {}
    ~WorkspaceLock() { if (held_) busy_.clear(std::memory_order_release); }
    explicit operator bool() const { return held_; }
    WorkspaceLock(const WorkspaceLock&) = delete;
    WorkspaceLock& operator=(const WorkspaceLock&) = delete;
private:
    std::atomic_flag &busy_;
    bool held_;
};

int validate(const rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *weights,
              const void *A, const void *C, bool swiglu) {
    if (!ws || !weights || !A || !C || A == C ||
        (swiglu && (ws->npu_reduce || ws->config.N % 64))) return RK_NPU_ERR_PARAM;
    const auto &mem = weights->packed.memory();
    const auto &wc = weights->packed.config();
    if (mem.ctx_id != ws->ctx_id || mem.iommu_domain_id != ws->domain_id)
        return RK_NPU_ERR_DOMAIN;
    if (wc.K != ws->config.K || wc.N != ws->config.N || wc.k_tile != ws->config.k_tile)
        return RK_NPU_ERR_PARAM;
    return RK_NPU_OK;
}

// Caller holds workspace ownership. Prescaled input is private to the FFN
// entrypoint, which has already checked every activation while collecting maxabs.
int run_locked(rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *weights,
        const void *A, void *C, bool half, rk_npu_w4a8_timings *metrics,
        bool swiglu = false, bool collect = false,
        const rk_npu_w4a8_workspace *scale_source = nullptr) {
    const auto &c = ws->config;
    RK_NPU_TRACE_EVENT("rknpu", "w4a8.run", "M", c.M, "K", c.K, "N", c.N,
        "m_tile", c.m_tile, "k_tile", c.k_tile, "n_tile", c.n_tile,
        "pipeline", c.pipeline, "cpu_threads", c.cpu_threads,
        "waves", (c.K+c.k_tile-1)/c.k_tile,
        "npu_mask", c.npu_core_mask, "input_layout", int(ws->layout),
        "npu_reduce", ws->npu_reduce, "fp16_input", half,
        "swiglu_output", swiglu, "output_N", swiglu ? c.N / 2 : c.N);
    const auto start = Clock::now();
    rk_npu_w4a8_timings t{};
    // The FFN entrypoint holds both workspaces, so upstream scale metadata
    // remains immutable throughout Down and can be consumed without a copy.
    const float *scales = scale_source ? scale_source->scales.data() : ws->scales.data();
    const float *inverse = scale_source ? scale_source->inverse.data() : ws->inverse.data();
    int rc = RK_NPU_OK;
    if (!scale_source) {
        RK_NPU_TRACE_EVENT("rknpu.detail", "w4a8.activation_scan", "M", c.M, "K", c.K);
        rc = rknpu2_matmul_open::detail::w4a8_scales(A, half, c.M, c.K, c.cpu_threads, ws->scales.data(), ws->inverse.data());
        t.activation_scan_us = elapsed(start);
    }
    if (!rc) {
        rknpu2_matmul_open::detail::W4A8Input input{A, half, c.M, c.K, scales, inverse, c.cpu_threads, ws->layout};
        rknpu2_matmul_open::detail::W4A8Reduction reduction{ws->accumulator.data(), weights->correction.data(),
                                       scales, weights->scales.data(), C, half,
                                       swiglu ? &ws->swiglu_pairs : nullptr,
                                       collect ? ws->row_maxima.data() : nullptr, c.M + 16};
        if (collect) std::fill(ws->row_maxima.begin(), ws->row_maxima.end(), 0u);
        rk_npu_i4_timings inner{};
        if(ws->chain) rc=ws->chain->run(weights->packed,input,reduction,inner);
        else rc = ws->executor.run(weights->packed, rknpu2_matmul_open::detail::w4a8_pack, &input, nullptr, &inner,
                              ws->npu_reduce ? nullptr : rknpu2_matmul_open::detail::w4a8_reduce, &reduction,
                              swiglu && ws->executor.plan().wave_count() == 2
                                  ? rknpu2_matmul_open::detail::w4a8_reduce_two : nullptr);
        t.quant_pack_us = inner.pack_us;
        t.sync_us = inner.sync_us;
        t.submit_us = inner.submit_us;
        t.reduce_us = inner.reduce_us;
        if (!rc && ws->npu_reduce) rc = ws->reducer.run(reduction,t);
        if (!rc && collect) {
            const auto begin = Clock::now();
            rc = rknpu2_matmul_open::detail::w4a8_finish_scales(c.M, c.cpu_threads,
                ws->row_maxima.data(), c.M + 16, ws->scales.data(), ws->inverse.data());
            t.reduce_us += elapsed(begin);
        }
    }
    t.total_us = elapsed(start);
    if (metrics)
        *metrics = t;
    return rc;
}
int run(rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *weights,
        const void *A, void *C, bool half, rk_npu_w4a8_timings *metrics, bool swiglu = false) {
    const int rc = validate(ws, weights, A, C, swiglu);
    if (rc) return rc;
    WorkspaceLock lock(*ws);
    if (!lock) return RK_NPU_ERR_BUSY;
    return run_locked(ws, weights, A, C, half, metrics, swiglu);
}
} // namespace

extern "C" void rk_npu_w4a8_config_init(rk_npu_w4a8_config *cfg, int M, int N, int K) {
    if (cfg)
        *cfg = {M, N, K, 64, 480, 1024, 1, 500, 1, 1};
}
extern "C" void rk_npu_w4a8_options_init(rk_npu_w4a8_options* options) {
    if(options) *options={sizeof(*options),RK_NPU_W4A8_INPUT_NATIVE,RK_NPU_W4A8_REDUCE_CPU};
}
extern "C" int rk_npu_w4a8_memory_query_ex(const rk_npu_w4a8_config* cfg,
    const rk_npu_w4a8_options* options, rk_npu_w4a8_memory_info_ex* out) {
    if(!cfg || !out) return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::detail::I4InputLayout layout; bool npu;
    int rc=options_config(options,cfg->M,layout,npu); if(rc) return rc;
    rk_npu_i4_config inner{};
    rc=backend_config(*cfg,inner,layout); if(rc) return rc;
    rk_npu_w4a8_memory_info_ex info{};
    info.effective_reduce_backend=npu?RK_NPU_W4A8_REDUCE_NPU:RK_NPU_W4A8_REDUCE_CPU;
    if(cfg->pipeline==RK_NPU_W4A8_CHAIN) {
        if(npu)return RK_NPU_ERR_PARAM;
        rc=rknpu2_matmul_open::detail::W4A8Chain::query(inner,layout,info.base.backend);
    } else rc=rknpu2_matmul_open::detail::query_i4(inner,info.base.backend,layout);
    if(rc) return rc;
    info.base.weight_metadata_bytes=uint64_t(cfg->N)*(sizeof(float)+sizeof(int32_t));
    info.base.cpu_scratch_bytes=uint64_t(cfg->M)*sizeof(float)*2;
    if(npu) {
        try { rc=rknpu2_matmul_open::detail::W4A8NpuReduce::query(inner,layout,info); }
        catch(...) { return RK_NPU_ERR_NOMEM; }
        if(rc) return rc;
    } else {
        info.base.cpu_scratch_bytes+=uint64_t(cfg->M)*((uint64_t(cfg->N)+63)/64*64)*sizeof(int32_t);
        if (cfg->N % 64 == 0)
            info.base.cpu_scratch_bytes+=(uint64_t(cfg->M)+16)*cfg->cpu_threads*sizeof(uint32_t);
    }
    if(info.base.cpu_scratch_bytes>SIZE_MAX || info.base.weight_metadata_bytes>SIZE_MAX)
        return RK_NPU_ERR_NOMEM;
    info.workspace_bytes=info.base.backend.input_bytes+info.base.backend.partial_bytes+
        info.base.backend.control_bytes+info.base.cpu_scratch_bytes+info.reduction_output_bytes+
        info.reduction_weight_bytes+info.reduction_control_bytes;
    *out=info; return RK_NPU_OK;
}
extern "C" int rk_npu_w4a8_memory_query(const rk_npu_w4a8_config* cfg, rk_npu_w4a8_memory_info* out) {
    if(!out) return RK_NPU_ERR_PARAM;
    rk_npu_w4a8_memory_info_ex info{};
    const int rc=rk_npu_w4a8_memory_query_ex(cfg,nullptr,&info);
    if(!rc) *out=info.base;
    return rc;
}
rk_npu_w4a8_weights* rknpu2_matmul_open::detail::create_w4a8_weights(rk_npu_iommu_domain* domain,
    const rk_npu_w4a8_weight_info& info, const int8_t* B, const float* scales,
    const w4_tune::Profile* profile) {
    const auto& actual=info.config;
    if(!domain || !B || !scales || !valid_shape(actual.K,actual.N)) return nullptr;
    try {
        uint64_t bytes = 0;
        if (rk_npu_i4_weights_memory_query(&actual, &bytes))
            return nullptr;
        auto w = std::make_unique<rk_npu_w4a8_weights>();
        if(profile) w->profile=*profile;
        else {
            rk_npu_w4a8_autotune_config c;
            rk_npu_w4a8_autotune_config_init(&c,1,actual.N,actual.K);
            // Legacy explicit-Ktile factories allow a tile wider than K.
            // Capture the full proof independently; packed.prepare below
            // still validates the actual requested partition as before.
            c.required_k_tile=0;c.k_tile_multiplier=info.multiplier;
            if(w4_tune::profile(c,B,scales,w->profile)) return nullptr;
        }
        w->info = info;
        w->scales.assign(scales, scales + actual.N);
        w->correction.assign(actual.N, 0);
        for (float s : w->scales)
            if (!std::isfinite(s) || !(s > 0))
                return nullptr;
        // Immutable integer correction, independent of token scales. No
        // external bias and no duplicate unpacked weight matrix are retained.
        for (int k = 0; k < actual.K; ++k)
            for (int n = 0; n < actual.N; ++n) {
                const int v = B[size_t(k) * actual.N + n];
                if (v < -8 || v > 7)
                    return nullptr;
                w->correction[n] += 8 * v; // |sum| <= 64*K under valid_shape()
            }
        if (w->packed.prepare(domain, actual, B, w->info.bound_relaxed != 0))
            return nullptr;
        return w.release();
    } catch (...) { return nullptr; }
}

extern "C" rk_npu_w4a8_weights *rk_npu_w4a8_weights_create(rk_npu_iommu_domain *domain,
                                                           const rk_npu_i4_weight_config *cfg,
                                                           const int8_t *B, const float *scales) {
    if (!domain || !cfg || !B || !scales || !valid_shape(cfg->K, cfg->N))
        return nullptr;
    try {
        rk_npu_i4_weight_config actual = *cfg;
        int safe = cfg->k_tile;
        double factor = 1;
        rknpu2_matmul_open::w4_tune::Profile profile;
        if (cfg->k_tile == 0) {
            rknpu2_matmul_open::detail::I4WeightBound bound;
            if (rknpu2_matmul_open::detail::bound_i4_weights(cfg->K, cfg->N, B, 0, bound))
                return nullptr;
            safe = bound.k_tile;
            if (const char *text = std::getenv("RK_NPU_W4A8_K_TILE_MULTIPLIER")) {
                errno = 0;
                char *end = nullptr;
                factor = std::strtod(text, &end);
                if (errno || end == text || *end || !std::isfinite(factor) || factor < 1)
                    return nullptr;
            }
            const int cap = std::min(rknpu2_matmul_open::detail::I4_K_TILE_MAX, ((cfg->K + 31) / 32) * 32);
            actual.k_tile = int(std::min(double(cap), safe * factor) / 32) * 32;
            profile.K=cfg->K;profile.N=cfg->N;profile.safe=safe;
            profile.safe_tiles=bound.safe_tiles;profile.multiplier=factor;profile.limit=actual.k_tile;
            rknpu2_matmul_open::tune::Hash hash;hash.mix(cfg->K);hash.mix(cfg->N);
            hash.bytes(B,size_t(cfg->K)*cfg->N);hash.bytes(scales,size_t(cfg->N)*sizeof(float));
            profile.fingerprint=hash.value;
        }
        return rknpu2_matmul_open::detail::create_w4a8_weights(domain,
            {actual, safe, factor, actual.k_tile > safe}, B, scales,
            cfg->k_tile==0?&profile:nullptr);
    } catch (...) {
        return nullptr;
    }
}
extern "C" void rk_npu_w4a8_weights_free(rk_npu_w4a8_weights *w) { delete w; }
extern "C" rk_npu_w4a8_tuning_profile* rk_npu_w4a8_tuning_profile_from_weights(
    const rk_npu_w4a8_weights* w) {
    if(!w) return nullptr;
    try { return new rk_npu_w4a8_tuning_profile{w->profile}; } catch(...) { return nullptr; }
}
extern "C" void rk_npu_w4a8_tuning_profile_free(rk_npu_w4a8_tuning_profile* p) { delete p; }
extern "C" int rk_npu_w4a8_weights_query(const rk_npu_w4a8_weights *w,
                                         rk_npu_w4a8_weight_info *out) {
    if (!w || !out)
        return RK_NPU_ERR_PARAM;
    *out = w->info;
    return RK_NPU_OK;
}
extern "C" rk_npu_w4a8_workspace *rk_npu_w4a8_workspace_create_ex(rk_npu_iommu_domain *domain,
    const rk_npu_w4a8_config *cfg, const rk_npu_w4a8_options* options) {
    if (!domain || !domain->ctx || !cfg) return nullptr;
    rk_npu_w4a8_memory_info_ex info{};
    if (rk_npu_w4a8_memory_query_ex(cfg,options,&info)) return nullptr;
    try {
        auto ws = std::make_unique<rk_npu_w4a8_workspace>();
        if(options_config(options,cfg->M,ws->layout,ws->npu_reduce)) return nullptr;
        ws->config = *cfg;
        ws->ctx_id = domain->ctx->id;
        ws->domain_id = domain->id;
        ws->scales.resize(cfg->M);
        ws->inverse.resize(cfg->M);
        if(!ws->npu_reduce)
            ws->accumulator.resize(size_t(cfg->M)*((size_t(cfg->N)+63)/64*64));
        rk_npu_i4_config inner{};
        if (backend_config(*cfg,inner,ws->layout)) return nullptr;
        if(cfg->pipeline==RK_NPU_W4A8_CHAIN) {
            if(ws->npu_reduce)return nullptr;
            ws->chain=std::make_unique<rknpu2_matmul_open::detail::W4A8Chain>();
            if(ws->chain->prepare(domain,inner,ws->layout))return nullptr;
        } else if(ws->executor.prepare(domain,inner,ws->npu_reduce,ws->layout))return nullptr;
        if(ws->npu_reduce && ws->reducer.prepare(domain,ws->executor.plan())) return nullptr;
        if(!ws->npu_reduce && cfg->N % 64 == 0) {
            ws->row_maxima.resize((size_t(cfg->M) + 16) * cfg->cpu_threads);
            uint32_t starts[3]{}, counts[3]{};
            ws->swiglu_pairs = rknpu2_matmul_open::detail::w4a8_swiglu_pairs(
                cfg->N, rknpu2_matmul_open::detail::i4_tiles(inner, starts, counts, ws->layout));
        }
        return ws.release();
    } catch (...) { return nullptr; }
}
extern "C" rk_npu_w4a8_workspace *rk_npu_w4a8_workspace_create(rk_npu_iommu_domain *domain,
    const rk_npu_w4a8_config *cfg) {
    return rk_npu_w4a8_workspace_create_ex(domain,cfg,nullptr);
}
extern "C" void rk_npu_w4a8_workspace_free(rk_npu_w4a8_workspace *ws) { delete ws; }
extern "C" int rk_npu_w4a8_run_f32(rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *w,
                                   const float *A, float *C, rk_npu_w4a8_timings *t) {
    return run(ws, w, A, C, false, t);
}
extern "C" int rk_npu_w4a8_run_f16(rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *w,
                                   const uint16_t *A, uint16_t *C, rk_npu_w4a8_timings *t) {
    return run(ws, w, A, C, true, t);
}

extern "C" int rk_npu_w4a8_run_swiglu_f32(rk_npu_w4a8_workspace *ws,
    const rk_npu_w4a8_weights *w, const float *A, float *middle, rk_npu_w4a8_timings *t) {
    return run(ws, w, A, middle, false, t, true);
}

extern "C" int rk_npu_w4a8_run_ffn_f32(rk_npu_w4a8_workspace *gate_up,
    const rk_npu_w4a8_weights *gate_up_weights, rk_npu_w4a8_workspace *down,
    const rk_npu_w4a8_weights *down_weights, const float *A, float *middle,
    float *C, rk_npu_w4a8_ffn_timings *metrics) {
    if (metrics) *metrics = {};
    if (gate_up == down || middle == C || A == C)
        return RK_NPU_ERR_PARAM;
    int rc = validate(gate_up, gate_up_weights, A, middle, true);
    if (rc) return rc;
    rc = validate(down, down_weights, middle, C, false);
    if (rc) return rc;
    if (gate_up->config.M != down->config.M || gate_up->config.N / 2 != down->config.K)
        return RK_NPU_ERR_PARAM;
    if (gate_up->ctx_id != down->ctx_id || gate_up->domain_id != down->domain_id)
        return RK_NPU_ERR_DOMAIN;
    WorkspaceLock gate_lock(*gate_up);
    if (!gate_lock) return RK_NPU_ERR_BUSY;
    WorkspaceLock down_lock(*down);
    if (!down_lock) return RK_NPU_ERR_BUSY;
    const auto begin = Clock::now();
    rk_npu_w4a8_ffn_timings t{};
    rc = run_locked(gate_up, gate_up_weights, A, middle, false, &t.gate_up, true, true);
    if (!rc) rc = run_locked(down, down_weights, middle, C, false, &t.down, false, false, gate_up);
    t.total_us = elapsed(begin);
    if (metrics) *metrics = t;
    return rc;
}

#include "rk_npu_w4a8.h"
#include "rk_npu_w4a8_internal.h"
#include "rk_npu_i4_executor.h"
#include "rk_npu_w4a8_cpu.h"
#include "rk_npu_w4a8_npu_reduce.h"
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
};
struct rk_npu_w4a8_workspace {
    rk_npu_w4a8_config config{};
    uint64_t ctx_id = 0;
    uint32_t domain_id = 0;
    std::vector<float> scales, inverse;
    std::vector<int32_t> accumulator;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    rknpu2_matmul_open::detail::I4Executor executor;
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
        cfg.m_tile > (layout == rknpu2_matmul_open::detail::I4InputLayout::Panel8 ? 256 : 64))
        return RK_NPU_ERR_PARAM;
    out = {cfg.M * 2,         cfg.N,          cfg.K,        cfg.m_tile * 2, cfg.k_tile, cfg.n_tile,
           cfg.npu_core_mask, cfg.timeout_ms, cfg.pipeline, cfg.cpu_threads};
    return RK_NPU_OK;
}

int run(rk_npu_w4a8_workspace *ws, const rk_npu_w4a8_weights *weights, const void *A, void *C,
        bool half, rk_npu_w4a8_timings *metrics) {
    if (!ws || !weights || !A || !C || A == C)
        return RK_NPU_ERR_PARAM;
    if (ws->busy.test_and_set(std::memory_order_acquire))
        return RK_NPU_ERR_BUSY;
    struct Unlock {
        std::atomic_flag &f;
        ~Unlock() { f.clear(std::memory_order_release); }
    } unlock{ws->busy};
    const auto &c = ws->config;
    const auto &wc = weights->packed.config();
    const auto &mem = weights->packed.memory();
    if (mem.ctx_id != ws->ctx_id || mem.iommu_domain_id != ws->domain_id)
        return RK_NPU_ERR_DOMAIN;
    if (wc.K != c.K || wc.N != c.N || wc.k_tile != c.k_tile)
        return RK_NPU_ERR_PARAM;
    const auto start = Clock::now();
    rk_npu_w4a8_timings t{};
    int rc =
        rknpu2_matmul_open::detail::w4a8_scales(A, half, c.M, c.K, c.cpu_threads, ws->scales.data(), ws->inverse.data());
    t.activation_scan_us = elapsed(start);
    if (!rc) {
        rknpu2_matmul_open::detail::W4A8Input input{A, half, c.M, c.K, ws->scales.data(), ws->inverse.data(), c.cpu_threads, ws->layout};
        rknpu2_matmul_open::detail::W4A8Reduction reduction{ws->accumulator.data(), weights->correction.data(),
                                       ws->scales.data(), weights->scales.data(), C, half};
        rk_npu_i4_timings inner{};
        rc = ws->executor.run(weights->packed, rknpu2_matmul_open::detail::w4a8_pack, &input, nullptr, &inner,
                              ws->npu_reduce ? nullptr : rknpu2_matmul_open::detail::w4a8_reduce, &reduction);
        t.quant_pack_us = inner.pack_us;
        t.sync_us = inner.sync_us;
        t.submit_us = inner.submit_us;
        t.reduce_us = inner.reduce_us;
        if (!rc && ws->npu_reduce) rc = ws->reducer.run(reduction,t);
    }
    t.total_us = elapsed(start);
    if (metrics)
        *metrics = t;
    return rc;
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
    rc=rknpu2_matmul_open::detail::query_i4(inner,info.base.backend,layout); if(rc) return rc;
    info.base.weight_metadata_bytes=uint64_t(cfg->N)*(sizeof(float)+sizeof(int32_t));
    info.base.cpu_scratch_bytes=uint64_t(cfg->M)*sizeof(float)*2;
    if(npu) {
        try { rc=rknpu2_matmul_open::detail::W4A8NpuReduce::query(inner,layout,info); }
        catch(...) { return RK_NPU_ERR_NOMEM; }
        if(rc) return rc;
    } else info.base.cpu_scratch_bytes+=uint64_t(cfg->M)*((uint64_t(cfg->N)+63)/64*64)*sizeof(int32_t);
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
    const rk_npu_w4a8_weight_info& info, const int8_t* B, const float* scales) {
    const auto& actual=info.config;
    if(!domain || !B || !scales || !valid_shape(actual.K,actual.N)) return nullptr;
    try {
        uint64_t bytes = 0;
        if (rk_npu_i4_weights_memory_query(&actual, &bytes))
            return nullptr;
        auto w = std::make_unique<rk_npu_w4a8_weights>();
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
        }
        return rknpu2_matmul_open::detail::create_w4a8_weights(domain,
            {actual, safe, factor, actual.k_tile > safe}, B, scales);
    } catch (...) {
        return nullptr;
    }
}
extern "C" void rk_npu_w4a8_weights_free(rk_npu_w4a8_weights *w) { delete w; }
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
        if (backend_config(*cfg,inner,ws->layout) ||
            ws->executor.prepare(domain,inner,ws->npu_reduce,ws->layout)) return nullptr;
        if(ws->npu_reduce && ws->reducer.prepare(domain,ws->executor.plan())) return nullptr;
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

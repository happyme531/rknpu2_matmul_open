#include "rk_npu_w4a4_linear.h"
#include "rk_npu_flatquant.h"
#include "rk_npu_i4_bounds.h"
#include "rk_npu_i4_executor.h"
#include "rk_npu_w4a4_linear_cpu.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

struct rk_npu_w4a4_linear_weights {
    rknpu2_matmul_open::detail::I4Weights packed;
    std::vector<float> scales;
    rknpu2_matmul_open::detail::FlatWeights transform;
    rk_npu_w4a4_linear_weight_info info{};
};
struct rk_npu_w4a4_linear_workspace {
    rk_npu_w4a4_linear_config config{};
    uint64_t ctx_id = 0;
    uint32_t domain_id = 0;
    std::vector<float> scales, inverse;
    std::vector<uint8_t> quantized;
    std::vector<int32_t> accumulator;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    rknpu2_matmul_open::detail::I4Executor executor;
    rknpu2_matmul_open::detail::FlatPlan transform;
};
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
bool valid_ratio(float x) { return x > 0 && x <= 1; }
int validate_transform(const rk_npu_w4a4_transform *t, int K, rknpu2_matmul_open::detail::FlatShape &out) {
    out = {};
    if (!t)
        return RK_NPU_OK;
    if (t->struct_size != sizeof(*t) || !valid_ratio(t->negative_clip_ratio) ||
        !valid_ratio(t->positive_clip_ratio))
        return RK_NPU_ERR_PARAM;
    if (!t->left_dim && !t->right_dim)
        return t->left || t->right ? RK_NPU_ERR_PARAM : RK_NPU_OK;
    if (!t->left || !t->right || int64_t(t->left_dim) * t->right_dim != K)
        return RK_NPU_ERR_PARAM;
    return rknpu2_matmul_open::detail::flat_shape(t->left_dim, t->right_dim, out);
}
rk_npu_i4_config integer_config(const rk_npu_w4a4_linear_config &c) {
    return {c.M,          c.N,        c.K,          c.m_tile, c.k_tile, c.n_tile, c.npu_core_mask,
            c.timeout_ms, c.pipeline, c.cpu_threads};
}
int memory_query(const rk_npu_w4a4_linear_config &cfg, const rknpu2_matmul_open::detail::FlatShape &s,
                 rk_npu_w4a4_linear_memory_info &info) {
    if (cfg.transform_batch < 0 || cfg.transform_batch > 256 || cfg.transform_ubatch < 0 ||
        cfg.transform_ubatch > 32)
        return RK_NPU_ERR_PARAM;
    int rc = rknpu2_matmul_open::detail::query_i4(integer_config(cfg), info.backend);
    if (rc)
        return rc;
    uint64_t packed = uint64_t(cfg.M) * align_up(cfg.K, 32) / 2;
    info.weight_metadata_bytes = uint64_t(cfg.N) * sizeof(float);
    info.cpu_scratch_bytes = packed + uint64_t(cfg.M) * 2 * sizeof(float) +
                             uint64_t(cfg.M) * align_up(cfg.N, 64) * sizeof(int32_t);
    if (info.cpu_scratch_bytes > SIZE_MAX)
        return RK_NPU_ERR_NOMEM;
    if (s.L) {
        info.weight_metadata_bytes += 2 * (uint64_t(s.rp) * s.rp + uint64_t(s.lp) * s.lp * 4);
        rc = rknpu2_matmul_open::detail::FlatPlan::query(cfg, s, info.transform_workspace_bytes);
        if (rc)
            return rc;
    }
    info.workspace_bytes = info.backend.input_bytes + info.backend.partial_bytes +
                           info.backend.control_bytes + info.cpu_scratch_bytes +
                           info.transform_workspace_bytes;
    return RK_NPU_OK;
}
int run(rk_npu_w4a4_linear_workspace *ws, const rk_npu_w4a4_linear_weights *weights, const void *A,
        void *C, bool half, rk_npu_w4a4_linear_timings *metrics) {
    if (!ws || !weights || !A || !C || A == C)
        return RK_NPU_ERR_PARAM;
    if (ws->busy.test_and_set(std::memory_order_acquire))
        return RK_NPU_ERR_BUSY;
    struct Unlock {
        std::atomic_flag &f;
        ~Unlock() { f.clear(std::memory_order_release); }
    } unlock{ws->busy};
    const auto &cfg = ws->config;
    const auto &wc = weights->packed.config();
    const auto &mem = weights->packed.memory();
    if (wc.K != cfg.K || wc.N != cfg.N || wc.k_tile != cfg.k_tile ||
        weights->info.left_dim != ws->transform.shape().L ||
        weights->info.right_dim != ws->transform.shape().R)
        return RK_NPU_ERR_PARAM;
    if (mem.ctx_id != ws->ctx_id || mem.iommu_domain_id != ws->domain_id)
        return RK_NPU_ERR_DOMAIN;
    const auto start = Clock::now();
    rk_npu_w4a4_linear_timings t{};
    rknpu2_matmul_open::detail::FlatMetrics ft{};
    const bool transform = weights->info.left_dim != 0;
    int rc = rknpu2_matmul_open::detail::flat_validate_input(A, half, cfg.M, cfg.K, cfg.cpu_threads, transform);
    const double validation = elapsed(start);
    if (!rc) {
        float neg = weights->info.negative_clip_ratio, pos = weights->info.positive_clip_ratio;
        rc = transform
                 ? ws->transform.run(weights->transform, A, half, neg, pos, ws->quantized.data(),
                                     ws->scales.data(), ws->inverse.data(), ft)
                 : rknpu2_matmul_open::detail::flat_quantize_identity(A, half, cfg.M, cfg.K, cfg.cpu_threads, neg, pos,
                                                 ws->quantized.data(), ws->scales.data(),
                                                 ws->inverse.data(), ft);
    }
    t.transform_scan_us = elapsed(start);
    t.transform_pack_us = ft.pack;
    t.transform_submit_us = ft.submit;
    t.activation_scan_us = validation + ft.scan;
    t.quant_pack_us = ft.quant;
    t.sync_us = ft.sync;
    if (!rc) {
        rknpu2_matmul_open::detail::W4A4Input input{ws->quantized.data(), cfg.M, cfg.K, cfg.cpu_threads};
        rknpu2_matmul_open::detail::W4A4Reduction reduction{ws->accumulator.data(), ws->scales.data(),
                                       weights->scales.data(), C, half};
        rk_npu_i4_timings inner{};
        rc = ws->executor.run(weights->packed, rknpu2_matmul_open::detail::w4a4_pack, &input, nullptr, &inner,
                              rknpu2_matmul_open::detail::w4a4_reduce, &reduction);
        t.quant_pack_us += inner.pack_us;
        t.sync_us += inner.sync_us;
        t.submit_us = inner.submit_us;
        t.reduce_dequant_us = inner.reduce_us;
    }
    t.total_us = elapsed(start);
    if (metrics)
        *metrics = t;
    return rc;
}
} // namespace
extern "C" void rk_npu_w4a4_linear_config_init(rk_npu_w4a4_linear_config *cfg, int M, int N,
                                               int K) {
    if (cfg)
        *cfg = {M, N, K, 128, 480, 1024, 1, 500, 1, 1, 0, 0};
}
extern "C" void rk_npu_w4a4_transform_init(rk_npu_w4a4_transform *t) {
    if (t)
        *t = {sizeof(*t), 0, 0, nullptr, nullptr, 1, 1};
}
extern "C" int rk_npu_w4a4_linear_memory_query(const rk_npu_w4a4_linear_config *cfg,
                                               const rk_npu_w4a4_transform *transform,
                                               rk_npu_w4a4_linear_memory_info *out) {
    if (!cfg || !out)
        return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::detail::FlatShape s;
    int rc = validate_transform(transform, cfg->K, s);
    if (rc)
        return rc;
    rk_npu_w4a4_linear_memory_info info{};
    rc = memory_query(*cfg, s, info);
    if (!rc)
        *out = info;
    return rc;
}
extern "C" rk_npu_w4a4_linear_weights *
rk_npu_w4a4_linear_weights_create(rk_npu_iommu_domain *domain, const rk_npu_i4_weight_config *cfg,
                                  const int8_t *B, const float *scales,
                                  const rk_npu_w4a4_transform *transform) {
    if (!domain || !domain->ctx || !cfg || !B || !scales || cfg->K < 1 || cfg->N < 1)
        return nullptr;
    try {
        rknpu2_matmul_open::detail::FlatShape s;
        if (validate_transform(transform, cfg->K, s))
            return nullptr;
        auto actual = *cfg;
        if (!actual.k_tile) {
            rknpu2_matmul_open::detail::I4WeightBound bound{};
            if (rknpu2_matmul_open::detail::bound_i4_weights(actual.K, actual.N, B, 0, bound))
                return nullptr;
            actual.k_tile = bound.k_tile;
        }
        uint64_t bytes = 0;
        if (rknpu2_matmul_open::detail::query_i4_weights(actual, bytes))
            return nullptr;
        auto w = std::make_unique<rk_npu_w4a4_linear_weights>();
        w->scales.assign(scales, scales + actual.N);
        for (float v : w->scales)
            if (!std::isfinite(v) || v <= 0)
                return nullptr;
        if (s.L && w->transform.prepare(domain, s.L, s.R, transform->left, transform->right))
            return nullptr;
        w->info = {actual, s.L, s.R, transform ? transform->negative_clip_ratio : 1.f,
                   transform ? transform->positive_clip_ratio : 1.f};
        if (w->packed.prepare(domain, actual, B))
            return nullptr;
        return w.release();
    } catch (...) {
        return nullptr;
    }
}
extern "C" void rk_npu_w4a4_linear_weights_free(rk_npu_w4a4_linear_weights *w) { delete w; }
extern "C" int rk_npu_w4a4_linear_weights_query(const rk_npu_w4a4_linear_weights *w,
                                                rk_npu_w4a4_linear_weight_info *out) {
    if (!w || !out)
        return RK_NPU_ERR_PARAM;
    *out = w->info;
    return RK_NPU_OK;
}
extern "C" rk_npu_w4a4_linear_workspace *
rk_npu_w4a4_linear_workspace_create(rk_npu_iommu_domain *domain,
                                    const rk_npu_w4a4_linear_config *cfg,
                                    const rk_npu_w4a4_linear_weights *weights) {
    if (!domain || !domain->ctx || !cfg || !weights)
        return nullptr;
    const auto &wc = weights->packed.config();
    const auto &mem = weights->packed.memory();
    if (wc.K != cfg->K || wc.N != cfg->N || wc.k_tile != cfg->k_tile ||
        mem.ctx_id != domain->ctx->id || mem.iommu_domain_id != domain->id)
        return nullptr;
    rk_npu_w4a4_linear_memory_info info{};
    if (memory_query(*cfg, weights->transform.shape(), info))
        return nullptr;
    try {
        auto ws = std::make_unique<rk_npu_w4a4_linear_workspace>();
        ws->config = *cfg;
        ws->ctx_id = domain->ctx->id;
        ws->domain_id = domain->id;
        ws->quantized.resize(size_t(cfg->M) * align_up(cfg->K, 32) / 2);
        ws->scales.resize(cfg->M);
        ws->inverse.resize(cfg->M);
        ws->accumulator.resize(size_t(cfg->M) * align_up(cfg->N, 64));
        if (ws->executor.prepare(domain, integer_config(*cfg)))
            return nullptr;
        if (weights->info.left_dim &&
            ws->transform.prepare(domain, *cfg, weights->transform.shape()))
            return nullptr;
        return ws.release();
    } catch (...) {
        return nullptr;
    }
}
extern "C" void rk_npu_w4a4_linear_workspace_free(rk_npu_w4a4_linear_workspace *ws) { delete ws; }
extern "C" int rk_npu_w4a4_linear_run_f32(rk_npu_w4a4_linear_workspace *ws,
                                          const rk_npu_w4a4_linear_weights *w, const float *A,
                                          float *C, rk_npu_w4a4_linear_timings *t) {
    return run(ws, w, A, C, false, t);
}
extern "C" int rk_npu_w4a4_linear_run_f16(rk_npu_w4a4_linear_workspace *ws,
                                          const rk_npu_w4a4_linear_weights *w, const uint16_t *A,
                                          uint16_t *C, rk_npu_w4a4_linear_timings *t) {
    return run(ws, w, A, C, true, t);
}

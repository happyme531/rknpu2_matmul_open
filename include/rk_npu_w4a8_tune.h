#ifndef RK_NPU_W4A8_TUNE_H
#define RK_NPU_W4A8_TUNE_H
#include "rk_npu_w4a8.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum rk_npu_w4a8_activation_type {
    RK_NPU_W4A8_F32 = 0,
    RK_NPU_W4A8_F16 = 1
} rk_npu_w4a8_activation_type;

typedef struct rk_npu_w4a8_autotune_config {
    int M, N, K;
    rk_npu_w4a8_activation_type activation_type;
    uint32_t allowed_npu_core_mask; /* 0: all; choose once, do not search */
    uint64_t allowed_cpu_core_mask; /* 0: current affinity; prefer up to 4 big cores */
    int warmup, loops, repeats;
    uint32_t timeout_ms;
    int verbose;
    int required_k_tile; /* 0: search; otherwise a 32-aligned tile <= min(align32(K),2048) */
    double k_tile_multiplier; /* 0 (default): read environment; >=1: explicit policy */
} rk_npu_w4a8_autotune_config;

typedef struct rk_npu_w4a8_strategy {
    rk_npu_w4a8_config config; /* selected K/N/logical-M tiles, pipeline always 1 */
    rk_npu_w4a8_activation_type activation_type;
    uint64_t cpu_core_mask; /* measured affinity; caller applies this when running */
    rk_npu_w4a8_weight_info weights;
    uint64_t weight_fingerprint; /* B codes and FP32 scales, not a security hash */
    uint32_t tuning_revision;
    double input_us, npu_us, sync_us, output_us, total_us;
    double jitter_pct, robust_us;
} rk_npu_w4a8_strategy;

/* Extended latency strategy carries the input layout selected by tuning.
 * Legacy strategy/functions remain native-only and keep their original ABI.
 * Use the _ex weight/workspace factories together; copying only base.config
 * into the legacy workspace factory loses the selected input layout. */
typedef struct rk_npu_w4a8_strategy_ex {
    rk_npu_w4a8_strategy base;
    rk_npu_w4a8_options execution; /* native or panel8 input, CPU reduction */
} rk_npu_w4a8_strategy_ex;

typedef struct rk_npu_w4a8_family_config {
    rk_npu_w4a8_autotune_config base; /* M ignored; dtype/resources/policy shared */
    int m_count;
    const int* m_values;
    const double* frequencies; /* relative call counts; NULL means equal */
} rk_npu_w4a8_family_config;
typedef struct rk_npu_w4a8_family_summary {
    rk_npu_w4a8_weight_info weights; /* one common K partition for all M values */
    uint64_t weight_fingerprint;
    double weighted_total_us, weighted_robust_us;
} rk_npu_w4a8_family_summary;

/* Defaults: f32, all NPU/current CPU resources, 2 warmups, 8 loops, 3 repeats,
 * timeout500ms, multiplier from the existing environment (1 when unset).
 * Set multiplier1 explicitly for a certified-only search regardless of env. */
void rk_npu_w4a8_autotune_config_init(rk_npu_w4a8_autotune_config*, int M, int N, int K);
void rk_npu_w4a8_family_config_init(rk_npu_w4a8_family_config*, int N, int K,
                                    int m_count, const int* m_values);

/* Actual B[K,N]/scale[N] are required for bounds and cache identity. A is a
 * deterministic synthetic calibration tensor. Search Ktile/Ntile/Mtile with
 * pipeline enabled and fixed resources. Time complete production calls;
 * exclude weight/workspace creation. Every timed output is checked against
 * a K<=480 reference. A submit failure stops tuning immediately.
 * A multiplier>1 is experimental: sample agreement is NOT an INT16 proof;
 * no per-run overflow detection/retry is added. Input arrays remain immutable
 * until return. Tune calls preserve the caller thread's CPU affinity.
 * Either fastest or stable may be NULL, but not both. */
int rk_npu_w4a8_autotune(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, rk_npu_w4a8_strategy* fastest, rk_npu_w4a8_strategy* stable);
int rk_npu_w4a8_autotune_cached(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path, int refresh,
    rk_npu_w4a8_strategy* fastest, rk_npu_w4a8_strategy* stable, int* cache_hit);
/* Cache operations validate current weight bounds as well as shape, dtype,
 * resources, policy, driver and tuning revision. No device work on a hit.
 * NULL cache_path uses a separate rk_npu_w4a8 cache directory. */
int rk_npu_w4a8_autotune_cache_load(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path,
    rk_npu_w4a8_strategy* fastest, rk_npu_w4a8_strategy* stable);
int rk_npu_w4a8_autotune_cache_save(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path,
    const rk_npu_w4a8_strategy* fastest, const rk_npu_w4a8_strategy* stable);

/* Revalidate the supplied weights and the chosen policy, then create exactly
 * the selected K partition (including an explicitly relaxed strategy).
 * Does not modify/read the environment. Create the workspace from strategy.config. */
rk_npu_w4a8_weights* rk_npu_w4a8_weights_create_tuned(rk_npu_iommu_domain*,
    const rk_npu_w4a8_strategy*, const int8_t* B, const float* w_scale);

/* Cached family search reuses single-M entries, sharing one selected Ktile.
 * Each requested summary requires a strategy array of m_count elements. */
int rk_npu_w4a8_autotune_family(rk_npu_ctx*, const rk_npu_w4a8_family_config*,
    const int8_t* B, const float* w_scale,
    rk_npu_w4a8_family_summary* fastest, rk_npu_w4a8_strategy* fastest_strategies,
    rk_npu_w4a8_family_summary* stable, rk_npu_w4a8_strategy* stable_strategies);
int rk_npu_w4a8_autotune_family_cached(rk_npu_ctx*, const rk_npu_w4a8_family_config*,
    const int8_t* B, const float* w_scale, const char* cache_directory, int refresh,
    rk_npu_w4a8_family_summary* fastest, rk_npu_w4a8_strategy* fastest_strategies,
    rk_npu_w4a8_family_summary* stable, rk_npu_w4a8_strategy* stable_strategies,
    int* cache_hits, int* cache_misses);

/* Search native plus panel8 and logical ubatches up to 256, within the
 * workspace geometry/CBUF budget. Resources are fixed, pipeline=1 and CPU
 * reduction only. Small ragged batches retain native candidates; supported
 * larger tails are measured against the same full-output reference.
 * Cache identity includes this search space; old/native-only cache entries
 * cannot stand in for an extended search. Remaining contracts match above. */
int rk_npu_w4a8_autotune_ex(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale,
    rk_npu_w4a8_strategy_ex* fastest, rk_npu_w4a8_strategy_ex* stable);
int rk_npu_w4a8_autotune_cached_ex(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path, int refresh,
    rk_npu_w4a8_strategy_ex* fastest, rk_npu_w4a8_strategy_ex* stable, int* cache_hit);
int rk_npu_w4a8_autotune_cache_load_ex(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path,
    rk_npu_w4a8_strategy_ex* fastest, rk_npu_w4a8_strategy_ex* stable);
int rk_npu_w4a8_autotune_cache_save_ex(rk_npu_ctx*, const rk_npu_w4a8_autotune_config*,
    const int8_t* B, const float* w_scale, const char* cache_path,
    const rk_npu_w4a8_strategy_ex* fastest, const rk_npu_w4a8_strategy_ex* stable);
rk_npu_w4a8_weights* rk_npu_w4a8_weights_create_tuned_ex(rk_npu_iommu_domain*,
    const rk_npu_w4a8_strategy_ex*, const int8_t* B, const float* w_scale);
/* Caller applies base.cpu_core_mask before workspace creation/execution.
 * Query memory with memory_query_ex(&s.base.config, &s.execution, ...). */
rk_npu_w4a8_workspace* rk_npu_w4a8_workspace_create_tuned_ex(rk_npu_iommu_domain*,
    const rk_npu_w4a8_strategy_ex*);
int rk_npu_w4a8_autotune_family_ex(rk_npu_ctx*, const rk_npu_w4a8_family_config*,
    const int8_t* B, const float* w_scale,
    rk_npu_w4a8_family_summary* fastest, rk_npu_w4a8_strategy_ex* fastest_strategies,
    rk_npu_w4a8_family_summary* stable, rk_npu_w4a8_strategy_ex* stable_strategies);
int rk_npu_w4a8_autotune_family_cached_ex(rk_npu_ctx*, const rk_npu_w4a8_family_config*,
    const int8_t* B, const float* w_scale, const char* cache_directory, int refresh,
    rk_npu_w4a8_family_summary* fastest, rk_npu_w4a8_strategy_ex* fastest_strategies,
    rk_npu_w4a8_family_summary* stable, rk_npu_w4a8_strategy_ex* stable_strategies,
    int* cache_hits, int* cache_misses);

#ifdef __cplusplus
}
#endif
#endif

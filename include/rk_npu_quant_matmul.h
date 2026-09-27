/*
 * Typed quantized matmul APIs for RK3588.
 * ======================================
 *
 * This header exposes three typed single-GEMM operators:
 *
 *   i8i8i32:   int8 A[M,K] * int8 B[K,N] -> int32 C[M,N]
 *   f16i8f16:  fp16 A[M,K] * int8 B[K,N] -> fp16 C[M,N]
 *   f32i8f32:  fp32 A[M,K] * int8 B[K,N] -> fp32 C[M,N]
 *
 * The first operator performs no numerical quantization or dequantization.
 * The latter two fuse CPU activation quantization/NPU-layout packing before
 * the NPU and exact INT32 reduction/dequantization/output conversion after it.
 * B is already symmetric signed INT8 with zero point 0 and one positive
 * per-output-channel scale.
 *
 * Execution workspaces are fixed to one exact M/strategy. Immutable packed
 * weights are fixed only to (K,N,Ktile), so one logical LLM weight can be
 * shared across workspaces for changing sequence lengths. A and C are compact
 * row-major host tensors supplied to each blocking run call. FP16 tensors use
 * raw IEEE binary16 bits stored in uint16_t.
 */
#ifndef RK_NPU_QUANT_MATMUL_H
#define RK_NPU_QUANT_MATMUL_H

#include "rk_npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rk_npu_matmul_op_kind {
    RK_NPU_MATMUL_I8I8I32 = 0,
    RK_NPU_MATMUL_F16I8F16_DYNAMIC = 1,
    RK_NPU_MATMUL_F16I8F16_STATIC = 2,
    RK_NPU_MATMUL_F32I8F32_DYNAMIC = 3,
    RK_NPU_MATMUL_F32I8F32_STATIC = 4,
} rk_npu_matmul_op_kind;

typedef enum rk_npu_matmul_a_layout {
    /* Quantized A is row-major [M, align32(Ktile)]. */
    RK_NPU_MATMUL_A_LAYOUT_NORMAL = 0,
    /* Quantized A is native [align32(Ktile)/16, M, 16]. */
    RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16 = 1,
    RK_NPU_MATMUL_A_LAYOUT_PANEL8 = 2,
    RK_NPU_MATMUL_A_LAYOUT_PANEL16 = 3,
} rk_npu_matmul_a_layout;

typedef enum rk_npu_matmul_c_layout {
    /* Zero preserves the existing native-C contract of initialized strategies. */
    RK_NPU_MATMUL_C_LAYOUT_NATIVE = 0,
    RK_NPU_MATMUL_C_LAYOUT_PANEL8 = 1,
    RK_NPU_MATMUL_C_LAYOUT_PANEL16 = 2,
} rk_npu_matmul_c_layout;

typedef struct rk_npu_matmul_autotune_config {
    int M;
    int N;
    int K;

    /* Resources are selected once, never benchmark-searched. Zero permits all
     * available cores. NPU uses the largest supported subset fitting N's 32-wide
     * groups (masks 1/2/4/3/7). CPU uses up to four permitted CPUs, preferring
     * big cores 4..7; an explicit narrow mask pins fewer workers. */
    uint32_t allowed_npu_core_mask;
    uint64_t allowed_cpu_core_mask;

    /* Each repeat records the average of `loops` independent blocking calls
     * after warmup.  A call may pipeline its own K waves, but never overlaps a
     * different matmul request.  Plan construction and static B packing are
     * outside the timed region. */
    int warmup;
    int loops;
    int repeats;
    uint32_t timeout_ms;
    int verbose;

    /* Zero searches all derived K tiles.  A positive value constrains the
     * search to one existing packed-weight partition. */
    int required_k_tile;
} rk_npu_matmul_autotune_config;

typedef struct rk_npu_matmul_weight_config {
    int K;
    int N;
    int k_tile;
} rk_npu_matmul_weight_config;

typedef struct rk_npu_matmul_family_autotune_config {
    int K;
    int N;
    int m_count;
    const int* m_values;
    /* Relative expected call counts. NULL means equal weight. */
    const double* frequencies;
    uint32_t allowed_npu_core_mask;
    uint64_t allowed_cpu_core_mask;
    int warmup;
    int loops;
    int repeats;
    uint32_t timeout_ms;
    int verbose;
} rk_npu_matmul_family_autotune_config;

typedef struct rk_npu_matmul_strategy {
    rk_npu_matmul_op_kind op_kind;
    int M;
    int N;
    int K;

    int k_tile;
    int n_tile;
    /* An autotuned execution choice. Public A remains row-major fp16/fp32/i8;
     * packing into this quantized NPU layout happens inside the workspace. */
    rk_npu_matmul_a_layout a_layout;
    int wave_count;
    int n_groups;
    uint32_t npu_core_mask;
    uint64_t cpu_core_mask;
    int cpu_threads;

    /* Median per-call times in microseconds.  total_us is single-call blocking
     * latency.  input/npu/output are accumulated stage work and may overlap,
     * so they are diagnostic values and do not sum to total_us. */
    double input_us;
    double npu_us;
    double sync_us;
    double output_us;
    double total_us;
    double jitter_pct;
    double robust_us;
    /* Internal output recipe; public C remains compact row-major. This field
     * extends the alpha C struct: rebuild callers together with the library. */
    rk_npu_matmul_c_layout c_layout;
} rk_npu_matmul_strategy;

typedef struct rk_npu_i8i8i32_weights rk_npu_i8i8i32_weights;
typedef struct rk_npu_f16i8f16_weights rk_npu_f16i8f16_weights;
typedef struct rk_npu_f32i8f32_weights rk_npu_f32i8f32_weights;
typedef struct rk_npu_matmul_workspace rk_npu_matmul_workspace;

typedef struct rk_npu_matmul_family_summary {
    rk_npu_matmul_weight_config weight_config;
    double weighted_total_us;
    double weighted_robust_us;
} rk_npu_matmul_family_summary;

/* Logical payload sizes exclude allocator page rounding and C++/thread state. */
typedef struct rk_npu_matmul_workspace_requirements {
    uint64_t bytes;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t control_bytes;
    uint32_t wave_count;
    uint32_t buffer_count;
} rk_npu_matmul_workspace_requirements;

typedef struct rk_npu_matmul_weight_requirements {
    uint64_t bytes;
    uint64_t packed_b_bytes;
    uint64_t scale_bytes;
    uint32_t wave_count;
    uint32_t buffer_count;
} rk_npu_matmul_weight_requirements;

/* Defaults: all NPU cores, current CPU affinity, warmup=2, loops=8,
 * repeats=3, timeout=500 ms, verbose=0. */
void rk_npu_matmul_autotune_config_init(
    rk_npu_matmul_autotune_config* cfg, int M, int N, int K);
void rk_npu_matmul_family_autotune_config_init(
    rk_npu_matmul_family_autotune_config* cfg, int N, int K,
    int m_count, const int* m_values);

/* Autotune searches K/N tiles and validated A/C layout recipes, with fixed
 * resources selected from the masks. It is shape/mask-only: callers do not
 * provide A, B, or scales.
 * Deterministic synthetic tensors generated from a fixed internal seed are
 * used for candidate validation and timing, then discarded before return.
 * Every final candidate is timed through the production intra-call pipeline:
 * A-slice i+1 packing and partial i-1 reduction may overlap NPU wave i.
 * `fastest` minimizes median total_us; `stable` penalizes run-to-run spread via
 * robust_us.  Either output may be NULL, but not both.  Returned strategies are
 * plain values and may be copied or serialized for later workspace creation. */
int rk_npu_i8i8i32_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable);
int rk_npu_f16i8f16_dynamic_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable);
int rk_npu_f16i8f16_static_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable);
int rk_npu_f32i8f32_dynamic_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable);
int rk_npu_f32i8f32_static_autotune(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable);

/* Versioned tuning-result serialization.  One cache file contains both the
 * fastest and stable strategies plus the shape, permission masks, measurement
 * settings, driver version, and an internal tuning revision.  Load returns
 * RK_NPU_ERR_CACHE_MISS when the file is absent, malformed, stale, or does not
 * match cfg/kind/current process affinity.  A cache miss is not a device error.
 * Save uses an atomic temporary-file rename; cache_path must not be empty. */
int rk_npu_matmul_autotune_cache_load(
    rk_npu_ctx* ctx, const char* cache_path,
    rk_npu_matmul_op_kind kind,
    const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_strategy* fastest,
    rk_npu_matmul_strategy* stable);
int rk_npu_matmul_autotune_cache_save(
    rk_npu_ctx* ctx, const char* cache_path,
    rk_npu_matmul_op_kind kind,
    const rk_npu_matmul_autotune_config* cfg,
    const rk_npu_matmul_strategy* fastest,
    const rk_npu_matmul_strategy* stable);

/* Convenience path: load a matching result, or run the ordinary autotuner and
 * save both strategies on a miss.  NULL cache_path selects
 * $XDG_CACHE_HOME/rk_npu_matmul (or $HOME/.cache/rk_npu_matmul) and a filename
 * derived from the full cache key.  refresh != 0 skips the initial load.
 * Cache write failure never hides an otherwise successful tuning result.
 * cache_hit is optional and receives 1 for a load or 0 for a real tune. */
int rk_npu_matmul_autotune_cached(
    rk_npu_ctx* ctx, const rk_npu_matmul_autotune_config* cfg,
    rk_npu_matmul_op_kind kind, const char* cache_path, int refresh,
    rk_npu_matmul_strategy* fastest, rk_npu_matmul_strategy* stable,
    int* cache_hit);

/* Jointly select one K partition for all M values. Strategy output arrays have
 * cfg->m_count entries in the same order as m_values. A summary and its array
 * must either both be NULL or both be non-NULL; at least one family is needed. */
int rk_npu_i8i8i32_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies);
int rk_npu_f16i8f16_dynamic_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies);
int rk_npu_f16i8f16_static_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies);
int rk_npu_f32i8f32_dynamic_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies);
int rk_npu_f32i8f32_static_autotune_family(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies);

/* Cached family tune reuses the ordinary single-M cache for every derived
 * (M, required K tile) subproblem, then recomputes the weighted family choice.
 * This permits partial reuse when M values are added and full reuse when only
 * frequencies change. NULL cache_directory uses the default per-user cache;
 * otherwise each subproblem is stored as one hashed file in that directory.
 * cache_hits/cache_misses are optional counts of single-M subproblems. */
int rk_npu_matmul_autotune_family_cached(
    rk_npu_ctx* ctx, const rk_npu_matmul_family_autotune_config* cfg,
    rk_npu_matmul_op_kind kind, const char* cache_directory, int refresh,
    rk_npu_matmul_family_summary* fastest_summary,
    rk_npu_matmul_strategy* fastest_strategies,
    rk_npu_matmul_family_summary* stable_summary,
    rk_npu_matmul_strategy* stable_strategies,
    int* cache_hits, int* cache_misses);

/* Workspace and weight memory/lifetimes are independent. A workspace is fixed
 * to one exact-M strategy and is non-reentrant: concurrent use returns BUSY.
 * A weight handle can serve any workspace in the same context and IOMMU domain
 * whose K/N/Ktile partition matches. Cross-domain runs return
 * RK_NPU_ERR_DOMAIN before submit. All handles must be freed before rk_npu_ctx. */
int rk_npu_matmul_workspace_memory_query(
    const rk_npu_matmul_strategy* strategy,
    rk_npu_matmul_workspace_requirements* out);
int rk_npu_i8i8i32_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out);
int rk_npu_f16i8f16_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out);
int rk_npu_f32i8f32_weights_memory_query(
    const rk_npu_matmul_weight_config* config,
    rk_npu_matmul_weight_requirements* out);
rk_npu_matmul_workspace* rk_npu_matmul_workspace_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_strategy* strategy);
void rk_npu_matmul_workspace_free(rk_npu_matmul_workspace* workspace);

/* --------------------------- int8 A * int8 B -> compact row-major int32 C */

rk_npu_i8i8i32_weights* rk_npu_i8i8i32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor);
/* Lossless normal DCOMP storage of the same B[K,N]. Same run/free functions.
 * Creation copies/encodes inputs. On first use of a new workspace N partition,
 * its compressed layout is built and cached; warm up before timing execution.
 * Tiles that do not shrink use raw native storage, without changing codes.
 * No duplicate raw weight copy is retained. Multiple used N partitions retain separate
 * compressed variants. Existing shape-only weight_memory_query is RAW-only. */
rk_npu_i8i8i32_weights* rk_npu_i8i8i32_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor);
void rk_npu_i8i8i32_weights_free(rk_npu_i8i8i32_weights* weights);

/* No quantization is performed.  Split-K partials are summed exactly modulo
 * 2^32, matching the NPU INT32 accumulator, and de-tiled into C_rowmajor[M,N]. */
int rk_npu_i8i8i32_run(rk_npu_matmul_workspace* workspace,
                        const rk_npu_i8i8i32_weights* weights,
                        const int8_t* A_rowmajor,
                        int32_t* C_rowmajor);

/* ---------------------------- fp16 A * int8 B -> compact row-major fp16 C */

/* B_rowmajor[K,N] is packed and w_scale[N] is copied into immutable storage.
 * Both inputs may be released after creation. Every scale must be finite and
 * positive. Dynamic and static fp16 workspaces share this weight type. */
rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale);
rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale);
/* Uncalibrated per-output-channel RTN + lossless DCOMP; this is not calibrated
 * V2/GPTQ. B is finite FP32 [K,N]. target_bpw must be finite in (0,8] and bounds
 * the creation-time full-N/K-wave payload (headers/alignment included, scales
 * excluded). Different execution N tiles may change BPW through headers and
 * codebooks. Returns NULL on invalid input, unmet search budget, or allocation
 * failure. Quantization uses nearest-even; zero columns use scale=1. */
rk_npu_f16i8f16_weights* rk_npu_f16i8f16_weights_create_from_f32_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const float* B_rowmajor, float target_bpw);
void rk_npu_f16i8f16_weights_free(rk_npu_f16i8f16_weights* weights);

/* Dynamic per-token symmetric activation quantization:
 *
 *   a_scale[m] = max(abs(A[m,:])) / 127       (all-zero row -> 1)
 *   Aq[m,k]    = clamp(round(A[m,k]/a_scale[m]), -127, 127)
 *   C[m,n]     = fp16(sum_k Aq[m,k]*B[k,n] * a_scale[m]*w_scale[n])
 *
 * Quantization writes each K slice directly into one of two NPU ping-pong
 * buffers; no intermediate compact INT8 A tensor is materialized. */
int rk_npu_f16i8f16_run_dynamic(
                                 rk_npu_matmul_workspace* workspace,
                                 const rk_npu_f16i8f16_weights* weights,
                                 const uint16_t* A_fp16_rowmajor,
                                 uint16_t* C_fp16_rowmajor);

/* Static per-token uses caller-provided a_scale[M] for both quantization and
 * output dequantization.  Every scale must be finite and positive and remains
 * caller-owned for the duration of this blocking call. */
int rk_npu_f16i8f16_run_static(
                                rk_npu_matmul_workspace* workspace,
                                const rk_npu_f16i8f16_weights* weights,
                                const uint16_t* A_fp16_rowmajor,
                                const float* a_scale,
                                uint16_t* C_fp16_rowmajor);

/* ---------------------------- fp32 A * int8 B -> compact row-major fp32 C */

rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale);
/* Same lossless and basic FP32 quantization contracts as the FP16 variants. */
rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const int8_t* B_rowmajor, const float* w_scale);
rk_npu_f32i8f32_weights* rk_npu_f32i8f32_weights_create_from_f32_compress(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_weight_config* config,
    const float* B_rowmajor, float target_bpw);
void rk_npu_f32i8f32_weights_free(rk_npu_f32i8f32_weights* weights);

int rk_npu_f32i8f32_run_dynamic(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f32i8f32_weights* weights,
    const float* A_fp32_rowmajor,
    float* C_fp32_rowmajor);

int rk_npu_f32i8f32_run_static(
    rk_npu_matmul_workspace* workspace,
    const rk_npu_f32i8f32_weights* weights,
    const float* A_fp32_rowmajor,
    const float* a_scale,
    float* C_fp32_rowmajor);

#ifdef __cplusplus
}
#endif
#endif /* RK_NPU_QUANT_MATMUL_H */

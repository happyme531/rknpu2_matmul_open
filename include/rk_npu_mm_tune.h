/* Explicit, complete-call tuning for compact, static-weight FP16/BF16 MM.
 * Uses the raw MM executor; no quantization or precision substitution. */
#ifndef RK_NPU_MM_TUNE_H
#define RK_NPU_MM_TUNE_H
#include "rk_npu_mm.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RK_NPU_MM_HOST_STORAGE = 0, /* A/C are dtype's uint16_t bit patterns */
    RK_NPU_MM_HOST_F32 = 1      /* convert A to dtype; widen C and add FP32 bias */
} rk_npu_mm_host_type;

typedef struct rk_npu_mm_autotune_config {
    int M, N, K; /* logical, compact row-major A[M,K], B[K,N], C[M,N] */
    rk_npu_mm_dtype dtype; /* logical F16/BF16 inputs and final output */
    rk_npu_mm_host_type host_type;
    int with_bias; /* optional FP32 bias[N]; storage output rounds after bias */
    uint32_t allowed_npu_core_mask; /* 0 means 7; fixed throughout search */
    uint64_t allowed_cpu_core_mask; /* 0: current affinity, up to 4 big cores */
    int warmup, loops, repeats;
    uint32_t timeout_ms;
    int allow_k_tiling; /* 0: full K; 1: FP16 partials + NPU FP16 Add tree.
                          Init reads RK_NPU_MM_K_TILING (off/0 or fp16/on/1),
                          enabled by default. Ignored for BF16. */
    int verbose; /* Progress/phase ETA on stderr; 1 adds candidates/scores,
                    2 adds repeat samples. Cache hits do not print progress. */
} rk_npu_mm_autotune_config;

typedef struct rk_npu_mm_strategy {
    int M, N, K;
    rk_npu_mm_dtype dtype;
    rk_npu_mm_host_type host_type;
    int with_bias;
    int m_tile; /* physical rows per call; final logical tail is zero-padded */
    int task_m_tile; /* 0: CBUF default; otherwise cap rows per hardware task */
    int fused_io; /* FP16/F32 host IO: fuse conversion with pack/collect */
    int k_tile; /* 0: full K; otherwise FP16 partials + NPU FP16 Add tree */
    rk_npu_mm_options options; /* tuned execution requires split_k=0 */
    uint64_t cpu_core_mask; /* caller applies this affinity and thread count */
    int cpu_threads;
    uint32_t tuning_revision;
    double total_us, jitter_pct, robust_us; /* complete logical call */
    double workspace_us, weight_us; /* construction/upload, excluded from total */
    double max_abs_error; /* full synthetic output versus independent CPU oracle */
} rk_npu_mm_strategy;

typedef struct rk_npu_mm_tuned_workspace rk_npu_mm_tuned_workspace;
typedef struct rk_npu_mm_tuned_weights rk_npu_mm_tuned_weights;

/* Defaults: storage FP16, no bias, FP16 K tiling enabled,
 * 1 warmup, 3 loops, 3 repeats, 500 ms timeout, all permitted NPU cores.
 * RK_NPU_MM_K_TILING=off/0 disables K tiling for numerical comparisons.
 * Enabled K tiling always uses FP16 partials and an NPU FP16 Add tree.
 * Environment is read during preparation, never inside tuned_run. */
void rk_npu_mm_autotune_config_init(rk_npu_mm_autotune_config*, int M, int N, int K);
/* Native/native M128 baseline used on cache misses. CPU-only. */
int rk_npu_mm_strategy_default(const rk_npu_mm_autotune_config*, rk_npu_mm_strategy*);

/* FP16 searches normal/native/panel8/panel16 A/C recipes, host M/N/K tiles,
 * hardware task M caps, NPU FP16 partial reduction and staged/fused FP32 IO.
 * A coarse layout/M/N/K pass seeds full-repeat task/IO refinement;
 * only refined candidates rank. K partitions and reduction stay in this library.
 * BF16 retains the normal/native, host M/N search and staged conversion.
 * Uses deterministic, dtype-rounded calibration and validates every output.
 * Measures the same tuned_run below, including conversion, padding, packing,
 * cache sync, submit/wait, collection and optional FP32 bias. Packed B creation
 * and workspace construction are reported separately. Submit failures abort.
 * CPU affinity and OpenMP settings are restored on return. */
int rk_npu_mm_autotune(rk_npu_iommu_domain*, const rk_npu_mm_autotune_config*,
                       rk_npu_mm_strategy* fastest, rk_npu_mm_strategy* stable);
/* Cache key includes logical M/N/K, dtype, host IO/bias, resource masks,
 * split policy, measurement policy, driver and scheduling revision. Entries
 * are shape profiles, independent of weight values. Cache load performs no
 * device work, search or writes. NULL directory uses rk_npu_mm's namespace. */
int rk_npu_mm_autotune_cache_load(rk_npu_ctx*, const rk_npu_mm_autotune_config*,
    const char* directory, rk_npu_mm_strategy* fastest, rk_npu_mm_strategy* stable);
int rk_npu_mm_autotune_cached(rk_npu_iommu_domain*, const rk_npu_mm_autotune_config*,
    const char* directory, int refresh, rk_npu_mm_strategy* fastest,
    rk_npu_mm_strategy* stable, int* cache_hit);
/* Library-owned resident weights. Uploads the baseline and K partitions
 * selected by previously tuned shapes of this resource family. directory=NULL
 * uploads the baseline only. Normal runs never repack or scan original weights.
 * B is compact dtype uint16_t B[K,N]; logical M in config does not bind weights. */
int rk_npu_mm_tuned_weights_create(rk_npu_iommu_domain*, const rk_npu_mm_autotune_config*,
    const uint16_t* B, uint64_t bytes, const char* directory, rk_npu_mm_tuned_weights** out);
void rk_npu_mm_tuned_weights_free(rk_npu_mm_tuned_weights*);
/* OK if resident weights support the strategy; CACHE_MISS for a new partition
 * not present when weights were loaded; PARAM for incompatible shapes/dtype. */
int rk_npu_mm_tuned_weights_match(const rk_npu_mm_tuned_weights*, const rk_npu_mm_strategy*);

/* B always contains dtype's compact uint16_t B[K,N], even for HOST_F32.
 * Packed weights are reusable across M/layout/N-tile choices with the same
 * dtype/K/N/split partition and domain, as specified by rk_npu_mm.h. Use this
 * factory again when changing split-K. ctx must outlive all device objects. */
int rk_npu_mm_tuned_packed_b_create(rk_npu_iommu_domain*, const rk_npu_mm_strategy*,
    const uint16_t* B, uint64_t bytes, rk_npu_mm_packed_b** out);
int rk_npu_mm_tuned_workspace_create(rk_npu_iommu_domain*, const rk_npu_mm_strategy*,
    rk_npu_mm_tuned_workspace** out);
void rk_npu_mm_tuned_workspace_free(rk_npu_mm_tuned_workspace*);
/* No allocation or policy search on run. A/C are compact, match host_type,
 * have M*K / M*N elements and must not overlap. bias is required iff with_bias.
 * Workspaces are non-reentrant. C is undefined on error. FP32 output widens
 * the dtype-rounded MM result. Storage output adds bias then rounds to dtype.
 * K tiling uses FP16 partials and an NPU FP16
 * pairwise Add tree, with one rounding per level. Set RK_NPU_MM_K_TILING=off
 * during preparation to compare with the full-K numerical path. */
int rk_npu_mm_tuned_run(rk_npu_mm_tuned_workspace*, const void* A, uint64_t a_bytes,
    const rk_npu_mm_packed_b*, const float* bias, void* C, uint64_t c_bytes);
/* Same complete call with library-owned weights; partition selection stays
 * inside the operator library. No allocation, repack, search or cache writes. */
int rk_npu_mm_tuned_run_weights(rk_npu_mm_tuned_workspace*, const void* A, uint64_t a_bytes,
    const rk_npu_mm_tuned_weights*, const float* bias, void* C, uint64_t c_bytes);

#ifdef __cplusplus
}
#endif
#endif

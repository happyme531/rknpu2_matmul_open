/* W4A8 linear: symmetric signed INT4 weights + FP32 per-output-channel scales.
 * FP32/FP16 activations are dynamically quantized per token to INT8, split
 * exactly into two signed INT4 rows, and executed by the W4A4 backend.
 *
 * No caller-supplied decomposition bias: creation computes 8*sum_k B[k,n].
 * This is not a model's additive linear bias; add that separately if needed.
 * No weight zero point, group-wise scales or model rotation is implied.
 */
#ifndef RK_NPU_W4A8_H
#define RK_NPU_W4A8_H
#include "rk_npu_matmul_i4.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct rk_npu_w4a8_config {
    int M, N, K;            /* logical token rows; K <= INT32_MAX/1024 */
    int m_tile;             /* logical tokens, legacy 1..64; _ex panel8 allows more; physical rows doubled */
    int k_tile;             /* resolved 32..2048, multiple of 32; fixed in shared weights */
    int n_tile;             /* multiple of 64, 64..4096 */
    uint32_t npu_core_mask; /* 1/2/4/3/7, same N64 alignment as W4A4 */
    uint32_t timeout_ms;
    int pipeline;    /* 0: serial; 1: overlapped pack/submit/reduce */
    int cpu_threads; /* 1..4; caller/OpenMP runtime controls CPU affinity */
} rk_npu_w4a8_config;

typedef struct rk_npu_w4a8_memory_info {
    rk_npu_i4_memory_info backend;  /* physical 2*M native buffers and tasks */
    uint64_t weight_metadata_bytes; /* FP32 scale[N] + INT32 correction[N] */
    uint64_t cpu_scratch_bytes;     /* native INT32 M*align64(N) + two float[M] arrays */
} rk_npu_w4a8_memory_info;

/* Opt-in execution policies. Existing APIs keep native input + CPU reduction.
 * Panel8 permits m_tile=4..256 in multiples of 4, within an 8-bank CBUF budget;
 * incomplete final groups use native input. Weights do not depend on layout.
 * NPU reduction retains all K partials, then uses the same selected NPU cores
 * for exact INT32 reduction. It does NOT reserve a separate reduction core.
 * For M<=16, a requested NPU reduction uses CPU reduction instead, without
 * allocating reduction buffers. Input layout and tile configuration are kept.
 * Otherwise supported: <=31 waves, each tile's logical_tokens*padded_N/8 <=4096.
 * Unsupported geometry returns PARAM from memory_query_ex; no CPU fallback.
 */
typedef enum rk_npu_w4a8_input_layout {
    RK_NPU_W4A8_INPUT_NATIVE = 0,
    RK_NPU_W4A8_INPUT_PANEL8 = 1
} rk_npu_w4a8_input_layout;
typedef enum rk_npu_w4a8_reduce_backend {
    RK_NPU_W4A8_REDUCE_CPU = 0,
    RK_NPU_W4A8_REDUCE_NPU = 1
} rk_npu_w4a8_reduce_backend;
typedef struct rk_npu_w4a8_options {
    uint32_t struct_size; /* initialize with options_init */
    rk_npu_w4a8_input_layout input_layout;
    rk_npu_w4a8_reduce_backend reduce_backend;
} rk_npu_w4a8_options;
typedef struct rk_npu_w4a8_memory_info_ex {
    rk_npu_w4a8_memory_info base; /* partial_bytes includes retained guard in NPU mode */
    uint64_t reduction_output_bytes; /* includes mapped DMA guard */
    uint64_t reduction_weight_bytes;
    uint64_t reduction_control_bytes;
    /* Sum of requested workspace device buffers + CPU tensor scratch.
     * Excludes shared packed weights/metadata, driver page rounding, host
     * descriptors/objects, allocator overhead and OpenMP/worker stacks.
     * Fixed for this workspace: no growth with runs or weight rebindings. */
    uint64_t workspace_bytes;
    rk_npu_w4a8_reduce_backend effective_reduce_backend; /* includes M<=16 policy */
} rk_npu_w4a8_memory_info_ex;
typedef struct rk_npu_w4a8_timings {
    double activation_scan_us; /* complete-row scale + finite-input check */
    double quant_pack_us, sync_us, submit_us, reduce_us;
    double dequant_us, total_us; /* stages overlap; do not sum to total.
                                  CPU mode: final dequant is fused into reduce_us.
                                  NPU mode: reduce_us is blocking Conv submission;
                                  dequant_us is CPU correction/scale/output. */
} rk_npu_w4a8_timings;
typedef struct rk_npu_w4a8_weights rk_npu_w4a8_weights;
typedef struct rk_npu_w4a8_workspace rk_npu_w4a8_workspace;

typedef struct rk_npu_w4a8_weight_info {
    rk_npu_i4_weight_config config; /* actual K/N/k_tile for the workspace */
    int safe_k_tile;                /* certified tile before experimental scaling */
    double multiplier;              /* default 1; read only for auto selection */
    int bound_relaxed;              /* output may saturate; no runtime check/retry */
} rk_npu_w4a8_weight_info;

void rk_npu_w4a8_config_init(rk_npu_w4a8_config *cfg, int M, int N, int K);
int rk_npu_w4a8_memory_query(const rk_npu_w4a8_config *cfg, rk_npu_w4a8_memory_info *out);
void rk_npu_w4a8_options_init(rk_npu_w4a8_options *options);
int rk_npu_w4a8_memory_query_ex(const rk_npu_w4a8_config *cfg,
                                const rk_npu_w4a8_options *options,
                                rk_npu_w4a8_memory_info_ex *out);

/* B[K,N]: one int8_t per logical INT4 code [-8,7], zero point 0.
 * w_scale[N]: finite, positive FP32 scales, copied unchanged.
 * Storage is native packed INT4; no persistent unpacked B copy. Original
 * B/scales may be released after creation. Uses the W4A4 K/N/k_tile key.
 * One immutable weight object serves both f32/f16 workspaces and any M.
 * cfg->k_tile=0 selects the largest weight-certified fixed tile <=2048.
 * Auto mode reads RK_NPU_W4A8_K_TILE_MULTIPLIER (finite >=1, default 1).
 * selected=floor(safe_k_tile*multiplier/32)*32, capped at min(align32(K),2048).
 * multiplier>1 is EXPERIMENTAL: it may permit INT16 saturation and incorrect
 * output. No runtime detection or retry is performed. Explicit k_tile ignores
 * the environment and is strictly checked. Query the resolved config before
 * creating a workspace (workspace k_tile=0 is not accepted).
 * NULL on invalid input/allocation failure. Context outlives all handles. */
rk_npu_w4a8_weights *rk_npu_w4a8_weights_create(rk_npu_iommu_domain *domain,
                                                const rk_npu_i4_weight_config *cfg, const int8_t *B,
                                                const float *w_scale);
void rk_npu_w4a8_weights_free(rk_npu_w4a8_weights *weights);
int rk_npu_w4a8_weights_query(const rk_npu_w4a8_weights *weights, rk_npu_w4a8_weight_info *info);
rk_npu_w4a8_workspace *rk_npu_w4a8_workspace_create(rk_npu_iommu_domain *domain,
                                                    const rk_npu_w4a8_config *cfg);
/* All tensor/device scratch is owned by the workspace, allocated here and
 * released by workspace_free. Reuse across sequential calls/layers having
 * the same M/N/K/k_tile and domain; each concurrent call needs its own workspace.
 * NULL options selects the legacy native/CPU policy. */
rk_npu_w4a8_workspace *rk_npu_w4a8_workspace_create_ex(rk_npu_iommu_domain *domain,
    const rk_npu_w4a8_config *cfg, const rk_npu_w4a8_options *options);
void rk_npu_w4a8_workspace_free(rk_npu_w4a8_workspace *workspace);

/* Blocking, non-reentrant (BUSY on concurrent use). A/C compact row-major,
 * non-aliasing, shape [M,K]/[M,N]. FP16 uses IEEE binary16 bits in uint16_t.
 * Reject NaN/Inf activations before submit. All-zero row uses scale=1.
 *
 * s[m] = max(maxabs(A[m,:])/127, FLT_MIN), for a nonzero row.
 * q = clamp(round_away(A * float(1/s)), -127,127).
 * The SAME full-row scale is used by every K slice. Quantization writes
 * packed high/low rows directly; no complete INT8 activation is materialized.
 * C = (float(sum_k q*B) * s[m]) * w_scale[n], then cast once for FP16.
 * With a certified tile, integer MSD reconstruction is exact; the experimental
 * multiplier waives this guarantee. Activation quantization and the final
 * floating-point conversion still round. FP16/FP32 outputs may overflow.
 * Failure leaves C unspecified. timings may be NULL. No per-run tensor/device allocation.
 */
int rk_npu_w4a8_run_f32(rk_npu_w4a8_workspace *workspace, const rk_npu_w4a8_weights *weights,
                        const float *A, float *C, rk_npu_w4a8_timings *timings);
int rk_npu_w4a8_run_f16(rk_npu_w4a8_workspace *workspace, const rk_npu_w4a8_weights *weights,
                        const uint16_t *A, uint16_t *C, rk_npu_w4a8_timings *timings);

#ifdef __cplusplus
}
#endif
#endif

/* FlatQuant-oriented W4A4 linear.
 *
 * The immutable handle owns signed INT4 weights, FP32 per-output-channel
 * scales and an optional Kronecker input transform. A complete call performs
 * transform + per-token range collection, clipped symmetric INT4 quantization
 * directly into the native NPU layout, exact split-K reduction, and fused
 * dequantization to FP32 or IEEE binary16 output.
 */
#ifndef RK_NPU_W4A4_LINEAR_H
#define RK_NPU_W4A4_LINEAR_H

#include "rk_npu_matmul_i4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rk_npu_w4a4_linear_config {
    int M, N, K;
    int m_tile, k_tile, n_tile;
    uint32_t npu_core_mask, timeout_ms;
    int pipeline, cpu_threads;
    /* Zero selects a bounded default. batch is the CPU/NPU preprocessing
     * pipeline block (<=256); ubatch groups tokens inside each transform
     * chain (<=32). These do not change the main INT4 M tile. */
    int transform_batch, transform_ubatch;
} rk_npu_w4a4_linear_config;

typedef struct rk_npu_w4a4_transform {
    uint32_t struct_size; /* initialize with rk_npu_w4a4_transform_init */
    int left_dim;
    int right_dim;      /* left_dim*right_dim must equal K */
    const float *left;  /* row-major [left_dim,left_dim] */
    const float *right; /* row-major [right_dim,right_dim] */
    /* FlatQuant LAC ratios after sigmoid, finite in (0,1]. Negative and
     * positive extrema are clipped independently before selecting maxabs/7. */
    float negative_clip_ratio;
    float positive_clip_ratio;
} rk_npu_w4a4_transform;

typedef struct rk_npu_w4a4_linear_memory_info {
    rk_npu_i4_memory_info backend;
    uint64_t weight_metadata_bytes;     /* w_scale + packed FP16 transform weights */
    uint64_t cpu_scratch_bytes;         /* INT4 A, token scales, INT32 accumulator */
    uint64_t workspace_bytes;           /* backend + CPU scratch + transform storage */
    uint64_t transform_workspace_bytes; /* fixed NPU transform buffers/control */
} rk_npu_w4a4_linear_memory_info;

typedef struct rk_npu_w4a4_linear_timings {
    double transform_scan_us; /* complete preprocessing wall time, including A4 packing */
    double quant_pack_us;
    double sync_us;
    double submit_us;
    double reduce_dequant_us;
    double total_us; /* stages overlap; do not sum the fields */
    double transform_pack_us;
    double transform_submit_us;
    double activation_scan_us;
} rk_npu_w4a4_linear_timings;

typedef struct rk_npu_w4a4_linear_weight_info {
    rk_npu_i4_weight_config config; /* resolved K/N/k_tile */
    int left_dim, right_dim;        /* zero means identity */
    float negative_clip_ratio, positive_clip_ratio;
} rk_npu_w4a4_linear_weight_info;

typedef struct rk_npu_w4a4_linear_weights rk_npu_w4a4_linear_weights;
typedef struct rk_npu_w4a4_linear_workspace rk_npu_w4a4_linear_workspace;

void rk_npu_w4a4_linear_config_init(rk_npu_w4a4_linear_config *cfg, int M, int N, int K);
/* NULL transform or init followed by left_dim=right_dim=0 selects identity. */
void rk_npu_w4a4_transform_init(rk_npu_w4a4_transform *transform);
int rk_npu_w4a4_linear_memory_query(const rk_npu_w4a4_linear_config *cfg,
                                    const rk_npu_w4a4_transform *transform,
                                    rk_npu_w4a4_linear_memory_info *out);

/* B is row-major [K,N], one int8_t per signed INT4 code. w_scale[N] is
 * finite and positive. Transform matrices and metadata are copied. k_tile=0
 * selects the largest weight-certified 32-aligned tile up to 2048. No unsafe
 * multiplier or runtime saturation retry is used. No linear bias is applied. */
rk_npu_w4a4_linear_weights *
rk_npu_w4a4_linear_weights_create(rk_npu_iommu_domain *domain, const rk_npu_i4_weight_config *cfg,
                                  const int8_t *B, const float *w_scale,
                                  const rk_npu_w4a4_transform *transform);
void rk_npu_w4a4_linear_weights_free(rk_npu_w4a4_linear_weights *weights);
int rk_npu_w4a4_linear_weights_query(const rk_npu_w4a4_linear_weights *weights,
                                     rk_npu_w4a4_linear_weight_info *out);

/* Weight metadata fixes transform buffer/command geometry at creation.
 * All storage and submit workers are prepared before the first run. */
rk_npu_w4a4_linear_workspace *
rk_npu_w4a4_linear_workspace_create(rk_npu_iommu_domain *domain,
                                    const rk_npu_w4a4_linear_config *cfg,
                                    const rk_npu_w4a4_linear_weights *weights);
void rk_npu_w4a4_linear_workspace_free(rk_npu_w4a4_linear_workspace *workspace);

/* A/C are compact row-major [M,K]/[M,N], non-aliasing. FP16 is carried as
 * uint16_t IEEE binary16 bits. Transform semantics match FlatQuant:
 * reshape each token to [left_dim,right_dim], then (left^T @ X) @ right.
 * Non-identity transforms use FP16 NPU GEMMs (including FP16 intermediate
 * rounding), with transpose fused into the first writeback. Dimensions L/R
 * are currently supported in 1..128, with padding handled internally.
 * FP32 input is rounded to FP16 before a non-identity transform. The identity
 * path scans/quantizes the original input without a dense transform or copy.
 *
 * xmin=min(A',0)*negative_clip_ratio; xmax=max(A',0)*positive_clip_ratio;
 * s=max(max(abs(xmin),xmax)/7,FLT_MIN) (s=1 for an all-zero row),
 * q=clamp(round-to-nearest-even(A'/s),-8,7),
 * C=(float(sum_k q*B)*s)*w_scale. Nonfinite inputs are rejected before any
 * submit; nonfinite transform outputs are rejected before the INT4 submit.
 * The workspace is prepared for the supplied weights' transform dimensions;
 * another weight handle with the same dimensions/config/domain may be used.
 * No per-run tensor or device allocation is performed. The workspace is
 * blocking and non-reentrant. Failure leaves C unspecified. */
int rk_npu_w4a4_linear_run_f32(rk_npu_w4a4_linear_workspace *workspace,
                               const rk_npu_w4a4_linear_weights *weights, const float *A, float *C,
                               rk_npu_w4a4_linear_timings *timings);
int rk_npu_w4a4_linear_run_f16(rk_npu_w4a4_linear_workspace *workspace,
                               const rk_npu_w4a4_linear_weights *weights, const uint16_t *A,
                               uint16_t *C, rk_npu_w4a4_linear_timings *timings);

#ifdef __cplusplus
}
#endif
#endif

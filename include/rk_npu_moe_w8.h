/* Routed W8A8 SwiGLU FFN, with an optional shared expert. Router is external.
 * All pointers are host pointers; calls block until output is ready.
 * Weights and scratch are independent so sequential layers reuse a workspace.
 */
#ifndef RK_NPU_MOE_W8_H
#define RK_NPU_MOE_W8_H
#include "rk_npu_common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum rk_npu_moe_middle {
    RK_NPU_MOE_MIDDLE_CPU = 0,
    /* FP16-rounded gate/up scales, approximate SiLU LUT on [-8,8].
     * FP32 main tensors, 4096x internal carrier. Requires intermediate=512.
     * Refreshes LUT immediately before use; no power-management writes.
     * This changes numerical results and requires model-quality validation.
     * Caller must serialize competing NPU users across this blocking call,
     * since SDK/other operators can change the hardware LUT. */
    RK_NPU_MOE_MIDDLE_NPU_LUT = 1
} rk_npu_moe_middle;

typedef struct rk_npu_moe_w8_weight_config {
    int hidden;       /* 64..1536, multiple of 32 */
    int intermediate; /* 32..512, multiple of 32 */
    int routed_experts; /* 1..128 */
    int shared_expert;  /* 0 or 1; if present, expert index routed_experts */
} rk_npu_moe_w8_weight_config;

typedef struct rk_npu_moe_w8_config {
    uint32_t struct_size;
    rk_npu_moe_w8_weight_config weights;
    int max_rows;      /* 1..256; each run may use any smaller positive count */
    int top_k;         /* 1..min(routed_experts,8) */
    int npu_cores;     /* 1/2/3, physical cores 0..npu_cores-1 */
    int cpu_threads;   /* 1..4; caller/OpenMP controls CPU affinity */
    int native_input;  /* 0 normal-A, 1 native K16/M16 */
    rk_npu_moe_middle middle;
    uint32_t timeout_ms;
} rk_npu_moe_w8_config;

typedef struct rk_npu_moe_w8_expert {
    /* Compact signed INT8, zero point 0, gate columns followed by up columns.
     * Original arrays may be released after weights_create returns. */
    const int8_t* gate_up; /* [hidden, 2*intermediate] */
    const float* gate_up_scales; /* [2*intermediate], positive finite */
    const int8_t* down; /* [intermediate, hidden] */
    const float* down_scales; /* [hidden], positive finite */
} rk_npu_moe_w8_expert;

typedef struct rk_npu_moe_w8_timings {
    double input_us, schedule_us, build_us, sync_us, gather_us;
    double gate_us, middle_npu_us, middle_cpu_us, down_us, combine_us;
    double lut_load_us;
    double total_us; /* Includes actual LUT refresh, unlike hot-path probe tables. */
    uint32_t gather_tasks, middle_tasks, gemm_tasks;
} rk_npu_moe_w8_timings;

typedef struct rk_npu_moe_w8_weights rk_npu_moe_w8_weights;
typedef struct rk_npu_moe_w8_workspace rk_npu_moe_w8_workspace;

/* Defaults: three NPU cores, four CPU threads, normal A, CPU middle,
 * NPU gather (always), timeout 1000ms. Config is copied at creation. */
void rk_npu_moe_w8_config_init(rk_npu_moe_w8_config* config,
    int hidden, int intermediate, int routed_experts, int shared_expert,
    int top_k, int max_rows);

/* Creates immutable packed weights for routed_experts+shared_expert entries.
 * Context must outlive every handle; input domain handle may be freed later.
 * Returns NULL on invalid configuration, coefficients or allocation failure. */
rk_npu_moe_w8_weights* rk_npu_moe_w8_weights_create(rk_npu_iommu_domain* domain,
    const rk_npu_moe_w8_weight_config* config, const rk_npu_moe_w8_expert* experts);
void rk_npu_moe_w8_weights_free(rk_npu_moe_w8_weights* weights);
rk_npu_moe_w8_workspace* rk_npu_moe_w8_workspace_create(rk_npu_iommu_domain* domain,
    const rk_npu_moe_w8_config* config);
void rk_npu_moe_w8_workspace_free(rk_npu_moe_w8_workspace* workspace);

/* Counts requested device payload bytes (no page rounding or CPU containers). */
uint64_t rk_npu_moe_w8_weights_device_bytes(const rk_npu_moe_w8_weights* weights);
uint64_t rk_npu_moe_w8_workspace_device_bytes(const rk_npu_moe_w8_workspace* workspace);

/* IDs/coefficients are row-major [rows,top_k]. IDs are in [0,routed_experts),
 * coefficients are finite and nonnegative; duplicates are allowed and summed
 * in slot order. The optional shared expert is appended with coefficient 1.
 * No implicit top-k normalization. Output is compact FP32 [rows,hidden].
 *
 * Quantized input: compact INT8 [rows,hidden], one positive finite FP32 scale
 * per source row. No re-quantization of input. FP32 input: quantized once per
 * token using nearest-even rounding and maxabs/127 (zero rows use scale 1).
 * The internal hidden activation uses the same nearest-even rule.
 * Input/output arrays must not overlap. Invalid input is rejected before
 * submission; a later failure leaves output unspecified. No reset or retry.
 * Workspace is non-reentrant (BUSY); immutable compatible weights can be
 * rebound on each run, but must belong to the same context and IOMMU domain.
 */
int rk_npu_moe_w8_run_quantized(rk_npu_moe_w8_workspace* workspace,
    const rk_npu_moe_w8_weights* weights, int rows, const int8_t* input,
    const float* row_scales, const int32_t* expert_ids, const float* route_weights,
    float* output, rk_npu_moe_w8_timings* timings);
int rk_npu_moe_w8_run_f32(rk_npu_moe_w8_workspace* workspace,
    const rk_npu_moe_w8_weights* weights, int rows, const float* input,
    const int32_t* expert_ids, const float* route_weights, float* output,
    rk_npu_moe_w8_timings* timings);

#ifdef __cplusplus
}
#endif
#endif

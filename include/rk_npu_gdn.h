/* Scalar-gated DeltaNet prefill: TF32 products, FP32 recurrent state.
 * CPU Conv/SiLU/L2, coefficients and triangular solve; NPU Gram,
 * state prediction, block output and state update. */
#ifndef RK_NPU_GDN_H
#define RK_NPU_GDN_H
#include "rk_npu_common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RK_NPU_GDN_TF32 = 0,
    /* Two TF32 input components; three products per logical GEMM. */
    RK_NPU_GDN_COMPENSATED_TF32 = 1,
} rk_npu_gdn_precision;
typedef enum {
    RK_NPU_GDN_HEAD_GROUPED = 0,    /* key_head = value_head / 2 */
    RK_NPU_GDN_HEAD_INTERLEAVED = 1 /* key_head = value_head % 16 */
} rk_npu_gdn_head_mapping;
typedef struct {
    int tokens;                       /* fixed prepared window, 1..2048 */
    int key_heads, value_heads;        /* currently 16/32 */
    int key_dim, value_dim;            /* currently 128/128 */
    int chunk_size;                    /* 16/32/64/128; default 64; see task limit below */
    int cpu_threads;                   /* 1..4; caller controls CPU affinity */
    uint32_t core_mask;                /* 1/2/4, 3 or 7; default 7 */
    uint32_t timeout_ms;               /* positive; default 6000 */
    rk_npu_gdn_precision precision;
    rk_npu_gdn_head_mapping head_mapping;
} rk_npu_gdn_config;
typedef struct {
    uint64_t query_bytes, key_bytes, value_bytes, gate_bytes, beta_bytes;
    uint64_t state_bytes, output_bytes;
    uint64_t projected_bytes, conv_weight_bytes, conv_history_bytes;
} rk_npu_gdn_sizes;
typedef struct {
    double total_us, prepare_us, coefficients_us, solve_us, collect_us;
    double hardware_us, gram_hardware_us, predict_hardware_us, update_hardware_us;
    double cpu_user_us, cpu_system_us;
} rk_npu_gdn_timings;
typedef struct rk_npu_gdn_workspace rk_npu_gdn_workspace;

/* Defaults: C7 16/32 heads, d128, grouped heads, C64, CPU4, three NPU cores.
 * Query validates the fixed shape without opening a device. A Gram chain is
 * limited to 4095 tasks: compensated chunk16 supports T<=816, chunk32 T<=1632;
 * chunk64/128 and ordinary TF32 support T<=2048. */
void rk_npu_gdn_config_init(rk_npu_gdn_config* config, int tokens);
int rk_npu_gdn_query(const rk_npu_gdn_config* config, rk_npu_gdn_sizes* sizes);
/* Preparation allocates all device/CPU scratch, persistent CPU workers and
 * register chains. No data-buffer allocation or graph construction in run.
 * The workspace retains the domain; ctx must outlive the workspace. Multiple
 * layer states can share one workspace of the same configuration. */
int rk_npu_gdn_prepare(rk_npu_iommu_domain* domain,
    const rk_npu_gdn_config* config, rk_npu_gdn_workspace** out);
void rk_npu_gdn_workspace_free(rk_npu_gdn_workspace* workspace);

/* One sequence, contiguous FP32 tensors:
 * Q/K [T,16,128], V/O [T,32,128], g/beta [T,32], state [32,128,128].
 * State axes are [value_head,value_channel,key_channel], key contiguous.
 * Q/K must already be L2-normalized. Q is unscaled; this operator applies
 * 1/sqrt(128) exactly once. g is log decay <=0 (-infinity is supported),
 * beta is already sigmoid in [0,1]. Other inputs must be finite.
 * State decay precedes delta=beta*(v-state*k), rank-1 update and output.
 * final_state may equal initial_state; other output/input spans must not
 * overlap. The caller allocates the spans returned by query().
 * Calls block and include packing, sync, submit and canonical state export.
 * Concurrent use of one workspace returns ERR_BUSY. Different workspaces
 * require caller scheduling on shared physical NPU cores. Free must not race
 * with a call. On failure O is
 * undefined; initial_state is unchanged, including in-place state calls.
 * A device failure requires recreating the workspace. timings may be NULL. */
int rk_npu_gdn_run(rk_npu_gdn_workspace* workspace,
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, const float* initial_state,
    float* output, float* final_state, rk_npu_gdn_timings* timings);

/* Fused CPU producer: K4 causal Conv + SiLU + fixed-order Q/K L2 norm
 * writes GDN native inputs directly. No full Conv or normalized Q/K tensor.
 * projected [T,8192] concatenates Q(2048), K(2048), V(4096) per token.
 * conv_weights [8192,4] are oldest-to-current taps, no bias;
 * history/final_history [8192,3] contain original projected inputs,
 * oldest first. Q/K norm uses epsilon=1e-6. g/beta/state follow run().
 * final_history may equal history. History and initial_state remain unchanged
 * on failure. Conv history and state are supplied and owned by the caller. */
int rk_npu_gdn_run_projected(rk_npu_gdn_workspace* workspace,
    const float* projected, const float* conv_weights, const float* history,
    const float* g, const float* beta, const float* initial_state,
    float* output, float* final_state, float* final_history,
    rk_npu_gdn_timings* timings);
#ifdef __cplusplus
}
#endif
#endif

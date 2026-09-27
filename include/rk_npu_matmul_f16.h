/*
 * rk_npu_matmul_f16 - fp16 x fp16 -> fp16 matmul on the RK3588 NPU
 * =================================================================
 * Direct register programming through the rknpu DRM driver.  A, B, C0, and D
 * are fp16 bit patterns passed as uint16_t because the C ABI has no fp16 type.
 *
 * Supported operations:
 *   D = A @ B
 *   D = (A @ B) .* C0   (vendor ConvMul-style fused elementwise multiply)
 *   D = (A @ B) .+ C0   (vendor ConvAdd-style fused elementwise add)
 *
 * The fused operand C0 is not read row-major by the hardware.  Always stage it
 * through rk_npu_matmul_f16_pack_operand().
 */
#ifndef RK_NPU_MATMUL_F16_H
#define RK_NPU_MATMUL_F16_H

#include "rk_npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RK_NPU_FUSE_NONE = 0,   /* D = A@B; no operand buffer is used. */
    RK_NPU_FUSE_MUL  = 1,   /* D = (A@B) .* C0; operand buffer is required. */
    RK_NPU_FUSE_ADD  = 2,   /* D = (A@B) .+ C0; operand buffer is required. */
} rk_npu_fuse_op;

typedef enum {
    RK_NPU_F16_A_LAYOUT_NORMAL = 0,
    /* Native feature layout: A_native[k/8][m][k%8]. */
    RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8 = 1,
} rk_npu_f16_a_layout;

typedef enum {
    RK_NPU_F16_D_LAYOUT_NORMAL_PADDED = 0,
    /* Native output layout: D_native[n/8][m][n%8]. */
    RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8 = 1,
} rk_npu_f16_d_layout;

/*
 * Complete fixed-shape fp16 plan configuration.
 *
 * M/N/K, op, n_tile, and core_mask determine buffer sizes, packing layout,
 * register programming, and task partitioning.  A prepared plan stores a copy
 * of the config.
 */
typedef struct rk_npu_matmul_f16_config {
    int M;
    int N;
    int K;
    rk_npu_fuse_op op;

    /* Native A/D layouts expose the fast AC-layout datapath used by the
     * official matmul runtime.  pack_a() and unpack_d() keep the public host
     * tensors compact row-major, so callers only select the device layout.
     * Fused MUL/ADD currently require both fields to stay NORMAL. */
    rk_npu_f16_a_layout a_layout;
    rk_npu_f16_d_layout d_layout;

    /* Optional output-channel tile.  Zero keeps one task group across aligned
     * N.  A positive value must be a multiple of 32 and no larger than
     * align32(N).  Tiled output is still unpacked to the same compact row-major
     * public tensor.  Fused MUL/ADD currently require zero. */
    int n_tile;

    /* Physical RK3588 NPU permission mask.  Supported values are 1/2/4, 3 and
     * 7.  Zero preserves source compatibility and means core 0 (mask 1).
     * Independent M/N/batch tasks are divided across the selected cores. */
    uint32_t core_mask;

    /* Blocking submit timeout.  Zero selects the library default (6000 ms). */
    uint32_t timeout_ms;
} rk_npu_matmul_f16_config;

typedef struct rk_npu_matmul_f16_plan rk_npu_matmul_f16_plan;
typedef struct rk_npu_matmul_f16_batch_plan rk_npu_matmul_f16_batch_plan;
typedef struct rk_npu_matmul_f16_splitk_plan rk_npu_matmul_f16_splitk_plan;

/* Initialize cfg for one fixed-shape fp16 matmul/fused-op plan. */
void rk_npu_matmul_f16_config_init(rk_npu_matmul_f16_config* cfg,
                                   int M, int N, int K, rk_npu_fuse_op op);

/* Compute NPU buffer sizes for cfg.
 * operand_bytes is 0 when cfg->op is RK_NPU_FUSE_NONE. */
int rk_npu_matmul_f16_query(const rk_npu_matmul_f16_config* cfg,
                            rk_npu_matmul_sizes* out);

/* Pack compact host A[M][K] and B[K][N] fp16 bit-pattern tensors into the NPU
 * feature/weight layouts.  The destination buffers must be CPU mapped and sized
 * according to query(). */
int rk_npu_matmul_f16_pack_a(const rk_npu_matmul_f16_config* cfg,
                             const uint16_t* A_rowmajor, rk_npu_mem* input);
int rk_npu_matmul_f16_pack_b(const rk_npu_matmul_f16_config* cfg,
                             const uint16_t* B_rowmajor, rk_npu_mem* weight);

/* Pack compact host C0[M][N] fp16 bit-pattern tensor into the DPU-RDMA
 * half-surface operand layout required by RK_NPU_FUSE_MUL/ADD. */
int rk_npu_matmul_f16_pack_operand(const rk_npu_matmul_f16_config* cfg,
                                   const uint16_t* C0_rowmajor, rk_npu_mem* operand);

/* Unpack D[M][N] from the NPU output buffer into compact row-major fp16 bit
 * patterns, or widen to fp32 for diagnostics/reference comparison. */
int rk_npu_matmul_f16_unpack_d(const rk_npu_matmul_f16_config* cfg,
                               const rk_npu_mem* output, uint16_t* D_rowmajor);
int rk_npu_matmul_f16_unpack_d_f32(const rk_npu_matmul_f16_config* cfg,
                                   const rk_npu_mem* output, float* D_rowmajor);
/* Plain normal-layout convenience path: de-pad D and add one compact FP16
 * bias[N] in the same CPU pass. */
int rk_npu_matmul_f16_unpack_d_add_bias(
    const rk_npu_matmul_f16_config* cfg, const rk_npu_mem* output,
    const uint16_t* bias_n, uint16_t* D_rowmajor);

/* Prepare/run/free a fixed-shape fp16 plan.  For RK_NPU_FUSE_NONE, pass NULL as
 * operand to run(); for MUL/ADD, operand must contain pack_operand() output. */
rk_npu_matmul_f16_plan* rk_npu_matmul_f16_prepare(rk_npu_iommu_domain* domain,
                                                  const rk_npu_matmul_f16_config* cfg);
int rk_npu_matmul_f16_run(rk_npu_ctx* ctx, rk_npu_matmul_f16_plan* plan,
                          rk_npu_mem* input, rk_npu_mem* weight,
                          rk_npu_mem* operand, rk_npu_mem* output);
void rk_npu_matmul_f16_plan_free(rk_npu_matmul_f16_plan* plan);

/* Same-shape distinct-weight FP16 BMM.  cfg describes one item and must use
 * RK_NPU_FUSE_NONE.  Host tensors are compact batch-major A[B,M,K], B[B,K,N]
 * and D[B,M,N].  Packed device buffers concatenate the per-item layouts.  One
 * prepared run submits the complete batch as one (optionally multicore) job. */
int rk_npu_matmul_f16_batch_query(int B,
                                  const rk_npu_matmul_f16_config* cfg,
                                  rk_npu_matmul_sizes* out);
int rk_npu_matmul_f16_batch_pack_a(int B,
                                   const rk_npu_matmul_f16_config* cfg,
                                   const uint16_t* A_bmk,
                                   rk_npu_mem* input);
int rk_npu_matmul_f16_batch_pack_b(int B,
                                   const rk_npu_matmul_f16_config* cfg,
                                   const uint16_t* B_bkn,
                                   rk_npu_mem* weight);
int rk_npu_matmul_f16_batch_unpack_d(int B,
                                     const rk_npu_matmul_f16_config* cfg,
                                     const rk_npu_mem* output,
                                     uint16_t* D_bmn);
rk_npu_matmul_f16_batch_plan* rk_npu_matmul_f16_batch_prepare(
    rk_npu_iommu_domain* domain, int B,
    const rk_npu_matmul_f16_config* cfg);
int rk_npu_matmul_f16_batch_run(rk_npu_ctx* ctx,
                                rk_npu_matmul_f16_batch_plan* plan,
                                rk_npu_mem* input,
                                rk_npu_mem* weight,
                                rk_npu_mem* output);
void rk_npu_matmul_f16_batch_plan_free(
    rk_npu_matmul_f16_batch_plan* plan);

/* Large-K, one-matrix path.  K is divided into two or three 32-aligned slices;
 * each slice is submitted as an independent partial GEMM and the host reduces
 * the FP16 partials in FP32.  This is intended for one slice per selected NPU
 * core.  The current implementation requires FUSE_NONE, normal A/D layouts,
 * and n_tile=0.  Device buffers contain all partials and are sized by query(). */
int rk_npu_matmul_f16_splitk_query(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    rk_npu_matmul_sizes* out);
int rk_npu_matmul_f16_splitk_pack_a(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* A_mk, rk_npu_mem* input);
int rk_npu_matmul_f16_splitk_pack_b(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* B_kn, rk_npu_mem* weight);
int rk_npu_matmul_f16_splitk_unpack_d(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* partial_output, uint16_t* D_mn);
int rk_npu_matmul_f16_splitk_unpack_d_f32(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* partial_output, float* D_mn);
/* Reduce partials in FP32, add FP16 bias[N], and round only the final result. */
int rk_npu_matmul_f16_splitk_unpack_d_add_bias(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* partial_output, const uint16_t* bias_n,
    uint16_t* D_mn);
rk_npu_matmul_f16_splitk_plan* rk_npu_matmul_f16_splitk_prepare(
    rk_npu_iommu_domain* domain, int split_count,
    const rk_npu_matmul_f16_config* cfg);
int rk_npu_matmul_f16_splitk_run(
    rk_npu_ctx* ctx, rk_npu_matmul_f16_splitk_plan* plan,
    rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* partial_output);
void rk_npu_matmul_f16_splitk_plan_free(
    rk_npu_matmul_f16_splitk_plan* plan);

#ifdef __cplusplus
}
#endif
#endif /* RK_NPU_MATMUL_F16_H */

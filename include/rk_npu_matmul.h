/*
 * rk_npu_matmul - int8 x int8 matmul on the RK3588 NPU
 * ====================================================
 * Direct register programming through the rknpu DRM driver (/dev/dri/cardN).
 * No librknnrt.so, no RKNN runtime, no compiler.
 *
 * This header temporarily preserves the distinct-weight BMM API.  New
 * single-GEMM code should use rk_npu_quant_matmul.h instead.
 */
#ifndef RK_NPU_MATMUL_H
#define RK_NPU_MATMUL_H

#include "rk_npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A/input buffer layout consumed by the NPU feature DMA. */
typedef enum {
    /* Padded row-major feature buffer:
     *   input[r * align_up(K,32) + k] = A_rowmajor[r * K + k]
     * Padding values are zeroed by the internal packer. */
    RK_NPU_I8_A_LAYOUT_NORMAL = 0,

    /* RK3588 native-A performance layout:
     *   input[(k/16) * M * 16 + r * 16 + (k%16)] = A_rowmajor[r * K + k]
     * K is padded to the internal align_in.  This layout must be paired with a
     * plan prepared from the same config because it changes CNA DMA geometry. */
    RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16 = 1,
    /* Row panels: [M/W, align32(K)/16, W, 16], W=8 or 16.
     * M and each task's M tile must be divisible by W. */
    RK_NPU_I8_A_LAYOUT_PANEL8 = 2,
    RK_NPU_I8_A_LAYOUT_PANEL16 = 3,
} rk_npu_matmul_i8_a_layout;

/* C/output buffer layout written by the DPU WDMA. */
typedef enum {
    /* Padded row-major NPU output:
     *   output[r * align_out + n] is one 4-byte fp32/int32 value.
     * The internal unpacker strips align_out padding. */
    RK_NPU_I8_C_LAYOUT_NORMAL_PADDED = 0,

    /* RK3588 native-C output layout:
     *   output[((n/4) * M + r) * 4 + (n%4)] is one 4-byte fp32/int32 value.
     * This is AC_layout=NATIVE-compatible and is usually faster for large N. */
    RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4 = 1,
    /* Row panels: [M/W, align32(N)/4, W, 4]. */
    RK_NPU_I8_C_LAYOUT_PANEL8 = 2,
    RK_NPU_I8_C_LAYOUT_PANEL16 = 3,
} rk_npu_matmul_i8_c_layout;

/* Four-byte output element type produced by the int8 datapath. */
typedef enum {
    /* CORE accumulates int32, then DPU converts to fp32 before writeback.
     * The internal unpacker writes float elements. */
    RK_NPU_I8_OUT_FP32 = 0,

    /* Raw int32 accumulator writeback.  Buffer sizes and layouts are identical
     * to fp32, but the internal unpacker writes int32_t elements. */
    RK_NPU_I8_OUT_INT32 = 1,
} rk_npu_matmul_i8_out_dtype;

/*
 * Complete fixed-shape int8 plan configuration.
 *
 * M/N/K are part of the config because they define buffer sizes, packing layout,
 * register programming, and task tiling.  A prepared plan stores a copy of this
 * struct; mutating the caller's config after prepare() has no effect.
 */
typedef struct rk_npu_matmul_i8_config {
    int M;
    int N;
    int K;
    rk_npu_matmul_i8_a_layout  a_layout;
    rk_npu_matmul_i8_c_layout  c_layout;
    rk_npu_matmul_i8_out_dtype out_dtype;
} rk_npu_matmul_i8_config;

typedef struct rk_npu_matmul_i8_batch_plan rk_npu_matmul_i8_batch_plan;

/* Initialize cfg for the default int8 path:
 * normal A layout, normal padded C layout, fp32 output, internal tiling. */
void rk_npu_matmul_i8_config_init(rk_npu_matmul_i8_config* cfg, int M, int N, int K);

/* Batched same-shape path.  cfg describes one batch item; B is the number of
 * independent GEMMs A[b]@B[b] concatenated in each packed device buffer.  This
 * legacy interface remains public until the MHA-specific BMM planner is
 * redesigned; it does not use the new single-GEMM autotuner. */
int rk_npu_matmul_i8_batch_query(int B, const rk_npu_matmul_i8_config* cfg,
                                 rk_npu_matmul_sizes* out);
int rk_npu_matmul_i8_batch_pack_a(int B, const rk_npu_matmul_i8_config* cfg,
                                  const int8_t* A_bmk, rk_npu_mem* input);
int rk_npu_matmul_i8_batch_pack_b(int B, const rk_npu_matmul_i8_config* cfg,
                                  const int8_t* B_bkn, rk_npu_mem* weight);
int rk_npu_matmul_i8_batch_unpack_c(int B, const rk_npu_matmul_i8_config* cfg,
                                    const rk_npu_mem* output, void* C_bmn_raw4);
rk_npu_matmul_i8_batch_plan* rk_npu_matmul_i8_batch_prepare(
    rk_npu_iommu_domain* domain, int B, const rk_npu_matmul_i8_config* cfg);
int rk_npu_matmul_i8_batch_run(rk_npu_ctx* ctx, rk_npu_matmul_i8_batch_plan* plan,
                               rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output);
void rk_npu_matmul_i8_batch_plan_free(rk_npu_matmul_i8_batch_plan* plan);

#ifdef __cplusplus
}
#endif
#endif /* RK_NPU_MATMUL_H */

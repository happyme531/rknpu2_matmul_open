/* Fused FP16 GEMM + FP32 channel bias + tanh GELU LUT.
 * Reuses resident FP16 weights; no host K partition or CPU activation pass. */
#ifndef RK_NPU_MM_GELU_H
#define RK_NPU_MM_GELU_H
#include "rk_npu_mm_tune.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct rk_npu_mm_gelu_config {
    int M, N, K;        /* logical compact A[M,K], resident B[K,N];
                          M/K in 1..65504, N in 1..32768; normal MM limits also apply */
    int n_tile;         /* 0: balance aligned N over allowed cores; otherwise multiple of 32 */
    uint32_t core_mask; /* 1/2/4/3/7; init defaults to 7 */
    uint32_t timeout_ms;
} rk_npu_mm_gelu_config;
typedef struct rk_npu_mm_gelu_workspace rk_npu_mm_gelu_workspace;
/* Borrowed, mapped FP32 carrier, valid until the next run or workspace free.
 * Logical C[r,c] = data[(r/8)*8*storage_columns + (c/4*8+r%8)*4+c%4] * scale.
 * Padding is not logical output. Collect directly into the caller's FP16/F32
 * tensor. Processing/output are FP32 to preserve zero codes and large tails;
 * the LUT itself is signed Q13, piecewise linear on [-4,4], with identity
 * positive tail and zero negative tail. This is an approximate GELU. */
typedef struct rk_npu_mm_gelu_view {
    const float *data;
    uint64_t elements;
    int rows, columns, storage_rows, storage_columns;
    float scale;
} rk_npu_mm_gelu_view;
void rk_npu_mm_gelu_config_init(rk_npu_mm_gelu_config *, int M, int N, int K);
int rk_npu_mm_gelu_prepare(rk_npu_iommu_domain *, const rk_npu_mm_gelu_config *,
                           rk_npu_mm_gelu_workspace **out);
void rk_npu_mm_gelu_free(rk_npu_mm_gelu_workspace *);
/* No allocation, search, weight upload or CPU GELU. A is compact FP16,
 * bias is FP32[N]. Workspace is non-reentrant, including view consumption:
 * caller serializes run and collection. ctx must outlive all device objects.
 * On error the view is empty; prior output is no longer guaranteed. */
int rk_npu_mm_gelu_run(rk_npu_mm_gelu_workspace *, const uint16_t *A, uint64_t a_bytes,
                       const rk_npu_mm_packed_b *, const float *bias, rk_npu_mm_gelu_view *);
int rk_npu_mm_gelu_run_weights(rk_npu_mm_gelu_workspace *, const uint16_t *A, uint64_t a_bytes,
                               const rk_npu_mm_tuned_weights *, const float *bias, rk_npu_mm_gelu_view *);
#ifdef __cplusplus
}
#endif
#endif

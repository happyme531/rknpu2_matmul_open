/* Additive raw MM/BMM API. Existing rk_npu_matmul_* interfaces are unchanged.
 * C[b] = op(A[b]) @ op(B[b]); no quantization, scales, bias or activation.
 * Implements FP16 x FP16 -> FP16, BF16 x BF16 -> BF16, TF32 x TF32 -> FP32.
 * FP16/BF16 use uint16_t bit patterns; TF32 inputs and FP32 outputs use float.
 * TF32 packing preserves raw FP32 inputs. For finite normal values the tested
 * RK3588 MAC input truncates 13 low significand bits (10 fraction bits), not RNE.
 * Other combinations return MM_ERR_UNSUPPORTED; no precision fallback.
 * The current TF32 backend requires aligned part_k<=4096; larger slices return
 * MM_ERR_UNSUPPORTED at plan creation. Use split-K or caller-side K blocking.
 */
#ifndef RK_NPU_MM_H
#define RK_NPU_MM_H
#include "rk_npu_common.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Specific to this API; other failures use the existing RK_NPU_ERR_* codes. */
#define RK_NPU_MM_ERR_UNSUPPORTED (-100)

typedef enum {
    RK_NPU_MM_F16 = 0, RK_NPU_MM_BF16 = 1, RK_NPU_MM_F32 = 2,
    RK_NPU_MM_I8 = 3, RK_NPU_MM_I4 = 4, RK_NPU_MM_I32 = 5,
    RK_NPU_MM_TF32 = 6 /* FP32 input container, TF32 arithmetic */
} rk_npu_mm_dtype;
typedef enum {
    RK_NPU_MM_LAYOUT_NORMAL = 0, RK_NPU_MM_LAYOUT_NATIVE = 1
} rk_npu_mm_layout;

typedef struct rk_npu_mm_desc {
    uint32_t struct_size;
    int M, N, K, batch_count;
    rk_npu_mm_dtype a_type, b_type, c_type;
    int trans_a, trans_b; /* 0 or 1; C is never transposed */
    /* All strides count storage ELEMENTS, not bytes. lda/ldb describe rows
     * BEFORE transpose. A is [M,K] or [K,M]; B is [K,N] or [N,K].
     * lda/ldb/ldc=0 selects the physical compact row width.
     * Input batch stride=0 broadcasts; nonzero batches must not overlap.
     * C stride=0 selects M*ldc. Negative strides are not supported. */
    uint64_t lda, ldb, ldc;
    uint64_t batch_stride_a, batch_stride_b, batch_stride_c;
} rk_npu_mm_desc;

typedef struct rk_npu_mm_options {
    uint32_t struct_size;
    uint32_t allowed_npu_core_mask; /* 1/2/4/3/7; default 7; 0 also means 7 */
    uint32_t timeout_ms; /* 0 selects 6000 ms */
    rk_npu_mm_layout a_layout, c_layout;
    int n_tile; /* 0: full aligned N; otherwise a positive multiple of 32 */
    /* -1: read RK_NPU_MM_SPLIT_K once at plan creation (unset/1: auto,
     * 0: disabled; other strings are errors). 0/1 explicitly disable/allow
     * auto, 2/3 force that split count and ignore the environment.
     * Auto uses all permitted cores for K>=2048 and batch_count<core_count.
     * Otherwise it uses one slice. This is a heuristic, not an autotuner.
     * NPU partials use c_type; reduction is CPU FP32, then RNE narrowing for
     * FP16/BF16 (no narrowing for FP32). Overflow/rounding may differ from unsplit
     * MM. Internal NPU accumulation order/precision is not a strict FP32 ABI;
     * BF16 NaN payload and signed-zero preservation are not guaranteed. */
    int split_k;
} rk_npu_mm_options;

typedef struct rk_npu_mm_info {
    rk_npu_mm_desc desc; /* normalized strides */
    rk_npu_mm_options options; /* resolved policy; no further getenv on run */
    int split_count, part_k;
    uint32_t actual_npu_core_mask;
    uint64_t host_a_bytes, host_b_bytes, host_c_bytes; /* minimum spans */
    /* Packed buffers contain [batch][split] items. output_bytes includes
     * all c_type partials. unpack_c performs any required reduction. */
    rk_npu_matmul_sizes device;
} rk_npu_mm_info;

typedef struct rk_npu_mm_plan rk_npu_mm_plan;
typedef struct rk_npu_mm_workspace rk_npu_mm_workspace;
typedef struct rk_npu_mm_packed_b rk_npu_mm_packed_b;
typedef enum {
    RK_NPU_MM_HOST_DYNAMIC = 0, /* owns A/B/C buffers */
    RK_NPU_MM_HOST_PACKED_B = 1, /* owns A/C; B comes from packed_b */
    RK_NPU_MM_DEVICE_ONLY = 2 /* owns only prepared commands/tasks */
} rk_npu_mm_workspace_mode;

/* Initializes compact, distinct-batch FP16 tensors; call again after changing
 * shape, or update strides yourself. Batch=1 is ordinary MM. */
void rk_npu_mm_desc_init(rk_npu_mm_desc*, int M, int N, int K, int batch_count);
void rk_npu_mm_options_init(rk_npu_mm_options*);
/* Plan creation/query/packing are CPU-only. options=NULL uses defaults.
 * Every create function sets *out=NULL on failure and returns an error code. */
int rk_npu_mm_plan_create(const rk_npu_mm_desc*, const rk_npu_mm_options*,
                          rk_npu_mm_plan** out);
int rk_npu_mm_plan_get_info(const rk_npu_mm_plan*, rk_npu_mm_info*);
void rk_npu_mm_plan_free(rk_npu_mm_plan*);

/* CPU layout conversion only; no allocation, device registration or sync.
 * rk_npu_mem may describe ordinary mapped host storage here. Host and packed
 * spans must not overlap. Output padding is preserved in the host tensor. */
int rk_npu_mm_pack_a(const rk_npu_mm_plan*, const void*, uint64_t host_bytes,
                     rk_npu_mem* packed);
int rk_npu_mm_pack_b(const rk_npu_mm_plan*, const void*, uint64_t host_bytes,
                     rk_npu_mem* packed);
int rk_npu_mm_unpack_c(const rk_npu_mm_plan*, const rk_npu_mem* packed,
                       void*, uint64_t host_bytes);

/* Objects copy the plan and retain the domain; the plan/domain handle may be
 * freed after creation, but ctx must outlive all objects. Calls are blocking.
 * Workspaces are non-reentrant (BUSY); free must not race with a call.
 * Packed B is immutable and reusable across M, strides, layouts, tiles and
 * core masks when dtype/K/N/batch_count/split_count/part_k and
 * ctx/domain match. FP16 and BF16 packed B objects are not interchangeable.
 * Broadcast B is currently duplicated in device storage for each batch. */
int rk_npu_mm_workspace_create(rk_npu_iommu_domain*, const rk_npu_mm_plan*,
                               rk_npu_mm_workspace_mode, rk_npu_mm_workspace** out);
void rk_npu_mm_workspace_free(rk_npu_mm_workspace*);
int rk_npu_mm_packed_b_create(rk_npu_iommu_domain*, const rk_npu_mm_plan*,
                              const void* B, uint64_t bytes, rk_npu_mm_packed_b** out);
void rk_npu_mm_packed_b_free(rk_npu_mm_packed_b*);

/* Complete host calls: packing, cache sync, submit, unpack/reduce. No per-run
 * buffer allocation. A/B must not overlap C. C is undefined on error.
 * Each entry requires the corresponding HOST_* workspace mode. */
int rk_npu_mm_run(rk_npu_mm_workspace*, const void* A, uint64_t a_bytes,
                  const void* B, uint64_t b_bytes, void* C, uint64_t c_bytes);
int rk_npu_mm_run_packed_b(rk_npu_mm_workspace*, const void* A, uint64_t a_bytes,
                           const rk_npu_mm_packed_b*, void* C, uint64_t c_bytes);
/* Any workspace mode. Device buffers must match this plan's packed layouts,
 * ctx/domain and queried sizes; output must not overlap either input.
 * Caller performs all data-buffer cache/ownership transitions. No pack, sync
 * or reduction occurs here; C holds c_type partials when split_count>1. */
int rk_npu_mm_run_device(rk_npu_mm_workspace*, rk_npu_mem* A,
                         rk_npu_mem* B, rk_npu_mem* C);

#ifdef __cplusplus
}
#endif
#endif

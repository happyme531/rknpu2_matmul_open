/* Experimental causal GQA Attention: compact FP16 Q/K/V -> compact FP32 O.
 * Flags explicitly select fixed-shift exp or stable row-max Attention.
 * No RKNN runtime/compiler/model files are required. */
#ifndef RK_NPU_ATTENTION_F16_H
#define RK_NPU_ATTENTION_F16_H
#include "rk_npu_common.h"
#ifdef __cplusplus
extern "C" {
#endif

#define RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL 1u
/* NPU QK/PV with stable row-max centering and native LUT exponential.
 * CPU exponential is used when the FP16 PV bound or task limit requires it. */
#define RK_NPU_ATTENTION_F16_STABLE_SOFTMAX 2u
typedef enum {
    RK_NPU_ATTENTION_F16_CAUSAL = 0,
    RK_NPU_ATTENTION_F16_NO_MASK = 1,
    RK_NPU_ATTENTION_F16_BOOLEAN_MASK = 2,
} rk_npu_attention_f16_mask_mode;
/* One batch, byte boolean data (nonzero=True=visible). Supports broadcasting
 * heads/queries/keys when the corresponding dimension is 1. Strides are bytes.
 * 2D [Q,K] and HF/SDPA [1,H,Q,K] map to these three dimensions. query_offset
 * selects a block of rows from a larger mask. version=0 refreshes every call;
 * nonzero versions may reuse an unchanged mask and must change after writes. */
typedef struct {
    const uint8_t* data;
    uint64_t bytes, head_stride, query_stride, key_stride, version;
    int heads, query_rows, key_length, query_offset;
} rk_npu_attention_f16_boolean_mask;
void rk_npu_attention_f16_boolean_mask_init(rk_npu_attention_f16_boolean_mask* mask,
    const uint8_t* data, uint64_t bytes, int heads, int query_rows, int key_length);
typedef struct {
    int query_heads, kv_heads, head_dim; /* Hq is a positive multiple of Hkv; D is a positive multiple of 32 */
    int max_query_rows;       /* logical rows per call, 1..128; default 32 */
    int kv_tile;              /* 256/512/1024/2048/4096 */
    int initial_capacity;     /* multiple of 32 */
    int max_capacity;         /* caller-provided allocation bound, multiple of 32 */
    uint32_t core_mask;       /* 1/2/4, dual 3, triple 7; 0 means 1.
                              * At least one KV head per selected core. */
    uint32_t timeout_ms;
    uint32_t flags;           /* select FIXED_SHIFT_EXPERIMENTAL or STABLE_SOFTMAX */
    float exp_shift;
    rk_npu_attention_f16_mask_mode mask_mode;
    float softmax_scale;      /* positive; zero selects 1/sqrt(head_dim) */
} rk_npu_attention_f16_config;
typedef struct {
    uint64_t key_bytes, value_bytes, tail_bytes;
    uint64_t query_bytes, score_bytes, mask_bytes, partial_bytes, output_bytes;
    int compute_length, physical_rows, tasks;
    uint32_t active_core_mask; /* requested cores bounded by KV heads */
} rk_npu_attention_f16_sizes;
typedef struct {
    int length, capacity, max_capacity;
    uint64_t key_bytes, value_bytes, tail_bytes, generation;
} rk_npu_attention_f16_cache_info;
typedef struct {
    double reserve_us, kv_update_us, prepare_us, tail_stage_us, mask_update_us;
    /* Stable mode submit_us includes NPU submits and CPU score processing/sync;
     * hardware_us sums only the driver-reported NPU intervals. */
    double query_pack_us, submit_us, hardware_us, output_sync_us, finish_us, total_us;
    int compute_length, tasks;
} rk_npu_attention_f16_timings;
typedef struct rk_npu_attention_f16_cache rk_npu_attention_f16_cache;
typedef struct rk_npu_attention_f16_workspace rk_npu_attention_f16_workspace;

/* Defaults describe Hq/Hkv=32/8,d128. flags defaults to zero: opt in explicitly. */
void rk_npu_attention_f16_config_init(rk_npu_attention_f16_config* cfg);
/* Pure CPU sizing/validation. kv_length is the causal prefix to compute, not
 * allocated cache capacity. Buffer sizes describe the largest internal segment;
 * output_bytes covers [query_heads,query_rows,head_dim], tasks estimates QK/PV
 * work for layout segments. Runtime numerical-mode splits and native LUT
 * execution can add tasks; run timings report their actual sum. */
int rk_npu_attention_f16_query(const rk_npu_attention_f16_config* cfg,
    int capacity, int query_rows, int kv_length, rk_npu_attention_f16_sizes* sizes);

/* Cache and workspace retain domain references; caller ctx must outlive both.
 * Multiple layer caches can share one workspace. Objects are non-reentrant:
 * concurrent use returns ERR_BUSY. Different workspaces still require caller
 * scheduling when sharing a physical core. */
int rk_npu_attention_f16_cache_create(rk_npu_iommu_domain* domain,
    const rk_npu_attention_f16_config* cfg, rk_npu_attention_f16_cache** out);
void rk_npu_attention_f16_cache_free(rk_npu_attention_f16_cache* cache);
int rk_npu_attention_f16_cache_get_info(rk_npu_attention_f16_cache* cache,
    rk_npu_attention_f16_cache_info* info);
/* Native KV survives reserve; automatic geometric growth is bounded by max_capacity. */
int rk_npu_attention_f16_cache_reserve(rk_npu_ctx* ctx,
    rk_npu_attention_f16_cache* cache, int capacity);
/* K/V are compact [kv_heads,length,head_dim]. length=0 accepts NULL pointers. */
int rk_npu_attention_f16_cache_load(rk_npu_ctx* ctx,
    rk_npu_attention_f16_cache* cache, const uint16_t* k, const uint16_t* v, int length);
/* K/V are compact [kv_heads,count,head_dim]; NULL is accepted only for count=0. */
int rk_npu_attention_f16_cache_append(rk_npu_ctx* ctx,
    rk_npu_attention_f16_cache* cache, const uint16_t* k, const uint16_t* v, int count);
/* Retain a prefix, 0<=length<=current length, without copying retained K/V or
 * changing capacity. Requires STABLE_SOFTMAX. Native LUT masking clears discarded
 * V/denominator rows; CPU masking excludes them. Tail staging is invalidated.
 * A poisoned cache still requires cache_load before reuse. */
int rk_npu_attention_f16_cache_truncate(rk_npu_attention_f16_cache* cache, int length);
int rk_npu_attention_f16_prepare(rk_npu_iommu_domain* domain,
    const rk_npu_attention_f16_config* cfg, rk_npu_attention_f16_workspace** out);
void rk_npu_attention_f16_workspace_free(rk_npu_attention_f16_workspace* workspace);
/* Optional warm-up: builds geometry and tail staging without a submit.
 * Geometry changes and reserve costs are also included when run/step does them. */
int rk_npu_attention_f16_prepare_query(rk_npu_ctx* ctx,
    rk_npu_attention_f16_workspace* workspace, rk_npu_attention_f16_cache* cache,
    int query_rows, int query_start);
/* Q/O are [query_heads,query_rows,head_dim]. query_start is the first absolute
 * query position for causal mode. Each causal row attends through query_start+row.
 * No-mask/boolean modes use all valid KV; query rows can exceed KV length.
 * The library splits query rows for native layout limits and restores compact
 * output order. Stable causal GQA also splits at softmax mode boundaries to
 * retain each row's single-token AR exp/normalization path. Future KV never
 * changes an earlier row's LUT/CPU or FP16-normalization mode.
 * Computing an earlier prefill chunk never reads reserved padding. */
int rk_npu_attention_f16_run(rk_npu_ctx* ctx,
    rk_npu_attention_f16_workspace* workspace, rk_npu_attention_f16_cache* cache,
    const uint16_t* q, int query_rows, int query_start, float* output,
    rk_npu_attention_f16_timings* timings, const rk_npu_attention_f16_boolean_mask* mask);
/* Same execution and FP32 softmax/output reductions; stores the final result
 * directly as compact FP16 [query_heads,query_rows,head_dim], 2 bytes/element.
 * No full-size FP32 output tensor is constructed. */
int rk_npu_attention_f16_run_half(rk_npu_ctx* ctx,
    rk_npu_attention_f16_workspace* workspace, rk_npu_attention_f16_cache* cache,
    const uint16_t* q, int query_rows, int query_start, uint16_t* output,
    rk_npu_attention_f16_timings* timings, const rk_npu_attention_f16_boolean_mask* mask);
/* Append one K/V[Hkv,head_dim] then run Q[Hq,head_dim]. Includes growth, graph updates,
 * cache sync and output processing. Submit/sync failures poison the affected
 * cache; reload it before reuse. Failed submits require a new workspace. */
int rk_npu_attention_f16_decode_step(rk_npu_ctx* ctx,
    rk_npu_attention_f16_workspace* workspace, rk_npu_attention_f16_cache* cache,
    const uint16_t* q, const uint16_t* k, const uint16_t* v, float* output,
    rk_npu_attention_f16_timings* timings, const rk_npu_attention_f16_boolean_mask* mask);
#ifdef __cplusplus
}
#endif
#endif

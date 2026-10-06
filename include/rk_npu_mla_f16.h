/* Experimental Ling-3.0-tiny MLA core, independent of the RKNN SDK.
 * FP16 Q and normalized latent KV + FP32 head gate logits -> FP32 output.
 * Q/rotary K are already rotated. Q/latent down-projections, RMSNorm, RoPE,
 * gate projection and final output projection belong to the caller. */
#ifndef RK_NPU_MLA_F16_H
#define RK_NPU_MLA_F16_H
#include "rk_npu_attention_f16.h"
#ifdef __cplusplus
extern "C" {
#endif

#define RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL 1u
#define RK_NPU_MLA_F16_STABLE_SOFTMAX 2u
typedef enum {
    RK_NPU_MLA_F16_EXPANDED = 0, /* per-head K192/V128, favor prefill */
    RK_NPU_MLA_F16_ABSORBED = 1, /* shared K576/C512, compressed cache */
} rk_npu_mla_f16_mode;
typedef struct {
    rk_npu_mla_f16_mode mode;
    int max_query_rows; /* expanded 1..128, absorbed 1..32; default32 */
    int kv_tile; /* 256/512/1024/2048/4096 */
    int initial_capacity, max_capacity; /* multiples of 32, max <=32768 */
    uint32_t core_mask, timeout_ms, flags; /* masks 0/1/2/4/3/7; 0 means 1 */
    float exp_shift; /* defaults to 4; fixed exp approximation, opt-in */
    rk_npu_attention_f16_mask_mode mask_mode;
} rk_npu_mla_f16_config;
typedef struct {
    uint64_t key_bytes,value_bytes,tail_bytes,score_bytes,mask_bytes,partial_bytes;
    uint64_t output_bytes;
    int compute_length,physical_rows,tasks; /* Attention graph tasks only */
} rk_npu_mla_f16_sizes;
typedef struct {
    int length,capacity,max_capacity;
    uint64_t key_bytes,value_bytes,tail_bytes;
} rk_npu_mla_f16_cache_info;
typedef struct {
    double kv_update_us,query_projection_us,attention_us,value_projection_us;
    double gate_output_us,total_us;
    rk_npu_attention_f16_timings attention;
} rk_npu_mla_f16_timings;
typedef struct rk_npu_mla_f16_weights rk_npu_mla_f16_weights;
typedef struct rk_npu_mla_f16_cache rk_npu_mla_f16_cache;
typedef struct rk_npu_mla_f16_workspace rk_npu_mla_f16_workspace;

void rk_npu_mla_f16_config_init(rk_npu_mla_f16_config* cfg);
int rk_npu_mla_f16_query(const rk_npu_mla_f16_config* cfg,int capacity,
    int query_rows,int kv_length,rk_npu_mla_f16_sizes* out);
/* kv_b is the HF kv_b_proj weight, compact [16,256,512]: K128 then V128
 * for each head. Pack once, immutable. Caches/workspaces retain weights and
 * their domain. Caller ctx must outlive all handles. For EXPANDED only,
 * cache_create/prepare also accept NULL weights when the caller supplies
 * already projected K/V through the expanded cache entry points below. */
int rk_npu_mla_f16_weights_create(rk_npu_iommu_domain* domain,
    const uint16_t* kv_b,rk_npu_mla_f16_weights** out);
void rk_npu_mla_f16_weights_free(rk_npu_mla_f16_weights* weights);
int rk_npu_mla_f16_cache_create(rk_npu_iommu_domain* domain,
    const rk_npu_mla_f16_config* cfg,rk_npu_mla_f16_weights* weights,
    rk_npu_mla_f16_cache** out);
void rk_npu_mla_f16_cache_free(rk_npu_mla_f16_cache* cache);
int rk_npu_mla_f16_cache_get_info(rk_npu_mla_f16_cache* cache,
    rk_npu_mla_f16_cache_info* out);
int rk_npu_mla_f16_cache_reserve(rk_npu_ctx* ctx,
    rk_npu_mla_f16_cache* cache,int capacity);
/* latent [count,512] is AFTER kv_a_layernorm; rope_key [count,64] is AFTER
 * RoPE. Load replaces cache; append preserves prefix and grows geometrically.
 * Absorbed keeps one shared copy; expanded performs NPU kv_b projections.
 * count=0 accepts NULL inputs. Sync/submit failures require cache reload. */
int rk_npu_mla_f16_cache_load(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* cache,
    const uint16_t* latent,const uint16_t* rope_key,int count);
int rk_npu_mla_f16_cache_append(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* cache,
    const uint16_t* latent,const uint16_t* rope_key,int count);
/* EXPANDED only: preserve the caller's projection/quantization policy.
 * Compact FP16 K [16,count,192], V [16,count,128], head-major, including
 * rotated K. No projection runs here. count=0 accepts NULL inputs. The
 * same reserve, synchronization and failure rules apply as load/append. */
int rk_npu_mla_f16_cache_load_expanded(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* cache,
    const uint16_t* keys,const uint16_t* values,int count);
int rk_npu_mla_f16_cache_append_expanded(rk_npu_ctx* ctx,rk_npu_mla_f16_cache* cache,
    const uint16_t* keys,const uint16_t* values,int count);
int rk_npu_mla_f16_prepare(rk_npu_iommu_domain* domain,
    const rk_npu_mla_f16_config* cfg,rk_npu_mla_f16_weights* weights,
    rk_npu_mla_f16_workspace** out);
void rk_npu_mla_f16_workspace_free(rk_npu_mla_f16_workspace* workspace);
/* Q [query_rows,16,192] (head content128 + rotated64), output
 * [query_rows,16,128]. Optional gate_logits [query_rows,16]: FP32 sigmoid
 * multiplied after attention and before the caller's output projection.
 * NULL gate_logits means ungated MLA. Boolean mask uses the Attention ABI.
 * The scale stays 1/sqrt(192), including absorbed QK's 576 channels.
 * Cache/workspace must have the same mode/tile/weights/domain. Objects are
 * non-reentrant; caller schedules workspaces sharing physical NPU cores. */
int rk_npu_mla_f16_run(rk_npu_ctx* ctx,rk_npu_mla_f16_workspace* workspace,
    rk_npu_mla_f16_cache* cache,const uint16_t* q,const float* gate_logits,
    int query_rows,int query_start,float* output,rk_npu_mla_f16_timings* timings,
    const rk_npu_attention_f16_boolean_mask* mask);
/* Append one latent/rotary key and attend it. Includes all packing, sync,
 * projection, allocation/growth and geometry updates inside this call. */
int rk_npu_mla_f16_decode_step(rk_npu_ctx* ctx,rk_npu_mla_f16_workspace* workspace,
    rk_npu_mla_f16_cache* cache,const uint16_t* q,const float* gate_logits,
    const uint16_t* latent,const uint16_t* rope_key,float* output,
    rk_npu_mla_f16_timings* timings,const rk_npu_attention_f16_boolean_mask* mask);
#ifdef __cplusplus
}
#endif
#endif

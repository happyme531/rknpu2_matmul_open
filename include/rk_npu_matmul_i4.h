/* Exact W4A4 GEMM. No floating-point quantization or model transforms.
 *
 * Host codes use int8_t elements in [-8,7]; device storage is packed INT4.
 * INT16 NPU partials are widened/reduced to compact row-major INT32 output.
 * Positive M/N/K tails are padded internally. K <= INT32_MAX/64 guarantees
 * that every possible signed INT4 dot product fits INT32.
 *
 * Immutable weights depend only on K/N/k_tile and may be shared across M,
 * M/N tilings and core masks. Workspaces are fixed-shape and non-reentrant.
 * The context must outlive its weights/workspaces. Calls are blocking.
 */
#ifndef RK_NPU_MATMUL_I4_H
#define RK_NPU_MATMUL_I4_H

#include "rk_npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rk_npu_i4_weight_config {
    int K, N;
    int k_tile; /* multiple of 32, 32..2048; >480 requires a weight-bound proof */
} rk_npu_i4_weight_config;

typedef struct rk_npu_i4_config {
    int M, N, K;
    int m_tile; /* 1..128 */
    int k_tile; /* 32..2048, multiple of 32; must match the validated weights */
    int n_tile; /* multiple of 64, 64..4096, per selected core */
    uint32_t npu_core_mask; /* 1/2/4/3/7; need >= popcount(mask) N64 blocks */
    uint32_t timeout_ms;
    int pipeline; /* 0: serial, 1: pack/reduce while next NPU wave executes */
    int cpu_threads; /* 1..4 OpenMP reduction workers; caller/runtime controls affinity */
} rk_npu_i4_config;

typedef struct rk_npu_i4_memory_info {
    uint64_t packed_weight_bytes;
    uint64_t input_bytes;   /* both ping-pong slots, logical sizes */
    uint64_t partial_bytes; /* INT16 ping-pong slots; CPU output is caller-owned */
    uint64_t control_bytes;
    uint32_t wave_count;
    uint32_t tasks_per_wave;
} rk_npu_i4_memory_info;

typedef struct rk_npu_i4_timings {
    double pack_us, sync_us, submit_us, reduce_us, total_us;
    /* Stage work may overlap: their sum is not total_us. submit_us includes
     * driver/host submission overhead, not just hardware execution. Creation
     * and offline B packing are excluded; run includes A packing and C reduction. */
} rk_npu_i4_timings;

typedef struct rk_npu_i4i4i32_weights rk_npu_i4i4i32_weights;
typedef struct rk_npu_i4_workspace rk_npu_i4_workspace;

void rk_npu_i4_config_init(rk_npu_i4_config* cfg, int M, int N, int K);
int rk_npu_i4_memory_query(const rk_npu_i4_config* cfg, rk_npu_i4_memory_info* out);
int rk_npu_i4_weights_memory_query(const rk_npu_i4_weight_config* cfg, uint64_t* bytes);

/* B[K,N] is validated and packed once. No persistent unpacked B copy is kept.
 * B may be released after creation. NULL indicates invalid config/codes or
 * allocation failure or unsafe INT16 partial bounds. For k_tile>480 creation
 * checks every column of every tile using 7P+8Q<=32767, 8P+7Q<=32768.
 * No scale/zero-point transformation is performed. */
rk_npu_i4i4i32_weights* rk_npu_i4i4i32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_i4_weight_config* cfg, const int8_t* B);
void rk_npu_i4i4i32_weights_free(rk_npu_i4i4i32_weights* weights);
rk_npu_i4_workspace* rk_npu_i4_workspace_create(
    rk_npu_iommu_domain* domain, const rk_npu_i4_config* cfg);
void rk_npu_i4_workspace_free(rk_npu_i4_workspace* workspace);

/* A[M,K] and C[M,N] must not alias; C contents are undefined on error. Codes
 * outside [-8,7] return PARAM, not masked/truncated. timings may be NULL.
 * Weights must match K/N/k_tile and the context/IOMMU domain. */
int rk_npu_i4i4i32_run(rk_npu_i4_workspace* workspace,
    const rk_npu_i4i4i32_weights* weights, const int8_t* A, int32_t* C,
    rk_npu_i4_timings* timings);

/* Fused preprocessing boundary: invoked serially on the calling thread for
 * each K wave / M panel, in increasing K then M order. dst is zero-initialized
 * [padded_k/32,rows,32] INT4, even element in HIGH nibble. Only logical K codes
 * need writing; leave the padded tail zero. Pointer is valid only during the
 * callback; the library owns synchronization and the device buffer lifetime.
 * Return RK_NPU_OK or an error, never throw or submit NPU work from a producer.
 * Full-row transforms/scales belong to caller state, not each K slice.
 * This supports future W4A8 row expansion/FlatQuant without a float ABI. */
typedef struct rk_npu_i4_input_tile {
    int m0, rows, k0, k, padded_k;
    uint8_t* dst;
    uint64_t bytes;
} rk_npu_i4_input_tile;
typedef int (*rk_npu_i4_input_producer)(void* user, const rk_npu_i4_input_tile* tile);

int rk_npu_i4i4i32_run_with_producer(rk_npu_i4_workspace* workspace,
    const rk_npu_i4i4i32_weights* weights, rk_npu_i4_input_producer producer,
    void* user, int32_t* C, rk_npu_i4_timings* timings);

/* CPU-only reference producer helper. A points to the full logical matrix;
 * lda >= tile->k0 + tile->k. Validates codes and writes the native K32 panel. */
int rk_npu_i4_pack_a_tile(const rk_npu_i4_input_tile* tile, const int8_t* A, int lda);

#ifdef __cplusplus
}
#endif
#endif

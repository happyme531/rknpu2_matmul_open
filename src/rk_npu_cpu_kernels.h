#ifndef RK_NPU_CPU_KERNELS_H
#define RK_NPU_CPU_KERNELS_H

#include <stdint.h>

namespace rknpu2_matmul_open::cpu {

/* Byte/element offsets for global native (panel_rows=0) or fixed row panels.
 * C panels include align32(N) padding between panels; global native has no
 * such inter-panel gap. Callers validate M divisibility and panel width. */
inline uint64_t i8_a_block_offset(int M, int align_in, int r, int kb,
                                  int panel_rows) {
    if (!panel_rows) return ((uint64_t)kb * M + r) * 16;
    const int mask = panel_rows - 1; // validated power-of-two widths 8/16
    return (uint64_t)(r & ~mask) * align_in +
           ((uint64_t)kb * panel_rows + (r & mask)) * 16;
}
inline uint64_t i8_c_block_offset(int M, int N, int r, int nb,
                                  int panel_rows) {
    if (!panel_rows) return ((uint64_t)nb * M + r) * 4;
    const int align_n = (N + 31) / 32 * 32;
    const int mask = panel_rows - 1;
    return (uint64_t)(r & ~mask) * align_n +
           ((uint64_t)nb * panel_rows + (r & mask)) * 4;
}

enum class QuantMode {
    DynamicPerToken = 0,
    StaticPerToken = 1,
    StaticTensor = 2,
};

enum class ActivationOp {
    None = 0,
    Relu = 1,
    Silu = 2,
    GeluTanh = 3,
};

void i8_pack_a_normal(int M, int K, int align_in, const int8_t* src, int8_t* dst);
void i8_pack_a_native_k16_m16(int M, int K, int align_in, const int8_t* src, int8_t* dst, int panel_rows = 0);

/*
 * Experimental LLM helper: quantize row-major fp16 activations and write native
 * int8 A layout in one pass over the output layout.
 *
 * Inputs:
 *   src_fp16      compact row-major fp16 bit patterns, shape [M,K].
 *   mode          DynamicPerToken computes scale_out[m] from max(abs(row));
 *                 StaticPerToken reads static_scale[m]; StaticTensor reads
 *                 static_scale[0]. A missing static_scale means scale 1.0.
 *   align_in      padded K used by the NPU int8 path, normally >= K and a
 *                 multiple of 32.
 *
 * Outputs:
 *   scale_out     optional per-row scale actually used. Dynamic mode writes
 *                 max_abs / 127, or 1.0 for an all-zero row.
 *   dst           native A buffer with layout (align_in/16, M, 16):
 *                 dst[(k/16)*M*16 + m*16 + k%16].
 *
 * Quantization is signed symmetric int8:
 *   q = clamp(round(fp16_value * (1 / scale)), -127, 127).
 * Padding lanes k >= K are zero. No cache synchronization is performed here.
 */
void i8_pack_a_f16_quant_native_k16_m16(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        QuantMode mode,
                                        const float* static_scale,
                                        float* scale_out,
                                        int8_t* dst);

/* Split-K dynamic-A helpers.  Scales are computed once over each complete
 * [K] row, then every K slice is quantized with that same per-row scale.
 * `src_row_stride` is measured in fp16 elements, allowing a slice starting at
 * `full_A + k0` to be packed without a temporary compact At buffer. */
void i8_compute_dynamic_per_token_scale_f16(int M, int K,
                                             const uint16_t* src_fp16,
                                             int src_row_stride,
                                             float* scale_out);
void i8_pack_a_f16_quant_normal_strided(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst);
/* Quantize row-major fp16 and directly write native A
 * (align_in/16, M, 16). Rows are handled in groups of four so each K16 block
 * is emitted with contiguous stores. */
void i8_pack_a_f16_quant_native_strided(int M, int K, int align_in,
                                        const uint16_t* src_fp16,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst, int panel_rows = 0);

/* Full dynamic-per-token scale plus all normal-A K slices in one row-parallel
 * OpenMP region.  This avoids one OpenMP team launch per K wave. */
void i8_pack_a_f16_dynamic_normal_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const uint16_t* src_fp16,
                                        float* scale_out,
                                        int8_t* const* dst_slices);
void i8_pack_a_f16_dynamic_native_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const uint16_t* src_fp16,
                                        float* scale_out,
                                        int8_t* const* dst_slices, int panel_rows = 0);

void i8_compute_dynamic_per_token_scale_f32(int M, int K,
                                             const float* src_f32,
                                             int src_row_stride,
                                             float* scale_out);
void i8_pack_a_f32_quant_normal_strided(int M, int K, int align_in,
                                        const float* src_f32,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst);
void i8_pack_a_f32_quant_native_strided(int M, int K, int align_in,
                                        const float* src_f32,
                                        int src_row_stride,
                                        const float* per_token_scale,
                                        int8_t* dst, int panel_rows = 0);
void i8_pack_a_f32_dynamic_normal_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const float* src_f32,
                                        float* scale_out,
                                        int8_t* const* dst_slices);
void i8_pack_a_f32_dynamic_native_split(int M, int full_K,
                                        int slice_count,
                                        const int* slice_k0,
                                        const int* slice_k,
                                        const int* slice_align_in,
                                        const float* src_f32,
                                        float* scale_out,
                                        int8_t* const* dst_slices, int panel_rows = 0);

/* Fused row-major A -> all normal-layout split-K buffers. */
void i8_pack_a_i8_normal_split(int M, int full_K,
                               int slice_count,
                               const int* slice_k0,
                               const int* slice_k,
                               const int* slice_align_in,
                               const int8_t* src_i8,
                               int8_t* const* dst_slices);

/* Static per-token fp16 quantization + all normal-layout split-K buffers in one
 * row-parallel pass.  per_token_scale[M] must have been validated by caller. */
void i8_pack_a_f16_static_normal_split(int M, int full_K,
                                       int slice_count,
                                       const int* slice_k0,
                                       const int* slice_k,
                                       const int* slice_align_in,
                                       const uint16_t* src_fp16,
                                       const float* per_token_scale,
                                       int8_t* const* dst_slices);
void i8_pack_b_native_n32_k32(int K, int N, int align_in, int align_out,
                              const int8_t* src, int8_t* dst);
void i8_unpack_c_normal(int M, int N, int align_out, const void* src, void* dst);
void i8_unpack_c_native_n4_m4(int M, int N, const void* src, void* dst, int panel_rows = 0);

/*
 * Experimental LLM helper: read native int32 C directly and fuse dequant,
 * optional bias, optional activation, and output layout conversion.
 *
 * Inputs:
 *   src_native    native C layout (ceil(N/4), M, 4):
 *                 src[((n/4)*M + m)*4 + n%4].
 *   a_scale       optional per-token activation scale [M], defaults to 1.0.
 *   w_scale       optional per-output-channel weight scale [N], defaults to 1.0.
 *   bias          optional fp32 bias [N], added after dequant.
 *   act           None, Relu, Silu, or tanh-approx GELU.
 *
 * Output:
 *   dst_fp16      compact row-major fp16 bit patterns [M,N].
 *
 * Formula:
 *   y = int32_acc * a_scale[m] * w_scale[n] + bias[n]
 *   dst = fp16(act(y))
 *
 * The fp32 variant below keeps the same formula but skips fp16 narrowing, which
 * is useful for accuracy debugging. No cache synchronization is performed here.
 */
void i8_unpack_c_native_i32_dequant_f16(int M, int N,
                                        const int32_t* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        uint16_t* dst_fp16, int panel_rows = 0);
void i8_unpack_c_native_i32_dequant_f32(int M, int N,
                                        const int32_t* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        float* dst_f32, int panel_rows = 0);

/*
 * Split-K helpers for native int32 C partials.  Accumulate keeps the native
 * layout so it can run behind the next NPU K-wave without materializing a
 * row-major buffer.  The reduce helpers sum all partials exactly (modulo int32,
 * matching the NEON/NPU accumulator representation), then fuse de-tile,
 * dequant, optional bias/activation, and the final fp16/fp32 store.
 *
 * `src_native[p]` must point to `(ceil(N/4), M, 4)` and partial_count must be
 * positive.  Padding lanes in the last N4 group are accumulated but not stored.
 */
void i8_accumulate_c_native_i32(int M, int N,
                                const int32_t* src_native,
                                int32_t* accum_native, int panel_rows = 0);
void i8_reduce_c_native_i32_dequant_f16(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        uint16_t* dst_fp16, int panel_rows = 0);
void i8_reduce_c_native_i32_dequant_f32(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        const float* a_scale,
                                        const float* w_scale,
                                        const float* bias,
                                        ActivationOp act,
                                        float* dst_f32, int panel_rows = 0);

/* Exact split-K sum plus native-C -> compact row-major int32 de-tile.  Addition
 * is modulo 2^32, matching the NPU accumulator representation. */
void i8_reduce_c_native_i32_to_rowmajor(int M, int N,
                                        int partial_count,
                                        const int32_t* const* src_native,
                                        int32_t* dst_i32, int panel_rows = 0);

void f16_pack_a_normal(int M, int K, int align_in, const uint16_t* src, uint16_t* dst);
void f16_pack_a_normal_batch(int B, int M, int K, int align_in,
                             const uint16_t* src, uint16_t* dst);
/* Split compact A[M,K] into split-major normal-layout buffers
 * [split_count,M,part_k].  part_k is the aligned K handled by every NPU core;
 * the tail of the final slice is zero-filled. */
void f16_pack_a_normal_split(int M, int K, int split_count, int part_k,
                             const uint16_t* src, uint16_t* dst);
void f16_pack_a_native_k8_m8(int M, int K, int align_in,
                             const uint16_t* src, uint16_t* dst);
void f16_pack_a_native_k8_m8_batch(int B, int M, int K, int align_in,
                                   const uint16_t* src, uint16_t* dst);
void f16_pack_b_native_n16_k32(int K, int N, int align_in, int align_out,
                               const uint16_t* src, uint16_t* dst);
void f16_pack_b_native_n16_k32_batch(int B, int K, int N,
                                     int align_in, int align_out,
                                     const uint16_t* src, uint16_t* dst);
void f16_pack_operand_half_surface(int M, int N, int align_out, int m_tile,
                                   const uint16_t* src, uint16_t* dst);
void f16_unpack_d(int M, int N, int align_out, const uint16_t* src, uint16_t* dst);
void f16_unpack_d_f32(int M, int N, int align_out, const uint16_t* src, float* dst);
void f16_unpack_d_strided(int M, int N, int align_out,
                          const uint16_t* src,
                          int dst_row_stride, int dst_col_offset,
                          uint16_t* dst);
void f16_unpack_d_f32_strided(int M, int N, int align_out,
                              const uint16_t* src,
                              int dst_row_stride, int dst_col_offset,
                              float* dst);
void f16_unpack_d_compact(int M, int N, int align_out,
                          const uint16_t* src, uint16_t* dst);
void f16_unpack_d_compact_add_bias(int M, int N, int align_out,
                                   const uint16_t* src,
                                   const uint16_t* bias,
                                   uint16_t* dst);
void f16_unpack_d_compact_strided(int M, int N, int align_out,
                                  const uint16_t* src,
                                  int dst_row_stride, int dst_col_offset,
                                  uint16_t* dst);
void f16_unpack_d_compact_batch(int B, int M, int N, int align_out,
                                uint64_t src_batch_stride,
                                const uint16_t* src, uint16_t* dst);
void f16_unpack_d_f32_compact(int M, int N, int align_out,
                              const uint16_t* src, float* dst);
void f16_unpack_d_f32_compact_strided(int M, int N, int align_out,
                                      const uint16_t* src,
                                      int dst_row_stride, int dst_col_offset,
                                      float* dst);
void f16_unpack_d_native_n8_m8(int M, int N,
                               const uint16_t* src, uint16_t* dst);
void f16_unpack_d_native_n8_m8_batch(int B, int M, int N,
                                     uint64_t src_batch_stride,
                                     const uint16_t* src, uint16_t* dst);
void f16_unpack_d_f32_native_n8_m8(int M, int N,
                                   const uint16_t* src, float* dst);
/* Sum split-major compact FP16 partials in FP32, de-pad rows, then emit one
 * compact row-major result.  src_split_stride is measured in uint16_t
 * elements and normally equals M*align_out. */
void f16_reduce_d_compact_split_f16(int M, int N, int align_out,
                                    int split_count,
                                    uint64_t src_split_stride,
                                    const uint16_t* src, uint16_t* dst);
void f16_reduce_d_compact_split_f16_add_bias(
    int M, int N, int align_out, int split_count,
    uint64_t src_split_stride, const uint16_t* src,
    const uint16_t* bias, uint16_t* dst);
void f16_reduce_d_compact_split_f32(int M, int N, int align_out,
                                    int split_count,
                                    uint64_t src_split_stride,
                                    const uint16_t* src, float* dst);

} /* namespace rknpu2_matmul_open::cpu */

#endif /* RK_NPU_CPU_KERNELS_H */

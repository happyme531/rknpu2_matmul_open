#ifndef RK_NPU_W4A8_CPU_H
#define RK_NPU_W4A8_CPU_H
#include "rk_npu_i4_plan.h"
namespace rknpu2_matmul_open::detail {
struct W4A8Input {
    const void *A = nullptr;
    bool half = false;
    int M = 0, K = 0;
    const float *scales = nullptr;
    const float *inverse = nullptr;
    int threads = 1; // private producer owns parallelism; public callback stays serial
    I4InputLayout layout = I4InputLayout::Native;
};
// Paired native N32 blocks, prepared once for a workspace's fixed geometry.
struct W4A8SwiGLUPair {
    uint64_t gate_partial, up_partial; // offsets in INT16 elements
    uint64_t gate_accumulator, up_accumulator; // offsets in INT32 elements
    int m0, rows, n0;
};
std::vector<W4A8SwiGLUPair> w4a8_swiglu_pairs(int N, const std::vector<I4Tile>&);
struct W4A8Reduction {
    int32_t *accumulator; // tile / N8 / logical M / 8, including N64 padding
    const int32_t *correction;
    const float *a_scale;
    const float *w_scale;
    void *output;
    bool half;
    const std::vector<W4A8SwiGLUPair> *swiglu_pairs = nullptr;
    uint32_t *row_maxima = nullptr; // private [cpu_threads, row_maxima_stride]
    int row_maxima_stride = 0;
};
int w4a8_scales(const void *A, bool half, int M, int K, int threads, float *scales, float *inverse);
int w4a8_pack(void *user, const rk_npu_i4_input_tile *tile);
int w4a8_reduce(void *user, const I4Plan &plan, int wave, const int16_t *partial);
// Private SwiGLU consumers assume geometry/state validated by workspace creation
// and public entrypoints; the executor guarantees two synchronized partials.
int w4a8_reduce_two(void *user, const I4Plan&, const int16_t*, const int16_t*);
int w4a8_reduce_two_tiles(const W4A8Reduction&, const rk_npu_i4_config&,
                          const int16_t*, const int16_t*);
int w4a8_reduce_swiglu(const W4A8Reduction&, const rk_npu_i4_config&, const int16_t*, bool first);
int w4a8_finish_scales(int M, int threads, const uint32_t *maxima, int stride,
                       float *scales, float *inverse);
// CPU-only boundary: NPU tiles describe storage, not the CPU work partition.
int w4a8_reduce_tiles(const W4A8Reduction&, const rk_npu_i4_config&,
                      const std::vector<I4Tile>&, int wave, int waves, const int16_t* partial);
void w4a8_dequant(int M, int N, const int32_t *accumulator, const float *a_scale,
                  const float *w_scale, bool half, void *output, int threads);
} // namespace rknpu2_matmul_open::detail
#endif

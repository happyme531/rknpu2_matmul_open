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
struct W4A8Reduction {
    int32_t *accumulator; // tile / N8 / logical M / 8, including N64 padding
    const int32_t *correction;
    const float *a_scale;
    const float *w_scale;
    void *output;
    bool half;
};
int w4a8_scales(const void *A, bool half, int M, int K, int threads, float *scales, float *inverse);
int w4a8_pack(void *user, const rk_npu_i4_input_tile *tile);
int w4a8_reduce(void *user, const I4Plan &plan, int wave, const int16_t *partial);
// CPU-only boundary: NPU tiles describe storage, not the CPU work partition.
int w4a8_reduce_tiles(const W4A8Reduction&, const rk_npu_i4_config&,
                      const std::vector<I4Tile>&, int wave, int waves, const int16_t* partial);
void w4a8_dequant(int M, int N, const int32_t *accumulator, const float *a_scale,
                  const float *w_scale, bool half, void *output, int threads);
} // namespace rknpu2_matmul_open::detail
#endif

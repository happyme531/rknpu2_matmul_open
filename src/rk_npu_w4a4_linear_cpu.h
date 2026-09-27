#ifndef RK_NPU_W4A4_LINEAR_CPU_H
#define RK_NPU_W4A4_LINEAR_CPU_H

#include "rk_npu_i4_plan.h"
#include "rk_npu_w4a4_linear.h"
#include <vector>

namespace rknpu2_matmul_open::detail {
struct W4A4Input {
    const uint8_t *packed = nullptr;
    int M = 0, K = 0;
    int threads = 1;
};
struct W4A4Reduction {
    int32_t *accumulator = nullptr;
    const float *a_scale = nullptr;
    const float *w_scale = nullptr;
    void *output = nullptr;
    bool half = false;
};

int w4a4_pack(void *user, const rk_npu_i4_input_tile *tile);
int w4a4_reduce(void *user, const I4Plan &plan, int wave, const int16_t *partial);
int w4a4_reduce_tiles(const W4A4Reduction &state, const rk_npu_i4_config &cfg,
                      const std::vector<I4Tile> &tiles, int wave, int waves,
                      const int16_t *partial);
} // namespace rknpu2_matmul_open::detail
#endif

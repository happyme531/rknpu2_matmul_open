#ifndef RK_NPU_I4_BOUNDS_H
#define RK_NPU_I4_BOUNDS_H
#include <cstdint>
namespace rknpu2_matmul_open::detail {
constexpr int I4_K_TILE_MAX = 2048; // native geometry: <=128 KiB packed A per task
constexpr int I4_K_TILE_UNIVERSAL = 480;
struct I4WeightBound {
    int k_tile = 0;
    int64_t positive = 0, negative_magnitude = 0;
    uint64_t safe_tiles = 0; // bit (kt/32-1): this exact fixed partition is certified
};
/* requested=0 selects the largest safe fixed 32-aligned partition <=2048.
 * Positive requested checks that partition. All codes are validated. Bounds
 * hold for every possible signed INT4 A, not just sampled activations.
 * Fixed-partition safety is not monotonic in k_tile; search all candidates. */
int bound_i4_weights(int K, int N, const int8_t *B, int requested, I4WeightBound &out);
} // namespace rknpu2_matmul_open::detail
#endif

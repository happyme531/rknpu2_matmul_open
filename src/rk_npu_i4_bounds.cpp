#include "rk_npu_i4_bounds.h"
#include "rk_npu_matmul_i4.h"
#include <algorithm>
#include <array>
#include <climits>
#include <vector>

namespace rknpu2_matmul_open::detail {
int bound_i4_weights(int K, int N, const int8_t *B, int requested, I4WeightBound &out) {
    if (!B || K < 1 || K > INT32_MAX / 64 || N < 1 || N > INT_MAX - 63 || requested < 0 ||
        requested > I4_K_TILE_MAX || requested % 32)
        return RK_NPU_ERR_PARAM;
    const int blocks = (K + 31) / 32;
    const int limit = requested ? requested / 32 : std::min(I4_K_TILE_MAX / 32, blocks);
    std::array<bool, I4_K_TILE_MAX / 32 + 1> safe{};
    std::array<int64_t, I4_K_TILE_MAX / 32 + 1> max_pos{}, max_neg{};
    for (int t = 1; t <= limit; ++t)
        safe[t] = !requested || t == limit;
    // Use column slabs to avoid retaining a full K*N prefix tensor.
    constexpr int slab = 128;
    for (int n0 = 0; n0 < N;) {
        const int width = std::min(slab, N - n0);
        std::vector<int64_t> pos(size_t(blocks + 1) * width), neg(pos.size());
        for (int b = 0; b < blocks; ++b) {
            auto *p = pos.data() + size_t(b + 1) * width;
            auto *q = neg.data() + size_t(b + 1) * width;
            std::copy(p - width, p, p);
            std::copy(q - width, q, q);
            for (int k = b * 32; k < std::min(K, (b + 1) * 32); ++k) {
                const auto *row = B + size_t(k) * N + n0;
                for (int n = 0; n < width; ++n) {
                    const int w = row[n];
                    if (w < -8 || w > 7)
                        return RK_NPU_ERR_PARAM;
                    p[n] += w >= 0 ? 7 * w : -8 * w;
                    q[n] += w >= 0 ? 8 * w : -7 * w;
                }
            }
        }
        for (int t = 1; t <= limit; ++t) {
            if (!safe[t])
                continue;
            for (int b = 0; b < blocks && safe[t]; b += t) {
                const int e = std::min(blocks, b + t);
                for (int n = 0; n < width; ++n) {
                    const int64_t hi = pos[size_t(e) * width + n] - pos[size_t(b) * width + n];
                    const int64_t lo = neg[size_t(e) * width + n] - neg[size_t(b) * width + n];
                    max_pos[t] = std::max(max_pos[t], hi);
                    max_neg[t] = std::max(max_neg[t], lo);
                    if (hi > 32767 || lo > 32768) {
                        safe[t] = false;
                        break;
                    }
                }
            }
        }
        n0 += width;
    }
    uint64_t mask=0;
    for(int t=1;t<=limit;++t) if(safe[t]) mask|=uint64_t(1)<<(t-1);
    for (int t = limit; t >= 1; --t)
        if (safe[t]) {
            out = {t * 32, max_pos[t], max_neg[t], mask};
            return RK_NPU_OK;
        }
    return RK_NPU_ERR_PARAM;
}
} // namespace rknpu2_matmul_open::detail

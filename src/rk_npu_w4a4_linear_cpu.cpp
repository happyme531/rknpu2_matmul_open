#include "rk_npu_w4a4_linear_cpu.h"
#include "rk_npu_i4_cpu.h"
#include <cstring>

namespace rknpu2_matmul_open::detail {
int w4a4_pack(void *user, const rk_npu_i4_input_tile *tile) {
    if (!user || !tile)
        return RK_NPU_ERR_PARAM;
    const auto &input = *static_cast<const W4A4Input *>(user);
    if (!input.packed || !tile->dst || input.M < 1 || input.K < 1 || input.threads < 1 ||
        input.threads > 4 || tile->m0 < 0 || tile->rows < 1 ||
        int64_t(tile->m0) + tile->rows > input.M || tile->k0 < 0 || tile->k0 % 32 || tile->k < 1 ||
        int64_t(tile->k0) + tile->k > input.K || tile->padded_k < tile->k || tile->padded_k % 32 ||
        tile->padded_k > I4_K_TILE_MAX ||
        int64_t(tile->k0) + tile->padded_k > align_up(input.K, 32) ||
        tile->bytes < uint64_t(tile->rows) * tile->padded_k / 2)
        return RK_NPU_ERR_PARAM;
#pragma omp parallel for num_threads(                                                              \
        input.threads) if (input.threads > 1 && int64_t(tile->rows) * tile->padded_k >= 65536)
    for (int kb = 0; kb < tile->padded_k / 32; ++kb) {
        const uint8_t *src = input.packed + (size_t(tile->k0 / 32 + kb) * input.M + tile->m0) * 16;
        std::memcpy(tile->dst + size_t(kb) * tile->rows * 16, src, size_t(tile->rows) * 16);
    }
    return RK_NPU_OK;
}

} // namespace rknpu2_matmul_open::detail

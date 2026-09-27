#ifndef RK_NPU_DCOMP_H
#define RK_NPU_DCOMP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rknpu2_matmul_open::detail {
constexpr uint32_t DCOMP_NORMAL_REGNUM = 0x55555555u;
struct DcompEncoded {
    std::vector<uint8_t> data;
    std::array<uint32_t, 16> amounts{};
};
/* Input/output is padded native N32/K32, not logical row-major weights. */
int dcomp_encode(const int8_t *native, int aligned_k, int aligned_n, DcompEncoded &out,
                 bool count_only = false);
int dcomp_decode(const uint8_t *data, size_t bytes, const std::array<uint32_t, 16> &amounts,
                 int aligned_k, int aligned_n, std::vector<int8_t> &native);
/* Basic, uncalibrated per-channel RTN. Searches a bounded set of integer
 * limits under the actual full-N, K-wave normal-codec payload budget.
 * Returns PARAM for invalid/nonfinite input or an unmet budget. */
int dcomp_quantize_f32(int K, int N, int k_tile, const float *weights_kn, float target_bpw,
                       std::vector<int8_t> &codes_kn, std::vector<float> &scales);
} // namespace rknpu2_matmul_open::detail
#endif

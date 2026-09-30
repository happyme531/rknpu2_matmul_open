#ifndef RK_NPU_BFLOAT_BITS_H
#define RK_NPU_BFLOAT_BITS_H
#include <cstdint>
#include <cstring>

namespace rknpu2_matmul_open::bits {
// Raw BF16 storage, widened exactly to FP32. No FP16 numerical conversion.
inline float bfloat_to_float(uint16_t value) {
    const uint32_t raw = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}
// RNE for finite values, retaining signed zero and infinity. Quiet NaNs and
// keep their sign/high payload; never round a NaN into an infinity or zero.
inline uint16_t float_to_bfloat(float value) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    if ((raw & 0x7f800000u) == 0x7f800000u && (raw & 0x007fffffu))
        return uint16_t((raw >> 16) | 0x0040u);
    return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}
}
#endif

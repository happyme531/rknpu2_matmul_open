#ifndef RK_NPU_HALF_BITS_H
#define RK_NPU_HALF_BITS_H
#include <cstdint>
#include <cstring>
/* Portable binary16 conversions for small CPU kernels, independent of NPU
 * math. Preserve signed zero; round f32->f16 to nearest-even. */
namespace rknpu2_matmul_open::bits {
inline float half_to_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 31, man = h & 1023, bits;
    if (!exp) {
        if (!man)
            bits = sign;
        else {
            int e = -14;
            while (!(man & 1024)) {
                man <<= 1;
                --e;
            }
            bits = sign | uint32_t(e + 127) << 23 | (man & 1023) << 13;
        }
    } else if (exp == 31)
        bits = sign | 0x7f800000 | man << 13;
    else
        bits = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, raw_exp = (x >> 23) & 255;
    uint32_t man = x & 0x7fffff;
    int exp = int(raw_exp) - 127 + 15;
    if (raw_exp == 255)
        return uint16_t(sign | 0x7c00 | (man ? ((man >> 13) | 0x200) : 0));
    if (exp <= 0) {
        if (exp < -10)
            return uint16_t(sign);
        man |= 0x800000;
        const int shift = 14 - exp;
        uint32_t hm = man >> shift, rb = 1u << (shift - 1);
        if ((man & rb) && ((man & (rb - 1)) || (hm & 1)))
            ++hm;
        return uint16_t(sign | hm);
    }
    if (exp >= 31)
        return uint16_t(sign | 0x7c00);
    uint32_t h = sign | uint32_t(exp) << 10 | man >> 13;
    if ((man & 0x1000) && ((man & 0xfff) || (h & 1)))
        ++h;
    return uint16_t(h);
}
} // namespace rknpu2_matmul_open::bits
#endif

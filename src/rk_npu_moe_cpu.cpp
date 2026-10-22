#include "rk_npu_moe_cpu.h"
#include "rk_npu_swiglu_fp32_neon.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace rknpu2_matmul_open::moe {
namespace {
#if defined(__aarch64__)
int32x4_t dot4(const MiddleRow& r, int col) {
    auto value = integer4(r.partial, r.rows, r.row, col, r.correction);
    for (int wave = 1; wave < r.waves; ++wave) {
        const auto* p = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const uint8_t*>(r.partial) + size_t(wave) * r.wave_stride);
        value = vaddq_s32(value, integer_i4_4(p, r.rows, r.row, col));
    }
    return value;
}
int8x16_t quant16(const float* values, float inverse) {
    const auto inv = vdupq_n_f32(inverse);
    const auto lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    const auto convert = [&](int offset) {
        // MoE retains nearest-even rounding, unlike dense W4A8's ties-away.
        return vqmovn_s32(vmaxq_s32(lo, vminq_s32(hi,
            vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values + offset), inv)))));
    };
    return vcombine_s8(vqmovn_s16(vcombine_s16(convert(0), convert(4))),
                       vqmovn_s16(vcombine_s16(convert(8), convert(12))));
}
void pack_nibbles(uint8x16_t values, uint8_t* output) {
    const auto even = vget_low_u8(vuzp1q_u8(values, values));
    const auto odd = vget_low_u8(vuzp2q_u8(values, values));
    vst1_u8(output, vorr_u8(vshl_n_u8(even, 4), odd));
}
#endif
} // namespace

float swiglu_quantize_row(const MiddleRow& r) {
    alignas(64) float hidden[512];
    uint32_t maximum = 0;
#if defined(__aarch64__)
    auto vmax = vdupq_n_u32(0);
    const auto sa = vdupq_n_f32(r.input_scale);
    for (int n = 0; n < r.channels; n += 16) {
        alignas(16) float gate[16], up[16];
        for (int j = 0; j < 16; j += 4) {
            const auto g = vmulq_f32(vmulq_f32(vcvtq_f32_s32(dot4(r, n + j)), sa),
                                      vld1q_f32(r.weight_scales + n + j));
            const auto u = vmulq_f32(vmulq_f32(vcvtq_f32_s32(dot4(r, r.channels + n + j)), sa),
                                      vld1q_f32(r.weight_scales + r.channels + n + j));
            vst1q_f32(gate + j, g);
            vst1q_f32(up + j, u);
        }
        vmax = vmaxq_u32(vmax, detail::swiglu_fp32::apply16<true>(gate, up, hidden + n));
    }
    maximum = vmaxvq_u32(vmax);
#else
    const auto dot = [&](int col) {
        int32_t value = integer_at(r.partial, r.rows, r.row, col, r.correction);
        for (int wave = 1; wave < r.waves; ++wave) {
            const auto* p = reinterpret_cast<const int32_t*>(
                reinterpret_cast<const uint8_t*>(r.partial) + size_t(wave) * r.wave_stride);
            value += integer_i4_at(p, r.rows, r.row, col);
        }
        return value;
    };
    for (int n = 0; n < r.channels; ++n) {
        const float gate = (float(dot(n)) * r.input_scale) * r.weight_scales[n];
        const float up = (float(dot(r.channels + n)) * r.input_scale) * r.weight_scales[r.channels + n];
        const float y = (gate / (1.f + std::exp(-gate))) * up;
        hidden[n] = y;
        uint32_t bits;
        std::memcpy(&bits, &y, sizeof(bits));
        maximum = std::max(maximum, bits & 0x7fffffffu);
    }
#endif
    float maxabs;
    std::memcpy(&maxabs, &maximum, sizeof(maxabs));
    // Preserve MoE row scaling, including scale=1 for all-zero rows.
    const float scale = maxabs ? maxabs / 127.f : 1.f, inverse = 1.f / scale;
#if defined(__aarch64__)
    for (int n = 0; n < r.channels; n += 16) {
        const auto q = quant16(hidden + n, inverse);
        if (r.layout == MiddleLayout::I4Native) {
            auto* p = reinterpret_cast<uint8_t*>(r.output) + size_t(n / 32) * r.rows * 32 + r.row * 32 + (n % 32) / 2;
            const auto bits = vreinterpretq_u8_s8(q);
            pack_nibbles(vshrq_n_u8(bits, 4), p);
            pack_nibbles(veorq_u8(vandq_u8(bits, vdupq_n_u8(15)), vdupq_n_u8(8)), p + 16);
        } else {
            auto* p = r.output + (r.layout == MiddleLayout::I8Native
                ? size_t(n / 16) * r.rows * 16 + r.row * 16 : size_t(r.row) * r.channels + n);
            vst1q_s8(p, q);
        }
    }
#else
    for (int n = 0; n < r.channels; ++n) {
        const float x = std::max(-127.f, std::min(127.f, hidden[n] * inverse));
        const float base = std::floor(x), fraction = x - base;
        int code = int(base);
        if (fraction > .5f || (fraction == .5f && (code & 1))) ++code;
        if (r.layout == MiddleLayout::I4Native) {
            auto* p = reinterpret_cast<uint8_t*>(r.output) + size_t(n / 32) * r.rows * 32 + r.row * 32 + (n % 32) / 2;
            const uint8_t q = uint8_t(code), h = q >> 4, l = (q & 15) ^ 8;
            if (n & 1) { p[0] |= h; p[16] |= l; }
            else { p[0] = h << 4; p[16] = l << 4; }
        } else {
            const size_t index = r.layout == MiddleLayout::I8Native
                ? size_t(n / 16) * r.rows * 16 + r.row * 16 + n % 16 : size_t(r.row) * r.channels + n;
            r.output[index] = int8_t(code);
        }
    }
#endif
    return scale;
}
} // namespace rknpu2_matmul_open::moe

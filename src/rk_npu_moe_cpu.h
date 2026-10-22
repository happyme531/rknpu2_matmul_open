#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::moe {
inline int32_t integer_i4_at(const int32_t* data,int rows,int row,int n) {
    const auto* p=reinterpret_cast<const int16_t*>(data)+size_t(n/8)*rows*16+row*16+n%8;
    return 16*int32_t(p[0])+p[8];
}
inline int32_t integer_at(const int32_t* data,int rows,int row,int n,const int32_t* correction) {
    if(!correction)return data[size_t(n/4)*rows*4+row*4+n%4];
    return integer_i4_at(data,rows,row,n)+correction[n];
}
#if defined(__aarch64__)
inline int32x4_t integer_i4_4(const int32_t* data,int rows,int row,int n) {
    const auto* p=reinterpret_cast<const int16_t*>(data)+size_t(n/8)*rows*16+row*16+n%8;
    return vaddw_s16(vmull_n_s16(vld1_s16(p),16),vld1_s16(p+8));
}
inline int32x4_t integer4(const int32_t* data,int rows,int row,int n,const int32_t* correction) {
    if(!correction)return vld1q_s32(data+size_t(n/4)*rows*4+row*4);
    return vaddq_s32(integer_i4_4(data,rows,row,n),vld1q_s32(correction+n));
}
#endif

enum class MiddleLayout { I8Normal, I8Native, I4Native };
struct MiddleRow {
    const int32_t *partial;
    const int32_t *correction; // null for W8; applied once for W4
    const float *weight_scales; // gate followed by up
    float input_scale;
    int rows, row, channels, waves;
    size_t wave_stride; // bytes between retained W4 K partials
    int8_t *output; // start of this job's Down input
    MiddleLayout layout;
};
// Validated MoE geometry: channels is a multiple of 32, at most 512.
// Returns the full-row A8 scale. NOTE: internal activations must remain finite
// and their nonzero row scale and reciprocal must be representable in FP32;
// this hot kernel does not validate intermediate values.
float swiglu_quantize_row(const MiddleRow&);
} // namespace rknpu2_matmul_open::moe

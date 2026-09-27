#ifndef RK_NPU_I4_CPU_H
#define RK_NPU_I4_CPU_H
#include "rk_npu_matmul_i4.h"
namespace rknpu2_matmul_open::detail {
int pack_i4_weights(int K, int N, int k0, int k, int padded_k, int padded_n,
                    const int8_t* B, uint8_t* dst);
void reduce_i4_partial(int rows, int logical_n, int padded_n,
                       const int16_t* src, int32_t* dst, int ldc, bool add);
}
#endif

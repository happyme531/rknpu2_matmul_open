#ifndef RK_NPU_I4_REGS_H
#define RK_NPU_I4_REGS_H
#include <cstdint>
#include "rk_npu_i4_layout.h"
#include <vector>
namespace rknpu2_matmul_open::detail {
constexpr int I4_BODY_WORDS = 108;
constexpr int I4_COMMAND_WORDS = 112;
constexpr int I4_WEIGHT_WORD = 34;
/* Native A[K/32,M,32], B[N/64,K/32,64,32], C[N/8,M,8] INT16.
 * Geometry: M<=128,K<=2048,N<=4096, K32/N64 aligned. Weight creation proves
 * INT16 bounds for K>480; the explicit W4A8 multiplier experiment may waive it.
 * Private Panel8 option: A[M/8,K/32,8,32], M<=512 with an 8-bank budget;
 * final M<8 groups keep native layout. B/C formats and command size are unchanged.
 * Produces register body only; ownership, PC chains and submits are separate. */
int make_i4_regs(int M, int K, int N, uint64_t a, uint64_t b, uint64_t c,
                  std::vector<uint64_t>& body, I4InputLayout layout = I4InputLayout::Native);
}
#endif

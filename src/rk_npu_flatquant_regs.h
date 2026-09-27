#ifndef RK_NPU_FLATQUANT_REGS_H
#define RK_NPU_FLATQUANT_REGS_H
#include "rk_npu_internal.h"
namespace rknpu2_matmul_open::detail {
constexpr int FLAT_COMMAND_WORDS = 128;
/* Native FP16 input; fold2 writes the transpose required by left x I2. */
std::vector<uint64_t> flat_regs(int m, int k, int n, int full_m, bool fold2, uint64_t a, uint64_t b,
                                uint64_t out);
} // namespace rknpu2_matmul_open::detail
#endif

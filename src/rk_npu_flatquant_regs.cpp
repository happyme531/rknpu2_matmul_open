#include "rk_npu_flatquant_regs.h"
#include <algorithm>

namespace rknpu2_matmul_open::detail {
std::vector<uint64_t> flat_regs(int m, int k, int n, int full_m, bool fold, uint64_t a, uint64_t b,
                                uint64_t out) {
    const int banks = std::max(1, ceil_div(m * k * 2, CBUF_BANK_SIZE));
    const uint32_t cbuf = ((12 - banks) << 4) | banks;
    const uint32_t conv = 0x120u | (m < full_m ? 0x20000000u : 0u);
    std::vector<uint64_t> v;
    v.reserve(112);
    auto cna = [&](uint32_t r, uint32_t x) { v.push_back(E(T_CNA, r, x)); };
    auto core = [&](uint32_t r, uint32_t x) { v.push_back(E(T_CORE, r, x)); };
    auto dpu = [&](uint32_t r, uint32_t x) { v.push_back(E(T_DPU, r, x)); };
    cna(0x1040, cbuf);
    cna(0x1104, 0);
    cna(0x1100, 0);
    cna(0x100c, conv);
    dpu(0x4004, 0xe);
    cna(0x100c, conv);
    cna(0x1010, 0x20);
    cna(0x1014, 9);
    cna(0x1020, (m << 16) | 1);
    cna(0x1024, ((k - 1) << 16) | k);
    cna(0x1028, m);
    cna(0x102c, m);
    cna(0x1030, k * n * 2);
    cna(0x1034, k * 2);
    cna(0x1038, 0x01010000 | n);
    cna(0x1040, cbuf);
    cna(0x1044, ceil_div(m * k, 32));
    cna(0x104c, 0xb);
    for (uint32_t r = 0x1050; r <= 0x105c; r += 4)
        cna(r, 0x10000);
    cna(0x1060, 0);
    cna(0x1064, 0);
    cna(0x1068, 0);
    cna(0x1070, uint32_t(a));
    cna(0x1074, 0);
    cna(0x1078, 0xf000f);
    cna(0x107c, m < full_m ? uint32_t(full_m) : 0x200u);
    cna(0x1080, m < full_m ? uint32_t(full_m - m) : 0x10000000u - uint32_t(full_m) * 3);
    cna(0x1084, (m << 16) | 1);
    cna(0x1088, k);
    cna(0x1100, 0);
    cna(0x1104, 0);
    cna(0x1110, uint32_t(b));
    for (uint32_t r = 0x1140; r <= 0x1184; r += 4)
        cna(r, 0);
    core(0x3010, 0x200);
    core(0x3014, m - 1);
    core(0x3018, n - 1);
    core(0x301c, 0);
    core(0x3030, 0);
    dpu(0x400c, 0x1e4);
    dpu(0x4010, 0x48000002);
    dpu(0x4014, 0);
    dpu(0x4020, uint32_t(out));
    dpu(0x4024, fold ? 16 : full_m * 16);
    dpu(0x4030, fold ? 0 : m - 1);
    dpu(0x4034, fold ? m - 1 : 0);
    dpu(0x4038, fold ? uint32_t(((n / 2 - 1) << 16) | (n / 2 - 1)) : 0);
    dpu(0x403c, ((n - 1) << 16) | (n - 1));
    dpu(0x4040, 0x53);
    for (uint32_t r = 0x4044; r <= 0x404c; r += 4)
        dpu(r, 0);
    dpu(0x4050, fold ? 0x080007fe : 0x126);
    dpu(0x4054, 0);
    dpu(0x4058,
        fold ? 0x08000000u | uint32_t((n / 16 - 1) << 16) | uint32_t(n / 2 - 1) : uint32_t(n - 1));
    dpu(0x405c, fold ? uint32_t((m / 4 - 1) << 16) : uint32_t(m - 1));
    dpu(0x4060, 0x53);
    dpu(0x4064, 0);
    dpu(0x4068, 0);
    dpu(0x406c, 0);
    dpu(0x4070, 0x383);
    dpu(0x4074, 0);
    dpu(0x4078, 1);
    dpu(0x407c, 0);
    dpu(0x4080, 0);
    dpu(0x4084, 0x10001);
    dpu(0x4088, 0);
    for (uint32_t r = 0x4090; r <= 0x40ac; r += 4)
        dpu(r, 0);
    dpu(0x40c0, fold ? 128 : full_m * 32);
    dpu(0x40c4, 0);
    for (uint32_t r = 0x4100; r <= 0x412c; r += 4)
        dpu(r, 0);
    return v;
}
} // namespace rknpu2_matmul_open::detail

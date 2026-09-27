#include "rk_npu_i4_regs.h"
#include "rk_npu_internal.h"
#include "rk_npu_i4_bounds.h"
#include <algorithm>

namespace rknpu2_matmul_open::detail {
int make_i4_regs(int m, int k, int n, uint64_t a, uint64_t b, uint64_t c,
                  std::vector<uint64_t>& out, I4InputLayout layout) {
    const bool panel = layout == I4InputLayout::Panel8 && m >= 8;
    if (layout != I4InputLayout::Native && layout != I4InputLayout::Panel8) return RK_NPU_ERR_PARAM;
    if (m < 1 || m > (panel ? 512 : 128) || k < 32 || k > I4_K_TILE_MAX || k % 32 ||
        n < 64 || n > 4096 || n % 64 || ((a | b | c) & 15) ||
        (a | b | c) > UINT32_MAX) return RK_NPU_ERR_PARAM;
    if (panel && (m % 8 || i4_panel_banks(m,k) > 8)) return RK_NPU_ERR_PARAM;
    const int width = panel ? 8 : m, height = panel ? m / 8 : 1;
    const int grains = panel ? std::min(height+1, (ceil_div(65536,4*k)+1)&~1) : 2;
    const uint32_t banks = panel ? i4_panel_banks(m,k) : std::max(1, ceil_div(m*k/2,32768));
    const uint32_t cbuf = ((12 - banks) << 4) | banks;
    out.clear(); out.reserve(I4_BODY_WORDS);
    auto put = [&](uint64_t target, uint32_t r, uint32_t v) { out.push_back(E(target, r, v)); };
    auto cna = [&](uint32_t r, uint32_t v) { put(T_CNA, r, v); };
    auto dpu = [&](uint32_t r, uint32_t v) { put(T_DPU, r, v); };
    auto core = [&](uint32_t r, uint32_t v) { put(T_CORE, r, v); };
    cna(0x1040,cbuf); cna(0x1104,0); cna(0x1100,0); cna(0x100c,panel?0x20000360:0x360);
    dpu(0x4004,0xe);
    cna(0x100c,panel?0x20000360:0x360); cna(0x1010,grains<<4); cna(0x1014,9);
    cna(0x1020,(width<<16)|height); cna(0x1024,((k-1)<<16)|k);
    cna(0x1028,width); cna(0x102c,m); cna(0x1030,k*n/2); cna(0x1034,k/2);
    cna(0x1038,0x01010000|n); cna(0x1040,cbuf); cna(0x1044,panel?i4_panel_entries(k):m*ceil_div(k,128));
    cna(0x104c,0xb);
    for (uint32_t r=0x1050;r<=0x105c;r+=4) cna(r,0x10000);
    cna(0x1060,0); cna(0x1064,0); cna(0x1068,0); cna(0x1070,uint32_t(a));
    cna(0x1074,0); cna(0x1078,0xf000f); cna(0x107c,panel?k/4:4*m); cna(0x1080,panel?0:(uint32_t(-3*m)&0xfffffff));
    cna(0x1084,(width<<16)|height); cna(0x1088,k); cna(0x1100,0); cna(0x1104,0); cna(0x1110,uint32_t(b));
    for (uint32_t r=0x1140;r<=0x1184;r+=4) cna(r,0);
    core(0x3010,0x601); core(0x3014,((height-1)<<16)|(width-1)); core(0x3018,n-1); core(0x301c,0); core(0x3030,0);
    dpu(0x400c,0x1e4); dpu(0x4010,0x38000006); dpu(0x4014,0);
    dpu(0x4020,uint32_t(c)); dpu(0x4024,16*m); dpu(0x4030,m-1); dpu(0x4034,0); dpu(0x4038,0);
    dpu(0x403c,((n-1)<<16)|(n-1)); dpu(0x4040,0x53); dpu(0x4044,0); dpu(0x4048,0); dpu(0x404c,0);
    dpu(0x4050,0x7fe); dpu(0x4054,0); dpu(0x4058,n-1); dpu(0x405c,m-1);
    dpu(0x4060,0x53); dpu(0x4064,0); dpu(0x4068,0); dpu(0x406c,0);
    dpu(0x4070,0x383); dpu(0x4074,0); dpu(0x4078,1); dpu(0x407c,0);
    dpu(0x4080,0); dpu(0x4084,1); dpu(0x4088,0);
    for (uint32_t r=0x4090;r<=0x40ac;r+=4) dpu(r,0);
    dpu(0x40c0,128*m); dpu(0x40c4,0);
    for (uint32_t r=0x4100;r<=0x412c;r+=4) dpu(r,0);
    return out.size()==I4_BODY_WORDS ? RK_NPU_OK : RK_NPU_ERR_PARAM;
}
}

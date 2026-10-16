#pragma once
#include "rk_npu_attention_f16_internal.h"
#include <cmath>
namespace rknpu2_matmul_open::attention {
using namespace detail;
inline std::vector<uint64_t> lut_exp_regs(int count, uint64_t input, uint64_t output, bool load) {
    std::vector<uint64_t> v;
    v.reserve(load ? 1095 : 67);
    auto d = [&](unsigned r, unsigned x) { v.push_back(E(T_DPU, r, x)); };
    auto r = [&](unsigned a, unsigned x) { v.push_back(E(T_RDMA, a, x)); };
    // Two 510-interval ranges cover [-16,-8] and [-8,0]; final entries guard
    // interpolation. Positive LUT codes avoid the zero-conversion anomaly.
    // The PV consumer removes the affine output bias from numerator/denominator.
    if (load) {
        static const std::vector<uint64_t> table = []{
            std::vector<uint64_t> words;words.reserve(1028);
            constexpr int bias=16;
            // Leave headroom for the carrier; the common gain cancels in finish().
            constexpr double gain=255./256.;
            for(int table=0;table<2;++table){
                words.push_back(E(T_DPU,0x4100,unsigned(2 | table) << 16));
                for(int i=0;i<513;++i){
                    const double x=-16.+table*8.+std::min(i,510)*(8./510.);
                    const int value=std::clamp(int(std::nearbyint(std::exp(x)*(32768.*gain)))+bias,bias,32767);
                    words.push_back(E(T_DPU,0x4104,unsigned(value)));
                }
            }
            return words;
        }();
        v.insert(v.end(),table.begin(),table.end());
    }
    const unsigned width = unsigned(count / 8 - 1);
    d(0x4004, 14); r(0x5004, 14);
    d(0x400c, 0x1e5); d(0x4010, 0x48000002); d(0x4014, 0);
    d(0x4020, unsigned(output)); d(0x4024, count * 2);
    d(0x4030, width); d(0x4034, 0); d(0x4038, 0); d(0x403c, 0x70007);
    d(0x4040, 0x53); d(0x4044, 0); d(0x4048, 0); d(0x404c, 0);
    d(0x4050, 2); d(0x4054, 0); d(0x4058, 7); d(0x405c, width);
    d(0x4060, 0x20040); d(0x4064, 0x80000000);
    d(0x4068, unsigned(half(2040.f)) << 16); d(0x406c, 0);
    d(0x4070, 0x302); d(0x4074, 0); d(0x4078, 1); d(0x407c, 0);
    d(0x4080, 0); d(0x4084, 0x10001); d(0x4088, 15 << 12);
    for (unsigned a = 0x4090; a <= 0x40ac; a += 4) d(a, 0);
    d(0x40c0, count * 2); d(0x40c4, 0);
    d(0x4108, 0x68); d(0x410c, 0x50500);
    d(0x4110, unsigned(-32640)); d(0x4114, unsigned(-16320));
    d(0x4118, unsigned(-16320)); d(0x411c, 64);
    d(0x4120, 0); d(0x4124, 0); d(0x4128, 0); d(0x412c, 0);
    r(0x500c, width); r(0x5010, 0); r(0x5014, 7); r(0x5018, unsigned(input));
    r(0x501c, 0); r(0x5020, 0); r(0x5028, 0); r(0x502c, 0);
    r(0x5034, 1); r(0x5038, 0); r(0x5040, 0); r(0x5044, 0x17849);
    r(0x5048, 0); r(0x504c, 0); r(0x5064, 0); r(0x5068, 0x01010101); r(0x506c, 0);
    return v;
}

}

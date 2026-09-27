#pragma once
#include "rk_npu_internal.h"
#include <stdexcept>
#include <vector>

namespace rknpu2_matmul_open::moe {
// Register configuration for one signed-INT8 row copy.
// Source is contiguous; destination can be row-major or native K16/M16.
struct I8Copy { uint64_t source, destination; int channels, destination_rows; };
inline void validate_i8_copy(const I8Copy& c) {
    if(c.channels<32 || c.channels>7680 || c.channels%32 ||
       c.destination_rows<1 || c.destination_rows>128 ||
       ((c.source|c.destination)&15) || c.source>UINT32_MAX || c.destination>UINT32_MAX)
        throw std::runtime_error("INT8 copy outside validated geometry");
}
inline std::vector<uint64_t> copy_i8_body(const I8Copy& c) {
    validate_i8_copy(c);
    using namespace rknpu2_matmul_open::detail;
    const uint32_t channels=uint32_t(c.channels-1), stride=uint32_t(c.destination_rows*16);
    std::vector<uint64_t> v;v.reserve(69);
    auto d=[&](uint32_t a,uint32_t x){v.push_back(::E(T_DPU,a,x));};
    auto r=[&](uint32_t a,uint32_t x){v.push_back(::E(T_RDMA,a,x));};
    d(0x4004,14);r(0x5004,14);d(0x400c,0x1e5);d(0x4010,0);d(0x4014,0);
    d(0x4020,uint32_t(c.destination));d(0x4024,stride);
    d(0x4030,0);d(0x4034,0);d(0x4038,0);d(0x403c,(channels<<16)|channels);
    d(0x4040,0x53);d(0x4044,0);d(0x4048,0);d(0x404c,0);d(0x4050,2);d(0x4054,0);
    d(0x4058,channels);d(0x405c,0);d(0x4060,0x53);d(0x4064,0);d(0x4068,0);d(0x406c,0);
    d(0x4070,0x383);d(0x4074,0);d(0x4078,1);d(0x407c,0);d(0x4080,0);d(0x4084,1);d(0x4088,0);
    for(uint32_t a=0x4090;a<0x40b0;a+=4)d(a,0);
    d(0x40c0,stride);d(0x40c4,0);
    for(uint32_t a=0x4100;a<0x4130;a+=4)d(a,0);
    r(0x500c,0);r(0x5010,0);r(0x5014,channels);r(0x5018,uint32_t(c.source));
    r(0x501c,0);r(0x5020,0);r(0x5028,0);r(0x502c,0);r(0x5034,1);r(0x5038,0);r(0x5040,0);
    r(0x5044,0x7801);r(0x5048,0);r(0x504c,0);r(0x5064,0);r(0x5068,0x01010101);r(0x506c,0);
    return v;
}
}

#pragma once
#include "rk_npu_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rknpu2_matmul_open::moe {
// Port of the validated row-fused DPU recipe in moe_dequant_probe/probe.py
// and activation.py. The GEMM first converts its integer accumulator to FP32.
// One native-N4 FP32 row becomes a compact gate or up row. The LUT uses
// [-8,8] and a 4096x output carrier for finer accuracy than the old 1024x table.
constexpr float middle_lut_scale=4096.f;
constexpr float middle_lut_index_scale=4096.f;
inline uint16_t half_bits(float value) {
#if defined(__aarch64__)
    const __fp16 half=static_cast<__fp16>(value);
#else
    const _Float16 half=static_cast<_Float16>(value);
#endif
    if(!std::isfinite(float(half)) || (value!=0 && half==0))
        throw std::runtime_error("DPU coefficient cannot be represented in FP16");
    uint16_t bits;
    std::memcpy(&bits,&half,sizeof(bits));
    return bits;
}
inline float half_value(uint16_t bits) {
#if defined(__aarch64__)
    __fp16 half;
#else
    _Float16 half;
#endif
    std::memcpy(&half,&bits,sizeof(bits));return float(half);
}
inline std::vector<uint64_t> middle_body(int rows,int channels,uint64_t input,
                                         uint64_t coeff,uint64_t output,float scale,bool gate) {
    if(rows<1 || rows>128 || channels!=512 || (input|coeff|output)>UINT32_MAX ||
       ((input|coeff|output)&15))throw std::runtime_error("DPU middle geometry/address");
    const uint32_t c=uint32_t(channels-1);
    const uint32_t hs=uint32_t(half_bits(scale))<<16;
    std::vector<uint64_t> v;v.reserve(96);
    auto d=[&](uint32_t reg,uint32_t value){v.push_back(::E(T_DPU,reg,value));};
    auto r=[&](uint32_t reg,uint32_t value){v.push_back(::E(T_RDMA,reg,value));};
    d(0x4004,14);d(0x400c,0x1e5);d(0x4010,(5u<<29)|(5u<<26)|5);d(0x4014,0);
    d(0x4020,uint32_t(output));d(0x4024,16);d(0x4030,0);d(0x4034,0);d(0x4038,0);
    d(0x403c,(c<<16)|c);d(0x4040,0x42);d(0x4044,0);d(0x4048,hs);d(0x404c,0);
    d(0x4050,2);d(0x4054,0);d(0x4058,c);d(0x405c,0);
    d(0x4060,gate?0x42:0x53);d(0x4064,0);d(0x4068,gate?1:0);d(0x406c,0);
    d(0x4070,gate?0x302:0x10c003c4);d(0x4074,0);d(0x4078,1);d(0x407c,0);
    d(0x4080,0);d(0x4084,1);d(0x4088,0);
    d(0x40c0,16);d(0x40c4,0);
    for(uint32_t reg=0x4090;reg<0x40b0;reg+=4)d(reg,0);
    r(0x5004,14);r(0x500c,0);r(0x5010,0);r(0x5014,c);
    r(0x5018,uint32_t(input));r(0x501c,0);r(0x5020,0);
    r(0x5028,gate?8:0);r(0x502c,gate?uint32_t(coeff):0);
    r(0x5034,gate?1:0x4000000c);r(0x5038,gate?0:uint32_t(coeff));r(0x5040,16);
    r(0x5044,(5u<<15)|(15u<<11)|(5u<<5)|1);r(0x5048,0);
    r(0x504c,uint32_t(rows-1)*16);r(0x5064,0);r(0x5068,0x01010101);r(0x506c,0);
    if(gate) {
        constexpr uint32_t fields[][2]={{0x4108,0x68},{0x410c,0x60600},{0x4110,0xffff8000},
            {0x4114,0},{0x4118,0},{0x411c,32768},{0x4120,0},{0x4124,0},
            {0x4128,8192u<<16},{0x412c,13u<<5}};
        for(const auto& field:fields)d(field[0],field[1]);
    }
    return v;
}
inline std::vector<uint64_t> middle_lut_load(uint64_t input,uint64_t output) {
    if((input|output)>UINT32_MAX || ((input|output)&15))
        throw std::runtime_error("LUT load address");
    std::vector<uint64_t> v;v.reserve(1028);
    for(int table=0;table<2;++table) {
        v.push_back(::E(T_DPU,0x4100,uint32_t((2|table)<<16)));
        for(int i=0;i<513;++i) {
            const double x=double(i+(table==0?-512:0))/64;
            const double y=x/(1+std::exp(-x));
            const int code=std::clamp(int(std::nearbyint(y*middle_lut_scale)),-32768,32767);
            v.push_back(::E(T_DPU,0x4104,uint32_t(code)));
        }
    }
    // Same eight-element FP32 SiLU command as activation.py:silu_regs(...,
    // load=True). The table writes alone do not constitute a DPU task.
    auto d=[&](uint32_t reg,uint32_t value){v.push_back(::E(T_DPU,reg,value));};
    auto r=[&](uint32_t reg,uint32_t value){v.push_back(::E(T_RDMA,reg,value));};
    d(0x4004,14);d(0x400c,0x1e5);d(0x4010,(5u<<29)|(5u<<26)|5);d(0x4014,0);
    d(0x4020,uint32_t(output));d(0x4024,32);d(0x4030,1);d(0x4034,0);d(0x4038,0);
    d(0x403c,0x30003);d(0x4040,0x53);d(0x4044,0);d(0x4048,0);d(0x404c,0);
    d(0x4050,2);d(0x4054,0);d(0x4058,3);d(0x405c,1);
    d(0x4060,0x20040);d(0x4064,0x80000000);d(0x4068,0x6c000000);d(0x406c,0);
    d(0x4070,0x302);d(0x4074,0);d(0x4078,1);d(0x407c,0);d(0x4080,0);d(0x4084,1);
    d(0x4088,10<<12);
    d(0x40c0,32);d(0x40c4,0);
    for(uint32_t reg=0x4090;reg<0x40b0;reg+=4)d(reg,0);
    constexpr uint32_t fields[][2]={{0x4108,0x68},{0x410c,0x60600},{0x4110,0xffff8000},
        {0x4114,0},{0x4118,0},{0x411c,32768},{0x4120,0},{0x4124,0},
        {0x4128,8192u<<16},{0x412c,13u<<5}};
    r(0x5004,14);r(0x500c,1);r(0x5010,0);r(0x5014,3);r(0x5018,uint32_t(input));
    r(0x501c,0);r(0x5020,0);r(0x5028,0);r(0x502c,0);r(0x5034,1);r(0x5038,0);
    r(0x5040,32);r(0x5044,(5u<<15)|(15u<<11)|(5u<<5)|1);r(0x5048,0);
    r(0x504c,0);r(0x5064,0);r(0x5068,0x01010101);r(0x506c,0);
    for(const auto& field:fields)d(field[0],field[1]);
    return v;
}
}

#include "rk_npu_matmul.h"
#include "rk_npu_matmul_f16.h"
#include "../src/rk_npu_cpu_kernels.h"
#include "../src/rk_npu_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int g_repeats = 3;
int g_stream_mib = 0;

int ceil_div(int x, int y) { return (x + y - 1) / y; }
int align_up(int x, int a) { return ceil_div(x, a) * a; }

struct I8Layout {
    int align_in;
    int align_out;
};

I8Layout i8_layout(int N, int K) {
    return {std::max(32, align_up(K, 32)), std::max(32, align_up(N, 32))};
}

struct F16Layout {
    int align_in;
    int align_out;
};

F16Layout f16_layout(int N, int K) {
    int aligned_k = std::max(32, align_up(K, 32));
    int align_out = std::max(32, align_up(N, 32));
    return {aligned_k, align_out};
}

template <typename T>
rk_npu_mem mem_view(std::vector<T>& v) {
    rk_npu_mem m{};
    m.vaddr = v.data();
    m.size = (uint64_t)v.size() * sizeof(T);
    return m;
}

void fill_bytes(void* p, size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    uint8_t* b = (uint8_t*)p;
    for (size_t i = 0; i < bytes; ++i) b[i] = (uint8_t)d(rng);
}

uint16_t ref_float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) {
        if (man == 0) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7c00u | (man >> 13) | 1u);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_man = man >> shift;
        const uint32_t round_bit = 1u << (shift - 1);
        if ((man & round_bit) && ((man & (round_bit - 1)) || (half_man & 1u)))
            ++half_man;
        return (uint16_t)(sign | half_man);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1000u) && ((man & 0x0fffu) || (half & 1u)))
        ++half;
    return (uint16_t)half;
}

float ref_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void fill_f16_values(std::vector<uint16_t>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (uint16_t& x : v) x = ref_float_to_half(dist(rng));
}

template <typename T>
bool same_vec(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s size mismatch %zu != %zu\n", label, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s mismatch at %zu: got=%lld ref=%lld\n",
                         label, i, (long long)a[i], (long long)b[i]);
            return false;
        }
    }
    return true;
}

bool close_vec_f32(const std::vector<float>& a, const std::vector<float>& b, const char* label, float tol = 1e-6f) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s size mismatch %zu != %zu\n", label, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const float da = a[i], db = b[i];
        const float diff = std::fabs(da - db);
        const float scale = std::max(1.0f, std::max(std::fabs(da), std::fabs(db)));
        if (!(diff <= tol * scale)) {
            std::fprintf(stderr, "%s mismatch at %zu: got=%g ref=%g diff=%g\n",
                         label, i, da, db, diff);
            return false;
        }
    }
    return true;
}

void ref_i8_pack_a_normal(int M, int K, int align_in, const int8_t* A, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * align_in, A + (size_t)r * K, (size_t)K);
}

void ref_i8_pack_a_native(int M, int K, int align_in, const int8_t* A, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int kb = 0; kb < align_in / 16; ++kb)
        for (int r = 0; r < M; ++r)
            for (int kk = 0; kk < 16; ++kk) {
                int k = kb * 16 + kk;
                dst[(size_t)kb * M * 16 + (size_t)r * 16 + kk] =
                    (k < K) ? A[(size_t)r * K + k] : 0;
            }
}

int8_t ref_quant_s8(float x, float scale) {
    if (!(scale > 0.0f)) return 0;
    float q = std::round(x * (1.0f / scale));
    q = std::max(-127.0f, std::min(127.0f, q));
    return (int8_t)q;
}

void ref_i8_pack_a_f16_quant_native(int M, int K, int align_in, const uint16_t* A,
                                    rknpu2_matmul_open::cpu::QuantMode mode, const float* static_scale,
                                    float* scales, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int r = 0; r < M; ++r) {
        float scale = 1.0f;
        if (mode == rknpu2_matmul_open::cpu::QuantMode::DynamicPerToken) {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::fabs(ref_half_to_float(A[(size_t)r * K + k])));
            scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        } else if (mode == rknpu2_matmul_open::cpu::QuantMode::StaticPerToken) {
            scale = static_scale ? static_scale[r] : 1.0f;
        } else {
            scale = static_scale ? static_scale[0] : 1.0f;
        }
        if (scales) scales[r] = scale;
        for (int kb = 0; kb < align_in / 16; ++kb)
            for (int kk = 0; kk < 16; ++kk) {
                const int k = kb * 16 + kk;
                if (k < K) {
                    const float v = ref_half_to_float(A[(size_t)r * K + k]);
                    dst[(size_t)kb * M * 16 + (size_t)r * 16 + kk] = ref_quant_s8(v, scale);
                }
            }
    }
}

void ref_i8_pack_a_f16_quant_native_strided(
    int M, int K, int align_in, const uint16_t* A, int row_stride,
    const float* scales, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int kb = 0; kb < align_in / 16; ++kb)
        for (int r = 0; r < M; ++r)
            for (int kk = 0; kk < 16; ++kk) {
                const int k = kb * 16 + kk;
                if (k < K) {
                    dst[((size_t)kb * M + r) * 16 + kk] = ref_quant_s8(
                        ref_half_to_float(A[(size_t)r * row_stride + k]),
                        scales[r]);
                }
            }
}

void ref_i8_pack_a_f32_quant_native_strided(
    int M, int K, int align_in, const float* A, int row_stride,
    const float* scales, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int kb = 0; kb < align_in / 16; ++kb)
        for (int r = 0; r < M; ++r)
            for (int kk = 0; kk < 16; ++kk) {
                const int k = kb * 16 + kk;
                if (k < K)
                    dst[((size_t)kb * M + r) * 16 + kk] = ref_quant_s8(
                        A[(size_t)r * row_stride + k], scales[r]);
            }
}

void ref_i8_dynamic_scale_f16(int M, int K, const uint16_t* A,
                              int row_stride, float* scales) {
    for (int r = 0; r < M; ++r) {
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k) {
            max_abs = std::max(
                max_abs, std::fabs(ref_half_to_float(A[(size_t)r * row_stride + k])));
        }
        scales[r] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
}

void ref_i8_pack_a_f16_quant_normal_strided(
    int M, int K, int align_in, const uint16_t* A, int row_stride,
    const float* scales, int8_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in);
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            dst[(size_t)r * align_in + k] = ref_quant_s8(
                ref_half_to_float(A[(size_t)r * row_stride + k]), scales[r]);
        }
    }
}

void ref_i8_pack_b(int K, int N, int align_in, int align_out, const int8_t* B, int8_t* dst) {
    std::memset(dst, 0, (size_t)align_out * align_in);
    for (int o = 0; o < align_out; ++o) {
        int a = o / 32, b = (o % 32) / 16, c = o % 16;
        for (int i = 0; i < align_in; ++i) {
            int d = i / 32, e = i % 32;
            size_t pf = (size_t)a * (align_in * 32) + (size_t)d * 1024
                      + (size_t)b * 512 + (size_t)c * 32 + e;
            dst[pf] = (o < N && i < K) ? B[(size_t)i * N + o] : 0;
        }
    }
}

void ref_i8_unpack_c_native(int M, int N, const uint32_t* src, uint32_t* dst) {
    for (int r = 0; r < M; ++r)
        for (int n = 0; n < N; ++n)
            dst[(size_t)r * N + n] = src[((size_t)(n / 4) * M + r) * 4 + (n % 4)];
}

float ref_activation(float x, rknpu2_matmul_open::cpu::ActivationOp act) {
    switch (act) {
    case rknpu2_matmul_open::cpu::ActivationOp::None:
        return x;
    case rknpu2_matmul_open::cpu::ActivationOp::Relu:
        return x > 0.0f ? x : 0.0f;
    case rknpu2_matmul_open::cpu::ActivationOp::Silu:
        return x / (1.0f + std::exp(-x));
    case rknpu2_matmul_open::cpu::ActivationOp::GeluTanh: {
        constexpr float kAlpha = 0.7978845608028654f;
        constexpr float kBeta = 0.044715f;
        const float x3 = x * x * x;
        return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x3)));
    }
    }
    return x;
}

void ref_i8_dequant_c_native_f32(int M, int N, const int32_t* src, const float* a_scale,
                                 const float* w_scale, const float* bias,
                                 rknpu2_matmul_open::cpu::ActivationOp act, float* dst) {
    for (int r = 0; r < M; ++r) {
        const float as = a_scale ? a_scale[r] : 1.0f;
        for (int n = 0; n < N; ++n) {
            const int32_t acc = src[((size_t)(n / 4) * M + r) * 4 + (n % 4)];
            float y = (float)acc * as * (w_scale ? w_scale[n] : 1.0f);
            if (bias) y += bias[n];
            dst[(size_t)r * N + n] = ref_activation(y, act);
        }
    }
}

void ref_i8_dequant_c_native_f16(int M, int N, const int32_t* src, const float* a_scale,
                                 const float* w_scale, const float* bias,
                                 rknpu2_matmul_open::cpu::ActivationOp act, uint16_t* dst) {
    for (int r = 0; r < M; ++r) {
        const float as = a_scale ? a_scale[r] : 1.0f;
        for (int n = 0; n < N; ++n) {
            const int32_t acc = src[((size_t)(n / 4) * M + r) * 4 + (n % 4)];
            float y = (float)acc * as * (w_scale ? w_scale[n] : 1.0f);
            if (bias) y += bias[n];
            dst[(size_t)r * N + n] = ref_float_to_half(ref_activation(y, act));
        }
    }
}

void ref_i8_reduce_c_native_i32(int M, int N, int partial_count,
                                const int32_t* const* src, int32_t* dst) {
    const size_t elems = (size_t)((N + 3) / 4) * M * 4;
    for (size_t i = 0; i < elems; ++i) {
        uint32_t acc = 0;
        for (int p = 0; p < partial_count; ++p) acc += (uint32_t)src[p][i];
        std::memcpy(dst + i, &acc, sizeof(acc));
    }
}

void quant_f16_to_i8_rowmajor_dynamic(int M, int K, const uint16_t* src, int8_t* dst, float* scales) {
    for (int r = 0; r < M; ++r) {
        const uint16_t* row = src + (size_t)r * K;
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::fabs(ref_half_to_float(row[k])));
        const float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        if (scales) scales[r] = scale;
        for (int k = 0; k < K; ++k)
            dst[(size_t)r * K + k] = ref_quant_s8(ref_half_to_float(row[k]), scale);
    }
}

void dequant_rowmajor_i32_to_f16(int M, int N, const int32_t* src, const float* a_scale,
                                 const float* w_scale, const float* bias,
                                 rknpu2_matmul_open::cpu::ActivationOp act, uint16_t* dst) {
#pragma omp parallel for schedule(static) if((uint64_t)M * N >= (1u << 15))
    for (int r = 0; r < M; ++r) {
        const float as = a_scale ? a_scale[r] : 1.0f;
        for (int n = 0; n < N; ++n) {
            float y = (float)src[(size_t)r * N + n] * as * (w_scale ? w_scale[n] : 1.0f);
            if (bias) y += bias[n];
            dst[(size_t)r * N + n] = ref_float_to_half(ref_activation(y, act));
        }
    }
}

void ref_f16_pack_a(int M, int K, int align_in, const uint16_t* A, uint16_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in * sizeof(uint16_t));
    for (int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * align_in, A + (size_t)r * K, (size_t)K * sizeof(uint16_t));
}

void ref_f16_pack_a_native(int M, int K, int align_in,
                            const uint16_t* A, uint16_t* dst) {
    std::memset(dst, 0, (size_t)M * align_in * sizeof(uint16_t));
    for (int k = 0; k < K; ++k)
        for (int r = 0; r < M; ++r)
            dst[((size_t)(k / 8) * M + r) * 8 + (k % 8)] =
                A[(size_t)r * K + k];
}

void ref_f16_pack_b(int K, int N, int align_in, int align_out, const uint16_t* B, uint16_t* dst) {
    std::memset(dst, 0, (size_t)align_out * align_in * sizeof(uint16_t));
    for (int o = 0; o < align_out; ++o) {
        int g = o / 16, j = o % 16;
        for (int i = 0; i < align_in; ++i) {
            int d = i / 32, e = i % 32;
            size_t pf = (size_t)g * (align_in * 16) + (size_t)d * (16 * 32) + (size_t)j * 32 + e;
            dst[pf] = (o < N && i < K) ? B[(size_t)i * N + o] : 0;
        }
    }
}

int operand_block_elems(int align_out, int tile_m) {
    return (align_out / 16) * 24 * tile_m;
}

void ref_f16_pack_operand(int M, int N, int align_out, int m_tile, const uint16_t* C0, uint16_t* dst) {
    uint64_t total = 0;
    for (int start = 0; start < M; start += m_tile)
        total += (uint64_t)operand_block_elems(align_out, std::min(m_tile, M - start));
    std::memset(dst, 0, (size_t)total * sizeof(uint16_t));

    uint64_t block_off = 0;
    for (int start = 0; start < M; start += m_tile) {
        int tile_m = std::min(m_tile, M - start);
        for (int r = 0; r < tile_m; ++r)
            for (int c = 0; c < N; ++c) {
                int b = c >> 3;
                uint64_t idx = (uint64_t)((b + 1) / 2) * (16 * tile_m)
                             + (uint64_t)(b / 2) * (8 * tile_m)
                             + (uint64_t)r * 8 + (c & 7);
                dst[block_off + idx] = C0[(size_t)(start + r) * N + c];
            }
        block_off += operand_block_elems(align_out, tile_m);
    }
}

void ref_f16_unpack_d(int M, int N, int align_out, const uint16_t* src, uint16_t* dst) {
    int row_stride = align_out * 2;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            dst[(size_t)r * N + c] = src[(size_t)r * row_stride + (c / 16) * 32 + (c % 16)];
}

void ref_f16_unpack_d_compact(int M, int N, int align_out,
                              const uint16_t* src, uint16_t* dst) {
    for (int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * N,
                    src + (size_t)r * align_out,
                    (size_t)N * sizeof(uint16_t));
}

void ref_f16_unpack_d_native(int M, int N,
                             const uint16_t* src, uint16_t* dst) {
    for (int r = 0; r < M; ++r)
        for (int n = 0; n < N; ++n)
            dst[(size_t)r * N + n] =
                src[((size_t)(n / 8) * M + r) * 8 + (n % 8)];
}

int f16_m_tile(int align_in) {
    constexpr int cbuf_banks = 12;
    constexpr int cbuf_bank_size = 256 * 128;
    int input_row_bytes = align_in * 2;
    int weight_banks = ceil_div(input_row_bytes * 32, cbuf_bank_size);
    weight_banks = std::min(std::max(weight_banks, 1), cbuf_banks - 1);
    return std::max(1, (cbuf_banks - weight_banks) *
                       cbuf_bank_size / input_row_bytes);
}

bool run_correctness() {
    bool ok = true;
    const int shapes[][3] = {
        {1, 16, 4}, {3, 31, 31}, {16, 257, 97}, {128, 1024, 513},
        {1, 17, 4099}, {33, 33, 4099}
    };
    for (auto& s : shapes) {
        const int M = s[0], K = s[1], N = s[2];
        I8Layout li8 = i8_layout(N, K);
        F16Layout lf16 = f16_layout(N, K);

        std::vector<int8_t> A8((size_t)M * K), B8((size_t)K * N);
        fill_bytes(A8.data(), A8.size(), 100 + M + K + N);
        fill_bytes(B8.data(), B8.size(), 200 + M + K + N);

        rk_npu_matmul_i8_config ci8{};
        rk_npu_matmul_i8_config_init(&ci8, M, N, K);
        rk_npu_matmul_sizes si8{};
        rknpu2_matmul_open::detail::query_i8(&ci8, &si8);

        std::vector<int8_t> got_a((size_t)M * li8.align_in), ref_a(got_a.size());
        rk_npu_mem mgot_a = mem_view(got_a);
        rknpu2_matmul_open::detail::pack_i8_a(&ci8, A8.data(), &mgot_a);
        ref_i8_pack_a_normal(M, K, li8.align_in, A8.data(), ref_a.data());
        ok &= same_vec(got_a, ref_a, "i8 pack A normal");

        ci8.a_layout = RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16;
        got_a.assign(got_a.size(), 0x55);
        ref_a.assign(ref_a.size(), 0x44);
        rknpu2_matmul_open::detail::pack_i8_a(&ci8, A8.data(), &mgot_a);
        ref_i8_pack_a_native(M, K, li8.align_in, A8.data(), ref_a.data());
        ok &= same_vec(got_a, ref_a, "i8 pack A native");

        std::vector<int8_t> got_b((size_t)li8.align_out * li8.align_in), ref_b(got_b.size());
        rk_npu_mem mgot_b = mem_view(got_b);
        rknpu2_matmul_open::detail::pack_i8_b(&ci8, B8.data(), &mgot_b);
        ref_i8_pack_b(K, N, li8.align_in, li8.align_out, B8.data(), ref_b.data());
        ok &= same_vec(got_b, ref_b, "i8 pack B native");

        std::vector<uint32_t> packed_c((size_t)M * li8.align_out), got_c((size_t)M * N), ref_c(got_c.size());
        for (size_t i = 0; i < packed_c.size(); ++i) packed_c[i] = (uint32_t)(0x12000000u + i);
        rk_npu_mem mc = mem_view(packed_c);
        ci8.c_layout = RK_NPU_I8_C_LAYOUT_NORMAL_PADDED;
        rknpu2_matmul_open::detail::unpack_i8_c(&ci8, &mc, got_c.data());
        for (int r = 0; r < M; ++r)
            std::memcpy(ref_c.data() + (size_t)r * N, packed_c.data() + (size_t)r * li8.align_out,
                        (size_t)N * sizeof(uint32_t));
        ok &= same_vec(got_c, ref_c, "i8 unpack C normal");

        std::vector<uint32_t> packed_cn((size_t)((N + 3) / 4) * M * 4);
        for (size_t i = 0; i < packed_cn.size(); ++i) packed_cn[i] = (uint32_t)(0x34000000u + i);
        rk_npu_mem mcn = mem_view(packed_cn);
        got_c.assign(got_c.size(), 0);
        ref_c.assign(ref_c.size(), 0);
        ci8.c_layout = RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;
        rknpu2_matmul_open::detail::unpack_i8_c(&ci8, &mcn, got_c.data());
        ref_i8_unpack_c_native(M, N, packed_cn.data(), ref_c.data());
        ok &= same_vec(got_c, ref_c, "i8 unpack C native");

        std::vector<int32_t> packed_i32((size_t)((N + 3) / 4) * M * 4);
        for (size_t i = 0; i < packed_i32.size(); ++i)
            packed_i32[i] = (int32_t)((i * 17u) % 2001u) - 1000;
        std::vector<float> a_scale((size_t)M), w_scale((size_t)N), bias((size_t)N);
        for (int r = 0; r < M; ++r) a_scale[r] = 0.001f + 0.000013f * (float)(r % 17);
        for (int n = 0; n < N; ++n) {
            w_scale[n] = 0.002f + 0.000007f * (float)(n % 23);
            bias[n] = ((n % 11) - 5) * 0.01f;
        }
        const rknpu2_matmul_open::cpu::ActivationOp acts[] = {
            rknpu2_matmul_open::cpu::ActivationOp::None,
            rknpu2_matmul_open::cpu::ActivationOp::Relu,
            rknpu2_matmul_open::cpu::ActivationOp::Silu,
            rknpu2_matmul_open::cpu::ActivationOp::GeluTanh,
        };
        for (rknpu2_matmul_open::cpu::ActivationOp act : acts) {
            std::vector<float> got_df32((size_t)M * N), ref_df32(got_df32.size());
            rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f32(M, N, packed_i32.data(), a_scale.data(),
                                                      w_scale.data(), bias.data(), act, got_df32.data());
            ref_i8_dequant_c_native_f32(M, N, packed_i32.data(), a_scale.data(),
                                        w_scale.data(), bias.data(), act, ref_df32.data());
            ok &= close_vec_f32(got_df32, ref_df32, "i8 fused C dequant f32");

            std::vector<uint16_t> got_df16((size_t)M * N), ref_df16(got_df16.size());
            rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f16(M, N, packed_i32.data(), a_scale.data(),
                                                      w_scale.data(), bias.data(), act, got_df16.data());
            ref_i8_dequant_c_native_f16(M, N, packed_i32.data(), a_scale.data(),
                                        w_scale.data(), bias.data(), act, ref_df16.data());
            ok &= same_vec(got_df16, ref_df16, "i8 fused C dequant f16");
        }

        {
            std::vector<float> got_df32((size_t)M * N), ref_df32(got_df32.size());
            rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f32(
                M, N, packed_i32.data(), a_scale.data(), w_scale.data(),
                nullptr, rknpu2_matmul_open::cpu::ActivationOp::None, got_df32.data());
            ref_i8_dequant_c_native_f32(
                M, N, packed_i32.data(), a_scale.data(), w_scale.data(),
                nullptr, rknpu2_matmul_open::cpu::ActivationOp::None, ref_df32.data());
            ok &= close_vec_f32(
                got_df32, ref_df32, "i8 fused C dequant f32 fast path");
        }


        std::vector<int32_t> partial1(packed_i32.size()), partial2(packed_i32.size()),
                             partial3(packed_i32.size()), partial4(packed_i32.size()),
                             reduced_ref(packed_i32.size()),
                             reduced_accum;
        for (size_t i = 0; i < packed_i32.size(); ++i) {
            partial1[i] = (int32_t)((i * 29u) % 3001u) - 1500;
            partial2[i] = (int32_t)((i * 43u) % 4001u) - 2000;
            partial3[i] = (int32_t)((i * 61u) % 5001u) - 2500;
            partial4[i] = (int32_t)((i * 73u) % 6001u) - 3000;
        }
        const int32_t* partials1[] = {packed_i32.data()};
        const int32_t* partials2[] = {packed_i32.data(), partial1.data()};
        const int32_t* partials4[] = {
            packed_i32.data(), partial1.data(), partial2.data(), partial3.data()
        };
        const int32_t* partials5[] = {
            packed_i32.data(), partial1.data(), partial2.data(), partial3.data(),
            partial4.data()
        };

        ref_i8_reduce_c_native_i32(M, N, 2, partials2, reduced_ref.data());
        reduced_accum = packed_i32;
        rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(M, N, partial1.data(), reduced_accum.data());
        ok &= same_vec(reduced_accum, reduced_ref, "i8 native C accumulate i32");

        for (int partial_count : {1, 2, 4, 5}) {
            const int32_t* const* partials = partial_count == 1 ? partials1
                                             : partial_count == 2 ? partials2
                                             : partial_count == 4 ? partials4
                                                                  : partials5;
            ref_i8_reduce_c_native_i32(M, N, partial_count, partials, reduced_ref.data());
            std::vector<int32_t> got_i32((size_t)M * N), ref_i32(got_i32.size());
            rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_to_rowmajor(
                M, N, partial_count, partials, got_i32.data());
            ref_i8_unpack_c_native(
                M, N, reinterpret_cast<const uint32_t*>(reduced_ref.data()),
                reinterpret_cast<uint32_t*>(ref_i32.data()));
            ok &= same_vec(got_i32, ref_i32,
                           "i8 split-K compact i32 reduce");
            for (rknpu2_matmul_open::cpu::ActivationOp act : acts) {
                std::vector<float> got_df32((size_t)M * N), ref_df32(got_df32.size());
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(
                    M, N, partial_count, partials, a_scale.data(), w_scale.data(),
                    bias.data(), act, got_df32.data());
                ref_i8_dequant_c_native_f32(M, N, reduced_ref.data(), a_scale.data(),
                                            w_scale.data(), bias.data(), act, ref_df32.data());
                ok &= close_vec_f32(got_df32, ref_df32, "i8 split-K fused C reduce f32");

                std::vector<uint16_t> got_df16((size_t)M * N), ref_df16(got_df16.size());
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
                    M, N, partial_count, partials, a_scale.data(), w_scale.data(),
                    bias.data(), act, got_df16.data());
                ref_i8_dequant_c_native_f16(M, N, reduced_ref.data(), a_scale.data(),
                                            w_scale.data(), bias.data(), act, ref_df16.data());
                ok &= same_vec(got_df16, ref_df16, "i8 split-K fused C reduce f16");
            }

            std::vector<uint16_t> got_fast((size_t)M * N), ref_fast(got_fast.size());
            rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
                M, N, partial_count, partials, a_scale.data(), w_scale.data(),
                nullptr, rknpu2_matmul_open::cpu::ActivationOp::None, got_fast.data());
            ref_i8_dequant_c_native_f16(
                M, N, reduced_ref.data(), a_scale.data(), w_scale.data(),
                nullptr, rknpu2_matmul_open::cpu::ActivationOp::None, ref_fast.data());
            ok &= same_vec(got_fast, ref_fast,
                           "i8 split-K no-bias C reduce f16");
        }

        std::vector<uint16_t> A16((size_t)M * K), B16((size_t)K * N);
        fill_f16_values(A16, 300 + M + K + N);
        fill_f16_values(B16, 400 + M + K + N);
        std::vector<float> A32(A16.size());
        for (size_t i = 0; i < A16.size(); ++i)
            A32[i] = ref_half_to_float(A16[i]);

        rk_npu_matmul_f16_config cf16{};
        rk_npu_matmul_f16_config_init(&cf16, M, N, K, RK_NPU_FUSE_NONE);

        std::vector<uint16_t> got_fa((size_t)M * lf16.align_in), ref_fa(got_fa.size());
        rk_npu_mem mfa = mem_view(got_fa);
        rk_npu_matmul_f16_pack_a(&cf16, A16.data(), &mfa);
        ref_f16_pack_a(M, K, lf16.align_in, A16.data(), ref_fa.data());
        ok &= same_vec(got_fa, ref_fa, "f16 pack A");

        if (K >= 3 * 32) {
            constexpr int split_count = 3;
            const int part_k = align_up(ceil_div(K, split_count), 32);
            rk_npu_matmul_sizes split_sizes{};
            ok &= rk_npu_matmul_f16_splitk_query(
                      split_count, &cf16, &split_sizes) == RK_NPU_OK;

            std::vector<uint16_t> got_split_a(
                split_sizes.input_bytes / sizeof(uint16_t), 0x55);
            std::vector<uint16_t> ref_split_a(got_split_a.size(), 0);
            rk_npu_mem split_a_mem = mem_view(got_split_a);
            ok &= rk_npu_matmul_f16_splitk_pack_a(
                      split_count, &cf16, A16.data(), &split_a_mem) == RK_NPU_OK;
            for (int split = 0; split < split_count; ++split) {
                const int k0 = split * part_k;
                const int valid = std::min(part_k, std::max(0, K - k0));
                for (int r = 0; r < M; ++r)
                    if (valid > 0)
                        std::memcpy(
                            ref_split_a.data() +
                                ((size_t)split * M + r) * part_k,
                            A16.data() + (size_t)r * K + k0,
                            (size_t)valid * sizeof(uint16_t));
            }
            ok &= same_vec(got_split_a, ref_split_a,
                           "f16 split-K pack A");

            const size_t partial_stride =
                split_sizes.output_bytes / split_count / sizeof(uint16_t);
            std::vector<uint16_t> partials(
                split_sizes.output_bytes / sizeof(uint16_t));
            fill_f16_values(partials, 350 + M + K + N);
            rk_npu_mem partial_mem = mem_view(partials);
            std::vector<uint16_t> got_reduce_f16((size_t)M * N);
            std::vector<uint16_t> ref_reduce_f16(got_reduce_f16.size());
            std::vector<uint16_t> got_reduce_bias((size_t)M * N);
            std::vector<uint16_t> ref_reduce_bias(got_reduce_bias.size());
            std::vector<float> got_reduce_f32((size_t)M * N);
            std::vector<float> ref_reduce_f32(got_reduce_f32.size());
            std::vector<uint16_t> split_bias((size_t)N);
            fill_f16_values(split_bias, 375 + M + K + N);
            ok &= rk_npu_matmul_f16_splitk_unpack_d(
                      split_count, &cf16, &partial_mem,
                      got_reduce_f16.data()) == RK_NPU_OK;
            ok &= rk_npu_matmul_f16_splitk_unpack_d_f32(
                      split_count, &cf16, &partial_mem,
                      got_reduce_f32.data()) == RK_NPU_OK;
            ok &= rk_npu_matmul_f16_splitk_unpack_d_add_bias(
                      split_count, &cf16, &partial_mem, split_bias.data(),
                      got_reduce_bias.data()) == RK_NPU_OK;
            for (int r = 0; r < M; ++r) {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int split = 0; split < split_count; ++split)
                        sum += ref_half_to_float(
                            partials[(size_t)split * partial_stride +
                                     (size_t)r * lf16.align_out + n]);
                    ref_reduce_f32[(size_t)r * N + n] = sum;
                    ref_reduce_f16[(size_t)r * N + n] =
                        ref_float_to_half(sum);
                    ref_reduce_bias[(size_t)r * N + n] =
                        ref_float_to_half(
                            sum + ref_half_to_float(split_bias[n]));
                }
            }
            ok &= same_vec(got_reduce_f16, ref_reduce_f16,
                           "f16 split-K compact reduce f16");
            ok &= close_vec_f32(got_reduce_f32, ref_reduce_f32,
                                "f16 split-K compact reduce f32");
            ok &= same_vec(got_reduce_bias, ref_reduce_bias,
                           "f16 split-K compact reduce+bias f16");
        }

        cf16.a_layout = RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;
        std::fill(got_fa.begin(), got_fa.end(), 0x55);
        std::fill(ref_fa.begin(), ref_fa.end(), 0x44);
        rk_npu_matmul_f16_pack_a(&cf16, A16.data(), &mfa);
        ref_f16_pack_a_native(M, K, lf16.align_in,
                               A16.data(), ref_fa.data());
        ok &= same_vec(got_fa, ref_fa, "f16 pack A native");
        cf16.a_layout = RK_NPU_F16_A_LAYOUT_NORMAL;

        std::vector<int8_t> got_fqa((size_t)M * li8.align_in), ref_fqa(got_fqa.size());
        std::vector<float> got_qscale((size_t)M), ref_qscale((size_t)M), static_scale((size_t)M);
        for (int r = 0; r < M; ++r) static_scale[r] = 0.02f + 0.0001f * (float)(r % 13);

        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_k16_m16(M, K, li8.align_in, A16.data(),
                                                  rknpu2_matmul_open::cpu::QuantMode::DynamicPerToken,
                                                  nullptr, got_qscale.data(), got_fqa.data());
        ref_i8_pack_a_f16_quant_native(M, K, li8.align_in, A16.data(),
                                       rknpu2_matmul_open::cpu::QuantMode::DynamicPerToken,
                                       nullptr, ref_qscale.data(), ref_fqa.data());
        ok &= same_vec(got_fqa, ref_fqa, "i8 fused A dynamic quant native");
        ok &= close_vec_f32(got_qscale, ref_qscale, "i8 fused A dynamic scales");

        std::vector<float> got_full_scale((size_t)M), ref_full_scale((size_t)M);
        rknpu2_matmul_open::cpu::i8_compute_dynamic_per_token_scale_f16(
            M, K, A16.data(), K, got_full_scale.data());
        ref_i8_dynamic_scale_f16(M, K, A16.data(), K, ref_full_scale.data());
        ok &= close_vec_f32(got_full_scale, ref_full_scale,
                            "i8 full-K dynamic scales");

        const int slice_k0 = K / 3;
        const int slice_k = K - slice_k0;
        const int slice_align = std::max(32, align_up(slice_k, 32));
        std::vector<int8_t> got_slice((size_t)M * slice_align, 0x55);
        std::vector<int8_t> ref_slice(got_slice.size(), 0x44);
        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_normal_strided(
            M, slice_k, slice_align, A16.data() + slice_k0, K,
            got_full_scale.data(), got_slice.data());
        ref_i8_pack_a_f16_quant_normal_strided(
            M, slice_k, slice_align, A16.data() + slice_k0, K,
            ref_full_scale.data(), ref_slice.data());
        ok &= same_vec(got_slice, ref_slice,
                       "i8 split-K dynamic quant normal strided");

        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_strided(
            M, slice_k, slice_align, A16.data() + slice_k0, K,
            got_full_scale.data(), got_slice.data());
        std::vector<int8_t> native_slice_ref(got_slice.size(), 0x44);
        ref_i8_pack_a_f16_quant_native_strided(
            M, slice_k, slice_align, A16.data() + slice_k0, K,
            ref_full_scale.data(), native_slice_ref.data());
        ok &= same_vec(got_slice, native_slice_ref,
                       "i8 split-K dynamic quant native strided");

        const int split_k0[] = {0, K / 3, (2 * K) / 3};
        const int split_k[] = {
            split_k0[1] - split_k0[0],
            split_k0[2] - split_k0[1],
            K - split_k0[2],
        };
        const int split_align[] = {
            std::max(32, align_up(split_k[0], 32)),
            std::max(32, align_up(split_k[1], 32)),
            std::max(32, align_up(split_k[2], 32)),
        };
        std::vector<int8_t> got_split[3], ref_split[3];
        int8_t* got_split_ptrs[3];
        for (int i = 0; i < 3; ++i) {
            got_split[i].assign((size_t)M * split_align[i], 0x55);
            ref_split[i].assign(got_split[i].size(), 0x44);
            got_split_ptrs[i] = got_split[i].data();
        }
        std::vector<float> got_split_scale((size_t)M), ref_split_scale((size_t)M);
        rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_normal_split(
            M, K, 3, split_k0, split_k, split_align, A16.data(),
            got_split_scale.data(), got_split_ptrs);
        ref_i8_dynamic_scale_f16(M, K, A16.data(), K, ref_split_scale.data());
        ok &= close_vec_f32(got_split_scale, ref_split_scale,
                            "i8 split-K fused dynamic scales");
        for (int i = 0; i < 3; ++i) {
            ref_i8_pack_a_f16_quant_normal_strided(
                M, split_k[i], split_align[i], A16.data() + split_k0[i], K,
                ref_split_scale.data(), ref_split[i].data());
            ok &= same_vec(got_split[i], ref_split[i],
                           "i8 split-K fused dynamic normal");
        }

        rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_native_split(
            M, K, 3, split_k0, split_k, split_align, A16.data(),
            got_split_scale.data(), got_split_ptrs);
        ok &= close_vec_f32(got_split_scale, ref_split_scale,
                            "i8 split-K fused dynamic native scales");
        std::vector<int8_t> ref_native_split[3];
        for (int i = 0; i < 3; ++i) {
            ref_native_split[i].resize(got_split[i].size());
            ref_i8_pack_a_f16_quant_native_strided(
                M, split_k[i], split_align[i], A16.data() + split_k0[i], K,
                ref_split_scale.data(), ref_native_split[i].data());
            ok &= same_vec(got_split[i], ref_native_split[i],
                           "i8 split-K fused dynamic native");
        }

        std::vector<float> got_f32_scale((size_t)M);
        rknpu2_matmul_open::cpu::i8_compute_dynamic_per_token_scale_f32(
            M, K, A32.data(), K, got_f32_scale.data());
        ok &= close_vec_f32(got_f32_scale, ref_full_scale,
                            "i8 f32 full-K dynamic scales");
        rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_normal_strided(
            M, slice_k, slice_align, A32.data() + slice_k0, K,
            got_f32_scale.data(), got_slice.data());
        ok &= same_vec(got_slice, ref_slice,
                       "i8 f32 split-K dynamic quant normal strided");

        rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_native_strided(
            M, slice_k, slice_align, A32.data() + slice_k0, K,
            got_f32_scale.data(), got_slice.data());
        ref_i8_pack_a_f32_quant_native_strided(
            M, slice_k, slice_align, A32.data() + slice_k0, K,
            ref_full_scale.data(), native_slice_ref.data());
        ok &= same_vec(got_slice, native_slice_ref,
                       "i8 f32 split-K dynamic quant native strided");

        rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_normal_split(
            M, K, 3, split_k0, split_k, split_align, A32.data(),
            got_f32_scale.data(), got_split_ptrs);
        ok &= close_vec_f32(got_f32_scale, ref_split_scale,
                            "i8 f32 split-K fused dynamic scales");
        for (int i = 0; i < 3; ++i)
            ok &= same_vec(got_split[i], ref_split[i],
                           "i8 f32 split-K fused dynamic normal");

        rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_native_split(
            M, K, 3, split_k0, split_k, split_align, A32.data(),
            got_f32_scale.data(), got_split_ptrs);
        ok &= close_vec_f32(got_f32_scale, ref_split_scale,
                            "i8 f32 split-K fused dynamic native scales");
        for (int i = 0; i < 3; ++i) {
            ref_i8_pack_a_f32_quant_native_strided(
                M, split_k[i], split_align[i], A32.data() + split_k0[i], K,
                ref_split_scale.data(), ref_native_split[i].data());
            ok &= same_vec(got_split[i], ref_native_split[i],
                           "i8 f32 split-K fused dynamic native");
        }

        for (int i = 0; i < 3; ++i) {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_normal_strided(
                M, split_k[i], split_align[i], A32.data() + split_k0[i], K,
                static_scale.data(), got_split[i].data());
            ref_i8_pack_a_f16_quant_normal_strided(
                M, split_k[i], split_align[i], A16.data() + split_k0[i], K,
                static_scale.data(), ref_split[i].data());
            ok &= same_vec(got_split[i], ref_split[i],
                           "i8 f32 split-K static-token normal");
        }

        for (int i = 0; i < 3; ++i) {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_native_strided(
                M, split_k[i], split_align[i], A32.data() + split_k0[i], K,
                static_scale.data(), got_split[i].data());
            ref_i8_pack_a_f32_quant_native_strided(
                M, split_k[i], split_align[i], A32.data() + split_k0[i], K,
                static_scale.data(), ref_native_split[i].data());
            ok &= same_vec(got_split[i], ref_native_split[i],
                           "i8 f32 split-K static-token native");
        }

        rknpu2_matmul_open::cpu::i8_pack_a_i8_normal_split(
            M, K, 3, split_k0, split_k, split_align, A8.data(),
            got_split_ptrs);
        for (int i = 0; i < 3; ++i) {
            std::fill(ref_split[i].begin(), ref_split[i].end(), 0);
            for (int r = 0; r < M; ++r) {
                std::memcpy(ref_split[i].data() + (size_t)r * split_align[i],
                            A8.data() + (size_t)r * K + split_k0[i],
                            (size_t)split_k[i]);
            }
            ok &= same_vec(got_split[i], ref_split[i],
                           "i8 split-K fused i8 normal");
        }

        rknpu2_matmul_open::cpu::i8_pack_a_f16_static_normal_split(
            M, K, 3, split_k0, split_k, split_align, A16.data(),
            static_scale.data(), got_split_ptrs);
        for (int i = 0; i < 3; ++i) {
            ref_i8_pack_a_f16_quant_normal_strided(
                M, split_k[i], split_align[i], A16.data() + split_k0[i], K,
                static_scale.data(), ref_split[i].data());
            ok &= same_vec(got_split[i], ref_split[i],
                           "i8 split-K fused static-token normal");
        }

        got_fqa.assign(got_fqa.size(), 0x55);
        ref_fqa.assign(ref_fqa.size(), 0x44);
        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_k16_m16(M, K, li8.align_in, A16.data(),
                                                  rknpu2_matmul_open::cpu::QuantMode::StaticPerToken,
                                                  static_scale.data(), got_qscale.data(), got_fqa.data());
        ref_i8_pack_a_f16_quant_native(M, K, li8.align_in, A16.data(),
                                       rknpu2_matmul_open::cpu::QuantMode::StaticPerToken,
                                       static_scale.data(), ref_qscale.data(), ref_fqa.data());
        ok &= same_vec(got_fqa, ref_fqa, "i8 fused A static-token quant native");
        ok &= close_vec_f32(got_qscale, ref_qscale, "i8 fused A static-token scales");

        const float tensor_scale = 0.03125f;
        got_fqa.assign(got_fqa.size(), 0x55);
        ref_fqa.assign(ref_fqa.size(), 0x44);
        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_k16_m16(M, K, li8.align_in, A16.data(),
                                                  rknpu2_matmul_open::cpu::QuantMode::StaticTensor,
                                                  &tensor_scale, got_qscale.data(), got_fqa.data());
        ref_i8_pack_a_f16_quant_native(M, K, li8.align_in, A16.data(),
                                       rknpu2_matmul_open::cpu::QuantMode::StaticTensor,
                                       &tensor_scale, ref_qscale.data(), ref_fqa.data());
        ok &= same_vec(got_fqa, ref_fqa, "i8 fused A static-tensor quant native");
        ok &= close_vec_f32(got_qscale, ref_qscale, "i8 fused A static-tensor scales");

        std::vector<uint16_t> got_fb((size_t)lf16.align_out * lf16.align_in), ref_fb(got_fb.size());
        rk_npu_mem mfb = mem_view(got_fb);
        rk_npu_matmul_f16_pack_b(&cf16, B16.data(), &mfb);
        ref_f16_pack_b(K, N, lf16.align_in, lf16.align_out, B16.data(), ref_fb.data());
        ok &= same_vec(got_fb, ref_fb, "f16 pack B native");

        constexpr int f16_batch = 3;
        const size_t b_src_stride = (size_t)K * N;
        const size_t b_dst_stride = (size_t)lf16.align_out * lf16.align_in;
        std::vector<uint16_t> batch_b(f16_batch * b_src_stride);
        std::vector<uint16_t> got_batch_b(f16_batch * b_dst_stride, 0x55);
        std::vector<uint16_t> ref_batch_b(got_batch_b.size(), 0x44);
        fill_f16_values(batch_b, 440 + M + K + N);
        rk_npu_mem mbatch_b = mem_view(got_batch_b);
        ok &= rk_npu_matmul_f16_batch_pack_b(
                  f16_batch, &cf16, batch_b.data(), &mbatch_b) == RK_NPU_OK;
        for (int b = 0; b < f16_batch; ++b)
            ref_f16_pack_b(K, N, lf16.align_in, lf16.align_out,
                           batch_b.data() + (size_t)b * b_src_stride,
                           ref_batch_b.data() + (size_t)b * b_dst_stride);
        ok &= same_vec(got_batch_b, ref_batch_b,
                       "f16 batch pack B native");

        std::vector<uint16_t> packed_compact(std::max(
            (size_t)M * lf16.align_out, (size_t)256 / sizeof(uint16_t)));
        std::vector<uint16_t> got_compact((size_t)M * N), ref_compact(got_compact.size());
        fill_f16_values(packed_compact, 450 + M + K + N);
        rk_npu_mem mcompact = mem_view(packed_compact);
        rk_npu_matmul_f16_unpack_d(&cf16, &mcompact, got_compact.data());
        ref_f16_unpack_d_compact(M, N, lf16.align_out,
                                 packed_compact.data(), ref_compact.data());
        ok &= same_vec(got_compact, ref_compact, "f16 unpack D compact");
        std::vector<uint16_t> compact_bias((size_t)N);
        std::vector<uint16_t> got_compact_bias((size_t)M * N);
        std::vector<uint16_t> ref_compact_bias(got_compact_bias.size());
        fill_f16_values(compact_bias, 460 + M + K + N);
        ok &= rk_npu_matmul_f16_unpack_d_add_bias(
                  &cf16, &mcompact, compact_bias.data(),
                  got_compact_bias.data()) == RK_NPU_OK;
        for (int r = 0; r < M; ++r)
            for (int n = 0; n < N; ++n)
                ref_compact_bias[(size_t)r * N + n] = ref_float_to_half(
                    ref_half_to_float(ref_compact[(size_t)r * N + n]) +
                    ref_half_to_float(compact_bias[n]));
        ok &= same_vec(got_compact_bias, ref_compact_bias,
                       "f16 unpack D compact+bias");

        cf16.d_layout = RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
        std::vector<uint16_t> packed_native((size_t)M * lf16.align_out);
        std::vector<uint16_t> got_native((size_t)M * N), ref_native(got_native.size());
        fill_bytes(packed_native.data(), packed_native.size() * sizeof(uint16_t),
                   475 + M + K + N);
        rk_npu_mem mnative = mem_view(packed_native);
        rk_npu_matmul_f16_unpack_d(&cf16, &mnative, got_native.data());
        ref_f16_unpack_d_native(M, N, packed_native.data(), ref_native.data());
        ok &= same_vec(got_native, ref_native, "f16 unpack D native");
        cf16.d_layout = RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;

        cf16.op = RK_NPU_FUSE_ADD;
        rk_npu_matmul_sizes sf16{};
        rk_npu_matmul_f16_query(&cf16, &sf16);
        std::vector<uint16_t> got_op(sf16.operand_bytes / sizeof(uint16_t)), ref_op(got_op.size());
        std::vector<uint16_t> C016((size_t)M * N);
        fill_f16_values(C016, 425 + M + K + N);
        rk_npu_mem mop = mem_view(got_op);
        rk_npu_matmul_f16_pack_operand(&cf16, C016.data(), &mop);
        const int fused_align_in = std::max(lf16.align_in, lf16.align_out);
        ref_f16_pack_operand(M, N, lf16.align_out,
                             f16_m_tile(fused_align_in),
                             C016.data(), ref_op.data());
        ok &= same_vec(got_op, ref_op, "f16 pack operand");

        std::vector<uint16_t> packed_d((size_t)M * lf16.align_out * 2), got_d((size_t)M * N), ref_d(got_d.size());
        fill_bytes(packed_d.data(), packed_d.size() * sizeof(uint16_t), 500 + M + K + N);
        rk_npu_mem md = mem_view(packed_d);
        rk_npu_matmul_f16_unpack_d(&cf16, &md, got_d.data());
        ref_f16_unpack_d(M, N, lf16.align_out, packed_d.data(), ref_d.data());
        ok &= same_vec(got_d, ref_d, "f16 unpack D");
    }
    return ok;
}

template <typename Fn>
double bench_us(int loops, Fn&& fn) {
    double best = 1e300;
    for (int rep = 0; rep < g_repeats; ++rep) {
        fn();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < loops; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / loops;
        if (us < best) best = us;
    }
    return best;
}

int omp_threads() {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    return threads;
}

void run_perf_all(int M, int K, int N, int loops) {
    I8Layout li8 = i8_layout(N, K);
    F16Layout lf16 = f16_layout(N, K);
    std::printf("cpu_kernels M=%d K=%d N=%d loops=%d repeats=%d omp_threads=%d\n",
                M, K, N, loops, g_repeats, omp_threads());

    std::vector<int8_t> A8((size_t)M * K), B8((size_t)K * N), PA8((size_t)M * li8.align_in),
                         PB8((size_t)li8.align_out * li8.align_in);
    std::vector<uint32_t> C8n((size_t)((N + 3) / 4) * M * 4), C8((size_t)M * N);
    fill_bytes(A8.data(), A8.size(), 1);
    fill_bytes(B8.data(), B8.size(), 2);
    fill_bytes(C8n.data(), C8n.size() * sizeof(uint32_t), 3);

    rk_npu_matmul_i8_config ci8{};
    rk_npu_matmul_i8_config_init(&ci8, M, N, K);
    rk_npu_mem ma8 = mem_view(PA8), mb8 = mem_view(PB8), mc8n = mem_view(C8n);
    ci8.a_layout = RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16;
    ci8.c_layout = RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;

    auto i8a = bench_us(loops, [&] { rknpu2_matmul_open::detail::pack_i8_a(&ci8, A8.data(), &ma8); });
    auto i8b = bench_us(loops, [&] { rknpu2_matmul_open::detail::pack_i8_b(&ci8, B8.data(), &mb8); });
    auto i8c = bench_us(loops, [&] { rknpu2_matmul_open::detail::unpack_i8_c(&ci8, &mc8n, C8.data()); });
    std::printf("  i8_pack_a_native   %9.2f us  %.2f GB/s\n", i8a, ((double)PA8.size() / i8a) / 1000.0);
    std::printf("  i8_pack_b_native   %9.2f us  %.2f GB/s\n", i8b, ((double)PB8.size() / i8b) / 1000.0);
    std::printf("  i8_unpack_c_native %9.2f us  %.2f GB/s\n", i8c, ((double)C8.size() * sizeof(uint32_t) / i8c) / 1000.0);

    std::vector<uint16_t> A16((size_t)M * K), B16((size_t)K * N), PA16((size_t)M * lf16.align_in),
                          PB16((size_t)lf16.align_out * lf16.align_in),
                          D16p((size_t)M * lf16.align_out * 2), D16((size_t)M * N);
    fill_bytes(A16.data(), A16.size() * sizeof(uint16_t), 4);
    fill_bytes(B16.data(), B16.size() * sizeof(uint16_t), 5);
    fill_bytes(D16p.data(), D16p.size() * sizeof(uint16_t), 6);

    rk_npu_matmul_f16_config cf16{};
    rk_npu_matmul_f16_config_init(&cf16, M, N, K, RK_NPU_FUSE_NONE);
    rk_npu_mem ma16 = mem_view(PA16), mb16 = mem_view(PB16), md16 = mem_view(D16p);
    auto f16a = bench_us(loops, [&] { rk_npu_matmul_f16_pack_a(&cf16, A16.data(), &ma16); });
    auto f16b = bench_us(loops, [&] { rk_npu_matmul_f16_pack_b(&cf16, B16.data(), &mb16); });
    auto f16d = bench_us(loops, [&] { rk_npu_matmul_f16_unpack_d(&cf16, &md16, D16.data()); });
    std::printf("  f16_pack_a         %9.2f us  %.2f GB/s\n", f16a, ((double)PA16.size() * sizeof(uint16_t) / f16a) / 1000.0);
    std::printf("  f16_pack_b_native  %9.2f us  %.2f GB/s\n", f16b, ((double)PB16.size() * sizeof(uint16_t) / f16b) / 1000.0);
    std::printf("  f16_unpack_d       %9.2f us  %.2f GB/s\n", f16d, ((double)D16.size() * sizeof(uint16_t) / f16d) / 1000.0);
}

void run_perf_acd(int M, int K, int N, int loops) {
    I8Layout li8 = i8_layout(N, K);
    F16Layout lf16 = f16_layout(N, K);
    std::printf("cpu_kernels_acd M=%d K=%d N=%d loops=%d repeats=%d omp_threads=%d\n",
                M, K, N, loops, g_repeats, omp_threads());

    std::vector<int8_t> A8((size_t)M * K), PA8((size_t)M * li8.align_in);
    std::vector<int32_t> C8n((size_t)((N + 3) / 4) * M * 4), C8((size_t)M * N);
    fill_bytes(A8.data(), A8.size(), 11);
    fill_bytes(C8n.data(), C8n.size() * sizeof(uint32_t), 13);
    for (size_t i = 0; i < C8n.size(); ++i)
        C8n[i] = (int32_t)((i * 17u) % 2001u) - 1000;

    rk_npu_matmul_i8_config ci8{};
    rk_npu_matmul_i8_config_init(&ci8, M, N, K);
    ci8.a_layout = RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16;
    ci8.c_layout = RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;
    rk_npu_mem ma8 = mem_view(PA8), mc8n = mem_view(C8n);

    auto i8a = bench_us(loops, [&] { rknpu2_matmul_open::detail::pack_i8_a(&ci8, A8.data(), &ma8); });
    auto i8c = bench_us(loops, [&] { rknpu2_matmul_open::detail::unpack_i8_c(&ci8, &mc8n, C8.data()); });
    std::printf("  i8_pack_a_native   %9.2f us  %.2f GB/s\n", i8a, ((double)PA8.size() / i8a) / 1000.0);
    std::printf("  i8_unpack_c_native %9.2f us  %.2f GB/s\n", i8c, ((double)C8.size() * sizeof(uint32_t) / i8c) / 1000.0);

    std::vector<uint16_t> A16((size_t)M * K), PA16((size_t)M * lf16.align_in),
                          D16p((size_t)M * lf16.align_out * 2), D16((size_t)M * N);
    fill_f16_values(A16, 14);
    fill_bytes(D16p.data(), D16p.size() * sizeof(uint16_t), 16);

    std::vector<int8_t> Qrow((size_t)M * K), PA8fused((size_t)M * li8.align_in),
                         PA8sep((size_t)M * li8.align_in);
    std::vector<float> qscale((size_t)M);
    auto fused_a = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_k16_m16(M, K, li8.align_in, A16.data(),
                                                  rknpu2_matmul_open::cpu::QuantMode::DynamicPerToken,
                                                  nullptr, qscale.data(), PA8fused.data());
    });
    auto sep_a = bench_us(loops, [&] {
        quant_f16_to_i8_rowmajor_dynamic(M, K, A16.data(), Qrow.data(), qscale.data());
        rknpu2_matmul_open::cpu::i8_pack_a_native_k16_m16(M, K, li8.align_in, Qrow.data(), PA8sep.data());
    });
    const int one_k0[] = {0};
    const int one_k[] = {K};
    const int one_align[] = {li8.align_in};
    int8_t* one_normal[] = {PA8sep.data()};
    int8_t* one_native[] = {PA8fused.data()};
    auto normal_f16_a = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_normal_split(
            M, K, 1, one_k0, one_k, one_align, A16.data(),
            qscale.data(), one_normal);
    });
    auto native4_f16_a = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_native_split(
            M, K, 1, one_k0, one_k, one_align, A16.data(),
            qscale.data(), one_native);
    });
    std::vector<float> A32((size_t)M * K);
    for (size_t i = 0; i < A32.size(); ++i)
        A32[i] = ref_half_to_float(A16[i]);
    auto normal_f32_a = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_normal_split(
            M, K, 1, one_k0, one_k, one_align, A32.data(),
            qscale.data(), one_normal);
    });
    auto native4_f32_a = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_native_split(
            M, K, 1, one_k0, one_k, one_align, A32.data(),
            qscale.data(), one_native);
    });

    std::vector<float> a_scale((size_t)M), w_scale((size_t)N), bias((size_t)N);
    for (int r = 0; r < M; ++r) a_scale[r] = 0.001f + 0.000013f * (float)(r % 17);
    for (int n = 0; n < N; ++n) {
        w_scale[n] = 0.002f + 0.000007f * (float)(n % 23);
        bias[n] = ((n % 11) - 5) * 0.01f;
    }
    std::vector<int32_t> C8row((size_t)M * N);
    std::vector<uint16_t> Deq16Fused((size_t)M * N), Deq16Sep((size_t)M * N);
    std::vector<float> Deq32Fused((size_t)M * N);
    auto fused_c_f32 = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f32(M, N, C8n.data(), a_scale.data(),
                                                  w_scale.data(), nullptr,
                                                  rknpu2_matmul_open::cpu::ActivationOp::None, Deq32Fused.data());
    });
    std::vector<float> Deq32Ref((size_t)M * N);
    ref_i8_dequant_c_native_f32(
        M, N, C8n.data(), a_scale.data(), w_scale.data(), nullptr,
        rknpu2_matmul_open::cpu::ActivationOp::None, Deq32Ref.data());
    double deq32_max_abs = 0.0;
    double deq32_max_rel = 0.0;
    for (size_t i = 0; i < Deq32Ref.size(); ++i) {
        const double abs_err = std::fabs(
            (double)Deq32Fused[i] - (double)Deq32Ref[i]);
        const double denom = std::max(1.0e-30, std::fabs((double)Deq32Ref[i]));
        deq32_max_abs = std::max(deq32_max_abs, abs_err);
        deq32_max_rel = std::max(deq32_max_rel, abs_err / denom);
    }
    if (!close_vec_f32(Deq32Fused, Deq32Ref, "i8 fused C dequant f32 perf shape"))
        std::exit(1);
    auto fused_c_none = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f16(M, N, C8n.data(), a_scale.data(),
                                                  w_scale.data(), bias.data(),
                                                  rknpu2_matmul_open::cpu::ActivationOp::None, Deq16Fused.data());
    });
    auto sep_c_none = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_n4_m4(M, N, C8n.data(), C8row.data());
        dequant_rowmajor_i32_to_f16(M, N, C8row.data(), a_scale.data(), w_scale.data(),
                                    bias.data(), rknpu2_matmul_open::cpu::ActivationOp::None, Deq16Sep.data());
    });
    auto fused_c_silu = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f16(M, N, C8n.data(), a_scale.data(),
                                                  w_scale.data(), bias.data(),
                                                  rknpu2_matmul_open::cpu::ActivationOp::Silu, Deq16Fused.data());
    });

    std::vector<int32_t> C8n1 = C8n, C8n2 = C8n, C8n3 = C8n, C8acc = C8n;
    for (size_t i = 0; i < C8n.size(); ++i) {
        C8n1[i] += (int32_t)(i % 31);
        C8n2[i] -= (int32_t)(i % 47);
        C8n3[i] += (int32_t)(i % 59);
    }
    const int32_t* reduce2_src[] = {C8n.data(), C8n1.data()};
    const int32_t* reduce4_src[] = {C8n.data(), C8n1.data(), C8n2.data(), C8n3.data()};
    auto accum_c_i32 = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(M, N, C8n1.data(), C8acc.data());
    });
    auto reduce2_c_none = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
            M, N, 2, reduce2_src, a_scale.data(), w_scale.data(), bias.data(),
            rknpu2_matmul_open::cpu::ActivationOp::None, Deq16Fused.data());
    });
    auto reduce4_c_none = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
            M, N, 4, reduce4_src, a_scale.data(), w_scale.data(), bias.data(),
            rknpu2_matmul_open::cpu::ActivationOp::None, Deq16Fused.data());
    });

    rk_npu_matmul_f16_config cf16{};
    rk_npu_matmul_f16_config_init(&cf16, M, N, K, RK_NPU_FUSE_NONE);
    rk_npu_mem ma16 = mem_view(PA16), md16 = mem_view(D16p);
    auto f16a = bench_us(loops, [&] { rk_npu_matmul_f16_pack_a(&cf16, A16.data(), &ma16); });
    auto f16d = bench_us(loops, [&] { rk_npu_matmul_f16_unpack_d(&cf16, &md16, D16.data()); });
    std::printf("  i8_fused_a_f16_dyn %9.2f us  %.2f GB/s\n", fused_a, ((double)A16.size() * sizeof(uint16_t) / fused_a) / 1000.0);
    std::printf("  i8_sep_a_f16_dyn   %9.2f us  %.2f GB/s\n", sep_a, ((double)A16.size() * sizeof(uint16_t) / sep_a) / 1000.0);
    std::printf("  i8_normal_a_f16_4r %9.2f us  %.2f GB/s\n", normal_f16_a,
                ((double)A16.size() * sizeof(uint16_t) / normal_f16_a) / 1000.0);
    std::printf("  i8_native_a_f16_4r %9.2f us  %.2f GB/s\n", native4_f16_a,
                ((double)A16.size() * sizeof(uint16_t) / native4_f16_a) / 1000.0);
    std::printf("  i8_normal_a_f32_4r %9.2f us  %.2f GB/s\n", normal_f32_a,
                ((double)A32.size() * sizeof(float) / normal_f32_a) / 1000.0);
    std::printf("  i8_native_a_f32_4r %9.2f us  %.2f GB/s\n", native4_f32_a,
                ((double)A32.size() * sizeof(float) / native4_f32_a) / 1000.0);
    std::printf("  i8_fused_c_deq_f32 %9.2f us  %.2f GB/s logical-read+write\n", fused_c_f32,
                ((double)Deq32Fused.size() * 8.0 / fused_c_f32) / 1000.0);
    std::printf("  i8_fused_c_f32_err max_abs=%.9g max_rel=%.9g\n",
                deq32_max_abs, deq32_max_rel);
    std::printf("  i8_fused_c_deq_f16 %9.2f us  %.2f GB/s\n", fused_c_none, ((double)Deq16Fused.size() * sizeof(uint16_t) / fused_c_none) / 1000.0);
    std::printf("  i8_sep_c_deq_f16   %9.2f us  %.2f GB/s\n", sep_c_none, ((double)Deq16Sep.size() * sizeof(uint16_t) / sep_c_none) / 1000.0);
    std::printf("  i8_fused_c_silu16  %9.2f us  %.2f GB/s\n", fused_c_silu, ((double)Deq16Fused.size() * sizeof(uint16_t) / fused_c_silu) / 1000.0);
    const double native_elems = (double)C8n.size();
    std::printf("  i8_accum_c_i32     %9.2f us  %.2f GB/s logical-warm-cache\n", accum_c_i32,
                native_elems * 12.0 / accum_c_i32 / 1000.0);
    std::printf("  i8_reduce2_deq_f16 %9.2f us  %.2f GB/s logical-warm-cache\n", reduce2_c_none,
                native_elems * 10.0 / reduce2_c_none / 1000.0);
    std::printf("  i8_reduce4_deq_f16 %9.2f us  %.2f GB/s logical-warm-cache\n", reduce4_c_none,
                native_elems * 18.0 / reduce4_c_none / 1000.0);

    if (g_stream_mib > 0) {
        /* Rotate through a working set larger than the CPU caches.  This is a
         * DDR-oriented approximation, not a dmabuf coherency measurement: the
         * production overlap benchmark must still include NPU writes + sync. */
        const size_t native_count = C8n.size();
        const size_t output_count = (size_t)M * N;
        const size_t bytes_per_slot = native_count * sizeof(int32_t) * 5
                                    + output_count * sizeof(uint16_t);
        const size_t target_bytes = (size_t)g_stream_mib << 20;
        const size_t slots = std::max<size_t>(2, (target_bytes + bytes_per_slot - 1) /
                                                 bytes_per_slot);
        std::vector<int32_t> stream_partial(slots * native_count * 4);
        std::vector<int32_t> stream_accum(slots * native_count);
        std::vector<uint16_t> stream_out(slots * output_count);
        for (size_t i = 0; i < stream_partial.size(); ++i)
            stream_partial[i] = (int32_t)((i * 17u) % 2001u) - 1000;
        for (size_t i = 0; i < stream_accum.size(); ++i)
            stream_accum[i] = (int32_t)((i * 29u) % 3001u) - 1500;

        size_t slot = 0;
        auto stream_accum_us = bench_us(loops, [&] {
            const size_t off = slot * native_count;
            rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(
                M, N, stream_partial.data() + off, stream_accum.data() + off);
            slot = (slot + 1) % slots;
        });
        slot = 0;
        auto stream_reduce2_us = bench_us(loops, [&] {
            const size_t off = slot * native_count;
            const int32_t* src[] = {
                stream_partial.data() + off,
                stream_partial.data() + slots * native_count + off,
            };
            rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
                M, N, 2, src, a_scale.data(), w_scale.data(), bias.data(),
                rknpu2_matmul_open::cpu::ActivationOp::None, stream_out.data() + slot * output_count);
            slot = (slot + 1) % slots;
        });
        slot = 0;
        auto stream_reduce4_us = bench_us(loops, [&] {
            const size_t off = slot * native_count;
            const int32_t* src[] = {
                stream_partial.data() + off,
                stream_partial.data() + slots * native_count + off,
                stream_partial.data() + 2 * slots * native_count + off,
                stream_partial.data() + 3 * slots * native_count + off,
            };
            rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(
                M, N, 4, src, a_scale.data(), w_scale.data(), bias.data(),
                rknpu2_matmul_open::cpu::ActivationOp::None, stream_out.data() + slot * output_count);
            slot = (slot + 1) % slots;
        });
        std::printf("  stream_working_set %d MiB slots=%zu (logical bytes, no NPU sync)\n",
                    g_stream_mib, slots);
        std::printf("  i8_accum_c_i32     %9.2f us  %.2f GB/s logical-stream\n", stream_accum_us,
                    native_elems * 12.0 / stream_accum_us / 1000.0);
        std::printf("  i8_reduce2_deq_f16 %9.2f us  %.2f GB/s logical-stream\n", stream_reduce2_us,
                    native_elems * 10.0 / stream_reduce2_us / 1000.0);
        std::printf("  i8_reduce4_deq_f16 %9.2f us  %.2f GB/s logical-stream\n", stream_reduce4_us,
                    native_elems * 18.0 / stream_reduce4_us / 1000.0);
    }
    std::printf("  f16_pack_a         %9.2f us  %.2f GB/s\n", f16a, ((double)PA16.size() * sizeof(uint16_t) / f16a) / 1000.0);
    std::printf("  f16_unpack_d       %9.2f us  %.2f GB/s\n", f16d, ((double)D16.size() * sizeof(uint16_t) / f16d) / 1000.0);
}

void run_perf_dequant_f32_only(int M, int K, int N, int loops) {
    std::printf("cpu_kernel_f32_dequant_only M=%d K=%d N=%d loops=%d repeats=%d omp_threads=%d\n",
                M, K, N, loops, g_repeats, omp_threads());
    std::vector<int32_t> src((size_t)((N + 3) / 4) * M * 4);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = (int32_t)((i * 17u) % 2001u) - 1000;
    std::vector<float> a_scale((size_t)M), w_scale((size_t)N);
    for (int r = 0; r < M; ++r)
        a_scale[r] = 0.001f + 0.000013f * (float)(r % 17);
    for (int n = 0; n < N; ++n)
        w_scale[n] = 0.002f + 0.000007f * (float)(n % 23);

    std::vector<float> got((size_t)M * N), ref(got.size());
    rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f32(
        M, N, src.data(), a_scale.data(), w_scale.data(), nullptr,
        rknpu2_matmul_open::cpu::ActivationOp::None, got.data());
    ref_i8_dequant_c_native_f32(
        M, N, src.data(), a_scale.data(), w_scale.data(), nullptr,
        rknpu2_matmul_open::cpu::ActivationOp::None, ref.data());
    if (!close_vec_f32(got, ref, "i8 fused C dequant f32 focused perf"))
        std::exit(1);

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double abs_err = std::fabs((double)got[i] - (double)ref[i]);
        const double denom = std::max(1.0e-30, std::fabs((double)ref[i]));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, abs_err / denom);
    }
    const double us = bench_us(loops, [&] {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_i32_dequant_f32(
            M, N, src.data(), a_scale.data(), w_scale.data(), nullptr,
            rknpu2_matmul_open::cpu::ActivationOp::None, got.data());
    });
    std::printf("  i8_fused_c_deq_f32 %9.2f us  %.2f GB/s logical-read+write\n",
                us, ((double)got.size() * 8.0 / us) / 1000.0);
    std::printf("  i8_fused_c_f32_err max_abs=%.9g max_rel=%.9g checksum=%.9g\n",
                max_abs, max_rel, (double)got[got.size() / 2]);
}

void run_perf_b(int K, int N, int loops) {
    constexpr int M = 1;
    I8Layout li8 = i8_layout(N, K);
    F16Layout lf16 = f16_layout(N, K);
    std::printf("cpu_kernels_b K=%d N=%d loops=%d repeats=%d omp_threads=%d\n",
                K, N, loops, g_repeats, omp_threads());

    std::vector<int8_t> B8((size_t)K * N), PB8((size_t)li8.align_out * li8.align_in);
    fill_bytes(B8.data(), B8.size(), 22);
    rk_npu_matmul_i8_config ci8{};
    rk_npu_matmul_i8_config_init(&ci8, M, N, K);
    rk_npu_mem mb8 = mem_view(PB8);
    auto i8b = bench_us(loops, [&] { rknpu2_matmul_open::detail::pack_i8_b(&ci8, B8.data(), &mb8); });
    std::printf("  i8_pack_b_native   %9.2f us  %.2f GB/s\n", i8b, ((double)PB8.size() / i8b) / 1000.0);

    std::vector<uint16_t> B16((size_t)K * N), PB16((size_t)lf16.align_out * lf16.align_in);
    fill_bytes(B16.data(), B16.size() * sizeof(uint16_t), 25);
    rk_npu_matmul_f16_config cf16{};
    rk_npu_matmul_f16_config_init(&cf16, M, N, K, RK_NPU_FUSE_NONE);
    rk_npu_mem mb16 = mem_view(PB16);
    auto f16b = bench_us(loops, [&] { rk_npu_matmul_f16_pack_b(&cf16, B16.data(), &mb16); });
    std::printf("  f16_pack_b_native  %9.2f us  %.2f GB/s\n", f16b, ((double)PB16.size() * sizeof(uint16_t) / f16b) / 1000.0);
}

void run_perf_suite(int loops) {
    run_perf_acd(128, 2048, 6144, loops);
    run_perf_acd(128, 6144, 2048, loops);
    run_perf_b(1024, 1024, loops);
}

} /* namespace */

int main(int argc, char** argv) {
    bool perf = false;
    bool suite = false;
    bool reduction_perf = false;
    bool f32_dequant_only = false;
    bool explicit_shape = false;
    int M = 128, K = 2048, N = 6144, loops = 50;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", o.c_str());
                std::exit(2);
            }
            return argv[i];
        };
        if (o == "--perf") perf = true;
        else if (o == "--reduction-perf") { perf = true; reduction_perf = true; }
        else if (o == "--f32-dequant-only") { perf = true; f32_dequant_only = true; }
        else if (o == "--perf-suite" || o == "--suite") { perf = true; suite = true; }
        else if (o == "--m") { M = std::atoi(next()); explicit_shape = true; }
        else if (o == "--k") { K = std::atoi(next()); explicit_shape = true; }
        else if (o == "--n") { N = std::atoi(next()); explicit_shape = true; }
        else if (o == "--loops") loops = std::atoi(next());
        else if (o == "--repeats") g_repeats = std::atoi(next());
        else if (o == "--stream-mib") g_stream_mib = std::atoi(next());
        else {
            std::fprintf(stderr, "unknown arg: %s\n", o.c_str());
            return 2;
        }
    }

    if (f32_dequant_only) {
        run_perf_dequant_f32_only(M, K, N, loops);
        return 0;
    }
    if (!run_correctness()) {
        std::fprintf(stderr, "cpu kernel correctness: FAIL\n");
        return 1;
    }
    if (g_repeats < 1) g_repeats = 1;
    if (g_stream_mib < 0) g_stream_mib = 0;
    std::printf("cpu kernel correctness: PASS\n");
    if (perf) {
        if (reduction_perf) run_perf_acd(M, K, N, loops);
        else if (suite || !explicit_shape) run_perf_suite(loops);
        else run_perf_all(M, K, N, loops);
    }
    return 0;
}

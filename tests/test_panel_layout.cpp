#include "../src/rk_npu_cpu_kernels.h"
#include "rk_npu_quant_matmul.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int up(int n) { return (n + 31) / 32 * 32; }
void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}

// Independent scalar layout oracle: emit complete channel blocks one panel at
// a time. Includes physical N/K padding and never calls production offsets.
template <class T> std::vector<T> panel(const std::vector<T> &row, int M, int C, int block, int W) {
    std::vector<T> out((size_t)M * up(C));
    size_t index = 0;
    for (int m = 0; m < M; m += W)
        for (int c = 0; c < up(C); c += block)
            for (int r = 0; r < W; ++r)
                for (int lane = 0; lane < block; ++lane)
                    out[index++] = c + lane < C ? row[(size_t)(m + r) * C + c + lane] : T{};
    return out;
}
uint16_t half_int(int x) {
    const uint16_t bits[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    return bits[std::abs(x)] | (x < 0 ? 0x8000 : 0);
}

void cpu_tests() {
    for (int M : {16, 32, 128})
        for (int K : {17, 64, 95, 1536, 2049})
            for (int W : {8, 16}) {
                std::vector<int8_t> a((size_t)M * K), got((size_t)M * up(K));
                std::vector<float> f(a.size()), scales(M, 1.0f), ns(M), ps(M);
                std::vector<uint16_t> h(a.size());
                for (size_t i = 0; i < a.size(); ++i) {
                    int x = int((i * 17 + i / K) % 15) - 7;
                    a[i] = x;
                    f[i] = float(x);
                    h[i] = half_int(x);
                }
                rknpu2_matmul_open::cpu::i8_pack_a_native_k16_m16(M, K, up(K), a.data(), got.data(), W);
                require(got == panel(a, M, K, 16, W), "raw A panel / padded K");
                std::vector<int8_t> normal(got.size());
                rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_normal_strided(M, K, up(K), f.data(), K, scales.data(),
                                                          normal.data());
                rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_native_strided(M, K, up(K), f.data(), K, scales.data(),
                                                          got.data(), W);
                require(got == panel(normal, M, up(K), 16, W), "FP32 static quant panel");
                rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_normal_strided(M, K, up(K), h.data(), K, scales.data(),
                                                          normal.data());
                rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_strided(M, K, up(K), h.data(), K, scales.data(),
                                                          got.data(), W);
                require(got == panel(normal, M, up(K), 16, W), "FP16 static quant panel");
                int k0[] = {0, K / 2}, ks[] = {K / 2, K - K / 2}, ak[] = {up(ks[0]), up(ks[1])};
                std::vector<int8_t> n0((size_t)M * ak[0]), n1((size_t)M * ak[1]), p0(n0.size()),
                    p1(n1.size());
                int8_t *normal_slices[] = {n0.data(), n1.data()};
                int8_t *panel_slices[] = {p0.data(), p1.data()};
                rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_normal_split(M, K, 2, k0, ks, ak, f.data(), ns.data(),
                                                          normal_slices);
                rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_native_split(M, K, 2, k0, ks, ak, f.data(), ps.data(),
                                                          panel_slices, W);
                require(ns == ps && p0 == panel(n0, M, ak[0], 16, W) &&
                            p1 == panel(n1, M, ak[1], 16, W),
                        "FP32 full-row scale / K tails");
                rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_normal_split(M, K, 2, k0, ks, ak, h.data(), ns.data(),
                                                          normal_slices);
                rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_native_split(M, K, 2, k0, ks, ak, h.data(), ps.data(),
                                                          panel_slices, W);
                require(ns == ps && p0 == panel(n0, M, ak[0], 16, W) &&
                            p1 == panel(n1, M, ak[1], 16, W),
                        "FP16 full-row scale / K tails");
            }
    for (int M : {16, 32, 128})
        for (int N : {17, 33, 96, 512, 2048, 4096})
            for (int W : {8, 16}) {
                std::vector<int32_t> row((size_t)M * N), out(row.size()), sum(row.size());
                for (size_t i = 0; i < row.size(); ++i)
                    row[i] = int(i * 37 % 8191) - 4095;
                auto packed = panel(row, M, N, 4, W), native = panel(row, M, N, 4, M), acc = packed;
                rknpu2_matmul_open::cpu::i8_unpack_c_native_n4_m4(M, N, packed.data(), out.data(), W);
                require(out == row, "C panel unpack / N padding");
                rknpu2_matmul_open::cpu::i8_accumulate_c_native_i32(M, N, packed.data(), acc.data(), W);
                const int32_t *pp[] = {acc.data(), packed.data()};
                const int32_t *nn[] = {native.data(), native.data(), native.data()};
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_to_rowmajor(M, N, 2, pp, out.data(), W);
                for (size_t i = 0; i < out.size(); ++i)
                    require(out[i] == row[i] * 3, "panel split-K accumulation");
                std::vector<float> as(M), ws(N), bias(N), fg(row.size()), fr(row.size());
                std::vector<uint16_t> hg(row.size()), hr(row.size());
                for (int m = 0; m < M; ++m)
                    as[m] = 0.015625f * (1 + m % 7);
                for (int n = 0; n < N; ++n) {
                    ws[n] = 0.03125f * (1 + n % 5);
                    bias[n] = (n % 3) * 0.125f;
                }
                for (auto act : {rknpu2_matmul_open::cpu::ActivationOp::None, rknpu2_matmul_open::cpu::ActivationOp::Relu}) {
                    rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(M, N, 3, nn, as.data(), ws.data(),
                                                              bias.data(), act, fr.data());
                    rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(M, N, 2, pp, as.data(), ws.data(),
                                                              bias.data(), act, fg.data(), W);
                    require(fg == fr, "fused panel reduce/dequant FP32");
                    rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(M, N, 3, nn, as.data(), ws.data(),
                                                              bias.data(), act, hr.data());
                    rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(M, N, 2, pp, as.data(), ws.data(),
                                                              bias.data(), act, hg.data(), W);
                    require(hg == hr, "fused panel reduce/dequant FP16");
                }
                const int32_t *onep[] = {packed.data()};
                const int32_t *onen[] = {native.data()};
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(M, N, 1, onen, as.data(), ws.data(),
                                                          nullptr, rknpu2_matmul_open::cpu::ActivationOp::None,
                                                          fr.data());
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f32(M, N, 1, onep, as.data(), ws.data(),
                                                          nullptr, rknpu2_matmul_open::cpu::ActivationOp::None,
                                                          fg.data(), W);
                require(fg == fr, "single-partial FP32 panel fastpath");
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(M, N, 1, onen, as.data(), ws.data(),
                                                          nullptr, rknpu2_matmul_open::cpu::ActivationOp::None,
                                                          hr.data());
                rknpu2_matmul_open::cpu::i8_reduce_c_native_i32_dequant_f16(M, N, 1, onep, as.data(), ws.data(),
                                                          nullptr, rknpu2_matmul_open::cpu::ActivationOp::None,
                                                          hg.data(), W);
                require(hg == hr, "single-partial FP16 panel fastpath");
            }
    std::puts("PASS panel CPU packing, padding, quantization, split-K and fused output");
}

struct Recipe {
    const char *name;
    rk_npu_matmul_a_layout a;
    rk_npu_matmul_c_layout c;
};
const Recipe recipes[] = {
    {"normal", RK_NPU_MATMUL_A_LAYOUT_NORMAL, RK_NPU_MATMUL_C_LAYOUT_NATIVE},
    {"native", RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16, RK_NPU_MATMUL_C_LAYOUT_NATIVE},
    {"pn8", RK_NPU_MATMUL_A_LAYOUT_PANEL8, RK_NPU_MATMUL_C_LAYOUT_NATIVE},
    {"pn16", RK_NPU_MATMUL_A_LAYOUT_PANEL16, RK_NPU_MATMUL_C_LAYOUT_NATIVE},
    {"pc8", RK_NPU_MATMUL_A_LAYOUT_NORMAL, RK_NPU_MATMUL_C_LAYOUT_PANEL8},
    {"p8", RK_NPU_MATMUL_A_LAYOUT_PANEL8, RK_NPU_MATMUL_C_LAYOUT_PANEL8},
    {"p16", RK_NPU_MATMUL_A_LAYOUT_PANEL16, RK_NPU_MATMUL_C_LAYOUT_PANEL16},
};

void board_case(rk_npu_ctx *ctx, rk_npu_iommu_domain *domain, int M, int K, int N, int kt, int nt,
                unsigned mask, int loops, bool compressed) {
    std::mt19937 rng(2819 + M + K + N);
    std::vector<int8_t> a((size_t)M * K), b((size_t)K * N);
    std::vector<int32_t> ref((size_t)M * N), out(ref.size());
    std::vector<float> af(a.size()), ws(N, 0.03125f), sf(M, 1.0f), fout(out.size());
    std::vector<uint16_t> ah(a.size()), hout(out.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = int(rng() % 15) - 7;
        af[i] = a[i];
        ah[i] = half_int(a[i]);
    }
    for (auto &x : b)
        x = compressed ? int(rng() % 15) - 7 : int(rng() % 255) - 127;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                ref[(size_t)m * N + n] += int(a[(size_t)m * K + k]) * b[(size_t)k * N + n];
    rk_npu_matmul_weight_config wc{K, N, kt};
    auto *w8 = compressed ? rk_npu_i8i8i32_weights_create_compress(domain, &wc, b.data())
                          : rk_npu_i8i8i32_weights_create(domain, &wc, b.data());
    auto *w32 = compressed
                    ? rk_npu_f32i8f32_weights_create_compress(domain, &wc, b.data(), ws.data())
                    : rk_npu_f32i8f32_weights_create(domain, &wc, b.data(), ws.data());
    auto *w16 = compressed
                    ? rk_npu_f16i8f16_weights_create_compress(domain, &wc, b.data(), ws.data())
                    : rk_npu_f16i8f16_weights_create(domain, &wc, b.data(), ws.data());
    require(w8 && w32 && w16, "board weight allocation");
    int panel_passed = 0;
    for (const auto &recipe : recipes) {
        rk_npu_matmul_strategy s{};
        s.M = M;
        s.K = K;
        s.N = N;
        s.k_tile = kt;
        s.n_tile = nt;
        s.a_layout = recipe.a;
        s.c_layout = recipe.c;
        s.npu_core_mask = mask;
        s.cpu_core_mask = 0xf0;
        s.cpu_threads = 4;
        s.wave_count = (K + kt - 1) / kt;
        s.n_groups = (up(N) + nt - 1) / nt;
        // Unsupported panel envelopes are rejected by query before any submit.
        rk_npu_matmul_workspace_requirements req{};
        s.op_kind = RK_NPU_MATMUL_I8I8I32;
        if (rk_npu_matmul_workspace_memory_query(&s, &req) != RK_NPU_OK)
            continue;
        if (recipe.a >= RK_NPU_MATMUL_A_LAYOUT_PANEL8 || recipe.c != RK_NPU_MATMUL_C_LAYOUT_NATIVE)
            ++panel_passed;
        auto *work = rk_npu_matmul_workspace_create(domain, &s);
        require(work, "board workspace");
        double total = 0;
        for (int i = 0; i < loops; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            require(rk_npu_i8i8i32_run(work, w8, a.data(), out.data()) == RK_NPU_OK,
                    "board INT8 run");
            total +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count();
            require(out == ref, "board INT32 exact / repeated submits");
        }
        rk_npu_matmul_workspace_free(work);
        s.op_kind = RK_NPU_MATMUL_F32I8F32_STATIC;
        work = rk_npu_matmul_workspace_create(domain, &s);
        require(work, "FP32 workspace");
        require(rk_npu_f32i8f32_run_static(work, w32, af.data(), sf.data(), fout.data()) ==
                    RK_NPU_OK,
                "FP32 panel run");
        for (size_t i = 0; i < out.size(); ++i)
            require(fout[i] == float(ref[i]) * 0.03125f, "board fused FP32 exact");
        rk_npu_matmul_workspace_free(work);
        // Dynamic paths compare to the same typed normal-layout production path.
        for (auto kind : {RK_NPU_MATMUL_F32I8F32_DYNAMIC, RK_NPU_MATMUL_F16I8F16_DYNAMIC}) {
            s.op_kind = kind;
            work = rk_npu_matmul_workspace_create(domain, &s);
            require(work, "dynamic workspace");
            auto base = s;
            base.a_layout = RK_NPU_MATMUL_A_LAYOUT_NORMAL;
            base.c_layout = RK_NPU_MATMUL_C_LAYOUT_NATIVE;
            auto *control = rk_npu_matmul_workspace_create(domain, &base);
            require(control, "dynamic control");
            if (kind == RK_NPU_MATMUL_F32I8F32_DYNAMIC) {
                std::vector<float> expected(out.size());
                require(rk_npu_f32i8f32_run_dynamic(control, w32, af.data(), expected.data()) ==
                            RK_NPU_OK,
                        "FP32 dynamic control");
                require(rk_npu_f32i8f32_run_dynamic(work, w32, af.data(), fout.data()) == RK_NPU_OK,
                        "FP32 dynamic panel");
                for (size_t i = 0; i < out.size(); ++i)
                    require(std::fabs(fout[i] - expected[i]) <=
                                1e-6f * std::max(1.0f, std::fabs(expected[i])),
                            "FP32 dynamic match");
            } else {
                std::vector<uint16_t> expected(out.size());
                require(rk_npu_f16i8f16_run_dynamic(control, w16, ah.data(), expected.data()) ==
                            RK_NPU_OK,
                        "FP16 dynamic control");
                require(rk_npu_f16i8f16_run_dynamic(work, w16, ah.data(), hout.data()) == RK_NPU_OK,
                        "FP16 dynamic panel");
                require(expected == hout, "FP16 dynamic match");
            }
            rk_npu_matmul_workspace_free(control);
            rk_npu_matmul_workspace_free(work);
        }
        std::printf(
            "PASS M=%d K=%d N=%d Kt=%d Nt=%d mask=%u layout=%s compressed=%d i8_total_us=%.3f\n", M,
            K, N, kt, nt, mask, recipe.name, int(compressed), total / loops);
        std::fflush(stdout);
    }
    require(panel_passed > 0, "board test must exercise a panel, not silently skip all recipes");
    rk_npu_i8i8i32_weights_free(w8);
    rk_npu_f32i8f32_weights_free(w32);
    rk_npu_f16i8f16_weights_free(w16);
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            cpu_tests();
            return 0;
        }
        require((argc == 9 || (argc == 10 && std::string(argv[9]) == "--compressed")) &&
                    std::string(argv[1]) == "--board",
                "usage: --board M K N Ktile Ntile mask loops [--compressed]");
        auto *ctx = rk_npu_open(nullptr);
        require(ctx, "open board");
        auto *domain = rk_npu_iommu_domain_create(ctx, 0);
        require(domain, "domain");
        board_case(ctx, domain, std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
                   std::atoi(argv[5]), std::atoi(argv[6]), std::strtoul(argv[7], nullptr, 0),
                   std::atoi(argv[8]), argc == 10);
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

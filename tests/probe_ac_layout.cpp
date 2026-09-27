/*
 * probe_ac_layout.cpp - split the vendor AC_layout knob into independent
 * A-input and output-C layout probes.
 *
 * This is intentionally a test-only raw-register probe.  It answers:
 *   normal A + normal C     baseline
 *   native A + normal C     does input layout alone matter/work?
 *   normal A + native C     does output layout alone matter/work?
 *   native A + native C     vendor AC_layout=1-like setup
 *
 * This probe's native-A register configuration is limited to M=128,K=1024.
 * Validate the register geometry for additional shapes before extending
 * the native-A test range.
 */
#include "test_common.h"
#include "../src/rk_npu_internal.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace rknpu2_matmul_open::test;

namespace {

struct Args {
    const char* dev = nullptr;
    int M = 128;
    int K = 1024;
    int N = 8192;
    int loops = 20;
    int timeout_ms = 800;
    bool fp32 = false;
};

struct Layout { int align_in, align_out; };

Layout layout_for(int N, int K) {
    return {
        std::max(MIN_CHANNEL_TILE, align_up(K, MIN_CHANNEL_TILE)),
        std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE)),
    };
}

int feature_grains_for(int tile_m, int input_row_bytes) {
    int two_bank_rows = (ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1;
    return std::max(1, std::min(tile_m + 1, two_bank_rows));
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (o == "--dev")     a.dev = next();
        else if (o == "--m")       a.M = std::atoi(next());
        else if (o == "--k")       a.K = std::atoi(next());
        else if (o == "--n")       a.N = std::atoi(next());
        else if (o == "--loops")   a.loops = std::atoi(next());
        else if (o == "--timeout") a.timeout_ms = std::atoi(next());
        else if (o == "--fp32")    a.fp32 = true;
        else { std::fprintf(stderr, "unknown option: %s\n", o.c_str()); std::exit(2); }
    }
    return a;
}

void pack_a_normal(int M, int K, int align_in, const int8_t* A, rk_npu_mem* input) {
    int8_t* dst = (int8_t*)input->vaddr;
    std::memset(dst, 0, (size_t)M * align_in);
    for (int m = 0; m < M; ++m)
        std::memcpy(dst + (size_t)m * align_in, A + (size_t)m * K, K);
}

void pack_a_native(int M, int K, int align_in, const int8_t* A, rk_npu_mem* input) {
    constexpr int subK = 16;
    int8_t* dst = (int8_t*)input->vaddr;
    std::memset(dst, 0, (size_t)M * align_in);
    for (int kb = 0; kb < align_in / subK; ++kb) {
        for (int m = 0; m < M; ++m) {
            for (int kk = 0; kk < subK; ++kk) {
                int k = kb * subK + kk;
                dst[(size_t)kb * M * subK + (size_t)m * subK + kk] =
                    (k < K) ? A[(size_t)m * K + k] : 0;
            }
        }
    }
}

void unpack_c_normal(int M, int N, int align_out, const rk_npu_mem* output,
                     bool out_i32, std::vector<float>& C) {
    const uint8_t* base = (const uint8_t*)output->vaddr;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const void* p = base + ((size_t)m * align_out + n) * 4;
            if (out_i32) {
                int32_t v; std::memcpy(&v, p, 4); C[(size_t)m * N + n] = (float)v;
            } else {
                float v; std::memcpy(&v, p, 4); C[(size_t)m * N + n] = v;
            }
        }
    }
}

void unpack_c_native(int M, int N, const rk_npu_mem* output,
                     bool out_i32, std::vector<float>& C) {
    const uint8_t* base = (const uint8_t*)output->vaddr;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            size_t idx = ((size_t)(n / 4) * M + m) * 4 + (n % 4);
            const void* p = base + idx * 4;
            if (out_i32) {
                int32_t v; std::memcpy(&v, p, 4); C[(size_t)m * N + n] = (float)v;
            } else {
                float v; std::memcpy(&v, p, 4); C[(size_t)m * N + n] = v;
            }
        }
    }
}

void make_regs(std::vector<uint64_t>& v, int M, int N, int K,
               uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma,
               bool a_native, bool c_native, bool out_i32) {
    Layout L = layout_for(N, K);
    const int align_in = L.align_in, align_out = L.align_out;
    const int data_banks = std::min(std::max(ceil_div(M * align_in, CBUF_BANK_SIZE), 1), RK_CBUF_BANKS - 1);
    const int normal_feature_grains = feature_grains_for(M, align_in);
    const int normal_line_stride = std::max(1, align_in / 16);
    const int normal_notch = align_out / 4 - 1;

    const uint32_t cna_conv1 = a_native ? 0u : (1u << 29);
    const uint32_t cna_conv2 = a_native ? (2u << 4) : ((uint32_t)normal_feature_grains << 4);
    const uint32_t data_size0 = a_native ? (((uint32_t)M << 16) | 1u) : ((1u << 16) | (uint32_t)M);
    const uint32_t data_size2 = a_native ? (uint32_t)M : 1u;
    const uint32_t cbuf1 = a_native ? (uint32_t)(M * 16) : (uint32_t)ceil_div(align_in, 64);
    const uint32_t dma_con1 = a_native ? (uint32_t)(M * 4) : (uint32_t)normal_line_stride;
    const uint32_t dma_con2 = a_native ? (0x10000000u - (uint32_t)M * 3u) : 0u;
    const uint32_t core_size0 = a_native ? (uint32_t)(M - 1) : ((uint32_t)(M - 1) << 16);

    const uint32_t dst_stride = c_native ? (uint32_t)(M * 16) : 0x10u;
    const uint32_t dst_w = c_native ? (uint32_t)(M - 1) : 0u;
    const uint32_t dst_h = c_native ? 0u : (uint32_t)(M - 1);
    const uint32_t dst_notch = c_native ? 0u : (((uint32_t)normal_notch << 16) | (uint32_t)normal_notch);
    const uint32_t wdma_size1 = c_native ? (uint32_t)(M - 1) : ((uint32_t)(M - 1) << 16);
    const uint32_t surface_add = c_native ? ((uint32_t)M << 7) : 0x80u;

    v.clear();
    v.push_back(E(T_DPU,  R_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON1, cna_conv1));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON2, cna_conv2));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON3, (1 << 3) | 1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE0, data_size0));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE1, ((uint32_t)(align_in - 1) << 16) | (uint32_t)align_in));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE2, data_size2));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE3, (uint32_t)M));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE0, (uint32_t)align_in * align_out));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE1, (uint32_t)align_in));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE2, (1u << 24) | (1u << 16) | (uint32_t)align_out));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON0, ((uint32_t)(RK_CBUF_BANKS - data_banks) << 4) | (uint32_t)data_banks));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON1, cbuf1));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON1, (1u << 16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON2, (1u << 16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON3, (1u << 16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON4, (1u << 16)));
    v.push_back(E(T_CNA,  R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_dma));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON0, (15u << 16) | 15u));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON1, dma_con1));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON2, dma_con2));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE0, data_size0));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE1, (uint32_t)align_in));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_ADDR0, (uint32_t)wt_dma));
    v.push_back(E(T_CORE, R_CORE_MISC_CFG, 1));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_0, core_size0));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_1, (uint32_t)align_out - 1));
    v.push_back(E(T_CORE, R_CORE_RESERVED_3030, 0));
    v.push_back(E(T_DPU,  R_FEATURE_MODE_CFG, (15 << 5) | (2 << 1)));
    v.push_back(E(T_DPU,  R_DATA_FORMAT, ((out_i32 ? 4u : 5u) << 29)));
    v.push_back(E(T_DPU,  R_DST_BASE_ADDR, (uint32_t)out_dma));
    v.push_back(E(T_DPU,  R_DST_SURF_STRIDE, dst_stride));
    v.push_back(E(T_DPU,  R_DATA_CUBE_WIDTH, dst_w));
    v.push_back(E(T_DPU,  R_DATA_CUBE_HEIGHT, dst_h));
    v.push_back(E(T_DPU,  R_DATA_CUBE_NOTCH, dst_notch));
    v.push_back(E(T_DPU,  R_DATA_CUBE_CHANNEL, ((uint32_t)(align_out - 1) << 16) | ((uint32_t)align_out - 1)));
    v.push_back(E(T_DPU,  R_BS_CFG, 0x53));
    v.push_back(E(T_DPU,  R_BS_OW_CFG, 0x7fc));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_0, (uint32_t)align_out - 1));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_1, wdma_size1));
    v.push_back(E(T_DPU,  R_BN_CFG, 0x53));
    v.push_back(E(T_DPU,  R_EW_CFG, 0x383));
    v.push_back(E(T_DPU,  R_OUT_CVT_SCALE, 1));
    v.push_back(E(T_DPU,  R_SURFACE_ADD, surface_add));
}

int prepare_task(rk_npu_iommu_domain* domain, int M, int N, int K, rk_npu_mem* input,
                 rk_npu_mem* weight, rk_npu_mem* output,
                 bool a_native, bool c_native,
                 bool out_i32, rk_npu_mem* regcmd, rk_npu_mem* task) {
    if (rk_npu_mem_alloc(domain, 4096, RK_NPU_MEM_NON_CACHEABLE, regcmd) != RK_NPU_OK) return RK_NPU_ERR_NOMEM;
    if (rk_npu_mem_alloc(domain, sizeof(rknpu_task), RK_NPU_MEM_KERNEL_MAPPING, task) != RK_NPU_OK) return RK_NPU_ERR_NOMEM;

    std::vector<uint64_t> body;
    make_regs(body, M, N, K, input->dma_addr, weight->dma_addr, output->dma_addr,
              a_native, c_native, out_i32);

    std::vector<std::vector<uint64_t>> bodies;
    bodies.push_back(body);
    std::vector<int> base = {0};
    std::memset(task->vaddr, 0, sizeof(rknpu_task));
    rknpu2_matmul_open::detail::write_chain((uint64_t*)regcmd->vaddr, (rknpu_task*)task->vaddr,
                       regcmd->dma_addr, bodies, base, {(6u << 1) | 1u, 0u, 0xdu});
    return RK_NPU_OK;
}

} /* namespace */

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.M != 128 || a.K != 1024) {
        std::fprintf(stderr, "probe_ac_layout native-A recipe is currently constrained to M=128,K=1024\n");
        return 2;
    }

    const bool out_i32 = !a.fp32;
    Layout L = layout_for(a.N, a.K);
    rk_npu_matmul_i8_config pack_cfg{};
    rk_npu_matmul_i8_config_init(&pack_cfg, a.M, a.N, a.K);
    pack_cfg.out_dtype = out_i32 ? RK_NPU_I8_OUT_INT32 : RK_NPU_I8_OUT_FP32;
    rk_npu_matmul_sizes sz{};
    if (rknpu2_matmul_open::detail::query_i8(&pack_cfg, &sz) != RK_NPU_OK) return 2;

    std::mt19937 rng(1234);
    std::vector<int8_t> A((size_t)a.M * a.K), B((size_t)a.K * a.N);
    fill_i8(A, rng);
    fill_i8(B, rng);

    rk_npu_ctx* ctx = rk_npu_open(a.dev);
    if (!ctx) {
        std::fprintf(stderr, "rk_npu_open failed\n");
        return 1;
    }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }

    struct Combo { const char* name; bool a_native; bool c_native; };
    const Combo combos[] = {
        {"A=normal C=normal", false, false},
        {"A=native C=normal", true,  false},
        {"A=normal C=native", false, true},
        {"A=native C=native", true,  true},
    };

    std::printf("shape M=%d K=%d N=%d dtype=int8->%s loops=%d timeout=%dms\n",
                a.M, a.K, a.N, out_i32 ? "int32" : "fp32", a.loops, a.timeout_ms);
    std::printf("%-20s %10s %9s %8s %s\n", "combo", "run_us", "GOPS", "verify", "detail");

    for (const Combo& c : combos) {
        rk_npu_mem input{}, weight{}, output{}, regcmd{}, task{};
        auto free_all = [&] {
            if (regcmd.vaddr) rk_npu_mem_free(ctx, &regcmd);
            if (task.vaddr)   rk_npu_mem_free(ctx, &task);
            if (input.vaddr)  rk_npu_mem_free(ctx, &input);
            if (weight.vaddr) rk_npu_mem_free(ctx, &weight);
            if (output.vaddr) rk_npu_mem_free(ctx, &output);
        };

        int rc = rk_npu_mem_alloc(domain, sz.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &input);
        if (rc == RK_NPU_OK) rc = rk_npu_mem_alloc(domain, sz.weight_bytes, RK_NPU_MEM_NON_CACHEABLE, &weight);
        if (rc == RK_NPU_OK) rc = rk_npu_mem_alloc(domain, sz.output_bytes, RK_NPU_MEM_NON_CACHEABLE, &output);
        if (rc != RK_NPU_OK) {
            std::printf("%-20s %10s %9s %8s alloc\n", c.name, "-", "-", "NOMEM");
            free_all();
            continue;
        }

        if (c.a_native) pack_a_native(a.M, a.K, L.align_in, A.data(), &input);
        else            pack_a_normal(a.M, a.K, L.align_in, A.data(), &input);
        rknpu2_matmul_open::detail::pack_i8_b(&pack_cfg, B.data(), &weight);

        rc = prepare_task(domain, a.M, a.N, a.K, &input, &weight, &output,
                          c.a_native, c.c_native, out_i32, &regcmd, &task);
        if (rc != RK_NPU_OK) {
            std::printf("%-20s %10s %9s %8s prepare\n", c.name, "-", "-", "NOMEM");
            free_all();
            continue;
        }

        std::memset(output.vaddr, 0, output.size);
        rc = rknpu2_matmul_open::detail::do_submit(ctx->fd, task.obj_addr, 1, 0,
                              (uint32_t)a.timeout_ms);
        if (rc != RK_NPU_OK) {
            std::printf("%-20s %10s %9s %8s submit\n", c.name, "-", "-", "SUBMIT");
            free_all();
            continue;
        }

        std::vector<float> C((size_t)a.M * a.N);
        if (c.c_native) unpack_c_native(a.M, a.N, &output, out_i32, C);
        else            unpack_c_normal(a.M, a.N, L.align_out, &output, out_i32, C);

        std::mt19937 vrng(5678);
        Verify v = verify(a.M, a.N, a.K, A.data(), B.data(), C.data(), 40000000ull, 4096, vrng);
        if (!v.ok) {
            std::printf("%-20s %10s %9s %8s max_abs=%.0f checked=%zu%s%s\n",
                        c.name, "-", "-", "WRONG", v.max_abs, v.checked,
                        v.sampled ? " sampled" : "", v.bad_value ? " bad" : "");
            free_all();
            continue;
        }

        {
            auto pp = timeit([&]{
                std::memset(output.vaddr, 0, output.size);
                rknpu2_matmul_open::detail::do_submit(ctx->fd, task.obj_addr, 1, 0,
                                 (uint32_t)a.timeout_ms);
            }, a.loops);
            double gops = 2.0 * a.M * a.N * a.K / (pp.first * 1e-6) / 1e9;
            std::printf("%-20s %10.2f %9.1f %8s checked=%zu%s\n",
                        c.name, pp.first, gops, "PASS", v.checked,
                        v.sampled ? " sampled" : "");
        }
        free_all();
    }

    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);
    return 0;
}

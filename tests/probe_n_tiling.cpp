/*
 * probe_n_tiling.cpp - probe whether N tiles can write directly into one
 * full-N packed output buffer.
 *
 * This is intentionally a low-level probe, not a public API test.  It builds
 * two N=256 int8 matmul tasks for a full N=512 result and tries several DPU
 * output-stride register variants.  PASS for a full-stride variant would mean
 * N-tiling can likely keep the existing output layout and avoid a CPU tile
 * scatter.  If only the compact-tile reference passes, N-tiling needs either a
 * new tiled output layout or more register work.
 */
#include "rk_npu_matmul.h"
#include "../src/rk_npu_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int M = 2;
constexpr int N_FULL = 512;
constexpr int N_TILE = 256;
constexpr int K = 64;
constexpr int NUM_TILES = N_FULL / N_TILE;
constexpr int BODY_QWORDS = 45;

struct Layout { int align_in, align_out; };

Layout layout_for(int N, int K_) {
    return { std::max(MIN_CHANNEL_TILE, align_up(K_, MIN_CHANNEL_TILE)),
             std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE)) };
}

void fill_inputs(std::vector<int8_t>& A, std::vector<int8_t>& B) {
    for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)((int)(i % 17) - 8);
    for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)((int)(i % 29) - 14);
}

void cpu_ref(const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<float>& C) {
    C.assign((size_t)M * N_FULL, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            int av = A[(size_t)m * K + k];
            for (int n = 0; n < N_FULL; ++n)
                C[(size_t)m * N_FULL + n] += (float)(av * (int)B[(size_t)k * N_FULL + n]);
        }
    }
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs((double)a[i] - (double)b[i]));
    return d;
}

uint64_t reg_word(uint64_t target, uint32_t reg, uint32_t value) {
    return (target << 48) | ((uint64_t)value << 16) | reg;
}

void set_reg(std::vector<uint64_t>& regs, uint32_t reg, uint32_t value) {
    for (uint64_t& w : regs) {
        if ((uint32_t)(w & 0xffff) == reg) {
            w = (w & 0xffff000000000000ull) | ((uint64_t)value << 16) | reg;
            return;
        }
    }
}

int notch_for(int align_out) {
    int groups = std::min(align_out / MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP);
    return 8 * groups - 1;
}

void make_tile_regs(std::vector<uint64_t>& v, uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma) {
    Layout L = layout_for(N_TILE, K);
    const int align_in = L.align_in, align_out = L.align_out;
    const int feature_grains = M + 1;
    const int data_banks = std::min(std::max(ceil_div(M * align_in, CBUF_BANK_SIZE), 1), RK_CBUF_BANKS - 1);
    const int line_stride = std::max(1, align_up(K, MIN_CHANNEL_TILE) / 16);
    const int notch_val = notch_for(align_out);

    v.clear();
    v.push_back(E(T_DPU,  R_S_POINTER, (1<<3)|(1<<2)|(1<<1)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON1, (1u<<29)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON2, (feature_grains<<4)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON3, (1<<3)|1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE0, (1u<<16)|M));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE1, ((uint32_t)(align_in-1)<<16)|align_in));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE2, 1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE3, M));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE0, (uint32_t)align_in*align_out));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE1, align_in));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE2, (1u<<24)|(1u<<16)|align_out));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON0, ((uint32_t)(RK_CBUF_BANKS-data_banks)<<4)|data_banks));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON1, ceil_div(align_in, 64)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON0, (1<<3)|(1<<1)|1));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON1, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON2, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON3, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON4, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_dma));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON0, (15u<<16)|15));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON1, line_stride));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON2, 0));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE0, (1u<<16)|M));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE1, align_in));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_ADDR0, (uint32_t)wt_dma));
    v.push_back(E(T_CORE, R_CORE_MISC_CFG, 1));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_0, ((uint32_t)(M-1)<<16)|0));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_1, align_out-1));
    v.push_back(E(T_CORE, R_CORE_RESERVED_3030, 0));
    v.push_back(E(T_DPU,  R_FEATURE_MODE_CFG, (15<<5)|(2<<1)));
    v.push_back(E(T_DPU,  R_DATA_FORMAT, (5u<<29)));
    v.push_back(E(T_DPU,  R_DST_BASE_ADDR, (uint32_t)out_dma));
    v.push_back(E(T_DPU,  R_DST_SURF_STRIDE, (1<<4)));
    v.push_back(E(T_DPU,  R_DATA_CUBE_WIDTH, 0));
    v.push_back(E(T_DPU,  R_DATA_CUBE_HEIGHT, M-1));
    v.push_back(E(T_DPU,  R_DATA_CUBE_NOTCH, ((uint32_t)notch_val<<16)|notch_val));
    v.push_back(E(T_DPU,  R_DATA_CUBE_CHANNEL, ((uint32_t)(align_out-1)<<16)|(align_out-1)));
    v.push_back(E(T_DPU,  R_BS_CFG, 0x53));
    v.push_back(E(T_DPU,  R_BS_OW_CFG, 0x7fc));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_0, align_out-1));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_1, ((uint32_t)(M-1)<<16)|0));
    v.push_back(E(T_DPU,  R_BN_CFG, 0x53));
    v.push_back(E(T_DPU,  R_EW_CFG, 0x383));
    v.push_back(E(T_DPU,  R_OUT_CVT_SCALE, 1));
    v.push_back(E(T_DPU,  R_SURFACE_ADD, 0x80));
}

enum class Variant {
    CompactReference,
    FullAddrOnly,
    FullNotch,
    FullSurfaceAdd,
    FullSurfStrideElems,
    FullSurfStrideBytes,
};

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::CompactReference:    return "compact_tile_reference";
        case Variant::FullAddrOnly:        return "full_addr_only";
        case Variant::FullNotch:           return "full_notch";
        case Variant::FullSurfaceAdd:      return "full_surface_add";
        case Variant::FullSurfStrideElems: return "full_surf_stride_elems";
        case Variant::FullSurfStrideBytes: return "full_surf_stride_bytes";
    }
    return "?";
}

void apply_variant(std::vector<uint64_t>& regs, Variant v) {
    const int full_align_out = layout_for(N_FULL, K).align_out;
    const int full_notch = notch_for(full_align_out);

    if (v == Variant::FullNotch || v == Variant::FullSurfaceAdd ||
        v == Variant::FullSurfStrideElems || v == Variant::FullSurfStrideBytes) {
        set_reg(regs, R_DATA_CUBE_NOTCH, ((uint32_t)full_notch << 16) | (uint32_t)full_notch);
    }

    if (v == Variant::FullSurfaceAdd) {
        /* Existing int8 recipe uses SURFACE_ADD=0x80, i.e. field value 8 for
         * align_out=256.  Try the analogous field value 16 for align_out=512. */
        set_reg(regs, R_SURFACE_ADD, (uint32_t)(full_align_out / 32) << 4);
    } else if (v == Variant::FullSurfStrideElems) {
        set_reg(regs, R_DST_SURF_STRIDE, (uint32_t)full_align_out << 4);
        set_reg(regs, R_SURFACE_ADD, (uint32_t)(full_align_out * 4) << 4);
    } else if (v == Variant::FullSurfStrideBytes) {
        set_reg(regs, R_DST_SURF_STRIDE, (uint32_t)(full_align_out * 4) << 4);
        set_reg(regs, R_SURFACE_ADD, (uint32_t)(full_align_out * 4) << 4);
    }
}

int run_variant(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                Variant variant, rk_npu_mem& in, rk_npu_mem& wt,
                rk_npu_mem& out, const std::vector<float>& ref, double& diff) {
    rk_npu_matmul_i8_config tile_cfg{};
    rk_npu_matmul_i8_config_init(&tile_cfg, M, N_TILE, K);
    rk_npu_matmul_i8_config full_cfg{};
    rk_npu_matmul_i8_config_init(&full_cfg, M, N_FULL, K);
    rk_npu_matmul_sizes tile_sz{};
    rknpu2_matmul_open::detail::query_i8(&tile_cfg, &tile_sz);
    const int per_task_qwords = align_up(BODY_QWORDS + 4, 2);

    rk_npu_mem regcmd{}, task{};
    int rc = rk_npu_mem_alloc(domain, (uint64_t)NUM_TILES * per_task_qwords * 8,
                              RK_NPU_MEM_NON_CACHEABLE, &regcmd);
    if (rc != RK_NPU_OK) return rc;
    rc = rk_npu_mem_alloc(domain, (uint64_t)NUM_TILES * sizeof(rknpu_task),
                          RK_NPU_MEM_KERNEL_MAPPING, &task);
    if (rc != RK_NPU_OK) {
        rk_npu_mem_free(ctx, &regcmd);
        return rc;
    }

    std::memset(out.vaddr, 0, out.size);
    std::vector<std::vector<uint64_t>> bodies(NUM_TILES);
    std::vector<int> base(NUM_TILES);
    for (int t = 0, off = 0; t < NUM_TILES; ++t) {
        const uint64_t wt_addr = wt.dma_addr + (uint64_t)t * tile_sz.weight_bytes;
        uint64_t out_addr = out.dma_addr;
        if (variant == Variant::CompactReference)
            out_addr += (uint64_t)t * tile_sz.output_bytes;
        else
            out_addr += (uint64_t)t * N_TILE * 4;  /* desired full output column offset */

        make_tile_regs(bodies[t], in.dma_addr, wt_addr, out_addr);
        apply_variant(bodies[t], variant);
        base[t] = off;
        off += per_task_qwords;
    }

    rknpu2_matmul_open::detail::write_chain((uint64_t*)regcmd.vaddr, (rknpu_task*)task.vaddr,
                       regcmd.dma_addr, bodies, base, { (6u<<1)|1u, 0u, 0xdu });
    rc = rknpu2_matmul_open::detail::do_submit(ctx->fd, task.obj_addr, NUM_TILES,
                          rk_npu_iommu_domain_id(domain), 500);
    if (rc != RK_NPU_OK) {
        rk_npu_mem_free(ctx, &task);
        rk_npu_mem_free(ctx, &regcmd);
        return rc;
    }

    std::vector<float> got((size_t)M * N_FULL);
    if (variant == Variant::CompactReference) {
        for (int t = 0; t < NUM_TILES; ++t) {
            rk_npu_mem tile_view = out;
            tile_view.vaddr = (void*)((uint8_t*)out.vaddr + (uint64_t)t * tile_sz.output_bytes);
            tile_view.dma_addr = out.dma_addr + (uint64_t)t * tile_sz.output_bytes;
            tile_view.size = tile_sz.output_bytes;
            std::vector<float> tmp((size_t)M * N_TILE);
            rknpu2_matmul_open::detail::unpack_i8_c(&tile_cfg, &tile_view, tmp.data());
            for (int m = 0; m < M; ++m)
                std::memcpy(got.data() + (size_t)m * N_FULL + t * N_TILE,
                            tmp.data() + (size_t)m * N_TILE,
                            (size_t)N_TILE * sizeof(float));
        }
    } else {
        rknpu2_matmul_open::detail::unpack_i8_c(&full_cfg, &out, got.data());
    }

    diff = max_abs_diff(got, ref);
    rk_npu_mem_free(ctx, &task);
    rk_npu_mem_free(ctx, &regcmd);
    return RK_NPU_OK;
}

} /* namespace */

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : nullptr;
    rk_npu_ctx* ctx = rk_npu_open(dev);
    if (!ctx) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
    if (!domain) { rk_npu_close(ctx); return 1; }

    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N_FULL);
    std::vector<float> ref;
    fill_inputs(A, B);
    cpu_ref(A, B, ref);

    rk_npu_matmul_i8_config full_cfg{};
    rk_npu_matmul_i8_config_init(&full_cfg, M, N_FULL, K);
    rk_npu_matmul_i8_config tile_cfg{};
    rk_npu_matmul_i8_config_init(&tile_cfg, M, N_TILE, K);
    rk_npu_matmul_sizes full_sz{}, tile_sz{};
    rknpu2_matmul_open::detail::query_i8(&full_cfg, &full_sz);
    rknpu2_matmul_open::detail::query_i8(&tile_cfg, &tile_sz);

    rk_npu_mem in{}, wt{}, out{};
    const uint64_t out_bytes = std::max(full_sz.output_bytes, tile_sz.output_bytes * (uint64_t)NUM_TILES);
    int rc = rk_npu_mem_alloc(domain, tile_sz.input_bytes, RK_NPU_MEM_NON_CACHEABLE, &in);
    rc |= rk_npu_mem_alloc(domain, tile_sz.weight_bytes * (uint64_t)NUM_TILES, RK_NPU_MEM_NON_CACHEABLE, &wt);
    rc |= rk_npu_mem_alloc(domain, out_bytes, RK_NPU_MEM_NON_CACHEABLE, &out);
    if (rc != RK_NPU_OK) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    rknpu2_matmul_open::detail::pack_i8_a(&tile_cfg, A.data(), &in);
    for (int t = 0; t < NUM_TILES; ++t) {
        std::vector<int8_t> Bt((size_t)K * N_TILE);
        for (int k = 0; k < K; ++k)
            std::memcpy(Bt.data() + (size_t)k * N_TILE,
                        B.data() + (size_t)k * N_FULL + t * N_TILE,
                        (size_t)N_TILE);
        rk_npu_mem wv = wt;
        wv.vaddr = (void*)((uint8_t*)wt.vaddr + (uint64_t)t * tile_sz.weight_bytes);
        wv.dma_addr = wt.dma_addr + (uint64_t)t * tile_sz.weight_bytes;
        wv.size = tile_sz.weight_bytes;
        rknpu2_matmul_open::detail::pack_i8_b(&tile_cfg, Bt.data(), &wv);
    }

    printf("== N-tiling full-stride output probe ==\n");
    printf("shape: M=%d K=%d N=%d split=%dx%d\n", M, K, N_FULL, NUM_TILES, N_TILE);

    bool any_full_pass = false;
    const Variant variants[] = {
        Variant::CompactReference,
        Variant::FullAddrOnly,
        Variant::FullNotch,
        Variant::FullSurfaceAdd,
        Variant::FullSurfStrideElems,
        Variant::FullSurfStrideBytes,
    };
    for (Variant v : variants) {
        double diff = -1.0;
        rc = run_variant(ctx, domain, v, in, wt, out, ref, diff);
        bool ok = (rc == RK_NPU_OK && diff == 0.0);
        if (v != Variant::CompactReference && ok) any_full_pass = true;
        printf("%-24s %s", variant_name(v), ok ? "PASS" : "FAIL");
        if (rc == RK_NPU_OK) printf(" max_abs=%g", diff);
        else printf(" rc=%s", rk_npu_strerror(rc));
        printf("\n");
    }

    rk_npu_mem_free(ctx, &out);
    rk_npu_mem_free(ctx, &wt);
    rk_npu_mem_free(ctx, &in);
    rk_npu_iommu_domain_free(domain);
    rk_npu_close(ctx);

    printf("full_stride_result: %s\n", any_full_pass ? "PASS" : "NO_PASS");
    return 0;
}

/*
 * rk_npu_matmul_f16.cpp - fp16 x fp16 -> fp16 matmul, optionally fused with an
 *                         elementwise MUL/ADD.
 * ============================================================================
 * The plain matmul is a verified port of allbilly's examples/gemm.py (fp16).
 * The fused elementwise path has hardware-validated operand layout and
 * conversion requirements, documented below and at the relevant registers.
 *
 *   matmul = a 1x1 convolution: feature A is H=M,W=1,C=K; weight B is N kernels
 *   of 1x1xK.  The CORE MACs accumulate in high precision; the DPU writes fp16.
 *
 * ---- Fused elementwise op (D = A@B  {*,+}  C0) --------------------------------
 * The DPU "EW" unit applies an elementwise op between the matmul result and a
 * second operand C0 that the DPU-RDMA engine streams from memory.  Three
 * requirements apply to this matmul geometry:
 *
 *  (1) DUAL-SURFACE operand read.  The operand must be fed through BOTH the
 *      MRDMA (SRC_BASE) and the ERDMA (EW_BASE) engines.  Feeding it through the
 *      ERDMA alone HANGS the W=1,H=M matmul cube: the EW
 *      pipeline stalls waiting for data and the DPU never raises its done IRQ.
 *      So: SRC_BASE = operand base, EW_BASE = operand base + one surface.
 *
 *  (2) ADD needs OP_CVT ACTIVE.  For MUL, EW_CFG=0x108003c4.  For ADD you must
 *      NOT set EW_OP_CVT_BYPASS (bit 8): EW_CFG=0x108202c0 with an identity
 *      operand-convert (EW_CVT_SCALE=1, EW_CVT_OFFSET=0).  With OP_CVT bypassed
 *      the ALU silently drops the operand (D == A@B, add ignored).
 *
 *  (3) The "half-surface" operand memory layout.  C0 is NOT row-major and NOT
 *      the same layout the DPU writes its output in.  See pack_operand() below.
 *      The surface strides also must be SURF_NOTCH = 2*M (not M) or alternate
 *      8-channel groups alias onto the same bytes.
 *
 * Device/memory/submit live in rk_npu_core.cpp.
 */
#include "rk_npu_matmul_f16.h"
#include "rk_npu_internal.h"
#include "rk_npu_cpu_kernels.h"

#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace {

/* fp16-specific CBUF budget / pipeline constants (from gemm.py) */
constexpr int FP16_BYTES               = 2;
constexpr int RK_MIN_WIDE_FEATURE_GRAINS = 80;   /* min pipeline depth to keep CSC->CMAC fed */
constexpr int GEMM_INPUT_BANKS         = RK_CBUF_BANKS - 2;
constexpr int GEMM_MAX_ALIGN_IN        = RK_CBUF_BANKS * MIN_CHANNEL_TILE;   /* 384 */
constexpr int FP16_LINE_STRIDE_GROUP_CAP = 40;
constexpr int FP16_NATIVE_A_DATA_BANKS = 4;
constexpr int FP16_NATIVE_M_TILE_MAX   = 1022;
constexpr int FP16_NATIVE_ROW_BYTES    = 16;

/* Fused EW_CFG settings; operand-conversion requirements are described above. */
constexpr uint32_t EW_CFG_BYPASS = 0x383;        /* EW fully bypassed (plain matmul)        */
constexpr uint32_t EW_CFG_MUL    = 0x108003c4;   /* OP_TYPE=mul, EDATA fp16, OP_CVT bypass  */
constexpr uint32_t EW_CFG_ADD    = 0x108202c0;   /* ALU add, EDATA fp16, OP_CVT ACTIVE      */
constexpr uint32_t ERDMA_CFG_FP16 = (1u<<30)|(2u<<2); /* DATA_MODE=1, DATA_SIZE=2(fp16)=0x40000008 */
constexpr uint32_t RDMA_FMC_FP16  = 0x17d40;     /* IN/PROC fp16, BURST_LEN=15, COMB_USE=5  */

struct Layout { int align_in, align_out, eff_k; };
/* NOTE the legacy fused-fp16 quirk: align_in is tied to
 * max(aligned_k, align_out), NOT just K.  The plain FC path keeps A/K compact. */
Layout gemm_layout(int N, int K, bool legacy_fused_layout) {
    int aligned_k = std::max(MIN_CHANNEL_TILE, align_up(K, MIN_CHANNEL_TILE));
    int align_out = std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE));
    int align_in  = legacy_fused_layout
                  ? std::max(aligned_k, align_out)
                  : aligned_k;
    int eff_k     = (align_in != aligned_k) ? align_in : K;
    return { align_in, align_out, eff_k };
}
int weight_banks_for(int input_row_bytes) {
    /* The FC path keeps one 32-output-channel weight atom in CBUF.  It uses
     * ceil(K * 2 * 32 / 32768) weight banks: K=1280 gets 3
     * weight + 9 data banks, while K=5120 gets 10 weight + 2 data banks. */
    const int64_t atom_bytes =
        (int64_t)input_row_bytes * MIN_CHANNEL_TILE;
    const int banks = (int)((atom_bytes + CBUF_BANK_SIZE - 1) /
                            CBUF_BANK_SIZE);
    /* Narrow K still needs two non-feature banks for the weight/pipeline
     * stream.  Letting A consume eleven banks makes the M=3000,K=384 Whisper
     * conv1 full tile submit-fail; the original gemm.py budget and hardware
     * both use at most ten data banks there. */
    return std::min(std::max(banks, 2), RK_CBUF_BANKS - 1);
}
int m_tile_for(const Layout& L, bool native_a) {
    int input_row_bytes = L.align_in * FP16_BYTES;
    if (native_a) {
        /* Native-A tasks budget at most four CBUF banks for A.
         * FP16's K/8 layout requires an even tile once two rows fit, and the
         * DATA_SIZE field tops out at 1022 rows: K=64 -> 1022, K=1280 -> 50,
         * K=5120 -> 12. */
        int rows = FP16_NATIVE_A_DATA_BANKS * CBUF_BANK_SIZE /
                   input_row_bytes;
        rows = std::min(rows, FP16_NATIVE_M_TILE_MAX);
        if (rows >= 2) rows &= ~1;
        return std::max(1, rows);
    }
    const int data_banks = RK_CBUF_BANKS - weight_banks_for(input_row_bytes);
    /* Reserve the weight banks first, then fill all remaining banks with A.
     * Unlike the old K>384 fallback, this remains useful at Whisper's large K:
     * K=1280 -> 115 rows/task and K=5120 -> 6 rows/task. */
    return std::max(1, data_banks * CBUF_BANK_SIZE / input_row_bytes);
}

bool a_native(const rk_npu_matmul_f16_config& cfg) {
    return cfg.a_layout == RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;
}

bool d_native(const rk_npu_matmul_f16_config& cfg) {
    return cfg.d_layout == RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
}

/* number of EW operand fp16 elements one task (tile_m rows) occupies (see pack_operand) */
inline int operand_block_elems(int align_out, int tile_m) {
    return (align_out / FEATURE_ATOMIC) * 24 * tile_m;   /* 24*m per 16-channel surface */
}

bool valid_config(const rk_npu_matmul_f16_config* cfg) {
    if (!cfg || cfg->M <= 0 || cfg->N <= 0 || cfg->K <= 0 ||
        (cfg->op != RK_NPU_FUSE_NONE &&
         cfg->op != RK_NPU_FUSE_MUL &&
         cfg->op != RK_NPU_FUSE_ADD) ||
        (cfg->a_layout != RK_NPU_F16_A_LAYOUT_NORMAL &&
         cfg->a_layout != RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8) ||
        (cfg->d_layout != RK_NPU_F16_D_LAYOUT_NORMAL_PADDED &&
         cfg->d_layout != RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8) ||
        (cfg->op != RK_NPU_FUSE_NONE &&
         (a_native(*cfg) || d_native(*cfg))))
        return false;
    if (cfg->n_tile < 0 || (cfg->n_tile > 0 &&
        ((cfg->n_tile % MIN_CHANNEL_TILE) != 0 ||
         cfg->n_tile > align_up(cfg->N, MIN_CHANNEL_TILE) ||
         cfg->op != RK_NPU_FUSE_NONE)))
        return false;
    const uint32_t mask = cfg->core_mask == 0 ? 1u : cfg->core_mask;
    return mask == 1u || mask == 2u || mask == 3u ||
           mask == 4u || mask == 7u;
}

rk_npu_matmul_f16_config default_config(int M, int N, int K, rk_npu_fuse_op op) {
    rk_npu_matmul_f16_config cfg{};
    cfg.M = M;
    cfg.N = N;
    cfg.K = K;
    cfg.op = op;
    cfg.a_layout = RK_NPU_F16_A_LAYOUT_NORMAL;
    cfg.d_layout = RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;
    cfg.n_tile = 0;
    cfg.core_mask = 1;
    cfg.timeout_ms = 6000;
    return cfg;
}

int config_n_tile(const rk_npu_matmul_f16_config& cfg) {
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg.N, MIN_CHANNEL_TILE));
    return cfg.n_tile > 0 ? std::min(cfg.n_tile, align_out) : align_out;
}

bool splitk_part_config(int split_count,
                        const rk_npu_matmul_f16_config* cfg,
                        rk_npu_matmul_f16_config* part_cfg) {
    if (!valid_config(cfg) || !part_cfg || split_count < 2 || split_count > 3 ||
        cfg->op != RK_NPU_FUSE_NONE ||
        cfg->a_layout != RK_NPU_F16_A_LAYOUT_NORMAL ||
        cfg->d_layout != RK_NPU_F16_D_LAYOUT_NORMAL_PADDED ||
        cfg->n_tile != 0 || cfg->K < split_count * MIN_CHANNEL_TILE)
        return false;
    *part_cfg = *cfg;
    part_cfg->K = align_up(ceil_div(cfg->K, split_count), MIN_CHANNEL_TILE);
    return true;
}

uint32_t effective_core_mask(uint32_t requested, int num_tasks) {
    uint32_t mask = requested == 0 ? 1u : requested;
    int cores = __builtin_popcount(mask & 7u);
    while (cores > num_tasks) {
        if (cores == 3) mask = 3u;
        else if (cores == 2) mask = 1u;
        cores = __builtin_popcount(mask & 7u);
    }
    return mask;
}

/*
 * One task's register body.  The plain FP16 FC path performs register
 * initialization (108 writes, including state-clearing writes required
 * by large K).  The legacy MUL/ADD path additionally programs the EW convert
 * registers and the whole DPU-RDMA operand-read block (see the dual-surface note
 * up top).
 */
void make_f16_regs(std::vector<uint64_t>& v, int m, int full_M, int N, int K,
                   uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma,
                   rk_npu_fuse_op op, uint64_t operand_dma,
                   int plan_data_banks, bool native_a, bool native_d) {
    Layout L = gemm_layout(N, K, op != RK_NPU_FUSE_NONE);
    const int align_in = L.align_in, align_out = L.align_out, eff_k = L.eff_k;
    const int input_row_bytes = align_in * FP16_BYTES;
    const bool fused = (op != RK_NPU_FUSE_NONE);
    native_a = native_a && !fused;
    native_d = native_d && !fused;

    /* output is fp16: OUT_PRECISION=2, element-size code size_e=1 */
    const uint32_t out_precision = 2, size_e = 1;

    int even_rows_per_two_banks = (ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1;
    /* m+1 is the hardware-validated conservative FC grain count. Too-small
     * values corrupt some intermediate large-K ranges (notably K=3840). */
    int feature_grains = fused
        ? std::max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)
        : (native_a ? 2 : m + 1);
    /* Keep the plan's first/full-tile partition on residual M tiles too. */
    int data_banks = native_a
        ? std::min(std::max(ceil_div(m * input_row_bytes, CBUF_BANK_SIZE), 1),
                   RK_CBUF_BANKS - 1)
        : plan_data_banks;
    int line_stride = fused
        ? 4 * std::min(ceil_div(eff_k, MIN_CHANNEL_TILE),
                       RK_LINE_STRIDE_GROUP_CAP)
        : 4 * ceil_div(eff_k, MIN_CHANNEL_TILE);
    int notch_val = fused
        ? 8 * std::min(align_out / MIN_CHANNEL_TILE,
                       RK_LINE_STRIDE_GROUP_CAP) - 1
        : align_out / 8 - 1;

    const bool native_a_split_m = native_a && m < full_M;
    const uint32_t cna_conv1 = (2u << 4) | (2u << 7) |
        ((!native_a || native_a_split_m) ? (1u << 29) : 0u);
    const uint32_t data_size0 = native_a
        ? ((uint32_t)m << 16) | 1u
        : (1u << 16) | (uint32_t)m;
    const uint32_t data_size2 = native_a ? (uint32_t)m : 1u;
    const uint32_t cbuf1 = native_a
        ? (uint32_t)ceil_div(m * align_in, MIN_CHANNEL_TILE)
        : (uint32_t)ceil_div(align_in, MIN_CHANNEL_TILE);
    const uint32_t dma_con1 = native_a
        ? (native_a_split_m ? (uint32_t)full_M : 0x200u)
        : (uint32_t)line_stride;
    const uint32_t dma_con2 = native_a
        ? (native_a_split_m ? (uint32_t)(full_M - m)
                            : (0x10000000u - (uint32_t)full_M * 3u))
        : 0u;
    const uint32_t core_size0 = native_a
        ? (uint32_t)(m - 1)
        : ((uint32_t)(m - 1) << 16);

    const uint32_t dst_surf_stride = native_d
        ? (uint32_t)(full_M * FP16_NATIVE_ROW_BYTES)
        : (1u << 4);
    const uint32_t dst_w = native_d ? (uint32_t)(m - 1) : 0u;
    const uint32_t dst_h = native_d ? 0u : (uint32_t)(m - 1);
    const uint32_t dst_notch = native_d
        ? 0u
        : ((uint32_t)notch_val << 16) | (uint32_t)notch_val;
    const uint32_t wdma_size1 = native_d
        ? (uint32_t)(m - 1)
        : ((uint32_t)(m - 1) << 16);
    const uint32_t surface_add = native_d
        ? ((uint32_t)full_M << 5)
        : (2u << 4);

    const uint32_t ew_cfg = (op == RK_NPU_FUSE_MUL) ? EW_CFG_MUL
                          : (op == RK_NPU_FUSE_ADD) ? EW_CFG_ADD : EW_CFG_BYPASS;

    if (!fused) {
        /* Preserve this initialization order. Several of these writes
         * look redundant, but omitting the CBUF/DCOMP preamble or stale-state
         * clears makes the large-K path depend on prior NPU register state. */
        v.clear();
        v.reserve(108);
        v.push_back(E(T_CNA,  R_CNA_CBUF_CON0, ((uint32_t)(RK_CBUF_BANKS-data_banks)<<4)|data_banks));
        v.push_back(E(T_CNA,  R_CNA_DCOMP_REGNUM, 0));
        v.push_back(E(T_CNA,  R_CNA_DCOMP_CTRL, 0));
        v.push_back(E(T_CNA,  R_CNA_CONV_CON1, cna_conv1));
        v.push_back(E(T_DPU,  R_S_POINTER, (1<<3)|(1<<2)|(1<<1)));
        v.push_back(E(T_CNA,  R_CNA_CONV_CON1, cna_conv1));
        v.push_back(E(T_CNA,  R_CNA_CONV_CON2, (feature_grains<<4)));
        v.push_back(E(T_CNA,  R_CNA_CONV_CON3, (1<<3)|1));
        v.push_back(E(T_CNA,  R_CNA_DATA_SIZE0, data_size0));
        v.push_back(E(T_CNA,  R_CNA_DATA_SIZE1, ((uint32_t)(align_in-1)<<16)|align_in));
        v.push_back(E(T_CNA,  R_CNA_DATA_SIZE2, data_size2));
        v.push_back(E(T_CNA,  R_CNA_DATA_SIZE3, m));
        v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE0, (uint32_t)input_row_bytes*align_out));
        v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE1, input_row_bytes));
        v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE2, (1u<<24)|(1u<<16)|align_out));
        v.push_back(E(T_CNA,  R_CNA_CBUF_CON0, ((uint32_t)(RK_CBUF_BANKS-data_banks)<<4)|data_banks));
        v.push_back(E(T_CNA,  R_CNA_CBUF_CON1, cbuf1));
        v.push_back(E(T_CNA,  R_CNA_CVT_CON0, (1<<3)|(1<<1)|1));
        v.push_back(E(T_CNA,  R_CNA_CVT_CON1, (1u<<16)));
        v.push_back(E(T_CNA,  R_CNA_CVT_CON2, (1u<<16)));
        v.push_back(E(T_CNA,  R_CNA_CVT_CON3, (1u<<16)));
        v.push_back(E(T_CNA,  R_CNA_CVT_CON4, (1u<<16)));
        v.push_back(E(T_CNA,  0x1060, 0));
        v.push_back(E(T_CNA,  0x1064, 0));
        v.push_back(E(T_CNA,  0x1068, 0));
        v.push_back(E(T_CNA,  R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_dma));
        v.push_back(E(T_CNA,  0x1074, 0));
        v.push_back(E(T_CNA,  R_CNA_DMA_CON0, (15u<<16)|15));
        v.push_back(E(T_CNA,  R_CNA_DMA_CON1, dma_con1));
        v.push_back(E(T_CNA,  R_CNA_DMA_CON2, dma_con2));
        v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE0, data_size0));
        v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE1, align_in));
        v.push_back(E(T_CNA,  R_CNA_DCOMP_CTRL, 0));
        v.push_back(E(T_CNA,  R_CNA_DCOMP_REGNUM, 0));
        v.push_back(E(T_CNA,  R_CNA_DCOMP_ADDR0, (uint32_t)wt_dma));
        for (uint32_t reg = 0x1140; reg < 0x1180; reg += 4)
            v.push_back(E(T_CNA, reg, 0));
        v.push_back(E(T_CNA,  0x1180, 0));
        v.push_back(E(T_CNA,  0x1184, 0));
        v.push_back(E(T_CORE, R_CORE_MISC_CFG, (2u<<8)));
        v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_0, core_size0));
        v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_1, align_out-1));
        v.push_back(E(T_CORE, 0x301c, 0));
        v.push_back(E(T_CORE, R_CORE_RESERVED_3030, 0));
        v.push_back(E(T_DPU,  R_FEATURE_MODE_CFG, (15<<5)|(2<<1)));
        v.push_back(E(T_DPU,  R_DATA_FORMAT, (out_precision<<29)|(2u<<26)|2u));
        v.push_back(E(T_DPU,  0x4014, 0));
        v.push_back(E(T_DPU,  R_DST_BASE_ADDR, (uint32_t)out_dma));
        v.push_back(E(T_DPU,  R_DST_SURF_STRIDE, dst_surf_stride));
        v.push_back(E(T_DPU,  R_DATA_CUBE_WIDTH, dst_w));
        v.push_back(E(T_DPU,  R_DATA_CUBE_HEIGHT, dst_h));
        v.push_back(E(T_DPU,  R_DATA_CUBE_NOTCH, dst_notch));
        v.push_back(E(T_DPU,  R_DATA_CUBE_CHANNEL, ((uint32_t)(align_out-1)<<16)|(align_out-1)));
        v.push_back(E(T_DPU,  R_BS_CFG, 0x53));
        v.push_back(E(T_DPU,  0x4044, 0));
        v.push_back(E(T_DPU,  0x4048, 0));
        v.push_back(E(T_DPU,  0x404c, 0));
        v.push_back(E(T_DPU,  R_BS_OW_CFG, (size_e<<8)|(size_e<<5)|(size_e<<2)|(1u<<1)));
        v.push_back(E(T_DPU,  0x4054, 0));
        v.push_back(E(T_DPU,  R_WDMA_SIZE_0, align_out-1));
        v.push_back(E(T_DPU,  R_WDMA_SIZE_1, wdma_size1));
        v.push_back(E(T_DPU,  R_BN_CFG, 0x53));
        v.push_back(E(T_DPU,  0x4064, 0));
        v.push_back(E(T_DPU,  0x4068, 0));
        v.push_back(E(T_DPU,  0x406c, 0));
        v.push_back(E(T_DPU,  R_EW_CFG, EW_CFG_BYPASS));
        v.push_back(E(T_DPU,  R_EW_CVT_OFFSET, 0));
        v.push_back(E(T_DPU,  R_EW_CVT_SCALE, 1));
        v.push_back(E(T_DPU,  R_EW_RELUX, 0));
        v.push_back(E(T_DPU,  0x4080, 0));
        v.push_back(E(T_DPU,  R_OUT_CVT_SCALE, (1u<<16)|1));
        v.push_back(E(T_DPU,  0x4088, 0));
        for (uint32_t reg = 0x4090; reg <= 0x40ac; reg += 4)
            v.push_back(E(T_DPU, reg, 0));
        v.push_back(E(T_DPU,  R_SURFACE_ADD, surface_add));
        v.push_back(E(T_DPU,  0x40c4, 0));
        for (uint32_t reg = 0x4100; reg <= 0x412c; reg += 4)
            v.push_back(E(T_DPU, reg, 0));
        return;
    }

    v.clear();
    // A preceding INT8 task may have enabled inline weight decompression.
    v.push_back(E(T_CNA, R_CNA_DCOMP_REGNUM, 0));
    v.push_back(E(T_CNA, R_CNA_DCOMP_CTRL, 0));
    v.push_back(E(T_DPU,  R_S_POINTER, (1<<3)|(1<<2)|(1<<1)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON1, (2u<<4)|(2u<<7)|(1u<<29)));     /* fp16 in/proc precision */
    v.push_back(E(T_CNA,  R_CNA_CONV_CON2, (feature_grains<<4)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON3, (1<<3)|1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE0, (1u<<16)|m));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE1, ((uint32_t)(align_in-1)<<16)|align_in));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE2, 1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE3, m));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE0, (uint32_t)input_row_bytes*align_out));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE1, input_row_bytes));
    v.push_back(E(T_CNA,  R_CNA_WEIGHT_SIZE2, (1u<<24)|(1u<<16)|align_out));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON0, ((uint32_t)(RK_CBUF_BANKS-data_banks)<<4)|data_banks));
    v.push_back(E(T_CNA,  R_CNA_CBUF_CON1, ceil_div(align_in, MIN_CHANNEL_TILE)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON0, (1<<3)|(1<<1)|1));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON1, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON2, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON3, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_CVT_CON4, (1u<<16)));
    v.push_back(E(T_CNA,  R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_dma));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON0, (15u<<16)|15));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON1, line_stride));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON2, 0));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE0, (1u<<16)|m));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE1, align_in));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_ADDR0, (uint32_t)wt_dma));
    v.push_back(E(T_CORE, R_CORE_MISC_CFG, (2u<<8)|(fused ? 1u : 0u)));    /* fp16 proc precision */
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_0, ((uint32_t)(m-1)<<16)|0));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_1, align_out-1));
    v.push_back(E(T_CORE, R_CORE_RESERVED_3030, 0));
    v.push_back(E(T_DPU,  R_FEATURE_MODE_CFG, (15<<5)|(2<<1)));
    v.push_back(E(T_DPU,  R_DATA_FORMAT, (out_precision<<29)|(2u<<26)|2u));/* OUT fp16, IN/PROC fp16 */
    v.push_back(E(T_DPU,  R_DST_BASE_ADDR, (uint32_t)out_dma));
    v.push_back(E(T_DPU,  R_DST_SURF_STRIDE, (1<<4)));
    v.push_back(E(T_DPU,  R_DATA_CUBE_WIDTH, 0));
    v.push_back(E(T_DPU,  R_DATA_CUBE_HEIGHT, m-1));
    v.push_back(E(T_DPU,  R_DATA_CUBE_NOTCH, ((uint32_t)notch_val<<16)|notch_val));
    v.push_back(E(T_DPU,  R_DATA_CUBE_CHANNEL, ((uint32_t)(align_out-1)<<16)|(align_out-1)));
    v.push_back(E(T_DPU,  R_BS_CFG, 0x53));
    v.push_back(E(T_DPU,  R_BS_OW_CFG, (size_e<<8)|(size_e<<5)|(size_e<<2)|(1u<<1)));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_0, align_out-1));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_1, ((uint32_t)(m-1)<<16)|0));
    v.push_back(E(T_DPU,  R_BN_CFG, 0x53));
    v.push_back(E(T_DPU,  R_EW_CFG, ew_cfg));
    if (fused) {
        /* identity operand convert: (operand - 0) * 1.  Required even for fp16. */
        v.push_back(E(T_DPU, R_EW_CVT_OFFSET, 0));
        v.push_back(E(T_DPU, R_EW_CVT_SCALE, 1));
        v.push_back(E(T_DPU, R_EW_RELUX, 0));
    }
    v.push_back(E(T_DPU,  R_OUT_CVT_SCALE, (1u<<16)|1));                   /* FP32->FP16 enable */
    v.push_back(E(T_DPU,  R_SURFACE_ADD, (fused ? 4u : 2u)<<4));

    if (fused) {
        /* DPU-RDMA operand-read block (dual surface; see file header notes 1 & 3). */
        const uint32_t WH = (uint32_t)m;                  /* cube W*H (W=1, H=m)            */
        const uint64_t ew_base = operand_dma + (uint64_t)m * FEATURE_ATOMIC * FP16_BYTES; /* +1 surface */
        v.push_back(E(T_RDMA, R_RDMA_S_POINTER, 0xe));
        v.push_back(E(T_RDMA, R_RDMA_W, 0));                               /* W-1 = 0        */
        v.push_back(E(T_RDMA, R_RDMA_H, m-1));
        v.push_back(E(T_RDMA, R_RDMA_C, align_out-1));
        v.push_back(E(T_RDMA, R_RDMA_SRC_BASE, (uint32_t)operand_dma));    /* MRDMA: surface 0 */
        v.push_back(E(T_RDMA, R_RDMA_BRDMA_CFG, 0));
        v.push_back(E(T_RDMA, R_RDMA_BS_BASE, 0));
        v.push_back(E(T_RDMA, R_RDMA_NRDMA_CFG, 0));
        v.push_back(E(T_RDMA, R_RDMA_BN_BASE, 0));
        v.push_back(E(T_RDMA, R_RDMA_ERDMA_CFG, ERDMA_CFG_FP16));
        v.push_back(E(T_RDMA, R_RDMA_EW_BASE, (uint32_t)ew_base));         /* ERDMA: surface 1+ */
        v.push_back(E(T_RDMA, R_RDMA_EW_SURF_STRIDE, WH<<4));
        v.push_back(E(T_RDMA, R_RDMA_FEATURE_MODE_CFG, RDMA_FMC_FP16));
        v.push_back(E(T_RDMA, R_RDMA_SRC_DMA_CFG, 0));
        v.push_back(E(T_RDMA, R_RDMA_SURF_NOTCH, (2u*WH)<<4));             /* 2*m: kill aliasing */
        v.push_back(E(T_RDMA, R_RDMA_PAD_CFG, 0));
        v.push_back(E(T_RDMA, R_RDMA_WEIGHT, 0x01010101));
        v.push_back(E(T_RDMA, R_RDMA_EW_SURF_NOTCH, (2u*WH)<<4));
    }
}

/* tiling bookkeeping for one run */
struct BuildMeta {
    int num_tasks, input_row_bytes, row_stride_bytes, align_out;
    uint64_t output_bytes;
    std::vector<int> base, body_size, start, tile;
    std::vector<uint64_t> input_off, weight_off, output_off, operand_off;
};

int build_regcmd(const rk_npu_matmul_f16_config& cfg,
                 uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma, uint64_t operand_dma,
                 rk_npu_mem* regcmd, rk_npu_mem* task, BuildMeta* meta,
                 int batch_count = 1) {
    if (batch_count <= 0) return RK_NPU_ERR_PARAM;
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    const rk_npu_fuse_op op = cfg.op;
    const bool native_a = a_native(cfg);
    const bool native_d = d_native(cfg);
    const int output_row_bytes_per_channel =
        op == RK_NPU_FUSE_NONE ? FP16_BYTES : 2 * FP16_BYTES;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(N, MIN_CHANNEL_TILE));
    const int n_tile = config_n_tile(cfg);
    const Layout input_layout = gemm_layout(
        std::min(n_tile, align_out), K, op != RK_NPU_FUSE_NONE);
    const int m_tile = m_tile_for(input_layout, native_a);
    const int plan_tile_m = std::min(M, m_tile);
    const int plan_data_banks = std::min(
        std::max(ceil_div(plan_tile_m * input_layout.align_in * FP16_BYTES,
                          CBUF_BANK_SIZE), 1),
        RK_CBUF_BANKS - 1);
    const int m_tasks = ceil_div(M, m_tile);
    const int n_tasks = ceil_div(align_out, n_tile);
    meta->align_out = align_out;
    meta->input_row_bytes = input_layout.align_in * FP16_BYTES;
    meta->row_stride_bytes = native_d
        ? FP16_NATIVE_ROW_BYTES
        : align_out * output_row_bytes_per_channel;
    meta->num_tasks = batch_count * n_tasks * m_tasks;

    rk_npu_matmul_sizes one{};
    if (rk_npu_matmul_f16_query(&cfg, &one) != RK_NPU_OK)
        return RK_NPU_ERR_PARAM;
    meta->output_bytes = one.output_bytes * (uint64_t)batch_count;
    if (regcmd->size < one.regcmd_bytes * (uint64_t)batch_count ||
        task->size < one.task_bytes * (uint64_t)batch_count)
        return RK_NPU_ERR_NOMEM;

    uint64_t* cmd  = (uint64_t*)regcmd->vaddr;
    rknpu_task* tk = (rknpu_task*)task->vaddr;
    std::memset(tk, 0, (size_t)meta->num_tasks * sizeof(rknpu_task));

    meta->base.resize(meta->num_tasks);
    meta->body_size.resize(meta->num_tasks);
    meta->start.resize(meta->num_tasks);
    meta->tile.resize(meta->num_tasks);
    meta->input_off.resize(meta->num_tasks);
    meta->weight_off.resize(meta->num_tasks);
    meta->output_off.resize(meta->num_tasks);
    meta->operand_off.resize(meta->num_tasks);
    std::vector<std::vector<uint64_t>> bodies(meta->num_tasks);
    int off = 0, ti = 0;
    for (int b = 0; b < batch_count; ++b) {
        const uint64_t batch_input_off = (uint64_t)b * one.input_bytes;
        const uint64_t batch_weight_off = (uint64_t)b * one.weight_bytes;
        const uint64_t batch_output_off = (uint64_t)b * one.output_bytes;
        const uint64_t batch_operand_off = (uint64_t)b * one.operand_bytes;
        uint64_t tile_output_off = 0;
        for (int n_start = 0; n_start < align_out; n_start += n_tile) {
            const int tile_n = std::min(n_tile, align_out - n_start);
            const Layout tile_layout = gemm_layout(
                tile_n, K, op != RK_NPU_FUSE_NONE);
            const uint64_t tile_weight_off =
                (uint64_t)n_start * input_layout.align_in * FP16_BYTES;
            uint64_t tile_operand_off = 0;
            for (int start = 0; start < M; start += m_tile, ++ti) {
                const int tile_m = std::min(m_tile, M - start);
                meta->input_off[ti] = batch_input_off +
                    (uint64_t)start * (native_a ? FP16_NATIVE_ROW_BYTES
                                                : meta->input_row_bytes);
                meta->weight_off[ti] = batch_weight_off + tile_weight_off;
                meta->output_off[ti] = batch_output_off + tile_output_off +
                    (uint64_t)start * (native_d
                        ? FP16_NATIVE_ROW_BYTES
                        : tile_layout.align_out * output_row_bytes_per_channel);
                meta->operand_off[ti] = batch_operand_off + tile_operand_off;
                make_f16_regs(bodies[ti], tile_m, M, tile_n, K,
                              in_dma + meta->input_off[ti],
                              wt_dma + meta->weight_off[ti],
                              out_dma + meta->output_off[ti],
                              op, operand_dma + meta->operand_off[ti],
                              plan_data_banks, native_a, native_d);
                meta->base[ti] = off;
                meta->body_size[ti] = (int)bodies[ti].size();
                meta->start[ti] = start;
                meta->tile[ti] = tile_m;
                off += align_up((int)bodies[ti].size() + 4, 2);
                tile_operand_off +=
                    (uint64_t)operand_block_elems(tile_layout.align_out, tile_m) *
                    FP16_BYTES;
            }
            tile_output_off +=
                (uint64_t)M * tile_layout.align_out *
                output_row_bytes_per_channel;
        }
    }
    /* fused ops enable the DPU-RDMA unit (0x1d); plain matmul uses 0xd. */
    rknpu2_matmul_open::detail::ChainCfg chain_cfg = (op == RK_NPU_FUSE_NONE) ? rknpu2_matmul_open::detail::ChainCfg{ (6u<<1)|1u, 0u, 0xdu }
                                                         : rknpu2_matmul_open::detail::ChainCfg{ 0x1du, 1u, 0x1du };
    rknpu2_matmul_open::detail::write_chain(cmd, tk, regcmd->dma_addr, bodies, meta->base, chain_cfg);
    return RK_NPU_OK;
}

} /* anonymous namespace */

/* -------------------------------------------------------------- query ---- */

extern "C" void rk_npu_matmul_f16_config_init(rk_npu_matmul_f16_config* cfg,
                                              int M, int N, int K, rk_npu_fuse_op op) {
    if (!cfg) return;
    *cfg = default_config(M, N, K, op);
}

extern "C" int rk_npu_matmul_f16_query(const rk_npu_matmul_f16_config* cfg,
                                       rk_npu_matmul_sizes* out) {
    if (!valid_config(cfg) || !out) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, N = cfg->N, K = cfg->K;
    const rk_npu_fuse_op op = cfg->op;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(N, MIN_CHANNEL_TILE));
    const int n_tile = config_n_tile(*cfg);
    Layout L = gemm_layout(std::min(n_tile, align_out), K,
                           op != RK_NPU_FUSE_NONE);
    int m_tile = m_tile_for(L, a_native(*cfg));
    int num_tasks = ceil_div(M, m_tile) * ceil_div(align_out, n_tile);

    /* body = 108 plain-path regs; legacy fused adds EW + DPU-RDMA regs. */
    const int body = (op == RK_NPU_FUSE_NONE) ? 108 : 45 + 3 + 18 + 2;
    const int per_task_qwords = align_up(body + 4, 2);

    out->input_bytes   = (uint64_t)M * L.align_in * FP16_BYTES;
    out->weight_bytes  = (uint64_t)align_out * L.align_in * FP16_BYTES;
    out->operand_bytes = (op == RK_NPU_FUSE_NONE)
                           ? 0
                           : (uint64_t)operand_block_elems(align_out, M) * FP16_BYTES;
    const int output_row_bytes_per_channel =
        op == RK_NPU_FUSE_NONE ? FP16_BYTES : 2 * FP16_BYTES;
    uint64_t out_b = (uint64_t)M * align_out *
                     output_row_bytes_per_channel;
    out->output_bytes  = out_b < 256 ? 256 : out_b;
    out->regcmd_bytes  = (uint64_t)num_tasks * per_task_qwords * 8;
    out->task_bytes    = (uint64_t)num_tasks * sizeof(rknpu_task);
    out->num_tasks     = num_tasks;
    return RK_NPU_OK;
}

/* --------------------------------------------------------------- pack ---- */

extern "C" int rk_npu_matmul_f16_pack_a(const rk_npu_matmul_f16_config* cfg,
                                        const uint16_t* A, rk_npu_mem* input) {
    if (!valid_config(cfg)||!A||!input||!input->vaddr) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, K = cfg->K;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    Layout L = gemm_layout(std::min(config_n_tile(*cfg), align_out), K,
                           cfg->op != RK_NPU_FUSE_NONE);
    if (input->size < (uint64_t)M * L.align_in * FP16_BYTES) return RK_NPU_ERR_NOMEM;
    uint16_t* dst = (uint16_t*)input->vaddr;
    if (a_native(*cfg))
        rknpu2_matmul_open::cpu::f16_pack_a_native_k8_m8(M, K, L.align_in, A, dst);
    else
        rknpu2_matmul_open::cpu::f16_pack_a_normal(M, K, L.align_in, A, dst);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_pack_b(const rk_npu_matmul_f16_config* cfg,
                                        const uint16_t* B, rk_npu_mem* weight) {
    if (!valid_config(cfg)||!B||!weight||!weight->vaddr) return RK_NPU_ERR_PARAM;
    const int N = cfg->N, K = cfg->K;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(N, MIN_CHANNEL_TILE));
    Layout L = gemm_layout(std::min(config_n_tile(*cfg), align_out), K,
                           cfg->op != RK_NPU_FUSE_NONE);
    const int align_in = L.align_in;
    if (weight->size < (uint64_t)align_out * align_in * FP16_BYTES) return RK_NPU_ERR_NOMEM;
    uint16_t* w = (uint16_t*)weight->vaddr;
    rknpu2_matmul_open::cpu::f16_pack_b_native_n16_k32(K, N, align_in, align_out, B, w);
    return RK_NPU_OK;
}

/*
 * Pack C0 (M*N row-major fp16) into the EW operand "half-surface" layout.
 *
 * The DPU-RDMA does NOT read C0 row-major. An A=0,B=0 marker test makes
 * D == operand and verifies the following FP16 element index for output
 * (r,c) within a task of `m` rows:
 *
 *     b   = c >> 3                       // 8-channel "half-block" index
 *     idx = ((b+1)/2)*(16*m) + (b/2)*(8*m) + r*8 + (c & 7)      // fp16 elements
 *
 * i.e. each 16-channel surface owns 24*m fp16 slots: the low 8 channels live in a
 * 16*m region (only 8*m used, 8*m gap) read via MRDMA from SRC_BASE, and the high
 * 8 channels live in the following 8*m region read via ERDMA from EW_BASE = base
 * + 16*m.  Channels c>=N (padding up to align_out) are left zero -- their outputs
 * are discarded on unpack.  M-tiling: each task's block is laid out the same way
 * with its own tile_m and concatenated (matching build_regcmd's operand_off).
 */
extern "C" int rk_npu_matmul_f16_pack_operand(const rk_npu_matmul_f16_config* cfg,
                                              const uint16_t* C0, rk_npu_mem* operand) {
    if (!valid_config(cfg)||!C0||!operand||!operand->vaddr) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, N = cfg->N, K = cfg->K;
    Layout L = gemm_layout(N, K, true);
    const int align_out = L.align_out;
    int m_tile = m_tile_for(L, false);
    rk_npu_matmul_sizes sz; rk_npu_matmul_f16_config op_cfg = *cfg;
    op_cfg.op = RK_NPU_FUSE_ADD;
    rk_npu_matmul_f16_query(&op_cfg, &sz);
    if (operand->size < sz.operand_bytes) return RK_NPU_ERR_NOMEM;

    uint16_t* buf = (uint16_t*)operand->vaddr;
    rknpu2_matmul_open::cpu::f16_pack_operand_half_surface(M, N, align_out, m_tile, C0, buf);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_unpack_d(const rk_npu_matmul_f16_config* cfg,
                                          const rk_npu_mem* output, uint16_t* D) {
    if (!valid_config(cfg)||!output||!output->vaddr||!D) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, N = cfg->N, K = cfg->K;
    const uint16_t* o = (const uint16_t*)output->vaddr;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(N, MIN_CHANNEL_TILE));
    const int n_tile = config_n_tile(*cfg);
    if (d_native(*cfg)) {
        rknpu2_matmul_open::cpu::f16_unpack_d_native_n8_m8(M, N, o, D);
        return RK_NPU_OK;
    }
    if (n_tile == align_out) {
        if (cfg->op == RK_NPU_FUSE_NONE)
            rknpu2_matmul_open::cpu::f16_unpack_d_compact(M, N, align_out, o, D);
        else
            rknpu2_matmul_open::cpu::f16_unpack_d(M, N, align_out, o, D);
        return RK_NPU_OK;
    }
    uint64_t tile_output_off = 0;
    for (int n_start = 0; n_start < align_out; n_start += n_tile) {
        const int tile_align_out = std::min(n_tile, align_out - n_start);
        const int valid_n = std::min(tile_align_out, N - n_start);
        if (valid_n > 0) {
            rknpu2_matmul_open::cpu::f16_unpack_d_compact_strided(
                M, valid_n, tile_align_out,
                o + tile_output_off / sizeof(uint16_t), N, n_start, D);
        }
        tile_output_off += (uint64_t)M * tile_align_out * FP16_BYTES;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_unpack_d_f32(const rk_npu_matmul_f16_config* cfg,
                                              const rk_npu_mem* output, float* D) {
    if (!valid_config(cfg)||!output||!output->vaddr||!D) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, N = cfg->N, K = cfg->K;
    const uint16_t* o = (const uint16_t*)output->vaddr;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(N, MIN_CHANNEL_TILE));
    const int n_tile = config_n_tile(*cfg);
    if (d_native(*cfg)) {
        rknpu2_matmul_open::cpu::f16_unpack_d_f32_native_n8_m8(M, N, o, D);
        return RK_NPU_OK;
    }
    if (n_tile == align_out) {
        if (cfg->op == RK_NPU_FUSE_NONE)
            rknpu2_matmul_open::cpu::f16_unpack_d_f32_compact(M, N, align_out, o, D);
        else
            rknpu2_matmul_open::cpu::f16_unpack_d_f32(M, N, align_out, o, D);
        return RK_NPU_OK;
    }
    uint64_t tile_output_off = 0;
    for (int n_start = 0; n_start < align_out; n_start += n_tile) {
        const int tile_align_out = std::min(n_tile, align_out - n_start);
        const int valid_n = std::min(tile_align_out, N - n_start);
        if (valid_n > 0) {
            rknpu2_matmul_open::cpu::f16_unpack_d_f32_compact_strided(
                M, valid_n, tile_align_out,
                o + tile_output_off / sizeof(uint16_t), N, n_start, D);
        }
        tile_output_off += (uint64_t)M * tile_align_out * FP16_BYTES;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_unpack_d_add_bias(
    const rk_npu_matmul_f16_config* cfg, const rk_npu_mem* output,
    const uint16_t* bias, uint16_t* D) {
    if (!valid_config(cfg) || !output || !output->vaddr || !bias || !D ||
        cfg->op != RK_NPU_FUSE_NONE || d_native(*cfg) || cfg->n_tile != 0)
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes sz{};
    int rc = rk_npu_matmul_f16_query(cfg, &sz);
    if (rc != RK_NPU_OK) return rc;
    if (output->size < sz.output_bytes) return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    rknpu2_matmul_open::cpu::f16_unpack_d_compact_add_bias(
        cfg->M, cfg->N, align_out,
        static_cast<const uint16_t*>(output->vaddr), bias, D);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_batch_query(
    int B, const rk_npu_matmul_f16_config* cfg, rk_npu_matmul_sizes* out) {
    if (B <= 0 || !valid_config(cfg) || cfg->op != RK_NPU_FUSE_NONE || !out)
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    int rc = rk_npu_matmul_f16_query(cfg, &one);
    if (rc != RK_NPU_OK) return rc;
    out->input_bytes = one.input_bytes * (uint64_t)B;
    out->weight_bytes = one.weight_bytes * (uint64_t)B;
    out->operand_bytes = 0;
    out->output_bytes = one.output_bytes * (uint64_t)B;
    out->regcmd_bytes = one.regcmd_bytes * (uint64_t)B;
    out->task_bytes = one.task_bytes * (uint64_t)B;
    out->num_tasks = one.num_tasks * B;
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_batch_pack_a(
    int B, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* A_bmk, rk_npu_mem* input) {
    if (B <= 0 || !valid_config(cfg) || cfg->op != RK_NPU_FUSE_NONE ||
        !A_bmk || !input || !input->vaddr)
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rk_npu_matmul_f16_query(cfg, &one) != RK_NPU_OK ||
        input->size < one.input_bytes * (uint64_t)B)
        return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    const Layout L = gemm_layout(
        std::min(config_n_tile(*cfg), align_out), cfg->K, false);
    uint16_t* dst = static_cast<uint16_t*>(input->vaddr);
    if (a_native(*cfg))
        rknpu2_matmul_open::cpu::f16_pack_a_native_k8_m8_batch(
            B, cfg->M, cfg->K, L.align_in, A_bmk, dst);
    else
        rknpu2_matmul_open::cpu::f16_pack_a_normal_batch(
            B, cfg->M, cfg->K, L.align_in, A_bmk, dst);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_batch_pack_b(
    int B, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* B_bkn, rk_npu_mem* weight) {
    if (B <= 0 || !valid_config(cfg) || cfg->op != RK_NPU_FUSE_NONE ||
        !B_bkn || !weight || !weight->vaddr)
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rk_npu_matmul_f16_query(cfg, &one) != RK_NPU_OK ||
        weight->size < one.weight_bytes * (uint64_t)B)
        return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    const Layout L = gemm_layout(
        std::min(config_n_tile(*cfg), align_out), cfg->K, false);
    rknpu2_matmul_open::cpu::f16_pack_b_native_n16_k32_batch(
        B, cfg->K, cfg->N, L.align_in, align_out, B_bkn,
        static_cast<uint16_t*>(weight->vaddr));
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_batch_unpack_d(
    int B, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* output, uint16_t* D_bmn) {
    if (B <= 0 || !valid_config(cfg) || cfg->op != RK_NPU_FUSE_NONE ||
        !output || !output->vaddr || !D_bmn)
        return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rk_npu_matmul_f16_query(cfg, &one) != RK_NPU_OK ||
        output->size < one.output_bytes * (uint64_t)B)
        return RK_NPU_ERR_NOMEM;
    const uint16_t* src = (const uint16_t*)output->vaddr;
    const uint64_t src_batch_stride = one.output_bytes / sizeof(uint16_t);
    if (d_native(*cfg)) {
        rknpu2_matmul_open::cpu::f16_unpack_d_native_n8_m8_batch(
            B, cfg->M, cfg->N, src_batch_stride, src, D_bmn);
        return RK_NPU_OK;
    }
    const int align_out = std::max(
        MIN_CHANNEL_TILE, align_up(cfg->N, MIN_CHANNEL_TILE));
    if (config_n_tile(*cfg) == align_out) {
        rknpu2_matmul_open::cpu::f16_unpack_d_compact_batch(
            B, cfg->M, cfg->N, align_out,
            src_batch_stride, src, D_bmn);
        return RK_NPU_OK;
    }
    for (int b = 0; b < B; ++b) {
        rk_npu_mem view{};
        int rc = rk_npu_mem_view(output, (uint64_t)b * one.output_bytes,
                                 one.output_bytes, &view);
        if (rc != RK_NPU_OK) return rc;
        rc = rk_npu_matmul_f16_unpack_d(
            cfg, &view, D_bmn + (size_t)b * cfg->M * cfg->N);
        if (rc != RK_NPU_OK) return rc;
    }
    return RK_NPU_OK;
}

/* ------------------------------------------------------------ split-K --- */

extern "C" int rk_npu_matmul_f16_splitk_query(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    rk_npu_matmul_sizes* out) {
    rk_npu_matmul_f16_config part_cfg{};
    if (!out || !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    return rk_npu_matmul_f16_batch_query(split_count, &part_cfg, out);
}

extern "C" int rk_npu_matmul_f16_splitk_pack_a(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* A, rk_npu_mem* input) {
    rk_npu_matmul_f16_config part_cfg{};
    rk_npu_matmul_sizes total{};
    if (!A || !input || !input->vaddr ||
        !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    int rc = rk_npu_matmul_f16_batch_query(split_count, &part_cfg, &total);
    if (rc != RK_NPU_OK) return rc;
    if (input->size < total.input_bytes) return RK_NPU_ERR_NOMEM;
    rknpu2_matmul_open::cpu::f16_pack_a_normal_split(
        cfg->M, cfg->K, split_count, part_cfg.K, A,
        static_cast<uint16_t*>(input->vaddr));
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_splitk_pack_b(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const uint16_t* B, rk_npu_mem* weight) {
    rk_npu_matmul_f16_config part_cfg{};
    rk_npu_matmul_sizes one{}, total{};
    if (!B || !weight || !weight->vaddr ||
        !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    int rc = rk_npu_matmul_f16_query(&part_cfg, &one);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_query(split_count, &part_cfg, &total);
    if (rc != RK_NPU_OK) return rc;
    if (weight->size < total.weight_bytes) return RK_NPU_ERR_NOMEM;

    const Layout L = gemm_layout(part_cfg.N, part_cfg.K, false);
    uint16_t* packed = static_cast<uint16_t*>(weight->vaddr);
    for (int split = 0; split < split_count; ++split) {
        const int k0 = split * part_cfg.K;
        const int valid_k = std::min(part_cfg.K, std::max(0, cfg->K - k0));
        uint16_t* out = packed +
            (uint64_t)split * one.weight_bytes / sizeof(uint16_t);
        if (valid_k > 0) {
            rknpu2_matmul_open::cpu::f16_pack_b_native_n16_k32(
                valid_k, cfg->N, L.align_in, L.align_out,
                B + (size_t)k0 * cfg->N, out);
        } else {
            std::memset(out, 0, (size_t)one.weight_bytes);
        }
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_splitk_unpack_d(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* output, uint16_t* D) {
    rk_npu_matmul_f16_config part_cfg{};
    rk_npu_matmul_sizes one{}, total{};
    if (!output || !output->vaddr || !D ||
        !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    int rc = rk_npu_matmul_f16_query(&part_cfg, &one);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_query(split_count, &part_cfg, &total);
    if (rc != RK_NPU_OK) return rc;
    if (output->size < total.output_bytes) return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    rknpu2_matmul_open::cpu::f16_reduce_d_compact_split_f16(
        cfg->M, cfg->N, align_out, split_count,
        one.output_bytes / sizeof(uint16_t),
        static_cast<const uint16_t*>(output->vaddr), D);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_splitk_unpack_d_f32(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* output, float* D) {
    rk_npu_matmul_f16_config part_cfg{};
    rk_npu_matmul_sizes one{}, total{};
    if (!output || !output->vaddr || !D ||
        !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    int rc = rk_npu_matmul_f16_query(&part_cfg, &one);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_query(split_count, &part_cfg, &total);
    if (rc != RK_NPU_OK) return rc;
    if (output->size < total.output_bytes) return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    rknpu2_matmul_open::cpu::f16_reduce_d_compact_split_f32(
        cfg->M, cfg->N, align_out, split_count,
        one.output_bytes / sizeof(uint16_t),
        static_cast<const uint16_t*>(output->vaddr), D);
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_f16_splitk_unpack_d_add_bias(
    int split_count, const rk_npu_matmul_f16_config* cfg,
    const rk_npu_mem* output, const uint16_t* bias, uint16_t* D) {
    rk_npu_matmul_f16_config part_cfg{};
    rk_npu_matmul_sizes one{}, total{};
    if (!output || !output->vaddr || !bias || !D ||
        !splitk_part_config(split_count, cfg, &part_cfg))
        return RK_NPU_ERR_PARAM;
    int rc = rk_npu_matmul_f16_query(&part_cfg, &one);
    if (rc == RK_NPU_OK)
        rc = rk_npu_matmul_f16_batch_query(split_count, &part_cfg, &total);
    if (rc != RK_NPU_OK) return rc;
    if (output->size < total.output_bytes) return RK_NPU_ERR_NOMEM;
    const int align_out = std::max(MIN_CHANNEL_TILE,
                                   align_up(cfg->N, MIN_CHANNEL_TILE));
    rknpu2_matmul_open::cpu::f16_reduce_d_compact_split_f16_add_bias(
        cfg->M, cfg->N, align_out, split_count,
        one.output_bytes / sizeof(uint16_t),
        static_cast<const uint16_t*>(output->vaddr), bias, D);
    return RK_NPU_OK;
}

/* ------------------------------------------------------ prepared (fast) --- */

struct F16Prepared {
    int batch_count = 1;
    int num_tasks = 0;
    uint64_t input_bytes = 0, weight_bytes = 0, output_bytes = 0;
    uint64_t task_obj_addr = 0;
    rk_npu_matmul_f16_config cfg{};
    uint64_t* cmd = nullptr;
    rk_npu_ctx* ctx = nullptr;
    uint32_t iommu_domain_id = 0;
    uint32_t core_mask = 1;
    uint32_t task_start[3]{};
    uint32_t task_count[3]{};
    rk_npu_mem regcmd{}, task{};
    std::vector<int> tile;
    std::vector<uint64_t> input_off, weight_off, output_off, operand_off;
    std::vector<int> feat_idx, dcomp_idx, dst_idx, src_idx, ew_idx;
};

struct rk_npu_matmul_f16_plan { F16Prepared p; };
struct rk_npu_matmul_f16_batch_plan { F16Prepared p; };
struct rk_npu_matmul_f16_splitk_plan { F16Prepared p; };

namespace {

void release_prepared(F16Prepared& p) {
    if (p.ctx) {
        if (p.regcmd.handle) rk_npu_mem_free(p.ctx, &p.regcmd);
        if (p.task.handle) rk_npu_mem_free(p.ctx, &p.task);
    }
}

bool configure_prepared(F16Prepared& p, const BuildMeta& meta) {
    const bool fused = p.cfg.op != RK_NPU_FUSE_NONE;
    const int window = fused ? (45 + 3 + 18) : 108;
    p.feat_idx.assign(meta.num_tasks, -1);
    p.dcomp_idx.assign(meta.num_tasks, -1);
    p.dst_idx.assign(meta.num_tasks, -1);
    p.src_idx.assign(meta.num_tasks, -1);
    p.ew_idx.assign(meta.num_tasks, -1);
    for (int ti = 0; ti < meta.num_tasks; ++ti) {
        const int base = meta.base[ti];
        for (int i = 0; i < window; ++i) {
            const uint32_t reg = (uint32_t)(p.cmd[base + i] & 0xffff);
            if      (reg == R_CNA_FEATURE_DATA_ADDR) p.feat_idx[ti]  = base + i;
            else if (reg == R_CNA_DCOMP_ADDR0)       p.dcomp_idx[ti] = base + i;
            else if (reg == R_DST_BASE_ADDR)         p.dst_idx[ti]   = base + i;
            else if (reg == R_RDMA_SRC_BASE)         p.src_idx[ti]   = base + i;
            else if (reg == R_RDMA_EW_BASE)          p.ew_idx[ti]    = base + i;
        }
        if (p.feat_idx[ti] < 0 || p.dcomp_idx[ti] < 0 ||
            p.dst_idx[ti] < 0 ||
            (fused && (p.src_idx[ti] < 0 || p.ew_idx[ti] < 0)))
            return false;
    }

    p.core_mask = effective_core_mask(p.cfg.core_mask, meta.num_tasks);
    const int core_count = __builtin_popcount(p.core_mask & 7u);
    int cursor = 0;
    for (int core = 0; core < core_count; ++core) {
        const int count = meta.num_tasks / core_count +
                          (core < meta.num_tasks % core_count ? 1 : 0);
        p.task_start[core] = (uint32_t)cursor;
        p.task_count[core] = (uint32_t)count;
        cursor += count;
    }

    /* Each selected core starts an independent PC chain.  Close the tail at
     * every partition boundary; all other links were written by write_chain. */
    for (int ti = 0; ti < meta.num_tasks; ++ti) {
        bool core_tail = false;
        for (int core = 0; core < core_count; ++core) {
            if ((uint32_t)(ti + 1) == p.task_start[core] + p.task_count[core]) {
                core_tail = true;
                break;
            }
        }
        if (!core_tail) continue;
        const int tail = meta.base[ti] + meta.body_size[ti];
        p.cmd[tail + 0] = E(T_NOP, 0, 0);
        p.cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, 0);
        p.cmd[tail + 2] = E(T_VERSION, 0, 0);
    }
    return true;
}

bool prepare_common(rk_npu_iommu_domain* domain, int batch_count,
                    const rk_npu_matmul_f16_config* cfg, F16Prepared& p) {
    if (!domain || !domain->ctx || batch_count <= 0 || !valid_config(cfg) ||
        (batch_count > 1 && cfg->op != RK_NPU_FUSE_NONE))
        return false;
    rk_npu_matmul_sizes one{}, total{};
    if (rk_npu_matmul_f16_query(cfg, &one) != RK_NPU_OK)
        return false;
    if (batch_count == 1) total = one;
    else if (rk_npu_matmul_f16_batch_query(batch_count, cfg, &total) != RK_NPU_OK)
        return false;

    p.ctx = domain->ctx;
    p.iommu_domain_id = domain->id;
    p.batch_count = batch_count;
    p.cfg = *cfg;
    p.input_bytes = total.input_bytes;
    p.weight_bytes = total.weight_bytes;
    p.output_bytes = total.output_bytes;
    if (rk_npu_mem_alloc(domain, total.regcmd_bytes,
                         RK_NPU_MEM_NON_CACHEABLE, &p.regcmd) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, total.task_bytes,
                         RK_NPU_MEM_KERNEL_MAPPING, &p.task) != RK_NPU_OK) {
        release_prepared(p);
        return false;
    }

    BuildMeta meta;
    if (build_regcmd(*cfg, 0, 0, 0, 0, &p.regcmd, &p.task, &meta,
                     batch_count) != RK_NPU_OK) {
        release_prepared(p);
        return false;
    }
    p.num_tasks = meta.num_tasks;
    p.task_obj_addr = p.task.obj_addr;
    p.cmd = (uint64_t*)p.regcmd.vaddr;
    p.tile = meta.tile;
    p.input_off = meta.input_off;
    p.weight_off = meta.weight_off;
    p.output_off = meta.output_off;
    p.operand_off = meta.operand_off;
    if (!configure_prepared(p, meta)) {
        release_prepared(p);
        return false;
    }
    return true;
}

int run_common(rk_npu_ctx* ctx, F16Prepared& p,
               rk_npu_mem* input, rk_npu_mem* weight,
               rk_npu_mem* operand, rk_npu_mem* output) {
    if (!ctx || ctx != p.ctx || !input || !weight || !output ||
        input->ctx_id != ctx->id || weight->ctx_id != ctx->id ||
        output->ctx_id != ctx->id ||
        (operand && operand->ctx_id != ctx->id))
        return RK_NPU_ERR_PARAM;
    if (input->iommu_domain_id != p.iommu_domain_id ||
        weight->iommu_domain_id != p.iommu_domain_id ||
        output->iommu_domain_id != p.iommu_domain_id ||
        (operand && operand->iommu_domain_id != p.iommu_domain_id))
        return RK_NPU_ERR_DOMAIN;
    const bool fused = p.cfg.op != RK_NPU_FUSE_NONE;
    if (fused && (!operand || !operand->vaddr)) return RK_NPU_ERR_PARAM;
    if (input->size < p.input_bytes || weight->size < p.weight_bytes ||
        output->size < p.output_bytes)
        return RK_NPU_ERR_NOMEM;

    for (int ti = 0; ti < p.num_tasks; ++ti) {
        const uint64_t in_addr = input->dma_addr + p.input_off[ti];
        const uint64_t wt_addr = weight->dma_addr + p.weight_off[ti];
        const uint64_t out_addr = output->dma_addr + p.output_off[ti];
        p.cmd[p.feat_idx[ti]] = E(T_CNA, R_CNA_FEATURE_DATA_ADDR,
                                  (uint32_t)in_addr);
        p.cmd[p.dcomp_idx[ti]] = E(T_CNA, R_CNA_DCOMP_ADDR0,
                                   (uint32_t)wt_addr);
        p.cmd[p.dst_idx[ti]] = E(T_DPU, R_DST_BASE_ADDR,
                                 (uint32_t)out_addr);
        if (fused) {
            const uint64_t src = operand->dma_addr + p.operand_off[ti];
            const uint64_t ew = src +
                (uint64_t)p.tile[ti] * FEATURE_ATOMIC * FP16_BYTES;
            p.cmd[p.src_idx[ti]] = E(T_RDMA, R_RDMA_SRC_BASE,
                                     (uint32_t)src);
            p.cmd[p.ew_idx[ti]] = E(T_RDMA, R_RDMA_EW_BASE,
                                    (uint32_t)ew);
        }
    }
    const uint32_t timeout = p.cfg.timeout_ms == 0 ? 6000 : p.cfg.timeout_ms;
    return rknpu2_matmul_open::detail::do_submit_multicore(
        ctx->fd, p.task_obj_addr, p.num_tasks, p.core_mask,
        p.task_start, p.task_count, p.iommu_domain_id, timeout);
}

} /* anonymous namespace */

extern "C" rk_npu_matmul_f16_plan* rk_npu_matmul_f16_prepare(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_f16_config* cfg) {
    rk_npu_matmul_f16_plan* p = new rk_npu_matmul_f16_plan();
    if (!prepare_common(domain, 1, cfg, p->p)) {
        delete p;
        return nullptr;
    }
    return p;
}

extern "C" int rk_npu_matmul_f16_run(rk_npu_ctx* ctx, rk_npu_matmul_f16_plan* plan,
                                     rk_npu_mem* input, rk_npu_mem* weight,
                                     rk_npu_mem* operand, rk_npu_mem* output) {
    if (!plan) return RK_NPU_ERR_PARAM;
    return run_common(ctx, plan->p, input, weight, operand, output);
}

extern "C" void rk_npu_matmul_f16_plan_free(rk_npu_matmul_f16_plan* plan) {
    if (!plan) return;
    release_prepared(plan->p);
    delete plan;
}

extern "C" rk_npu_matmul_f16_batch_plan* rk_npu_matmul_f16_batch_prepare(
    rk_npu_iommu_domain* domain, int B,
    const rk_npu_matmul_f16_config* cfg) {
    rk_npu_matmul_f16_batch_plan* p = new rk_npu_matmul_f16_batch_plan();
    if (!prepare_common(domain, B, cfg, p->p)) {
        delete p;
        return nullptr;
    }
    return p;
}

extern "C" int rk_npu_matmul_f16_batch_run(
    rk_npu_ctx* ctx, rk_npu_matmul_f16_batch_plan* plan,
    rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output) {
    if (!plan) return RK_NPU_ERR_PARAM;
    return run_common(ctx, plan->p, input, weight, nullptr, output);
}

extern "C" void rk_npu_matmul_f16_batch_plan_free(
    rk_npu_matmul_f16_batch_plan* plan) {
    if (!plan) return;
    release_prepared(plan->p);
    delete plan;
}

extern "C" rk_npu_matmul_f16_splitk_plan* rk_npu_matmul_f16_splitk_prepare(
    rk_npu_iommu_domain* domain, int split_count,
    const rk_npu_matmul_f16_config* cfg) {
    rk_npu_matmul_f16_config part_cfg{};
    if (!splitk_part_config(split_count, cfg, &part_cfg)) return nullptr;
    rk_npu_matmul_f16_splitk_plan* p = new rk_npu_matmul_f16_splitk_plan();
    if (!prepare_common(domain, split_count, &part_cfg, p->p)) {
        delete p;
        return nullptr;
    }
    return p;
}

extern "C" int rk_npu_matmul_f16_splitk_run(
    rk_npu_ctx* ctx, rk_npu_matmul_f16_splitk_plan* plan,
    rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output) {
    if (!plan) return RK_NPU_ERR_PARAM;
    return run_common(ctx, plan->p, input, weight, nullptr, output);
}

extern "C" void rk_npu_matmul_f16_splitk_plan_free(
    rk_npu_matmul_f16_splitk_plan* plan) {
    if (!plan) return;
    release_prepared(plan->p);
    delete plan;
}

/*
 * rk_npu_matmul_i8.cpp - int8 x int8 -> fp32 matmul on the RK3588 NPU.
 * ===================================================================
 * Native INT8_MM_INT8_TO_FLOAT32 mode: int8 MACs accumulate to int32 in the
 * CORE, and the DPU output converter casts int32 -> fp32 on chip in the same
 * fused op.  Verified port of allbilly's experimental/gemm_int8.py with the DPU
 * output precision set to fp32.  Device/memory/submit live in rk_npu_core.cpp.
 *
 *     C(M,N) = A(M,K) @ B(K,N),   A,B int8 row-major,   C fp32 row-major
 */
#include "rk_npu_matmul.h"
#include "rk_npu_internal.h"
#include "rk_npu_cpu_kernels.h"
#include "rk_npu_dcomp.h"

#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace {

/* Raw and DCOMP use the same prepared body, including explicit mode disable
 * for raw tasks. This also permits alternating handles in one workspace. */
constexpr int I8_BODY_REGS = 65;

/* int8-specific CBUF budget */
constexpr int INT8_CBUF_ENTRY_VALUES = 64;
/* TODO(m-tiling): performance-tune these hyperparameters by sweeping tile_m on
 * real RK3588 boards.  They are deliberately conservative for the first correct
 * implementation pass. */
constexpr int INT8_M_TILE_MAX   = 128;
/* CBUF data-bank budget for the input M tile.  tune_m_tile measured the real
 * correctness/submit boundary at 8 banks (tile_m*align_in <= 8*32768 = 256 KiB)
 * for align_in <= 4096: e.g. K=2048 max tile 128, K=4096 max tile 64, both exact
 * 8-bank points; 16 banks gives WRONG/SUBMIT.  4 banks (the old value) left half
 * the CBUF unused -> ~1.10-1.21x slower on K=2048/4096.  RK_CBUF_BANKS=12 leaves
 * 4 banks for the weight stream. */
constexpr int INT8_M_TILE_BANKS = 8;
constexpr int INT8_A_NATIVE_M_TILE_BANKS = 4;
/* The 8-bank budget above holds only up to align_in 4096 if the old
 * feature_grains heuristic is used. With CNA_CONV_CON2=0x1e0
 * (feature_grains=30), board sweeps verify 32 rows through K=6144 and
 * 16 rows through K=8192. Beyond 8192
 * stays at 1 row until we have evidence. */
constexpr int INT8_M_TILE_SAFE_ALIGN_IN  = 4096;   /* 8-bank budget valid up to here  */
constexpr int INT8_M_TILE_LARGE32_ALIGN_IN = 6144; /* tile=32 verified up to here     */
constexpr int INT8_M_TILE_HUGE_ALIGN_IN  = 8192;   /* tile=16 verified up to here     */
constexpr int INT8_M_TILE_LARGE_K_ROWS   = 32;
constexpr int INT8_M_TILE_HUGE_K_ROWS    = 16;
constexpr int INT8_FEATURE_GRAINS_LARGE_K = 30;

/* Native output: one row's slot is 4 elements = 16 bytes (int32 or fp32).
 * Native A: one row's slot inside each K/16 block is 16 int8 bytes. */
constexpr int NATIVE_ROW_BYTES = 16;

struct Layout { int align_in, align_out; };
Layout gemm_layout(int N, int K) {
    Layout L;
    L.align_in  = std::max(MIN_CHANNEL_TILE, align_up(K, MIN_CHANNEL_TILE));
    L.align_out = std::max(MIN_CHANNEL_TILE, align_up(N, MIN_CHANNEL_TILE));
    return L;
}

int a_panel(const rk_npu_matmul_i8_config& cfg) {
    return cfg.a_layout == RK_NPU_I8_A_LAYOUT_PANEL8 ? 8 :
           cfg.a_layout == RK_NPU_I8_A_LAYOUT_PANEL16 ? 16 : 0;
}
int c_panel(const rk_npu_matmul_i8_config& cfg) {
    return cfg.c_layout == RK_NPU_I8_C_LAYOUT_PANEL8 ? 8 :
           cfg.c_layout == RK_NPU_I8_C_LAYOUT_PANEL16 ? 16 : 0;
}

bool valid_config(const rk_npu_matmul_i8_config* cfg) {
    if (!(cfg && cfg->M > 0 && cfg->N > 0 && cfg->K > 0
        && (cfg->a_layout == RK_NPU_I8_A_LAYOUT_NORMAL ||
            cfg->a_layout == RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16 || a_panel(*cfg))
        && (cfg->c_layout == RK_NPU_I8_C_LAYOUT_NORMAL_PADDED ||
            cfg->c_layout == RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4 || c_panel(*cfg))
        && (cfg->out_dtype == RK_NPU_I8_OUT_FP32 ||
            cfg->out_dtype == RK_NPU_I8_OUT_INT32))) return false;
    const int aw = a_panel(*cfg), cw = c_panel(*cfg);
    if (aw && !(cfg->c_layout == RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4 || cw == aw)) return false;
    if (!aw && cw && !(cfg->a_layout == RK_NPU_I8_A_LAYOUT_NORMAL && cw == 8)) return false;
    const int width = std::max(aw, cw);
    if (width && cfg->M % width) return false;
    if (width && cfg->K > 8192) return false;
    // Initial panel envelope deliberately excludes observed large-K failures.
    if ((aw == 8 && cfg->K > 2048) || (aw == 16 && cfg->K > 4096)) return false;
    if (cw && (int64_t)(align_up(cfg->N, 32) / 4 - 1) * cw > 8191) return false;
    return true;
}

bool a_native(const rk_npu_matmul_i8_config& cfg) {
    return cfg.a_layout == RK_NPU_I8_A_LAYOUT_NATIVE_K16_M16;
}
bool c_native(const rk_npu_matmul_i8_config& cfg) {
    return cfg.c_layout == RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;
}
bool out_int32(const rk_npu_matmul_i8_config& cfg) {
    return cfg.out_dtype == RK_NPU_I8_OUT_INT32;
}

rk_npu_matmul_i8_config default_config(int M, int N, int K) {
    rk_npu_matmul_i8_config cfg{};
    cfg.M = M;
    cfg.N = N;
    cfg.K = K;
    cfg.a_layout = RK_NPU_I8_A_LAYOUT_NORMAL;
    cfg.c_layout = RK_NPU_I8_C_LAYOUT_NORMAL_PADDED;
    cfg.out_dtype = RK_NPU_I8_OUT_FP32;
    return cfg;
}

int m_tile_for(const Layout& L, const rk_npu_matmul_i8_config& cfg) {
    if (a_native(cfg)) {
        /* Native-A feature tiles use a four-bank CBUF budget:
         * K=1024 -> M=128, K=2048 -> M=64, K=6144 -> M=20, K=8192 -> M=16.
         * Keep rows 4-aligned once possible; split tiles use the native-A DMA
         * stride registers below. */
        if (L.align_in > INT8_M_TILE_HUGE_ALIGN_IN) return 1;
        int rows = INT8_A_NATIVE_M_TILE_BANKS * CBUF_BANK_SIZE / L.align_in;
        rows = std::min(INT8_M_TILE_MAX, rows);
        if (rows >= 4) rows &= ~3;
        if (c_panel(cfg)) rows = rows / c_panel(cfg) * c_panel(cfg);
        return std::max(1, rows);
    }

    /* TODO(m-tiling): validate the remaining K>6144 boundary and implement
     * native-A split-tile register updates before raising the K=8192 tile again. */
    const int rows_by_cbuf = INT8_M_TILE_BANKS * CBUF_BANK_SIZE / L.align_in;  /* int8: 1 byte/elem */
    /* TODO(m-tiling): tune INT8_M_TILE_MAX by sweeping correctness/perf on the board.
     * The current 128-row cap keeps PC chains short
     * without assuming all larger M blocks are safe.  INT8_M_TILE_BANKS is also
     * intentionally conservative: early tests show maxing out the CBUF data-bank
     * budget can still submit-fail on very large K.  This can leave performance
     * on the table and should be revisited after the first correct tiling pass. */
    int rows = L.align_in > INT8_M_TILE_HUGE_ALIGN_IN ? 1 :
               L.align_in > INT8_M_TILE_LARGE32_ALIGN_IN ? INT8_M_TILE_HUGE_K_ROWS :
               L.align_in > INT8_M_TILE_SAFE_ALIGN_IN ? INT8_M_TILE_LARGE_K_ROWS :
               std::min(INT8_M_TILE_MAX, rows_by_cbuf);
    const int width = std::max(a_panel(cfg), c_panel(cfg));
    if (width) rows = rows / width * width;
    return std::max(1, rows);
}
int n_tile_for(const Layout& L, int strategy_n_tile) {
    if (strategy_n_tile <= 0 || strategy_n_tile >= L.align_out)
        return L.align_out;
    return std::max(MIN_CHANNEL_TILE,
                    align_up(strategy_n_tile, MIN_CHANNEL_TILE));
}

int feature_grains_for(int tile_m, int input_row_bytes) {
    if (input_row_bytes > INT8_M_TILE_SAFE_ALIGN_IN && tile_m >= INT8_M_TILE_HUGE_K_ROWS)
        return INT8_FEATURE_GRAINS_LARGE_K;

    /* For narrow K the m+1 grain count is known-good. For wider K, cap
     * CNA_CONV_CON2 by roughly the number of rows that fit in two
     * CBUF banks: M=128,K=1024 uses feature_grains=64, not 129. */
    int two_bank_rows = (ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1;
    return std::max(1, std::min(tile_m + 1, two_bank_rows));
}

/* One task's body: matmul as a fused 1x1 conv, with explicit DCOMP state. */
void make_gemm_regs(std::vector<uint64_t>& v, int m, int full_M, int full_N, int tile_N, int K,
                    uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma,
                    const rk_npu_matmul_i8_config& cfg, bool full_dcomp_state = true) {
    Layout fullL = gemm_layout(full_N, K);
    const int align_in = fullL.align_in;
    const int align_out = std::max(MIN_CHANNEL_TILE, align_up(tile_N, MIN_CHANNEL_TILE));
    const int input_row_bytes = align_in;             /* int8 */
    const int aw = a_panel(cfg), cw = c_panel(cfg);
    const int feature_grains = aw
                             ? std::max(1, std::min(m / aw + 1,
                               (ceil_div(2 * CBUF_BANK_SIZE, aw * align_in) + 1) & ~1))
                             : a_native(cfg)
                             ? 2 : feature_grains_for(m, input_row_bytes);
    const int data_banks = std::min(
        std::max(ceil_div(m * input_row_bytes, CBUF_BANK_SIZE), 1),
        RK_CBUF_BANKS - 1);
    int line_stride = std::max(1, align_up(K, MIN_CHANNEL_TILE) / 16);
    /* DPU row notch is the fp32 row stride in 4-element atomics minus one.
     * The old RK_LINE_STRIDE_GROUP_CAP clamp happened to work for one-row
     * tasks, but corrupts multi-row M tiles once N exceeds 416. The row-major
     * stride gives N=1024 -> notch 0xff and N=8192 -> notch 0x7ff. */
    const int notch_val = fullL.align_out / 4 - 1;

    const bool a_native_split_m = a_native(cfg) && m < full_M;
    const uint32_t cna_conv1 = a_native(cfg) ? (a_native_split_m ? (1u << 29) : 0u) : (1u << 29);
    const uint32_t cna_conv2 = (uint32_t)feature_grains << 4;
    const uint32_t data_size0 = aw ? ((uint32_t)aw << 16) | (uint32_t)(m / aw)
                                  : a_native(cfg) ? (((uint32_t)m << 16) | 1u) : ((1u << 16) | (uint32_t)m);
    const uint32_t data_size2 = aw ? (uint32_t)aw : a_native(cfg) ? (uint32_t)m : 1u;
    const uint32_t cbuf1_auto = aw ? (uint32_t)ceil_div(aw * align_in, INT8_CBUF_ENTRY_VALUES)
                                 : a_native(cfg) ? (uint32_t)ceil_div(m * align_in, INT8_CBUF_ENTRY_VALUES)
                                             : (uint32_t)ceil_div(align_in, INT8_CBUF_ENTRY_VALUES);
    const uint32_t cbuf1 = cbuf1_auto;
    const uint32_t dma_con1 = aw ? (uint32_t)(aw * align_in / 16)
                                 : a_native(cfg) ? (a_native_split_m ? (uint32_t)full_M : 0x200u)
                                           : (uint32_t)line_stride;
    const uint32_t dma_con2 = a_native(cfg) ? (a_native_split_m ? (uint32_t)(full_M - m)
                                                               : (0x10000000u - (uint32_t)full_M * 3u))
                                           : 0u;
    const uint32_t dma_con0 = (15u << 16) | 15u;
    const uint32_t core_size0 = aw ? ((uint32_t)(m / aw - 1) << 16) | (uint32_t)(aw - 1)
                                   : a_native(cfg) ? (uint32_t)(m - 1) : ((uint32_t)(m - 1) << 16);

    const uint32_t dst_surf_stride = cw ? cw * 16u : c_native(cfg) ? (uint32_t)(full_M * NATIVE_ROW_BYTES) : (1u << 4);
    const uint32_t dst_w = cw ? cw - 1u : c_native(cfg) ? (uint32_t)(m - 1) : 0u;
    const uint32_t dst_h = cw ? m / cw - 1u : c_native(cfg) ? 0u : (uint32_t)(m - 1);
    const uint32_t notch = (uint32_t)notch_val * (cw ? cw : 1);
    const uint32_t dst_notch = c_native(cfg) ? 0u : ((notch << 16) | notch);
    const uint32_t wdma_size1 = (dst_h << 16) | dst_w;
    const uint32_t surface_add = cw ? (uint32_t)cw << 7 : c_native(cfg) ? ((uint32_t)full_M << 7) : 0x80u;
    const uint32_t bs_cfg = 0x53u;
    const uint32_t bs_ow_cfg = 0x7fcu;
    const uint32_t bn_cfg = 0x53u;
    const uint32_t ew_cfg = 0x383u;

    v.clear();
    v.push_back(E(T_CNA, R_CNA_DCOMP_REGNUM, 0));
    v.push_back(E(T_CNA, R_CNA_DCOMP_CTRL, 0));
    v.push_back(E(T_DPU,  R_S_POINTER, (1<<3)|(1<<2)|(1<<1)));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON1, cna_conv1));                /* int8 in/proc */
    v.push_back(E(T_CNA,  R_CNA_CONV_CON2, cna_conv2));
    v.push_back(E(T_CNA,  R_CNA_CONV_CON3, (1<<3)|1));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE0, data_size0));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE1, ((uint32_t)(align_in-1)<<16)|align_in));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE2, data_size2));
    v.push_back(E(T_CNA,  R_CNA_DATA_SIZE3, (uint32_t)m));
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
    v.push_back(E(T_CNA,  R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_dma));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON0, dma_con0));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON1, dma_con1));
    v.push_back(E(T_CNA,  R_CNA_DMA_CON2, dma_con2));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE0, data_size0));
    v.push_back(E(T_CNA,  R_CNA_FC_DATA_SIZE1, align_in));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_CTRL, 0));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_REGNUM, 0));
    v.push_back(E(T_CNA,  R_CNA_DCOMP_ADDR0, (uint32_t)wt_dma));
    if (full_dcomp_state)
        for (int lane = 0; lane < 16; ++lane)
            v.push_back(E(T_CNA, R_CNA_DCOMP_AMOUNT0 + lane * 4, 0));
    v.push_back(E(T_CORE, R_CORE_MISC_CFG, 1));                        /* int8 proc */
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_0, core_size0));
    v.push_back(E(T_CORE, R_CORE_DATAOUT_SIZE_1, align_out-1));
    v.push_back(E(T_CORE, R_CORE_RESERVED_3030, 0));
    v.push_back(E(T_DPU,  R_FEATURE_MODE_CFG, (15<<5)|(2<<1)));
    v.push_back(E(T_DPU,  R_DATA_FORMAT,
                  (out_int32(cfg) ? 4u : 5u) << 29));  /* OUT_PRECISION: 4 int32 / 5 fp32 */
    v.push_back(E(T_DPU,  R_DST_BASE_ADDR, (uint32_t)out_dma));
    v.push_back(E(T_DPU,  R_DST_SURF_STRIDE, dst_surf_stride));
    v.push_back(E(T_DPU,  R_DATA_CUBE_WIDTH, dst_w));
    v.push_back(E(T_DPU,  R_DATA_CUBE_HEIGHT, dst_h));
    v.push_back(E(T_DPU,  R_DATA_CUBE_NOTCH, dst_notch));
    v.push_back(E(T_DPU,  R_DATA_CUBE_CHANNEL, ((uint32_t)(align_out-1)<<16)|(align_out-1)));
    v.push_back(E(T_DPU,  R_BS_CFG, bs_cfg));
    v.push_back(E(T_DPU,  R_BS_OW_CFG, bs_ow_cfg));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_0, align_out-1));
    v.push_back(E(T_DPU,  R_WDMA_SIZE_1, wdma_size1));
    v.push_back(E(T_DPU,  R_BN_CFG, bn_cfg));
    v.push_back(E(T_DPU,  R_EW_CFG, ew_cfg));                           /* EW fully bypassed */
    v.push_back(E(T_DPU,  R_OUT_CVT_SCALE, 1));
    v.push_back(E(T_DPU,  R_SURFACE_ADD, surface_add));
}

struct BuildMeta {
    int num_tasks, input_row_bytes, row_stride_bytes;
    uint64_t output_bytes;
    std::vector<int> base, body_size, start;
    std::vector<uint64_t> input_off, weight_off, output_off;
};

int build_regcmd(const rk_npu_matmul_i8_config& cfg,
                 uint64_t in_dma, uint64_t wt_dma, uint64_t out_dma,
                 rk_npu_mem* regcmd, rk_npu_mem* task, BuildMeta* meta,
                 int batch_count = 1, int strategy_n_tile = 0) {
    if (batch_count <= 0) return RK_NPU_ERR_PARAM;
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    Layout L = gemm_layout(N, K);
    meta->input_row_bytes  = L.align_in;
    meta->row_stride_bytes = L.align_out * 4;        /* fp32 */
    const int m_tile = m_tile_for(L, cfg);
    const int n_tile = n_tile_for(L, strategy_n_tile);
    const int tasks_per_batch = ceil_div(M, m_tile);
    const int n_tasks = ceil_div(L.align_out, n_tile);
    meta->num_tasks = tasks_per_batch * n_tasks * batch_count;

    rk_npu_matmul_sizes sz{};
    const int body = I8_BODY_REGS;
    const int per_task_qwords = align_up(body + 4, 2);
    sz.input_bytes = (uint64_t)M * L.align_in;
    sz.weight_bytes = (uint64_t)L.align_out * L.align_in;
    sz.output_bytes = std::max<uint64_t>((uint64_t)M * L.align_out * 4, 256);
    sz.regcmd_bytes = (uint64_t)(tasks_per_batch * n_tasks) * per_task_qwords * 8;
    sz.task_bytes = (uint64_t)(tasks_per_batch * n_tasks) * sizeof(rknpu_task);
    meta->output_bytes = sz.output_bytes * (uint64_t)batch_count;
    if (regcmd->size < sz.regcmd_bytes * (uint64_t)batch_count ||
        task->size < sz.task_bytes * (uint64_t)batch_count) return RK_NPU_ERR_NOMEM;

    uint64_t* cmd  = (uint64_t*)regcmd->vaddr;
    rknpu_task* tk = (rknpu_task*)task->vaddr;
    std::memset(tk, 0, (size_t)meta->num_tasks * sizeof(rknpu_task));

    meta->base.resize(meta->num_tasks);
    meta->body_size.resize(meta->num_tasks);
    meta->start.resize(meta->num_tasks);
    meta->input_off.resize(meta->num_tasks);
    meta->weight_off.resize(meta->num_tasks);
    meta->output_off.resize(meta->num_tasks);
    std::vector<std::vector<uint64_t>> bodies(meta->num_tasks);
    int off = 0, ti = 0;
    for (int b = 0; b < batch_count; ++b) {
        const uint64_t batch_input_off  = (uint64_t)b * sz.input_bytes;
        const uint64_t batch_weight_off = (uint64_t)b * sz.weight_bytes;
        const uint64_t batch_output_off = (uint64_t)b * sz.output_bytes;
        for (int n_start = 0; n_start < L.align_out; n_start += n_tile) {
            int tile_n = std::min(n_tile, L.align_out - n_start);
            const uint64_t n_weight_off = (uint64_t)(n_start / MIN_CHANNEL_TILE) * L.align_in * MIN_CHANNEL_TILE;
            const uint64_t n_output_off = c_native(cfg)
                                        ? (uint64_t)n_start * M * 4
                                        : (uint64_t)n_start * (c_panel(cfg) ? c_panel(cfg) : 1) * 4;
            for (int start = 0; start < M; start += m_tile, ++ti) {
                int tile_m = std::min(m_tile, M - start);
                /* Store per-task offsets relative to the caller's packed batch buffers.
                 * The initial regcmd is built from the prepare-time base DMA values
                 * (usually 0); run() patches these same registers for real buffers. */
                meta->input_off[ti]  = batch_input_off + (uint64_t)start * (a_native(cfg) ? NATIVE_ROW_BYTES
                                                                                          : meta->input_row_bytes);
                meta->weight_off[ti] = batch_weight_off + n_weight_off;
                meta->output_off[ti] = batch_output_off + n_output_off
                                     + (uint64_t)start * (c_native(cfg) ? NATIVE_ROW_BYTES
                                                                        : meta->row_stride_bytes);

                make_gemm_regs(bodies[ti], tile_m, M, N, tile_n, K,
                               in_dma  + meta->input_off[ti],
                               wt_dma  + meta->weight_off[ti],
                               out_dma + meta->output_off[ti],
                               cfg);
                meta->base[ti] = off;
                meta->body_size[ti] = (int)bodies[ti].size();
                meta->start[ti] = start;
                off += align_up((int)bodies[ti].size() + 4, 2);
            }
        }
    }
    /* plain matmul: enable CNA+CORE+DPU (0xd), no DPU-RDMA */
    rknpu2_matmul_open::detail::write_chain(cmd, tk, regcmd->dma_addr, bodies, meta->base, { (6u<<1)|1u, 0u, 0xdu });
    return RK_NPU_OK;
}

} /* anonymous namespace */

int rknpu2_matmul_open::detail::build_i8_group(const std::vector<I8GroupItem>& items,
                          rk_npu_mem* regcmd, rk_npu_mem* task,
                          uint32_t task_start[3], uint32_t task_count[3]) {
    if (items.empty() || !regcmd || !task || !regcmd->vaddr || !task->vaddr ||
        !task_start || !task_count || (task->flags & RK_NPU_MEM_F_VIEW))
        return RK_NPU_ERR_PARAM;
    if (regcmd->ctx_id != task->ctx_id ||
        regcmd->iommu_domain_id != task->iommu_domain_id) return RK_NPU_ERR_DOMAIN;
    std::vector<std::vector<uint64_t>> bodies[3];
    std::vector<int> bases[3];
    int qwords = 0, tasks = 0;
    for (int core = 0; core < 3; ++core) {
        task_start[core] = tasks;
        for (const auto& item : items) {
            if (item.core < 0 || item.core > 2 || !valid_config(item.cfg) ||
                !item.input || !item.weight || !item.output) return RK_NPU_ERR_PARAM;
            if (item.core != core) continue;
            for (auto* mem : {item.input, item.weight, item.output}) {
                if (mem->ctx_id != task->ctx_id ||
                    mem->iommu_domain_id != task->iommu_domain_id)
                    return RK_NPU_ERR_DOMAIN;
            }
            const auto& cfg = *item.cfg;
            const auto layout = gemm_layout(cfg.N, cfg.K);
            const uint64_t input_bytes = uint64_t(cfg.M) * layout.align_in;
            const uint64_t weight_bytes = uint64_t(layout.align_in) * layout.align_out;
            const uint64_t output_bytes = std::max<uint64_t>(256,
                uint64_t(cfg.M) * layout.align_out * 4);
            if (item.input->size < input_bytes || item.weight->size < weight_bytes ||
                item.output->size < output_bytes) return RK_NPU_ERR_NOMEM;
            if ((item.input->dma_addr | item.weight->dma_addr |
                 item.output->dma_addr) & 15) return RK_NPU_ERR_PARAM;
            const int mt = m_tile_for(layout, cfg), nt = n_tile_for(layout, 0);
            for (int n = 0; n < layout.align_out; n += nt) {
                for (int m = 0; m < cfg.M; m += mt) {
                    bodies[core].emplace_back();
                    auto& body = bodies[core].back();
                    // Raw-only grouped callers reserve 64 qwords/task. Keep
                    // explicit CTRL disable but omit unused AMOUNT registers.
                    make_gemm_regs(body, std::min(mt, cfg.M-m), cfg.M, cfg.N,
                        std::min(nt, layout.align_out-n), cfg.K,
                        item.input->dma_addr + uint64_t(m) *
                            (a_native(cfg) ? 16 : layout.align_in),
                        item.weight->dma_addr + uint64_t(n) * layout.align_in,
                        item.output->dma_addr + (c_native(cfg)
                            ? uint64_t(n)*cfg.M*4 + uint64_t(m)*16
                            : uint64_t(m)*layout.align_out*4 + uint64_t(n)*
                              (c_panel(cfg) ? c_panel(cfg) : 1)*4), cfg, false);
                    bases[core].push_back(qwords);
                    qwords += align_up(int(body.size()) + 4, 2);
                    ++tasks;
                    if (uint64_t(qwords)*8 > regcmd->size ||
                        uint64_t(tasks)*sizeof(rknpu_task) > task->size)
                        return RK_NPU_ERR_NOMEM;
                }
            }
        }
        task_count[core] = tasks - task_start[core];
    }
    std::memset(task->vaddr, 0, size_t(tasks)*sizeof(rknpu_task));
    for (int core = 0; core < 3; ++core)
        write_chain(static_cast<uint64_t*>(regcmd->vaddr),
            static_cast<rknpu_task*>(task->vaddr) + task_start[core],
            regcmd->dma_addr, bodies[core], bases[core], {13u, 0u, 13u});
    return RK_NPU_OK;
}

/* ------------------------------------------------------------- config ---- */

extern "C" void rk_npu_matmul_i8_config_init(rk_npu_matmul_i8_config* cfg, int M, int N, int K) {
    if (!cfg) return;
    *cfg = default_config(M, N, K);
}

/* -------------------------------------------------------------- query ---- */

int rknpu2_matmul_open::detail::query_i8(const rk_npu_matmul_i8_config* cfg, rk_npu_matmul_sizes* out) {
    return rknpu2_matmul_open::detail::query_i8_n_tiled(cfg, 0, out);
}

int rknpu2_matmul_open::detail::query_i8_n_tiled(const rk_npu_matmul_i8_config* cfg, int strategy_n_tile,
                            rk_npu_matmul_sizes* out) {
    if (!valid_config(cfg) || !out) return RK_NPU_ERR_PARAM;
    Layout L = gemm_layout(cfg->N, cfg->K);
    int m_tile = m_tile_for(L, *cfg);
    int n_tile = n_tile_for(L, strategy_n_tile);
    int num_tasks = ceil_div(cfg->M, m_tile) * ceil_div(L.align_out, n_tile);

    const int body = I8_BODY_REGS;
    const int per_task_qwords = align_up(body + 4, 2);

    out->input_bytes   = (uint64_t)cfg->M * L.align_in;
    out->weight_bytes  = (uint64_t)L.align_out * L.align_in;
    out->operand_bytes = 0;                                  /* int8 matmul has no fused operand */
    uint64_t out_b     = (uint64_t)cfg->M * L.align_out * 4;
    out->output_bytes  = out_b < 256 ? 256 : out_b;
    out->regcmd_bytes  = (uint64_t)num_tasks * per_task_qwords * 8;
    out->task_bytes    = (uint64_t)num_tasks * sizeof(rknpu_task);
    out->num_tasks     = num_tasks;
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_i8_batch_query(int B, const rk_npu_matmul_i8_config* cfg,
                                            rk_npu_matmul_sizes* out) {
    if (B <= 0 || !valid_config(cfg) || !out) return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    int rc = rknpu2_matmul_open::detail::query_i8(cfg, &one);
    if (rc != RK_NPU_OK) return rc;

    /* Batch buffers are deliberately a simple concatenation of single-GEMM
     * packed blocks. That makes dmabuf suballocation predictable and keeps the
     * batch API compatible with the existing pack/unpack helpers. */
    out->input_bytes   = one.input_bytes   * (uint64_t)B;
    out->weight_bytes  = one.weight_bytes  * (uint64_t)B;
    out->operand_bytes = 0;
    out->output_bytes  = one.output_bytes  * (uint64_t)B;
    out->regcmd_bytes  = one.regcmd_bytes  * (uint64_t)B;
    out->task_bytes    = one.task_bytes    * (uint64_t)B;
    out->num_tasks     = one.num_tasks * B;
    return RK_NPU_OK;
}

/* --------------------------------------------------------------- pack ---- */

int rknpu2_matmul_open::detail::pack_i8_a(const rk_npu_matmul_i8_config* cfg,
                     const int8_t* A, rk_npu_mem* input) {
    if (!valid_config(cfg) || !A || !input || !input->vaddr) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, K = cfg->K;
    Layout L = gemm_layout(cfg->N, K);
    if (input->size < (uint64_t)M * L.align_in) return RK_NPU_ERR_NOMEM;
    int8_t* dst = (int8_t*)input->vaddr;
    if (a_native(*cfg) || a_panel(*cfg)) {
        rknpu2_matmul_open::cpu::i8_pack_a_native_k16_m16(M, K, L.align_in, A, dst, a_panel(*cfg));
    } else {
        rknpu2_matmul_open::cpu::i8_pack_a_normal(M, K, L.align_in, A, dst);
    }
    return RK_NPU_OK;
}

int rknpu2_matmul_open::detail::pack_i8_b(const rk_npu_matmul_i8_config* cfg,
                     const int8_t* B, rk_npu_mem* weight) {
    if (!valid_config(cfg) || !B || !weight || !weight->vaddr) return RK_NPU_ERR_PARAM;
    const int N = cfg->N, K = cfg->K;
    Layout L = gemm_layout(N, K);
    const int align_in = L.align_in, align_out = L.align_out;
    if (weight->size < (uint64_t)align_out * align_in) return RK_NPU_ERR_NOMEM;
    int8_t* w = (int8_t*)weight->vaddr;
    rknpu2_matmul_open::cpu::i8_pack_b_native_n32_k32(K, N, align_in, align_out, B, w);
    return RK_NPU_OK;
}

int rknpu2_matmul_open::detail::unpack_i8_c(const rk_npu_matmul_i8_config* cfg,
                       const rk_npu_mem* output, void* C) {
    if (!valid_config(cfg) || !output || !output->vaddr || !C) return RK_NPU_ERR_PARAM;
    const int M = cfg->M, N = cfg->N;
    Layout L = gemm_layout(N, cfg->K);
    const uint8_t* src = (const uint8_t*)output->vaddr;
    if (c_native(*cfg) || c_panel(*cfg)) {
        rknpu2_matmul_open::cpu::i8_unpack_c_native_n4_m4(M, N, src, C, c_panel(*cfg));
    } else {
        rknpu2_matmul_open::cpu::i8_unpack_c_normal(M, N, L.align_out, src, C);
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_i8_batch_pack_a(int B, const rk_npu_matmul_i8_config* cfg,
                                             const int8_t* A_bmk, rk_npu_mem* input) {
    if (B<=0||!valid_config(cfg)||!A_bmk||!input||!input->vaddr) return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rknpu2_matmul_open::detail::query_i8(cfg, &one) != RK_NPU_OK) return RK_NPU_ERR_PARAM;
    if (input->size < one.input_bytes * (uint64_t)B) return RK_NPU_ERR_NOMEM;

    for (int b = 0; b < B; ++b) {
        rk_npu_mem view = *input;
        view.vaddr = (void*)((uint8_t*)input->vaddr + (uint64_t)b * one.input_bytes);
        view.dma_addr = input->dma_addr + (uint64_t)b * one.input_bytes;
        view.size = one.input_bytes;
        int rc = rknpu2_matmul_open::detail::pack_i8_a(cfg, A_bmk + (size_t)b * cfg->M * cfg->K, &view);
        if (rc != RK_NPU_OK) return rc;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_i8_batch_pack_b(int B, const rk_npu_matmul_i8_config* cfg,
                                             const int8_t* B_bkn, rk_npu_mem* weight) {
    if (B<=0||!valid_config(cfg)||!B_bkn||!weight||!weight->vaddr) return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rknpu2_matmul_open::detail::query_i8(cfg, &one) != RK_NPU_OK) return RK_NPU_ERR_PARAM;
    if (weight->size < one.weight_bytes * (uint64_t)B) return RK_NPU_ERR_NOMEM;

    for (int b = 0; b < B; ++b) {
        rk_npu_mem view = *weight;
        view.vaddr = (void*)((uint8_t*)weight->vaddr + (uint64_t)b * one.weight_bytes);
        view.dma_addr = weight->dma_addr + (uint64_t)b * one.weight_bytes;
        view.size = one.weight_bytes;
        int rc = rknpu2_matmul_open::detail::pack_i8_b(cfg, B_bkn + (size_t)b * cfg->K * cfg->N, &view);
        if (rc != RK_NPU_OK) return rc;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_matmul_i8_batch_unpack_c(int B, const rk_npu_matmul_i8_config* cfg,
                                               const rk_npu_mem* output, void* C_bmn) {
    if (B<=0||!valid_config(cfg)||!output||!output->vaddr||!C_bmn) return RK_NPU_ERR_PARAM;
    rk_npu_matmul_sizes one{};
    if (rknpu2_matmul_open::detail::query_i8(cfg, &one) != RK_NPU_OK) return RK_NPU_ERR_PARAM;
    if (output->size < one.output_bytes * (uint64_t)B) return RK_NPU_ERR_NOMEM;

    for (int b = 0; b < B; ++b) {
        rk_npu_mem view = *output;
        view.vaddr = (void*)((uint8_t*)output->vaddr + (uint64_t)b * one.output_bytes);
        view.dma_addr = output->dma_addr + (uint64_t)b * one.output_bytes;
        view.size = one.output_bytes;
        void* dst = (void*)((uint8_t*)C_bmn + (size_t)b * cfg->M * cfg->N * 4);
        int rc = rknpu2_matmul_open::detail::unpack_i8_c(cfg, &view, dst);
        if (rc != RK_NPU_OK) return rc;
    }
    return RK_NPU_OK;
}

/* ------------------------------------------------------ prepared (fast) --- */

struct rk_npu_matmul_i8_plan {
    int num_tasks, input_row_bytes, row_stride_bytes, n_tile;
    uint64_t output_bytes, task_obj_addr;
    rk_npu_matmul_i8_config cfg{};
    uint64_t* cmd;
    rk_npu_ctx* ctx;
    uint32_t iommu_domain_id;
    rk_npu_mem regcmd{}, task{};
    std::vector<uint64_t> input_off, weight_off, output_off;
    std::vector<int> base, body_size, start, feat_idx, dcomp_idx, dst_idx;
    uint32_t prebound_core_mask = 0;
    uint32_t prebound_task_start[3] = {0, 0, 0};
    uint32_t prebound_task_count[3] = {0, 0, 0};
};

struct rk_npu_matmul_i8_batch_plan {
    int batch_count, num_tasks, input_row_bytes, row_stride_bytes;
    uint64_t output_bytes, task_obj_addr;
    rk_npu_matmul_i8_config cfg{};
    uint64_t* cmd;
    rk_npu_ctx* ctx;
    uint32_t iommu_domain_id;
    rk_npu_mem regcmd{}, task{};
    std::vector<uint64_t> input_off, weight_off, output_off;
    std::vector<int> feat_idx, dcomp_idx, dst_idx;
};

namespace rknpu2_matmul_open::detail {
static void patch_dcomp(rk_npu_matmul_i8_plan* plan, int ti,
                        const I8CompressedTile* tile);
}

bool find_patch_indices(uint64_t* cmd, const std::vector<int>& base, int window,
                        std::vector<int>& feat_idx,
                        std::vector<int>& dcomp_idx,
                        std::vector<int>& dst_idx) {
    const int n = (int)base.size();
    feat_idx.assign(n, -1);
    dcomp_idx.assign(n, -1);
    dst_idx.assign(n, -1);
    for (int ti = 0; ti < n; ++ti) {
        int b = base[ti];
        for (int i = 0; i < window; ++i) {
            uint32_t reg = (uint32_t)(cmd[b + i] & 0xffff);
            if      (reg == R_CNA_FEATURE_DATA_ADDR) feat_idx[ti]  = b + i;
            else if (reg == R_CNA_DCOMP_ADDR0)       dcomp_idx[ti] = b + i;
            else if (reg == R_DST_BASE_ADDR)         dst_idx[ti]   = b + i;
            if (feat_idx[ti] >= 0 && dcomp_idx[ti] >= 0 && dst_idx[ti] >= 0) break;
        }
        if (feat_idx[ti] < 0 || dcomp_idx[ti] < 0 || dst_idx[ti] < 0) return false;
    }
    return true;
}

namespace {

rk_npu_matmul_i8_plan* prepare_i8_impl(rk_npu_iommu_domain* domain,
                                       const rk_npu_matmul_i8_config* cfg,
                                       int strategy_n_tile) {
    if (!domain || !domain->ctx || !valid_config(cfg)) return nullptr;
    rk_npu_ctx* ctx = domain->ctx;
    rk_npu_matmul_sizes sz;
    if (rknpu2_matmul_open::detail::query_i8_n_tiled(cfg, strategy_n_tile, &sz) != RK_NPU_OK) return nullptr;

    rk_npu_matmul_i8_plan* p = new rk_npu_matmul_i8_plan();
    p->ctx = ctx;
    p->iommu_domain_id = domain->id;
    p->cfg = *cfg;
    if (rk_npu_mem_alloc(domain, sz.regcmd_bytes, RK_NPU_MEM_NON_CACHEABLE, &p->regcmd) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sz.task_bytes, RK_NPU_MEM_KERNEL_MAPPING, &p->task) != RK_NPU_OK) {
        rknpu2_matmul_open::detail::free_i8_plan(p);
        return nullptr;
    }

    BuildMeta meta;
    if (build_regcmd(*cfg, 0, 0, 0, &p->regcmd, &p->task, &meta,
                     1, strategy_n_tile) != RK_NPU_OK) {
        rknpu2_matmul_open::detail::free_i8_plan(p);
        return nullptr;
    }

    p->num_tasks       = meta.num_tasks;
    p->input_row_bytes = meta.input_row_bytes;
    p->row_stride_bytes= meta.row_stride_bytes;
    p->n_tile          = n_tile_for(gemm_layout(cfg->N, cfg->K), strategy_n_tile);
    p->output_bytes    = meta.output_bytes;
    p->task_obj_addr   = p->task.obj_addr;
    p->cmd             = (uint64_t*)p->regcmd.vaddr;
    p->start           = meta.start;
    p->base            = meta.base;
    p->body_size       = meta.body_size;
    p->input_off       = meta.input_off;
    p->weight_off      = meta.weight_off;
    p->output_off      = meta.output_off;
    if (!find_patch_indices(p->cmd, meta.base, I8_BODY_REGS, p->feat_idx, p->dcomp_idx, p->dst_idx)) {
        rknpu2_matmul_open::detail::free_i8_plan(p);
        return nullptr;
    }
    return p;
}

} /* anonymous namespace */

rk_npu_matmul_i8_plan* rknpu2_matmul_open::detail::prepare_i8(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_i8_config* cfg) {
    return prepare_i8_impl(domain, cfg, 0);
}

rk_npu_matmul_i8_plan* rknpu2_matmul_open::detail::prepare_i8_n_tiled(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_i8_config* cfg, int n_tile) {
    return prepare_i8_impl(domain, cfg, n_tile);
}

int rknpu2_matmul_open::detail::run_i8(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
                  rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output) {
    if (!ctx || !plan || ctx != plan->ctx || !input || !weight || !output ||
        input->ctx_id != ctx->id || weight->ctx_id != ctx->id ||
        output->ctx_id != ctx->id) return RK_NPU_ERR_PARAM;
    if (input->iommu_domain_id != plan->iommu_domain_id ||
        weight->iommu_domain_id != plan->iommu_domain_id ||
        output->iommu_domain_id != plan->iommu_domain_id)
        return RK_NPU_ERR_DOMAIN;
    if (output->size < plan->output_bytes) return RK_NPU_ERR_NOMEM;
    /* Full GEMM tasks overwrite the complete padded/native output region; avoid
     * clearing the output dmabuf in the timed run path. */
    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        uint64_t in_addr  = input->dma_addr  + plan->input_off[ti];
        uint64_t wt_addr  = weight->dma_addr + plan->weight_off[ti];
        uint64_t out_addr = output->dma_addr + plan->output_off[ti];
        rknpu2_matmul_open::detail::patch_dcomp(plan, ti, nullptr);
        plan->cmd[plan->feat_idx[ti]]  = E(T_CNA, R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_addr);
        plan->cmd[plan->dcomp_idx[ti]] = E(T_CNA, R_CNA_DCOMP_ADDR0,       (uint32_t)wt_addr);
        plan->cmd[plan->dst_idx[ti]]   = E(T_DPU, R_DST_BASE_ADDR,         (uint32_t)out_addr);
    }
    return rknpu2_matmul_open::detail::do_submit(ctx->fd, plan->task_obj_addr, plan->num_tasks,
                            plan->iommu_domain_id);
}

namespace rknpu2_matmul_open::detail {

static void patch_dcomp(rk_npu_matmul_i8_plan* plan, int ti,
                        const I8CompressedTile* tile) {
    const int base = plan->base[ti], addr = plan->dcomp_idx[ti];
    const uint32_t ctrl = tile ? 1u : 0u;
    const uint32_t regnum = tile ? DCOMP_NORMAL_REGNUM : 0u;
    plan->cmd[base] = E(T_CNA, R_CNA_DCOMP_REGNUM, regnum);
    plan->cmd[base + 1] = E(T_CNA, R_CNA_DCOMP_CTRL, ctrl);
    plan->cmd[addr - 2] = E(T_CNA, R_CNA_DCOMP_CTRL, ctrl);
    plan->cmd[addr - 1] = E(T_CNA, R_CNA_DCOMP_REGNUM, regnum);
    for (int lane = 0; lane < 16; ++lane)
        plan->cmd[addr + 1 + lane] = E(T_CNA, R_CNA_DCOMP_AMOUNT0 + lane * 4,
                                      tile ? tile->amounts[lane] : 0u);
}

int bind_i8_core_mask_n_split_io(rk_npu_matmul_i8_plan* plan,
                                 rk_npu_mem* input, rk_npu_mem* output,
                                 uint32_t core_mask) {
    const int core_count = __builtin_popcount(core_mask & 7u);
    if (!plan || !plan->ctx || !input || !output ||
        input->ctx_id != plan->ctx->id || output->ctx_id != plan->ctx->id ||
        core_mask == 0 || (core_mask & ~7u) != 0 || core_count < 1 || core_count > 3)
        return RK_NPU_ERR_PARAM;
    if (input->iommu_domain_id != plan->iommu_domain_id ||
        output->iommu_domain_id != plan->iommu_domain_id)
        return RK_NPU_ERR_DOMAIN;
    if (output->size < plan->output_bytes) return RK_NPU_ERR_NOMEM;

    const Layout L = gemm_layout(plan->cfg.N, plan->cfg.K);
    const int m_tile = m_tile_for(L, plan->cfg);
    const int n_tile = plan->n_tile;
    const int m_tasks = ceil_div(plan->cfg.M, m_tile);
    const int n_groups = ceil_div(L.align_out, n_tile);
    if (n_groups < core_count || m_tasks * n_groups != plan->num_tasks)
        return RK_NPU_ERR_PARAM;

    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        const uint64_t in_addr  = input->dma_addr  + plan->input_off[ti];
        const uint64_t out_addr = output->dma_addr + plan->output_off[ti];
        plan->cmd[plan->feat_idx[ti]]  = E(T_CNA, R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_addr);
        plan->cmd[plan->dst_idx[ti]]   = E(T_DPU, R_DST_BASE_ADDR,         (uint32_t)out_addr);
    }

    std::fill(std::begin(plan->prebound_task_start),
              std::end(plan->prebound_task_start), 0u);
    std::fill(std::begin(plan->prebound_task_count),
              std::end(plan->prebound_task_count), 0u);
    int task_cursor = 0;
    const int groups_per_core = n_groups / core_count;
    const int extra_groups = n_groups % core_count;
    for (int core = 0; core < core_count; ++core) {
        const int core_groups = groups_per_core + (core < extra_groups ? 1 : 0);
        plan->prebound_task_start[core] = (uint32_t)task_cursor;
        plan->prebound_task_count[core] = (uint32_t)(core_groups * m_tasks);
        task_cursor += (int)plan->prebound_task_count[core];
    }

    /* Rebuild every link so rebinding an existing plan to a different mask
     * cannot leave stale closed tails from an earlier core partition. */
    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        bool core_tail = false;
        for (int core = 0; core < core_count; ++core) {
            const uint32_t end = plan->prebound_task_start[core] +
                                 plan->prebound_task_count[core];
            if ((uint32_t)(ti + 1) == end) {
                core_tail = true;
                break;
            }
        }
        const int tail = plan->base[ti] + plan->body_size[ti];
        if (core_tail) {
            plan->cmd[tail + 0] = E(T_NOP, 0, 0);
            plan->cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, 0);
            plan->cmd[tail + 2] = E(T_VERSION, 0, 0);
        } else {
            const uint64_t next_addr = plan->regcmd.dma_addr +
                                       (uint64_t)plan->base[ti + 1] * 8;
            plan->cmd[tail + 0] = E(T_PC_REG, R_PC_BASE_ADDRESS,
                                    (uint32_t)(next_addr & 0xfffffff0));
            plan->cmd[tail + 1] = E(
                T_PC_REG, R_PC_REGISTER_AMOUNTS,
                ceil_div(plan->body_size[ti + 1], 2) + 1);
            plan->cmd[tail + 2] = E(T_VERSION, 0, 0);
        }
        /* tail+3 is OPERATION_ENABLE and remains unchanged. */
    }
    plan->prebound_core_mask = core_mask;
    return RK_NPU_OK;
}

int run_i8_core_mask_n_split_prebound(rk_npu_ctx* ctx,
                                      rk_npu_matmul_i8_plan* plan,
                                      rk_npu_mem* weight,
                                      uint32_t timeout_ms) {
    if (!ctx || !plan || ctx != plan->ctx || !weight ||
        weight->ctx_id != ctx->id || plan->prebound_core_mask == 0 ||
        timeout_ms == 0)
        return RK_NPU_ERR_PARAM;
    if (weight->iommu_domain_id != plan->iommu_domain_id)
        return RK_NPU_ERR_DOMAIN;

    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        const uint64_t wt_addr = weight->dma_addr + plan->weight_off[ti];
        patch_dcomp(plan, ti, nullptr);
        plan->cmd[plan->dcomp_idx[ti]] =
            E(T_CNA, R_CNA_DCOMP_ADDR0, (uint32_t)wt_addr);
    }

    return do_submit_multicore(ctx->fd, plan->task_obj_addr, plan->num_tasks,
                               plan->prebound_core_mask,
                               plan->prebound_task_start,
                               plan->prebound_task_count,
                               plan->iommu_domain_id, timeout_ms);
}

int run_i8_compressed_prebound(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
                               const rk_npu_mem* arena,
                               const std::vector<I8CompressedTile>& tiles,
                               uint32_t timeout_ms) {
    if (!ctx || !plan || ctx != plan->ctx || !arena || arena->ctx_id != ctx->id ||
        plan->prebound_core_mask == 0 || timeout_ms == 0) return RK_NPU_ERR_PARAM;
    if (arena->iommu_domain_id != plan->iommu_domain_id) return RK_NPU_ERR_DOMAIN;
    const auto layout = gemm_layout(plan->cfg.N, plan->cfg.K);
    const int groups = ceil_div(layout.align_out, plan->n_tile);
    if (tiles.size() != size_t(groups)) return RK_NPU_ERR_PARAM;
    for (const auto& tile : tiles)
        if (tile.offset > arena->size || tile.bytes > arena->size - tile.offset ||
            ((arena->dma_addr + tile.offset) & 63)) return RK_NPU_ERR_PARAM;
    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        const size_t group = plan->weight_off[ti] / (uint64_t(layout.align_in) * plan->n_tile);
        if (group >= tiles.size()) return RK_NPU_ERR_PARAM;
        const auto& tile = tiles[group];
        patch_dcomp(plan, ti, tile.compressed ? &tile : nullptr);
        plan->cmd[plan->dcomp_idx[ti]] = E(T_CNA, R_CNA_DCOMP_ADDR0,
                                         uint32_t(arena->dma_addr + tile.offset));
    }
    return do_submit_multicore(ctx->fd, plan->task_obj_addr, plan->num_tasks,
                               plan->prebound_core_mask, plan->prebound_task_start,
                               plan->prebound_task_count, plan->iommu_domain_id, timeout_ms);
}

int run_i8_core_mask_n_split(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
                             rk_npu_mem* input, rk_npu_mem* weight,
                             rk_npu_mem* output, uint32_t core_mask,
                             uint32_t timeout_ms) {
    if (!ctx || !plan || ctx != plan->ctx || !weight ||
        weight->ctx_id != ctx->id)
        return RK_NPU_ERR_PARAM;
    int rc = bind_i8_core_mask_n_split_io(plan, input, output, core_mask);
    if (rc != RK_NPU_OK) return rc;
    return run_i8_core_mask_n_split_prebound(ctx, plan, weight, timeout_ms);
}

} /* namespace rknpu2_matmul_open::detail */

void rknpu2_matmul_open::detail::free_i8_plan(rk_npu_matmul_i8_plan* plan) {
    if (!plan) return;
    if (plan->ctx) {
        if (plan->regcmd.handle) rk_npu_mem_free(plan->ctx, &plan->regcmd);
        if (plan->task.handle) rk_npu_mem_free(plan->ctx, &plan->task);
    }
    delete plan;
}

extern "C" rk_npu_matmul_i8_batch_plan* rk_npu_matmul_i8_batch_prepare(
    rk_npu_iommu_domain* domain, int B, const rk_npu_matmul_i8_config* cfg) {
    if (!domain || !domain->ctx || B<=0 || !valid_config(cfg)) return nullptr;
    rk_npu_ctx* ctx = domain->ctx;
    rk_npu_matmul_sizes sz;
    if (rk_npu_matmul_i8_batch_query(B, cfg, &sz) != RK_NPU_OK) return nullptr;

    rk_npu_matmul_i8_batch_plan* p = new rk_npu_matmul_i8_batch_plan();
    p->ctx = ctx;
    p->iommu_domain_id = domain->id;
    p->batch_count = B;
    p->cfg = *cfg;
    if (rk_npu_mem_alloc(domain, sz.regcmd_bytes, RK_NPU_MEM_NON_CACHEABLE, &p->regcmd) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sz.task_bytes, RK_NPU_MEM_KERNEL_MAPPING, &p->task) != RK_NPU_OK) {
        rk_npu_matmul_i8_batch_plan_free(p);
        return nullptr;
    }

    BuildMeta meta;
    if (build_regcmd(*cfg, 0, 0, 0, &p->regcmd, &p->task, &meta, B) != RK_NPU_OK) {
        rk_npu_matmul_i8_batch_plan_free(p);
        return nullptr;
    }

    p->num_tasks        = meta.num_tasks;
    p->input_row_bytes  = meta.input_row_bytes;
    p->row_stride_bytes = meta.row_stride_bytes;
    p->output_bytes     = meta.output_bytes;
    p->task_obj_addr    = p->task.obj_addr;
    p->cmd              = (uint64_t*)p->regcmd.vaddr;
    p->input_off        = meta.input_off;
    p->weight_off       = meta.weight_off;
    p->output_off       = meta.output_off;
    if (!find_patch_indices(p->cmd, meta.base, I8_BODY_REGS, p->feat_idx, p->dcomp_idx, p->dst_idx)) {
        rk_npu_matmul_i8_batch_plan_free(p);
        return nullptr;
    }
    return p;
}

extern "C" int rk_npu_matmul_i8_batch_run(rk_npu_ctx* ctx, rk_npu_matmul_i8_batch_plan* plan,
                                          rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output) {
    if (!ctx || !plan || ctx != plan->ctx || !input || !weight || !output ||
        input->ctx_id != ctx->id || weight->ctx_id != ctx->id ||
        output->ctx_id != ctx->id) return RK_NPU_ERR_PARAM;
    if (input->iommu_domain_id != plan->iommu_domain_id ||
        weight->iommu_domain_id != plan->iommu_domain_id ||
        output->iommu_domain_id != plan->iommu_domain_id)
        return RK_NPU_ERR_DOMAIN;
    if (output->size < plan->output_bytes) return RK_NPU_ERR_NOMEM;

    for (int ti = 0; ti < plan->num_tasks; ++ti) {
        /* Offsets already include both the batch block stride and the intra-M
         * tile row offset.  Patching the full PC-chain here is the only per-run
         * address work before the single submit. */
        uint64_t in_addr  = input->dma_addr  + plan->input_off[ti];
        uint64_t wt_addr  = weight->dma_addr + plan->weight_off[ti];
        uint64_t out_addr = output->dma_addr + plan->output_off[ti];
        plan->cmd[plan->feat_idx[ti]]  = E(T_CNA, R_CNA_FEATURE_DATA_ADDR, (uint32_t)in_addr);
        plan->cmd[plan->dcomp_idx[ti]] = E(T_CNA, R_CNA_DCOMP_ADDR0,       (uint32_t)wt_addr);
        plan->cmd[plan->dst_idx[ti]]   = E(T_DPU, R_DST_BASE_ADDR,         (uint32_t)out_addr);
    }
    return rknpu2_matmul_open::detail::do_submit(ctx->fd, plan->task_obj_addr, plan->num_tasks,
                            plan->iommu_domain_id);
}

extern "C" void rk_npu_matmul_i8_batch_plan_free(rk_npu_matmul_i8_batch_plan* plan) {
    if (!plan) return;
    if (plan->ctx) {
        if (plan->regcmd.handle) rk_npu_mem_free(plan->ctx, &plan->regcmd);
        if (plan->task.handle) rk_npu_mem_free(plan->ctx, &plan->task);
    }
    delete plan;
}

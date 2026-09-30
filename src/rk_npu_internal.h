/*
 * rk_npu_internal.h - PRIVATE shared internals (not installed, not a public API).
 * ============================================================================
 * The rknpu DRM ioctl ABI, the register-command word encoding, the device
 * context, and the PC-chain / submit helpers that every datapath module
 * (int8 matmul, fp16 fused matmul) links against via rk_npu_core.cpp.
 *
 * Register-command model (shared by all ops)
 * ------------------------------------------
 * The NPU executes a "task": a contiguous array of 64-bit register-write words
 * in a dmabuf, terminated by a short PC-chain tail.  Each word is
 *
 *     word = (target << 48) | (value << 16) | reg_offset
 *
 * where `target` selects a hardware block (CNA/CORE/DPU/RDMA/PC/...).  Multiple
 * tasks are PC-chained (each tail points at the next task's regcmd address); the
 * driver walks the chain in one submit.  We use that for M (row) tiling.
 */
#ifndef RK_NPU_INTERNAL_H
#define RK_NPU_INTERNAL_H

#include "rk_npu_common.h"
#include <atomic>
#include <stdint.h>
#include <vector>

/* ----------------------------------------------------------- DRM ioctl ABI */
#pragma pack(push, 1)
struct rknpu_mem_create { uint32_t handle; uint32_t flags; uint64_t size;
                          uint64_t obj_addr; uint64_t dma_addr; uint64_t sram_size;
                          int32_t iommu_domain_id; uint32_t core_mask; };
struct rknpu_mem_map    { uint32_t handle; uint32_t reserved; uint64_t offset; };
struct rknpu_mem_destroy { uint32_t handle; uint32_t reserved; uint64_t obj_addr; };
struct rknpu_mem_sync { uint32_t flags; uint32_t reserved; uint64_t obj_addr;
                        uint64_t offset; uint64_t size; };
struct drm_prime_handle { uint32_t handle; uint32_t flags; int32_t fd; };
struct drm_gem_close { uint32_t handle; uint32_t pad; };
struct rknpu_action     { uint32_t flags; uint32_t value; };
struct rknpu_subcore_task { uint32_t task_start; uint32_t task_number; };
struct rknpu_submit {
    uint32_t flags; uint32_t timeout; uint32_t task_start; uint32_t task_number;
    uint32_t task_counter; int32_t priority; uint64_t task_obj_addr;
    uint32_t iommu_domain_id; uint32_t reserved; uint64_t task_base_addr;
    int64_t hw_elapse_time; uint32_t core_mask; int32_t fence_fd;
    rknpu_subcore_task subcore_task[5];
};
struct rknpu_task {
    uint32_t flags; uint32_t op_idx; uint32_t enable_mask; uint32_t int_mask;
    uint32_t int_clear; uint32_t int_status; uint32_t regcfg_amount;
    uint32_t regcfg_offset; uint64_t regcmd_addr;
};
#pragma pack(pop)

/* Official rknpu action enum. */
constexpr uint32_t RKNPU_GET_DRV_VERSION = 1;
constexpr uint32_t RKNPU_ACT_RESET    = 6;
constexpr uint32_t RKNPU_GET_IOMMU_EN = 18;
constexpr uint32_t RKNPU_JOB_PC       = 1u << 0;
constexpr uint32_t RKNPU_JOB_PINGPONG = 1u << 2;

/* _IOWR('d', nr, sizeof(T)) with the standard _IOC bit layout. */
template <typename T>
constexpr unsigned long rk_iowr(unsigned nr) {
    return (3ul << 30) | ((unsigned long)sizeof(T) << 16) | ((unsigned long)'d' << 8) | nr;
}
template <typename T>
constexpr unsigned long rk_iow(unsigned nr) {
    return (1ul << 30) | ((unsigned long)sizeof(T) << 16) |
           ((unsigned long)'d' << 8) | nr;
}
#define IOCTL_ACTION      rk_iowr<rknpu_action>(0x40)
#define IOCTL_SUBMIT      rk_iowr<rknpu_submit>(0x41)
#define IOCTL_MEM_CREATE  rk_iowr<rknpu_mem_create>(0x42)
#define IOCTL_MEM_MAP     rk_iowr<rknpu_mem_map>(0x43)
#define IOCTL_MEM_DESTROY rk_iowr<rknpu_mem_destroy>(0x44)
#define IOCTL_MEM_SYNC    rk_iowr<rknpu_mem_sync>(0x45)
#define IOCTL_GEM_CLOSE   rk_iow<drm_gem_close>(0x09)
#define IOCTL_PRIME_HANDLE_TO_FD rk_iowr<drm_prime_handle>(0x2d)
#define IOCTL_PRIME_FD_TO_HANDLE rk_iowr<drm_prime_handle>(0x2e)

/* ----------------------------------------------------- register encoding */

/* target stream IDs (bits 48..63 of each register word) */
constexpr uint64_t T_CNA = 0x0201, T_CORE = 0x0801, T_DPU = 0x1001, T_RDMA = 0x2001;
constexpr uint64_t T_PC  = 0x0081, T_PC_REG = 0x0101, T_VERSION = 0x0041, T_NOP = 0x0001;

/* register offsets shared across datapaths */
enum : uint32_t {
    R_OPERATION_ENABLE = 0x0008, R_PC_BASE_ADDRESS = 0x0010, R_PC_REGISTER_AMOUNTS = 0x0014,
    /* DPU (0x40xx) */
    R_S_POINTER = 0x4004, R_FEATURE_MODE_CFG = 0x400c, R_DATA_FORMAT = 0x4010,
    R_DST_BASE_ADDR = 0x4020, R_DST_SURF_STRIDE = 0x4024,
    R_DATA_CUBE_WIDTH = 0x4030, R_DATA_CUBE_HEIGHT = 0x4034,
    R_DATA_CUBE_NOTCH = 0x4038, R_DATA_CUBE_CHANNEL = 0x403c,
    R_BS_CFG = 0x4040, R_BS_OW_CFG = 0x4050, R_WDMA_SIZE_0 = 0x4058, R_WDMA_SIZE_1 = 0x405c,
    R_BN_CFG = 0x4060,
    R_EW_CFG = 0x4070, R_EW_CVT_OFFSET = 0x4074, R_EW_CVT_SCALE = 0x4078, R_EW_RELUX = 0x407c,
    R_OUT_CVT_SCALE = 0x4084, R_SURFACE_ADD = 0x40c0,
    /* DPU-RDMA (0x50xx) -- the fused-op second operand read engine */
    R_RDMA_S_POINTER = 0x5004, R_RDMA_W = 0x500c, R_RDMA_H = 0x5010, R_RDMA_C = 0x5014,
    R_RDMA_SRC_BASE = 0x5018, R_RDMA_BRDMA_CFG = 0x501c, R_RDMA_BS_BASE = 0x5020,
    R_RDMA_NRDMA_CFG = 0x5028, R_RDMA_BN_BASE = 0x502c, R_RDMA_ERDMA_CFG = 0x5034,
    R_RDMA_EW_BASE = 0x5038, R_RDMA_EW_SURF_STRIDE = 0x5040, R_RDMA_FEATURE_MODE_CFG = 0x5044,
    R_RDMA_SRC_DMA_CFG = 0x5048, R_RDMA_SURF_NOTCH = 0x504c, R_RDMA_PAD_CFG = 0x5064,
    R_RDMA_WEIGHT = 0x5068, R_RDMA_EW_SURF_NOTCH = 0x506c,
    /* CNA (0x10xx) */
    R_CNA_CONV_CON1 = 0x100c, R_CNA_CONV_CON2 = 0x1010, R_CNA_CONV_CON3 = 0x1014,
    R_CNA_DATA_SIZE0 = 0x1020, R_CNA_DATA_SIZE1 = 0x1024, R_CNA_DATA_SIZE2 = 0x1028, R_CNA_DATA_SIZE3 = 0x102c,
    R_CNA_WEIGHT_SIZE0 = 0x1030, R_CNA_WEIGHT_SIZE1 = 0x1034, R_CNA_WEIGHT_SIZE2 = 0x1038,
    R_CNA_CBUF_CON0 = 0x1040, R_CNA_CBUF_CON1 = 0x1044,
    R_CNA_CVT_CON0 = 0x104c, R_CNA_CVT_CON1 = 0x1050, R_CNA_CVT_CON2 = 0x1054,
    R_CNA_CVT_CON3 = 0x1058, R_CNA_CVT_CON4 = 0x105c,
    R_CNA_FEATURE_DATA_ADDR = 0x1070, R_CNA_DMA_CON0 = 0x1078, R_CNA_DMA_CON1 = 0x107c, R_CNA_DMA_CON2 = 0x1080,
    R_CNA_FC_DATA_SIZE0 = 0x1084, R_CNA_FC_DATA_SIZE1 = 0x1088,
    R_CNA_DCOMP_CTRL = 0x1100, R_CNA_DCOMP_REGNUM = 0x1104, R_CNA_DCOMP_ADDR0 = 0x1110,
    R_CNA_DCOMP_AMOUNT0 = 0x1140,
    /* CORE (0x30xx) */
    R_CORE_MISC_CFG = 0x3010, R_CORE_DATAOUT_SIZE_0 = 0x3014, R_CORE_DATAOUT_SIZE_1 = 0x3018, R_CORE_RESERVED_3030 = 0x3030,
};

/* CBUF / tiling geometry (NVDLA-derived) */
constexpr int MIN_CHANNEL_TILE         = 32;
constexpr int CBUF_BANK_SIZE           = 256 * 128;   /* 32768 bytes */
constexpr int RK_CBUF_BANKS            = 12;
constexpr int RK_LINE_STRIDE_GROUP_CAP = 13;
constexpr int FEATURE_ATOMIC           = 16;          /* output channels per surface */

inline uint64_t E(uint64_t target, uint32_t reg, uint32_t value) {
    return (target << 48) | ((uint64_t)value << 16) | reg;
}
inline int ceil_div(int x, int y) { return (x + y - 1) / y; }
inline int align_up(int x, int a) { return ceil_div(x, a) * a; }

/* ------------------------------------------------------------- context */
struct rk_npu_ctx {
    int fd;
    uint64_t id;
    uint32_t driver_version;
    bool iommu_enabled;
};
struct rk_npu_iommu_domain {
    rk_npu_ctx* ctx;
    uint32_t id;
    std::atomic<uint32_t> refs;
};
struct rk_npu_matmul_i8_config;
struct rk_npu_matmul_i8_plan;

/* ----------------------------------------------- shared core helpers */
namespace rknpu2_matmul_open::detail {

/* Private single-GEMM INT8 surface used by typed plans and low-level tests. */
int query_i8(const rk_npu_matmul_i8_config* cfg, rk_npu_matmul_sizes* out);
int pack_i8_a(const rk_npu_matmul_i8_config* cfg, const int8_t* A,
              rk_npu_mem* input);
int pack_i8_b(const rk_npu_matmul_i8_config* cfg, const int8_t* B,
              rk_npu_mem* weight);
int unpack_i8_c(const rk_npu_matmul_i8_config* cfg,
                const rk_npu_mem* output, void* C_raw4);
rk_npu_matmul_i8_plan* prepare_i8(rk_npu_iommu_domain* domain,
                                  const rk_npu_matmul_i8_config* cfg);
int run_i8(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
           rk_npu_mem* input, rk_npu_mem* weight, rk_npu_mem* output);
void free_i8_plan(rk_npu_matmul_i8_plan* plan);

struct I8CompressedTile {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    bool compressed = true;
    uint32_t amounts[16] = {};
};
int run_i8_compressed_prebound(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
                               const rk_npu_mem* arena,
                               const std::vector<I8CompressedTile>& tiles,
                               uint32_t timeout_ms);

/* Experimental grouped GEMM builder: independent shapes/weights, explicit
 * physical-core ownership. Caller owns all data and the reusable command/task
 * arenas, and performs data cache synchronization. Does not submit or allocate
 * device memory. Used by tools/ling_moe_probe; not a public ABI. */
struct I8GroupItem {
    const rk_npu_matmul_i8_config* cfg;
    const rk_npu_mem* input;
    const rk_npu_mem* weight;
    const rk_npu_mem* output;
    int core;
};
int build_i8_group(const std::vector<I8GroupItem>& items,
                   rk_npu_mem* regcmd, rk_npu_mem* task,
                   uint32_t task_start[3], uint32_t task_count[3]);

/* Cacheable page-backed native GEM buffer in one explicit IOMMU domain.
 * CPU/device ownership is represented by RKNPU_MEM_SYNC calls around each CPU
 * access interval. */
struct DomainDataBuffer {
    enum class CpuAccess : uint8_t {
        None = 0,
        Read = 1,
        Write = 2,
        ReadWrite = 3,
    };

    rk_npu_mem mem{};
    rk_npu_iommu_domain* domain = nullptr;
    CpuAccess cpu_access = CpuAccess::None;

    DomainDataBuffer() = default;
    DomainDataBuffer(const DomainDataBuffer&) = delete;
    DomainDataBuffer& operator=(const DomainDataBuffer&) = delete;
    DomainDataBuffer(DomainDataBuffer&& other) noexcept;
    DomainDataBuffer& operator=(DomainDataBuffer&& other) noexcept;

    int alloc(rk_npu_iommu_domain* domain, uint64_t size);
    int begin_cpu_read();
    int begin_cpu_write();
    int begin_cpu_readwrite();
    int end_cpu_access();
    int release(rk_npu_ctx* ctx);
};

/* reset + blocking PC-chain ping-pong submit; timeout in ms. */
int do_submit(int fd, uint64_t task_obj_addr, int num_tasks,
              uint32_t iommu_domain_id, uint32_t timeout_ms = 6000);

/* Explicit physical-core submit.  Ranges are contiguous task-array intervals
 * in increasing selected-core order.  Supported masks are 1/2/4, 3 and 7;
 * RK3588's vendor ABI reads subcore_task[2..4] for the three-core case. */
int do_submit_multicore(int fd, uint64_t task_obj_addr, int num_tasks,
                        uint32_t core_mask, const uint32_t task_start[3],
                        const uint32_t task_count[3], uint32_t iommu_domain_id,
                        uint32_t timeout_ms = 500);

/* Private N-tiled INT8 plan surface.  N tiling is an execution-strategy detail,
 * not a register-tuning ABI, so production/autotune code uses these helpers
 * while the ordinary public matmul API always selects the full aligned N. */
int query_i8_n_tiled(const rk_npu_matmul_i8_config* cfg, int n_tile,
                     rk_npu_matmul_sizes* out);
rk_npu_matmul_i8_plan* prepare_i8_n_tiled(
    rk_npu_iommu_domain* domain, const rk_npu_matmul_i8_config* cfg, int n_tile);

/* Copy one prebound N group's existing GEMM recipe for a mixed task chain.
 * B stays relocatable; A/C addresses and all layout registers are preserved. */
struct I8GemmBody {
    std::vector<uint64_t> regs;
    uint64_t weight_offset = 0;
};
int export_i8_n_group(const rk_npu_matmul_i8_plan* plan, int group,
                       std::vector<I8GemmBody>& bodies);

void retain_domain(rk_npu_iommu_domain* domain);
void release_domain(rk_npu_iommu_domain* domain);

/* Execute the contiguous N groups of an internally N-tiled plan on the
 * physical NPU cores selected by bits 0..2 of core_mask. */
int run_i8_core_mask_n_split(rk_npu_ctx* ctx, rk_npu_matmul_i8_plan* plan,
                             rk_npu_mem* input, rk_npu_mem* weight,
                             rk_npu_mem* output, uint32_t core_mask,
                             uint32_t timeout_ms = 500);
int bind_i8_core_mask_n_split_io(rk_npu_matmul_i8_plan* plan,
                                 rk_npu_mem* input, rk_npu_mem* output,
                                 uint32_t core_mask);
int run_i8_core_mask_n_split_prebound(rk_npu_ctx* ctx,
                                      rk_npu_matmul_i8_plan* plan,
                                      rk_npu_mem* weight,
                                      uint32_t timeout_ms = 500);

/*
 * How a task's PC-chain tail is written.  The tail's OPERATION_ENABLE selects
 * which hardware units run:
 *   plain matmul:  op_enable = (6<<1)|1 = 0xd   (CNA+CORE+DPU)
 *   fused matmul:  op_enable = 0x1d             (+ the DPU-RDMA operand engine)
 * The driver's per-task descriptor also carries op_idx / enable_mask which must
 * agree (0/0xd plain, 1/0x1d fused).
 */
struct ChainCfg { uint32_t op_enable; uint32_t op_idx; uint32_t enable_mask; };

/*
 * Write `bodies` (one register vector per task) into `cmd` at the precomputed
 * qword offsets `base`, append each task's 4-qword PC-chain tail, and fill the
 * `tk` task descriptors.  `regcmd_dma` is the regcmd buffer's NPU address.
 */
void write_chain(uint64_t* cmd, rknpu_task* tk, uint64_t regcmd_dma,
                 const std::vector<std::vector<uint64_t>>& bodies,
                 const std::vector<int>& base, const ChainCfg& cfg);

} /* namespace rknpu2_matmul_open::detail */

#endif /* RK_NPU_INTERNAL_H */

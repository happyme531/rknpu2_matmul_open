/*
 * rk_npu_common - shared C ABI types for the rk_npu_* NPU libraries
 * ================================================================
 * Device context, dmabuf wrapper, error codes and the device/memory helpers
 * that are common to every datapath (int8 matmul, fp16 fused matmul, ...).
 *
 * The library talks to the RK3588 NPU directly through the rknpu DRM driver
 * (/dev/dri/cardN) -- no librknnrt.so, no RKNN runtime, no compiler.  Each
 * datapath lives in its own header/translation unit:
 *
 *     rk_npu_matmul.h      int8 x int8 -> fp32 matmul
 *     rk_npu_matmul_f16.h  fp16 x fp16 -> fp16 matmul, optionally fused with an
 *                          elementwise MUL/ADD against a third operand (ConvMul/ConvAdd)
 *     rk_npu_add_rmsnorm_f16.h  fp16 native residual Add + RMSNorm
 */
#ifndef RK_NPU_COMMON_H
#define RK_NPU_COMMON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- error codes ---- */
#define RK_NPU_OK            0
#define RK_NPU_ERR_OPEN     (-1)   /* could not open the DRM device          */
#define RK_NPU_ERR_IOCTL    (-2)   /* an ioctl failed (see errno)            */
#define RK_NPU_ERR_PARAM    (-3)   /* bad argument                           */
#define RK_NPU_ERR_NOMEM    (-4)   /* a provided buffer is too small         */
#define RK_NPU_ERR_SUBMIT   (-5)   /* the NPU job failed / timed out         */
#define RK_NPU_ERR_BUSY     (-6)   /* a non-reentrant workspace is in use    */
#define RK_NPU_ERR_DOMAIN   (-7)   /* objects belong to different IOMMU domains */
#define RK_NPU_ERR_IO       (-8)   /* ordinary file I/O failed               */
#define RK_NPU_ERR_CACHE_MISS (-9) /* tuning cache absent, stale, or invalid */

/* ---- allocation flags for rk_npu_mem_alloc (mirror the driver's flags) ---- */
#define RK_NPU_MEM_NON_CONTIGUOUS  0x1   /* page-backed allocation mapped by the IOMMU */
#define RK_NPU_MEM_NON_CACHEABLE   0x0   /* uncached CPU mapping; no cache maintenance needed */
#define RK_NPU_MEM_CACHEABLE       0x2   /* cached CPU mapping; call rk_npu_mem_sync around NPU use */
#define RK_NPU_MEM_KERNEL_MAPPING  0x8   /* required for the task buffer                        */
#define RK_NPU_MEM_IOMMU           0x10  /* explicitly request IOMMU-visible storage           */

/* CACHEABLE alone selects the driver's force-contiguous dma_alloc_attrs path,
 * whose CPU reads are slow on RK3588.  Production CPU-visible data uses the
 * page-backed GEM path selected by NON_CONTIGUOUS. */
#define RK_NPU_MEM_DATA_DEFAULT \
    (RK_NPU_MEM_NON_CONTIGUOUS | RK_NPU_MEM_CACHEABLE | RK_NPU_MEM_IOMMU)
#define RK_NPU_MEM_DEFAULT RK_NPU_MEM_DATA_DEFAULT

#define RK_NPU_IOMMU_DOMAIN_COUNT 16u

/* ---- rk_npu_mem.flags ownership/state bits ---- */
#define RK_NPU_MEM_F_OWN_RKNPU     (1u << 0)  /* owns an rknpu GEM/object registration */
#define RK_NPU_MEM_F_IMPORTED      (1u << 1)  /* registration came from an external dmabuf */
#define RK_NPU_MEM_F_VIEW          (1u << 2)  /* non-owning subrange view */
#define RK_NPU_MEM_F_MAPPED        (1u << 3)  /* vaddr is valid */
#define RK_NPU_MEM_F_CACHEABLE     (1u << 4)  /* CPU mapping is cacheable; explicit sync required */

typedef enum {
    RK_NPU_SYNC_TO_DEVICE   = 1u << 0,
    RK_NPU_SYNC_FROM_DEVICE = 1u << 1,
} rk_npu_sync_dir;

/* Opaque device context. */
typedef struct rk_npu_ctx rk_npu_ctx;
typedef struct rk_npu_iommu_domain rk_npu_iommu_domain;

/*
 * A view of NPU-visible memory. Base memories own an rknpu registration; views
 * are non-owning subranges over a base memory. The underlying dmabuf can be
 * allocated as native GEM storage by this library or imported from its owner.
 *   handle   - rknpu GEM handle in this rk_npu_ctx
 *   flags    - RK_NPU_MEM_F_* state/ownership bits
 *   obj_addr - kernel object address (used as the submit's task object for task buffers)
 *   dma_addr - NPU/IOMMU address for this view
 *   vaddr    - userspace mapping for this view
 *   size     - byte size of this view
 *   offset   - byte offset from the base object
 *   capacity - byte size of the base object
 *   ctx_id   - owning rk_npu_ctx cookie; one rk_npu_mem is not cross-context
 *   iommu_domain_id - 32-bit IOVA space containing this allocation/view
 */
typedef struct {
    uint32_t handle;
    uint32_t flags;
    uint64_t obj_addr;
    uint64_t dma_addr;
    void*    vaddr;
    uint64_t size;
    uint64_t offset;
    uint64_t capacity;
    uint64_t ctx_id;
    uint32_t iommu_domain_id;
    uint32_t reserved;
} rk_npu_mem;

/* Buffer sizes required for one matmul (filled by each datapath's *_query). */
typedef struct {
    uint64_t input_bytes;    /* packed A  (feature)            */
    uint64_t weight_bytes;   /* packed B  (weights)            */
    uint64_t operand_bytes;  /* packed C0 (fused operand) or 0 */
    uint64_t output_bytes;   /* packed C  (result)             */
    uint64_t regcmd_bytes;   /* register command stream        */
    uint64_t task_bytes;     /* task descriptor array          */
    int      num_tasks;      /* PC-chained tasks (M tiling)    */
} rk_npu_matmul_sizes;

/* ---------------------------------------------------------------- device */

/* Open the NPU. dev may be NULL to use the default ("/dev/dri/card1").
 * Returns NULL on failure. */
rk_npu_ctx* rk_npu_open(const char* dev);
void        rk_npu_close(rk_npu_ctx* ctx);

/* Reference one of the driver's sixteen independent 32-bit IOVA spaces.
 * Nonzero domains require rknpu 0.9.8+ with IOMMU enabled.  Objects created
 * from a domain retain it internally; ctx must still outlive every object. */
rk_npu_iommu_domain* rk_npu_iommu_domain_create(rk_npu_ctx* ctx,
                                                 uint32_t domain_id);
void rk_npu_iommu_domain_free(rk_npu_iommu_domain* domain);
uint32_t rk_npu_iommu_domain_id(const rk_npu_iommu_domain* domain);

/* Human-readable string for an RK_NPU_ERR_* code. */
const char* rk_npu_strerror(int code);

/* ---------------------------------------------------------------- memory */

/* Allocate an NPU-visible dmabuf owned by this library. alloc_flags is a bitmask
 * of RK_NPU_MEM_* driver allocation flags.  Cacheable buffers need explicit
 * rk_npu_mem_sync(..., TO_DEVICE) after CPU writes and
 * rk_npu_mem_sync(..., FROM_DEVICE) before CPU reads. */
int rk_npu_mem_alloc(rk_npu_iommu_domain* domain, uint64_t size,
                     uint32_t alloc_flags, rk_npu_mem* out);

/* Import an existing dmabuf into one NPU IOMMU domain.  import_flags is a
 * bitmask of RK_NPU_MEM_* driver flags.  The caller keeps
 * ownership of dmabuf_fd and cpu_addr.  cpu_addr may be NULL for device-only
 * use; otherwise it must map at least size bytes and remains valid until
 * rk_npu_mem_free().  The imported object owns only its RKNPU registration.
 * The external owner must perform its heap's CPU/device cache-ownership
 * protocol (for example DMA_BUF_IOCTL_SYNC); rk_npu_mem_sync() is guaranteed
 * only for native RKNPU allocations. */
int rk_npu_mem_import_dmabuf(rk_npu_iommu_domain* domain, int dmabuf_fd,
                             void* cpu_addr, uint64_t size,
                             uint32_t import_flags, rk_npu_mem* out);

/* Export a base memory as a new dmabuf fd. Caller owns and must close *out_fd. */
int rk_npu_mem_export_dmabuf(rk_npu_ctx* ctx, const rk_npu_mem* mem, int* out_fd);

/* Create a non-owning subrange view. Views are valid for data/regcmd buffers;
 * task buffers should remain base allocations owned by the internal plan. */
int rk_npu_mem_view(const rk_npu_mem* base, uint64_t offset, uint64_t size, rk_npu_mem* out_view);

/* Synchronize CPU writes/reads for cacheable native buffers. */
int rk_npu_mem_sync(rk_npu_ctx* ctx, const rk_npu_mem* mem, rk_npu_sync_dir dir);

/* Release an owning base registration, or clear a non-owning view. */
int rk_npu_mem_free(rk_npu_ctx* ctx, rk_npu_mem* mem);

#ifdef __cplusplus
}
#endif
#endif /* RK_NPU_COMMON_H */

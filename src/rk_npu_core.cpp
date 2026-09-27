/*
 * rk_npu_core.cpp - shared device / memory / submit implementation.
 * Used by every datapath module (rk_npu_matmul_i8.cpp, rk_npu_matmul_f16.cpp).
 */
#include "rk_npu_internal.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <new>

namespace {

uint64_t ctx_id(const rk_npu_ctx* ctx) {
    return (uint64_t)(uintptr_t)ctx;
}

bool same_ctx(const rk_npu_ctx* ctx, const rk_npu_mem* mem) {
    return ctx && mem && (mem->ctx_id == 0 || mem->ctx_id == ctx_id(ctx));
}

int map_mem(rk_npu_ctx* ctx, rk_npu_mem* out) {
    rknpu_mem_map mm; std::memset(&mm, 0, sizeof(mm));
    mm.handle = out->handle;
    if (ioctl(ctx->fd, IOCTL_MEM_MAP, &mm) < 0) return RK_NPU_ERR_IOCTL;

    void* va = mmap(nullptr, out->capacity, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, (off_t)mm.offset);
    if (va == MAP_FAILED) return RK_NPU_ERR_IOCTL;
    out->vaddr = va;
    out->flags |= RK_NPU_MEM_F_MAPPED;
    return RK_NPU_OK;
}

void close_gem_handle(int fd, uint32_t handle) {
    if (!handle) return;
    drm_gem_close gc; std::memset(&gc, 0, sizeof(gc));
    gc.handle = handle;
    ioctl(fd, IOCTL_GEM_CLOSE, &gc);
}

} /* namespace */

/* ------------------------------------------------------------ device ---- */

extern "C" rk_npu_ctx* rk_npu_open(const char* dev) {
    if (!dev) dev = "/dev/dri/card1";
    int fd = ::open(dev, O_RDWR);
    if (fd < 0) return nullptr;
    rk_npu_ctx* ctx = (rk_npu_ctx*)std::calloc(1, sizeof(rk_npu_ctx));
    if (!ctx) { ::close(fd); return nullptr; }
    ctx->fd = fd;
    ctx->id = ctx_id(ctx);
    rknpu_action action{};
    action.flags = RKNPU_GET_DRV_VERSION;
    if (::ioctl(fd, IOCTL_ACTION, &action) == 0)
        ctx->driver_version = action.value;
    action = {};
    action.flags = RKNPU_GET_IOMMU_EN;
    if (::ioctl(fd, IOCTL_ACTION, &action) == 0)
        ctx->iommu_enabled = action.value != 0;
    return ctx;
}

extern "C" void rk_npu_close(rk_npu_ctx* ctx) {
    if (!ctx) return;
    if (ctx->fd >= 0) ::close(ctx->fd);
    std::free(ctx);
}

extern "C" const char* rk_npu_strerror(int code) {
    switch (code) {
        case RK_NPU_OK:          return "ok";
        case RK_NPU_ERR_OPEN:    return "could not open DRM device";
        case RK_NPU_ERR_IOCTL:   return "ioctl failed";
        case RK_NPU_ERR_PARAM:   return "bad argument";
        case RK_NPU_ERR_NOMEM:   return "buffer too small";
        case RK_NPU_ERR_SUBMIT:  return "NPU job failed/timed out";
        case RK_NPU_ERR_BUSY:    return "workspace is already running";
        case RK_NPU_ERR_DOMAIN:  return "IOMMU domain mismatch or unavailable";
        case RK_NPU_ERR_IO:      return "file I/O failed";
        case RK_NPU_ERR_CACHE_MISS: return "tuning cache miss";
        default:                 return "unknown error";
    }
}

extern "C" rk_npu_iommu_domain* rk_npu_iommu_domain_create(
    rk_npu_ctx* ctx, uint32_t domain_id) {
    if (!ctx || domain_id >= RK_NPU_IOMMU_DOMAIN_COUNT) return nullptr;
    /* Driver version encoding is major*10000 + minor*100 + patch. */
    if (domain_id != 0 && (!ctx->iommu_enabled || ctx->driver_version < 908))
        return nullptr;
    rk_npu_iommu_domain* domain = new (std::nothrow) rk_npu_iommu_domain();
    if (!domain) return nullptr;
    domain->ctx = ctx;
    domain->id = domain_id;
    domain->refs.store(1, std::memory_order_relaxed);
    return domain;
}

extern "C" void rk_npu_iommu_domain_free(rk_npu_iommu_domain* domain) {
    rknpu2_matmul_open::detail::release_domain(domain);
}

extern "C" uint32_t rk_npu_iommu_domain_id(
    const rk_npu_iommu_domain* domain) {
    return domain ? domain->id : UINT32_MAX;
}

/* ------------------------------------------------------------- memory ---- */

extern "C" int rk_npu_mem_alloc(rk_npu_iommu_domain* domain, uint64_t size,
                                  uint32_t alloc_flags, rk_npu_mem* out) {
    if (!domain || !domain->ctx || !out || size == 0) return RK_NPU_ERR_PARAM;
    rk_npu_ctx* ctx = domain->ctx;
    std::memset(out, 0, sizeof(*out));
    rknpu_mem_create mc; std::memset(&mc, 0, sizeof(mc));
    mc.flags = alloc_flags;
    mc.size = size;
    mc.iommu_domain_id = (int32_t)domain->id;
    if (ioctl(ctx->fd, IOCTL_MEM_CREATE, &mc) < 0) return RK_NPU_ERR_IOCTL;

    out->handle   = mc.handle;
    out->flags    = RK_NPU_MEM_F_OWN_RKNPU;
    if (alloc_flags & RK_NPU_MEM_CACHEABLE) out->flags |= RK_NPU_MEM_F_CACHEABLE;
    out->obj_addr = mc.obj_addr;
    out->dma_addr = mc.dma_addr;
    out->size     = mc.size;
    out->capacity = mc.size;
    out->ctx_id   = ctx->id;
    out->iommu_domain_id = domain->id;
    if (mc.dma_addr > UINT32_MAX || mc.size == 0 ||
        mc.size - 1 > UINT32_MAX - mc.dma_addr) {
        rk_npu_mem_free(ctx, out);
        return RK_NPU_ERR_DOMAIN;
    }
    int rc = map_mem(ctx, out);
    if (rc != RK_NPU_OK) {
        rk_npu_mem_free(ctx, out);
        return rc;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_mem_import_dmabuf(
    rk_npu_iommu_domain* domain, int dmabuf_fd, void* cpu_addr, uint64_t size,
    uint32_t import_flags, rk_npu_mem* out) {
    if (!domain || !domain->ctx || !out || dmabuf_fd < 0 || size == 0)
        return RK_NPU_ERR_PARAM;
    rk_npu_ctx* ctx = domain->ctx;
    std::memset(out, 0, sizeof(*out));

    drm_prime_handle ph; std::memset(&ph, 0, sizeof(ph));
    ph.fd = dmabuf_fd;
    if (ioctl(ctx->fd, IOCTL_PRIME_FD_TO_HANDLE, &ph) < 0)
        return RK_NPU_ERR_IOCTL;

    rknpu_mem_create mc; std::memset(&mc, 0, sizeof(mc));
    mc.handle = ph.handle;
    mc.flags = import_flags;
    mc.size = size;
    mc.iommu_domain_id = (int32_t)domain->id;
    if (ioctl(ctx->fd, IOCTL_MEM_CREATE, &mc) < 0) {
        close_gem_handle(ctx->fd, ph.handle);
        return RK_NPU_ERR_IOCTL;
    }

    out->handle = mc.handle;
    out->flags = RK_NPU_MEM_F_OWN_RKNPU | RK_NPU_MEM_F_IMPORTED;
    if (import_flags & RK_NPU_MEM_CACHEABLE)
        out->flags |= RK_NPU_MEM_F_CACHEABLE;
    out->obj_addr = mc.obj_addr;
    out->dma_addr = mc.dma_addr;
    out->vaddr = cpu_addr;
    if (cpu_addr) out->flags |= RK_NPU_MEM_F_MAPPED;
    out->size = size;
    out->capacity = size;
    out->ctx_id = ctx->id;
    out->iommu_domain_id = domain->id;
    if (out->dma_addr > UINT32_MAX || out->size == 0 ||
        out->size - 1 > UINT32_MAX - out->dma_addr) {
        rk_npu_mem_free(ctx, out);
        return RK_NPU_ERR_DOMAIN;
    }
    return RK_NPU_OK;
}

extern "C" int rk_npu_mem_export_dmabuf(rk_npu_ctx* ctx, const rk_npu_mem* mem, int* out_fd) {
    if (!ctx || !mem || !out_fd || (mem->flags & RK_NPU_MEM_F_VIEW) || !same_ctx(ctx, mem))
        return RK_NPU_ERR_PARAM;
    drm_prime_handle ph; std::memset(&ph, 0, sizeof(ph));
    ph.handle = mem->handle;
    ph.flags = O_CLOEXEC;
    ph.fd = -1;
    if (ioctl(ctx->fd, IOCTL_PRIME_HANDLE_TO_FD, &ph) < 0) return RK_NPU_ERR_IOCTL;
    *out_fd = ph.fd;
    return RK_NPU_OK;
}

extern "C" int rk_npu_mem_view(const rk_npu_mem* base, uint64_t offset, uint64_t size, rk_npu_mem* out_view) {
    if (!base || !out_view || size == 0 || !base->capacity || offset > base->size ||
        size > base->size - offset)
        return RK_NPU_ERR_PARAM;
    *out_view = *base;
    out_view->flags = (base->flags & (RK_NPU_MEM_F_IMPORTED |
                                      RK_NPU_MEM_F_MAPPED |
                                      RK_NPU_MEM_F_CACHEABLE)) | RK_NPU_MEM_F_VIEW;
    out_view->dma_addr = base->dma_addr + offset;
    out_view->vaddr = base->vaddr ? (void*)((uint8_t*)base->vaddr + offset) : nullptr;
    out_view->offset = base->offset + offset;
    out_view->size = size;
    return RK_NPU_OK;
}

extern "C" int rk_npu_mem_sync(rk_npu_ctx* ctx, const rk_npu_mem* mem, rk_npu_sync_dir dir) {
    if (!ctx || !mem || !same_ctx(ctx, mem) ||
        !(dir == RK_NPU_SYNC_TO_DEVICE || dir == RK_NPU_SYNC_FROM_DEVICE))
        return RK_NPU_ERR_PARAM;
    rknpu_mem_sync s; std::memset(&s, 0, sizeof(s));
    s.flags = (uint32_t)dir;
    s.obj_addr = mem->obj_addr;
    s.offset = mem->offset;
    s.size = mem->size;
    if (ioctl(ctx->fd, IOCTL_MEM_SYNC, &s) < 0) return RK_NPU_ERR_IOCTL;
    return RK_NPU_OK;
}

extern "C" int rk_npu_mem_free(rk_npu_ctx* ctx, rk_npu_mem* mem) {
    if (!ctx || !mem) return RK_NPU_ERR_PARAM;
    if (mem->flags & RK_NPU_MEM_F_VIEW) {
        std::memset(mem, 0, sizeof(*mem));
        return RK_NPU_OK;
    }
    if (!same_ctx(ctx, mem)) return RK_NPU_ERR_PARAM;

    if ((mem->flags & RK_NPU_MEM_F_MAPPED) && mem->vaddr &&
        !(mem->flags & RK_NPU_MEM_F_IMPORTED))
        munmap(mem->vaddr, mem->capacity ? mem->capacity : mem->size);

    int rc = RK_NPU_OK;
    if ((mem->flags & RK_NPU_MEM_F_OWN_RKNPU) && mem->handle && mem->obj_addr) {
        rknpu_mem_destroy md; std::memset(&md, 0, sizeof(md));
        md.handle = mem->handle;
        md.obj_addr = mem->obj_addr;
        if (ioctl(ctx->fd, IOCTL_MEM_DESTROY, &md) < 0) rc = RK_NPU_ERR_IOCTL;
    }
    std::memset(mem, 0, sizeof(*mem));
    return rc;
}

/* ------------------------------------------ domain-native data buffer ---- */

namespace rknpu2_matmul_open::detail {

void retain_domain(rk_npu_iommu_domain* domain) {
    if (domain) domain->refs.fetch_add(1, std::memory_order_relaxed);
}

void release_domain(rk_npu_iommu_domain* domain) {
    if (domain && domain->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete domain;
}

DomainDataBuffer::DomainDataBuffer(DomainDataBuffer&& other) noexcept {
    mem = other.mem;
    domain = other.domain;
    cpu_access = other.cpu_access;
    std::memset(&other.mem, 0, sizeof(other.mem));
    other.domain = nullptr;
    other.cpu_access = CpuAccess::None;
}

DomainDataBuffer& DomainDataBuffer::operator=(DomainDataBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (mem.vaddr || mem.handle || domain) std::abort();
    mem = other.mem;
    domain = other.domain;
    cpu_access = other.cpu_access;
    std::memset(&other.mem, 0, sizeof(other.mem));
    other.domain = nullptr;
    other.cpu_access = CpuAccess::None;
    return *this;
}

int DomainDataBuffer::alloc(rk_npu_iommu_domain* owner, uint64_t size) {
    if (!owner || !owner->ctx || size == 0 || mem.vaddr || mem.handle || domain)
        return RK_NPU_ERR_PARAM;
    const int rc = rk_npu_mem_alloc(owner, size, RK_NPU_MEM_DATA_DEFAULT, &mem);
    if (rc != RK_NPU_OK) return rc;
    domain = owner;
    retain_domain(domain);
    return rc;
}

static int begin_cpu_access(DomainDataBuffer& buffer,
                            DomainDataBuffer::CpuAccess access) {
    if (!buffer.domain || !buffer.mem.vaddr ||
        access == DomainDataBuffer::CpuAccess::None)
        return RK_NPU_ERR_PARAM;
    if (buffer.cpu_access == access) return RK_NPU_OK;
    if (buffer.cpu_access != DomainDataBuffer::CpuAccess::None)
        return RK_NPU_ERR_PARAM;
    if (access == DomainDataBuffer::CpuAccess::Read ||
        access == DomainDataBuffer::CpuAccess::ReadWrite) {
        const int rc = rk_npu_mem_sync(buffer.domain->ctx, &buffer.mem,
                                       RK_NPU_SYNC_FROM_DEVICE);
        if (rc != RK_NPU_OK) return rc;
    }
    buffer.cpu_access = access;
    return RK_NPU_OK;
}

int DomainDataBuffer::begin_cpu_read() {
    return begin_cpu_access(*this, CpuAccess::Read);
}

int DomainDataBuffer::begin_cpu_write() {
    return begin_cpu_access(*this, CpuAccess::Write);
}

int DomainDataBuffer::begin_cpu_readwrite() {
    return begin_cpu_access(*this, CpuAccess::ReadWrite);
}

int DomainDataBuffer::end_cpu_access() {
    if (cpu_access == CpuAccess::None) return RK_NPU_OK;
    if (!domain || !mem.vaddr) return RK_NPU_ERR_PARAM;
    int rc = RK_NPU_OK;
    if (cpu_access == CpuAccess::Write || cpu_access == CpuAccess::ReadWrite)
        rc = rk_npu_mem_sync(domain->ctx, &mem, RK_NPU_SYNC_TO_DEVICE);
    cpu_access = CpuAccess::None;
    return rc;
}

int DomainDataBuffer::release(rk_npu_ctx* ctx) {
    if (!ctx) return RK_NPU_ERR_PARAM;
    int rc = end_cpu_access();
    if (mem.vaddr || mem.handle) {
        const int free_rc = rk_npu_mem_free(ctx, &mem);
        if (rc == RK_NPU_OK && free_rc != RK_NPU_OK) rc = free_rc;
    }
    release_domain(domain);
    domain = nullptr;
    cpu_access = CpuAccess::None;
    return rc;
}

} /* namespace rknpu2_matmul_open::detail */

/* ------------------------------------------------- submit / PC chain ---- */

namespace rknpu2_matmul_open::detail {

int do_submit(int fd, uint64_t task_obj_addr, int num_tasks,
              uint32_t iommu_domain_id, uint32_t timeout_ms) {
    if (iommu_domain_id >= RK_NPU_IOMMU_DOMAIN_COUNT) return RK_NPU_ERR_PARAM;
    rknpu_submit s; std::memset(&s, 0, sizeof(s));
    s.flags       = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;
    s.timeout     = timeout_ms;
    s.task_number = (uint32_t)num_tasks;
    s.task_obj_addr = task_obj_addr;
    s.iommu_domain_id = iommu_domain_id;
    s.core_mask   = 1;                            /* single core */
    s.fence_fd    = -1;
    s.subcore_task[0] = { 0, (uint32_t)num_tasks };
    s.subcore_task[1] = { (uint32_t)num_tasks, 0 };
    s.subcore_task[2] = { (uint32_t)num_tasks, 0 };
    if (ioctl(fd, IOCTL_SUBMIT, &s) < 0) {
        rknpu_action act; std::memset(&act, 0, sizeof(act));
        act.flags = RKNPU_ACT_RESET;
        ioctl(fd, IOCTL_ACTION, &act);
        return RK_NPU_ERR_SUBMIT;
    }
    return RK_NPU_OK;
}

int do_submit_multicore(int fd, uint64_t task_obj_addr, int num_tasks,
                        uint32_t core_mask, const uint32_t task_start[3],
                        const uint32_t task_count[3], uint32_t iommu_domain_id,
                        uint32_t timeout_ms) {
    const int core_count = __builtin_popcount(core_mask & 7u);
    if (core_mask == 0 || (core_mask & ~7u) != 0 ||
        core_mask == 5u || core_mask == 6u ||
        core_count < 1 || core_count > 3 || num_tasks <= 0 ||
        !task_start || !task_count ||
        iommu_domain_id >= RK_NPU_IOMMU_DOMAIN_COUNT) return RK_NPU_ERR_PARAM;

    uint32_t covered = 0;
    for (int core = 0; core < core_count; ++core) {
        if (task_count[core] == 0 || task_start[core] != covered)
            return RK_NPU_ERR_PARAM;
        covered += task_count[core];
    }
    if (covered != (uint32_t)num_tasks) return RK_NPU_ERR_PARAM;

    rknpu_submit s; std::memset(&s, 0, sizeof(s));
    s.flags         = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;
    s.timeout       = timeout_ms;
    s.task_number   = (uint32_t)num_tasks;
    s.task_obj_addr = task_obj_addr;
    s.iommu_domain_id = iommu_domain_id;
    s.core_mask     = core_mask;
    s.fence_fd      = -1;

    int logical_core = 0;
    for (int physical_core = 0; physical_core < 3; ++physical_core) {
        if ((core_mask & (1u << physical_core)) == 0) continue;
        /* This driver accepts 0x3 as its only two-core execution mask; it reads
         * logical slots 0/1.  A three-core job consumes slots 2/3/4, while a
         * single physical-core job uses that core's slot.  Masks 0x5/0x6 are
         * rejected above because both timed out in the board matrix test. */
        const int slot = core_count == 3 ? physical_core + 2
                       : core_count == 2 ? logical_core
                                         : physical_core;
        s.subcore_task[slot] = {task_start[logical_core], task_count[logical_core]};
        ++logical_core;
    }
    if (ioctl(fd, IOCTL_SUBMIT, &s) < 0) {
        rknpu_action act; std::memset(&act, 0, sizeof(act));
        act.flags = RKNPU_ACT_RESET;
        ioctl(fd, IOCTL_ACTION, &act);
        return RK_NPU_ERR_SUBMIT;
    }
    return RK_NPU_OK;
}

void write_chain(uint64_t* cmd, rknpu_task* tk, uint64_t regcmd_dma,
                 const std::vector<std::vector<uint64_t>>& bodies,
                 const std::vector<int>& base, const ChainCfg& cfg) {
    const int num = (int)bodies.size();
    for (int ti = 0; ti < num; ++ti) {
        const std::vector<uint64_t>& body = bodies[ti];
        const int b = base[ti];
        for (size_t i = 0; i < body.size(); ++i) cmd[b + i] = body[i];

        /* 4-qword PC-chain tail. The last qword enables the units for this task. */
        uint64_t en = E(T_PC, R_OPERATION_ENABLE, cfg.op_enable);
        uint64_t tail[4];
        if (ti + 1 < num) {
            uint64_t next_addr = regcmd_dma + (uint64_t)base[ti + 1] * 8;
            int next_len = (int)bodies[ti + 1].size();
            tail[0] = E(T_PC_REG, R_PC_BASE_ADDRESS, (uint32_t)(next_addr & 0xFFFFFFF0));
            tail[1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, ceil_div(next_len, 2) + 1);
            tail[2] = E(T_VERSION, 0, 0);
            tail[3] = en;
        } else {
            tail[0] = E(T_NOP, 0, 0);
            tail[1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, 0);   /* end of chain */
            tail[2] = E(T_VERSION, 0, 0);
            tail[3] = en;
        }
        for (int i = 0; i < 4; ++i) cmd[b + body.size() + i] = tail[i];

        tk[ti].regcmd_addr   = regcmd_dma + (uint64_t)b * 8;
        tk[ti].regcfg_amount = (uint32_t)body.size();
        tk[ti].op_idx        = cfg.op_idx;
        tk[ti].enable_mask   = cfg.enable_mask;
        tk[ti].int_mask      = (1u << 8) | (1u << 9);  /* DPU_0 | DPU_1 done IRQs */
        tk[ti].int_clear     = 0x1ffff;
    }
}

} /* namespace rknpu2_matmul_open::detail */

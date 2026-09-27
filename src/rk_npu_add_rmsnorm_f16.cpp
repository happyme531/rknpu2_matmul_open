#include "rk_npu_add_rmsnorm_f16.h"
#include "rk_npu_internal.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

namespace {

struct MemoryTemplate { uint32_t handle; uint32_t old_base; uint64_t size; };
struct TaskTemplate {
    const uint64_t* words;
    size_t count;
    uint32_t op_idx;
    uint32_t enable;
};
struct ShapeTemplate {
    int M, D;
    const MemoryTemplate* memories;
    size_t memory_count;
    const TaskTemplate* tasks;
    size_t task_count;
};

#include "rk_npu_add_rmsnorm_templates.inc"

struct DynamicPatch {
    size_t qword;
    uint32_t handle;
    uint32_t offset;
};

const ShapeTemplate* find_template(int M, int D) {
    for (const auto& item : kAddRmsNormTemplates)
        if (item.M == M && item.D == D) return &item;
    return nullptr;
}

const MemoryTemplate* find_memory(const ShapeTemplate* shape, uint32_t handle) {
    for (size_t i = 0; i < shape->memory_count; ++i)
        if (shape->memories[i].handle == handle) return &shape->memories[i];
    return nullptr;
}

const MemoryTemplate* memory_for_value(const ShapeTemplate* shape,
                                       uint32_t value) {
    for (size_t i = 0; i < shape->memory_count; ++i) {
        const auto& mem = shape->memories[i];
        if (value >= mem.old_base &&
            (uint64_t)value < (uint64_t)mem.old_base + mem.size)
            return &mem;
    }
    return nullptr;
}

uint32_t reg_value(uint64_t qword) { return (uint32_t)(qword >> 16); }
uint32_t reg_addr(uint64_t qword) { return (uint32_t)(qword & 0xffffu); }
uint64_t reg_target(uint64_t qword) { return (qword >> 48) & 0xffffu; }

bool valid_config(const rk_npu_add_rmsnorm_f16_config* cfg) {
    return cfg && find_template(cfg->M, cfg->D) &&
           std::fabs(cfg->eps - 1.0e-5f) <= 1.0e-12f;
}

uint64_t required_regcmd_bytes(const ShapeTemplate* shape) {
    uint64_t qwords = 0;
    for (size_t i = 0; i < shape->task_count; ++i)
        qwords += (uint64_t)align_up((int)shape->tasks[i].count + 4, 2);
    return qwords * sizeof(uint64_t);
}

bool overlaps(const rk_npu_mem* a, const rk_npu_mem* b, uint64_t bytes) {
    const uint64_t ae = a->dma_addr + bytes;
    const uint64_t be = b->dma_addr + bytes;
    return a->dma_addr < be && b->dma_addr < ae;
}

}  // namespace

struct rk_npu_add_rmsnorm_f16_plan {
    rk_npu_ctx* ctx = nullptr;
    rk_npu_iommu_domain* domain = nullptr;
    rk_npu_add_rmsnorm_f16_config cfg{};
    const ShapeTemplate* shape = nullptr;
    rk_npu_mem internal{}, weight{}, regcmd{}, task{};
    std::vector<DynamicPatch> patches;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

namespace {

void release_plan(rk_npu_add_rmsnorm_f16_plan* plan) {
    if (!plan) return;
    if (plan->ctx) {
        if (plan->internal.handle) rk_npu_mem_free(plan->ctx, &plan->internal);
        if (plan->weight.handle) rk_npu_mem_free(plan->ctx, &plan->weight);
        if (plan->regcmd.handle) rk_npu_mem_free(plan->ctx, &plan->regcmd);
        if (plan->task.handle) rk_npu_mem_free(plan->ctx, &plan->task);
    }
    if (plan->domain) rknpu2_matmul_open::detail::release_domain(plan->domain);
}

int write_weights(rk_npu_add_rmsnorm_f16_plan* plan, const uint16_t* gamma) {
    auto* weight = static_cast<uint16_t*>(plan->weight.vaddr);
    std::memset(weight, 0, (size_t)plan->weight.size);
    uint32_t gamma_old = 0, broadcast_old = 0, sum_old = 0;
    uint32_t broadcast_bytes = 0, sum_bytes = 0;
    for (size_t ti = 0; ti < plan->shape->task_count; ++ti) {
        const auto& task = plan->shape->tasks[ti];
        uint32_t dcomp = 0, wsize = 0;
        for (size_t i = 0; i < task.count; ++i) {
            const uint64_t qword = task.words[i];
            const auto target = reg_target(qword);
            const auto addr = reg_addr(qword);
            const auto value = reg_value(qword);
            if (target == T_RDMA && addr == R_RDMA_BS_BASE) gamma_old = value;
            if (target == T_CNA && addr == R_CNA_DCOMP_ADDR0) dcomp = value;
            if (target == T_CNA && addr == R_CNA_WEIGHT_SIZE0) wsize = value;
        }
        if (wsize == (uint32_t)plan->cfg.D * 2u) {
            sum_old = dcomp;
            sum_bytes = wsize;
        } else if (wsize == (uint32_t)plan->cfg.D * 16u) {
            broadcast_old = dcomp;
            broadcast_bytes = wsize;
        }
    }
    const auto* weight_mem = find_memory(plan->shape, 2);
    if (!weight_mem || !gamma_old || !broadcast_old || !sum_old ||
        sum_bytes != (uint32_t)plan->cfg.D * 2u ||
        broadcast_bytes != (uint32_t)plan->cfg.D * 16u)
        return RK_NPU_ERR_PARAM;
    const uint32_t gamma_off = gamma_old - weight_mem->old_base;
    const uint32_t broadcast_off = broadcast_old - weight_mem->old_base;
    const uint32_t sum_off = sum_old - weight_mem->old_base;
    if ((uint64_t)gamma_off + plan->cfg.D * 2u > plan->weight.size ||
        (uint64_t)broadcast_off + broadcast_bytes > plan->weight.size ||
        (uint64_t)sum_off + sum_bytes > plan->weight.size)
        return RK_NPU_ERR_NOMEM;
    std::memcpy((uint8_t*)weight + gamma_off, gamma,
                (size_t)plan->cfg.D * sizeof(uint16_t));
    uint16_t* broadcast = (uint16_t*)((uint8_t*)weight + broadcast_off);
    for (int d = 0; d < plan->cfg.D; ++d) broadcast[(size_t)d * 8] = 0x3c00;
    uint16_t* sum = (uint16_t*)((uint8_t*)weight + sum_off);
    for (int d = 0; d < plan->cfg.D; ++d) sum[d] = 0x3f80;
    return RK_NPU_OK;
}

int build_plan(rk_npu_add_rmsnorm_f16_plan* plan) {
    uint64_t* cmd = static_cast<uint64_t*>(plan->regcmd.vaddr);
    auto* tasks = static_cast<rknpu_task*>(plan->task.vaddr);
    std::memset(cmd, 0, (size_t)plan->regcmd.size);
    std::memset(tasks, 0, (size_t)plan->task.size);
    std::vector<int> base(plan->shape->task_count);
    int off = 0;
    for (size_t ti = 0; ti < plan->shape->task_count; ++ti) {
        base[ti] = off;
        off += align_up((int)plan->shape->tasks[ti].count + 4, 2);
    }
    for (size_t ti = 0; ti < plan->shape->task_count; ++ti) {
        const auto& item = plan->shape->tasks[ti];
        for (size_t i = 0; i < item.count; ++i) {
            uint64_t qword = item.words[i];
            const uint32_t value = reg_value(qword);
            const auto* mem = memory_for_value(plan->shape, value);
            if (mem) {
                const uint32_t delta = value - mem->old_base;
                if (mem->handle == 2)
                    qword = E(reg_target(qword), reg_addr(qword),
                              (uint32_t)(plan->weight.dma_addr + delta));
                else if (mem->handle == 3)
                    qword = E(reg_target(qword), reg_addr(qword),
                              (uint32_t)(plan->internal.dma_addr + delta));
                else
                    plan->patches.push_back(
                        {static_cast<size_t>(base[ti]) + i, mem->handle, delta});
            }
            cmd[base[ti] + i] = qword;
        }
        const size_t tail = (size_t)base[ti] + item.count;
        if (ti + 1 < plan->shape->task_count) {
            const uint64_t next = plan->regcmd.dma_addr + (uint64_t)base[ti + 1] * 8;
            cmd[tail + 0] = E(T_PC_REG, R_PC_BASE_ADDRESS,
                              (uint32_t)(next & 0xfffffff0u));
            cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS,
                              ceil_div((int)plan->shape->tasks[ti + 1].count, 2) + 1);
            cmd[tail + 2] = E(T_VERSION, 0, 0);
        } else {
            cmd[tail + 0] = E(T_NOP, 0, 0);
            cmd[tail + 1] = E(T_PC_REG, R_PC_REGISTER_AMOUNTS, 0);
            cmd[tail + 2] = E(T_VERSION, 0, 0);
        }
        cmd[tail + 3] = E(T_PC, R_OPERATION_ENABLE, item.enable);
        tasks[ti].regcmd_addr = plan->regcmd.dma_addr + (uint64_t)base[ti] * 8;
        tasks[ti].regcfg_amount = (uint32_t)item.count;
        tasks[ti].op_idx = item.op_idx;
        tasks[ti].enable_mask = item.enable;
        tasks[ti].int_mask = 0x300;
        tasks[ti].int_clear = 0x1ffff;
    }
    return RK_NPU_OK;
}

}  // namespace

extern "C" void rk_npu_add_rmsnorm_f16_config_init(
    rk_npu_add_rmsnorm_f16_config* cfg, int M, int D) {
    if (!cfg) return;
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->M = M;
    cfg->D = D;
    cfg->eps = 1.0e-5f;
    cfg->timeout_ms = 6000;
}

extern "C" int rk_npu_add_rmsnorm_f16_query(
    const rk_npu_add_rmsnorm_f16_config* cfg,
    rk_npu_add_rmsnorm_f16_sizes* out) {
    if (!valid_config(cfg) || !out) return RK_NPU_ERR_PARAM;
    const auto* shape = find_template(cfg->M, cfg->D);
    const auto* weight = find_memory(shape, 2);
    const auto* internal = find_memory(shape, 3);
    if (!weight || !internal) return RK_NPU_ERR_PARAM;
    out->tensor_bytes = (uint64_t)cfg->M * cfg->D * sizeof(uint16_t);
    out->internal_bytes = internal->size;
    out->weight_bytes = weight->size;
    out->regcmd_bytes = required_regcmd_bytes(shape);
    out->task_bytes = shape->task_count * sizeof(rknpu_task);
    out->num_tasks = (int)shape->task_count;
    return RK_NPU_OK;
}

extern "C" rk_npu_add_rmsnorm_f16_plan*
rk_npu_add_rmsnorm_f16_plan_create(
    rk_npu_iommu_domain* domain,
    const rk_npu_add_rmsnorm_f16_config* cfg,
    const uint16_t* gamma) {
    if (!domain || !domain->ctx || !valid_config(cfg) || !gamma) return nullptr;
    auto* plan = new (std::nothrow) rk_npu_add_rmsnorm_f16_plan();
    if (!plan) return nullptr;
    plan->ctx = domain->ctx;
    plan->domain = domain;
    plan->cfg = *cfg;
    plan->shape = find_template(cfg->M, cfg->D);
    rknpu2_matmul_open::detail::retain_domain(domain);
    rk_npu_add_rmsnorm_f16_sizes sizes{};
    if (rk_npu_add_rmsnorm_f16_query(cfg, &sizes) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sizes.internal_bytes, RK_NPU_MEM_NON_CACHEABLE,
                         &plan->internal) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sizes.weight_bytes, RK_NPU_MEM_NON_CACHEABLE,
                         &plan->weight) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sizes.regcmd_bytes, RK_NPU_MEM_NON_CACHEABLE,
                         &plan->regcmd) != RK_NPU_OK ||
        rk_npu_mem_alloc(domain, sizes.task_bytes, RK_NPU_MEM_KERNEL_MAPPING,
                         &plan->task) != RK_NPU_OK ||
        write_weights(plan, gamma) != RK_NPU_OK ||
        build_plan(plan) != RK_NPU_OK) {
        release_plan(plan);
        delete plan;
        return nullptr;
    }
    return plan;
}

extern "C" int rk_npu_add_rmsnorm_f16_run(
    rk_npu_ctx* ctx, rk_npu_add_rmsnorm_f16_plan* plan,
    const rk_npu_mem* x, const rk_npu_mem* residual,
    rk_npu_mem* residual_out, rk_npu_mem* norm_out) {
    if (!ctx || !plan || ctx != plan->ctx || !x || !residual ||
        !residual_out || !norm_out) return RK_NPU_ERR_PARAM;
    const uint64_t bytes = (uint64_t)plan->cfg.M * plan->cfg.D * 2;
    const rk_npu_mem* all[] = {x, residual, residual_out, norm_out};
    for (const auto* mem : all) {
        if (mem->ctx_id != ctx->id || mem->size < bytes) return RK_NPU_ERR_NOMEM;
        if (mem->iommu_domain_id != plan->domain->id) return RK_NPU_ERR_DOMAIN;
    }
    if (overlaps(x, residual_out, bytes) || overlaps(x, norm_out, bytes) ||
        overlaps(residual, residual_out, bytes) || overlaps(residual, norm_out, bytes) ||
        overlaps(residual_out, norm_out, bytes)) return RK_NPU_ERR_PARAM;
    if (plan->busy.test_and_set(std::memory_order_acquire)) return RK_NPU_ERR_BUSY;
    uint64_t* cmd = static_cast<uint64_t*>(plan->regcmd.vaddr);
    for (const auto& patch : plan->patches) {
        const rk_npu_mem* mem = nullptr;
        if (patch.handle == 4) mem = x;
        else if (patch.handle == 5) mem = residual;
        else if (patch.handle == 6) mem = residual_out;
        else if (patch.handle == 7) mem = norm_out;
        if (!mem) {
            plan->busy.clear(std::memory_order_release);
            return RK_NPU_ERR_PARAM;
        }
        const uint64_t old = cmd[patch.qword];
        cmd[patch.qword] = E(reg_target(old), reg_addr(old),
                             (uint32_t)(mem->dma_addr + patch.offset));
    }
    const uint32_t timeout = plan->cfg.timeout_ms ? plan->cfg.timeout_ms : 6000;
    const int rc = rknpu2_matmul_open::detail::do_submit(ctx->fd, plan->task.obj_addr,
                                    (int)plan->shape->task_count,
                                    plan->domain->id, timeout);
    plan->busy.clear(std::memory_order_release);
    return rc;
}

extern "C" void rk_npu_add_rmsnorm_f16_plan_free(
    rk_npu_add_rmsnorm_f16_plan* plan) {
    if (!plan) return;
    release_plan(plan);
    delete plan;
}

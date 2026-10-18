#include "rk_npu_mm_gelu.h"
#include "rk_npu_cpu_kernels.h"
#include "rk_npu_float_backend.h"
#include "rk_npu_internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace detail = rknpu2_matmul_open::detail;
namespace {
void check(int rc) {
    if (rc)
        throw rc;
}
uint32_t get(const std::vector<uint64_t> &body, uint64_t target, uint32_t reg) {
    for (auto word : body)
        if (word >> 48 == target && uint16_t(word) == reg)
            return uint32_t(word >> 16);
    throw std::logic_error("register missing");
}
void set(std::vector<uint64_t> &body, uint64_t target, uint32_t reg, uint32_t value) {
    for (auto &word : body)
        if (word >> 48 == target && uint16_t(word) == reg) {
            word = E(target, reg, value);
            return;
        }
    throw std::logic_error("register missing");
}
std::vector<uint64_t> lut() {
    std::vector<uint64_t> result;
    for (int bank = 0; bank < 2; ++bank) {
        result.push_back(E(T_DPU, 0x4100, uint32_t(2 | bank) << 16));
        for (int i = 0; i <= 512; ++i) {
            const double x = double(i + (bank ? 0 : -512)) / 128.;
            const double y = .5 * x * (1. + std::tanh(.7978845608028654 * (x + .044715 * x * x * x)));
            int code = std::clamp(int(std::nearbyint(y * 8192.)), -32768, 32767);
            if (!bank && !i)
                code = 0;
            result.push_back(E(T_DPU, 0x4104, uint32_t(code)));
        }
    }
    return result;
}
} // namespace

struct rk_npu_mm_gelu_workspace {
    rk_npu_iommu_domain *domain = nullptr;
    rk_npu_mm_gelu_config config{};
    rk_npu_mem input{}, output{}, bias{}, commands{}, tasks{};
    int aligned_n = 0, aligned_k = 0, physical_m = 0, count = 0;
    uint32_t starts[3]{}, counts[3]{}, mask = 0;
    // Word indices and offsets into the existing packed B. Patch only the
    // non-cacheable command buffer when changing the layer, never the weights.
    std::vector<std::pair<size_t, uint32_t>> relocations;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    ~rk_npu_mm_gelu_workspace() {
        if (domain) {
            for (auto *mem : {&tasks, &commands, &bias, &output, &input})
                rk_npu_mem_free(domain->ctx, mem);
            detail::release_domain(domain);
        }
    }
};

extern "C" void rk_npu_mm_gelu_config_init(rk_npu_mm_gelu_config *c, int m, int n, int k) {
    if (c)
        *c = {m, n, k, 0, 7, 500};
}
extern "C" void rk_npu_mm_gelu_free(rk_npu_mm_gelu_workspace *w) { delete w; }
extern "C" int rk_npu_mm_gelu_prepare(rk_npu_iommu_domain *domain, const rk_npu_mm_gelu_config *c,
                                      rk_npu_mm_gelu_workspace **out) {
    if (!out)
        return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!domain || !domain->ctx || !c || c->M < 1 || c->M > 65504 || c->N < 1 || c->N > 32768 || c->K < 1 ||
        c->K > 65504 || !c->core_mask || c->core_mask > 7 || c->core_mask == 5 || c->core_mask == 6 ||
        !c->timeout_ms || c->n_tile < 0 || c->n_tile > 65504 || c->n_tile % 32)
        return RK_NPU_ERR_PARAM;
    try {
        auto owner = std::make_unique<rk_npu_mm_gelu_workspace>();
        auto &p = *owner;
        p.domain = domain;
        detail::retain_domain(domain);
        p.config = *c;
        p.aligned_n = align_up(c->N, 32);
        p.aligned_k = align_up(c->K, 32);
        p.physical_m = align_up(c->M, 8);
        rk_npu_matmul_f16_config cfg{};
        rk_npu_matmul_f16_config_init(&cfg, p.physical_m, c->N, c->K, RK_NPU_FUSE_NONE);
        cfg.a_layout = RK_NPU_F16_A_LAYOUT_PANEL8;
        cfg.d_layout = RK_NPU_F16_D_LAYOUT_PANEL8;
        cfg.core_mask = 1;
        cfg.timeout_ms = c->timeout_ms;
        const int allowed = __builtin_popcount(c->core_mask);
        cfg.n_tile = c->n_tile ? c->n_tile : align_up(ceil_div(p.aligned_n, allowed), 32);
        rk_npu_matmul_sizes sizes{};
        check(rk_npu_matmul_f16_query(&cfg, &sizes));
        check(rk_npu_mem_alloc(domain, sizes.input_bytes, RK_NPU_MEM_DATA_DEFAULT, &p.input));
        check(rk_npu_mem_alloc(domain, sizes.output_bytes * 2, RK_NPU_MEM_DATA_DEFAULT, &p.output));
        check(rk_npu_mem_alloc(domain, p.aligned_n * sizeof(float), RK_NPU_MEM_DATA_DEFAULT, &p.bias));
        std::memset(p.input.vaddr, 0, p.input.size);
        std::memset(p.bias.vaddr, 0, p.bias.size);
        rk_npu_mem placeholder{};
        placeholder.ctx_id = domain->ctx->id;
        placeholder.iommu_domain_id = domain->id;
        placeholder.size = sizes.weight_bytes;
        std::vector<detail::RegisterTask> source;
        check(detail::emit_f16_batch_tasks(domain, 1, &cfg, &p.input, &placeholder, &p.output, source));
        const int cores = std::min(allowed, int(source.size()));
        if (!cores)
            return RK_NPU_ERR_PARAM;
        std::vector<std::vector<uint64_t>> bodies;
        std::vector<int> bases;
        const auto table = lut();
        int at = 0, first = 0, used = 0;
        for (int core = 0; core < 3 && used < cores; ++core) {
            if (!(c->core_mask & (1u << core)))
                continue;
            p.mask |= 1u << core;
            const int slot = used++;
            p.starts[slot] = first;
            p.counts[slot] = source.size() / cores + (slot < int(source.size() % cores));
            for (int task = 0; task < int(p.counts[slot]); ++task, ++first) {
                auto body = std::move(source[first].body);
                const auto weight_offset = get(body, T_CNA, R_CNA_DCOMP_ADDR0);
                const int n0 = weight_offset / (p.aligned_k * 2);
                const uint32_t shape = get(body, T_CORE, R_CORE_DATAOUT_SIZE_0);
                const uint32_t channels = get(body, T_DPU, R_DATA_CUBE_CHANNEL) & 0xffff;
                const uint32_t offset = get(body, T_DPU, R_DST_BASE_ADDR) - p.output.dma_addr;
                set(body, T_DPU, R_DST_BASE_ADDR, uint32_t(p.output.dma_addr + offset * 2));
                set(body, T_DPU, R_DATA_FORMAT, 0xb4000005);
                set(body, T_DPU, R_BS_CFG, 0x20150);
                set(body, T_DPU, R_BS_OW_CFG, 0x36e);
                set(body, T_DPU, R_BN_CFG, 0x20040);
                set(body, T_DPU, 0x4064, 0x80000000);
                set(body, T_DPU, 0x4068, 0x6c000000);
                set(body, T_DPU, R_EW_CFG, 0x302);
                set(body, T_DPU, R_OUT_CVT_SCALE, 1);
                set(body, T_DPU, 0x4088, 0);
                const uint32_t notch = (p.aligned_n / 4 - 1) * 8;
                set(body, T_DPU, R_DATA_CUBE_NOTCH, (notch << 16) | notch);
                set(body, T_DPU, R_SURFACE_ADD, get(body, T_DPU, R_SURFACE_ADD) * 2);
                const std::pair<uint32_t, uint32_t> fields[] = {
                    {0x4108, 0x68},    {0x410c, 0x50500}, {0x4110, uint32_t(-16384)},
                    {0x4114, 0},       {0x4118, 0},       {0x411c, 16384},
                    {0x4120, 0},       {0x4124, 0},       {0x4128, 16384u << 16},
                    {0x412c, 13u << 5}};
                for (auto [reg, value] : fields)
                    set(body, T_DPU, reg, value);
                const std::pair<uint32_t, uint32_t> rdma[] = {{0x5004, 14},
                                                              {0x500c, shape & 0xffff},
                                                              {0x5010, shape >> 16},
                                                              {0x5014, channels},
                                                              {0x5018, 0},
                                                              {0x501c, 2},
                                                              {0x5020, uint32_t(p.bias.dma_addr + n0 * 4)},
                                                              {0x5028, 0},
                                                              {0x502c, 0},
                                                              {0x5034, 1},
                                                              {0x5038, 0},
                                                              {0x5040, 16},
                                                              {0x5044, 0x2f8b0},
                                                              {0x5048, 0},
                                                              {0x504c, 0},
                                                              {0x5064, 0},
                                                              {0x5068, 0x01010101},
                                                              {0x506c, 0}};
                for (auto [reg, value] : rdma)
                    body.push_back(E(T_RDMA, reg, value));

                // Reload on each core's first task, on every run. Other ops
                // and runtime PM may change LUT state between invocations.
                if (!task)
                    body.insert(body.begin(), table.begin(), table.end());
                for (size_t j = 0; j < body.size(); ++j)
                    if (body[j] >> 48 == T_CNA && uint16_t(body[j]) == R_CNA_DCOMP_ADDR0)
                        p.relocations.emplace_back(at + j, weight_offset);
                bases.push_back(at);
                at = align_up(at + int(body.size()) + 4, 2);
                bodies.push_back(std::move(body));
            }
        }
        p.count = bodies.size();
        if (p.count > 4095)
            return RK_NPU_ERR_PARAM;
        check(rk_npu_mem_alloc(domain, uint64_t(at) * 8, RK_NPU_MEM_NON_CACHEABLE, &p.commands));
        check(rk_npu_mem_alloc(domain, p.count * sizeof(rknpu_task), RK_NPU_MEM_KERNEL_MAPPING, &p.tasks));
        std::memset(p.tasks.vaddr, 0, p.tasks.size);
        for (int core = 0; core < 3; ++core) {
            if (!p.counts[core])
                continue;
            const int begin = p.starts[core], end = begin + p.counts[core];
            detail::write_chain(static_cast<uint64_t *>(p.commands.vaddr),
                                static_cast<rknpu_task *>(p.tasks.vaddr) + begin, p.commands.dma_addr,
                                {bodies.begin() + begin, bodies.begin() + end},
                                {bases.begin() + begin, bases.begin() + end}, {0x1d, 0, 0x1d});
        }
        *out = owner.release();
        return RK_NPU_OK;
    } catch (int rc) {
        return rc;
    } catch (const std::bad_alloc &) {
        return RK_NPU_ERR_NOMEM;
    } catch (const std::logic_error &) {
        return RK_NPU_ERR_PARAM;
    }
}
extern "C" int rk_npu_mm_gelu_run(rk_npu_mm_gelu_workspace *w, const uint16_t *a, uint64_t bytes,
                                  const rk_npu_mm_packed_b *b, const float *bias, rk_npu_mm_gelu_view *view) {
    if (!view)
        return RK_NPU_ERR_PARAM;
    *view = {};
    if (!w || !a || !bias || reinterpret_cast<uintptr_t>(a) % 2 || reinterpret_cast<uintptr_t>(bias) % 4)
        return RK_NPU_ERR_PARAM;
    if (bytes < uint64_t(w->config.M) * w->config.K * 2)
        return RK_NPU_ERR_NOMEM;
    if (w->busy.test_and_set())
        return RK_NPU_ERR_BUSY;
    struct Unlock {
        std::atomic_flag &flag;
        ~Unlock() { flag.clear(); }
    } unlock{w->busy};
    const rk_npu_mem *weight = nullptr;
    int rc = detail::float_packed_b_view(b, w->domain, w->config.N, w->config.K, &weight);
    if (rc)
        return rc;
    auto *commands = static_cast<uint64_t *>(w->commands.vaddr);
    for (const auto &item : w->relocations)
        commands[item.first] = E(T_CNA, R_CNA_DCOMP_ADDR0, uint32_t(weight->dma_addr + item.second));
    rknpu2_matmul_open::cpu::f16_pack_a_native_k8_m8(w->config.M, w->config.K, w->aligned_k, a,
                                                     static_cast<uint16_t *>(w->input.vaddr), 8);
    std::memcpy(w->bias.vaddr, bias, w->config.N * sizeof(float));
    if ((rc = rk_npu_mem_sync(w->domain->ctx, &w->input, RK_NPU_SYNC_TO_DEVICE)))
        return rc;
    if ((rc = rk_npu_mem_sync(w->domain->ctx, &w->bias, RK_NPU_SYNC_TO_DEVICE)))
        return rc;
    rc = detail::do_submit_multicore(w->domain->ctx->fd, w->tasks.obj_addr, w->count, w->mask, w->starts,
                                     w->counts, w->domain->id, w->config.timeout_ms, nullptr);
    if (rc)
        return rc;
    if ((rc = rk_npu_mem_sync(w->domain->ctx, &w->output, RK_NPU_SYNC_FROM_DEVICE)))
        return rc;
    *view = {static_cast<const float *>(w->output.vaddr),
             uint64_t(w->physical_m) * w->aligned_n,
             w->config.M,
             w->config.N,
             w->physical_m,
             w->aligned_n,
             1.f / 8192.f};
    return RK_NPU_OK;
}

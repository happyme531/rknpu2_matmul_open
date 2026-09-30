#include "rk_npu_i8_dpu_chain.h"
#include "rk_npu_kn_plan.h"
#include "rk_npu_moe_regs.h"
#include <algorithm>
#include <cstring>

namespace rknpu2_matmul_open::detail {
namespace {
constexpr int DPU_CHANNELS = 1024;
// Conservative storage bound, including the PC tail and alignment.
constexpr int DPU_COMMAND_WORDS = 96;
void patch(std::vector<uint64_t>& body, uint32_t reg, uint32_t value) {
    for (auto& word : body)
        if ((word & 65535u) == reg) {
            word = (word & 0xffff00000000ffffull) | (uint64_t(value) << 16);
            return;
        }
    throw std::runtime_error("missing DPU register");
}
std::vector<uint64_t> add_body(int rows, int channels, uint64_t a,
                               uint64_t b, uint64_t c, bool fp32) {
    auto v = moe::middle_body(rows, channels, a, b, c, 1.f, false);
    patch(v, 0x4010, ((fp32 ? 5u : 4u) << 29) | (4u << 26) | 4u);
    patch(v, 0x4024, rows * 16);
    patch(v, 0x4030, rows - 1);
    patch(v, 0x4040, 0x53);
    patch(v, 0x4048, 0);
    patch(v, 0x405c, rows - 1);
    patch(v, 0x4070, 0x10c203c0);
    patch(v, 0x40c0, rows * 16);
    patch(v, 0x500c, rows - 1);
    patch(v, 0x5040, rows * 16);
    patch(v, 0x5044, 0x27881);
    patch(v, 0x504c, 0);
    return v;
}
int panel_rows(const I8KnPlanConfig& c) {
    return c.c_layout == RK_NPU_I8_C_LAYOUT_PANEL8 ? 8 :
           c.c_layout == RK_NPU_I8_C_LAYOUT_PANEL16 ? 16 : c.M;
}
}

void I8DpuChain::memory(const I8KnPlanConfig& c, uint64_t gemm_commands,
                        uint64_t gemm_tasks, uint64_t& control, uint64_t& data) {
    const int n = align_up(c.N, 32), waves = ceil_div(c.K, c.k_tile);
    const int nt = std::min(n, align_up(c.n_tile, 32));
    uint64_t chunks = 0;
    for (int n0 = 0; n0 < n; n0 += nt)
        chunks += ceil_div(std::min(nt, n - n0), DPU_CHANNELS);
    const uint64_t dpu_tasks = chunks *
        (uint64_t(waves - 1) * (c.M / panel_rows(c)) +
         (c.npu_dequant ? c.M : 0));
    control = gemm_commands + gemm_tasks +
        dpu_tasks * (DPU_COMMAND_WORDS * 8 + sizeof(rknpu_task));
    data = c.npu_dequant ? uint64_t(n) * (c.M + 1) * 4 : 0;
}

I8DpuChain::~I8DpuChain() {
    if (!domain_) return;
    result_.release(domain_->ctx);
    coefficients_.release(domain_->ctx);
    rk_npu_mem_free(domain_->ctx, &tasks_);
    rk_npu_mem_free(domain_->ctx, &commands_);
    release_domain(domain_);
}

int I8DpuChain::prepare(rk_npu_iommu_domain* domain, const I8KnPlanConfig& c,
                        const std::vector<rk_npu_matmul_i8_plan*>& waves,
                        const std::vector<rk_npu_mem>& outputs) {
    if (domain_ || !domain || waves.empty() || outputs.size() != waves.size())
        return RK_NPU_ERR_PARAM;
    domain_ = domain;
    retain_domain(domain_);
    M_ = c.M; N_ = c.N; padded_n_ = align_up(c.N, 32);
    core_mask_ = c.npu_core_mask; timeout_ms_ = c.timeout_ms;
    int rc = RK_NPU_OK;
    if (c.npu_dequant) {
        coefficient_scales_.resize(N_);
        row_scales_.resize(M_);
        rc = coefficients_.alloc(domain_, uint64_t(padded_n_) * 4);
        if (rc == RK_NPU_OK) rc = result_.alloc(domain_, uint64_t(M_) * padded_n_ * 4);
        if (rc != RK_NPU_OK) return rc;
    }
    const int nt = std::min(padded_n_, align_up(c.n_tile, 32));
    const int groups = ceil_div(padded_n_, nt), cores = __builtin_popcount(core_mask_);
    const int rows = panel_rows(c);
    std::vector<std::vector<uint64_t>> bodies[3];
    std::vector<int> bases[3], enables[3];
    int offset = 0, group = 0;
    auto append = [&](int core, std::vector<uint64_t> body, int enable,
                       int wave, uint64_t weight_offset, int row) {
        if (enable == 24 && body.size() + 4 > DPU_COMMAND_WORDS)
            throw std::runtime_error("DPU command budget");
        for (int j = 0; j < int(body.size()); ++j) {
            const auto reg = body[j] & 65535u;
            if (wave >= 0 && reg == R_CNA_DCOMP_ADDR0)
                weights_.push_back({offset + j, wave, weight_offset});
            if (row >= 0 && reg == 0x4048) rows_.push_back({offset + j, row});
        }
        bases[core].push_back(offset);
        offset += align_up(int(body.size()) + 4, 2);
        enables[core].push_back(enable);
        bodies[core].push_back(std::move(body));
    };
    try {
        for (int core = 0; core < cores; ++core) {
            starts_[core] = task_count_;
            const int end_group = group + groups / cores + (core < groups % cores);
            for (; group < end_group; ++group) {
                const int n0 = group * nt, width = std::min(nt, padded_n_ - n0);
                for (int w = 0; w < int(waves.size()); ++w) {
                    std::vector<I8GemmBody> gemms;
                    rc = export_i8_n_group(waves[w], group, gemms);
                    if (rc != RK_NPU_OK) return rc;
                    for (auto& body : gemms) {
                        if (waves.size() == 1 && c.npu_dequant)
                            patch(body.regs, 0x4010, 5u << 29);
                        append(core, std::move(body.regs), 13, w, body.weight_offset, -1);
                    }
                    if (!w) continue;
                    for (int m0 = 0; m0 < M_; m0 += rows)
                        for (int n = n0; n < n0 + width; n += DPU_CHANNELS) {
                            const uint64_t pos = uint64_t(m0) * padded_n_ * 4 + uint64_t(n) * rows * 4;
                            append(core, add_body(rows, std::min(DPU_CHANNELS, n0 + width - n),
                                outputs[0].dma_addr + pos, outputs[w].dma_addr + pos,
                                outputs[0].dma_addr + pos,
                                c.npu_dequant && w + 1 == int(waves.size())), 24, -1, 0, -1);
                        }
                }
                if (c.npu_dequant) {
                    for (int m = 0; m < M_; ++m)
                        for (int n = n0; n < n0 + width; n += DPU_CHANNELS) {
                            const uint64_t pos = uint64_t(m / rows * rows) * padded_n_ * 4 +
                                                  uint64_t(n) * rows * 4 + uint64_t(m % rows) * 16;
                            append(core, moe::middle_body(rows, std::min(DPU_CHANNELS, n0 + width - n),
                                outputs[0].dma_addr + pos, coefficients_.mem.dma_addr + uint64_t(n) * 4,
                                result_.mem.dma_addr + (uint64_t(m) * padded_n_ + n) * 4,
                                1.f, false), 24, -1, 0, m);
                        }
                }
            }
            counts_[core] = uint32_t(bodies[core].size());
            task_count_ += counts_[core];
        }
    } catch (const std::exception&) { return RK_NPU_ERR_PARAM; }
    rc = rk_npu_mem_alloc(domain_, uint64_t(offset) * 8, RK_NPU_MEM_NON_CACHEABLE, &commands_);
    if (rc == RK_NPU_OK)
        rc = rk_npu_mem_alloc(domain_, uint64_t(task_count_) * sizeof(rknpu_task),
                              RK_NPU_MEM_KERNEL_MAPPING, &tasks_);
    if (rc != RK_NPU_OK) return rc;
    auto* cmd = static_cast<uint64_t*>(commands_.vaddr);
    auto* tasks = static_cast<rknpu_task*>(tasks_.vaddr);
    std::memset(tasks, 0, size_t(task_count_) * sizeof(*tasks));
    for (int core = 0; core < cores; ++core) {
        write_chain(cmd, tasks + starts_[core], commands_.dma_addr,
                     bodies[core], bases[core], {13, 0, 13});
        for (int i = 0; i < int(bodies[core].size()); ++i) {
            const uint32_t enable = enables[core][i];
            cmd[bases[core][i] + bodies[core][i].size() + 3] = E(T_PC, R_OPERATION_ENABLE, enable);
            tasks[starts_[core] + i].enable_mask = enable;
            tasks[starts_[core] + i].op_idx = enable == 24 ? 1 : 0;
        }
    }
    return RK_NPU_OK;
}

int I8DpuChain::run(const I8KnWeights& weights, const float* a_scale,
                    const float* w_scale, bool dequant) {
    if (!domain_ || !commands_.vaddr || weights.compressed() ||
        dequant != bool(result_.mem.handle)) return RK_NPU_ERR_PARAM;
    auto* cmd = static_cast<uint64_t*>(commands_.vaddr);
    for (const auto& p : weights_) {
        const auto* weight = weights.wave(p.wave);
        if (!weight || weight->ctx_id != domain_->ctx->id || weight->iommu_domain_id != domain_->id)
            return RK_NPU_ERR_DOMAIN;
        cmd[p.index] = E(T_CNA, R_CNA_DCOMP_ADDR0, uint32_t(weight->dma_addr + p.offset));
    }
    if (dequant) {
        if (!a_scale || !w_scale) return RK_NPU_ERR_PARAM;
        try {
            for (int m = 0; m < M_; ++m)
                row_scales_[m] = uint32_t(moe::half_bits(a_scale[m])) << 16;
        } catch (const std::exception&) { return RK_NPU_ERR_PARAM; }
        if (!coefficients_valid_ ||
            std::memcmp(coefficient_scales_.data(), w_scale, size_t(N_) * sizeof(float))) {
            coefficients_valid_ = false;
            int rc = coefficients_.begin_cpu_write();
            if (rc != RK_NPU_OK) return rc;
            try {
                auto* coeff = static_cast<uint32_t*>(coefficients_.mem.vaddr);
                for (int n = 0; n < padded_n_; ++n)
                    coeff[n] = n < N_ ? moe::half_bits(w_scale[n]) : 0;
            } catch (const std::exception&) { rc = RK_NPU_ERR_PARAM; }
            const int end = coefficients_.end_cpu_access();
            if (rc != RK_NPU_OK) return rc;
            if (end != RK_NPU_OK) return end;
            std::memcpy(coefficient_scales_.data(), w_scale, size_t(N_) * sizeof(float));
            coefficients_valid_ = true;
        }
        for (const auto& p : rows_)
            cmd[p.index] = E(T_DPU, 0x4048, row_scales_[p.row]);
        const int rc = result_.end_cpu_access();
        if (rc != RK_NPU_OK) return rc;
    }
    return do_submit_multicore(domain_->ctx->fd, tasks_.obj_addr, task_count_,
        core_mask_, starts_, counts_, domain_->id, timeout_ms_);
}

int I8DpuChain::copy_dequant(void* output, bool fp16) {
    if (!output || !result_.mem.handle) return RK_NPU_ERR_PARAM;
    int rc = result_.begin_cpu_read();
    if (rc != RK_NPU_OK) return rc;
    const float* src = static_cast<const float*>(result_.mem.vaddr);
    if (fp16) {
#if defined(__aarch64__)
        auto* dst = static_cast<__fp16*>(output);
#else
        auto* dst = static_cast<_Float16*>(output);
#endif
        #pragma omp parallel for if(M_ > 1)
        for (int m = 0; m < M_; ++m)
            for (int n = 0; n < N_; ++n) dst[size_t(m) * N_ + n] = src[size_t(m) * padded_n_ + n];
    } else {
        auto* dst = static_cast<float*>(output);
        #pragma omp parallel for if(M_ > 1)
        for (int m = 0; m < M_; ++m)
            std::memcpy(dst + size_t(m) * N_, src + size_t(m) * padded_n_, size_t(N_) * 4);
    }
    return result_.end_cpu_access();
}
}

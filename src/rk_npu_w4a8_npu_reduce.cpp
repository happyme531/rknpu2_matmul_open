#include "rk_npu_w4a8_npu_reduce.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
std::vector<uint64_t> reduction_regs(int rows, int waves, int storage_rows, uint64_t a, uint64_t b,
                                     uint64_t c) {
    using namespace rknpu2_matmul_open::detail;
    const int banks = ceil_div(rows * 64 * (waves + 1), 32768), stride = align_up(rows, 4) * 16;
    std::vector<uint64_t> v;
    auto add = [&](uint64_t target, uint32_t reg, uint64_t value) {
        v.push_back(E(target, reg, uint32_t(value)));
    };
    add(T_CNA, 0x1100, 0);
    add(T_CNA, 0x1104, 0);
    add(T_DPU, 0x4004, 14);
    add(T_CNA, 0x100c, storage_rows == rows ? 0x90 : 0x20000090);
    add(T_CNA, 0x1010, (waves + 1) << 4);
    add(T_CNA, 0x1014, 10);
    add(T_CNA, 0x1020, (rows * 2 << 16) | waves);
    add(T_CNA, 0x1024, (7 << 16) | 16);
    add(T_CNA, 0x1028, rows);
    add(T_CNA, 0x102c, rows);
    add(T_CNA, 0x1030, waves * 512);
    add(T_CNA, 0x1034, waves * 64);
    add(T_CNA, 0x1038, (2 << 24) | (waves << 16) | 8);
    add(T_CNA, 0x1040, ((12 - banks) << 4) | banks);
    add(T_CNA, 0x1044, rows);
    add(T_CNA, 0x104c, 11);
    for (int reg = 0x1050; reg <= 0x105c; reg += 4)
        add(T_CNA, reg, 0x10000);
    add(T_CNA, 0x1070, a);
    add(T_CNA, 0x1078, 0xf000f);
    add(T_CNA, 0x107c, storage_rows == rows ? rows * 8 : storage_rows * 2);
    add(T_CNA, 0x1080, storage_rows * 2 * std::max(waves - 4, 0));
    add(T_CNA, 0x1084, (rows * 2 << 16) | waves);
    add(T_CNA, 0x1088, 16);
    add(T_CNA, 0x1110, b);
    add(T_CNA, 0x1180, 7);
    add(T_CORE, 0x3010, 0x101);
    add(T_CORE, 0x3014, rows - 1);
    add(T_CORE, 0x3018, 15);
    add(T_CORE, 0x301c, 0);
    add(T_CORE, 0x3030, 0);
    add(T_DPU, 0x400c, 0x1e4);
    add(T_DPU, 0x4010, 0x84000001);
    add(T_DPU, 0x4014, 0);
    add(T_DPU, 0x4020, c);
    add(T_DPU, 0x4024, stride);
    add(T_DPU, 0x4030, rows - 1);
    add(T_DPU, 0x4034, 0);
    add(T_DPU, 0x4038, 0);
    add(T_DPU, 0x403c, (7 << 16) | 15);
    add(T_DPU, 0x4040, 0x53);
    add(T_DPU, 0x4044, 0);
    add(T_DPU, 0x4048, 0);
    add(T_DPU, 0x404c, 0);
    add(T_DPU, 0x4050, 0x36e);
    add(T_DPU, 0x4054, 0);
    add(T_DPU, 0x4058, 15);
    add(T_DPU, 0x405c, rows - 1);
    add(T_DPU, 0x4060, 0x53);
    add(T_DPU, 0x4064, 0);
    add(T_DPU, 0x4068, 0);
    add(T_DPU, 0x406c, 0);
    add(T_DPU, 0x4070, 0x383);
    add(T_DPU, 0x4074, 0);
    add(T_DPU, 0x4078, 1);
    add(T_DPU, 0x407c, 0);
    add(T_DPU, 0x4080, 0);
    add(T_DPU, 0x4084, 1);
    add(T_DPU, 0x4088, 0);
    add(T_DPU, 0x40c0, stride * 2);
    add(T_DPU, 0x40c4, 0);
    return v;
}
} // namespace

namespace rknpu2_matmul_open::detail {
int W4A8NpuReduce::describe(const rk_npu_i4_config &cfg, const std::vector<I4Tile> &tiles,
                            std::vector<Part> &parts, rk_npu_w4a8_memory_info_ex &info) {
    const int waves = ceil_div(cfg.K, cfg.k_tile);
    if (waves > 31)
        return RK_NPU_ERR_PARAM;
    int limit = 512;
    while (limit * 64 * (waves + 1) > 11 * 32768)
        limit /= 2;
    uint64_t bytes = 0, guard = 256 * 1024;
    for (const auto &t : tiles) {
        if (t.m % 2 || t.m0 % 2 || t.n % 64)
            return RK_NPU_ERR_PARAM;
        const int storage = t.m * t.n / 16;
        // 4096 crop pitch is board-verified; 8192 failed even without overlap.
        if (storage > 4096)
            return RK_NPU_ERR_PARAM;
        guard = std::max(guard, uint64_t(t.m) * t.n * 2 * waves);
        for (int r = 0; r < storage; r += limit) {
            const int rows = std::min(limit, storage - r);
            parts.push_back({t, r, rows, bytes});
            bytes += uint64_t(align_up(rows, 4)) * 64;
        }
    }
    const uint64_t command_words =
        align_up(int(reduction_regs(1, waves, 1, 0, 0, 0).size()) + 4, 2);
    if (parts.size() < size_t(__builtin_popcount(cfg.npu_core_mask)))
        return RK_NPU_ERR_PARAM;
    if (parts.size() > INT_MAX / command_words)
        return RK_NPU_ERR_NOMEM;
    info.base.backend.partial_bytes = uint64_t(cfg.M) * align_up(cfg.N, 64) * 2 * waves + guard;
    info.reduction_output_bytes = bytes + 256 * 1024;
    info.reduction_weight_bytes = uint64_t(waves) * 512;
    info.reduction_control_bytes = parts.size() * (command_words * 8 + sizeof(rknpu_task));
    if (info.base.backend.partial_bytes > UINT32_MAX || info.reduction_output_bytes > UINT32_MAX ||
        info.reduction_control_bytes > UINT32_MAX)
        return RK_NPU_ERR_NOMEM;
    return RK_NPU_OK;
}
int W4A8NpuReduce::query(const rk_npu_i4_config &cfg, I4InputLayout layout,
                         rk_npu_w4a8_memory_info_ex &info) {
    if (ceil_div(cfg.K, cfg.k_tile) > 31)
        return RK_NPU_ERR_PARAM;
    if (uint64_t(cfg.M) * align_up(cfg.N, 64) * 2 * ceil_div(cfg.K, cfg.k_tile) > UINT32_MAX)
        return RK_NPU_ERR_NOMEM;
    uint32_t starts[3]{}, counts[3]{};
    std::vector<Part> parts;
    return describe(cfg, i4_tiles(cfg, starts, counts, layout), parts, info);
}
W4A8NpuReduce::~W4A8NpuReduce() {
    if (!domain_)
        return;
    rk_npu_mem_free(domain_->ctx, &regcmd_);
    rk_npu_mem_free(domain_->ctx, &tasks_);
    reduced_.release(domain_->ctx);
    weights_.release(domain_->ctx);
    release_domain(domain_);
}
int W4A8NpuReduce::prepare(rk_npu_iommu_domain *domain, const I4Plan &p) {
    if (domain_ || !domain || !domain->ctx || !p.retains_partials())
        return RK_NPU_ERR_PARAM;
    domain_ = domain;
    retain_domain(domain_);
    cfg_ = p.config();
    const int waves = p.wave_count();
    rk_npu_w4a8_memory_info_ex info{};
    int rc = describe(cfg_, p.tiles(), parts_, info);
    if (rc)
        return rc;
    rc = reduced_.alloc(domain_, info.reduction_output_bytes);
    if (rc)
        return rc;
    rc = weights_.alloc(domain_, info.reduction_weight_bytes);
    if (rc)
        return rc;
    rc = weights_.begin_cpu_write();
    if (rc)
        return rc;
    auto *w = static_cast<int16_t *>(weights_.mem.vaddr);
    std::memset(w, 0, size_t(waves) * 512);
    for (int wave = 0; wave < waves; ++wave)
        for (int c = 0; c < 8; ++c) {
            w[((wave * 2) * 8 + c) * 16 + c] = 16;
            w[((wave * 2 + 1) * 8 + c) * 16 + c] = 1;
        }
    rc = weights_.end_cpu_access();
    if (rc)
        return rc;
    std::vector<std::vector<uint64_t>> bodies;
    std::vector<int> bases;
    int offset = 0;
    for (const auto &t : parts_) {
        bases.push_back(offset);
        bodies.push_back(reduction_regs(t.rows, waves, t.tile.m * t.tile.n / 16,
                                        p.retained_partials().dma_addr +
                                            t.tile.output_offset * waves + uint64_t(t.row0) * 32,
                                        weights_.mem.dma_addr, reduced_.mem.dma_addr + t.offset));
        offset += align_up(int(bodies.back().size()) + 4, 2);
    }
    rc = rk_npu_mem_alloc(domain_, uint64_t(offset) * 8, 0, &regcmd_);
    if (rc)
        return rc;
    rc = rk_npu_mem_alloc(domain_, parts_.size() * sizeof(rknpu_task), RK_NPU_MEM_KERNEL_MAPPING,
                          &tasks_);
    if (rc)
        return rc;
    std::memset(tasks_.vaddr, 0, size_t(tasks_.size));
    const int cores = __builtin_popcount(cfg_.npu_core_mask), count = int(parts_.size());
    int start = 0;
    for (int c = 0; c < cores; ++c) {
        const int n = count / cores + int(c < count % cores);
        starts_[c] = start;
        counts_[c] = n;
        std::vector<std::vector<uint64_t>> group(bodies.begin() + start,
                                                 bodies.begin() + start + n);
        std::vector<int> positions(bases.begin() + start, bases.begin() + start + n);
        write_chain(static_cast<uint64_t *>(regcmd_.vaddr),
                    static_cast<rknpu_task *>(tasks_.vaddr) + start, regcmd_.dma_addr, group,
                    positions, {13, 0, 13});
        start += n;
    }
    return RK_NPU_OK;
}
void W4A8NpuReduce::output(const W4A8Reduction &d) {
    const auto *src = static_cast<const int32_t *>(reduced_.mem.vaddr);
#pragma omp parallel for num_threads(cfg_.cpu_threads)                                             \
    schedule(static) if (cfg_.cpu_threads > 1 && int64_t((cfg_.M / 2)) * cfg_.N >= 65536)
    for (int i = 0; i < int(parts_.size()); ++i) {
        const auto &p = parts_[i];
        const auto *a = src + p.offset / 4;
        const int pm = p.tile.m / 2, pr = align_up(p.rows, 4);
        const int nb_first = p.row0 / pm, nb_end = (p.row0 + p.rows + pm - 1) / pm;
        for (int nb0 = nb_first; nb0 < nb_end; nb0 += 4)
            for (int mr = 0; mr < pm; ++mr)
                for (int nb = nb0; nb < std::min(nb0 + 4, nb_end); ++nb) {
                    const int r = nb * pm + mr - p.row0;
                    if (r < 0 || r >= p.rows)
                        continue;
                    const int m = p.tile.m0 / 2 + mr, n = p.tile.n0 + nb * 8;
                    for (int lane = 0; lane < 8 && n + lane < cfg_.N; lane += 4) {
                        const auto *q = a + (lane / 4 * pr + r) * 4;
#if defined(__aarch64__)
                        if (n + lane + 4 <= cfg_.N) {
                            const auto sum =
                                vaddq_s32(vld1q_s32(q), vld1q_s32(d.correction + n + lane));
                            const auto y =
                                vmulq_f32(vmulq_f32(vcvtq_f32_s32(sum), vdupq_n_f32(d.a_scale[m])),
                                          vld1q_f32(d.w_scale + n + lane));
                            if (!d.half)
                                vst1q_f32(static_cast<float *>(d.output) + size_t(m) * cfg_.N + n +
                                              lane,
                                          y);
                            else {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                                vst1_u16(static_cast<uint16_t *>(d.output) + size_t(m) * cfg_.N +
                                             n + lane,
                                         vreinterpret_u16_f16(vcvt_f16_f32(y)));
#else
                                float values[4];
                                vst1q_f32(values, y);
                                for (int j = 0; j < 4; ++j)
                                    static_cast<uint16_t *>(
                                        d.output)[size_t(m) * cfg_.N + n + lane + j] =
                                        rknpu2_matmul_open::bits::float_to_half(values[j]);
#endif
                            }
                            continue;
                        }
#endif
                        for (int j = 0; j < 4 && n + lane + j < cfg_.N; ++j) {
                            const int32_t sum = q[j] + d.correction[n + lane + j];
                            const float y = (float(sum) * d.a_scale[m]) * d.w_scale[n + lane + j];
                            const auto ix = size_t(m) * cfg_.N + n + lane + j;
                            if (d.half)
                                static_cast<uint16_t *>(d.output)[ix] = rknpu2_matmul_open::bits::float_to_half(y);
                            else
                                static_cast<float *>(d.output)[ix] = y;
                        }
                    }
                }
    }
}
int W4A8NpuReduce::run(const W4A8Reduction &data, rk_npu_w4a8_timings &t) {
    auto start = Clock::now();
    int rc =
        do_submit_multicore(domain_->ctx->fd, tasks_.obj_addr, int(parts_.size()),
                            cfg_.npu_core_mask, starts_, counts_, domain_->id, cfg_.timeout_ms);
    t.reduce_us += elapsed(start); // blocking Conv submit, not CPU work
    if (!rc) {
        start = Clock::now();
        rc = reduced_.begin_cpu_read();
        t.sync_us += elapsed(start);
        if (!rc) {
            start = Clock::now();
            output(data);
            t.dequant_us += elapsed(start);
            start = Clock::now();
            rc = reduced_.end_cpu_access();
            t.sync_us += elapsed(start);
        }
    }
    return rc;
}
} // namespace rknpu2_matmul_open::detail

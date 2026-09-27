#include "rk_npu_flatquant.h"
#include "rk_npu_cpu_kernels.h"
#include "rk_npu_flatquant_regs.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rknpu2_matmul_open::detail {
namespace {
using Clock = std::chrono::steady_clock;
double us(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
int batch_size(const rk_npu_w4a4_linear_config &c) {
    return std::min(c.M, c.transform_batch ? c.transform_batch : 64);
}
int group_size(const rk_npu_w4a4_linear_config &c, const FlatShape &s, int batch) {
    if (c.transform_ubatch)
        return c.transform_ubatch;
    if (batch < 16)
        return 1;
    return s.lp <= 48 ? (batch >= 48 ? 16 : 8) : (batch > 32 ? 8 : 4);
}
int tile1(const FlatShape &s) { return std::min(1022, 4 * CBUF_BANK_SIZE / (s.rp * 2)) / 4 * 4; }
int tile2(const FlatShape &s) { return std::min(1022, 4 * CBUF_BANK_SIZE / (s.lp * 4)) / 2 * 2; }
int task_count(int rows, int group, const FlatShape &s) {
    int tasks = 0;
    for (int m = 0; m < rows; m += group) {
        int g = std::min(group, rows - m);
        tasks += ceil_div(g * s.lp, tile1(s)) + ceil_div(g * s.rp / 2, tile2(s));
    }
    return tasks;
}
} // namespace
int flat_shape(int L, int R, FlatShape &out) {
    if (L < 1 || R < 1 || L > 128 || R > 128)
        return RK_NPU_ERR_PARAM;
    out = {L, R, align_up(L, 16), align_up(R, 32)};
    return RK_NPU_OK;
}
FlatWeights::~FlatWeights() {
    if (arena_.domain)
        arena_.release(arena_.domain->ctx);
}
int FlatWeights::prepare(rk_npu_iommu_domain *domain, int L, int R, const float *left,
                         const float *right) {
    if (!domain || !left || !right || arena_.domain)
        return RK_NPU_ERR_PARAM;
    int rc = flat_shape(L, R, shape_);
    if (rc)
        return rc;
    const int lp = shape_.lp, rp = shape_.rp;
    std::vector<uint16_t> a(size_t(rp) * rp), b(size_t(lp) * lp * 4);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < R; ++j) {
            float x = right[size_t(i) * R + j];
            if (!std::isfinite(x) || std::fabs(x) > 65504.f)
                return RK_NPU_ERR_PARAM;
            a[size_t(i) * rp + j] = rknpu2_matmul_open::bits::float_to_half(x);
        }
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j) {
            float x = left[size_t(i) * L + j];
            if (!std::isfinite(x) || std::fabs(x) > 65504.f)
                return RK_NPU_ERR_PARAM;
            b[size_t(2 * i) * (2 * lp) + 2 * j] = b[size_t(2 * i + 1) * (2 * lp) + 2 * j + 1] =
                rknpu2_matmul_open::bits::float_to_half(x);
        }
    rc = arena_.alloc(domain, uint64_t(a.size() + b.size()) * 2);
    if (rc)
        return rc;
    rc = arena_.begin_cpu_write();
    if (rc)
        return rc;
    auto *dst = static_cast<uint16_t *>(arena_.mem.vaddr);
    rknpu2_matmul_open::cpu::f16_pack_b_native_n16_k32(rp, rp, rp, rp, a.data(), dst);
    rknpu2_matmul_open::cpu::f16_pack_b_native_n16_k32(2 * lp, 2 * lp, 2 * lp, 2 * lp, b.data(), dst + a.size());
    return arena_.end_cpu_access();
}
int FlatPlan::query(const rk_npu_w4a4_linear_config &c, const FlatShape &s, uint64_t &bytes) {
    FlatShape checked;
    if (flat_shape(s.L, s.R, checked) || c.M < 1 || c.transform_batch < 0 ||
        c.transform_batch > 256 || c.transform_ubatch < 0 || c.transform_ubatch > 32)
        return RK_NPU_ERR_PARAM;
    int batch = batch_size(c), group = group_size(c, s, batch), tail = c.M % batch;
    int slots = c.pipeline && c.M > batch ? 2 : 1;
    int tasks = task_count(batch, group, s) + (tail ? task_count(tail, group, s) : 0);
    bytes = slots * (uint64_t(batch) * s.lp * s.rp * 2 * 3 +
                     uint64_t(tasks) * (FLAT_COMMAND_WORDS * 8 + sizeof(rknpu_task)));
    return bytes > UINT32_MAX ? RK_NPU_ERR_NOMEM : RK_NPU_OK;
}
FlatPlan::~FlatPlan() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    if (thread_.joinable())
        thread_.join();
    if (!domain_)
        return;
    for (auto &slot : slots_) {
        for (auto *p : {&slot.full, &slot.tail}) {
            rk_npu_mem_free(domain_->ctx, &p->regs);
            rk_npu_mem_free(domain_->ctx, &p->tasks);
        }
        slot.input.release(domain_->ctx);
        slot.middle.release(domain_->ctx);
        slot.output.release(domain_->ctx);
    }
    release_domain(domain_);
}
int FlatPlan::build(Commands &cmd, Slot &slot, int rows) {
    for (int m = 0; m < rows; m += ubatch_)
        cmd.groups.push_back({m, std::min(ubatch_, rows - m)});
    int cores = std::min(int(cmd.groups.size()), __builtin_popcount(cfg_.npu_core_mask));
    cmd.mask = cfg_.npu_core_mask;
    if (cores < __builtin_popcount(cmd.mask))
        cmd.mask = cores == 2 ? 3u : (cmd.mask & (~cmd.mask + 1u));
    std::vector<std::vector<uint64_t>> bodies;
    std::vector<int> bases, per_group;
    for (const auto &g : cmd.groups) {
        int start = int(bodies.size());
        uint64_t offset = uint64_t(g.m0) * shape_.lp * shape_.rp * 2;
        const int h = g.rows * shape_.lp, second = g.rows * shape_.rp / 2;
        auto add = [&](std::vector<uint64_t> regs, uint64_t weight_offset) {
            int base = int(bodies.size()) * FLAT_COMMAND_WORDS;
            for (int j = 0; j < int(regs.size()); ++j)
                if ((regs[j] & 65535) == 0x1110)
                    cmd.bindings.emplace_back(base + j, weight_offset);
            bases.push_back(base);
            bodies.push_back(std::move(regs));
        };
        for (int r = 0; r < h; r += tile1(shape_))
            add(flat_regs(std::min(tile1(shape_), h - r), shape_.rp, shape_.rp, h, true,
                          slot.input.mem.dma_addr + offset + uint64_t(r) * 16, 0,
                          slot.middle.mem.dma_addr + offset + uint64_t(r) * shape_.rp * 2),
                0);
        for (int r = 0; r < second; r += tile2(shape_))
            add(flat_regs(std::min(tile2(shape_), second - r), shape_.lp * 2, shape_.lp * 2, second,
                          false, slot.middle.mem.dma_addr + offset + uint64_t(r) * 16, 0,
                          slot.output.mem.dma_addr + offset + uint64_t(r) * 16),
                uint64_t(shape_.rp) * shape_.rp * 2);
        per_group.push_back(int(bodies.size()) - start);
    }
    cmd.tasks_count = int(bodies.size());
    for (const auto &body : bodies)
        if (body.size() + 4 > FLAT_COMMAND_WORDS)
            return RK_NPU_ERR_PARAM;
    int rc =
        rk_npu_mem_alloc(domain_, uint64_t(cmd.tasks_count) * FLAT_COMMAND_WORDS * 8, 0, &cmd.regs);
    if (rc)
        return rc;
    rc = rk_npu_mem_alloc(domain_, uint64_t(cmd.tasks_count) * sizeof(rknpu_task),
                          RK_NPU_MEM_KERNEL_MAPPING, &cmd.tasks);
    if (rc)
        return rc;
    std::memset(cmd.regs.vaddr, 0, size_t(cmd.regs.size));
    std::memset(cmd.tasks.vaddr, 0, size_t(cmd.tasks.size));
    int gi = 0, ti = 0;
    for (int c = 0; c < cores; ++c) {
        int groups = int(per_group.size()) / cores + int(c < int(per_group.size()) % cores),
            count = 0;
        for (int j = 0; j < groups; ++j)
            count += per_group[gi++];
        cmd.starts[c] = ti;
        cmd.counts[c] = count;
        std::vector<std::vector<uint64_t>> part(bodies.begin() + ti, bodies.begin() + ti + count);
        std::vector<int> positions(bases.begin() + ti, bases.begin() + ti + count);
        write_chain(static_cast<uint64_t *>(cmd.regs.vaddr),
                    static_cast<rknpu_task *>(cmd.tasks.vaddr) + ti, cmd.regs.dma_addr, part,
                    positions, {13, 0, 13});
        ti += count;
    }
    return RK_NPU_OK;
}
int FlatPlan::prepare(rk_npu_iommu_domain *domain, const rk_npu_w4a4_linear_config &cfg,
                      const FlatShape &shape) {
    uint64_t bytes = 0;
    int rc = query(cfg, shape, bytes);
    if (rc || !domain || domain_)
        return rc ? rc : RK_NPU_ERR_PARAM;
    domain_ = domain;
    retain_domain(domain_);
    cfg_ = cfg;
    shape_ = shape;
    batch_ = batch_size(cfg);
    ubatch_ = group_size(cfg, shape, batch_);
    blocks_ = ceil_div(cfg.M, batch_);
    tail_ = cfg.M % batch_;
    slot_count_ = cfg.pipeline && blocks_ > 1 ? 2 : 1;
    for (int i = 0; i < slot_count_; ++i) {
        auto &s = slots_[i];
        uint64_t capacity = uint64_t(batch_) * shape_.lp * shape_.rp * 2;
        for (auto *buffer : {&s.input, &s.middle, &s.output}) {
            rc = buffer->alloc(domain_, capacity);
            if (rc)
                return rc;
        }
        rc = build(s.full, s, batch_);
        if (rc)
            return rc;
        if (tail_) {
            rc = build(s.tail, s, tail_);
            if (rc)
                return rc;
        }
    }
    if (slot_count_ == 2)
        thread_ = std::thread(&FlatPlan::worker, this);
    return RK_NPU_OK;
}
int FlatPlan::bind(const FlatWeights &w) {
    const auto &mem = w.memory();
    const auto &s = w.shape();
    if (mem.ctx_id != domain_->ctx->id || mem.iommu_domain_id != domain_->id)
        return RK_NPU_ERR_DOMAIN;
    if (s.L != shape_.L || s.R != shape_.R)
        return RK_NPU_ERR_PARAM;
    if (weight_dma_ == mem.dma_addr)
        return RK_NPU_OK;
    for (int i = 0; i < slot_count_; ++i)
        for (auto *cmd : {&slots_[i].full, &slots_[i].tail})
            for (const auto &binding : cmd->bindings)
                static_cast<uint64_t *>(cmd->regs.vaddr)[binding.first] =
                    E(T_CNA, 0x1110, uint32_t(mem.dma_addr + binding.second));
    weight_dma_ = mem.dma_addr;
    return RK_NPU_OK;
}
FlatPlan::Commands &FlatPlan::commands(int b) {
    return tail_ && b == blocks_ - 1 ? slots_[b % slot_count_].tail : slots_[b % slot_count_].full;
}
int FlatPlan::submit_job(int b) {
    auto &c = commands(b);
    return do_submit_multicore(domain_->ctx->fd, c.tasks.obj_addr, c.tasks_count, c.mask, c.starts,
                               c.counts, domain_->id, cfg_.timeout_ms);
}
int FlatPlan::pack_job(int b, const void *A, bool half, FlatMetrics &t) {
    auto &input = slots_[b % slot_count_].input;
    auto start = Clock::now();
    int rc = input.begin_cpu_write();
    t.sync += us(start);
    if (rc)
        return rc;
    start = Clock::now();
    flat_pack_input(A, half, cfg_.K, b * batch_, shape_, commands(b).groups, cfg_.cpu_threads,
                    static_cast<uint16_t *>(input.mem.vaddr));
    t.pack += us(start);
    start = Clock::now();
    rc = input.end_cpu_access();
    t.sync += us(start);
    return rc;
}
int FlatPlan::quant_job(int b, float negative, float positive, uint8_t *packed, float *scales,
                        float *inverse, FlatMetrics &t) {
    auto &output = slots_[b % slot_count_].output;
    auto start = Clock::now();
    int rc = output.begin_cpu_read();
    t.sync += us(start);
    if (rc)
        return rc;
    rc = flat_quantize_native(static_cast<const uint16_t *>(output.mem.vaddr), shape_,
                              commands(b).groups, b * batch_, cfg_.M, cfg_.cpu_threads, negative,
                              positive, packed, scales, inverse, t);
    start = Clock::now();
    int end = output.end_cpu_access();
    t.sync += us(start);
    return rc ? rc : end;
}
void FlatPlan::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        cv_.wait(lock, [&] { return stop_ || pending_; });
        if (stop_)
            return;
        int b = block_;
        pending_ = false;
        lock.unlock();
        auto start = Clock::now();
        int rc = submit_job(b);
        double elapsed = us(start);
        lock.lock();
        result_ = rc;
        submit_us_ = elapsed;
        done_ = true;
        cv_.notify_all();
    }
}
void FlatPlan::launch(int b) {
    std::lock_guard<std::mutex> lock(mutex_);
    block_ = b;
    done_ = false;
    pending_ = true;
    cv_.notify_all();
}
int FlatPlan::wait(FlatMetrics &t) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return done_; });
    t.submit += submit_us_;
    return result_;
}
int FlatPlan::run(const FlatWeights &weights, const void *A, bool half, float negative,
                  float positive, uint8_t *packed, float *scales, float *inverse, FlatMetrics &t) {
    auto begin = Clock::now();
    int rc = bind(weights);
    if (rc)
        return rc;
    if (slot_count_ == 1) {
        for (int b = 0; b < blocks_ && !rc; ++b) {
            rc = pack_job(b, A, half, t);
            if (rc)
                break;
            auto start = Clock::now();
            rc = submit_job(b);
            t.submit += us(start);
            if (!rc)
                rc = quant_job(b, negative, positive, packed, scales, inverse, t);
        }
    } else {
        rc = pack_job(0, A, half, t);
        if (rc)
            return rc;
        launch(0);
        for (int b = 0; b < blocks_; ++b) {
            if (b + 1 < blocks_)
                rc = pack_job(b + 1, A, half, t);
            int submitted = wait(t);
            if (!rc)
                rc = submitted;
            if (rc)
                break;
            if (b + 1 < blocks_)
                launch(b + 1);
            rc = quant_job(b, negative, positive, packed, scales, inverse, t);
            if (rc) {
                if (b + 1 < blocks_)
                    wait(t);
                break;
            }
        }
    }
    t.total = us(begin);
    return rc;
}
} // namespace rknpu2_matmul_open::detail

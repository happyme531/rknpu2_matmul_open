#include "rk_npu_kn_plan.h"
#include "rk_npu_cpu_kernels.h"
#include "rk_npu_dcomp.h"

#include <algorithm>
#include <cstring>

namespace rknpu2_matmul_open::detail {

namespace {

rk_npu_matmul_i8_config make_wave_config(const I8KnPlanConfig& cfg, int kt) {
    rk_npu_matmul_i8_config mm{};
    rk_npu_matmul_i8_config_init(&mm, cfg.M, cfg.N, kt);
    mm.out_dtype = RK_NPU_I8_OUT_INT32;
    mm.a_layout = cfg.a_layout;
    mm.c_layout = cfg.c_layout;
    return mm;
}

rk_npu_matmul_i8_config make_weight_wave_config(int N, int kt) {
    rk_npu_matmul_i8_config mm{};
    rk_npu_matmul_i8_config_init(&mm, 1, N, kt);
    mm.out_dtype = RK_NPU_I8_OUT_INT32;
    mm.a_layout = RK_NPU_I8_A_LAYOUT_NORMAL;
    mm.c_layout = RK_NPU_I8_C_LAYOUT_NATIVE_N4_M4;
    return mm;
}

void make_b_slice(int N, int k0, int kt, const int8_t* B,
                  std::vector<int8_t>& slice) {
    slice.resize((size_t)kt * N);
    for (int k = 0; k < kt; ++k) {
        std::memcpy(slice.data() + (size_t)k * N,
                    B + (size_t)(k0 + k) * N, (size_t)N);
    }
}

} /* namespace */

int query_i8_kn_memory(const I8KnPlanConfig& config, I8KnMemoryInfo* out) {
    if (!out || config.M <= 0 || config.N <= 0 || config.K <= 0 ||
        config.k_tile <= 0 || config.k_tile > config.K || config.n_tile <= 0 ||
        config.npu_core_mask == 0 || (config.npu_core_mask & ~7u) != 0 ||
        config.npu_core_mask == 5u || config.npu_core_mask == 6u ||
        config.timeout_ms == 0)
        return RK_NPU_ERR_PARAM;

    const int align_n = std::max(MIN_CHANNEL_TILE,
                                 align_up(config.N, MIN_CHANNEL_TILE));
    const int n_tile = std::min(align_n,
                                align_up(config.n_tile, MIN_CHANNEL_TILE));
    if (ceil_div(align_n, n_tile) < __builtin_popcount(config.npu_core_mask))
        return RK_NPU_ERR_PARAM;

    I8KnMemoryInfo info{};
    uint64_t max_input_bytes = 0;
    uint64_t max_output_bytes = 0;
    for (int offset = 0; offset < config.K; offset += config.k_tile) {
        const int kt = std::min(config.k_tile, config.K - offset);
        const rk_npu_matmul_i8_config mm = make_wave_config(config, kt);
        rk_npu_matmul_sizes sizes{};
        const int rc = query_i8_n_tiled(&mm, n_tile, &sizes);
        if (rc != RK_NPU_OK) return rc;
        max_input_bytes = std::max(max_input_bytes, sizes.input_bytes);
        max_output_bytes = std::max(max_output_bytes, sizes.output_bytes);
        info.weight_bytes += sizes.weight_bytes;
        info.control_bytes += sizes.regcmd_bytes + sizes.task_bytes;
        ++info.wave_count;
        info.workspace_buffer_count += 2; /* regcmd + task for this wave */
    }
    const uint32_t input_slots = std::min<uint32_t>(info.wave_count, 2);
    const uint32_t output_slots = std::min<uint32_t>(info.wave_count, 3);
    info.input_bytes = max_input_bytes * input_slots;
    info.output_bytes = max_output_bytes * output_slots;
    info.workspace_buffer_count += input_slots + output_slots;
    info.weight_buffer_count = 1; /* one packed-B arena per matrix */
    *out = info;
    return RK_NPU_OK;
}

int query_i8_kn_weight_memory(const I8KnWeightConfig& config,
                              I8KnMemoryInfo* out) {
    if (!out || config.K <= 0 || config.N <= 0 || config.k_tile <= 0 ||
        config.k_tile > config.K)
        return RK_NPU_ERR_PARAM;
    I8KnMemoryInfo info{};
    for (int k0 = 0; k0 < config.K; k0 += config.k_tile) {
        const int kt = std::min(config.k_tile, config.K - k0);
        const rk_npu_matmul_i8_config mm =
            make_weight_wave_config(config.N, kt);
        rk_npu_matmul_sizes sizes{};
        const int rc = query_i8_n_tiled(&mm, config.N, &sizes);
        if (rc != RK_NPU_OK) return rc;
        info.weight_bytes += sizes.weight_bytes;
        ++info.wave_count;
    }
    info.weight_buffer_count = 1;
    *out = info;
    return RK_NPU_OK;
}

I8KnPlan::~I8KnPlan() {
    release();
}

int I8KnPlan::prepare(rk_npu_iommu_domain* domain,
                      const I8KnPlanConfig& config) {
    if (!domain || !domain->ctx || ctx_ || config.M <= 0 || config.N <= 0 ||
        config.K <= 0 || config.k_tile <= 0 || config.k_tile > config.K ||
        config.n_tile <= 0 || config.npu_core_mask == 0 ||
        (config.npu_core_mask & ~7u) != 0 || config.npu_core_mask == 5u ||
        config.npu_core_mask == 6u ||
        config.timeout_ms == 0)
        return RK_NPU_ERR_PARAM;

    const int align_n = std::max(MIN_CHANNEL_TILE,
                                 align_up(config.N, MIN_CHANNEL_TILE));
    const int n_tile = std::min(align_n,
                                align_up(config.n_tile, MIN_CHANNEL_TILE));
    const int n_cores = __builtin_popcount(config.npu_core_mask);
    if (ceil_div(align_n, n_tile) < n_cores)
        return RK_NPU_ERR_PARAM;

    ctx_ = domain->ctx;
    domain_ = domain;
    retain_domain(domain_);
    config_ = config;
    config_.n_tile = n_tile;
    const int count = ceil_div(config.K, config.k_tile);
    waves_.resize((size_t)count);
    a_scale_.resize((size_t)config_.M, 1.0f);
    int rc = RK_NPU_OK;
    uint64_t max_input_bytes = 0;
    uint64_t max_output_bytes = 0;
    for (int w = 0, offset = 0; rc == RK_NPU_OK && w < count; ++w) {
        Wave& wave = waves_[(size_t)w];
        wave.k0 = offset;
        wave.kt = std::min(config.k_tile, config.K - offset);
        wave.mm_cfg = make_wave_config(config_, wave.kt);
        rc = query_i8_n_tiled(&wave.mm_cfg, config_.n_tile, &wave.sizes);
        if (rc != RK_NPU_OK) break;
        max_input_bytes = std::max(max_input_bytes, wave.sizes.input_bytes);
        max_output_bytes = std::max(max_output_bytes, wave.sizes.output_bytes);
        offset += wave.kt;
    }
    input_slot_count_ = std::min(count, 2);
    output_slot_count_ = std::min(count, 3);
    for (int slot = 0; rc == RK_NPU_OK && slot < input_slot_count_; ++slot)
        rc = input_slot_[slot].alloc(domain_, max_input_bytes);
    for (int slot = 0; rc == RK_NPU_OK && slot < output_slot_count_; ++slot)
        rc = output_slot_[slot].alloc(domain_, max_output_bytes);
    for (int w = 0; rc == RK_NPU_OK && w < count; ++w) {
        Wave& wave = waves_[(size_t)w];
        wave.input = input_slot_[input_slot_for_wave(w)].mem;
        wave.output = output_slot_[output_slot_for_wave(w)].mem;
        wave.plan = prepare_i8_n_tiled(domain_, &wave.mm_cfg, config_.n_tile);
        if (!wave.plan) {
            rc = RK_NPU_ERR_NOMEM;
            break;
        }
        rc = bind_i8_core_mask_n_split_io(
            wave.plan, &wave.input, &wave.output, config_.npu_core_mask);
    }
    if (rc == RK_NPU_OK) wave_packed_.assign((size_t)count, 0);
    if (rc != RK_NPU_OK) release();
    return rc;
}

int I8KnPlan::pack_wave_i8_impl(Wave &wave, const int8_t *A_rowmajor) {
    rk_npu_mem &input = wave.input;
    const int align_in = std::max(MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
    int8_t *dst = (int8_t *)input.vaddr;
    std::memset(dst, 0, (size_t)config_.M * align_in);
    if (config_.a_layout == RK_NPU_I8_A_LAYOUT_NORMAL) {
        for (int m = 0; m < config_.M; ++m) {
            std::memcpy(dst + (size_t)m * align_in, A_rowmajor + (size_t)m * config_.K + wave.k0,
                        (size_t)wave.kt);
        }
    } else {
        constexpr int sub_k = 16;
        const int kb_count = align_in / sub_k;
        const int rows_block = a_panel_width() ? a_panel_width() : config_.M;
        // Keep a complete panel's stores adjacent instead of scattering each
        // K16 block across all M panels. Global native keeps its old traversal.
        for (int r0 = 0; r0 < config_.M; r0 += rows_block) {
            for (int kb = 0; kb < kb_count; ++kb) {
                const int local_k = kb * sub_k;
                const int valid = std::min(sub_k, std::max(0, wave.kt - local_k));
                if (valid <= 0)
                    continue;
                for (int m = r0; m < r0 + rows_block; ++m) {
                    std::memcpy(
                        dst + rknpu2_matmul_open::cpu::i8_a_block_offset(config_.M, align_in, m, kb, a_panel_width()),
                        A_rowmajor + (size_t)m * config_.K + wave.k0 + local_k, (size_t)valid);
                }
            }
        }
    }
    return RK_NPU_OK;
}

int I8KnPlan::prepare_f16_dynamic(const uint16_t* A_fp16_rowmajor) {
    if (!ctx_ || !A_fp16_rowmajor)
        return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::cpu::i8_compute_dynamic_per_token_scale_f16(
        config_.M, config_.K, A_fp16_rowmajor, config_.K,
        a_scale_.data());
    return RK_NPU_OK;
}

int I8KnPlan::prepare_and_pack_wave0_f16_dynamic(
    const uint16_t* A_fp16_rowmajor) {
    if (!ctx_ || !A_fp16_rowmajor || waves_.empty())
        return RK_NPU_ERR_PARAM;

    Wave& wave = waves_[0];
    const int slot = input_slot_for_wave(0);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        const int slice_k0[] = {wave.k0};
        const int slice_k[] = {wave.kt};
        const int slice_align_in[] = {align_in};
        int8_t* dst_slices[] = {
            static_cast<int8_t*>(wave.input.vaddr)};
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_native_split(
                config_.M, config_.K, 1, slice_k0, slice_k, slice_align_in,
                A_fp16_rowmajor, a_scale_.data(), dst_slices, a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_dynamic_normal_split(
                config_.M, config_.K, 1, slice_k0, slice_k, slice_align_in,
                A_fp16_rowmajor, a_scale_.data(), dst_slices);
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[0] = 1;
    return rc;
}

int I8KnPlan::pack_wave_i8(int wave_index,
                           const int8_t* A_rowmajor) {
    if (!ctx_ || !A_rowmajor || wave_index < 0 ||
        wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    const int slot = input_slot_for_wave(wave_index);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK)
        rc = pack_wave_i8_impl(waves_[(size_t)wave_index], A_rowmajor);
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[(size_t)wave_index] = 1;
    return rc;
}

int I8KnPlan::pack_wave_f16_dynamic(
    int wave_index, const uint16_t* A_fp16_rowmajor) {
    if (!ctx_ || !A_fp16_rowmajor ||
        wave_index < 0 || wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    Wave& wave = waves_[(size_t)wave_index];
    const int slot = input_slot_for_wave(wave_index);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_strided(
                config_.M, wave.kt, align_in, A_fp16_rowmajor + wave.k0,
                config_.K, a_scale_.data(),
                static_cast<int8_t*>(wave.input.vaddr), a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_normal_strided(
                config_.M, wave.kt, align_in, A_fp16_rowmajor + wave.k0,
                config_.K, a_scale_.data(),
                static_cast<int8_t*>(wave.input.vaddr));
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[(size_t)wave_index] = 1;
    return rc;
}

int I8KnPlan::pack_wave_f16_static(
    int wave_index, const uint16_t* A_fp16_rowmajor,
    const float* per_token_scale) {
    if (!ctx_ || !A_fp16_rowmajor || !per_token_scale ||
        wave_index < 0 || wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    Wave& wave = waves_[(size_t)wave_index];
    const int slot = input_slot_for_wave(wave_index);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_native_strided(
                config_.M, wave.kt, align_in, A_fp16_rowmajor + wave.k0,
                config_.K, per_token_scale,
                static_cast<int8_t*>(wave.input.vaddr), a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f16_quant_normal_strided(
                config_.M, wave.kt, align_in, A_fp16_rowmajor + wave.k0,
                config_.K, per_token_scale,
                static_cast<int8_t*>(wave.input.vaddr));
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[(size_t)wave_index] = 1;
    return rc;
}

int I8KnPlan::prepare_f32_dynamic(const float* A_fp32_rowmajor) {
    if (!ctx_ || !A_fp32_rowmajor)
        return RK_NPU_ERR_PARAM;
    rknpu2_matmul_open::cpu::i8_compute_dynamic_per_token_scale_f32(
        config_.M, config_.K, A_fp32_rowmajor, config_.K, a_scale_.data());
    return RK_NPU_OK;
}

int I8KnPlan::prepare_and_pack_wave0_f32_dynamic(
    const float* A_fp32_rowmajor) {
    if (!ctx_ || !A_fp32_rowmajor || waves_.empty())
        return RK_NPU_ERR_PARAM;
    Wave& wave = waves_[0];
    const int slot = input_slot_for_wave(0);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        const int slice_k0[] = {wave.k0};
        const int slice_k[] = {wave.kt};
        const int slice_align_in[] = {align_in};
        int8_t* dst_slices[] = {static_cast<int8_t*>(wave.input.vaddr)};
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_native_split(
                config_.M, config_.K, 1, slice_k0, slice_k, slice_align_in,
                A_fp32_rowmajor, a_scale_.data(), dst_slices, a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_dynamic_normal_split(
                config_.M, config_.K, 1, slice_k0, slice_k, slice_align_in,
                A_fp32_rowmajor, a_scale_.data(), dst_slices);
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[0] = 1;
    return rc;
}

int I8KnPlan::pack_wave_f32_dynamic(
    int wave_index, const float* A_fp32_rowmajor) {
    if (!ctx_ || !A_fp32_rowmajor || wave_index < 0 ||
        wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    Wave& wave = waves_[(size_t)wave_index];
    const int slot = input_slot_for_wave(wave_index);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_native_strided(
                config_.M, wave.kt, align_in, A_fp32_rowmajor + wave.k0,
                config_.K, a_scale_.data(),
                static_cast<int8_t*>(wave.input.vaddr), a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_normal_strided(
                config_.M, wave.kt, align_in, A_fp32_rowmajor + wave.k0,
                config_.K, a_scale_.data(),
                static_cast<int8_t*>(wave.input.vaddr));
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[(size_t)wave_index] = 1;
    return rc;
}

int I8KnPlan::pack_wave_f32_static(
    int wave_index, const float* A_fp32_rowmajor,
    const float* per_token_scale) {
    if (!ctx_ || !A_fp32_rowmajor || !per_token_scale ||
        wave_index < 0 || wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    Wave& wave = waves_[(size_t)wave_index];
    const int slot = input_slot_for_wave(wave_index);
    int rc = input_slot_[slot].begin_cpu_write();
    if (rc == RK_NPU_OK) {
        const int align_in = std::max(
            MIN_CHANNEL_TILE, align_up(wave.kt, MIN_CHANNEL_TILE));
        if (config_.a_layout != RK_NPU_I8_A_LAYOUT_NORMAL) {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_native_strided(
                config_.M, wave.kt, align_in, A_fp32_rowmajor + wave.k0,
                config_.K, per_token_scale,
                static_cast<int8_t*>(wave.input.vaddr), a_panel_width());
        } else {
            rknpu2_matmul_open::cpu::i8_pack_a_f32_quant_normal_strided(
                config_.M, wave.kt, align_in, A_fp32_rowmajor + wave.k0,
                config_.K, per_token_scale,
                static_cast<int8_t*>(wave.input.vaddr));
        }
    }
    const int end_rc = input_slot_[slot].end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc == RK_NPU_OK) wave_packed_[(size_t)wave_index] = 1;
    return rc;
}

int I8KnPlan::run_wave(int wave_index, const I8KnWeights& weights) {
    if (!ctx_ || wave_index < 0 || wave_index >= (int)waves_.size() ||
        !wave_packed_[(size_t)wave_index] ||
        !weights.compatible_wave(*this, wave_index))
        return RK_NPU_ERR_PARAM;
    const int slot = output_slot_for_wave(wave_index);
    int rc = input_slot_[input_slot_for_wave(wave_index)].end_cpu_access();
    if (rc == RK_NPU_OK) rc = output_slot_[slot].end_cpu_access();
    if (rc != RK_NPU_OK) return rc;
    Wave& wave = waves_[(size_t)wave_index];
    if (weights.compressed())
        return weights.run_compressed(wave_index, config_.n_tile, wave.plan,
                                       config_.timeout_ms);
    const rk_npu_mem* weight = weights.wave(wave_index);
    if (!weight) return RK_NPU_ERR_PARAM;
    return run_i8_core_mask_n_split_prebound(
        ctx_, wave.plan, const_cast<rk_npu_mem*>(weight), config_.timeout_ms);
}

int I8KnPlan::output_slot_for_wave(int wave) const {
    if (wave <= 0) return 0;
    return 1 + ((wave - 1) & 1);
}

int I8KnPlan::input_slot_for_wave(int wave) const {
    return wave & 1;
}

int I8KnPlan::begin_wave_output_cpu_read(int wave) {
    if (!ctx_ || wave < 0 || wave >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    return output_slot_[output_slot_for_wave(wave)].begin_cpu_read();
}

int I8KnPlan::begin_wave_output_cpu_readwrite(int wave) {
    if (!ctx_ || wave < 0 || wave >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    return output_slot_[output_slot_for_wave(wave)].begin_cpu_readwrite();
}

int I8KnPlan::end_wave_output_cpu_access(int wave) {
    if (!ctx_ || wave < 0 || wave >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    return output_slot_[output_slot_for_wave(wave)].end_cpu_access();
}

int I8KnPlan::unpack_partial_i32(int wave_index,
                                 std::vector<int32_t>& dst) const {
    if (!ctx_ || wave_index < 0 || wave_index >= (int)waves_.size())
        return RK_NPU_ERR_PARAM;
    const Wave& wave = waves_[(size_t)wave_index];
    const DomainDataBuffer::CpuAccess access =
        output_slot_[output_slot_for_wave(wave_index)].cpu_access;
    if (access != DomainDataBuffer::CpuAccess::Read &&
        access != DomainDataBuffer::CpuAccess::ReadWrite)
        return RK_NPU_ERR_PARAM;
    dst.assign((size_t)config_.M * config_.N, 0);
    return unpack_i8_c(
        &wave.mm_cfg, &wave.output,
        reinterpret_cast<float*>(dst.data()));
}

int I8KnPlan::release() {
    if (!ctx_) return RK_NPU_OK;
    int rc = RK_NPU_OK;
    for (auto it = waves_.rbegin(); it != waves_.rend(); ++it) {
        Wave& wave = *it;
        if (wave.plan) free_i8_plan(wave.plan);
        wave.plan = nullptr;
    }
    waves_.clear();
    for (int slot = output_slot_count_ - 1; slot >= 0; --slot) {
        const int r = output_slot_[slot].release(ctx_);
        if (rc == RK_NPU_OK && r != RK_NPU_OK) rc = r;
    }
    for (int slot = input_slot_count_ - 1; slot >= 0; --slot) {
        const int r = input_slot_[slot].release(ctx_);
        if (rc == RK_NPU_OK && r != RK_NPU_OK) rc = r;
    }
    a_scale_.clear();
    wave_packed_.clear();
    ctx_ = nullptr;
    release_domain(domain_);
    domain_ = nullptr;
    config_ = I8KnPlanConfig{};
    input_slot_count_ = 0;
    output_slot_count_ = 0;
    return rc;
}

int I8KnPlan::k0(int wave) const {
    return wave >= 0 && wave < (int)waves_.size() ? waves_[(size_t)wave].k0 : -1;
}

int I8KnPlan::kt(int wave) const {
    return wave >= 0 && wave < (int)waves_.size() ? waves_[(size_t)wave].kt : -1;
}

const int32_t* I8KnPlan::partial_i32(int wave) const {
    if (wave < 0 || wave >= (int)waves_.size()) return nullptr;
    return (const int32_t*)waves_[(size_t)wave].output.vaddr;
}

const float* I8KnPlan::a_scale() const {
    return !a_scale_.empty() ? a_scale_.data() : nullptr;
}

I8KnWeights::~I8KnWeights() {
    release();
}

int I8KnWeights::prepare(rk_npu_iommu_domain* domain,
                         const I8KnWeightConfig& config,
                         const int8_t* B_rowmajor) {
    if (!domain || !domain->ctx || !B_rowmajor || ctx_ || config.K <= 0 || config.N <= 0 ||
        config.k_tile <= 0 || config.k_tile > config.K)
        return RK_NPU_ERR_PARAM;

    ctx_ = domain->ctx;
    domain_ = domain;
    retain_domain(domain_);
    config_ = config;
    I8KnMemoryInfo memory{};
    int rc = query_i8_kn_weight_memory(config_, &memory);
    if (rc != RK_NPU_OK) {
        release();
        return rc;
    }
    waves_.resize((size_t)memory.wave_count);
    rc = arena_.alloc(domain_, memory.weight_bytes);
    uint64_t weight_offset = 0;
    for (size_t w = 0; rc == RK_NPU_OK && w < waves_.size(); ++w) {
        const int k0 = (int)w * config_.k_tile;
        const int kt = std::min(config_.k_tile, config_.K - k0);
        const rk_npu_matmul_i8_config mm =
            make_weight_wave_config(config_.N, kt);
        rk_npu_matmul_sizes sizes{};
        rc = query_i8_n_tiled(&mm, config_.N, &sizes);
        const uint64_t bytes = sizes.weight_bytes;
        if (rc != RK_NPU_OK) break;
        rc = rk_npu_mem_view(&arena_.mem, weight_offset, bytes, &waves_[w]);
        weight_offset += bytes;
    }
    if (rc == RK_NPU_OK) rc = arena_.begin_cpu_write();
    std::vector<int8_t> b_slice;
    for (size_t w = 0; rc == RK_NPU_OK && w < waves_.size(); ++w) {
        const int k0 = (int)w * config_.k_tile;
        const int kt = std::min(config_.k_tile, config_.K - k0);
        const rk_npu_matmul_i8_config mm =
            make_weight_wave_config(config_.N, kt);
        make_b_slice(config_.N, k0, kt,
                     B_rowmajor, b_slice);
        rc = pack_i8_b(&mm, b_slice.data(), &waves_[w]);
    }
    const int end_rc = arena_.end_cpu_access();
    if (rc == RK_NPU_OK && end_rc != RK_NPU_OK) rc = end_rc;
    if (rc != RK_NPU_OK) release();
    return rc;
}

int I8KnWeights::release() {
    if (!ctx_) return RK_NPU_OK;
    compressed_layouts_.clear();
    compressed_ = false;
    const int rc = arena_.release(ctx_);
    waves_.clear();
    config_ = I8KnWeightConfig{};
    ctx_ = nullptr;
    release_domain(domain_);
    domain_ = nullptr;
    return rc;
}

bool I8KnWeights::compatible(const I8KnPlan& workspace) const {
    if (compressed_)
        return same_domain(workspace) && config_.K == workspace.config_.K &&
               config_.N == workspace.config_.N && config_.k_tile == workspace.config_.k_tile;
    if (!same_domain(workspace) ||
        config_.K != workspace.config_.K ||
        config_.N != workspace.config_.N ||
        config_.k_tile != workspace.config_.k_tile ||
        waves_.size() != workspace.waves_.size())
        return false;
    for (size_t w = 0; w < waves_.size(); ++w) {
        const I8KnPlan::Wave& wave = workspace.waves_[w];
        if (wave.k0 != (int)w * config_.k_tile ||
            wave.kt != std::min(config_.k_tile, config_.K - wave.k0) ||
            waves_[w].size < wave.sizes.weight_bytes)
            return false;
    }
    return true;
}

bool I8KnWeights::compatible_wave(
    const I8KnPlan& workspace, int wave_index) const {
    if (compressed_)
        return compatible(workspace) && wave_index >= 0 && wave_index < workspace.wave_count();
    if (!same_domain(workspace) ||
        config_.K != workspace.config_.K ||
        config_.N != workspace.config_.N ||
        config_.k_tile != workspace.config_.k_tile ||
        waves_.size() != workspace.waves_.size() || wave_index < 0 ||
        wave_index >= (int)waves_.size())
        return false;
    const I8KnPlan::Wave& wave = workspace.waves_[(size_t)wave_index];
    return wave.k0 == wave_index * config_.k_tile &&
           wave.kt == std::min(config_.k_tile, config_.K - wave.k0) &&
           waves_[(size_t)wave_index].size >= wave.sizes.weight_bytes;
}

bool I8KnWeights::same_domain(const I8KnPlan& workspace) const {
    return ctx_ && ctx_ == workspace.ctx_ && domain_ && workspace.domain_ &&
           domain_->id == workspace.domain_->id;
}

const rk_npu_mem* I8KnWeights::wave(int index) const {
    return index >= 0 && index < (int)waves_.size()
         ? &waves_[(size_t)index] : nullptr;
}

int I8KnWeights::build_compressed_layout(int n_tile, const int8_t* B,
    const CompressedLayout* source, std::shared_ptr<CompressedLayout>& result) const {
    auto layout = std::make_shared<CompressedLayout>();
    layout->n_tile = n_tile;
    std::vector<uint8_t> payload;
    const int an = align_up(config_.N, 32);
    for (int k0 = 0, wi = 0; k0 < config_.K; ++wi) {
        const int kt = std::min(config_.k_tile, config_.K - k0), ak = align_up(kt, 32);
        std::vector<int8_t> packed;
        if (B) {
            packed.resize(size_t(ak) * an);
            rknpu2_matmul_open::cpu::i8_pack_b_native_n32_k32(kt, config_.N, ak, an,
                B + size_t(k0) * config_.N, packed.data());
        } else {
            if (!source || size_t(wi) >= source->waves.size()) return RK_NPU_ERR_PARAM;
            packed.reserve(size_t(ak) * an);
            int n0 = 0;
            for (const auto& tile : source->waves[wi]) {
                std::array<uint32_t, 16> amounts;
                std::copy(tile.amounts, tile.amounts + 16, amounts.begin());
                std::vector<int8_t> part;
                const int nt = std::min(source->n_tile, an - n0);
                const auto* data = static_cast<const uint8_t*>(source->arena.mem.vaddr) + tile.offset;
                if (tile.compressed) {
                    const int rc = dcomp_decode(data, tile.bytes, amounts, ak, nt, part);
                    if (rc != RK_NPU_OK) return rc;
                } else {
                    if (tile.bytes != size_t(ak) * nt) return RK_NPU_ERR_PARAM;
                    part.assign(reinterpret_cast<const int8_t*>(data),
                                reinterpret_cast<const int8_t*>(data) + tile.bytes);
                }
                packed.insert(packed.end(), part.begin(), part.end()); n0 += nt;
            }
            if (packed.size() != size_t(ak) * an) return RK_NPU_ERR_PARAM;
        }
        layout->waves.emplace_back();
        for (int n0 = 0; n0 < an; n0 += n_tile) {
            const int nt = std::min(n_tile, an - n0);
            DcompEncoded encoded;
            const int rc = dcomp_encode(packed.data() + size_t(n0) * ak, ak, nt, encoded);
            if (rc != RK_NPU_OK) return rc;
            I8CompressedTile tile{};
            tile.offset = payload.size();
            // Do not expand high-entropy or tiny tiles. In particular a 32x32
            // normal stream has a 2048-byte minimum, exceeding its raw tile.
            tile.compressed = encoded.data.size() < size_t(ak) * nt;
            tile.bytes = tile.compressed ? encoded.data.size() : size_t(ak) * nt;
            std::copy(encoded.amounts.begin(), encoded.amounts.end(), tile.amounts);
            layout->waves.back().push_back(tile);
            if (tile.compressed)
                payload.insert(payload.end(), encoded.data.begin(), encoded.data.end());
            else {
                const auto* raw = reinterpret_cast<const uint8_t*>(packed.data()) + size_t(n0) * ak;
                payload.insert(payload.end(), raw, raw + tile.bytes);
            }
        }
        k0 += kt;
    }
    int rc = layout->arena.alloc(domain_, payload.size());
    if (rc == RK_NPU_OK) rc = layout->arena.begin_cpu_write();
    if (rc == RK_NPU_OK)
        std::memcpy(layout->arena.mem.vaddr, payload.data(), payload.size());
    const int end_rc = layout->arena.end_cpu_access();
    if (rc == RK_NPU_OK) rc = end_rc;
    if (rc != RK_NPU_OK) return rc;
    result = std::move(layout);
    return RK_NPU_OK;
}

int I8KnWeights::prepare_compress(rk_npu_iommu_domain* domain,
    const I8KnWeightConfig& config, const int8_t* B) {
    if (!domain || !domain->ctx || !B || ctx_ || config.K <= 0 || config.N <= 0 ||
        config.K > INT32_MAX - 32 || config.N > INT32_MAX - 32 ||
        config.k_tile <= 0 || config.k_tile > config.K) return RK_NPU_ERR_PARAM;
    ctx_ = domain->ctx; domain_ = domain; retain_domain(domain_);
    config_ = config; compressed_ = true;
    std::shared_ptr<CompressedLayout> layout;
    const int nt = align_up(config.N, 32);
    int rc = build_compressed_layout(nt, B, nullptr, layout);
    if (rc == RK_NPU_OK) compressed_layouts_.emplace(nt, std::move(layout));
    else release();
    return rc;
}

int I8KnWeights::run_compressed(int wi, int nt, rk_npu_matmul_i8_plan* plan,
                                uint32_t timeout_ms) const {
    std::shared_ptr<CompressedLayout> layout;
    try {
        std::lock_guard<std::mutex> lock(compressed_mutex_);
        auto it = compressed_layouts_.find(nt);
        if (it != compressed_layouts_.end()) layout = it->second;
        else {
            if (compressed_layouts_.empty()) return RK_NPU_ERR_PARAM;
            auto source = compressed_layouts_.begin()->second;
            int rc = source->arena.begin_cpu_read();
            if (rc == RK_NPU_OK) {
                try { rc = build_compressed_layout(nt, nullptr, source.get(), layout); }
                catch (...) { source->arena.end_cpu_access(); throw; }
            }
            const int end_rc = source->arena.end_cpu_access();
            if (rc == RK_NPU_OK) rc = end_rc;
            if (rc != RK_NPU_OK) return rc;
            compressed_layouts_.emplace(nt, layout);
            // The creation-time full-N stream is only a seed. Drop unused seeds
            // after first binding, avoiding a permanent second full weight copy.
            for (auto p = compressed_layouts_.begin(); p != compressed_layouts_.end();)
                if (p->first != nt && !p->second->used) p = compressed_layouts_.erase(p);
                else ++p;
        }
        layout->used = true;
    } catch (const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
    if (wi < 0 || size_t(wi) >= layout->waves.size()) return RK_NPU_ERR_PARAM;
    return run_i8_compressed_prebound(ctx_, plan, &layout->arena.mem,
                                      layout->waves[wi], timeout_ms);
}

uint64_t I8KnWeights::stored_bytes() const {
    if (!compressed_) return arena_.mem.size;
    std::lock_guard<std::mutex> lock(compressed_mutex_);
    uint64_t total = 0;
    for (const auto& item : compressed_layouts_)
        for (const auto& wave : item.second->waves)
            for (const auto& tile : wave) total += tile.bytes;
    return total;
}

} /* namespace rknpu2_matmul_open::detail */

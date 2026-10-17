#if defined(__aarch64__)
#include "rk_npu_gdn_cpu.h"
namespace rknpu2_matmul_open::gdn {
template<bool split, int Chunk>
class Prefill final : public Engine {
    rk_npu_iommu_domain* domain;
    rk_npu_gdn_config cfg;
    static constexpr int C = Chunk, N = (D + C + 31) / 32 * 32;
    int T, blocks;
    Workers & workers;
    Buffer ga, gal, gc, gx;
    Buffer state, state_hi, state_lo, identity;
    Buffer b1, b1l, p1, p1x, a2, a2l, b2, b2l, p2, p2x, gate_hi, gate_lo;
    Chain grams, update, update_add, update_single;
    GroupedChain grouped_update;
    std::vector<std::unique_ptr<Chain>> predictions;
    std::vector<float> lower, delta, prefix;
    // The native FP32 buffer holds Z with M = state_scale[h] * Z.
    // State export restores M while converting the native layout.
    std::vector<float> state_scale = std::vector<float>(HV, 1.f);
    std::vector<int> scale_update = std::vector<int>(HV, 0);
    bool add_only = false, all_scale = false, single_pass = false;

    static void store4(float * main, float * low, size_t at, float32x4_t value) {
        vst1q_f32(main + at, value);
        if (split) {
            auto high = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(value), vdupq_n_u32(0xffffe000)));
            vst1q_f32(low + at, vsubq_f32(value, high));
        }
    }
    void product(std::vector<std::vector<uint64_t>> & bodies, std::vector<int> & enables,
                 int m, int n, int k, uint64_t a, uint64_t al, uint64_t b, uint64_t bl,
                 uint64_t out, uint64_t scratch) {
        auto add = [&](std::vector<uint64_t> body, int enable) {
            bodies.push_back(std::move(body)); enables.push_back(enable);
        };
        add(gemm(m, n, k, a, b, out), 13);
        if (split) {
            add(gemm(m, n, k, a, bl, scratch), 13);
            add(dpu(m, n, out, scratch, out), 24);
            add(gemm(m, n, k, al, b, scratch), 13);
            add(dpu(m, n, out, scratch, out), 24);
        }
    }
    void allocate(Buffer & buffer, size_t floats) { buffer.init(domain, floats * 4); }
    void allocate_pair(Buffer & main, Buffer & low, size_t floats) {
        allocate(main, floats);
        if (split) allocate(low, floats);
    }
    void pack_grams(const Inputs & x) {
        const float scale = 1.f / std::sqrt(float(D));
        workers.run([&](int w) {
            const int blocks = this->blocks, T = this->T;
            const float * const q_data = x.q, * const k_data = x.k;
            float * const a_data = ga.ptr(), * const a_low = gal.ptr();
            float * const b_data = b1.ptr(), * const b_low = b1l.ptr();
            for (int job = w; job < blocks * HK; job += workers.count()) {
                const int block = job / HK, head = job % HK, base = block * C;
                const int valid = std::min(C, T - base);
                const size_t a = size_t(job) * C * D, b = size_t(job) * D * 2 * C;
                for (int t0 = 0; t0 < valid; t0 += 16) for (int k0 = 0; k0 < D; k0 += 16)
                    for (int t = t0; t < std::min(t0 + 16, valid); t += 4)
                        for (int k = k0; k < k0 + 16; k += 4)
                            for (int lane = 0; lane < 4 && t + lane < valid; ++lane) {
                                const size_t at = (size_t(base + t + lane) * HK + head) * D + k;
                                const auto q = vmulq_n_f32(vld1q_f32(q_data + at), scale);
                                const auto key = vld1q_f32(k_data + at);
                                store4(a_data, a_low, a + native_a(t + lane, k, C), key);
                                store4(b_data, b_low, b + bi(k, t + lane, D), key);
                                store4(b_data, b_low, b + bi(k, C + t + lane, D), q);
                            }
            }
        });
    }
    template<bool native_keys>
    void coefficients(const Inputs & x, int block) {
        const int base = block * C, valid = std::min(C, T - base);
        workers.run([&](int w) {
            const float * const k_data = x.k;
            const float * const g_data = x.g, * const beta_data = x.beta;
            const float * const gram_data = gc.ptr();
            float * const lower_data = lower.data();
            float * const prefix_data = prefix.data();
            float * const b2_data = b2.ptr(), * const b2_low = b2l.ptr();
            uint16_t * const gates_high = reinterpret_cast<uint16_t *>(gate_hi.ptr());
            uint16_t * const gates_low = reinterpret_cast<uint16_t *>(gate_lo.ptr());
            for (int h = w; h < HV; h += workers.count()) {
                const int kh = cfg.head_mapping == RK_NPU_GDN_HEAD_GROUPED ? h / 2 : h % HK;
                const float * gram = gram_data + size_t(block * HK + kh) * 2 * C * C;
                float * l = lower_data + size_t(h) * C * C;
                if (valid < C) {
                    std::fill_n(b2_data + size_t(h) * C * N, C * N, 0.f);
                    if (split) {
                        std::fill_n(b2_low + size_t(h) * C * N, C * N, 0.f);
                    }
                }
                float alpha[C]{}, suffix[C];
                const float old_scale = state_scale[h];
                float gamma = 1;
                for (int t = 0; t < valid; ++t) {
                    alpha[t] = std::exp(g_data[size_t(base + t) * HV + h]);
                    gamma *= alpha[t];
                    prefix_data[h * C + t] = gamma * old_scale;
                }
                float next_scale = gamma * old_scale, scale_factor = 1.f;
                // Exact power-of-two rebasing keeps inverse weights bounded by 65536.
                if (next_scale < 0x1p-16f) {
                    if (next_scale > 0.f && next_scale * 0x1p14f >= 0x1p-16f) {
                        next_scale *= 0x1p14f;
                        scale_factor = 0x1p-14f;
                    } else {
                        scale_factor = next_scale;
                        next_scale = 1.f;
                    }
                }
                // Partial chunks export the ordinary state without inverse-scale amplification.
                if (valid < C && block + 1 == blocks) {
                    scale_factor = gamma * old_scale;
                    next_scale = 1.f;
                }
                state_scale[h] = next_scale;
                // Four query positions share a native Gram vector across key rows.
                for (int t0 = 0; t0 < valid; t0 += 4) {
                    auto factor = vdupq_n_f32(1.f);
                    const int32_t token_lanes[4]{t0, t0 + 1, t0 + 2, t0 + 3};
                    const auto token = vld1q_s32(token_lanes);
                    for (int j = t0; j >= 0; j -= 4) {
                        float32x4_t decay[4];
                        for (int lane = 3; lane >= 0; --lane) {
                            if (j == t0) {
                                const auto active = vcgeq_s32(token, vdupq_n_s32(j + lane));
                                decay[lane] = vbslq_f32(active, factor, vdupq_n_f32(0.f));
                                factor = vmulq_f32(factor, vbslq_f32(active,
                                    vdupq_n_f32(alpha[j + lane]), vdupq_n_f32(1.f)));
                            } else {
                                decay[lane] = factor;
                                factor = vmulq_n_f32(factor, alpha[j + lane]);
                            }
                        }
                        float32x4_t qk[4], kk[4];
                        for (int lane = 0; lane < 4; ++lane) {
                            qk[lane] = vmulq_f32(vld1q_f32(gram + native_a(j + lane, C + t0, C)), decay[lane]);
                            kk[lane] = vmulq_f32(vld1q_f32(gram + native_a(j + lane, t0, C)), decay[lane]);
                        }
                        transpose(qk); transpose(kk);
                        for (int row = 0; row < 4 && t0 + row < valid; ++row) {
                            const int t = t0 + row;
                            store4(b2_data, b2_low, size_t(h) * C * N + bi(j, D + t, C), qk[row]);
                            vst1q_f32(l + t * C + j, vmulq_n_f32(kk[row], beta_data[size_t(base + t) * HV + h]));
                        }
                    }
                    for (int row = 0; row < 4 && t0 + row < valid; ++row)
                        l[(t0 + row) * C + t0 + row] = 0;
                }
                float product = 1;
                for (int j = valid - 1; j >= 0; --j) { suffix[j] = product / next_scale; product *= alpha[j]; }
                for (int j0 = 0; j0 < valid; j0 += 16) for (int k0 = 0; k0 < D; k0 += 16)
                    for (int j = j0; j < std::min(j0 + 16, valid); j += 4) for (int k = k0; k < k0 + 16; k += 4) {
                    float32x4_t rows[4];
                    for (int lane = 0; lane < 4; ++lane) {
                        const size_t src = native_keys
                            ? size_t(block * HK + kh) * C * D + native_a(j + lane, k, C)
                            : (size_t(base + j + lane) * HK + kh) * D + k;
                        const float * keys = native_keys ? ga.ptr() : k_data;
                        rows[lane] = j + lane < valid ? vmulq_n_f32(vld1q_f32(keys + src), suffix[j + lane]) : vdupq_n_f32(0);
                    }
                    transpose(rows);
                    for (int lane = 0; lane < 4; ++lane)
                        store4(b2_data, b2_low, size_t(h) * C * N + bi(j, k + lane, C), rows[lane]);
                }
                const uint16_t high = hb::float_to_half(scale_factor);
                scale_update[h] = scale_factor == 1.f ? 0 : hb::half_to_float(high) == scale_factor ? 1 : 2;
                const uint16_t low = hb::float_to_half((scale_factor - hb::half_to_float(high)) * 0x1p16f);
                auto * high_out = gates_high + h * D * 2;
                auto * low_out = gates_low + h * D * 2;
                for (int k = 0; k < D; ++k) {
                    high_out[k / 4 * 8 + k % 4] = high;
                    low_out[k / 4 * 8 + k % 4] = low;
                }
            }
        });
        add_only = std::none_of(scale_update.begin(), scale_update.end(), [](int v) { return v; });
        all_scale = std::all_of(scale_update.begin(), scale_update.end(), [](int v) { return v == 2; });
        single_pass = std::none_of(scale_update.begin(), scale_update.end(), [](int v) { return v == 2; });
    }
    void solve(const Inputs & x, int block) {
        const int base = block * C, valid = std::min(C, T - base);
        workers.run([&](int w) {
            const float * const raw_data = p1.ptr(), * const value_data = x.v;
            const float * const beta_data = x.beta, * const lower_data = lower.data();
            const float * const prefix_data = prefix.data();
            float * const delta_data = delta.data();
            float * const a_data = a2.ptr(), * const a_low = a2l.ptr();
            for (int h = w; h < HV; h += workers.count()) {
                const float * raw = raw_data + size_t(h) * D * 2 * C;
                float * d = delta_data + size_t(h) * C * D;
                const float * l = lower_data + size_t(h) * C * C;
                if (valid < C) std::fill_n(d, C * D, 0.f);
                for (int t = 0; t < valid; t += 4) for (int v = 0; v < D; v += 4) {
                    auto key_tile = vld4q_f32(raw + native_a(v, t, D));
                    for (int j = 0; j < 4 && t + j < valid; ++j) {
                        const float beta = beta_data[size_t(base + t + j) * HV + h];
                        const float gate = prefix_data[h * C + t + j];
                        const float * value = value_data + (size_t(base + t + j) * HV + h) * D + v;
                        vst1q_f32(d + (size_t(v / 32) * C + t + j) * 32 + v % 32,
                            vmulq_n_f32(vsubq_f32(vld1q_f32(value), vmulq_n_f32(key_tile.val[j], gate)), beta));
                    }
                }
                // Each value strip keeps its solved rows contiguous in L1.
                for (int v = 0; v < D; v += 32) {
                    float * strip = d + size_t(v / 32) * C * 32;
                    for (int t = 0; t < valid; ++t) {
                        float32x4_t acc[8];
                        for (int lane = 0; lane < 8; ++lane)
                            acc[lane] = vld1q_f32(strip + t * 32 + lane * 4);
                        for (int j = 0; j < t; ++j) for (int lane = 0; lane < 8; ++lane)
                            acc[lane] = vfmsq_n_f32(acc[lane], vld1q_f32(strip + j * 32 + lane * 4), l[t * C + j]);
                        for (int lane = 0; lane < 8; ++lane) vst1q_f32(strip + t * 32 + lane * 4, acc[lane]);
                    }
                }
                for (int t = 0; t < C; t += 4) for (int v = 0; v < D; v += 4) {
                    float32x4_t rows[4];
                    for (int lane = 0; lane < 4; ++lane)
                        rows[lane] = vld1q_f32(d + (size_t(v / 32) * C + t + lane) * 32 + v % 32);
                    transpose(rows);
                    for (int lane = 0; lane < 4; ++lane)
                        store4(a_data, a_low, size_t(h) * C * D + native_a(v + lane, t, D), rows[lane]);
                }
            }
        });
    }
    void collect(int block, float* output) {
        const int base = block * C, valid = std::min(C, T - base);
        workers.run([&](int w) {
            const float * const raw_data = p2.ptr(), * const old_data = p1.ptr();
            const float * const prefix_data = prefix.data();
            float * const out = output;
            for (int h = w; h < HV; h += workers.count()) {
                const float * raw = raw_data + HV * SS + size_t(h) * D * (N - D);
                const float * old = old_data + size_t(h) * D * 2 * C;
                for (int t = 0; t < valid; t += 4) for (int v = 0; v < D; v += 4) {
                    auto tile = vld4q_f32(raw + native_a(v, t, D));
                    auto old_tile = vld4q_f32(old + native_a(v, C + t, D));
                    for (int j = 0; j < 4 && t + j < valid; ++j)
                        vst1q_f32(out + (size_t(base + t + j) * HV + h) * D + v,
                            vaddq_f32(tile.val[j], vmulq_n_f32(old_tile.val[j], prefix_data[h * C + t + j])));
                }
            }
        });
    }
public:
    Prefill(rk_npu_iommu_domain* d,const rk_npu_gdn_config& config,Workers& pool)
        : domain(d),cfg(config),T(config.tokens),blocks((T+C-1)/C),
          workers(pool),lower(HV*C*C),delta(HV*C*D),prefix(HV*C) {
        const int cores=__builtin_popcount(cfg.core_mask);
        allocate_pair(ga, gal, size_t(blocks) * HK * C * D);
        allocate(gc, size_t(blocks) * HK * 2 * C * C);
        if (split) allocate(gx, size_t(blocks) * HK * 2 * C * C);
        allocate(state, HV * SS);
        allocate(state_hi, HV * SS);
        if (split) { allocate(state_lo, HV * SS); allocate(identity, SS); }
        allocate_pair(b1, b1l, size_t(blocks) * HK * D * 2 * C);
        allocate(p1, HV * D * 2 * C);
        if (split) allocate(p1x, HV * D * 2 * C);
        allocate_pair(a2, a2l, HV * C * D);
        allocate_pair(b2, b2l, HV * C * N);
        allocate(p2, HV * D * N);
        if (split) allocate(p2x, HV * D * N);
        allocate(gate_hi, HV * D);
        allocate(gate_lo, HV * D);
        if (split) {
            for (int k = 0; k < D; ++k) identity.ptr()[bi(k, k, D)] = 1.f;
            identity.sync(RK_NPU_SYNC_TO_DEVICE);
        }
        std::vector<std::vector<std::vector<uint64_t>>> g(3), u(3), ua(3), us(3);
        std::vector<std::vector<int>> ge(3), ue(3), uae(3), use(3);
        std::vector<std::vector<std::vector<uint64_t>>> grouped(3);
        std::vector<std::vector<int>> grouped_enables(3);
        std::vector<StateGroup> tail_groups[3];
        int front_count[3]{};
        for (int job = 0; job < blocks * HK; ++job) {
            const int core = job % cores;
            product(g[core], ge[core], C, 2 * C, D,
                ga.addr(size_t(job) * C * D), gal.addr(size_t(job) * C * D),
                b1.addr(size_t(job) * D * 2 * C), b1l.addr(size_t(job) * D * 2 * C),
                gc.addr(size_t(job) * 2 * C * C), gx.addr(size_t(job) * 2 * C * C));
        }
        for (int core = 0; core < cores; ++core) {
            const int first = core * (HV / cores) + std::min(core, HV % cores);
            const int count = HV / cores + (core < HV % cores);
            for (int h = first; h < first + count; ++h) {
                // Compact state increments allow one DPU pass per core.
                product(u[core], ue[core], D, D, C,
                    a2.addr(size_t(h) * D * C), a2l.addr(size_t(h) * D * C),
                    b2.addr(size_t(h) * C * N), b2l.addr(size_t(h) * C * N),
                    p2.addr(size_t(h) * SS), p2x.addr(size_t(h) * SS));
                product(u[core], ue[core], D, N - D, C,
                    a2.addr(size_t(h) * D * C), a2l.addr(size_t(h) * D * C),
                    b2.addr(size_t(h) * C * N + D * C), b2l.addr(size_t(h) * C * N + D * C),
                    p2.addr(HV * SS + size_t(h) * D * (N - D)), p2x.addr(HV * SS + size_t(h) * D * (N - D)));
                if (!split) {
                    // Both GEMMs consume the same complete native delta tile.
                    auto & body = u[core].back();
                    for (auto & word : body) if ((word & 65535) == R_CNA_CBUF_CON0)
                        word |= uint64_t(1u << 12) << 16;
                }
            }
            grouped[core] = u[core]; grouped_enables[core] = ue[core];
            front_count[core] = grouped[core].size();
            for (int h = first; h < first + count; h += STATE_GROUP) {
                const int size = std::min(STATE_GROUP, first + count - h);
                const int add_index = grouped[core].size();
                grouped[core].push_back(dpu(D, D * size, state.addr(h * SS), p2.addr(h * SS), state.addr(h * SS)));
                grouped[core].push_back(dpu(D, D * size, state.addr(h * SS), p2.addr(h * SS),
                    state.addr(h * SS), false, gate_hi.addr(h * D)));
                grouped[core].push_back(dpu(D, D * size, state.addr(h * SS), p2.addr(h * SS),
                    state_hi.addr(h * SS), false, gate_hi.addr(h * D)));
                grouped[core].push_back(dpu(D, D * size, state.addr(h * SS), state_hi.addr(h * SS),
                    state.addr(h * SS), false, gate_lo.addr(h * D), true));
                grouped_enables[core].insert(grouped_enables[core].end(), 4, 24);
                tail_groups[core].push_back({h, size, add_index, add_index + 1, add_index + 2, add_index + 3});
            }
            us[core] = u[core]; use[core] = ue[core];
            us[core].push_back(dpu(D, D * count, state.addr(first * SS), p2.addr(first * SS),
                state.addr(first * SS), false, gate_hi.addr(first * D)));
            use[core].push_back(24);
            ua[core] = u[core]; uae[core] = ue[core];
            ua[core].push_back(dpu(D, D * count, state.addr(first * SS), p2.addr(first * SS), state.addr(first * SS)));
            uae[core].push_back(24);
            u[core].push_back(dpu(D, D * count, state.addr(first * SS), p2.addr(first * SS),
                state_hi.addr(first * SS), false, gate_hi.addr(first * D)));
            ue[core].push_back(24);
            u[core].push_back(dpu(D, D * count, state.addr(first * SS), state_hi.addr(first * SS),
                state.addr(first * SS), false, gate_lo.addr(first * D), true));
            ue[core].push_back(24);
        }
        grams.build(domain, g, ge,cfg.core_mask,cfg.timeout_ms);
        for (int block = 0; block < blocks; ++block) {
            std::vector<std::vector<std::vector<uint64_t>>> p(3);
            std::vector<std::vector<int>> pe(3);
            for (int core = 0; core < cores; ++core) {
                const int first = core * (HV / cores) + std::min(core, HV % cores);
                const int count = HV / cores + (core < HV % cores);
                for (int h = first; h < first + count; ++h) {
                    if (split) {
                        p[core].push_back(gemm(D, D, D, state.addr(h * SS), identity.addr(), state_hi.addr(h * SS)));
                        pe[core].push_back(13);
                        p[core].push_back(dpu(D, D, state.addr(h * SS), state_hi.addr(h * SS), state_lo.addr(h * SS), true));
                        pe[core].push_back(24);
                    }
                    const size_t at = size_t(block * HK + (cfg.head_mapping == RK_NPU_GDN_HEAD_GROUPED ? h / 2 : h % HK)) * D * 2 * C;
                    product(p[core], pe[core], D, 2 * C, D,
                        state.addr(h * SS), state_lo.addr(h * SS), b1.addr(at), b1l.addr(at),
                        p1.addr(size_t(h) * D * 2 * C), p1x.addr(size_t(h) * D * 2 * C));
                    // Adjacent value heads share the complete K/Q weight tile.
                    if (!split && cfg.head_mapping == RK_NPU_GDN_HEAD_GROUPED && h > first && h % 2 == 1) {
                        auto & body = p[core].back();
                        for (auto & word : body) if ((word & 65535) == R_CNA_CBUF_CON0)
                            word |= uint64_t(1u << 13) << 16;
                    }
                }
            }
            auto prediction = std::make_unique<Chain>();
            prediction->build(domain, p, pe,cfg.core_mask,cfg.timeout_ms);
            predictions.push_back(std::move(prediction));
        }
        update.build(domain, u, ue,cfg.core_mask,cfg.timeout_ms);
        update_add.build(domain, ua, uae,cfg.core_mask,cfg.timeout_ms);
        update_single.build(domain, us, use,cfg.core_mask,cfg.timeout_ms);
        grouped_update.build(domain, grouped, grouped_enables, front_count, tail_groups,cfg.core_mask,cfg.timeout_ms);
    }
    Times run(const Inputs& x,const Projected* producer,float* output,float* final_state) override {
        if(producer) return run_with_producer<true>(x,output,final_state,
            [&](float* a,float* al,float* b,float* bl) {
                produce_native<split,C>(workers,T,*producer,a,al,b,bl);
            });
        return run_with_producer<false>(x,output,final_state,[](float*,float*,float*,float*){});
    }
    // A native producer writes K native A and [K^T,Q^T] native B, including
    // the split-TF32 residuals. Coefficients then consume K from native A.
    template<bool native_keys, class Producer>
    Times run_with_producer(const Inputs & x, float* output,
                            float* final_state, Producer && produce) {
        Times times;
        const auto begin = Clock::now();
        std::fill(state_scale.begin(), state_scale.end(), 1.f);
        workers.run([&](int w) {
            for (int h = w; h < HV; h += workers.count()) for (int v0 = 0; v0 < D; v0 += 16) for (int k0 = 0; k0 < D; k0 += 16)
                for (int v = v0; v < v0 + 16; ++v) for (int k = k0; k < k0 + 16; k += 4)
                    vst1q_f32(state.ptr() + h * SS + si(v, k), vld1q_f32(x.state + h * SS + v * D + k));
        });
        state.sync(RK_NPU_SYNC_TO_DEVICE);
        auto mark = Clock::now();
        times.handoff += micros(begin, mark);
        produce(ga.ptr(), gal.ptr(), b1.ptr(), b1l.ptr());
        if constexpr (!native_keys) pack_grams(x);
        times.pack_compute = micros(mark, Clock::now());
        ga.sync(RK_NPU_SYNC_TO_DEVICE); b1.sync(RK_NPU_SYNC_TO_DEVICE);
        if (split) { gal.sync(RK_NPU_SYNC_TO_DEVICE); b1l.sync(RK_NPU_SYNC_TO_DEVICE); }
        auto next = Clock::now(); times.gram_pack = micros(mark, next); mark = next;
        times.gram_driver += grams.run(); times.driver += times.gram_driver;
        gc.sync(RK_NPU_SYNC_FROM_DEVICE);
        next = Clock::now(); times.gram = micros(mark, next);
        for (int block = 0; block < blocks; ++block) {
            mark = Clock::now(); coefficients<native_keys>(x, block);
            times.coefficients_compute += micros(mark, Clock::now());
            b2.sync(RK_NPU_SYNC_TO_DEVICE);
            gate_hi.sync(RK_NPU_SYNC_TO_DEVICE); gate_lo.sync(RK_NPU_SYNC_TO_DEVICE);
            if (split) b2l.sync(RK_NPU_SYNC_TO_DEVICE);
            next = Clock::now(); times.coefficients += micros(mark, next); mark = next;
            const double prediction_hw = predictions[block]->run();
            times.predict_driver += prediction_hw; times.driver += prediction_hw;
            p1.sync(RK_NPU_SYNC_FROM_DEVICE);
            next = Clock::now(); times.predict += micros(mark, next); mark = next;
            solve(x, block);
            times.solve_compute += micros(mark, Clock::now());
            a2.sync(RK_NPU_SYNC_TO_DEVICE);
            if (split) a2l.sync(RK_NPU_SYNC_TO_DEVICE);
            next = Clock::now(); times.solve += micros(mark, next); mark = next;
            const double update_hw = add_only ? update_add.run() : single_pass ? update_single.run() : all_scale ? update.run() : grouped_update.run(scale_update);
            times.update_driver += update_hw; times.driver += update_hw;
            p2.sync(RK_NPU_SYNC_FROM_DEVICE);
            const auto collect_begin = Clock::now();
            collect(block, output);
            times.collect_compute += micros(collect_begin, Clock::now());
            times.update += micros(mark, Clock::now());
        }
        mark = Clock::now(); state.sync(RK_NPU_SYNC_FROM_DEVICE);
        workers.run([&](int w) {
            for (int h = w; h < HV; h += workers.count()) for (int v0 = 0; v0 < D; v0 += 16) for (int k0 = 0; k0 < D; k0 += 16)
                for (int v = v0; v < v0 + 16; ++v) for (int k = k0; k < k0 + 16; k += 4)
                    vst1q_f32(final_state + h * SS + v * D + k,
                        vmulq_n_f32(vld1q_f32(state.ptr() + h * SS + si(v, k)), state_scale[h]));
        });
        const auto end = Clock::now(); times.handoff += micros(mark, end); times.total = micros(begin, end);
        return times;
    }
};

template<bool split>
std::unique_ptr<Engine> prepare_precision(rk_npu_iommu_domain* d,const rk_npu_gdn_config& c,Workers& workers) {
    switch(c.chunk_size) {
    case 16: return std::make_unique<Prefill<split,16>>(d,c,workers);
    case 32: return std::make_unique<Prefill<split,32>>(d,c,workers);
    case 64: return std::make_unique<Prefill<split,64>>(d,c,workers);
    case 128: return std::make_unique<Prefill<split,128>>(d,c,workers);
    default: throw Error{RK_NPU_ERR_PARAM};
    }
}
std::unique_ptr<Engine> prepare_engine(rk_npu_iommu_domain* d,const rk_npu_gdn_config& c,Workers& workers) {
    return c.precision==RK_NPU_GDN_COMPENSATED_TF32
        ? prepare_precision<true>(d,c,workers) : prepare_precision<false>(d,c,workers);
}
} // namespace rknpu2_matmul_open::gdn
#else
#include "rk_npu_gdn_internal.h"
namespace rknpu2_matmul_open::gdn {
std::unique_ptr<Engine> prepare_engine(rk_npu_iommu_domain*,const rk_npu_gdn_config&,Workers&) {
    throw Error{RK_NPU_ERR_PARAM};
}
}
#endif

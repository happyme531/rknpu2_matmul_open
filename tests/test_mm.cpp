#include "rk_npu_mm.h"
#include "rk_npu_matmul_f16.h"
#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_bfloat_bits.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bits = rknpu2_matmul_open::bits;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + \
    " at line " + std::to_string(__LINE__)); } while (0)
using Plan = std::unique_ptr<rk_npu_mm_plan, decltype(&rk_npu_mm_plan_free)>;
using Workspace = std::unique_ptr<rk_npu_mm_workspace, decltype(&rk_npu_mm_workspace_free)>;
using PackedB = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;

struct Fixture {
    Plan p{nullptr, rk_npu_mm_plan_free};
    rk_npu_mm_info i{};
    std::vector<uint16_t> a, b, c;
    Fixture(int m, int n, int k, int batches, int trans, int broadcast,
            int layouts, int nt, int split, uint32_t mask = 7, bool compact = false,
            rk_npu_mm_dtype type = RK_NPU_MM_F16) {
        rk_npu_mm_desc d;
        rk_npu_mm_desc_init(&d, m, n, k, batches);
        d.a_type = d.b_type = d.c_type = type;
        d.trans_a = trans & 1; d.trans_b = (trans >> 1) & 1;
        d.lda = (d.trans_a ? m : k) + 3;
        d.ldb = (d.trans_b ? k : n) + 5;
        d.ldc = n + 7;
        d.batch_stride_a = broadcast & 1 ? 0 : (d.trans_a ? k : m) * d.lda + 11;
        d.batch_stride_b = broadcast & 2 ? 0 : (d.trans_b ? n : k) * d.ldb + 13;
        d.batch_stride_c = m * d.ldc + 17;
        if (compact) {
            d.lda = d.trans_a ? m : k; d.ldb = d.trans_b ? k : n; d.ldc = n;
            d.batch_stride_a = broadcast & 1 ? 0 : (uint64_t)m * k;
            d.batch_stride_b = broadcast & 2 ? 0 : (uint64_t)k * n;
            d.batch_stride_c = (uint64_t)m * n;
        }
        rk_npu_mm_options o;
        rk_npu_mm_options_init(&o);
        o.a_layout = layouts & 1 ? RK_NPU_MM_LAYOUT_NATIVE : RK_NPU_MM_LAYOUT_NORMAL;
        o.c_layout = layouts & 2 ? RK_NPU_MM_LAYOUT_NATIVE : RK_NPU_MM_LAYOUT_NORMAL;
        o.n_tile = nt; o.split_k = split; o.allowed_npu_core_mask = mask;
        rk_npu_mm_plan* raw = nullptr;
        CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == 0);
        p.reset(raw);
        CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0);
        a.resize(i.host_a_bytes / 2 + 16, 0x7e00);
        b.resize(i.host_b_bytes / 2 + 16, 0x7e00);
        c.resize(i.host_c_bytes / 2 + 16, 0x3555);
        for (int batch = 0; batch < batches; ++batch) {
            for (int r = 0; r < m; ++r)
                for (int q = 0; q < k; ++q)
                    a[ai(batch, r, q)] = encode(float((r * 7 + q * 3 + batch * 5) % 23 - 11) / 37);
            for (int q = 0; q < k; ++q)
                for (int s = 0; s < n; ++s)
                    b[bi(batch, q, s)] = encode(float((q * 5 + s * 11 + batch * 3) % 19 - 9) / 43);
        }
    }
    uint16_t encode(float x) const {
        return i.desc.a_type == RK_NPU_MM_BF16 ? bits::float_to_bfloat(x) : bits::float_to_half(x);
    }
    float decode(uint16_t x) const {
        return i.desc.a_type == RK_NPU_MM_BF16 ? bits::bfloat_to_float(x) : bits::half_to_float(x);
    }
    uint64_t ai(int batch, int m, int k) const {
        const auto& d = i.desc;
        return batch * d.batch_stride_a + (d.trans_a ? k * d.lda + m : m * d.lda + k);
    }
    uint64_t bi(int batch, int k, int n) const {
        const auto& d = i.desc;
        return batch * d.batch_stride_b + (d.trans_b ? n * d.ldb + k : k * d.ldb + n);
    }
    uint64_t ci(int batch, int m, int n) const {
        return batch * i.desc.batch_stride_c + m * i.desc.ldc + n;
    }
    void verify() const {
        const auto& d = i.desc;
        std::vector<bool> used(c.size(), false);
        double max_err = 0;
        for (int batch = 0; batch < d.batch_count; ++batch)
            for (int m = 0; m < d.M; ++m)
                for (int n = 0; n < d.N; ++n) {
                    double ref = 0;
                    for (int k = 0; k < d.K; ++k)
                        ref += double(decode(a[ai(batch, m, k)])) * decode(b[bi(batch, k, n)]);
                    const double got = decode(c[ci(batch, m, n)]);
                    CHECK(std::isfinite(got));
                    max_err = std::max(max_err, std::fabs(got - ref));
                    const double atol = d.c_type == RK_NPU_MM_BF16 ? 0.015 : 0.003;
                    const double rtol = d.c_type == RK_NPU_MM_BF16 ? 0.02 : 0.005;
                    CHECK(std::fabs(got - ref) <= atol + rtol * std::fabs(ref));
                    used[ci(batch, m, n)] = true;
                }
        for (size_t j = 0; j < c.size(); ++j)
            if (!used[j]) CHECK(c[j] == 0x3555);
        std::printf(" max_abs=%.6f", max_err);
    }
};

rk_npu_mem host_mem(std::vector<uint16_t>& v, uint64_t bytes) {
    rk_npu_mem m{}; m.vaddr = v.data(); m.size = bytes; return m;
}

void layout_test(Fixture& f) {
    const auto& i = f.i;
    const auto& d = i.desc;
    std::vector<uint16_t> a(i.device.input_bytes / 2 + 8, 0x5555);
    std::vector<uint16_t> b(i.device.weight_bytes / 2 + 8, 0x5555);
    auto am = host_mem(a, i.device.input_bytes), bm = host_mem(b, i.device.weight_bytes);
    CHECK(rk_npu_mm_pack_a(f.p.get(), f.a.data(), i.host_a_bytes, &am) == 0);
    CHECK(rk_npu_mm_pack_b(f.p.get(), f.b.data(), i.host_b_bytes, &bm) == 0);
    CHECK(a.back() == 0x5555 && b.back() == 0x5555);
    rk_npu_matmul_f16_config cfg;
    rk_npu_matmul_f16_config_init(&cfg, d.M, d.N, i.part_k, RK_NPU_FUSE_NONE);
    cfg.a_layout = i.options.a_layout == RK_NPU_MM_LAYOUT_NATIVE ?
        RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8 : RK_NPU_F16_A_LAYOUT_NORMAL;
    cfg.d_layout = i.options.c_layout == RK_NPU_MM_LAYOUT_NATIVE ?
        RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8 : RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;
    cfg.n_tile = i.options.n_tile;
    rk_npu_matmul_sizes one{};
    CHECK(rk_npu_matmul_f16_query(&cfg, &one) == 0);
    std::vector<uint16_t> compact_a(d.M * i.part_k), compact_b(i.part_k * d.N);
    std::vector<uint16_t> ra(one.input_bytes / 2), rb(one.weight_bytes / 2);
    auto ram = host_mem(ra, one.input_bytes), rbm = host_mem(rb, one.weight_bytes);
    for (int batch = 0; batch < d.batch_count; ++batch)
        for (int split = 0; split < i.split_count; ++split) {
            std::fill(compact_a.begin(), compact_a.end(), 0);
            std::fill(compact_b.begin(), compact_b.end(), 0);
            for (int k = 0; k < i.part_k && split * i.part_k + k < d.K; ++k) {
                for (int m = 0; m < d.M; ++m)
                    compact_a[m * i.part_k + k] = f.a[f.ai(batch, m, split * i.part_k + k)];
                for (int n = 0; n < d.N; ++n)
                    compact_b[k * d.N + n] = f.b[f.bi(batch, split * i.part_k + k, n)];
            }
            CHECK(rk_npu_matmul_f16_pack_a(&cfg, compact_a.data(), &ram) == 0);
            CHECK(rk_npu_matmul_f16_pack_b(&cfg, compact_b.data(), &rbm) == 0);
            const int item = batch * i.split_count + split;
            CHECK(std::equal(ra.begin(), ra.end(), a.begin() + item * ra.size()));
            CHECK(std::equal(rb.begin(), rb.end(), b.begin() + item * rb.size()));
        }
    // Compare new scatter/reduction with the established per-item unpacker,
    // using arbitrary packed data rather than mirroring the new index formula.
    std::vector<uint16_t> output(i.device.output_bytes / 2), item_c(d.M * d.N);
    for (size_t j = 0; j < output.size(); ++j)
        output[j] = f.encode(float(int(j % 31) - 15) / 16);
    auto om = host_mem(output, i.device.output_bytes);
    CHECK(rk_npu_mm_unpack_c(f.p.get(), &om, f.c.data(), i.host_c_bytes) == 0);
    std::vector<bool> written(f.c.size(), false);
    for (int batch = 0; batch < d.batch_count; ++batch) {
        std::vector<float> sums(d.M * d.N, 0);
        for (int split = 0; split < i.split_count; ++split) {
            rk_npu_mem view = om;
            view.vaddr = output.data() + (batch * i.split_count + split) * one.output_bytes / 2;
            view.size = one.output_bytes;
            CHECK(rk_npu_matmul_f16_unpack_d(&cfg, &view, item_c.data()) == 0);
            for (size_t j = 0; j < sums.size(); ++j) sums[j] += f.decode(item_c[j]);
        }
        for (int m = 0; m < d.M; ++m)
            for (int n = 0; n < d.N; ++n) {
                CHECK(f.c[f.ci(batch, m, n)] == f.encode(sums[m * d.N + n]));
                written[f.ci(batch, m, n)] = true;
            }
    }
    for (size_t j = 0; j < f.c.size(); ++j) if (!written[j]) CHECK(f.c[j] == 0x3555);
    CHECK(rk_npu_mm_pack_a(f.p.get(), f.a.data(), i.host_a_bytes - 1, &am) == RK_NPU_ERR_NOMEM);
    am.size--;
    CHECK(rk_npu_mm_pack_a(f.p.get(), f.a.data(), i.host_a_bytes, &am) == RK_NPU_ERR_NOMEM);
    CHECK(rk_npu_mm_unpack_c(f.p.get(), &om, output.data(), i.host_c_bytes) == RK_NPU_ERR_PARAM);
}

void policy_test() {
    rk_npu_mm_desc d;
    rk_npu_mm_desc_init(&d, 3, 65, 2049, 1);
    rk_npu_mm_plan* raw = nullptr;
    unsetenv("RK_NPU_MM_SPLIT_K");
    CHECK(rk_npu_mm_plan_create(&d, nullptr, &raw) == 0);
    Plan p(raw, rk_npu_mm_plan_free);
    rk_npu_mm_info i;
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0);
    CHECK(i.split_count == 3 && i.part_k == 704 && i.actual_npu_core_mask == 7);
    setenv("RK_NPU_MM_SPLIT_K", "0", 1);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.split_count == 3);
    CHECK(rk_npu_mm_plan_create(&d, nullptr, &raw) == 0);
    p.reset(raw);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.split_count == 1);
    setenv("RK_NPU_MM_SPLIT_K", "true", 1);
    CHECK(rk_npu_mm_plan_create(&d, nullptr, &raw) == RK_NPU_ERR_PARAM && raw == nullptr);
    rk_npu_mm_options o;
    rk_npu_mm_options_init(&o); o.split_k = 2;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == 0);
    p.reset(raw);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.split_count == 2);
    o.allowed_npu_core_mask = 4;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_ERR_PARAM);
    o.split_k = 1;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == 0);
    p.reset(raw);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.split_count == 1 && i.actual_npu_core_mask == 4);
    o.allowed_npu_core_mask = 7;
    d.lda = UINT64_MAX;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_ERR_PARAM);
    d.lda = 0; d.batch_count = 2; d.batch_stride_a = 1;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_ERR_PARAM);
    rk_npu_mm_desc_init(&d, 3, 65, 2049, 3);
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == 0);
    p.reset(raw);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.split_count == 1);
    for (auto type : {RK_NPU_MM_BF16, RK_NPU_MM_F32, RK_NPU_MM_I8, RK_NPU_MM_I4}) {
        d.a_type = type;
        CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_MM_ERR_UNSUPPORTED);
    }
    d.a_type = d.b_type = d.c_type = RK_NPU_MM_BF16;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == 0);
    p.reset(raw);
    CHECK(rk_npu_mm_plan_get_info(p.get(), &i) == 0 && i.desc.c_type == RK_NPU_MM_BF16);
    d.c_type = RK_NPU_MM_F32;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_MM_ERR_UNSUPPORTED);
    d.b_type = d.c_type = RK_NPU_MM_F16;
    d.a_type = d.b_type = d.c_type = RK_NPU_MM_F32;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_MM_ERR_UNSUPPORTED);
    d.a_type = d.b_type = d.c_type = RK_NPU_MM_F16; d.M = INT32_MAX;
    CHECK(rk_npu_mm_plan_create(&d, &o, &raw) == RK_NPU_MM_ERR_UNSUPPORTED);
    unsetenv("RK_NPU_MM_SPLIT_K");
}

void bfloat_cpu_tests() {
    // Exhaust every finite BF16 bit pattern and both infinities; widening and
    // round-trip must preserve sign and subnormals. NaNs must remain NaNs.
    for (uint32_t h = 0; h <= 0xffff; ++h) {
        const float x = bits::bfloat_to_float(uint16_t(h));
        uint32_t raw; std::memcpy(&raw, &x, 4);
        CHECK(raw == h << 16);
        const uint16_t got = bits::float_to_bfloat(x);
        if ((h & 0x7f80) == 0x7f80 && (h & 0x7f))
            CHECK((got & 0x7f80) == 0x7f80 && (got & 0x7f));
        else CHECK(got == h);
    }
    const uint32_t input[] = {0x3f808000,0x3f818000,0xbf808000,0xbf818000,
                             0x00008000,0x00018000,0x7f7fffff,0x7f800001,0xff800001};
    const uint16_t expected[] = {0x3f80,0x3f82,0xbf80,0xbf82,0x0000,0x0002,0x7f80,0x7fc0,0xffc0};
    for (size_t j = 0; j < sizeof(input)/sizeof(input[0]); ++j) {
        float x; std::memcpy(&x, &input[j], 4);
        CHECK(bits::float_to_bfloat(x) == expected[j]);
    }
    Fixture split(1, 1, 96, 1, 0, 0, 0, 0, 3, 7, true, RK_NPU_MM_BF16);
    std::vector<uint16_t> partial(split.i.device.output_bytes/2, 0);
    auto pm = host_mem(partial, split.i.device.output_bytes);
    const size_t step = partial.size()/3;
    partial[0] = 0x7180; partial[step] = 0xf180; partial[2*step] = 0x4000;
    CHECK(rk_npu_mm_unpack_c(split.p.get(), &pm, split.c.data(), split.i.host_c_bytes) == 0);
    CHECK(split.c[0] == 0x4000); // BF16 2^100 - 2^100 + 2; FP16 decoding cannot pass.
    partial[0] = 0x3f81; partial[step] = 0x3b80; partial[2*step] = 0;
    CHECK(rk_npu_mm_unpack_c(split.p.get(), &pm, split.c.data(), split.i.host_c_bytes) == 0);
    CHECK(split.c[0] == 0x3f82); // odd halfway significand rounds up to even
    Fixture raw(1, 8, 32, 1, 0, 0, 0, 0, 0, 1, true, RK_NPU_MM_BF16);
    const uint16_t patterns[] = {0x8000,0x0001,0x7f7f,0x7f80,0xff80,0x7f81,0x7fc1,0x7180};
    std::vector<uint16_t> output(raw.i.device.output_bytes/2, 0);
    std::copy(std::begin(patterns), std::end(patterns), output.begin());
    auto om = host_mem(output, raw.i.device.output_bytes);
    CHECK(rk_npu_mm_unpack_c(raw.p.get(), &om, raw.c.data(), raw.i.host_c_bytes) == 0);
    CHECK(std::equal(std::begin(patterns), std::end(patterns), raw.c.begin()));
}

void cpu_tests() {
    policy_test();
    bfloat_cpu_tests();
    int cases = 0;
    for (auto type : {RK_NPU_MM_F16, RK_NPU_MM_BF16}) {
    for (int trans = 0; trans < 4; ++trans)
        for (int layout = 0; layout < 4; ++layout)
            for (int split : {0, 2, 3}) {
                Fixture f(5, 65, 97, 2, trans, trans, layout, 32, split, 7, false, type);
                layout_test(f); ++cases;
            }
    Fixture tiny(1, 1, 1, 1, 0, 0, 0, 0, 0, 7, false, type);
    layout_test(tiny);
    Fixture normal(7, 80, 33, 3, 2, 2, 0, 0, 0, 7, false, type);
    layout_test(normal);
    Fixture automatic(3, 35, 2049, 1, 3, 0, 3, 32, -1, 7, false, type);
    layout_test(automatic);
    for (int layout = 0; layout < 4; ++layout) {
        Fixture compact(7, 80, 33, 3, 0, 0, layout, 32, 0, 7, true, type);
        layout_test(compact); ++cases;
    }
    cases += 3;
    }
    std::printf("CPU PASS: %d FP16/BF16 layout cases plus conversion, policy and guards\n", cases);
}

Workspace workspace(rk_npu_iommu_domain* domain, const Fixture& f, rk_npu_mm_workspace_mode mode) {
    rk_npu_mm_workspace* w = nullptr;
    CHECK(rk_npu_mm_workspace_create(domain, f.p.get(), mode, &w) == 0);
    return Workspace(w, rk_npu_mm_workspace_free);
}
PackedB weights(rk_npu_iommu_domain* domain, const Fixture& f) {
    rk_npu_mm_packed_b* b = nullptr;
    CHECK(rk_npu_mm_packed_b_create(domain, f.p.get(), f.b.data(), f.i.host_b_bytes, &b) == 0);
    return PackedB(b, rk_npu_mm_packed_b_free);
}
void board_case(rk_npu_iommu_domain* domain, Fixture& f, const char* name) {
    auto w = workspace(domain, f, RK_NPU_MM_HOST_DYNAMIC);
    const auto start = std::chrono::steady_clock::now();
    CHECK(rk_npu_mm_run(w.get(), f.a.data(), f.i.host_a_bytes,
                        f.b.data(), f.i.host_b_bytes, f.c.data(), f.i.host_c_bytes) == 0);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s %s split=%d mask=%u", f.i.desc.a_type == RK_NPU_MM_BF16 ? "BF16" : "FP16",
                name, f.i.split_count, f.i.actual_npu_core_mask);
    f.verify();
    auto packed = weights(domain, f);
    auto wp = workspace(domain, f, RK_NPU_MM_HOST_PACKED_B);
    const auto expected = f.c;
    CHECK(rk_npu_mm_run_packed_b(wp.get(), f.a.data(), f.i.host_a_bytes,
                                packed.get(), f.c.data(), f.i.host_c_bytes) == 0);
    CHECK(f.c == expected);
    // Reuse cacheable buffers with changed input; no stale data across calls.
    f.a[f.ai(0, 0, 0)] = f.encode(1.0f);
    CHECK(rk_npu_mm_run(w.get(), f.a.data(), f.i.host_a_bytes,
                        f.b.data(), f.i.host_b_bytes, f.c.data(), f.i.host_c_bytes) == 0);
    f.verify();
    CHECK(rk_npu_mm_run(w.get(), f.a.data(), f.i.host_a_bytes - 1,
                        f.b.data(), f.i.host_b_bytes, f.c.data(), f.i.host_c_bytes) == RK_NPU_ERR_NOMEM);
    CHECK(rk_npu_mm_run(w.get(), f.a.data(), f.i.host_a_bytes,
                        f.b.data(), f.i.host_b_bytes, f.a.data(), f.i.host_c_bytes) == RK_NPU_ERR_PARAM);
    std::printf(" complete_call_first_us=%.1f PASS\n", us);
}
void board_tests(rk_npu_mm_dtype type) {
    std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr), rk_npu_close);
    CHECK(ctx);
    std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
        rk_npu_iommu_domain_create(ctx.get(), 0), rk_npu_iommu_domain_free);
    CHECK(domain);
    Fixture f1(5, 65, 97, 2, 3, 2, 0, 32, 0, 7, false, type);
    board_case(domain.get(), f1, "strided_bmm");
    Fixture f2(3, 80, 2049, 1, 2, 0, 0, 0, -1, 7, false, type);
    board_case(domain.get(), f2, "auto_split_mm");
    Fixture f3(5, 65, 97, 2, 1, 1, 3, 32, 3, 7, false, type);
    board_case(domain.get(), f3, "native_split_bmm");
    Fixture f4(7, 80, 33, 1, 0, 0, 2, 32, 0, 4, false, type);
    board_case(domain.get(), f4, "single_core2");
    Fixture f5(3, 65, 2049, 2, 3, 2, 1, 32, 2, 3, false, type);
    board_case(domain.get(), f5, "two_core_split_bmm");
    Fixture f6(99, 65, 2049, 1, 0, 0, 3, 32, -1, 7, false, type);
    board_case(domain.get(), f6, "split_with_mn_tiling");
    Fixture f7(7, 80, 33, 3, 0, 0, 3, 32, 0, 7, true, type);
    board_case(domain.get(), f7, "compact_fast_path");
    setenv("RK_NPU_MM_SPLIT_K", "0", 1);
    Fixture f8(3, 80, 2049, 1, 2, 0, 0, 0, -1, 7, false, type);
    CHECK(f8.i.split_count == 1);
    board_case(domain.get(), f8, "env_disabled");
    unsetenv("RK_NPU_MM_SPLIT_K");
    // B packing is independent of M, A/C layout and N tiling.
    auto b = weights(domain.get(), f2);
    Fixture other(7, 80, 2049, 1, 2, 0, 3, 32, 3, 7, false, type);
    auto w = workspace(domain.get(), other, RK_NPU_MM_HOST_PACKED_B);
    CHECK(rk_npu_mm_run_packed_b(w.get(), other.a.data(), other.i.host_a_bytes,
                                b.get(), other.c.data(), other.i.host_c_bytes) == 0);
    std::printf("packed_b_reuse_across_m"); other.verify(); std::printf(" PASS\n");
    Fixture incompatible(7, 80, 2049, 1, 2, 0, 0, 0, 0, 7, false, type);
    auto bad = workspace(domain.get(), incompatible, RK_NPU_MM_HOST_PACKED_B);
    CHECK(rk_npu_mm_run_packed_b(bad.get(), incompatible.a.data(), incompatible.i.host_a_bytes,
                                b.get(), incompatible.c.data(), incompatible.i.host_c_bytes) == RK_NPU_ERR_PARAM);
    Fixture wrong_type(7, 80, 2049, 1, 2, 0, 3, 32, 3, 7, false,
                      type == RK_NPU_MM_BF16 ? RK_NPU_MM_F16 : RK_NPU_MM_BF16);
    auto typed = workspace(domain.get(), wrong_type, RK_NPU_MM_HOST_PACKED_B);
    const auto untouched = wrong_type.c;
    CHECK(rk_npu_mm_run_packed_b(typed.get(), wrong_type.a.data(), wrong_type.i.host_a_bytes,
                                b.get(), wrong_type.c.data(), wrong_type.i.host_c_bytes) == RK_NPU_ERR_PARAM);
    CHECK(wrong_type.c == untouched);

    // Device-only mode, explicit ownership and split partial unpacking.
    auto wd = workspace(domain.get(), f3, RK_NPU_MM_DEVICE_ONLY);
    rk_npu_mem a{}, wb{}, c{};
    CHECK(rk_npu_mem_alloc(domain.get(), f3.i.device.input_bytes, RK_NPU_MEM_DATA_DEFAULT, &a) == 0);
    CHECK(rk_npu_mem_alloc(domain.get(), f3.i.device.weight_bytes, RK_NPU_MEM_DATA_DEFAULT, &wb) == 0);
    CHECK(rk_npu_mem_alloc(domain.get(), f3.i.device.output_bytes, RK_NPU_MEM_DATA_DEFAULT, &c) == 0);
    CHECK(rk_npu_mm_pack_a(f3.p.get(), f3.a.data(), f3.i.host_a_bytes, &a) == 0);
    CHECK(rk_npu_mm_pack_b(f3.p.get(), f3.b.data(), f3.i.host_b_bytes, &wb) == 0);
    CHECK(rk_npu_mem_sync(ctx.get(), &a, RK_NPU_SYNC_TO_DEVICE) == 0);
    CHECK(rk_npu_mem_sync(ctx.get(), &wb, RK_NPU_SYNC_TO_DEVICE) == 0);
    CHECK(rk_npu_mm_run_device(wd.get(), &a, &wb, &c) == 0);
    CHECK(rk_npu_mem_sync(ctx.get(), &c, RK_NPU_SYNC_FROM_DEVICE) == 0);
    CHECK(rk_npu_mm_unpack_c(f3.p.get(), &c, f3.c.data(), f3.i.host_c_bytes) == 0);
    std::printf("device_only"); f3.verify(); std::printf(" PASS\n");
    auto wrong = a; wrong.iommu_domain_id = 1;
    CHECK(rk_npu_mm_run_device(wd.get(), &wrong, &wb, &c) == RK_NPU_ERR_DOMAIN);
    CHECK(rk_npu_mm_run_device(wd.get(), &a, &wb, &a) != 0);
    rk_npu_mem_free(ctx.get(), &a); rk_npu_mem_free(ctx.get(), &wb); rk_npu_mem_free(ctx.get(), &c);
    if (type == RK_NPU_MM_BF16) {
        // Exact powers outside FP16's range, through the actual public API.
        // Each row has one nonzero K term; split and unsplit must agree.
        for (int splits : {0, 3}) {
            Fixture range(4, 32, 96, 1, 3, 0, 3, 32, splits, 7, false, type);
            std::fill(range.a.begin(), range.a.end(), 0);
            std::fill(range.b.begin(), range.b.end(), 0);
            range.a[range.ai(0, 0, 0)] = 0x6780;  // 2^80
            range.a[range.ai(0, 1, 32)] = 0x2b80; // 2^-40
            range.a[range.ai(0, 2, 64)] = 0x7f7f; // maximum finite BF16
            range.a[range.ai(0, 3, 0)] = 0x3f80;
            for (int n = 0; n < 32; ++n) {
                range.b[range.bi(0, 0, n)] = 0x4980;  // 2^20
                range.b[range.bi(0, 32, n)] = 0x2b80; // 2^-40
                range.b[range.bi(0, 64, n)] = 0x3f80;
            }
            auto rw = workspace(domain.get(), range, RK_NPU_MM_HOST_DYNAMIC);
            CHECK(rk_npu_mm_run(rw.get(), range.a.data(), range.i.host_a_bytes,
                                range.b.data(), range.i.host_b_bytes,
                                range.c.data(), range.i.host_c_bytes) == 0);
            const uint16_t expected[] = {0x7180, 0x1780, 0x7f7f, 0x4980};
            for (int m = 0; m < 4; ++m) for (int n = 0; n < 32; ++n)
                CHECK(range.c[range.ci(0, m, n)] == expected[m]);
        }
        std::printf("BF16 range with/without split exact PASS\n");
        Fixture f16(5, 65, 97, 2, 3, 2, 0, 32, 0);
        board_case(domain.get(), f16, "fp16_after_bf16_same_context");
    }
    // Workspace/packed B own copies; freeing plan/domain handles is safe.
    other.p.reset(); domain.reset();
    CHECK(rk_npu_mm_run_packed_b(w.get(), other.a.data(), other.i.host_a_bytes,
                                b.get(), other.c.data(), other.i.host_c_bytes) == 0);
    std::printf("BOARD ALL PASS\n");
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1], "--board")) {
            board_tests(RK_NPU_MM_F16);
            board_tests(RK_NPU_MM_BF16);
        }
        else if (argc == 2 && !std::strcmp(argv[1], "--cpu")) cpu_tests();
        else { std::fprintf(stderr, "usage: test_mm --cpu|--board\n"); return 2; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    return 0;
}

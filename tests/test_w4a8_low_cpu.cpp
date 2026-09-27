#include "../src/rk_npu_half_bits.h"
#include "../src/rk_npu_w4a8_cpu.h"
#include "rk_npu_w4a8.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
// Interpose device allocations to verify the public query and reuse contract.
static uint64_t allocations = 0, requested = 0, live = 0;
static int fail_after = -1;
static std::unordered_map<uint64_t, uint64_t> sizes;
extern "C" int rk_npu_mem_alloc(rk_npu_iommu_domain *d, uint64_t bytes, uint32_t flags,
                                rk_npu_mem *mem) {
    static auto real =
        reinterpret_cast<decltype(&rk_npu_mem_alloc)>(dlsym(RTLD_NEXT, "rk_npu_mem_alloc"));
    if (fail_after == 0)
        return RK_NPU_ERR_NOMEM;
    if (fail_after > 0)
        --fail_after;
    const int rc = real(d, bytes, flags, mem);
    if (!rc) {
        ++allocations;
        requested += bytes;
        live += bytes;
        sizes[mem->obj_addr] = bytes;
    }
    return rc;
}
extern "C" int rk_npu_mem_free(rk_npu_ctx *ctx, rk_npu_mem *mem) {
    static auto real =
        reinterpret_cast<decltype(&rk_npu_mem_free)>(dlsym(RTLD_NEXT, "rk_npu_mem_free"));
    const auto key = mem ? mem->obj_addr : 0;
    const int rc = real(ctx, mem);
    if (!rc) {
        auto it = sizes.find(key);
        if (it != sizes.end()) {
            live -= it->second;
            sizes.erase(it);
        }
    }
    return rc;
}
static void host() {
    rk_npu_w4a8_config c;
    rk_npu_w4a8_config_init(&c, 128, 4096, 11008);
    c.k_tile = 2048;
    c.npu_core_mask = 7;
    c.m_tile = 128;
    c.n_tile = 256;
    rk_npu_w4a8_options opt;
    rk_npu_w4a8_options_init(&opt);
    opt.input_layout = RK_NPU_W4A8_INPUT_PANEL8;
    opt.reduce_backend = RK_NPU_W4A8_REDUCE_NPU;
    rk_npu_w4a8_memory_info_ex info{};
    CHECK(!rk_npu_w4a8_memory_query_ex(&c, &opt, &info));
    CHECK(info.base.cpu_scratch_bytes == 128 * 8);
    CHECK(info.base.backend.partial_bytes == 128ull * 4096 * 4 * 6 + 128 * 256 * 4 * 6);
    CHECK(info.workspace_bytes == info.base.backend.input_bytes + info.base.backend.partial_bytes +
                                      info.base.backend.control_bytes +
                                      info.base.cpu_scratch_bytes + info.reduction_output_bytes +
                                      info.reduction_weight_bytes + info.reduction_control_bytes);
    c.n_tile = 512;
    CHECK(rk_npu_w4a8_memory_query_ex(&c, &opt, &info) == RK_NPU_ERR_PARAM);
    opt.reduce_backend = RK_NPU_W4A8_REDUCE_CPU;
    CHECK(!rk_npu_w4a8_memory_query_ex(&c, &opt, &info));
    c.m_tile = 256;
    CHECK(rk_npu_w4a8_memory_query_ex(&c, &opt, &info) == 0); // actual M is only128
    c.M = 256;
    CHECK(rk_npu_w4a8_memory_query_ex(&c, &opt, &info) == RK_NPU_ERR_PARAM); // >8 banks
    c.k_tile = 992;
    CHECK(!rk_npu_w4a8_memory_query_ex(&c, &opt, &info));
    opt.struct_size = 0;
    CHECK(rk_npu_w4a8_memory_query_ex(&c, &opt, &info) == RK_NPU_ERR_PARAM);
    rk_npu_w4a8_options_init(&opt);
    opt.reduce_backend = RK_NPU_W4A8_REDUCE_NPU;
    c.M = 17;
    c.m_tile = 1;
    c.k_tile = 32;
    c.K = 32 * 32;
    CHECK(rk_npu_w4a8_memory_query_ex(&c, &opt, &info) == RK_NPU_ERR_PARAM); // 32 waves
    c.M = 16;
    CHECK(!rk_npu_w4a8_memory_query_ex(&c, &opt, &info));
    CHECK(info.effective_reduce_backend == RK_NPU_W4A8_REDUCE_CPU && !info.reduction_output_bytes);
    const auto small = info;
    opt.reduce_backend = RK_NPU_W4A8_REDUCE_CPU;
    CHECK(!rk_npu_w4a8_memory_query_ex(&c, &opt, &info));
    CHECK(info.workspace_bytes == small.workspace_bytes);
    for (int M : {1, 3, 4, 5, 65, 127, 128, 129, 257})
        for (int K : {1, 32, 33, 64, 65, 96, 97, 160, 192, 224, 480, 992, 2048}) {
            std::vector<float> a(size_t(M) * K), scale(M), inv(M);
            for (size_t i = 0; i < a.size(); ++i)
                a[i] = float(int(i % 255) - 127);
            CHECK(!rknpu2_matmul_open::detail::w4a8_scales(a.data(), false, M, K, 1, scale.data(), inv.data()));
            rknpu2_matmul_open::detail::W4A8Input in{a.data(),     false,      M, K,
                                scale.data(), inv.data(), 1, rknpu2_matmul_open::detail::I4InputLayout::Panel8};
            int pk = (K + 31) / 32 * 32;
            for (int m0 = 0; m0 < 2 * M;) {
                const int rows = rknpu2_matmul_open::detail::i4_m_rows(2 * M - m0, 256, in.layout);
                std::vector<uint8_t> dst(size_t(rows) * pk / 2 + 16, 0xa5);
                rk_npu_i4_input_tile t{m0, rows, 0, K, pk, dst.data(), uint64_t(rows) * pk / 2};
                CHECK(!rknpu2_matmul_open::detail::w4a8_pack(&in, &t));
                auto nibble = [&](size_t pos) {
                    int v = (dst[pos / 2] >> ((pos & 1) ? 0 : 4)) & 15;
                    return v >= 8 ? v - 16 : v;
                };
                for (int r = 0; r < rows; r += 2)
                    for (int k = 0; k < pk; ++k) {
                        const size_t pos =
                            (rows >= 8 ? size_t(r / 8) * 8 * pk + (k / 32 * 8 + r % 8) * 32
                                       : size_t(k / 32 * rows + r) * 32) +
                            k % 32;
                        const int h = nibble(pos), l = nibble(pos + 32);
                        if (k >= K)
                            CHECK(h == 0 && l == 0);
                        else
                            CHECK(16 * h + l + 8 ==
                                  int(std::clamp(std::round(a[size_t((m0 + r) / 2) * K + k] *
                                                            inv[(m0 + r) / 2]),
                                                 -127.f, 127.f)));
                    }
                CHECK(std::all_of(dst.end() - 16, dst.end(), [](uint8_t v) { return v == 0xa5; }));
                m0 += rows;
            }
        }
    std::puts("PASS host: panel8 packing/tails, mode/geometry validation, memory query");
}
static void one(rk_npu_iommu_domain *domain, int M, int K, int N, int kt, int mt, int nt, int mask,
                bool panel, int repeats = 3) {
    rk_npu_w4a8_config cfg;
    rk_npu_w4a8_config_init(&cfg, M, N, K);
    cfg.k_tile = kt;
    cfg.m_tile = mt;
    cfg.n_tile = nt;
    cfg.npu_core_mask = mask;
    cfg.cpu_threads = M * N >= 65536 ? 4 : 1;
    rk_npu_w4a8_options opt;
    rk_npu_w4a8_options_init(&opt);
    opt.input_layout = panel ? RK_NPU_W4A8_INPUT_PANEL8 : RK_NPU_W4A8_INPUT_NATIVE;
    opt.reduce_backend = RK_NPU_W4A8_REDUCE_NPU;
    rk_npu_w4a8_memory_info_ex info{};
    CHECK(!rk_npu_w4a8_memory_query_ex(&cfg, &opt, &info));
    std::vector<int8_t> b(size_t(K) * N), b2(b.size());
    std::vector<float> a(size_t(M) * K), s(N), s2(N), c(size_t(M) * N + 32);
    std::vector<uint16_t> ah(a.size()), ch(c.size());
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = int((i * 31 + i / N) % 16) - 8;
        b2[i] = -2;
    }
    // Large K uses small weights to keep the full-call oracle certified.
    if (kt > 480) {
        for (auto &v : b)
            v = v % 2;
        std::fill(b2.begin(), b2.end(), -1);
    }
    for (int n = 0; n < N; ++n) {
        s[n] = .003f * (1 + n % 7);
        s2[n] = .004f * (1 + n % 3);
    }
    rk_npu_i4_weight_config wc{K, N, kt};
    auto *w = rk_npu_w4a8_weights_create(domain, &wc, b.data(), s.data());
    auto *w2 = rk_npu_w4a8_weights_create(domain, &wc, b2.data(), s2.data());
    CHECK(w && w2);
    const uint64_t before = requested, baseline = live, creation_count = allocations;
    auto *ws = rk_npu_w4a8_workspace_create_ex(domain, &cfg, &opt);
    CHECK(ws);
    CHECK(requested - before == info.workspace_bytes - info.base.cpu_scratch_bytes);
    CHECK(info.effective_reduce_backend ==
          (M <= 16 ? RK_NPU_W4A8_REDUCE_CPU : RK_NPU_W4A8_REDUCE_NPU));
    if (M <= 16)
        CHECK(!info.reduction_output_bytes && !info.reduction_weight_bytes &&
              !info.reduction_control_bytes);
    const auto count = allocations;
    for (int repeat = 0; repeat < repeats; ++repeat)
        for (bool half : {false, true}) {
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = i < size_t(K) ? 0.f : float(int((i * 13 + repeat) % 511) - 255) * .125f;
                ah[i] = rknpu2_matmul_open::bits::float_to_half(a[i]);
            }
            std::fill(c.begin(), c.end(), -12345.f);
            std::fill(ch.begin(), ch.end(), 0x5555);
            const bool second = repeat % 2;
            const auto &B = second ? b2 : b;
            const auto &S = second ? s2 : s;
            const int rc =
                half ? rk_npu_w4a8_run_f16(ws, second ? w2 : w, ah.data(), ch.data() + 16, nullptr)
                     : rk_npu_w4a8_run_f32(ws, second ? w2 : w, a.data(), c.data() + 16, nullptr);
            CHECK(rc == RK_NPU_OK);
            CHECK(allocations == count);
            for (int m = 0; m < M; ++m) {
                float maximum = 0;
                for (int k = 0; k < K; ++k)
                    maximum = std::max(maximum, std::fabs(a[size_t(m) * K + k]));
                const float scale = maximum ? std::max(maximum / 127.f, FLT_MIN) : 1.f,
                            inv = 1.f / scale;
                std::vector<int> q(K);
                for (int k = 0; k < K; ++k)
                    q[k] = int(std::clamp(std::round(a[size_t(m) * K + k] * inv), -127.f, 127.f));
                for (int n = 0; n < N; ++n) {
                    int64_t sum = 0;
                    for (int k = 0; k < K; ++k)
                        sum += q[k] * B[size_t(k) * N + n];
                    const float ref = (float(sum) * scale) * S[n];
                    const size_t ix = size_t(m) * N + n + 16;
                    if (half ? ch[ix] != rknpu2_matmul_open::bits::float_to_half(ref) : c[ix] != ref) {
                        std::fprintf(stderr, "m=%d n=%d ref=%g got=%g half=%d\n", m, n, ref,
                                     half ? rknpu2_matmul_open::bits::half_to_float(ch[ix]) : c[ix], half);
                        throw std::runtime_error("NPU reduce output mismatch; stop");
                    }
                }
            }
            for (int i = 0; i < 16; ++i)
                CHECK(c[i] == -12345.f && c[c.size() - 1 - i] == -12345.f && ch[i] == 0x5555 &&
                      ch[ch.size() - 1 - i] == 0x5555);
        }
    rk_npu_w4a8_workspace_free(ws);
    CHECK(live == baseline);
    if (M == 17 && K == 480) {
        for (uint64_t i = 0; i < count - creation_count; ++i) {
            fail_after = int(i);
            CHECK(!rk_npu_w4a8_workspace_create_ex(domain, &cfg, &opt));
            fail_after = -1;
            CHECK(live == baseline);
        }
        std::puts("PASS cleanup after failure at each workspace device allocation");
    }
    rk_npu_w4a8_weights_free(w2);
    rk_npu_w4a8_weights_free(w);
    CHECK(live == 0);
    std::printf("PASS M=%d K=%d N=%d kt=%d mt=%d nt=%d mask=%d panel=%d reuse=%d bytes=%llu "
                "effective=%d full-exact f32/f16 alloc/free\n",
                M, K, N, kt, mt, nt, mask, panel, repeats, (unsigned long long)info.workspace_bytes,
                int(info.effective_reduce_backend));
    std::fflush(stdout);
}
int main(int argc, char **argv) {
    try {
        host();
        if (argc == 2 && !std::strcmp(argv[1], "--host"))
            return 0;
        auto *ctx = rk_npu_open(nullptr);
        CHECK(ctx);
        auto *domain = rk_npu_iommu_domain_create(ctx, 0);
        CHECK(domain);
        one(domain, 17, 480, 64, 480, 4, 64, 1, false);
        if (argc < 2 || std::strcmp(argv[1], "--canary")) {
            for (int k : {1, 33, 65, 97, 160, 192, 224})
                one(domain, 21, k, 65, 480, 4, 64, 1, true);
            one(domain, 1, 513, 193, 256, 4, 64, 7, true, 20);
            one(domain, 16, 513, 193, 256, 4, 64, 7, true);
            one(domain, 17, 513, 193, 256, 4, 64, 7, true);
            for (int mask : {2, 4, 3, 7})
                one(domain, 19, 1025, 257, 480, 4, 64, mask, true);
            one(domain, 65, 1921, 1025, 480, 64, 512, 7, false);
            one(domain, 129, 2401, 1025, 480, 128, 256, 7, true);
            one(domain, 257, 993, 257, 992, 256, 128, 7, true);
            one(domain, 129, 2049, 257, 2048, 128, 256, 7, true);
            one(domain, 21, 31 * 32, 65, 32, 4, 64, 1, true);
        }
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

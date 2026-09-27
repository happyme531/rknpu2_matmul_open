#include "../src/rk_npu_flatquant.h"
#include "../src/rk_npu_half_bits.h"
#include "rk_npu_w4a4_linear.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Header {
    char magic[8];
    int32_t K, N, L, R;
    float negative_clip, positive_clip;
};
using W = std::unique_ptr<rk_npu_w4a4_linear_weights, decltype(&rk_npu_w4a4_linear_weights_free)>;
using P =
    std::unique_ptr<rk_npu_w4a4_linear_workspace, decltype(&rk_npu_w4a4_linear_workspace_free)>;

template <class T> void read(std::ifstream &in, std::vector<T> &values) {
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(values.size() * sizeof(T)));
    if (!in)
        throw std::runtime_error("truncated input blob");
}

static std::vector<float> transformed(const std::vector<float> &a, int M, const Header &h,
                                      const std::vector<float> &left,
                                      const std::vector<float> &right) {
    std::vector<float> stage(size_t(M) * h.K), out(stage.size());
    for (int m = 0; m < M; ++m)
        for (int l = 0; l < h.L; ++l)
            for (int r = 0; r < h.R; ++r) {
                const float x = a[(size_t(m) * h.L + l) * h.R + r];
                for (int j = 0; j < h.R; ++j)
                    stage[(size_t(m) * h.L + l) * h.R + j] +=
                        x * rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(right[r * h.R + j]));
            }
    for (auto &v : stage)
        v = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(v));
    for (int m = 0; m < M; ++m)
        for (int o = 0; o < h.L; ++o)
            for (int l = 0; l < h.L; ++l) {
                const float x = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(left[l * h.L + o]));
                for (int r = 0; r < h.R; ++r)
                    out[(size_t(m) * h.L + o) * h.R + r] +=
                        x * stage[(size_t(m) * h.L + l) * h.R + r];
            }
    for (auto &v : out)
        v = rknpu2_matmul_open::bits::half_to_float(rknpu2_matmul_open::bits::float_to_half(v));
    return out;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s LINEAR.bin M [loops]\n", argv[0]);
        return 2;
    }
    try {
        const int M = std::stoi(argv[2]), loops = argc > 3 ? std::stoi(argv[3]) : 5;
        if (M < 1 || M > 256 || loops < 1)
            throw std::runtime_error("invalid M/loops");
        std::ifstream in(argv[1], std::ios::binary);
        Header h{};
        in.read(reinterpret_cast<char *>(&h), sizeof(h));
        if (!in || std::memcmp(h.magic, "W4A4FQ1", 7) || h.K != h.L * h.R)
            throw std::runtime_error("invalid input blob");
        std::vector<int8_t> b(size_t(h.K) * h.N);
        std::vector<float> ws(h.N), left(size_t(h.L) * h.L), right(size_t(h.R) * h.R);
        read(in, b);
        read(in, ws);
        read(in, left);
        read(in, right);

        auto *ctx = rk_npu_open(nullptr);
        if (!ctx)
            throw std::runtime_error("rk_npu_open failed");
        auto *domain = rk_npu_iommu_domain_create(ctx, 0);
        if (!domain)
            throw std::runtime_error("domain create failed");
        rk_npu_w4a4_transform tr;
        rk_npu_w4a4_transform_init(&tr);
        tr.left_dim = h.L;
        tr.right_dim = h.R;
        tr.left = left.data();
        tr.right = right.data();
        tr.negative_clip_ratio = h.negative_clip;
        tr.positive_clip_ratio = h.positive_clip;
        rk_npu_i4_weight_config wc{h.K, h.N, 2048};
        W weights(rk_npu_w4a4_linear_weights_create(domain, &wc, b.data(), ws.data(), &tr),
                  rk_npu_w4a4_linear_weights_free);
        if (!weights)
            throw std::runtime_error("weight creation failed (including INT16 bound)");
        rk_npu_w4a4_linear_config cfg;
        rk_npu_w4a4_linear_config_init(&cfg, M, h.N, h.K);
        cfg.k_tile = 2048;
        cfg.m_tile = std::min(M, 128);
        cfg.n_tile = 1408;
        cfg.npu_core_mask = 7;
        cfg.cpu_threads = M >= 16 ? 4 : 1;
        cfg.pipeline = 1;
        if (argc > 4)
            cfg.transform_batch = std::stoi(argv[4]);
        P workspace(rk_npu_w4a4_linear_workspace_create(domain, &cfg, weights.get()),
                    rk_npu_w4a4_linear_workspace_free);
        if (!workspace)
            throw std::runtime_error("workspace creation failed");

        std::vector<float> source(size_t(M) * h.K);
        std::vector<uint16_t> a(source.size()), c(size_t(M) * h.N);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = float(int((i * 131 + 17) % 2049) - 1024) / 512;
            a[i] = rknpu2_matmul_open::bits::float_to_half(source[i]);
            source[i] = rknpu2_matmul_open::bits::half_to_float(a[i]);
        }
        rk_npu_w4a4_linear_timings warm{};
        int rc =
            rk_npu_w4a4_linear_run_f16(workspace.get(), weights.get(), a.data(), c.data(), &warm);
        if (rc)
            throw std::runtime_error("warmup submit failed: " + std::to_string(rc));
        rk_npu_w4a4_linear_timings sum{};
        for (int i = 0; i < loops; ++i) {
            rk_npu_w4a4_linear_timings t{};
            rc = rk_npu_w4a4_linear_run_f16(workspace.get(), weights.get(), a.data(), c.data(), &t);
            if (rc)
                throw std::runtime_error("timed submit failed: " + std::to_string(rc));
            sum.transform_scan_us += t.transform_scan_us;
            sum.quant_pack_us += t.quant_pack_us;
            sum.sync_us += t.sync_us;
            sum.submit_us += t.submit_us;
            sum.reduce_dequant_us += t.reduce_dequant_us;
            sum.total_us += t.total_us;
            sum.transform_pack_us += t.transform_pack_us;
            sum.transform_submit_us += t.transform_submit_us;
            sum.activation_scan_us += t.activation_scan_us;
        }

        const auto x = transformed(source, M, h, left, right);
        std::vector<int> columns;
        if (M == 1) {
            columns.resize(h.N);
            std::iota(columns.begin(), columns.end(), 0);
        } else {
            columns = {0,       1,       7,           31,       63,      h.N / 7,
                       h.N / 3, h.N / 2, h.N * 2 / 3, h.N - 65, h.N - 2, h.N - 1};
        }
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        // Independently recompute integer dot products from the NPU-transformed
        // activation codes. This checks the complete Linear exactly while the
        // scalar FP16-boundary oracle below diagnoses floating MAC rounding.
        {
            rknpu2_matmul_open::detail::FlatWeights fw;
            if (fw.prepare(domain, h.L, h.R, left.data(), right.data()))
                throw std::runtime_error("oracle weights");
            rknpu2_matmul_open::detail::FlatPlan fp;
            auto oracle_cfg = cfg;
            oracle_cfg.transform_batch = 32;
            if (fp.prepare(domain, oracle_cfg, fw.shape()))
                throw std::runtime_error("oracle plan");
            std::vector<uint8_t> codes(size_t(M) * ((h.K + 31) / 32) * 16);
            std::vector<float> token_scale(M), inverse(M);
            rknpu2_matmul_open::detail::FlatMetrics oracle_times;
            if (fp.run(fw, a.data(), true, h.negative_clip, h.positive_clip, codes.data(),
                       token_scale.data(), inverse.data(), oracle_times))
                throw std::runtime_error("oracle transform");
            for (int m = 0; m < M; ++m)
                for (int n : columns) {
                    int64_t dot = 0;
                    for (int k = 0; k < h.K; ++k) {
                        size_t i = (size_t(k / 32) * M + m) * 16 + (k % 32) / 2;
                        int q = (codes[i] >> ((k & 1) ? 0 : 4)) & 15;
                        if (q >= 8)
                            q -= 16;
                        dot += q * int(b[size_t(k) * h.N + n]);
                    }
                    if (c[size_t(m) * h.N + n] !=
                        rknpu2_matmul_open::bits::float_to_half((float(dot) * token_scale[m]) * ws[n]))
                        throw std::runtime_error("exact integer oracle mismatch");
                }
        }
        size_t mismatches = 0;
        double error2 = 0, reference2 = 0, max_error = 0;
        for (int m = 0; m < M; ++m) {
            float xmin = 0, xmax = 0;
            for (int k = 0; k < h.K; ++k) {
                xmin = std::min(xmin, x[size_t(m) * h.K + k]);
                xmax = std::max(xmax, x[size_t(m) * h.K + k]);
            }
            const float scale = std::max(-xmin * h.negative_clip, xmax * h.positive_clip) / 7;
            for (int n : columns) {
                int64_t dot = 0;
                for (int k = 0; k < h.K; ++k) {
                    const int q = int(std::clamp(
                        std::nearbyint(x[size_t(m) * h.K + k] * (1.0f / scale)), -8.f, 7.f));
                    dot += q * int(b[size_t(k) * h.N + n]);
                }
                const uint16_t expected = rknpu2_matmul_open::bits::float_to_half((float(dot) * scale) * ws[n]);
                float actual = rknpu2_matmul_open::bits::half_to_float(c[size_t(m) * h.N + n]);
                float ref = rknpu2_matmul_open::bits::half_to_float(expected);
                double error = double(actual) - ref;
                error2 += error * error;
                reference2 += double(ref) * ref;
                max_error = std::max(max_error, std::abs(error));
                if (c[size_t(m) * h.N + n] != expected) {
                    if (mismatches < 4)
                        std::printf("reference_difference m=%d n=%d expected=%.9g actual=%.9g\n", m,
                                    n, ref, actual);
                    ++mismatches;
                }
            }
        }
        std::printf("reference_fp16_boundary mismatches=%zu relative_l2=%.9g "
                    "max_abs=%.9g\n",
                    mismatches, std::sqrt(error2 / std::max(reference2, 1e-30)), max_error);
        if (!std::isfinite(error2) || std::sqrt(error2 / std::max(reference2, 1e-30)) > .005)
            throw std::runtime_error("sampled output exceeds FP16-transform reference tolerance");
        const double d = loops;
        std::printf("PASS FlatQuant W4A4 K=%d N=%d transform=%dx%d M=%d loops=%d "
                    "sampled=%zu\n",
                    h.K, h.N, h.L, h.R, M, loops, size_t(M) * columns.size());
        std::printf("avg_ms total=%.6f transform_scan=%.6f quant_pack=%.6f "
                    "submit=%.6f reduce_dequant=%.6f sync=%.6f\n",
                    sum.total_us / d / 1000, sum.transform_scan_us / d / 1000,
                    sum.quant_pack_us / d / 1000, sum.submit_us / d / 1000,
                    sum.reduce_dequant_us / d / 1000, sum.sync_us / d / 1000);
        std::printf("transform_ms input_pack=%.6f submit=%.6f scan=%.6f "
                    "exact_integer_oracle=PASS\n",
                    sum.transform_pack_us / d / 1000, sum.transform_submit_us / d / 1000,
                    sum.activation_scan_us / d / 1000);
        workspace.reset();
        weights.reset();
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(ctx);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

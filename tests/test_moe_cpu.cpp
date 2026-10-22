#include "../src/rk_npu_moe_cpu.h"
#include "../src/rk_npu_swiglu_fp32_neon.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
namespace moe = rknpu2_matmul_open::moe;
namespace detail = rknpu2_matmul_open::detail;

static void one(int rows, int channels, int waves, bool w4, moe::MiddleLayout layout, bool zero) {
    const size_t stride = size_t(rows) * 2 * channels;
    std::vector<int32_t> native(stride * waves), correction(2 * channels), sums(stride);
    std::vector<float> scales(2 * channels), gate(channels), up(channels), hidden(channels);
    for (int n = 0; n < 2 * channels; ++n) {
        correction[n] = zero ? 0 : n % 101 - 50;
        scales[n] = .002f + float(n % 11) * .00017f;
    }
    for (int row = 0; row < rows; ++row) for (int n = 0; n < 2 * channels; ++n) {
        int32_t total = w4 ? correction[n] : 0;
        for (int wave = 0; wave < waves; ++wave) {
            const int hi = zero ? 0 : (n * 13 + row * 17 + wave * 19) % 601 - 300;
            const int lo = zero ? 0 : (n * 11 + row * 7 + wave * 3) % 1001 - 500;
            if (w4) {
                auto* p = reinterpret_cast<int16_t*>(native.data() + wave * stride);
                const size_t offset = size_t(n / 8) * rows * 16 + row * 16 + n % 8;
                p[offset] = int16_t(hi); p[offset + 8] = int16_t(lo);
                total += 16 * hi + lo;
            } else {
                const int32_t value = hi * 13 + lo;
                native[size_t(n / 4) * rows * 4 + row * 4 + n % 4] = value;
                total += value;
            }
        }
        sums[size_t(row) * 2 * channels + n] = total;
    }
    for (int row = 0; row < rows; ++row) {
        const float sa = .015f + row * .0001f;
        for (int n = 0; n < channels; ++n) {
            gate[n] = (float(sums[size_t(row) * 2 * channels + n]) * sa) * scales[n];
            up[n] = (float(sums[size_t(row) * 2 * channels + channels + n]) * sa) * scales[channels + n];
        }
#if defined(__aarch64__)
        for (int n = 0; n < channels; n += 16)
            detail::swiglu_fp32::apply16<false>(gate.data() + n, up.data() + n, hidden.data() + n);
#else
        for (int n = 0; n < channels; ++n) hidden[n] = (gate[n] / (1.f + std::exp(-gate[n]))) * up[n];
#endif
        float maximum = 0;
        for (float v : hidden) maximum = std::max(maximum, std::fabs(v));
        const float expected_scale = maximum ? maximum / 127.f : 1.f;
        std::vector<uint8_t> output(size_t(rows) * channels + 64, 0x5a);
        auto* base = output.data() + 32;
        const moe::MiddleRow input{native.data(), w4 ? correction.data() : nullptr, scales.data(),
            sa, rows, row, channels, waves, stride * sizeof(int32_t), reinterpret_cast<int8_t*>(base), layout};
        const float actual_scale = moe::swiglu_quantize_row(input);
        CHECK(!std::memcmp(&expected_scale, &actual_scale, sizeof(float)));
        std::vector<bool> written(size_t(rows) * channels);
        for (int n = 0; n < channels; ++n) {
            const int expected = int(std::lrint(std::clamp(hidden[n] * (1.f / expected_scale), -127.f, 127.f)));
            int got;
            if (layout == moe::MiddleLayout::I4Native) {
                const size_t offset = size_t(n / 32) * rows * 32 + row * 32 + (n % 32) / 2;
                const int shift = n % 2 ? 0 : 4;
                const int h = ((base[offset] >> shift) & 15) ^ 8;
                const int l = ((base[offset + 16] >> shift) & 15) ^ 8;
                got = 16 * (h - 8) + (l - 8) + 8;
                written[offset] = written[offset + 16] = true;
            } else {
                const size_t offset = layout == moe::MiddleLayout::I8Native
                    ? size_t(n / 16) * rows * 16 + row * 16 + n % 16 : size_t(row) * channels + n;
                got = int8_t(base[offset]); written[offset] = true;
            }
            if (got != expected) {
                std::fprintf(stderr, "rows=%d channels=%d waves=%d w4=%d layout=%d zero=%d row=%d col=%d got=%d expected=%d hidden=%a scale=%a\n",
                    rows, channels, waves, int(w4), int(layout), int(zero), row, n, got, expected,
                    double(hidden[n]), double(expected_scale));
                CHECK(got == expected);
            }
        }
        for (size_t i = 0; i < written.size(); ++i) if (!written[i]) CHECK(base[i] == 0x5a);
        for (int i = 0; i < 32; ++i) CHECK(output[i] == 0x5a && output[output.size() - 1 - i] == 0x5a);
    }
}

int main() try {
    for (int rows : {1, 5, 33}) for (int channels : {32, 64, 512})
        for (auto layout : {moe::MiddleLayout::I8Normal, moe::MiddleLayout::I8Native, moe::MiddleLayout::I4Native})
            for (bool zero : {false, true}) {
                one(rows, channels, 1, false, layout, zero);
                for (int waves : {1, 2, 4}) one(rows, channels, waves, true, layout, zero);
            }
    std::puts("PASS MoE middle: W8/W4 multiwave, exact scales/codes, three layouts, zero, row guards");
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }

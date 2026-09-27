#include "rk_npu_dcomp.h"
#include "rk_npu_common.h"
#include "rk_npu_cpu_kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace rknpu2_matmul_open::detail {
namespace {
constexpr unsigned patterns[8] = {0, 1, 7, 3, 6, 10, 18, 2};
constexpr unsigned lengths[8] = {2, 2, 3, 3, 3, 4, 5, 5};
size_t align64(size_t x) { return (x + 63) & ~size_t(63); }
bool valid_shape(int k, int n) {
    return k > 0 && n > 0 && k % 32 == 0 && n % 32 == 0 &&
           uint64_t(k) * n <= std::numeric_limits<size_t>::max() / 16;
}
unsigned magnitude(unsigned residual) { return residual < 128 ? residual : 255 - residual; }
unsigned width(unsigned mag) { return mag ? 32u - __builtin_clz(mag) : 0; }
template <class Fn> void each_lane(int k, int n, int lane, Fn fn) {
    const int parity = lane / 8, quad = lane % 8;
    for (int nb = 0; nb < n / 32; ++nb)
        for (int kb = 0; kb < k / 32; ++kb)
            for (int half = 0; half < 2; ++half)
                for (int group = 0; group < 8; ++group) {
                    const size_t base = (size_t(nb) * (k / 32) + kb) * 1024 + half * 512 +
                                        (group * 2 + parity) * 32 + quad * 4;
                    for (int j = 0; j < 4; ++j)
                        fn(base + j);
                }
}
void put(uint8_t *data, size_t &pos, unsigned value, unsigned bits) {
    while (bits) {
        const unsigned take = std::min(bits, 8u - unsigned(pos & 7));
        data[pos >> 3] |= uint8_t((value & ((1u << take) - 1)) << (pos & 7));
        pos += take;
        value >>= take;
        bits -= take;
    }
}
// Deterministic nearest-even, independent of the caller's FP rounding mode.
int nearest_even(float x) {
    const int lo = int(std::floor(x));
    const float frac = x - float(lo);
    return lo + (frac > 0.5f || (frac == 0.5f && (lo & 1)));
}
} // namespace

int dcomp_encode(const int8_t *native, int k, int n, DcompEncoded &out, bool count_only) {
    if (!native || !valid_shape(k, n))
        return RK_NPU_ERR_PARAM;
    out = {};
    for (int lane = 0; lane < 16; ++lane) {
        std::array<uint64_t, 256> hist{};
        each_lane(k, n, lane, [&](size_t i) { ++hist[uint8_t(native[i])]; });
        unsigned modal = unsigned(std::max_element(hist.begin(), hist.end()) - hist.begin());
        std::array<uint64_t, 8> whist{};
        for (unsigned b = 0; b < 256; ++b)
            whist[width(magnitude(uint8_t(b - modal)))] += hist[b];
        std::array<unsigned, 8> book{}, inverse{};
        for (unsigned i = 0; i < 8; ++i)
            book[i] = i;
        std::sort(book.begin(), book.end(), [&](unsigned a, unsigned b) {
            return whist[a] != whist[b] ? whist[a] > whist[b] : a < b;
        });
        for (unsigned i = 0; i < 8; ++i)
            inverse[book[i]] = i;
        size_t bits = 128;
        for (unsigned w = 0; w < 8; ++w)
            bits += whist[w] * (lengths[inverse[w]] + 1 + (w > 1 ? w - 1 : 0));
        const size_t blocks = std::max<size_t>(8, (bits + 127) / 128);
        if (blocks - 1 > UINT32_MAX)
            return RK_NPU_ERR_PARAM;
        out.amounts[lane] = uint32_t(blocks - 1);
        const size_t offset = out.data.size(), bytes = align64(blocks * 16);
        out.data.resize(offset + bytes, 0);
        if (count_only)
            continue;
        auto *dst = out.data.data() + offset;
        dst[1] = uint8_t(modal);
        for (unsigned i = 0; i < 8; ++i)
            dst[4 + i / 2] |= uint8_t(book[i] << ((i & 1) * 4));
        unsigned symbol[256], nbits[256];
        for (unsigned b = 0; b < 256; ++b) {
            const unsigned r = uint8_t(b - modal), mag = magnitude(r), w = width(mag);
            const unsigned idx = inverse[w], prefix = lengths[idx];
            symbol[b] = patterns[idx] | ((r >= 128) << prefix);
            if (w > 1)
                symbol[b] |= (mag & ((1u << (w - 1)) - 1)) << (prefix + 1);
            nbits[b] = prefix + 1 + (w > 1 ? w - 1 : 0);
        }
        size_t pos = 128;
        each_lane(k, n, lane, [&](size_t i) {
            const auto b = uint8_t(native[i]);
            put(dst, pos, symbol[b], nbits[b]);
        });
    }
    return RK_NPU_OK;
}

int dcomp_decode(const uint8_t *data, size_t bytes, const std::array<uint32_t, 16> &amounts, int k,
                 int n, std::vector<int8_t> &native) {
    if (!data || !valid_shape(k, n))
        return RK_NPU_ERR_PARAM;
    native.resize(size_t(k) * n);
    size_t offset = 0;
    for (int lane = 0; lane < 16; ++lane) {
        const uint64_t valid_bytes = (uint64_t(amounts[lane]) + 1) * 16;
        if (valid_bytes < 128 || valid_bytes > bytes - offset)
            return RK_NPU_ERR_PARAM;
        const size_t lane_bytes = align64(size_t(valid_bytes));
        if (lane_bytes > bytes - offset)
            return RK_NPU_ERR_PARAM;
        const auto *src = data + offset;
        if (src[0] != 0)
            return RK_NPU_ERR_PARAM;
        unsigned book[8], seen = 0;
        for (unsigned i = 0; i < 8; ++i) {
            book[i] = (src[4 + i / 2] >> ((i & 1) * 4)) & 15;
            if (book[i] > 7 || (seen & (1u << book[i])))
                return RK_NPU_ERR_PARAM;
            seen |= 1u << book[i];
        }
        struct Symbol {
            uint8_t value, bits;
        } table[4096];
        for (unsigned bits = 0; bits < 4096; ++bits) {
            unsigned idx = 0;
            while (idx < 8 && (bits & ((1u << lengths[idx]) - 1)) != patterns[idx])
                ++idx;
            if (idx == 8)
                return RK_NPU_ERR_PARAM;
            const unsigned prefix = lengths[idx], w = book[idx];
            const unsigned sign = (bits >> prefix) & 1;
            const unsigned low = w > 1 ? (bits >> (prefix + 1)) & ((1u << (w - 1)) - 1) : 0;
            const unsigned mag = w ? (1u << (w - 1)) | low : 0;
            const int residual = sign ? -int(mag) - 1 : int(mag);
            table[bits] = {uint8_t(residual + src[1]), uint8_t(prefix + 1 + (w > 1 ? w - 1 : 0))};
        }
        bool ok = true;
        size_t pos = 128;
        each_lane(k, n, lane, [&](size_t i) {
            if (!ok)
                return;
            const size_t byte = pos >> 3;
            if (byte >= lane_bytes) {
                ok = false;
                return;
            }
            unsigned bits = src[byte];
            if (byte + 1 < lane_bytes)
                bits |= unsigned(src[byte + 1]) << 8;
            if (byte + 2 < lane_bytes)
                bits |= unsigned(src[byte + 2]) << 16;
            const auto symbol = table[(bits >> (pos & 7)) & 4095];
            if (pos + symbol.bits > valid_bytes * 8) {
                ok = false;
                return;
            }
            pos += symbol.bits;
            native[i] = int8_t(symbol.value < 128 ? int(symbol.value) : int(symbol.value) - 256);
        });
        if (!ok)
            return RK_NPU_ERR_PARAM;
        offset += lane_bytes;
    }
    return offset == bytes ? RK_NPU_OK : RK_NPU_ERR_PARAM;
}

int dcomp_quantize_f32(int K, int N, int ktile, const float *w, float target,
                       std::vector<int8_t> &codes, std::vector<float> &scales) {
    if (!w || K <= 0 || N <= 0 || K > INT32_MAX - 32 || N > INT32_MAX - 32 || ktile <= 0 ||
        ktile > K || !std::isfinite(target) || target <= 0 || target > 8)
        return RK_NPU_ERR_PARAM;
    std::vector<float> maxabs(N, 0), candidate_scales(N);
    for (size_t i = 0; i < size_t(K) * N; ++i) {
        if (!std::isfinite(w[i]))
            return RK_NPU_ERR_PARAM;
        maxabs[i % N] = std::max(maxabs[i % N], std::abs(w[i]));
    }
    struct Result {
        bool done = false;
        double bpw = 0, error = 0;
    } results[128];
    std::vector<int8_t> candidate(size_t(K) * N), packed;
    auto evaluate = [&](int limit) -> Result & {
        auto &result = results[limit];
        if (result.done)
            return result;
        result.done = true;
        for (int n = 0; n < N; ++n) {
            candidate_scales[n] =
                maxabs[n] == 0 ? 1.0f
                               : std::max(maxabs[n] / limit, std::numeric_limits<float>::min());
        }
        for (size_t i = 0; i < candidate.size(); ++i) {
            const float s = candidate_scales[i % N];
            const float z = std::max(-float(limit), std::min(float(limit), w[i] / s));
            candidate[i] = int8_t(nearest_even(z));
            const double e = double(w[i]) - double(s) * candidate[i];
            result.error += e * e;
        }
        uint64_t bytes = 0;
        const int an = (N + 31) / 32 * 32;
        for (int k0 = 0; k0 < K;) {
            const int kt = std::min(ktile, K - k0), ak = (kt + 31) / 32 * 32;
            packed.resize(size_t(ak) * an);
            rknpu2_matmul_open::cpu::i8_pack_b_native_n32_k32(kt, N, ak, an, candidate.data() + size_t(k0) * N,
                                            packed.data());
            DcompEncoded encoded;
            if (dcomp_encode(packed.data(), ak, an, encoded, true) != RK_NPU_OK) {
                result.bpw = std::numeric_limits<double>::infinity();
                return result;
            }
            bytes += encoded.data.size();
            k0 += kt;
        }
        result.bpw = double(bytes) * 8 / (double(K) * N);
        return result;
    };
    // Codec rate is not strictly monotonic. This is a bounded heuristic search,
    // not an optimality or infeasibility proof; all accepted rates are exact.
    int low = 1, high = 127;
    evaluate(low);
    evaluate(high);
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (evaluate(mid).bpw <= target)
            low = mid;
        else
            high = mid;
    }
    for (int c = std::max(1, low - 2); c <= std::min(127, high + 2); ++c)
        evaluate(c);
    int best = 0;
    for (int c = 1; c <= 127; ++c)
        if (results[c].done && results[c].bpw <= target &&
            (!best || results[c].error < results[best].error))
            best = c;
    if (!best)
        return RK_NPU_ERR_PARAM;
    scales.resize(N);
    codes.resize(candidate.size());
    for (int n = 0; n < N; ++n)
        scales[n] =
            maxabs[n] == 0 ? 1.0f : std::max(maxabs[n] / best, std::numeric_limits<float>::min());
    for (size_t i = 0; i < codes.size(); ++i)
        codes[i] = int8_t(
            nearest_even(std::max(-float(best), std::min(float(best), w[i] / scales[i % N]))));
    return RK_NPU_OK;
}
} // namespace rknpu2_matmul_open::detail

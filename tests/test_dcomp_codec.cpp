#include "../src/rk_npu_cpu_kernels.h"
#include "../src/rk_npu_dcomp.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                              \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
template <class T> bool read_file(const char *path, std::vector<T> &v) {
    std::ifstream f(path, std::ios::binary);
    return bool(f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T)));
}
template <class T> bool write_file(const std::string &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    return bool(f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T)));
}
int main(int argc, char **argv) {
    if (argc == 7 && std::string(argv[1]) == "--rate") {
        int k = std::atoi(argv[2]), n = std::atoi(argv[3]), ktile = std::atoi(argv[4]),
            ntile = std::atoi(argv[5]);
        CHECK(k > 0 && n > 0 && ktile > 0 && ntile > 0 && ntile % 32 == 0);
        std::vector<int8_t> v(size_t(k) * n), packed;
        CHECK(read_file(argv[6], v));
        uint64_t bytes = 0;
        unsigned raw = 0, compressed = 0;
        int an = (n + 31) / 32 * 32;
        for (int k0 = 0; k0 < k; k0 += ktile) {
            int kt = std::min(ktile, k - k0), ak = (kt + 31) / 32 * 32;
            packed.resize(size_t(ak) * an);
            rknpu2_matmul_open::cpu::i8_pack_b_native_n32_k32(kt, n, ak, an, v.data() + size_t(k0) * n,
                                            packed.data());
            for (int n0 = 0; n0 < an; n0 += ntile) {
                int nt = std::min(ntile, an - n0);
                rknpu2_matmul_open::detail::DcompEncoded e;
                CHECK(rknpu2_matmul_open::detail::dcomp_encode(packed.data() + size_t(n0) * ak, ak, nt, e, true) == 0);
                size_t raw_bytes = size_t(ak) * nt;
                bytes += std::min(raw_bytes, e.data.size());
                if (e.data.size() < raw_bytes)
                    ++compressed;
                else
                    ++raw;
            }
        }
        std::printf("{\"payload_bytes\":%llu,\"payload_bpw\":%.9f,\"scale_bytes\":%llu,"
                    "\"compressed_tiles\":%u,\"raw_tiles\":%u}\n",
                    (unsigned long long)bytes, double(bytes) * 8 / (double(k) * n),
                    (unsigned long long)n * 4, compressed, raw);
        return 0;
    }
    if (argc == 7 && std::string(argv[1]) == "--encode") {
        int k = std::atoi(argv[2]), n = std::atoi(argv[3]);
        CHECK(k > 0 && n > 0);
        std::vector<int8_t> v(size_t(k) * n), decoded;
        CHECK(read_file(argv[4], v));
        rknpu2_matmul_open::detail::DcompEncoded e;
        CHECK(rknpu2_matmul_open::detail::dcomp_encode(v.data(), k, n, e) == 0);
        CHECK(rknpu2_matmul_open::detail::dcomp_decode(e.data.data(), e.data.size(), e.amounts, k, n, decoded) == 0 &&
              v == decoded);
        CHECK(write_file(argv[5], e.data));
        std::vector<uint32_t> a(e.amounts.begin(), e.amounts.end());
        CHECK(write_file(argv[6], a));
        return 0;
    }
    if (argc == 8 && std::string(argv[1]) == "--quantize") {
        int k = std::atoi(argv[2]), n = std::atoi(argv[3]), kt = std::atoi(argv[4]);
        CHECK(k > 0 && n > 0);
        std::vector<float> w(size_t(k) * n), scales;
        std::vector<int8_t> codes;
        CHECK(read_file(argv[5], w));
        CHECK(rknpu2_matmul_open::detail::dcomp_quantize_f32(k, n, kt, w.data(), std::strtof(argv[6], nullptr), codes,
                                        scales) == 0);
        CHECK(write_file(std::string(argv[7]) + ".codes.bin", codes));
        CHECK(write_file(std::string(argv[7]) + ".scales.bin", scales));
        return 0;
    }
    CHECK(argc == 1);
    std::mt19937 rng(20260913);
    int cases = 0;
    for (int k : {32, 64, 96, 256})
        for (int n : {32, 64, 160}) {
            for (int mode = 0; mode < 8; ++mode) {
                std::vector<int8_t> v(size_t(k) * n), decoded;
                for (size_t i = 0; i < v.size(); ++i) {
                    const int val = mode == 0   ? 0
                                    : mode == 1 ? -128
                                    : mode == 2 ? 127
                                    : mode == 3 ? int(i % 256) - 128
                                    : mode == 4 ? int(rng() % 256) - 128
                                    : mode == 5 ? int(rng() % 31) - 15
                                    : mode == 6 ? (i % 101 ? 3 : -127)
                                                : (i % 2 ? -1 : 0);
                    v[i] = int8_t(val);
                }
                rknpu2_matmul_open::detail::DcompEncoded e, count;
                CHECK(rknpu2_matmul_open::detail::dcomp_encode(v.data(), k, n, e) == 0);
                CHECK(rknpu2_matmul_open::detail::dcomp_encode(v.data(), k, n, count, true) == 0);
                CHECK(count.data.size() == e.data.size() && count.amounts == e.amounts);
                CHECK(rknpu2_matmul_open::detail::dcomp_decode(e.data.data(), e.data.size(), e.amounts, k, n, decoded) ==
                      0);
                CHECK(decoded == v);
                CHECK(rknpu2_matmul_open::detail::dcomp_decode(e.data.data(), e.data.size() - 1, e.amounts, k, n,
                                          decoded) != 0);
                auto bad = e;
                bad.data[0] = 1;
                CHECK(rknpu2_matmul_open::detail::dcomp_decode(bad.data.data(), bad.data.size(), bad.amounts, k, n,
                                          decoded) != 0);
                bad = e;
                bad.data[4] = 0;
                CHECK(rknpu2_matmul_open::detail::dcomp_decode(bad.data.data(), bad.data.size(), bad.amounts, k, n,
                                          decoded) != 0);
                ++cases;
            }
        }
    std::vector<float> w(256 * 128), scales;
    std::vector<int8_t> codes;
    for (auto &x : w)
        x = float(int(rng() % 10001) - 5000) / 3000;
    for (int k = 0; k < 256; ++k)
        w[k * 128] = 0;
    CHECK(rknpu2_matmul_open::detail::dcomp_quantize_f32(256, 128, 128, w.data(), 6.5f, codes, scales) == 0);
    CHECK(scales[0] == 1);
    for (int k = 0; k < 256; ++k)
        CHECK(codes[k * 128] == 0);
    CHECK(rknpu2_matmul_open::detail::dcomp_quantize_f32(256, 128, 128, w.data(), 0.01f, codes, scales) != 0);
    CHECK(rknpu2_matmul_open::detail::dcomp_quantize_f32(256, 128, 128, w.data(), 0, codes, scales) != 0);
    w[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(rknpu2_matmul_open::detail::dcomp_quantize_f32(256, 128, 128, w.data(), 6.5f, codes, scales) != 0);
    std::printf("PASS codec: %d full-byte roundtrips, malformed streams, RTN validation\n", cases);
    return 0;
}

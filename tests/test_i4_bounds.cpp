#include "../src/rk_npu_i4_bounds.h"
#include "rk_npu_matmul_i4.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error(#x);                                                          \
    } while (0)
int main() {
    try {
        rknpu2_matmul_open::detail::I4WeightBound b;
        std::vector<int8_t> w(2048, -8);
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 0, b));
        CHECK(b.k_tile == 480);
        CHECK(rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 512, b) == RK_NPU_ERR_PARAM);
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(511, 1, w.data(), 512, b));
        CHECK(b.positive == 32704);
        std::fill(w.begin(), w.end(), 4);
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 0, b));
        CHECK(b.k_tile == 1024 && b.negative_magnitude == 32768);
        std::fill(w.begin(), w.end(), 0);
        std::fill(w.begin() + 624, w.begin() + 1424, -8);
        CHECK(rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 768, b) == RK_NPU_ERR_PARAM);
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 1024, b));
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(2048, 1, w.data(), 0, b));
        CHECK(b.k_tile == 1120);
        w.assign(2048 * 129, 0);
        for (int k = 0; k < 2048; ++k)
            w[k * 129 + 128] = -8;
        CHECK(!rknpu2_matmul_open::detail::bound_i4_weights(2048, 129, w.data(), 0, b));
        CHECK(b.k_tile == 480);
        w.back() = 8;
        CHECK(rknpu2_matmul_open::detail::bound_i4_weights(2048, 129, w.data(), 0, b) == RK_NPU_ERR_PARAM);
        std::puts("PASS weight bounds: exact signed limits, K tail, non-monotonic partitions, last "
                  "column, invalid code");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

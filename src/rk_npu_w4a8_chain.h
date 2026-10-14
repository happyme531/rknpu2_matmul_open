#pragma once
#include "rk_npu_w4a8_cpu.h"
#include <memory>

namespace rknpu2_matmul_open::detail {
// Complete packed A and wave-major native partials. Each selected core owns
// one PC chain over all K waves; CPU ownership transfers only at the edges.
class W4A8Chain {
public:
    W4A8Chain();
    ~W4A8Chain();
    static int query(const rk_npu_i4_config&, I4InputLayout, rk_npu_i4_memory_info&);
    int prepare(rk_npu_iommu_domain*, const rk_npu_i4_config&, I4InputLayout);
    int run(const I4Weights&, const W4A8Input&, const W4A8Reduction&, rk_npu_i4_timings&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

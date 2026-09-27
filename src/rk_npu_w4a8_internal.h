#pragma once
#include "rk_npu_w4a8.h"
namespace rknpu2_matmul_open::detail {
// Only internal callers with a checked per-weight policy may relax the bound.
rk_npu_w4a8_weights* create_w4a8_weights(rk_npu_iommu_domain*,
    const rk_npu_w4a8_weight_info&, const int8_t*, const float*);
}

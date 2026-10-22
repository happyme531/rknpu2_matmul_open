#pragma once
#include "rk_npu_moe_w8.h"
// Experimental mixed MoE: routed W4 codes in int8_t, shared W8.
// Uses the existing run/free/timing functions. H/I must be multiples of 64;
// routed gate/up uses certified K<=512 partials, shared W8 uses full K.
// CPU middle, normal shared input only. Original weight codes/scales retained.
// No new installed API until complete-model performance is established.
extern "C" rk_npu_moe_w8_weights* rk_npu_moe_w4_weights_create(
    rk_npu_iommu_domain*, const rk_npu_moe_w8_weight_config*, const rk_npu_moe_w8_expert*);
extern "C" rk_npu_moe_w8_workspace* rk_npu_moe_w4_workspace_create(
    rk_npu_iommu_domain*, const rk_npu_moe_w8_config*);

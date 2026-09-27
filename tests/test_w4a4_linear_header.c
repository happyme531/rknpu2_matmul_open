#include "rk_npu_w4a4_linear.h"

void w4a4_linear_header_is_c(void) {
    rk_npu_w4a4_linear_config cfg;
    rk_npu_w4a4_transform transform;
    rk_npu_w4a4_linear_memory_info memory;
    rk_npu_w4a4_linear_config_init(&cfg, 1, 64, 32);
    rk_npu_w4a4_transform_init(&transform);
    (void)rk_npu_w4a4_linear_memory_query(&cfg, &transform, &memory);
}

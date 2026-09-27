#include "rk_npu_w4a8.h"
#include "rk_npu_w4a8_tune.h"
int w4a8_header_is_c(void) {
    rk_npu_w4a8_config cfg;
    rk_npu_w4a8_memory_info info;
    rk_npu_w4a8_config_init(&cfg, 1, 64, 32);
    rk_npu_w4a8_strategy_ex strategy = {0};
    rk_npu_w4a8_options_init(&strategy.execution);
    rk_npu_w4a8_options options;
    rk_npu_w4a8_memory_info_ex extended;
    rk_npu_w4a8_options_init(&options);
    return rk_npu_w4a8_memory_query(&cfg, &info) |
        rk_npu_w4a8_memory_query_ex(&cfg, &options, &extended);
}

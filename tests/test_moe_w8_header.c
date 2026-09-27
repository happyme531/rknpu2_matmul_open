#include "rk_npu_moe_w8.h"
void moe_header_contract(void) {
    rk_npu_moe_w8_config config;
    rk_npu_moe_w8_config_init(&config,1536,512,128,1,8,256);
    config.middle=RK_NPU_MOE_MIDDLE_NPU_LUT;
    (void)config;
}

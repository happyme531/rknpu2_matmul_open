#include "rk_npu_attention_f16.h"
void attention_header_c(void){
    rk_npu_attention_f16_config c;
    rk_npu_attention_f16_config_init(&c);
    c.flags=RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL;
    c.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
    c.query_heads=12;c.kv_heads=4;c.head_dim=96;c.softmax_scale=1.f/96;
    rk_npu_attention_f16_boolean_mask mask;
    uint8_t visible=1;
    rk_npu_attention_f16_boolean_mask_init(&mask,&visible,1,1,1,1);
}

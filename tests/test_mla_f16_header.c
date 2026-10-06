#include "rk_npu_mla_f16.h"
int mla_header_contract(void){
    rk_npu_mla_f16_config c;rk_npu_mla_f16_config_init(&c);
    rk_npu_mla_f16_sizes s;return rk_npu_mla_f16_query(&c,4096,1,4096,&s);
}

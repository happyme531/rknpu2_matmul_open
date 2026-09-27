#include "rk_npu_matmul_i4.h"
int i4_header_is_c(void) {
    rk_npu_i4_config cfg;
    rk_npu_i4_memory_info sizes;
    rk_npu_i4_config_init(&cfg,1,64,32);
    return rk_npu_i4_memory_query(&cfg,&sizes);
}

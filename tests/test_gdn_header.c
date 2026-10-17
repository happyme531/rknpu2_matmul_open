#include "rk_npu_gdn.h"
int rk_npu_gdn_header_c(void) {
    rk_npu_gdn_config config;
    rk_npu_gdn_sizes sizes;
    rk_npu_gdn_config_init(&config,128);
    return rk_npu_gdn_query(&config,&sizes);
}

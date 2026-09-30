#include "rk_npu_mm.h"
#include "rk_npu_matmul_f16.h"
#include "rk_npu_quant_matmul.h"

int rk_mm_header_smoke(void) {
    rk_npu_mm_desc d;
    rk_npu_mm_options o;
    rk_npu_mm_plan* p = 0;
    rk_npu_mm_info i;
    rk_npu_mm_desc_init(&d, 3, 5, 7, 2);
    rk_npu_mm_options_init(&o);
    int rc = rk_npu_mm_plan_create(&d, &o, &p);
    if (!rc) rc = rk_npu_mm_plan_get_info(p, &i);
    rk_npu_mm_plan_free(p);
    return rc;
}

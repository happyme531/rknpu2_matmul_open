#ifndef RK_NPU_FLOAT_REDUCE_H
#define RK_NPU_FLOAT_REDUCE_H
#include "rk_npu_common.h"
#include <memory>
namespace rknpu2_matmul_open::detail {
// Device-resident FP16 pairwise tree over identically packed partial slabs.
// Partial storage must outlive this object; run is blocking/non-reentrant.
class FloatReduce {
public:
    FloatReduce();
    ~FloatReduce();
    int prepare(rk_npu_iommu_domain*, const rk_npu_mem& partials,
                int count, uint64_t slab_bytes, uint32_t core_mask, uint32_t timeout_ms);
    int run();
    const rk_npu_mem& output() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
#endif

#include "rk_npu_matmul_i4.h"
#include "rk_npu_i4_executor.h"
#include <memory>
#include <new>
#include <system_error>

struct rk_npu_i4i4i32_weights { rknpu2_matmul_open::detail::I4Weights impl; };
struct rk_npu_i4_workspace { rknpu2_matmul_open::detail::I4Executor impl; };

extern "C" void rk_npu_i4_config_init(rk_npu_i4_config* cfg, int M, int N, int K) {
    if (cfg) *cfg={M,N,K,128,480,1024,1,500,1,1};
}
extern "C" int rk_npu_i4_memory_query(const rk_npu_i4_config* cfg, rk_npu_i4_memory_info* out) {
    if (!cfg || !out) return RK_NPU_ERR_PARAM;
    return rknpu2_matmul_open::detail::query_i4(*cfg,*out);
}
extern "C" int rk_npu_i4_weights_memory_query(const rk_npu_i4_weight_config* cfg, uint64_t* bytes) {
    if (!cfg || !bytes) return RK_NPU_ERR_PARAM;
    return rknpu2_matmul_open::detail::query_i4_weights(*cfg,*bytes);
}
extern "C" rk_npu_i4i4i32_weights* rk_npu_i4i4i32_weights_create(
    rk_npu_iommu_domain* domain, const rk_npu_i4_weight_config* cfg, const int8_t* B) {
    if (!domain || !cfg || !B) return nullptr;
    try {
        auto w=std::make_unique<rk_npu_i4i4i32_weights>();
        if (w->impl.prepare(domain,*cfg,B)) return nullptr;
        return w.release();
    } catch (...) { return nullptr; }
}
extern "C" void rk_npu_i4i4i32_weights_free(rk_npu_i4i4i32_weights* weights) { delete weights; }
extern "C" rk_npu_i4_workspace* rk_npu_i4_workspace_create(
    rk_npu_iommu_domain* domain, const rk_npu_i4_config* cfg) {
    if (!domain || !cfg) return nullptr;
    try {
        auto w=std::make_unique<rk_npu_i4_workspace>();
        if (w->impl.prepare(domain,*cfg)) return nullptr;
        return w.release();
    } catch (...) { return nullptr; }
}
extern "C" void rk_npu_i4_workspace_free(rk_npu_i4_workspace* workspace) { delete workspace; }

extern "C" int rk_npu_i4i4i32_run_with_producer(rk_npu_i4_workspace* workspace,
    const rk_npu_i4i4i32_weights* weights, rk_npu_i4_input_producer producer,
    void* user, int32_t* C, rk_npu_i4_timings* times) {
    if (!workspace || !weights) return RK_NPU_ERR_PARAM;
    return workspace->impl.run(weights->impl,producer,user,C,times);
}
namespace {
struct Rows { const int8_t* A; int K; };
int pack_rows(void* opaque, const rk_npu_i4_input_tile* tile) {
    const auto& rows=*static_cast<Rows*>(opaque);
    return rk_npu_i4_pack_a_tile(tile,rows.A,rows.K);
}
}
extern "C" int rk_npu_i4i4i32_run(rk_npu_i4_workspace* workspace,
    const rk_npu_i4i4i32_weights* weights, const int8_t* A, int32_t* C, rk_npu_i4_timings* times) {
    if (!workspace || !A) return RK_NPU_ERR_PARAM;
    Rows rows{A,workspace->impl.K()};
    return rk_npu_i4i4i32_run_with_producer(workspace,weights,pack_rows,&rows,C,times);
}

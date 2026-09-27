#pragma once
#include "rk_npu_w4a8_tune.h"
#include "rk_npu_autotune_common.h"
#include "rk_npu_autotune_cache.h"
#include "rk_npu_internal.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

namespace rknpu2_matmul_open::w4_tune {
constexpr uint32_t revision=3; // layout-aware candidates, strategy and cache encoding
using Strategy = rk_npu_w4a8_strategy_ex;
inline Strategy extend(const rk_npu_w4a8_strategy& s) {
    return {s,{sizeof(rk_npu_w4a8_options),RK_NPU_W4A8_INPUT_NATIVE,RK_NPU_W4A8_REDUCE_CPU}};
}
inline Strategy extend(const Strategy& s) { return s; }
inline void publish(rk_npu_w4a8_strategy* out,const Strategy& s) { if(out) *out=s.base; }
inline void publish(Strategy* out,const Strategy& s) { if(out) *out=s; }
struct Candidate {
    rk_npu_w4a8_config config;
    rk_npu_w4a8_options execution;
};
struct Profile {
    int K=0,N=0,safe=0,limit=0;
    double multiplier=1;
    uint64_t fingerprint=0,safe_tiles=0;
    bool certified(int k) const {return k>=32 && k<=2048 && k%32==0 && (safe_tiles&(uint64_t(1)<<(k/32-1)));}
    bool permits(int k) const {return k>=32 && k<=limit && k%32==0 && (certified(k)||multiplier>1);}
    rk_npu_w4a8_weight_info info(int k) const {return {{K,N,k},safe,multiplier,!certified(k)};}
};
struct Resources {uint32_t npu=0;uint64_t cpu=0,allowed_cpu=0;int threads=0;};
using Weight=std::unique_ptr<rk_npu_w4a8_weights,decltype(&rk_npu_w4a8_weights_free)>;
using Workspace=std::unique_ptr<rk_npu_w4a8_workspace,decltype(&rk_npu_w4a8_workspace_free)>;
bool valid(const rk_npu_w4a8_autotune_config&);
int profile(const rk_npu_w4a8_autotune_config&,const int8_t*,const float*,Profile&);
Resources resources(const rk_npu_w4a8_autotune_config&);
std::vector<int> k_candidates(const rk_npu_w4a8_autotune_config&,const Profile&);
std::vector<Candidate> candidates(const rk_npu_w4a8_autotune_config&,const Profile&,const Resources&,bool panel8=false);
bool valid_strategy(const Strategy&,const rk_npu_w4a8_autotune_config&,const Profile&,const Resources&,bool panel8=false);
bool valid_execution(const Strategy&,bool panel8);
std::string key(rk_npu_ctx*,const rk_npu_w4a8_autotune_config&,const Profile&,const Resources&,bool panel8=false);
std::string cache_path(const std::string& key,const char* directory=nullptr);
int load(const std::string& path,const std::string& key,const rk_npu_w4a8_autotune_config&,
         const Profile&,const Resources&,Strategy&,Strategy&,bool panel8=false);
int save(const std::string& path,const std::string& key,const Strategy&,const Strategy&);

struct Data {
    int M,K,N; bool half;
    std::vector<float> a,c,reference;
    std::vector<uint16_t> ah,ch,reference_h;
    Data(const rk_npu_w4a8_autotune_config&);
    int run(rk_npu_w4a8_workspace*,const rk_npu_w4a8_weights*,rknpu2_matmul_open::tune::Sample&);
    bool same() const;
    void capture();
    bool verify_reference(const int8_t* B,const float* scale) const;
};
// All entries below are called on a dedicated tuning thread. Resource masks
// are resolved on the caller before affinity is pinned.
int tune(rk_npu_ctx*,const rk_npu_w4a8_autotune_config&,const int8_t*,const float*,
         const Profile&,const Resources&,Strategy&,Strategy&,bool panel8=false);
int cached(rk_npu_ctx*,const rk_npu_w4a8_autotune_config&,const int8_t*,const float*,
           const Profile&,const Resources&,const char* path,int refresh,
           Strategy&,Strategy&,int* hit,bool panel8=false);
int on_worker(const Resources&,const std::function<int()>&);
} // namespace rknpu2_matmul_open::w4_tune

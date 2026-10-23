#include "rk_npu_attention_f16.h"
#include "mla_f16_test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace bits = mla_test::bits;
using mla_test::Device;
using mla_test::require;
using mla_test::check;
using Cache = mla_test::Handle<rk_npu_attention_f16_cache, rk_npu_attention_f16_cache_free>;
using Work = mla_test::Handle<rk_npu_attention_f16_workspace, rk_npu_attention_f16_workspace_free>;
Cache cache(Device& d, const rk_npu_attention_f16_config& c) {
    rk_npu_attention_f16_cache* out = nullptr;
    check(rk_npu_attention_f16_cache_create(d.domain, &c, &out));
    return Cache(out, rk_npu_attention_f16_cache_free);
}
Work work(Device& d, const rk_npu_attention_f16_config& c) {
    rk_npu_attention_f16_workspace* out = nullptr;
    check(rk_npu_attention_f16_prepare(d.domain, &c, &out));
    return Work(out, rk_npu_attention_f16_workspace_free);
}
rk_npu_attention_f16_cache_info info(rk_npu_attention_f16_cache* c) {
    rk_npu_attention_f16_cache_info out{};
    check(rk_npu_attention_f16_cache_get_info(c, &out));
    return out;
}
bool equal(const rk_npu_attention_f16_cache_info& a, const rk_npu_attention_f16_cache_info& b) {
    return a.length == b.length && a.capacity == b.capacity && a.max_capacity == b.max_capacity &&
        a.key_bytes == b.key_bytes && a.value_bytes == b.value_bytes && a.tail_bytes == b.tail_bytes &&
        a.generation == b.generation;
}
std::vector<uint16_t> data(size_t count, int seed, float scale) {
    std::vector<uint16_t> out(count);
    for (size_t i = 0; i < count; ++i)
        out[i] = bits::float_to_half(std::sin(float((i * 19 + seed) % 1009) * .031f) * scale);
    return out;
}
std::vector<uint16_t> row(const std::vector<uint16_t>& x, int heads, int rows, int d, int r) {
    std::vector<uint16_t> out(size_t(heads) * d);
    for (int h = 0; h < heads; ++h)
        std::copy_n(x.data() + (size_t(h) * rows + r) * d, d, out.data() + h * d);
    return out;
}
std::vector<float> execute(Device& dev, const rk_npu_attention_f16_config& c,
    rk_npu_attention_f16_cache* kv, rk_npu_attention_f16_workspace* w, bool batch,
    const std::vector<uint16_t>& q, const std::vector<uint16_t>& k, const std::vector<uint16_t>& v,
    int rows, int start, int* task_count = nullptr) {
    mla_test::GuardOutput output(size_t(c.query_heads) * rows * c.head_dim);
    int tasks = 0;
    if (batch) {
        check(rk_npu_attention_f16_cache_append(dev.ctx, kv, k.data(), v.data(), rows));
        rk_npu_attention_f16_timings timing{};
        check(rk_npu_attention_f16_run(dev.ctx, w, kv, q.data(), rows, start, output.ptr(), &timing, nullptr));
        tasks += timing.tasks;
    } else {
        for (int r = 0; r < rows; ++r) {
            auto qr = row(q, c.query_heads, rows, c.head_dim, r);
            auto kr = row(k, c.kv_heads, rows, c.head_dim, r);
            auto vr = row(v, c.kv_heads, rows, c.head_dim, r);
            mla_test::GuardOutput result(qr.size());
            rk_npu_attention_f16_timings timing{};
            check(rk_npu_attention_f16_decode_step(dev.ctx, w, kv, qr.data(), kr.data(), vr.data(), result.ptr(), &timing, nullptr));
            tasks += timing.tasks;
            const auto values = result.values();
            for (int h = 0; h < c.query_heads; ++h)
                std::copy_n(values.data() + h * c.head_dim, c.head_dim,
                    output.ptr() + (size_t(h) * rows + r) * c.head_dim);
        }
    }
    if (task_count) *task_count = tasks;
    return output.values();
}
void batch_case(Device& dev, const char* name, int start, int rows, int dimension,
    int query_heads, int kv_heads, bool lut, int high, int core = 7) {
    setenv("RK_NPU_ATTN_LUT_EXP", lut ? "1" : "0", 1);
    rk_npu_attention_f16_config c;
    rk_npu_attention_f16_config_init(&c);
    c.flags = RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    c.query_heads = query_heads; c.kv_heads = kv_heads; c.head_dim = dimension;
    c.max_query_rows = std::max(32, rows); c.core_mask = core; c.kv_tile = 512;
    c.initial_capacity = 32; c.max_capacity = std::max(128, (start + rows + 31) / 32 * 32);
    auto a = cache(dev, c), b = cache(dev, c), oracle = cache(dev, c);
    auto wa = work(dev, c), wb = work(dev, c), wo = work(dev, c);
    auto pk = data(size_t(kv_heads) * start * dimension, 17, .6f);
    auto pv = data(pk.size(), 19, .5f);
    auto q = data(size_t(query_heads) * rows * dimension, 21, .7f);
    auto k = data(size_t(kv_heads) * rows * dimension, 23, .6f);
    auto v = data(k.size(), 25, .5f);
    if (high == 1) v[0] = bits::float_to_half(64.f);
    if (high == 2) v[(rows - 1) * dimension] = bits::float_to_half(64.f);
    if (high == 3) pv[0] = bits::float_to_half(64.f);
    if (high == 4) v[(rows - 1) * dimension] = bits::float_to_half(65000.f);
    if (high == 5) pv[0] = bits::float_to_half(65000.f);
    for (auto* kv : {a.get(), b.get(), oracle.get()})
        check(rk_npu_attention_f16_cache_load(dev.ctx, kv, pk.data(), pv.data(), start));
    // LUT policy belongs to the cache. A later environment change must not
    // incorrectly qualify an already-created CPU-exp cache.
    setenv("RK_NPU_ATTN_LUT_EXP", "1", 1);
    int batch_tasks=0,serial_tasks=0;
    const auto actual = execute(dev, c, a.get(), wa.get(), true, q, k, v, rows, start, &batch_tasks);
    const auto repeated = execute(dev, c, b.get(), wb.get(), true, q, k, v, rows, start);
    const auto serial = execute(dev, c, oracle.get(), wo.get(), false, q, k, v, rows, start, &serial_tasks);
    require(actual == repeated && actual == serial, "native causal batch differs from AR");
    require(equal(info(a.get()), info(b.get())), "batch changed final KV memory information");
    if (!high && rows > 1 && start + rows <= 1024)
        require(batch_tasks < serial_tasks, "uniform CPU/LUT batch was serialized");
    if (!lut && !high && start >= 1024 && rows > 1)
        require(batch_tasks < serial_tasks, "uniform long CPU batch was serialized");
    if (!lut && !high && start == 1021 && rows == 4)
        require(batch_tasks == 28, "CPU boundary should split 3 short rows plus 1 long row");
    auto next_q = data(size_t(query_heads) * dimension, 31, .7f);
    std::vector<float> x(next_q.size()), y(x.size());
    check(rk_npu_attention_f16_run(dev.ctx, wa.get(), a.get(), next_q.data(), 1, start + rows - 1, x.data(), nullptr, nullptr));
    check(rk_npu_attention_f16_run(dev.ctx, wo.get(), oracle.get(), next_q.data(), 1, start + rows - 1, y.data(), nullptr, nullptr));
    require(x == y, "final KV contents differ from AR");
    if (high == 2) {
        check(rk_npu_attention_f16_cache_truncate(a.get(), start + 1));
        auto first_v = row(v, kv_heads, rows, dimension, 0);
        auto first_q = row(q, query_heads, rows, dimension, 0);
        std::vector<float> replay(first_q.size());
        check(rk_npu_attention_f16_run(dev.ctx, wa.get(), a.get(), first_q.data(), 1, start, replay.data(), nullptr, nullptr));
        int changed = 0;
        for (int h = 0; h < query_heads; ++h) for (int d = 0; d < dimension; ++d) {
            const float ref = serial[(size_t(h) * rows) * dimension + d];
            const double delta = double(replay[size_t(h) * dimension + d]) - ref;
            changed += delta != 0;
        }
        require(changed == 0, "partial commit retained discarded high-V mode");
        check(rk_npu_attention_f16_cache_truncate(b.get(), start));
        auto first_k = row(k, kv_heads, rows, dimension, 0);
        std::vector<float> abort_output(first_q.size());
        check(rk_npu_attention_f16_decode_step(dev.ctx, wb.get(), b.get(), first_q.data(), first_k.data(), first_v.data(), abort_output.data(), nullptr, nullptr));
        require(abort_output == replay, "abort retained discarded high-V mode");
        std::printf("ROLLBACK_CHECK start=%d high_future_removed=1 partial_AR_exact=1 abort_AR_exact=1\n", start);
    }
    std::printf("PASS %s core=%d lut=%d rows=%d start=%d batch_tasks=%d serial_tasks=%d final_kv_bytes_equal=1 all_outputs_AR_exact=1\n",
        name, core, int(lut), rows, start, batch_tasks, serial_tasks);
    std::fflush(stdout);
}
void masked_future_case(Device& dev, int start, int core) {
    const int qh = 16, kh = 4, d = 32, length = start + 8, visible = start + 1;
    setenv("RK_NPU_ATTN_LUT_EXP", "1", 1);
    rk_npu_attention_f16_config c;
    rk_npu_attention_f16_config_init(&c);
    c.flags = RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    c.query_heads = qh; c.kv_heads = kh; c.head_dim = d; c.core_mask = core;
    c.initial_capacity = c.max_capacity = (length + 31) / 32 * 32;
    auto kv = cache(dev, c); auto w = work(dev, c);
    auto k = data(size_t(kh) * length * d, 41, .6f);
    auto v = data(k.size(), 43, .5f);
    auto q = data(size_t(qh) * d, 47, .7f);
    std::vector<uint16_t> pk(size_t(kh) * visible * d), pv(pk.size());
    for (int h = 0; h < kh; ++h) {
        std::copy_n(k.data() + size_t(h) * length * d, visible * d, pk.data() + size_t(h) * visible * d);
        std::copy_n(v.data() + size_t(h) * length * d, visible * d, pv.data() + size_t(h) * visible * d);
    }
    mla_test::GuardOutput reference(q.size());
    check(rk_npu_attention_f16_cache_load(dev.ctx, kv.get(), pk.data(), pv.data(), visible));
    check(rk_npu_attention_f16_run(dev.ctx, w.get(), kv.get(), q.data(), 1, start, reference.ptr(), nullptr, nullptr));
    const auto expected = reference.values();
    for (int variant : {1, 0, -1, 64, 65000}) {
        auto future = v;
        for (int h = 0; h < kh; ++h) for (int j = visible; j < length; ++j)
            for (int channel = 0; channel < d; ++channel) {
                auto& value = future[(size_t(h) * length + j) * d + channel];
                if (!variant) value = 0;
                else if (variant < 0) value ^= 0x8000;
                else if (variant > 1) value = bits::float_to_half(float(variant));
            }
        mla_test::GuardOutput actual(q.size());
        check(rk_npu_attention_f16_cache_load(dev.ctx, kv.get(), k.data(), future.data(), length));
        check(rk_npu_attention_f16_run(dev.ctx, w.get(), kv.get(), q.data(), 1, start, actual.ptr(), nullptr, nullptr));
        require(actual.values() == expected, "masked future V changed a visible-prefix output");
    }
    std::printf("PASS masked_future core=%d start=%d prefix/complete/zero/negative_future_AR_exact=1\n", core, start);
}
void bulk_checkpoint_case(Device& dev, int start) {
    const int qh = 16, kh = 4, d = 32, rows = 8, length = start + rows;
    setenv("RK_NPU_ATTN_LUT_EXP", "1", 1);
    rk_npu_attention_f16_config c;
    rk_npu_attention_f16_config_init(&c);
    c.flags = RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    c.query_heads = qh; c.kv_heads = kh; c.head_dim = d; c.core_mask = 7;
    c.initial_capacity = 32; c.max_capacity = (length + 31) / 32 * 32 + 64;
    auto k = data(size_t(kh) * length * d, 53, .6f);
    auto v = data(k.size(), 59, .5f);
    v[(length - 1) * d] = bits::float_to_half(64.f);
    auto q = data(size_t(qh) * d, 61, .7f);
    std::vector<uint16_t> pk(size_t(kh) * start * d), pv(pk.size());
    std::vector<uint16_t> nk(size_t(kh) * rows * d), nv(nk.size());
    for (int h = 0; h < kh; ++h) {
        std::copy_n(k.data() + size_t(h) * length * d, start * d, pk.data() + size_t(h) * start * d);
        std::copy_n(v.data() + size_t(h) * length * d, start * d, pv.data() + size_t(h) * start * d);
        std::copy_n(k.data() + (size_t(h) * length + start) * d, rows * d, nk.data() + size_t(h) * rows * d);
        std::copy_n(v.data() + (size_t(h) * length + start) * d, rows * d, nv.data() + size_t(h) * rows * d);
    }
    const auto first_k = row(k, kh, length, d, start), first_v = row(v, kh, length, d, start);
    auto fresh = cache(dev, c); auto fw = work(dev, c);
    mla_test::GuardOutput reference(q.size());
    check(rk_npu_attention_f16_cache_load(dev.ctx, fresh.get(), pk.data(), pv.data(), start));
    check(rk_npu_attention_f16_decode_step(dev.ctx, fw.get(), fresh.get(), q.data(), first_k.data(), first_v.data(), reference.ptr(), nullptr, nullptr));
    const auto expected = reference.values();
    for (bool load : {false, true}) for (int keep : {start, start + 1}) {
        auto kv = cache(dev, c); auto w = work(dev, c);
        if (load) check(rk_npu_attention_f16_cache_load(dev.ctx, kv.get(), k.data(), v.data(), length));
        else {
            check(rk_npu_attention_f16_cache_load(dev.ctx, kv.get(), pk.data(), pv.data(), start));
            check(rk_npu_attention_f16_cache_append(dev.ctx, kv.get(), nk.data(), nv.data(), rows));
        }
        check(rk_npu_attention_f16_cache_reserve(dev.ctx, kv.get(), c.max_capacity));
        check(rk_npu_attention_f16_cache_truncate(kv.get(), keep));
        mla_test::GuardOutput actual(q.size());
        if (keep == start)
            check(rk_npu_attention_f16_decode_step(dev.ctx, w.get(), kv.get(), q.data(), first_k.data(), first_v.data(), actual.ptr(), nullptr, nullptr));
        else check(rk_npu_attention_f16_run(dev.ctx, w.get(), kv.get(), q.data(), 1, start, actual.ptr(), nullptr, nullptr));
        require(actual.values() == expected, "bulk checkpoint rollback differs from fresh AR");
        std::printf("PASS bulk_checkpoint start=%d method=%s keep=%d reserve/prefix/AR_exact=1\n", start, load ? "load" : "append", keep);
    }
}
void cpu() {
    auto* c = reinterpret_cast<rk_npu_attention_f16_cache*>(1);
    auto* w = reinterpret_cast<rk_npu_attention_f16_workspace*>(1);
    require(rk_npu_attention_f16_run(nullptr, nullptr, c, nullptr, 1, 0, nullptr, nullptr, nullptr) == RK_NPU_ERR_PARAM, "null workspace");
    require(rk_npu_attention_f16_run(nullptr, w, nullptr, nullptr, 1, 0, nullptr, nullptr, nullptr) == RK_NPU_ERR_PARAM, "null cache");
    require(rk_npu_attention_f16_prepare_query(nullptr, nullptr, c, 1, 0) == RK_NPU_ERR_PARAM, "null prepare workspace");
    std::puts("causal Attention CPU parameter contract PASS");
}
void preloaded_prefix_case(Device& dev, bool lut) {
    // Internal 32-row segments share a graph within this request's visible
    // prefix. Large future V must not affect dispatch, results or sizing.
    constexpr int qh=16,kh=4,d=32,rows=128,length=512;
    setenv("RK_NPU_ATTN_LUT_EXP",lut?"1":"0",1);
    rk_npu_attention_f16_config c;rk_npu_attention_f16_config_init(&c);
    c.flags=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;c.query_heads=qh;c.kv_heads=kh;c.head_dim=d;
    c.core_mask=7;c.initial_capacity=c.max_capacity=length;c.max_query_rows=rows;
    auto kv=cache(dev,c);auto batch=work(dev,c),serial=work(dev,c);
    auto k=data(size_t(kh)*length*d,67,.6f),v=data(k.size(),71,.5f);
    for(int h=0;h<kh;++h)for(int pos=384;pos<length;++pos)
        std::fill_n(v.data()+(size_t(h)*length+pos)*d,d,bits::float_to_half(65000.f));
    check(rk_npu_attention_f16_cache_load(dev.ctx,kv.get(),k.data(),v.data(),length));
    for(int start:{0,128,256}){
        auto q=data(size_t(qh)*rows*d,73+start,.7f);mla_test::GuardOutput output(q.size());
        check(rk_npu_attention_f16_prepare_query(dev.ctx,batch.get(),kv.get(),rows,start));
        rk_npu_attention_f16_timings t{};
        check(rk_npu_attention_f16_run(dev.ctx,batch.get(),kv.get(),q.data(),rows,start,output.ptr(),&t,nullptr));
        rk_npu_attention_f16_sizes sizes{};
        check(rk_npu_attention_f16_query(&c,length,rows,start+rows,&sizes));
        require(t.compute_length==start+rows && t.compute_length<=sizes.compute_length,
            "prefix graph reuse exceeded public query sizing");
        std::vector<float> expected(q.size());
        for(int r=0;r<rows;++r){
            auto qr=row(q,qh,rows,d,r);mla_test::GuardOutput one(qr.size());
            check(rk_npu_attention_f16_run(dev.ctx,serial.get(),kv.get(),qr.data(),1,start+r,one.ptr(),nullptr,nullptr));
            const auto values=one.values();
            for(int h=0;h<qh;++h)std::copy_n(values.data()+h*d,d,expected.data()+(size_t(h)*rows+r)*d);
        }
        require(output.values()==expected,"preloaded future changed batched causal output");
        std::printf("PASS preloaded_prefix lut=%d start=%d compute_length=%d high_future_AR_exact=1\n",int(lut),start,t.compute_length);
    }
}
void board() {
    // run_board.py retains this FD across exec; reuse its lowlevel lock.
    std::unique_ptr<mla_test::BoardLock> lowlevel;
    if (const char* inherited = std::getenv("LLM_INFER_LOCK_FD"))
        require(fcntl(std::stoi(inherited), F_GETFD) >= 0, "runner lock FD is closed");
    else lowlevel.reset(new mla_test::BoardLock("/tmp/rknpu_lowlevel_submit.lock"));
    mla_test::BoardLock legacy("/tmp/rk3588_npu_submit.lock");
    Device dev;
    for(bool lut:{true,false})preloaded_prefix_case(dev,lut);
    batch_case(dev, "C7_small", 5, 8, 256, 16, 4, true, 0);
    for (int core : {1, 3}) batch_case(dev, "C7_small_core", 5, 8, 256, 16, 4, true, 0, core);
    batch_case(dev, "empty_prefix_padding_M", 0, 3, 32, 4, 4, true, 0);
    batch_case(dev, "K32_boundary_padding_M", 31, 3, 32, 4, 4, true, 0);
    for (int start : {1016,1021,1024})
        batch_case(dev, "CPU_short_long", start, 4, 256, 16, 4, false, 0);
    batch_case(dev, "CPU_boundary_R8", 1021, 8, 32, 16, 4, false, 0);
    batch_case(dev, "new_high_V", 5, 8, 256, 16, 4, true, 1);
    for (int start : {0,5,27,31,509,511,1021,1024})
        batch_case(dev, "future_high_V", start, 8, 32, 16, 4, true, 2);
    batch_case(dev, "historical_high_V", 5, 8, 256, 16, 4, true, 3);
    for (int start : {5,1024}) for (int high : {2,4,5})
        batch_case(dev, "CPU_V_mode", start, 8, 32, 16, 4, false, high);
    batch_case(dev, "long_LUT_fits", 32759, 8, 32, 16, 4, true, 0);
    batch_case(dev, "long_descriptor_boundary", 32736, 32, 32, 16, 4, true, 0);
    batch_case(dev, "internal_GQA_32_plus_1", 5, 33, 32, 16, 4, true, 0);
    for (int core : {1, 3, 7}) for (int start : {5, 31, 511}) masked_future_case(dev, start, core);
    for (int start : {31, 511}) bulk_checkpoint_case(dev, start);
}
} // namespace
int main(int argc, char** argv) try {
    require(argc <= 2, "usage: test_attention_f16_causal [--cpu|--board]");
    const std::string mode = argc > 1 ? argv[1] : "--cpu";
    require(mode == "--cpu" || mode == "--board", "usage: test_attention_f16_causal [--cpu|--board]");
    cpu();
    if (mode == "--board") board();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}

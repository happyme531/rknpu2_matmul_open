#include "rk_npu_quant_matmul.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct Args {
    int M = 128, N = 2560, K = 9216;
    int warmup = 2, loops = 8, repeats = 3, timeout_ms = 500;
    uint32_t npu_mask = 7;
    uint64_t cpu_mask = 0;
    rk_npu_matmul_op_kind kind = RK_NPU_MATMUL_F16I8F16_DYNAMIC;
    const char* dev = nullptr;
    const char* cache_path = nullptr;
    bool no_cache = false;
    bool refresh_cache = false;
};

long long parse_int(const char* s) {
    if (!s || !*s) return -1;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 0);
    return end && *end == 0 ? v : -1;
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto value = [&]() -> long long {
            return i + 1 < argc ? parse_int(argv[++i]) : -1;
        };
        if (option == "--m") a.M = (int)value();
        else if (option == "--n") a.N = (int)value();
        else if (option == "--k") a.K = (int)value();
        else if (option == "--warmup") a.warmup = (int)value();
        else if (option == "--loops") a.loops = (int)value();
        else if (option == "--repeats") a.repeats = (int)value();
        else if (option == "--timeout-ms") a.timeout_ms = (int)value();
        else if (option == "--npu-mask") a.npu_mask = (uint32_t)value();
        else if (option == "--cpu-mask") a.cpu_mask = (uint64_t)value();
        else if (option == "--dev") {
            if (i + 1 >= argc) return false;
            a.dev = argv[++i];
        } else if (option == "--cache-path") {
            if (i + 1 >= argc) return false;
            a.cache_path = argv[++i];
        } else if (option == "--no-cache") {
            a.no_cache = true;
        } else if (option == "--refresh-cache") {
            a.refresh_cache = true;
        } else if (option == "--op") {
            if (i + 1 >= argc) return false;
            const std::string op = argv[++i];
            if (op == "i8i8i32") a.kind = RK_NPU_MATMUL_I8I8I32;
            else if (op == "dynamic") a.kind = RK_NPU_MATMUL_F16I8F16_DYNAMIC;
            else if (op == "static") a.kind = RK_NPU_MATMUL_F16I8F16_STATIC;
            else if (op == "f32-dynamic") a.kind = RK_NPU_MATMUL_F32I8F32_DYNAMIC;
            else if (op == "f32-static") a.kind = RK_NPU_MATMUL_F32I8F32_STATIC;
            else return false;
        } else if (option == "--help" || option == "-h") {
            std::printf(
                "usage: %s --op i8i8i32|dynamic|static|f32-dynamic|f32-static "
                "[--m M --n N --k K]\n"
                "          [--npu-mask MASK --cpu-mask MASK] "
                "[--warmup W --loops L --repeats R]\n"
                "          [--cache-path FILE | --no-cache] [--refresh-cache]\n",
                argv[0]);
            std::exit(0);
        } else return false;
    }
    return a.M > 0 && a.N > 0 && a.K > 0 && a.warmup >= 0 &&
           a.loops > 0 && a.repeats > 0 && a.timeout_ms > 0 &&
           a.npu_mask > 0 && (a.npu_mask & ~7u) == 0 &&
           !(a.no_cache && (a.cache_path || a.refresh_cache));
}

const char* op_name(rk_npu_matmul_op_kind kind) {
    switch (kind) {
    case RK_NPU_MATMUL_I8I8I32: return "i8i8i32";
    case RK_NPU_MATMUL_F16I8F16_DYNAMIC: return "f16i8f16_dynamic";
    case RK_NPU_MATMUL_F16I8F16_STATIC: return "f16i8f16_static";
    case RK_NPU_MATMUL_F32I8F32_DYNAMIC: return "f32i8f32_dynamic";
    case RK_NPU_MATMUL_F32I8F32_STATIC: return "f32i8f32_static";
    }
    return "unknown";
}

const char* a_name(rk_npu_matmul_a_layout layout) {
    switch (layout) {
    case RK_NPU_MATMUL_A_LAYOUT_NORMAL: return "normal";
    case RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16: return "native";
    case RK_NPU_MATMUL_A_LAYOUT_PANEL8: return "panel8";
    case RK_NPU_MATMUL_A_LAYOUT_PANEL16: return "panel16";
    }
    return "invalid";
}
const char* c_name(rk_npu_matmul_c_layout layout) {
    return layout == RK_NPU_MATMUL_C_LAYOUT_NATIVE ? "native" :
           layout == RK_NPU_MATMUL_C_LAYOUT_PANEL8 ? "panel8" : "panel16";
}

void print_strategy(const char* label, const rk_npu_matmul_strategy& s) {
    const double tops = s.total_us > 0
        ? 2.0 * s.M * s.N * s.K / s.total_us / 1.0e6 : 0;
    std::printf(
        "%s %s A=%s C=%s Kt=%d waves=%d Nt=%d groups=%d npu=0x%x cpu=0x%llx/%d\n"
        "  total=%.2f us robust=%.2f us jitter=%.2f%% TOPS=%.3f "
        "[input=%.2f npu=%.2f sync=%.2f output=%.2f us]\n",
        label, op_name(s.op_kind),
        a_name(s.a_layout), c_name(s.c_layout),
        s.k_tile, s.wave_count, s.n_tile, s.n_groups,
        s.npu_core_mask, (unsigned long long)s.cpu_core_mask, s.cpu_threads,
        s.total_us, s.robust_us, s.jitter_pct, tops,
        s.input_us, s.npu_us, s.sync_us, s.output_us);
}

void print_memory(const rk_npu_matmul_strategy& s) {
    rk_npu_matmul_workspace_requirements workspace{};
    rk_npu_matmul_weight_requirements weight{};
    const rk_npu_matmul_weight_config wc{s.K, s.N, s.k_tile};
    if (rk_npu_matmul_workspace_memory_query(&s, &workspace) != RK_NPU_OK)
        return;
    const int rc = s.op_kind == RK_NPU_MATMUL_I8I8I32
        ? rk_npu_i8i8i32_weights_memory_query(&wc, &weight)
        : rk_npu_f16i8f16_weights_memory_query(&wc, &weight);
    if (rc != RK_NPU_OK) return;
    std::printf(
        "MEMORY workspace=%.3f MiB [A=%.3f C=%.3f control=%.3f] "
        "weight=%.3f MiB [B=%.3f scale=%.3f] buffers=%u+%u\n",
        workspace.bytes / 1048576.0,
        workspace.input_bytes / 1048576.0,
        workspace.output_bytes / 1048576.0,
        workspace.control_bytes / 1048576.0,
        weight.bytes / 1048576.0,
        weight.packed_b_bytes / 1048576.0,
        weight.scale_bytes / 1048576.0,
        workspace.buffer_count, weight.buffer_count);
}

} /* namespace */

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;
    rk_npu_ctx* ctx = rk_npu_open(a.dev);
    if (!ctx) return 2;
    rk_npu_matmul_autotune_config cfg{};
    rk_npu_matmul_autotune_config_init(&cfg, a.M, a.N, a.K);
    cfg.allowed_npu_core_mask = a.npu_mask;
    cfg.allowed_cpu_core_mask = a.cpu_mask;
    cfg.warmup = a.warmup;
    cfg.loops = a.loops;
    cfg.repeats = a.repeats;
    cfg.timeout_ms = (uint32_t)a.timeout_ms;
    cfg.verbose = 1;

    rk_npu_matmul_strategy fastest{}, stable{};
    int cache_hit = 0;
    int rc = a.no_cache
        ? RK_NPU_ERR_PARAM
        : rk_npu_matmul_autotune_cached(
              ctx, &cfg, a.kind, a.cache_path, a.refresh_cache ? 1 : 0,
              &fastest, &stable, &cache_hit);
    if (a.no_cache) {
        switch (a.kind) {
        case RK_NPU_MATMUL_I8I8I32:
            rc = rk_npu_i8i8i32_autotune(ctx, &cfg, &fastest, &stable); break;
        case RK_NPU_MATMUL_F16I8F16_DYNAMIC:
            rc = rk_npu_f16i8f16_dynamic_autotune(ctx, &cfg, &fastest, &stable); break;
        case RK_NPU_MATMUL_F16I8F16_STATIC:
            rc = rk_npu_f16i8f16_static_autotune(ctx, &cfg, &fastest, &stable); break;
        case RK_NPU_MATMUL_F32I8F32_DYNAMIC:
            rc = rk_npu_f32i8f32_dynamic_autotune(ctx, &cfg, &fastest, &stable); break;
        case RK_NPU_MATMUL_F32I8F32_STATIC:
            rc = rk_npu_f32i8f32_static_autotune(ctx, &cfg, &fastest, &stable); break;
        }
    }
    if (rc == RK_NPU_OK) {
        if (!a.no_cache)
            std::printf("TUNING_CACHE %s\n", cache_hit ? "HIT" : "MISS");
        print_strategy("BEST_LATENCY", fastest);
        print_memory(fastest);
        print_strategy("BEST_STABLE", stable);
        if (fastest.k_tile != stable.k_tile || fastest.n_tile != stable.n_tile ||
            fastest.npu_core_mask != stable.npu_core_mask ||
            fastest.cpu_core_mask != stable.cpu_core_mask ||
            fastest.cpu_threads != stable.cpu_threads)
            print_memory(stable);
    } else {
        std::fprintf(stderr, "autotune failed: %s\n", rk_npu_strerror(rc));
    }
    rk_npu_close(ctx);
    return rc == RK_NPU_OK ? 0 : 1;
}

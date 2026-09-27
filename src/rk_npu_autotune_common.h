#pragma once
#include "rk_npu_common.h"
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

namespace rknpu2_matmul_open::tune {
struct Sample {
    double input_us=0, npu_us=0, sync_us=0, output_us=0, total_us=0;
};
struct Score : Sample { double jitter_pct=0, robust_us=0; };
struct Options { int warmup, loops, repeats; };
// Validation is outside the measured backend call. Submit failures propagate.
int measure(const Options&, const std::function<int(Sample&)>& run,
            const std::function<bool()>& validate, Score&);
double median(std::vector<double>);
uint64_t process_cpu_mask();
std::vector<int> preferred_cpus(uint64_t allowed);
uint64_t first_cpu_mask(const std::vector<int>&,int count);
uint32_t fixed_npu_mask(uint32_t allowed,int N,int channel_alignment=32);
void pin_openmp_team(uint64_t cpu_mask,int threads);
bool set_thread_cpu_mask(uint64_t cpu_mask);
uint64_t double_bits(double);
double bits_double(uint64_t);
struct Hash {
    uint64_t value=1469598103934665603ull; // preserve existing W8 cache identity
    void bytes(const void*,size_t);
    void mix(uint64_t);
};
} // namespace rknpu2_matmul_open::tune

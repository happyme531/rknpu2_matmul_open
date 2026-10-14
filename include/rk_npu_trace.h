#pragma once

#include <cstdint>

#ifndef RK_NPU_ENABLE_TRACE
#define RK_NPU_ENABLE_TRACE 0
#endif

#if RK_NPU_ENABLE_TRACE
#include "perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("llm").SetDescription("Measured model calls and CPU operators"),
    perfetto::Category("rknpu").SetDescription("Complete NPU operator calls"),
    perfetto::Category("rknpu.detail").SetDescription("NPU submissions, workers and layout stages"));

#define RK_NPU_TRACE_EVENT(...) TRACE_EVENT(__VA_ARGS__)
#define RK_NPU_TRACE_BEGIN(...) TRACE_EVENT_BEGIN(__VA_ARGS__)
#define RK_NPU_TRACE_END(...) TRACE_EVENT_END(__VA_ARGS__)
#define RK_NPU_TRACE_COUNTER(...) TRACE_COUNTER(__VA_ARGS__)
#else
#define RK_NPU_TRACE_EVENT(...) do {} while (0)
#define RK_NPU_TRACE_BEGIN(...) do {} while (0)
#define RK_NPU_TRACE_END(...) do {} while (0)
#define RK_NPU_TRACE_COUNTER(...) do {} while (0)
#endif

namespace rk_npu_trace {
bool available();
// One in-process recording at a time. Start and stop run outside sample timers.
void start(const char *path);
void stop();
// Unique flow IDs for worker handoffs within this process.
std::uint64_t next_id();
} // namespace rk_npu_trace

#include "rk_npu_trace.h"
#include <stdexcept>

#if RK_NPU_ENABLE_TRACE
#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace {
std::unique_ptr<perfetto::TracingSession> session;
std::ofstream output;
std::once_flag initialized;
std::atomic<std::uint64_t> flow_id{1};
}
#endif

namespace rk_npu_trace {
bool available() { return RK_NPU_ENABLE_TRACE; }

void start(const char *path) {
#if RK_NPU_ENABLE_TRACE
    if (session) throw std::runtime_error("a Perfetto recording is already active");
    if (!path || !*path) throw std::runtime_error("Perfetto trace path is empty");
    output.open(path,std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(std::string("cannot open Perfetto trace: ")+path);
    std::call_once(initialized,[] {
        perfetto::TracingInitArgs args;
        args.backends = perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
    });
    perfetto::TraceConfig config;
    auto *buffer = config.add_buffers();
    buffer->set_size_kb(64 * 1024);
    buffer->set_fill_policy(perfetto::TraceConfig::BufferConfig::DISCARD);
    auto *source = config.add_data_sources()->mutable_config();
    source->set_name("track_event");
    perfetto::protos::gen::TrackEventConfig events;
    events.add_disabled_categories("*");
    for (const char *category : {"llm","rknpu","rknpu.detail"}) events.add_enabled_categories(category);
    source->set_track_event_config_raw(events.SerializeAsString());
    session = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
    session->Setup(config);
    session->StartBlocking();
#else
    (void)path;
    throw std::runtime_error("Perfetto tracing requires a build with RK_NPU_ENABLE_TRACE=ON");
#endif
}

void stop() {
#if RK_NPU_ENABLE_TRACE
    if (!session) return;
    perfetto::TrackEvent::Flush();
    const bool flushed = session->FlushBlocking();
    session->StopBlocking();
    const auto trace = session->ReadTraceBlocking();
    session.reset();
    output.write(trace.data(),static_cast<std::streamsize>(trace.size()));
    output.close();
    if (!output) throw std::runtime_error("failed writing Perfetto trace");
    if (!flushed) throw std::runtime_error("Perfetto flush timed out; trace may be incomplete");
#endif
}

std::uint64_t next_id() {
#if RK_NPU_ENABLE_TRACE
    return flow_id.fetch_add(1,std::memory_order_relaxed);
#else
    return 0;
#endif
}
} // namespace rk_npu_trace

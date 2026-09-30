#include "rk_npu_mm.h"
#include "rk_npu_matmul_f16.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include "rk_npu_bfloat_bits.h"
#include "rk_npu_float_backend.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace detail = rknpu2_matmul_open::detail;
namespace bits = rknpu2_matmul_open::bits;

struct rk_npu_mm_plan {
    rk_npu_mm_info info{};
    rk_npu_matmul_f16_config cfg{}; // one batch/split item
    rk_npu_matmul_sizes one{};
    int items = 0;
};

struct rk_npu_mm_workspace {
    rk_npu_mm_plan plan;
    rk_npu_iommu_domain* domain = nullptr;
    rk_npu_mm_workspace_mode mode{};
    detail::FloatBatchPlan* execution = nullptr;
    rk_npu_mem a{}, b{}, c{};
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    ~rk_npu_mm_workspace() {
        detail::float_batch_free(execution);
        if (domain) {
            rk_npu_mem_free(domain->ctx, &a);
            rk_npu_mem_free(domain->ctx, &b);
            rk_npu_mem_free(domain->ctx, &c);
            detail::release_domain(domain);
        }
    }
};

struct rk_npu_mm_packed_b {
    rk_npu_mm_plan plan;
    rk_npu_iommu_domain* domain = nullptr;
    rk_npu_mem mem{};
    ~rk_npu_mm_packed_b() {
        if (domain) {
            rk_npu_mem_free(domain->ctx, &mem);
            detail::release_domain(domain);
        }
    }
};

namespace {
int dtype_bytes(rk_npu_mm_dtype type) {
    return type == RK_NPU_MM_F32 || type == RK_NPU_MM_TF32 ? 4 : 2;
}
detail::FloatPrecision precision_for(rk_npu_mm_dtype type) {
    return type == RK_NPU_MM_TF32 ? detail::FloatPrecision::TF32 :
           type == RK_NPU_MM_BF16 ? detail::FloatPrecision::BF16 : detail::FloatPrecision::F16;
}
bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b; return true;
}
bool mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (a && b > UINT64_MAX / a) return false;
    out = a * b; return true;
}
bool span(int rows, int cols, int batches, uint64_t& ld, uint64_t& stride,
          bool output, uint64_t& bytes, int element_bytes) {
    if (!ld) ld = cols;
    if (ld < (uint64_t)cols) return false;
    uint64_t extent, total;
    if (!mul(rows - 1, ld, extent) || !add(extent, cols, extent)) return false;
    if (output && !stride && !mul(rows, ld, stride)) return false;
    if (batches > 1 && stride && stride < extent) return false;
    if (!mul(batches - 1, stride, total) || !add(total, extent, total) ||
        !mul(total, element_bytes, bytes)) return false;
    return bytes <= PTRDIFF_MAX;
}
bool addressable(const void* ptr, uint64_t bytes, int alignment = 2) {
    return ptr && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(ptr) &&
           (reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
}
bool overlap(const void* a, uint64_t an, const void* b, uint64_t bn) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + bn && y < x + an;
}
int conversion_buffers(const void* host, uint64_t bytes, uint64_t required,
                       const rk_npu_mem* dev, uint64_t packed, int alignment = 2) {
    if (!dev || !addressable(host, required, alignment) || !addressable(dev->vaddr, packed, alignment))
        return RK_NPU_ERR_PARAM;
    if (bytes < required || dev->size < packed) return RK_NPU_ERR_NOMEM;
    return overlap(host, required, dev->vaddr, packed) ? RK_NPU_ERR_PARAM : RK_NPU_OK;
}
int build_plan(const rk_npu_mm_desc& src, const rk_npu_mm_options& opt,
               rk_npu_mm_plan& p) {
    if (src.struct_size != sizeof(src) || opt.struct_size != sizeof(opt) ||
        src.M <= 0 || src.N <= 0 || src.K <= 0 || src.batch_count <= 0 ||
        (src.trans_a != 0 && src.trans_a != 1) ||
        (src.trans_b != 0 && src.trans_b != 1)) return RK_NPU_ERR_PARAM;
    for (auto type : {src.a_type, src.b_type, src.c_type})
        if (type < RK_NPU_MM_F16 || type > RK_NPU_MM_TF32) return RK_NPU_ERR_PARAM;
    const bool native16 = (src.a_type == RK_NPU_MM_F16 || src.a_type == RK_NPU_MM_BF16) &&
                          src.c_type == src.a_type;
    const bool tf32 = src.a_type == RK_NPU_MM_TF32 && src.c_type == RK_NPU_MM_F32;
    if ((!native16 && !tf32) || src.b_type != src.a_type)
        return RK_NPU_MM_ERR_UNSUPPORTED;
    // Bound legacy 16-bit register fields and its signed-int task arithmetic.
    if (src.M > 65504 || src.N > 65504 || src.K > 65504)
        return RK_NPU_MM_ERR_UNSUPPORTED;
    p.info.desc = src;
    auto& d = p.info.desc;
    const int bytes = dtype_bytes(d.a_type), ka = 64 / bytes;
    if (!span(d.trans_a ? d.K : d.M, d.trans_a ? d.M : d.K, d.batch_count,
              d.lda, d.batch_stride_a, false, p.info.host_a_bytes, bytes) ||
        !span(d.trans_b ? d.N : d.K, d.trans_b ? d.K : d.N, d.batch_count,
              d.ldb, d.batch_stride_b, false, p.info.host_b_bytes, bytes) ||
        !span(d.M, d.N, d.batch_count, d.ldc, d.batch_stride_c, true,
              p.info.host_c_bytes, bytes)) return RK_NPU_ERR_PARAM;
    p.info.options = opt;
    auto& o = p.info.options;
    if (!o.allowed_npu_core_mask) o.allowed_npu_core_mask = 7;
    const uint32_t mask = o.allowed_npu_core_mask;
    if (mask != 1 && mask != 2 && mask != 4 && mask != 3 && mask != 7)
        return RK_NPU_ERR_PARAM;
    for (auto layout : {o.a_layout, o.c_layout})
        if (layout != RK_NPU_MM_LAYOUT_NORMAL && layout != RK_NPU_MM_LAYOUT_NATIVE)
            return RK_NPU_ERR_PARAM;
    const int align_n = align_up(d.N, 32);
    if (o.n_tile < 0 || o.n_tile > align_n || o.n_tile % 32 ||
        o.split_k < -1 || o.split_k > 3) return RK_NPU_ERR_PARAM;
    if (o.split_k == -1) {
        const char* env = std::getenv("RK_NPU_MM_SPLIT_K");
        if (!env || !std::strcmp(env, "1")) o.split_k = 1;
        else if (!std::strcmp(env, "0")) o.split_k = 0;
        else return RK_NPU_ERR_PARAM;
    }
    const int cores = __builtin_popcount(mask);
    int splits = o.split_k >= 2 ? o.split_k :
        (o.split_k && d.K >= 2048 && d.batch_count < cores ? cores : 1);
    if (splits > cores || (splits > 1 && d.K < splits * ka)) return RK_NPU_ERR_PARAM;
    p.info.split_count = splits;
    p.info.part_k = splits == 1 ? d.K : align_up(ceil_div(d.K, splits), ka);
    // K=5120 unsplit timed out in the initial TF32 CBUF recipe. Keep each
    // slice within the validated range; never silently force another split.
    if (tf32 && align_up(p.info.part_k, 16) > 4096) return RK_NPU_MM_ERR_UNSUPPORTED;
    uint64_t tasks_bound = (uint64_t)d.batch_count * splits * d.M *
                          (o.n_tile ? ceil_div(align_n, o.n_tile) : 1);
    if (tasks_bound > INT_MAX / 112) return RK_NPU_MM_ERR_UNSUPPORTED;
    p.items = d.batch_count * splits;
    rk_npu_matmul_f16_config_init(&p.cfg, d.M, d.N, p.info.part_k, RK_NPU_FUSE_NONE);
    p.cfg.a_layout = o.a_layout == RK_NPU_MM_LAYOUT_NATIVE ?
        RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8 : RK_NPU_F16_A_LAYOUT_NORMAL;
    p.cfg.d_layout = o.c_layout == RK_NPU_MM_LAYOUT_NATIVE ?
        RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8 : RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;
    p.cfg.n_tile = o.n_tile;
    p.cfg.core_mask = mask;
    p.cfg.timeout_ms = o.timeout_ms;
    const auto precision = precision_for(d.a_type);
    int rc = detail::float_batch_query(1, &p.cfg, precision, &p.one);
    if (rc) return rc;
    rc = detail::float_batch_query(p.items, &p.cfg, precision, &p.info.device);
    if (rc) return rc;
    const auto& z = p.info.device;
    for (auto bytes : {z.input_bytes, z.weight_bytes, z.output_bytes,
                      z.regcmd_bytes, z.task_bytes})
        if (bytes > INT32_MAX) return RK_NPU_MM_ERR_UNSUPPORTED;
    p.info.actual_npu_core_mask = mask;
    while (__builtin_popcount(p.info.actual_npu_core_mask) > z.num_tasks)
        p.info.actual_npu_core_mask = p.info.actual_npu_core_mask == 7 ? 3 : 1;
    return RK_NPU_OK;
}

template<class T>
int pack_typed(const rk_npu_mm_plan* p, const void* host, uint64_t bytes,
         rk_npu_mem* packed, bool weight) {
    if (!p) return RK_NPU_ERR_PARAM;
    const auto& i = p->info;
    const auto& d = i.desc;
    const uint64_t required = weight ? i.host_b_bytes : i.host_a_bytes;
    const uint64_t total = weight ? i.device.weight_bytes : i.device.input_bytes;
    int rc = conversion_buffers(host, bytes, required, packed, total, sizeof(T));
    if (rc) return rc;
    const auto* src = static_cast<const T*>(host);
    auto* dst = static_cast<T*>(packed->vaddr);
    // Preserve the existing vectorized batch packers for compact unsplit IO.
    if constexpr (sizeof(T) == 2) { if (i.split_count == 1) {
        if (weight && !d.trans_b && d.ldb == (uint64_t)d.N &&
            (d.batch_count == 1 || d.batch_stride_b == (uint64_t)d.K * d.N))
            return rk_npu_matmul_f16_batch_pack_b(d.batch_count, &p->cfg, src, packed);
        if (!weight && !d.trans_a && d.lda == (uint64_t)d.K &&
            (d.batch_count == 1 || d.batch_stride_a == (uint64_t)d.M * d.K))
            return rk_npu_matmul_f16_batch_pack_a(d.batch_count, &p->cfg, src, packed);
    } }
    std::memset(dst, 0, total);
    const int ka = 64 / sizeof(T), atom = 16 / sizeof(T);
    const int ak = align_up(i.part_k, ka);
    const uint64_t per = (weight ? p->one.weight_bytes : p->one.input_bytes) / sizeof(T);
    // Transpose and source strides are fused into the device-layout write.
    // Each worker owns a whole N16 weight tile or A row; writes are disjoint.
    const int units = weight ? ceil_div(d.N, 16) : d.M;
    const int64_t work = (int64_t)p->items * units;
#pragma omp parallel for schedule(static) if(total >= (1u << 17))
    for (int64_t job = 0; job < work; ++job) {
        const int item = job / units, u = job % units;
        const int batch = item / i.split_count, split = item % i.split_count;
        const int k0 = split * i.part_k;
        const int valid_k = std::min(i.part_k, d.K - k0);
        T* out = dst + (uint64_t)item * per;
        const T* in = src + (uint64_t)batch *
                            (weight ? d.batch_stride_b : d.batch_stride_a);
        if (weight) {
            for (int n = u * 16; n < std::min(d.N, u * 16 + 16); ++n)
                for (int k = 0; k < valid_k; ++k) {
                    const uint64_t s = d.trans_b ? (uint64_t)n * d.ldb + k0 + k :
                                                   (uint64_t)(k0 + k) * d.ldb + n;
                    const uint64_t t = ((uint64_t)(n / 16) * (ak / ka) + k / ka) * 16 * ka +
                                       (n % 16) * ka + k % ka;
                    out[t] = in[s];
                }
        } else {
            for (int k = 0; k < valid_k; ++k) {
                const uint64_t s = d.trans_a ? (uint64_t)(k0 + k) * d.lda + u :
                                               (uint64_t)u * d.lda + k0 + k;
                const uint64_t t = i.options.a_layout == RK_NPU_MM_LAYOUT_NATIVE ?
                    ((uint64_t)(k / atom) * d.M + u) * atom + k % atom : (uint64_t)u * ak + k;
                out[t] = in[s];
            }
        }
    }
    return RK_NPU_OK;
}

int pack(const rk_npu_mm_plan* p, const void* host, uint64_t bytes,
         rk_npu_mem* packed, bool weight) {
    if (!p) return RK_NPU_ERR_PARAM;
    return p->info.desc.a_type == RK_NPU_MM_TF32 ?
        pack_typed<float>(p, host, bytes, packed, weight) :
        pack_typed<uint16_t>(p, host, bytes, packed, weight);
}

int unpack_f32(const rk_npu_mm_plan& p, const rk_npu_mem* packed, void* host, uint64_t bytes) {
    const auto& i = p.info;
    const auto& d = i.desc;
    const int rc = conversion_buffers(host, bytes, i.host_c_bytes, packed, i.device.output_bytes, 4);
    if (rc) return rc;
    const auto* in = static_cast<const float*>(packed->vaddr);
    auto* out = static_cast<float*>(host);
    const int an = align_up(d.N, 32), nt = i.options.n_tile ? i.options.n_tile : an;
    const uint64_t per = p.one.output_bytes / 4;
    const int64_t rows = (int64_t)d.batch_count * d.M;
#pragma omp parallel for schedule(static) if(rows * d.N >= (1u << 17))
    for (int64_t row = 0; row < rows; ++row) {
        const int batch = row / d.M, m = row % d.M;
        for (int n = 0; n < d.N; ++n) {
            const int n0 = n / nt * nt, width = std::min(nt, an - n0);
            const uint64_t offset = i.options.c_layout == RK_NPU_MM_LAYOUT_NATIVE ?
                ((uint64_t)(n / 4) * d.M + m) * 4 + n % 4 :
                (uint64_t)d.M * n0 + (uint64_t)m * width + n - n0;
            const uint64_t base = (uint64_t)batch * i.split_count * per + offset;
            float value = in[base];
            for (int s = 1; s < i.split_count; ++s) value += in[base + s * per];
            out[(uint64_t)batch * d.batch_stride_c + (uint64_t)m * d.ldc + n] = value;
        }
    }
    return RK_NPU_OK;
}

struct BusyGuard {
    rk_npu_mm_workspace* w;
    explicit BusyGuard(rk_npu_mm_workspace* p) : w(p) {
        if (w && w->busy.test_and_set(std::memory_order_acquire)) w = nullptr;
    }
    ~BusyGuard() { if (w) w->busy.clear(std::memory_order_release); }
};

int device_buffers(rk_npu_mm_workspace& w, rk_npu_mem* a, rk_npu_mem* b, rk_npu_mem* c) {
    if (!a || !b || !c) return RK_NPU_ERR_PARAM;
    const auto& z = w.plan.info.device;
    rk_npu_mem* mems[] = {a, b, c};
    const uint64_t sizes[] = {z.input_bytes, z.weight_bytes, z.output_bytes};
    for (int j = 0; j < 3; ++j) {
        const auto& m = *mems[j];
        if (m.ctx_id != w.domain->ctx->id) return RK_NPU_ERR_PARAM;
        if (m.iommu_domain_id != w.domain->id) return RK_NPU_ERR_DOMAIN;
        if (m.size < sizes[j]) return RK_NPU_ERR_NOMEM;
        if (m.dma_addr > UINT32_MAX || sizes[j] > (UINT64_C(1) << 32) - m.dma_addr ||
            m.dma_addr % 16) return RK_NPU_ERR_PARAM;
    }
    for (int j = 0; j < 2; ++j)
        if (c->dma_addr < mems[j]->dma_addr + sizes[j] &&
            mems[j]->dma_addr < c->dma_addr + sizes[2]) return RK_NPU_ERR_PARAM;
    return RK_NPU_OK;
}
int submit(rk_npu_mm_workspace& w, rk_npu_mem* a, rk_npu_mem* b, rk_npu_mem* c) {
    int rc = device_buffers(w, a, b, c);
    if (rc) return rc;
    return detail::float_batch_run(w.domain->ctx, w.execution, a, b, c);
}
int host_run(rk_npu_mm_workspace* w, const void* a, uint64_t ab,
             const void* b, uint64_t bb, const rk_npu_mm_packed_b* packed,
             void* c, uint64_t cb) {
    if (!w) return RK_NPU_ERR_PARAM;
    BusyGuard lock(w);
    if (!lock.w) return RK_NPU_ERR_BUSY;
    const auto& i = w->plan.info;
    const int alignment = dtype_bytes(i.desc.a_type);
    if (w->mode != (packed ? RK_NPU_MM_HOST_PACKED_B : RK_NPU_MM_HOST_DYNAMIC))
        return RK_NPU_ERR_PARAM;
    if (!addressable(a, i.host_a_bytes, alignment) || !addressable(c, i.host_c_bytes, alignment) ||
        (!packed && !addressable(b, i.host_b_bytes, alignment))) return RK_NPU_ERR_PARAM;
    if (ab < i.host_a_bytes || cb < i.host_c_bytes ||
        (!packed && bb < i.host_b_bytes)) return RK_NPU_ERR_NOMEM;
    if (overlap(a, i.host_a_bytes, c, i.host_c_bytes) ||
        (!packed && overlap(b, i.host_b_bytes, c, i.host_c_bytes))) return RK_NPU_ERR_PARAM;
    if (packed) {
        const auto& q = packed->plan.info;
        if (packed->domain->ctx != w->domain->ctx) return RK_NPU_ERR_PARAM;
        if (packed->domain->id != w->domain->id) return RK_NPU_ERR_DOMAIN;
        if (q.desc.a_type != i.desc.a_type || q.desc.b_type != i.desc.b_type ||
            q.desc.c_type != i.desc.c_type ||
            q.desc.K != i.desc.K || q.desc.N != i.desc.N ||
            q.desc.batch_count != i.desc.batch_count || q.split_count != i.split_count ||
            q.part_k != i.part_k) return RK_NPU_ERR_PARAM;
    }
    rk_npu_mem* weight = packed ? const_cast<rk_npu_mem*>(&packed->mem) : &w->b;
    int rc = rk_npu_mm_pack_a(&w->plan, a, ab, &w->a);
    if (!rc && !packed) rc = rk_npu_mm_pack_b(&w->plan, b, bb, &w->b);
    if (!rc) rc = rk_npu_mem_sync(w->domain->ctx, &w->a, RK_NPU_SYNC_TO_DEVICE);
    if (!rc && !packed) rc = rk_npu_mem_sync(w->domain->ctx, &w->b, RK_NPU_SYNC_TO_DEVICE);
    if (!rc) rc = submit(*w, &w->a, weight, &w->c);
    if (!rc) rc = rk_npu_mem_sync(w->domain->ctx, &w->c, RK_NPU_SYNC_FROM_DEVICE);
    if (!rc) rc = rk_npu_mm_unpack_c(&w->plan, &w->c, c, cb);
    return rc;
}
} // namespace

extern "C" void rk_npu_mm_desc_init(rk_npu_mm_desc* d, int m, int n, int k, int batches) {
    if (!d) return;
    *d = {};
    d->struct_size = sizeof(*d);
    d->M = m; d->N = n; d->K = k; d->batch_count = batches;
    if (m > 0 && k > 0) d->batch_stride_a = (uint64_t)m * k;
    if (k > 0 && n > 0) d->batch_stride_b = (uint64_t)k * n;
}
extern "C" void rk_npu_mm_options_init(rk_npu_mm_options* o) {
    if (!o) return;
    *o = {};
    o->struct_size = sizeof(*o);
    o->allowed_npu_core_mask = 7;
    o->split_k = -1;
}
extern "C" int rk_npu_mm_plan_create(const rk_npu_mm_desc* d, const rk_npu_mm_options* o,
                                      rk_npu_mm_plan** out) {
    if (!out) return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!d) return RK_NPU_ERR_PARAM;
    rk_npu_mm_options defaults{};
    rk_npu_mm_options_init(&defaults);
    std::unique_ptr<rk_npu_mm_plan> p(new (std::nothrow) rk_npu_mm_plan);
    if (!p) return RK_NPU_ERR_NOMEM;
    int rc = build_plan(*d, o ? *o : defaults, *p);
    if (!rc) *out = p.release();
    return rc;
}
extern "C" int rk_npu_mm_plan_get_info(const rk_npu_mm_plan* p, rk_npu_mm_info* out) {
    if (!p || !out) return RK_NPU_ERR_PARAM;
    *out = p->info; return RK_NPU_OK;
}
extern "C" void rk_npu_mm_plan_free(rk_npu_mm_plan* p) { delete p; }
extern "C" int rk_npu_mm_pack_a(const rk_npu_mm_plan* p, const void* a, uint64_t n, rk_npu_mem* b) {
    return pack(p, a, n, b, false);
}
extern "C" int rk_npu_mm_pack_b(const rk_npu_mm_plan* p, const void* a, uint64_t n, rk_npu_mem* b) {
    return pack(p, a, n, b, true);
}
extern "C" int rk_npu_mm_unpack_c(const rk_npu_mm_plan* p, const rk_npu_mem* packed,
                                   void* host, uint64_t bytes) {
    if (!p) return RK_NPU_ERR_PARAM;
    if (p->info.desc.c_type == RK_NPU_MM_F32) return unpack_f32(*p, packed, host, bytes);
    const auto& i = p->info;
    const auto& d = i.desc;
    int rc = conversion_buffers(host, bytes, i.host_c_bytes, packed, i.device.output_bytes);
    if (rc) return rc;
    const auto* in = static_cast<const uint16_t*>(packed->vaddr);
    auto* out = static_cast<uint16_t*>(host);
    if (i.split_count == 1 && d.ldc == (uint64_t)d.N &&
        (d.batch_count == 1 || d.batch_stride_c == (uint64_t)d.M * d.N)) {
        // The legacy tiled unpacker creates memory views. CPU-only callers
        // need only vaddr/size, so supply its capacity metadata locally.
        rk_npu_mem view = *packed;
        if (!view.capacity) view.capacity = view.size;
        return rk_npu_matmul_f16_batch_unpack_d(d.batch_count, &p->cfg, &view, out);
    }
    const int an = align_up(d.N, 32), nt = i.options.n_tile ? i.options.n_tile : an;
    const uint64_t part_stride = p->one.output_bytes / 2;
    const int64_t rows = (int64_t)d.batch_count * d.M;
    const auto decode = d.c_type == RK_NPU_MM_BF16 ? bits::bfloat_to_float : bits::half_to_float;
    const auto encode = d.c_type == RK_NPU_MM_BF16 ? bits::float_to_bfloat : bits::float_to_half;
#pragma omp parallel for schedule(static) if(rows * d.N >= (1u << 17))
    for (int64_t row = 0; row < rows; ++row) {
        const int batch = row / d.M, m = row % d.M;
        for (int n = 0; n < d.N; ++n) {
            uint64_t offset;
            if (i.options.c_layout == RK_NPU_MM_LAYOUT_NATIVE)
                offset = ((uint64_t)(n / 8) * d.M + m) * 8 + n % 8;
            else {
                const int n0 = n / nt * nt, width = std::min(nt, an - n0);
                offset = (uint64_t)d.M * n0 + (uint64_t)m * width + n - n0;
            }
            const uint64_t base = (uint64_t)batch * i.split_count * part_stride + offset;
            uint16_t value = in[base];
            if (i.split_count > 1) {
                float sum = decode(value);
                for (int s = 1; s < i.split_count; ++s)
                    sum += decode(in[base + s * part_stride]);
                value = encode(sum);
            }
            out[(uint64_t)batch * d.batch_stride_c + (uint64_t)m * d.ldc + n] = value;
        }
    }
    return RK_NPU_OK;
}
extern "C" int rk_npu_mm_workspace_create(rk_npu_iommu_domain* domain,
    const rk_npu_mm_plan* p, rk_npu_mm_workspace_mode mode, rk_npu_mm_workspace** out) {
    if (!out) return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!domain || !domain->ctx || !p || mode < RK_NPU_MM_HOST_DYNAMIC ||
        mode > RK_NPU_MM_DEVICE_ONLY) return RK_NPU_ERR_PARAM;
    try {
        std::unique_ptr<rk_npu_mm_workspace> w(new rk_npu_mm_workspace);
        w->plan = *p; w->mode = mode; w->domain = domain;
        detail::retain_domain(domain);
        int rc = RK_NPU_OK;
        if (mode != RK_NPU_MM_DEVICE_ONLY) {
            rc = rk_npu_mem_alloc(domain, p->info.device.input_bytes, RK_NPU_MEM_DATA_DEFAULT, &w->a);
            if (!rc) rc = rk_npu_mem_alloc(domain, p->info.device.output_bytes, RK_NPU_MEM_DATA_DEFAULT, &w->c);
            if (!rc && mode == RK_NPU_MM_HOST_DYNAMIC)
                rc = rk_npu_mem_alloc(domain, p->info.device.weight_bytes, RK_NPU_MEM_DATA_DEFAULT, &w->b);
        }
        if (rc) return rc;
        const auto precision = precision_for(p->info.desc.a_type);
        rc = detail::float_batch_prepare(domain, p->items, &p->cfg, precision, &w->execution);
        if (rc) return rc;
        *out = w.release(); return RK_NPU_OK;
    } catch (const std::bad_alloc&) { return RK_NPU_ERR_NOMEM; }
}
extern "C" void rk_npu_mm_workspace_free(rk_npu_mm_workspace* w) { delete w; }
extern "C" int rk_npu_mm_packed_b_create(rk_npu_iommu_domain* domain,
    const rk_npu_mm_plan* p, const void* b, uint64_t bytes, rk_npu_mm_packed_b** out) {
    if (!out) return RK_NPU_ERR_PARAM;
    *out = nullptr;
    if (!domain || !domain->ctx || !p ||
        !addressable(b, p->info.host_b_bytes, dtype_bytes(p->info.desc.a_type)))
        return RK_NPU_ERR_PARAM;
    if (bytes < p->info.host_b_bytes) return RK_NPU_ERR_NOMEM;
    std::unique_ptr<rk_npu_mm_packed_b> w(new (std::nothrow) rk_npu_mm_packed_b);
    if (!w) return RK_NPU_ERR_NOMEM;
    w->plan = *p; w->domain = domain;
    detail::retain_domain(domain);
    int rc = rk_npu_mem_alloc(domain, p->info.device.weight_bytes, RK_NPU_MEM_DATA_DEFAULT, &w->mem);
    if (!rc) rc = rk_npu_mm_pack_b(p, b, bytes, &w->mem);
    if (!rc) rc = rk_npu_mem_sync(domain->ctx, &w->mem, RK_NPU_SYNC_TO_DEVICE);
    if (!rc) *out = w.release();
    return rc;
}
extern "C" void rk_npu_mm_packed_b_free(rk_npu_mm_packed_b* w) { delete w; }
extern "C" int rk_npu_mm_run(rk_npu_mm_workspace* w, const void* a, uint64_t ab,
    const void* b, uint64_t bb, void* c, uint64_t cb) {
    return host_run(w, a, ab, b, bb, nullptr, c, cb);
}
extern "C" int rk_npu_mm_run_packed_b(rk_npu_mm_workspace* w, const void* a, uint64_t ab,
    const rk_npu_mm_packed_b* b, void* c, uint64_t cb) {
    if (!b) return RK_NPU_ERR_PARAM;
    return host_run(w, a, ab, nullptr, 0, b, c, cb);
}
extern "C" int rk_npu_mm_run_device(rk_npu_mm_workspace* w, rk_npu_mem* a,
    rk_npu_mem* b, rk_npu_mem* c) {
    if (!w) return RK_NPU_ERR_PARAM;
    BusyGuard lock(w);
    if (!lock.w) return RK_NPU_ERR_BUSY;
    return submit(*w, a, b, c);
}

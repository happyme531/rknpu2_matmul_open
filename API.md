# API and implementation notes

See [README](README.md) for building, testing and the namespace convention.
Historical measurement notes named below belong to the parent lowlevel research
workspace and are not bundled with this standalone library. Their measurements
are configuration-specific historical results, not fresh release benchmarks.

RK3588 NPU matrix multiplication by direct register programming through the
`rknpu` DRM driver.  It does not depend on `librknnrt.so`, an RKNN model, or the
RKNN compiler.

The main single-GEMM interface is `include/rk_npu_quant_matmul.h` and exposes
five fixed-shape operators:

```text
int8 A[M,K]  * int8 B[K,N] -> int32 C[M,N]
fp16 A[M,K]  * int8 B[K,N] -> fp16  C[M,N]  (dynamic per-token A scale)
fp16 A[M,K]  * int8 B[K,N] -> fp16  C[M,N]  (caller-supplied per-token A scale)
fp32 A[M,K]  * int8 B[K,N] -> fp32  C[M,N]  (dynamic per-token A scale)
fp32 A[M,K]  * int8 B[K,N] -> fp32  C[M,N]  (caller-supplied per-token A scale)
```

The NPU performs all INT8 MACs and emits native-layout INT32 partials.  CPU
kernels directly pack/quantize A into the autotuned normal or native NPU A
layout and fuse exact split-K
reduction, output de-tiling, dequantization, and FP16/FP32 output conversion.
The old
non-cacheable data path and staging copies are not used. CPU-visible NPU data
uses native page-backed GEM with `NON_CONTIGUOUS|CACHEABLE|IOMMU`; this matches
system dma-heap CPU bandwidth while allowing every allocation to select one of
the driver's sixteen 32-bit IOMMU domains.

Additional interfaces:

- `include/rk_npu_matmul.h` temporarily keeps distinct-weight BMM
  `A[b] @ B[b]`; its redesign is deferred because BMM needs a different tuner.
- `include/rk_npu_matmul_f16.h` keeps the raw FP16×FP16 NPU path, optional
  fused FP16 MUL/ADD, and distinct-weight FP16 BMM.
- `include/rk_npu_add_rmsnorm_f16.h` exposes the validated FP16 native
  residual Add + RMSNorm plan for `M=1/128, D=4096, eps=1e-5`.
  Other shapes and full-model quality are not yet validated. See
  the operator evidence (historical note: `fused_add_rmsnorm_native_2026-08-27.md`).

## Modules

| File | Purpose |
|------|---------|
| `include/rk_npu_quant_matmul.h` | public typed single-GEMM C ABI |
| `src/rk_npu_quant_matmul.cpp` | isolated-call autotuner and blocking executor |
| `src/rk_npu_kn_plan.cpp` | persistent split-K / physical-core N-shard plan |
| `src/rk_npu_cpu_kernels.cpp` | fused OpenMP/NEON quantize, pack, reduction and dequantize kernels |
| `src/rk_npu_dcomp.cpp` | standalone normal DCOMP codec and uncalibrated FP32 weight quantization |
| `include/rk_npu_matmul.h` | temporarily retained distinct-weight BMM C ABI |
| `include/rk_npu_matmul_f16.h` | raw FP16 NPU matmul and fused MUL/ADD C ABI |
| `include/rk_npu_add_rmsnorm_f16.h` | FP16 native Add + RMSNorm dual-output C ABI |
| `src/rk_npu_add_rmsnorm_f16.cpp` | prepared fixed-shape Add + RMSNorm register chain |
| `src/rk_npu_core.cpp` | device, IOMMU-domain native GEM, submit, reset and PC-chain helpers |

## Build

The data-side CPU kernels use OpenMP.  Build natively on the board:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or cross-compile with the supplied toolchain:

```bash
cmake -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=aarch64-toolchain.cmake
cmake --build build-aarch64 -j
```

The main artifacts are `librk_npu_matmul.so`, `librk_npu_matmul.a`, `demo`,
`bench`, `bench_llm_i8`, `test_quant_matmul`, and `tune_quant_matmul`.  The public
quant header is also compiled as plain C by the `test_quant_header_c` build
target.

### Quick INT8 LLM benchmark

`bench_llm_i8` runs hidden, FFN-up, and FFN-down matrices at sequence lengths
1, 4, and 128.  M=1/4 report logical INT8 weight bandwidth; M=128 reports GOPS.
The timed public API call includes A packing, NPU work, split-K reduction, and C
unpacking, but excludes one-time B packing.  Every row also checks sampled
outputs against an exact INT32 CPU reference.

```bash
./build/bench_llm_i8
./build/bench_llm_i8 --loops 20 --warmup 3 --npu-mask 7
```

The tool uses a conservative fixed `Ktile<=2048`, `Ntile<=3072` strategy so it
starts quickly; use the autotune API below when comparing deployment-optimal
strategies.

## Integer W4A4 backend

`rk_npu_matmul_i4.h` provides exact `i4i4i32` execution: signed INT4 codes,
resident packed weights, reusable workspaces, configurable M/K/N tiles,
single/multicore execution and an optional CPU/NPU pipeline. Safe INT16 NPU
partials are widened and reduced to INT32. It includes a native input producer
boundary for future fused preprocessing, without a floating-point quantization
API. See W4A4 API, implementation boundaries, tests and benchmarks (historical note: `w4a4_cpp_2026-09-20.md`).

`rk_npu_w4a4_linear.h` adds a FlatQuant-oriented floating Linear boundary. Dense
transforms run as two FP16 NPU GEMMs with transpose fused into writeback. CPU
kernels scan that native output and pack INT4 without a full FP32 activation
copy. Preprocessing uses fixed double buffers; the main INT4 execution retains
its K-wave pipeline. Partial sums stay in native INT32 layout until fused final
dequantization/output. The old dense CPU transform has been removed.

Create workspaces with `rk_npu_w4a4_linear_workspace_create(domain, &cfg, weights)`;
weight metadata fixes transform geometry before execution. This updates the
initial W4A4 Linear ABI: recompile callers and initialize its dedicated config
with `rk_npu_w4a4_linear_config_init`. `transform_batch` and `transform_ubatch`
control bounded preprocessing tiling. Integer-only W4A4 and W4A8 retain their
ABI. See the NPU Linear implementation, memory and complete-call comparison (historical note: `w4a4_npu_linear_2026-09-23.md`).

## W4A8 with FP32/FP16 activations

`rk_npu_w4a8.h` adds shared signed INT4 weights with FP32 per-output-channel
scales, dynamic per-token INT8 activations, and matching FP32/FP16 outputs.
It executes two signed INT4 rows per token through the W4A4 backend and fuses
their reconstruction into split-K reduction. The weight handle computes its
own integer correction; callers do not supply a decomposition bias. Model
linear bias, if present, is separate. See W4A8 API, numeric contract and board
results (historical note: `w4a8_cpp_2026-09-20.md`).

Weight `k_tile=0` selects a weight-certified tile up to 2048; query the resolved
tile before creating the workspace. `RK_NPU_W4A8_K_TILE_MULTIPLIER=1.5` optionally
enlarges that tile for experiments, waiving the INT16 saturation guarantee.
The default is 1; explicit positive tiles remain strictly checked. See the
multiplier comparison (historical note: `w4a8_multiplier_2026-09-20.md`) and
CPU optimization measurements (historical note: `w4a8_cpu_2026-09-20.md`).

`rk_npu_w4a8_tune.h` provides weight-aware autotuning, cached fastest/stable
strategies, and joint K selection for multiple M values. It searches K/N/M
tiles with **pipeline always enabled** and fixed CPU/NPU resource permissions.
The original APIs/strategy ABI retain native-only search. The `_ex` APIs return
`rk_npu_w4a8_strategy_ex`, searching native plus panel8 and logical ubatches up
to 256 within the CBUF budget. They use CPU reduction as the latency objective.
Create both weights and workspace with `rk_npu_w4a8_weights_create_tuned_ex`
and `rk_npu_w4a8_workspace_create_tuned_ex`; replaying only `base.config` through
the legacy workspace factory loses the input layout. Extended single-M/family
caches preserve layout and are distinct from native-only caches (format v2,
tuning revision 3). Old cache entries require retuning.

The `tune_w4a8` CLI defaults to extended search; `--native-only` selects the
legacy search. It accepts raw INT4 codes plus FP32 scales, or synthetic inputs
for experiments. See panel8 autotune API and paired measurements (historical note: `w4a8_layout_autotune_2026-09-22.md`)
and earlier tuned decode/prefill comparisons (historical note: `w4a8_autotune_2026-09-20.md`).

## CPU usage and OpenMP waiting

For latency-sensitive CPU/NPU pipelines that also need to leave CPU capacity
available, start the process with GNU OpenMP workers sleeping while idle:

```bash
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ./your_program
```

Set these before starting the process. `OMP_WAIT_POLICY=PASSIVE` selects passive
waiting; GNU libgomp's `GOMP_SPINCOUNT=0` disables its initial spin period. These
settings affect OpenMP users throughout the process; the library does not set
them automatically. They control waiting, **not the number of compute threads**:
set the W4A8 config or selected strategy's `cpu_threads` separately.

On the measured RK3588 W4A8 prefill cases, these settings reduced process CPU
time by 24–29% with complete-call latency changing by -0.3% to +2.4%. FP16
M=128/K=11008/N=4096 used about 378% CPU with default waiting versus 281% with
passive waiting, both around 4.0 ms (100% means one CPU core). A two-thread
CPU config was about 4.43 ms / 173% CPU. These are configuration-specific
measurements, not a guarantee for other workloads.

Use the same waiting policy during tuning and inference. Waiting policy is not
part of the current tuning cache key; use `tune_w4a8 --refresh` after changing it.
See the CPU usage experiment (historical note: `w4a8_split_reduce_2026-09-22.md`) and
reduction/output optimization (historical note: `w4a8_cpu_reduce_2026-09-22.md`) for the
measurement protocol, full-call latency, and CPU-time tradeoffs.

## Compressed weights

Use the `_create_compress` version of an existing typed weight constructor:

```c
rk_npu_f32i8f32_weights* w = rk_npu_f32i8f32_weights_create_compress(
    domain, &weight_config, B_i8_rowmajor, w_scale);
/* Same workspace, run and free API as ordinary weights. */
int rc = rk_npu_f32i8f32_run_dynamic(workspace, w, A_f32, C_f32);
rk_npu_f32i8f32_weights_free(w);
```

`i8i8i32` and `f16i8f16` also have `_create_compress` constructors. All retain
the exact supplied INT8 values; scaled variants copy `w_scale[N]`. Inputs are
compact **B[K,N]** and may be released after creation. Tiles that do not shrink
remain raw native INT8. NPU decompression adds no quantization error.

For ordinary FP32 weights, the two scaled families additionally provide
`_weights_create_from_f32_compress(domain, config, B_f32, target_bpw)`.
This uses **uncalibrated per-channel RTN**, not the calibrated V2 quantizer.
Its finite `(0,8]` BPW target covers the creation-time K-wave/full-N payload,
including lane headers/alignment but excluding scales. Invalid inputs or an
unmet search budget return NULL. Prefer the INT8+scale entry point to preserve
an existing calibrated quantization result.

The first run for each new N partition prepares and caches that compressed
layout; warm up before timing. Different M and core masks reuse a matching
partition. Multiple used N partitions retain multiple copies. No duplicate
raw matrix is retained. Changing N partitions can change actual BPW, and the
existing shape-only memory query/autotuner still describes raw weights.

See correctness, timing, quality and startup costs (historical note: `dcomp_cpp_library_2026-09-13.md`).
Tests: `test_dcomp_codec` (host or board), `test_compress` (RK3588).

## Autotune API

Autotune takes only shape, measurement settings, and CPU/NPU permission masks.
It deliberately does **not** take user input tensors: deterministic synthetic
A, B, activation scales, and weight scales are generated internally from a
fixed seed and discarded before return.

```c
#include "rk_npu_quant_matmul.h"

rk_npu_ctx *ctx = rk_npu_open(NULL);
rk_npu_matmul_autotune_config tc;
rk_npu_matmul_autotune_config_init(&tc, M, N, K);
tc.allowed_npu_core_mask = 0x7;   /* permission, not a forced choice */
tc.allowed_cpu_core_mask = 0xf0;  /* Linux CPU4..7 */

rk_npu_matmul_strategy fastest, stable;
int rc = rk_npu_f16i8f16_dynamic_autotune(
    ctx, &tc, &fastest, &stable);
```

Use the matching entry point for each operator:

```c
rk_npu_i8i8i32_autotune(...);
rk_npu_f16i8f16_dynamic_autotune(...);
rk_npu_f16i8f16_static_autotune(...);
rk_npu_f32i8f32_dynamic_autotune(...);
rk_npu_f32i8f32_static_autotune(...);
```

`fastest` minimizes median independent blocking-call latency.  `stable`
minimizes a jitter-penalized score. CPU/NPU resources are fixed before tuning.
The timed region includes input quantize/pack, all NPU waves, dma-buf
ownership sync, and output reduction/conversion.  K waves within that one call
use the production CPU/NPU pipeline.  Static B packing and plan construction
are not timed; separate matmul requests are never overlapped.

### Search algorithm

The search space is derived from the dataflow instead of exposing arbitrary
register values:

1. Select resources once from the permission masks. NPU chooses the widest
   supported subset (`7`, `3`, then a permitted single core) that fits N's
   32-channel groups. CPU uses up to four permitted CPUs, preferring cores 4–7.
   There is **no timing search over core masks or thread counts**. Supply a
   narrower permission mask to use fewer resources.
2. Estimate the CBUF transition and derive the small aligned K-tile grid.
   Add a <=2048-wave anchor, respecting `required_k_tile`. The interval
   `(3072,4096)`, containing the recorded K3840 submit failure, is conservatively
   excluded for M>4, including K tails.
   This conservative policy is not a universal hardware safety formula.
3. Derive N tiles from the fixed core count and `Ntile <= 3072`.
4. Enumerate complete layout recipes: normal A/native C, native A/native C,
   `pn8`, `pn16`, `pc8`, `p8`, `p16`. A layout is part of each recipe, not an
   additional Cartesian search. Normal/native remain useful fallbacks.
5. Validate every K-wave's dimensions before submit. Current panel-A guards:
   width 8 requires Ktile<=2048; width 16 requires Ktile<=4096. M and M tiles
   must be divisible by the panel width. Panel C also requires its full-row DPU
   notch to fit 13 bits; CPU addressing includes align32(N) padding.
6. Require two exact INT32 comparisons against the anchor, then measure every
   legal candidate through the complete production pipeline. There is no
   NPU-only top-1 pruning. Validate typed output after every timed call (outside
   its measured interval), so repeated-submit or CPU-conversion errors exclude
   the candidate. Submit failures still stop tuning.

`pnW` means panel A plus existing native C; `pc8` means normal A plus panel C;
`pW` means both sides use panels. Within an M panel the layouts are
`A[M/W,align32(Ktile)/16,W,16]` and `C[M/W,align32(N)/4,W,4]`.
CPU kernels fuse panel addresses into quantize/pack and reduce/dequant, with no
extra tensor-sized transpose. Dynamic scales are still computed over full K.
The search skips panel recipes for M1/M4 where experiments found no benefit.

These guards exclude known bad configurations, but do not replace correctness
checks. A faster NPU layout can lose after CPU conversion; returning a legacy
recipe is intentional. The shape-only tuner continues to profile raw weights,
not a caller's particular DCOMP payload.

The returned strategy is a plain value and may be cached or serialized. Its
`a_layout` and `c_layout` are execution details; public A/C remain compact row-major.
`c_layout=0` means the previous native C format. The alpha strategy struct was
extended: rebuild callers together with the library; do not reuse old binary
struct serializations. Tuning cache format v2 / revision 3 rejects older entries. The
CPU mask still has to be available in the process affinity when the workspace
is created.

Board validation and measured pipeline results for the panel search are recorded
in the integration note (historical note: `quant_autotune_panels_2026-09-17.md`).

### Tuning result cache

`rk_npu_matmul_autotune_cached` serializes the existing `fastest` and `stable`
strategy values; it does not introduce a second plan representation.  On a
matching cache hit it returns those values without allocating NPU buffers or
running any tuning candidates:

```c
int cache_hit = 0;
int rc = rk_npu_matmul_autotune_cached(
    ctx, &tc, RK_NPU_MATMUL_F16I8F16_DYNAMIC,
    NULL,              /* default per-user cache path */
    0,                 /* nonzero forces retuning */
    &fastest, &stable, &cache_hit);
```

The default file lives below `$XDG_CACHE_HOME/rk_npu_matmul`, falling back to
`$HOME/.cache/rk_npu_matmul`.  Its filename hashes the full cache key.  A key
contains operator/shape, effective CPU affinity, allowed NPU mask,
`required_k_tile`, measurement settings, driver version, format version, and a
manually bumped tuning revision.  Consequently a result is shareable by all
weights with the same shape, but is not silently reused after a relevant
driver, search-space, pipeline, or CPU-kernel change.

The file itself is versioned text with individually encoded fields rather than
a raw `fwrite(sizeof(rk_npu_matmul_strategy))` binary image. Writes use a temporary
file plus atomic rename.  Missing, malformed, stale, or mismatched files are
ordinary cache misses; failure to save a newly tuned result does not discard
that result.  Applications that manage paths themselves can use
`rk_npu_matmul_autotune_cache_load` and
`rk_npu_matmul_autotune_cache_save` directly.

The `tune_quant_matmul` CLI enables this cache by default.  Use
`--cache-path FILE` for an explicit file, `--refresh-cache` to replace an entry,
or `--no-cache` for the original always-tune behavior.

For changing LLM sequence lengths, family autotune accepts a list of M values
and optional relative call frequencies.  It selects one common K tile for the
packed weights and returns an independently optimized execution strategy for
each M.  A later M can be added with ordinary autotune by setting
`required_k_tile` to the existing weight configuration.

```c
int m_values[] = {1, 32, 128};
double frequencies[] = {100.0, 4.0, 1.0};
rk_npu_matmul_family_autotune_config fc;
rk_npu_matmul_family_autotune_config_init(&fc, N, K, 3, m_values);
fc.frequencies = frequencies;
rk_npu_matmul_family_summary family;
rk_npu_matmul_strategy per_m[3];
rk_npu_f16i8f16_dynamic_autotune_family(
    ctx, &fc, &family, per_m, NULL, NULL);
```

`rk_npu_matmul_autotune_family_cached` runs the same family selection while
reusing the single-M cache entry for every derived `(M, required_k_tile)` pair.
It does not serialize a second family-specific result format.  Adding an M only
tunes the missing pairs; changing frequencies reuses all measurements and only
recomputes the weighted winner.  `cache_directory == NULL` uses the normal
per-user cache.  An explicit value is a directory containing the hashed
single-M entries.  Optional hit/miss counters report partial reuse.

`family.weight_config` is used once when packing each logical layer weight;
`per_m[i]` creates the cached workspace for `m_values[i]`.

## Typed execution

Packed weights and execution workspaces have independent lifetimes.  One
immutable weight handle owns B (and fp16-path weight scales) for one
`(K,N,Ktile)` partition.  It can serve any exact-M workspace with that
partition, even when N tile and CPU/NPU masks differ.  The caller caches one
workspace per M or application-defined padded bucket.

Both objects are also bound to one `rk_npu_iommu_domain`. One submit can only
dereference addresses from its active domain, so packed B, A slots, partial C,
regcmd and task descriptors must all use the same domain. A mismatch is rejected
with `RK_NPU_ERR_DOMAIN` before submission.

```c
rk_npu_iommu_domain *domain = rk_npu_iommu_domain_create(ctx, domain_id);
/* Create all weights and workspaces resident in this address space. */
```

The handle may be released after object creation because weights/workspaces
retain its state. `rk_npu_ctx` must remain alive until every object is freed.
The driver switches the entire NPU device, not individual cores, so different
domains provide capacity rather than concurrent execution.

A uses two native-GEM allocation ping-pong slots; the bytes inside use the
strategy's normal/native A layout. C uses one native-INT32 accumulator plus
two partial ping-pong slots (only the slots required by the wave count are
allocated).  Separate slots allow CPU packing/reduction to overlap NPU access
without violating cache ownership.  Each logical matrix packs all B
waves into one weight arena exactly once.  Regcmd/task buffers stay per M
workspace because they contain M-dependent task geometry.

### INT8 × INT8 → INT32

```c
rk_npu_matmul_weight_config wc = {K, N, fastest.k_tile};

rk_npu_matmul_workspace *workspace =
    rk_npu_matmul_workspace_create(domain, &fastest);
rk_npu_i8i8i32_weights *weight0 =
    rk_npu_i8i8i32_weights_create(domain, &wc, B0_i8_rowmajor);
rk_npu_i8i8i32_weights *weight1 =
    rk_npu_i8i8i32_weights_create(domain, &wc, B1_i8_rowmajor);

for (...) {
    rk_npu_i8i8i32_run(workspace, weight0, A0_i8_rowmajor, C0_i32_rowmajor);
    rk_npu_i8i8i32_run(workspace, weight1, A1_i8_rowmajor, C1_i32_rowmajor);
}

rk_npu_i8i8i32_weights_free(weight1);
rk_npu_i8i8i32_weights_free(weight0);
rk_npu_matmul_workspace_free(workspace);
rk_npu_iommu_domain_free(domain);
```

There is no quantization or dequantization.  Split-K partials are added modulo
`2^32`, exactly matching the NPU INT32 accumulator, while native NPU output is
converted directly to compact row-major C.

### FP16 × INT8 → FP16, dynamic per-token

```c
rk_npu_matmul_workspace *workspace =
    rk_npu_matmul_workspace_create(domain, &fastest);
rk_npu_matmul_weight_config wc = {K, N, fastest.k_tile};
rk_npu_f16i8f16_weights *weights = rk_npu_f16i8f16_weights_create(
    domain, &wc, B_i8_rowmajor, w_scale_per_output_channel);

rk_npu_f16i8f16_run_dynamic(workspace, weights, A_fp16_bits, C_fp16_bits);
rk_npu_f16i8f16_weights_free(weights);
rk_npu_matmul_workspace_free(workspace);
```

For every row `m`:

```text
a_scale[m] = max(abs(A[m,:])) / 127       (all-zero row uses 1)
Aq[m,k]    = clamp(round(A[m,k] / a_scale[m]), -127, 127)
C[m,n]     = fp16(acc_i32[m,n] * a_scale[m] * w_scale[n])
```

Quantization writes directly into each split-K A buffer in the strategy's
normal/native layout; there is no compact INT8 A intermediate.

### FP16 × INT8 → FP16, static per-token

```c
rk_npu_matmul_workspace *workspace =
    rk_npu_matmul_workspace_create(domain, &fastest);
rk_npu_matmul_weight_config wc = {K, N, fastest.k_tile};
rk_npu_f16i8f16_weights *weights = rk_npu_f16i8f16_weights_create(
    domain, &wc, B_i8_rowmajor, w_scale_per_output_channel);

rk_npu_f16i8f16_run_static(
    workspace, weights, A_fp16_bits, a_scale_per_token, C_fp16_bits);
rk_npu_f16i8f16_weights_free(weights);
rk_npu_matmul_workspace_free(workspace);
```

`a_scale` has shape `[M]` and is supplied on every call.  Static per-tensor was
intentionally not exposed: it loses substantially more accuracy and the board
microbench found no meaningful speed advantage over static per-token.

For both FP16 APIs, `B[K,N]` is packed and positive finite `w_scale[N]` is copied
during weight creation; both caller inputs may then be released.  FP16 tensors
are raw IEEE binary16 bits in `uint16_t`.  Calls are blocking.  One workspace is
non-reentrant: accidental concurrent runs return `RK_NPU_ERR_BUSY`.  Weight and
workspace handles are independent, but both must outlive a call and their
`rk_npu_ctx`.

### FP32 × INT8 → FP32

The FP32 API mirrors the FP16 dynamic/static per-token contracts while sharing
the same split-K/NPU plan and packed INT8 weight representation:

```c
rk_npu_matmul_workspace *workspace =
    rk_npu_matmul_workspace_create(domain, &fastest);
rk_npu_matmul_weight_config wc = {K, N, fastest.k_tile};
rk_npu_f32i8f32_weights *weights = rk_npu_f32i8f32_weights_create(
    domain, &wc, B_i8_rowmajor, w_scale_per_output_channel);

rk_npu_f32i8f32_run_dynamic(
    workspace, weights, A_fp32_rowmajor, C_fp32_rowmajor);
/* Or supply a_scale[M] explicitly with rk_npu_f32i8f32_run_static(). */

rk_npu_f32i8f32_weights_free(weights);
rk_npu_matmul_workspace_free(workspace);
```

Dynamic FP32 scale scan and the first K-wave pack share one OpenMP region.
Subsequent waves are packed just in time into the two ping-pong A slots. The
autotuned native-A kernel fuses quantization with K16 layout conversion. Native
INT32 partials use the existing fused FP32 reduction/dequantization kernel, so
this path performs no FP16 conversion.

## CLI examples

```bash
LD_LIBRARY_PATH=build ./build/tune_quant_matmul --op dynamic \
  --m 128 --k 9216 --n 2560 --npu-mask 0x7 --cpu-mask 0xf0

LD_LIBRARY_PATH=build ./build/tune_quant_matmul --op i8i8i32 \
  --m 128 --k 2560 --n 9216 --warmup 1 --loops 2 --repeats 1

LD_LIBRARY_PATH=build ./build/tune_quant_matmul --op f32-dynamic \
  --m 128 --k 8192 --n 512 --npu-mask 0x7 --cpu-mask 0xf0

LD_LIBRARY_PATH=build ./build/test_quant_matmul
LD_LIBRARY_PATH=build ./build/test_batch
```

The process needs access to the NPU DRM node, normally `/dev/dri/card1`.

## Legacy distinct-weight BMM

`include/rk_npu_matmul.h` still exposes one-submit BMM where every batch item has
its own A and B:

```c
rk_npu_matmul_i8_config cfg;
rk_npu_matmul_i8_config_init(&cfg, M, N, K);

rk_npu_matmul_sizes sizes;
rk_npu_matmul_i8_batch_query(batch, &cfg, &sizes);
rk_npu_matmul_i8_batch_pack_a(batch, &cfg, A_bmk, &input);
rk_npu_matmul_i8_batch_pack_b(batch, &cfg, B_bkn, &weight);

rk_npu_matmul_i8_batch_plan *plan =
rk_npu_matmul_i8_batch_prepare(domain, batch, &cfg);
rk_npu_matmul_i8_batch_run(ctx, plan, &input, &weight, &output);
rk_npu_matmul_i8_batch_unpack_c(batch, &cfg, &output, C_bmn);
rk_npu_matmul_i8_batch_plan_free(plan);
```

This legacy API uses caller-managed dma-bufs and explicit sync.  It is retained
only until a BMM-specific shape/topology tuner is designed; it does not share
the typed single-GEMM autotune logic.

## Raw FP16 NPU path

`include/rk_npu_matmul_f16.h` implements FP16×FP16→FP16 and optional fused
elementwise MUL/ADD against a third FP16 operand.  It is independent from the
weight-quantized interface above.  Its non-obvious operand layout and register
recipe are documented in `src/rk_npu_matmul_f16.cpp` and exercised by
`demo_f16`.

Plain FP16 keeps `align_in=align32(K)` independent of N.  `n_tile` optionally
splits aligned N into multiples of 32 no larger than `align32(N)`; tiled device
outputs are collected by the same compact row-major unpack API. `core_mask`
selects one or more physical NPU cores (`1`, `2`, `4`, `3`, or `7`), and
independent M/N/batch tasks are partitioned across them.

Normal A/D are joined by independently selectable official-runtime layouts:
`A_native[K/8,M,8]` and `D_native[N/8,M,8]`. Host inputs and outputs remain
compact row-major because the CPU kernels perform the conversion. The batch
pack/unpack kernels use one OpenMP team across all items, and normal unpack can
fuse FP16 bias addition into the same pass.

Distinct-weight FP16 BMM uses the same configuration for one item and compact
batch-major host tensors `A[B,M,K]`, `B[B,K,N]`, and `D[B,M,N]`:

```c
rk_npu_matmul_f16_config cfg;
rk_npu_matmul_f16_config_init(&cfg, M, N, K, RK_NPU_FUSE_NONE);
cfg.n_tile = 128;
cfg.core_mask = 7;

rk_npu_matmul_sizes sizes;
rk_npu_matmul_f16_batch_query(batch, &cfg, &sizes);
rk_npu_matmul_f16_batch_pack_a(batch, &cfg, A_bmk, &input);
rk_npu_matmul_f16_batch_pack_b(batch, &cfg, B_bkn, &weight);

rk_npu_matmul_f16_batch_plan *plan =
    rk_npu_matmul_f16_batch_prepare(domain, batch, &cfg);
rk_npu_matmul_f16_batch_run(ctx, plan, &input, &weight, &output);
rk_npu_matmul_f16_batch_unpack_d(batch, &cfg, &output, D_bmn);
rk_npu_matmul_f16_batch_plan_free(plan);
```

`test_batch_f16` covers distinct per-batch weights, N tails, M tiling, and
multi-core submission on real hardware. `bench_f16` additionally reports NPU,
A/B packing, output collection, cacheable-buffer, native-layout, and split-K
timings for one explicit shape.

Large-K single matrices can use the explicit two/three-way split-K API. It
submits every K slice as one batch item, then sums FP16 partial outputs in FP32
on the CPU. `rk_npu_matmul_f16_splitk_unpack_d_add_bias()` folds bias into that
final reduction and rounds only once. The current split-K API intentionally
requires plain normal A/D and `n_tile=0`.

External producer/consumer buffers can be registered with
`rk_npu_mem_import_dmabuf()`. The import borrows the caller's mmap and owns only
the RKNPU registration. Cache ownership for a system-heap import remains the
caller's responsibility through `DMA_BUF_IOCTL_SYNC`; `rk_npu_mem_sync()` is
for native RKNPU objects and is not portable to every external heap.

## Current limits

- Each packed matrix must fit in one 4 GiB IOVA domain. Total model weights may
  span up to sixteen domains; a single matrix is not split across domains.
- For LLM placement, group consecutive layers into the same domain and budget
  packed weights plus that domain's shared workspace peak. Start with a
  3.75 GiB placement ceiling to leave IOVA/control headroom.
  `TODO(iommu-domain-budget)`: replace this conservative integration default
  after measuring fragmentation with complete model loading.
- Only symmetric signed INT8 weights with zero point 0 and positive per-output-
  channel scales are supported by `f16i8f16`.
- Dynamic and static activation quantization are per-token.  Bias and activation
  fusion are not part of this API.
- The current executor is isolated/blocking.  Inter-request CPU/NPU overlap is
  deliberately left to a later scheduler instead of being hidden in autotune.
- INT8 BMM remains on its legacy interface; FP16 BMM is available from
  `rk_npu_matmul_f16.h` but does not yet have a shape/topology autotuner.
- `TODO(fp16-bmm-autotune)`: replace integration-side native-layout heuristics
  with a BMM tuner that includes pack/collect and external-dmabuf costs.
- `TODO(fp16-full-stride-n-tile)`: N-tiled DPU tasks still cannot scatter
  directly into one full-row-stride output. The hardware probe found no passing
  register combination, so zero-copy external C currently uses full N.
- `TODO(fp16-splitk-fp32-partial)`: evaluate FP32 partial output for callers
  that need to avoid the one FP16 rounding at each K slice boundary.

The derivation, discarded register/SRAM directions, K/N two-dimensional tiling
results, CPU/NPU role split, and historical two-lane experiments are in
`../docs/int8_single_core_large_k_small_n_findings_2026-07-10.md` and
`../docs/int8_kn_tiling_parallel_pipeline.md`.


## W4A8 optional low-CPU workspace

`rk_npu_w4a8_workspace_create_ex` accepts explicit input-layout and reduction
options. `RK_NPU_W4A8_REDUCE_NPU` retains GEMM partials in workspace-owned buffers
and reduces them to INT32 on the same selected NPU cores. All scratch is allocated
at workspace creation, reused across runs/compatible weight rebindings, and freed
with the workspace. No per-run tensor/device allocations or growing weight cache.
For **M <= 16**, a requested NPU reduction uses CPU reduction and allocates no NPU
reduction buffers; layout and explicit tile settings are preserved.
`rk_npu_w4a8_memory_query_ex` reports the effective backend and fixed buffer bytes
(including DMA guards, excluding shared weights, driver rounding and host overhead).

A tested prefill configuration is `INPUT_PANEL8`, logical `m_tile=128`, `n_tile=256`,
`pipeline=1`. NPU reduction requires at most 31 K waves and each actual tile's
`logical_M*padded_N/8 <= 4096`; query rejects unsupported geometry. Legacy workspace/autotune APIs retain native input/CPU reduction; the extended
latency autotuner also searches panel8 and still uses CPU reduction. Use
`OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0` when reducing CPU use is a priority;
the library does not change the process OpenMP environment.

See API example, memory accounting and paired CPU/latency measurements (historical note: `w4a8_low_cpu_2026-09-22.md`).

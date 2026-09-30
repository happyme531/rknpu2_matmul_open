# rknpu2_matmul_open test framework

Run commands below from the library root. For CPU-only testing on a Linux
host, see [the host build instructions](../README.md#cpu-only-tests-on-a-linux-host).
The parent workspace historical notes referenced here are not bundled with
the standalone library.

Two executables sweep the `int8 × int8 → fp32` matmul over a wide M / N / K grid:

| binary | what it does |
|--------|--------------|
| `test_correctness` | runs every shape, verifies vs an exact CPU reference, prints a PASS/FAIL map + a "largest passing N" capability envelope |
| `test_perf` | times the fixed-shape plan path per shape, reports avg us + GOPS, verifies once. `--official`/`--i32`/`--ac-native` also bench the vendor `rknn_matmul` (see below) |
| `test_mem_api` | runs native-GEM subrange views in domains 0/1/2 and a cross-context dmabuf import |
| `test_cpu_kernels` | host-only correctness/perf check for the OpenMP A/B pack and C/D unpack kernels; use `--perf` for microbench output |
| `test_batch` | verifies batched same-shape BMM and compares B single submits vs one batched submit |
| `test_batch_f16` | verifies distinct-weight FP16 BMM, N/M tiling, output tails, and multicore submission |
| `test_mm --cpu` / `--board` | FP16/BF16 MM/BMM: transpose/strides, broadcast, split-K, typed packed B reuse, device mode, conversion/range and ownership; only `--cpu` is registered with CTest |
| `test_mm_tf32 --cpu` / `--board` | TF32 dtype with FP32 containers/output: byte-exact packing oracle, K16 tails, multi-core/BMM/split-K, truncation boundaries, ranges and mixed-weight rejection |
| `test_mm_header_c` | new raw MM/BMM header alongside legacy C headers |
| `bench_f16` | explicit FP16 shape benchmark with normal/native A/D, cacheable memory, batch, multicore, N tiling, and split-K |
| `test_quant_matmul` | board correctness smoke for typed APIs, weight/workspace sharing, mismatch/BUSY guards, 100 domain switches, and autotuners |
| `tune_quant_matmul` | CLI for single-call pipelined autotuning of `i8i8i32`, dynamic `f16i8f16`, or static-per-token `f16i8f16` |
| `test_quant_header_c` | compile-only target that keeps `rk_npu_quant_matmul.h` valid C |
| `test_f16_header_c` | compile-only target that keeps the FP16, split-K, and dmabuf APIs valid C |
| `test_i4_cpu`, `test_i4_bounds` | host/board native INT4 packing, INT16 reduction and weight-certified K bounds |
| `test_i4` | board integer W4A4, tails, multicore, producer callbacks and pipeline correctness |
| `test_w4a8_cpu` | host/board FP32/FP16 quantization, parallel packing, tails and half conversion |
| `test_w4a8_reduction` | independent native INT16/INT32 reduction oracle, first/middle/final waves, FP32/FP16, 1–4 threads, M/N tails, guards and sums above 2^24 |
| `test_w4a8`, `test_w4a8_multiplier` | board MSD output, reusable weights/workspaces, larger tiles and explicit saturation-risk multiplier |
| `test_i4_header_c`, `test_w4a8_header_c` | compile-only targets for the integer and floating W4 C headers |
| `test_autotune_cpu` | shared timing/failure handling, W8 cache compatibility, W4 bounds and cache invalidation, caller affinity |
| `test_w4a8_tune` | board W4 tuning, selected-weight construction, full outputs, cache hits, family reuse and sampled saturation rejection |

Shared scaffolding (dimension sweeps, reference, classification, buffers) lives in
`test_common.h`; the vendor-comparison wrapper in `official_matmul.h`.

## Typed split-K / multicore-N tuning

The typed autotune functions accept only shape, NPU/CPU permission masks,
and measurement parameters.  They create deterministic synthetic A/B/scales
internally; callers do not need to provide model tensors.  The search derives a
small K-tile grid around the CBUF M-tile capacity transition and full A/C layout
recipes. CPU/NPU masks and thread counts are selected once, not benchmarked.
The tuner rejects every
candidate that is not bit-exact against a safe INT32 hardware anchor, then ranks
the remaining candidates by one complete blocking-call latency with intra-call
K-wave overlap enabled.

Family autotune takes multiple M values plus optional relative call frequencies.
It minimizes weighted end-to-end latency under one common K tile and returns a
separate N-tile/layout strategy for every M, using deterministic fixed resources.  `required_k_tile` constrains a
later single-M tune to an already packed weight partition.  The cached family
entry point reuses these same single-M records across family calls, including
partial reuse when the M list grows or frequencies change.

`tune_quant_matmul` contains no tuning logic of its own.  It caches the two
returned strategy values below the per-user cache directory by default and
prints `TUNING_CACHE HIT` or `TUNING_CACHE MISS`.  Use `--cache-path FILE`,
`--refresh-cache`, or `--no-cache` to control that behavior.  It then prints
`BEST_LATENCY` (minimum median) and `BEST_STABLE` (median plus half-range jitter
penalty), followed by workspace and immutable-weight
memory requirements.  Timed calls include activation
quantization/packing when applicable, all NPU waves, dma-buf ownership sync, and
exact INT32 reduction/de-tile/output conversion.  B packing and plan creation
remain outside the timed region.  Stage counters are accumulated work; because
A packing, NPU execution, and partial accumulation overlap, they do not sum to
the measured wall-clock total.

Non-divisible final K waves are supported. The default API measurement is eight
requests × three repeats after two warmups:

```bash
LD_LIBRARY_PATH=build ./build/tune_quant_matmul --op dynamic \
  --m 128 --k 9216 --n 2560 --npu-mask 0x7 --cpu-mask 0xf0

# Short search for a smoke/regression run:
LD_LIBRARY_PATH=build ./build/tune_quant_matmul --op i8i8i32 \
  --m 128 --k 2560 --n 9216 --warmup 1 --loops 2 --repeats 1
```

Two packed-A ping-pong buffers, ACC plus two partial-C ping-pong buffers, and
each matrix's packed-B arena use native page-backed RKNPU GEM with
`NON_CONTIGUOUS|CACHEABLE|IOMMU`; the old system dma-heap import inside this
INT8 typed-workspace allocator and the uncached-data/staging-copy paths are
removed. External producer/consumer dmabufs remain supported by the common
memory API. Separate A/C slots make CPU/device
cache ownership valid while different slots overlap. All K slices of a
row share one full-K activation scale.
Per-slice scales would make exact split-K reduction invalid.  The current public
API intentionally measures one blocking request;
the earlier two-request/two-lane throughput experiment remains documented in
`../../docs/int8_kn_tiling_parallel_pipeline.md` as historical evidence.

## CPU pack/unpack kernels

The public pack/unpack APIs call OpenMP-backed CPU kernels in
`src/rk_npu_cpu_kernels.cpp`. They cover int8 normal/native A, int8 native B,
int8 int32/fp32 raw C unpack, fp16 A/B/operand pack, and fp16 D unpack.
`test_cpu_kernels` compares those APIs against independent reference layouts
without opening the NPU.

By default `--perf` runs a small scenario suite: large asymmetric A/C/D shapes
from the native-layout experiments, plus an attention-like `1024x1024` B-pack
case. Pass `--m/--k/--n` to benchmark one explicit shape instead. Timings report
the best run across `--repeats` repeats, which reduces scheduler/devfreq noise on
these short CPU kernels:

```bash
LD_LIBRARY_PATH=build OMP_NUM_THREADS=4 taskset -c 4-7 \
  ./build/test_cpu_kernels --perf --loops 160 --repeats 3

LD_LIBRARY_PATH=build OMP_NUM_THREADS=4 taskset -c 4-7 \
  ./build/test_cpu_kernels --perf --m 128 --k 2048 --n 6144 --loops 160
```

See `docs/cpu_pack_unpack_kernels.md` for current board-side measurements.

## Vendor comparison (`test_perf --official`)

Builds against the vendor `librknnrt` and times the same GEMM through `rknn_matmul`
for a side-by-side `our/off` ratio. Board-only; see `docs/int8_vs_official.md` for
the build flags, the librknnrt-version gotcha, and the finding that we're at
single-core parity with the vendor (the headline ~1.8 TOPS is its `--ac-native`
output path). Configure with `-DRK_NPU_OFFICIAL=ON -DRKNN_INCLUDE_DIR=... -DRKNN_LIB=...`.

## Dimension sweep

The requested grid (segmented, inclusive, deduplicated):

```
M : 1–4 step 1, 4–16 step 4, 16–128 step 16, 128–1024 step 128      (21 values)
K : 16–128 step 16, 128–1024 step 128, 1024–8192 step 512           (29 values)
N : same value set as K                                             (29 values)
```

`test_correctness` sweeps the full cartesian product (21 × 29 × 29 = 17 661 shapes).
The step sizes are only a guideline — the hardware has its own alignment limits, so
many shapes are *expected* to fail until N/K tiling exists. The point of the sweep is
to map exactly where PASS turns to FAIL.

Pass `--exp` (or `--exponential`) for a smaller exponential grid:

```
M : 1,2,4,8,16,32,64,128,256,512,1024
K/N : 16,32,64,128,256,512,1024,2048,4096,8192
```

## Build & run (on the board)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
LD_LIBRARY_PATH=build ./build/test_correctness            # full map, ~minutes
LD_LIBRARY_PATH=build ./build/test_perf                   # coarse perf table, seconds
LD_LIBRARY_PATH=build ./build/test_mem_api                # domains 0/1/2 native-GEM view smoke test
LD_LIBRARY_PATH=build ./build/test_batch                  # batched single-submit smoke/perf test
LD_LIBRARY_PATH=build ./build/test_quant_matmul            # typed API/autotune smoke
```

Cross-compiling on the x86 host links against a newer glibc than the board ships, so
**build natively on the board** (or use a matching sysroot).

## Correctness output

```
M=2
  K=64    | PPPPPPPPPPXXXXXXXXXXXXXXXXXXX     <- one char per N column
...
legend: P=pass  X=WRONG  T=submit/timeout  M=nomem  A=alloc-fail  !=param  .=skip(guard)

-- largest passing N per (M,K) --   (the capability envelope)
```

The reference accumulates int8 MACs in **int64** and casts once to fp32 — matching the
NPU, which accumulates in int32 and does a single int32→fp32 convert on chip (so PASS
means a bit-exact match). Big shapes are verified by sampling a random subset of output
positions (plus a full NaN/Inf scan) so verification never dominates the sweep.

## Useful flags (both binaries)

```
--dev PATH            DRM device (default /dev/dri/card1)
--m LO:HI  --n LO:HI  --k LO:HI    restrict each axis to a sub-range
--exp                 use the smaller exponential grid
--csv PATH            write per-shape results as CSV
--max-weight-mb MB    resource guard: skip shapes whose packed B exceeds this
--max-out-mb / --max-in-mb / --max-tasks    other guard knobs
```

`test_correctness` extras: `--samples N`, `--full-budget MACS`, `--seed N`, `--only-fail`.
`test_perf` extras: `--full` (sweep the entire cartesian, not the coarse subset), `--loops N`.
Int8 layout knobs: `--a-native` packs/consumes A as `(K/16,M,16)`, and
`--c-native` writes/unpacks C as `(N/4,M,4)`. `test_perf --official --ac-native`
uses the vendor's coupled `AC_layout=NATIVE` path for comparison.

## Notes / gotchas

- **Data buffers are allocated once and reused.** Plans allocate/free their own
  internal `regcmd`/`task` scratch, while A/B/C buffers are sized to the guard
  caps up front.  The shared test buffers use `RK_NPU_MEM_DEFAULT`
  (cacheable) and explicitly call `rk_npu_mem_sync()` after packing inputs and
  before unpacking outputs; the library run path does not sync implicitly.
- The **resource guard** skips shapes that don't fit those cap buffers (shown as `.`).
  Raise `--max-weight-mb` etc. to push the envelope further (and the one-time allocation
  grows accordingly).
- A too-large shape returns `RK_NPU_ERR_SUBMIT` after the driver's ~6 s timeout. The
  library issues a real NPU reset only after a submit failure, so a bad shape should
  recover and show `T` without paying reset cost on every successful job.

## Panel-layout regression

`test_panel_layout` without arguments is a host/board CPU regression for independent
scalar layout oracles, padded N/K, full-token quantization, split-K accumulation,
and fused FP16/FP32 output. It is registered with CTest.

On RK3588, `test_panel_layout --board M K N Ktile Ntile mask loops [--compressed]`
checks all legal recipes against exact CPU INT32 GEMM, plus typed static/dynamic
outputs. `tools/layout_probe/validate_autotune.py` runs the recorded shape/mask
coverage under an exclusive lock and temporarily fixed governors, restores them
on failure, and stops on any failing command.

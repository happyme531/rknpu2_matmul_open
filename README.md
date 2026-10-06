# rknpu2_matmul_open

[![Proudly Vibe Coded - Plasma Mix](https://vibecoded.fyi/badges/flat/main/proudly-vibe-coded-plasma-mix.svg)](https://vibecoded.fyi/)
[![Coded with Codex](https://vibecoded.fyi/badges/flat/agents/codex.svg)](https://vibecoded.fyi/)

Fully open source high-performance matrix multiplication library for LLM inference for the RK3588 RKNPU2.

## Features

- fp16/bf16/tf32 batched matmul kernels
- w8a8 int8 per-channel per-token quantized matmul kernel, with hardware weight (de)compression support
- w4a8 per-channel per-token quantized matmul kernel implemented with Mixed-precision Split-activation Decomposition
- w4a4 FlatQuant linear kernel
- w8a8/w4a8 quantized MoE kernel with fused SwiGLU activation
- Very fast fused fp16 GQA (MHA, MQA) kernel
- Fast fused fp16 MLA/gated MLA kernel, supporting both expanded and absorbed mode
- Optimized performance (at least I tried to do so)
- Some of these kernels support auto-tuning for even better performance

## Usage

Since this library is 100% coded by LLM, just clone this repo then fire up your coding agents like OpenAI Codex, Claude Code, Cursor, OpenCode, dsh etc. They will know what to do next.

## Note

- This is a very experimental project and all the things and APIs could change.
- The code may be buggy.
- Memory usage may be larger than needed (shouldn't be disastrous)

## TODO

- Optimized internal memory usage
- Add support for mainline `rocket` kernel driver
- More fused kernels
- Finding if more existing CPU processing can be offloaded to NPU
- Write a research paper if possible

## License

Original contributions to this project are licensed under the
[GNU Affero General Public License, version 3 only](LICENSE)
(`AGPL-3.0-only`). The software is provided without warranty.

Third-party material retains its applicable rights and licensing terms.
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and the
outstanding upstream licensing clarification.

## References

**RK3588 and NPU software**

- Rockchip RK3588 Technical Reference Manual V1.0 — [Part 1](https://dl.khadas.com/products/edge2/datasheet/rockchip-rk3588-trm-v1.0-part1.pdf) and [Part 2](https://dl.khadas.com/products/edge2/datasheet/rockchip-rk3588-trm-v1.0-part2.pdf), hosted by Khadas; SoC architecture and register documentation.
- [allbilly/rk3588](https://github.com/allbilly/rk3588) — RK3588 NPU programming examples, including FP16 and INT8 matrix multiplication.
- [allbilly/npu](https://github.com/allbilly/npu) — C userspace driver and operator examples for the RK3588 NPU.
- [allbilly/rknpu_driver](https://github.com/allbilly/rknpu_driver) — RKNPU kernel driver source.
- [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) — RK3588 NPU matrix multiplication examples and LLM integration experiments.
- [Mesa Rocket](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/gallium/drivers/rocket) ([GitHub mirror](https://github.com/chaotic-cx/mesa-mirror/tree/main/src/gallium/drivers/rocket)) — Mesa's Gallium NPU backend for RK3588.
- [Armbian linux-rockchip](https://github.com/armbian/linux-rockchip) — Rockchip Linux kernel tree used by Armbian, including RKNPU, DRM/GEM and DMA buffer interfaces.
- [Ling-3.0-tiny-RKNN / Ling3RKNN](https://huggingface.co/Sariel00/Ling-3.0-tiny-RKNN) — C++ Ling-3.0-tiny inference engine for RK3588, including quantized linear and MoE execution; the public project corresponding to the local MindNano reference snapshot.

**Accelerator architecture and compilers**

- NVDLA — [Documentation](https://github.com/nvdla/doc), [hardware](https://github.com/nvdla/hw), [software](https://github.com/nvdla/sw) and [virtual platform](https://github.com/nvdla/vp) for the NVIDIA Deep Learning Accelerator.
- [soDLA](https://github.com/soDLA-publishment/soDLA) — Chisel implementation of NVDLA with Chipyard integration.
- [ONNC](https://github.com/ONNC/onnc) — ONNX compilation framework for deep learning accelerators, including an NVDLA backend.

---
# AI maintained documentation start (they're not good at writing documentation)

Open userspace matrix multiplication for the **RK3588 RKNPU2**, with direct
register programming through the Linux `rknpu` DRM driver. The library does
not require `librknnrt.so`, an RKNN model, or the RKNN compiler. Its main focus
is matrix multiplication for LLM inference.

The NPU executes matrix products; CPU OpenMP/NEON kernels handle packing,
quantization, split-K reduction and output conversion. Prepared weights and
workspaces can be reused across calls.

Typed W8A8 also has optional NPU split-K reduction and dequantization, both
disabled by default. Set `RK_NPU_I8_NPU_REDUCE=1` before tuning and creating
workspaces to use an INT32 DPU accumulation chain. Add
`RK_NPU_W8A8_NPU_DEQUANT=1` to perform dequantization on the NPU using
**FP16-rounded scales**; enabling dequantization also enables the chain.
The switches apply to uncompressed W8 weights with `M<=128`, `K<=131071`;
raw INT8 output supports the reduction switch. Existing layout recipes remain
available, and each backend has a distinct tuning cache identity. For limits
and precision details, see [the typed API header](include/rk_npu_quant_matmul.h).
The optional chain packs all dynamic FP16/FP32 K slices in one CPU parallel
region. NPU dequantization reuses synchronized channel coefficients while
their contents remain unchanged; switching weights refreshes them. See the
optimization measurements (historical parent-workspace note: `w8a8_dpu_simple_opt_2026-09-29.md`).
Every variable the library reads is listed in [ENV.md](ENV.md).

## Supported interfaces

| Interface | Public header | Notes |
| --- | --- | --- |
| Typed W8A8 GEMM | [rk_npu_quant_matmul.h](include/rk_npu_quant_matmul.h) | INT8 → INT32, or FP16/FP32 input/output with dynamic or supplied per-token activation scales; autotuning and lossless DCOMP weights |
| Integer W4A4 | [rk_npu_matmul_i4.h](include/rk_npu_matmul_i4.h) | Signed INT4 × INT4 → INT32, certified INT16 partial bounds and split-K |
| Floating W4A8 | [rk_npu_w4a8.h](include/rk_npu_w4a8.h) | Packed INT4 weights, dynamic INT8 activations, FP16/FP32 output |
| W4A8 autotuning | [rk_npu_w4a8_tune.h](include/rk_npu_w4a8_tune.h) | Weight-aware tile/layout search and strategy caches |
| FlatQuant W4A4 Linear | [rk_npu_w4a4_linear.h](include/rk_npu_w4a4_linear.h) | NPU FP16 transforms followed by INT4 execution |
| Routed W8 MoE | [rk_npu_moe_w8.h](include/rk_npu_moe_w8.h) | SwiGLU experts and optional shared expert; [MoE guide](docs/MOE.md) |
| Raw FP16 GEMM/BMM | [rk_npu_matmul_f16.h](include/rk_npu_matmul_f16.h) | Optional fused MUL/ADD and explicit split-K |
| Experimental FP16 Attention | [rk_npu_attention_f16.h](include/rk_npu_attention_f16.h) | Causal/no-mask/broadcast boolean GQA, shared workspace and growing native KV; [guide](docs/ATTENTION.md) |
| Experimental FP16 Gated MLA | [rk_npu_mla_f16.h](include/rk_npu_mla_f16.h) | Ling tiny H16/QK192/V128, expanded or absorbed latent512 KV; [guide](docs/MLA.md) |
| Unified raw MM/BMM (new) | [rk_npu_mm.h](include/rk_npu_mm.h) | FP16/BF16/TF32, transpose/strides, dynamic or typed packed B, multicore and default-enabled automatic split-K; [guide](docs/MM.md) |
| Legacy INT8 BMM | [rk_npu_matmul.h](include/rk_npu_matmul.h) | Distinct weights per batch item |
| Add + RMSNorm | [rk_npu_add_rmsnorm_f16.h](include/rk_npu_add_rmsnorm_f16.h) | Fixed shapes: M=1/128, D=4096, eps=1e-5 |

See [API and implementation notes](docs/API.md) for ownership, layouts, tuning,
compression, execution examples and detailed limits. Experimental private W4
MoE support is described separately in the [MoE guide](docs/MOE.md).

## Build on RK3588

Requirements: Linux with a compatible Rockchip `rknpu` DRM driver, CMake 3.16+,
a C compiler, a C++17 compiler, and OpenMP. NPU execution requires access to
the NPU DRM device (the API defaults to `/dev/dri/card1`; device numbering can
vary). Nonzero IOMMU domains require IOMMU support and driver version 0.9.8+.

Run these commands from the library root. In the parent research workspace
this directory remains `rk_npu_matmul/`; a standalone GitHub checkout may be
named `rknpu2_matmul_open/`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/demo
./build/test_quant_matmul
```

The library artifacts remain **`librk_npu_matmul.so`** and
**`librk_npu_matmul.a`**. The existing `rk_npu_matmul` and
`rk_npu_matmul_static` CMake targets are retained for callers.

To embed the library in another CMake project:

```cmake
add_subdirectory(path/to/rknpu2_matmul_open rknpu2_matmul_open-build EXCLUDE_FROM_ALL)
target_link_libraries(your_app PRIVATE rk_npu_matmul_static)
```

The existing install rule installs the shared library and public headers:

```bash
cmake --install build --prefix /your/install/prefix
```

### Cross-compilation

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=aarch64-toolchain.cmake
cmake --build build-aarch64 -j4
```

The supplied toolchain uses `aarch64-linux-gnu-gcc/g++`. Use a sysroot matching
the board's glibc and OpenMP runtime when deploying cross-built binaries.
Compilation alone does not validate NPU execution.

### CPU-only tests on a Linux host

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-host --target rknpu2_matmul_open_cpu_tests -j4
OMP_NUM_THREADS=4 ctest --test-dir build-host --output-on-failure
```

This target builds all 15 registered CPU tests and the public C header checks.
It does not require an NPU. Use this target on x86: several hardware-only FP16
programs in the default full build use the AArch64 `__fp16` type.
NPU tests are explicit executables, not part of CTest; see the
[test guide](tests/README.md). CPU test success is not a board acceptance test.

## Quick INT8 LLM benchmark

```bash
./build/bench_llm_i8 --loops 20 --warmup 3 --npu-mask 7
```

`bench_llm_i8` measures hidden, FFN-up and FFN-down matrices at sequence lengths
1, 4 and 128. M=1/4 report logical INT8 weight bandwidth; M=128 reports GOPS.
Timing includes A packing, NPU execution, split-K reduction and C unpacking,
and excludes one-time B packing. Each row checks sampled outputs against an
exact INT32 CPU reference. The fixed strategy uses Ktile≤2048 and Ntile≤3072;
use the autotuner for deployment-specific strategies.

For OpenMP workers to sleep while idle, set the waiting policy before starting
the process, for example:

```bash
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ./build/bench_llm_i8
```

Keep the same policy during tuning and inference; the library does not change
it automatically. Set the API's CPU thread count separately.

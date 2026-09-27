# Routed W8A8 MoE API

Public C header: [`include/rk_npu_moe_w8.h`](include/rk_npu_moe_w8.h).
The operator implements routed SwiGLU experts plus an optional shared expert.
The caller computes routing IDs and coefficients; NPU gather is the default
and only dispatch implementation in this API.

## Ownership and reuse

`rk_npu_moe_w8_weights_create` copies and packs an array of expert descriptors.
Each descriptor supplies gate/up INT8 `[H,2*I]`, down INT8 `[I,H]`, and positive
FP32 per-output-channel scales. Gate columns precede up columns. The source
arrays may be released after creation.

`rk_npu_moe_w8_workspace_create` allocates scratch for up to `max_rows` tokens.
A workspace accepts different immutable weight handles with matching geometry
on successive calls. Both must belong to the same context and IOMMU domain.
Weights do not depend on batch length or A layout. Context lifetime covers all
handles; callers may release the original domain handle after creation.

Supported envelope: H=64..1536 and I=32..512 in multiples of32, 1..128 routed
experts, top-k≤8, and zero or one shared expert of the same intermediate width.
`max_rows` is 1..256. Normal/native-A use conservative M tiles128/84.
Smaller shapes and fewer active experts use only the required NPU cores.

## Execution

```c
rk_npu_moe_w8_config config;
rk_npu_moe_w8_config_init(&config, 1536, 512, 128, 1, 8, 256);
/* descriptors has 129 entries; entry128 is the shared expert. */
rk_npu_moe_w8_weights *weights =
    rk_npu_moe_w8_weights_create(domain, &config.weights, descriptors);
rk_npu_moe_w8_workspace *workspace =
    rk_npu_moe_w8_workspace_create(domain, &config);
/* Check both handles before use. IDs and coefficients are [rows,8]. */
int status = rk_npu_moe_w8_run_f32(workspace, weights, rows, input,
                                  ids, coefficients, output, NULL);
```

`run_f32` quantizes each complete input row once using nearest-even rounding.
`run_quantized` accepts compact INT8 input and one FP32 scale per source row,
without changing its codes/scales. Both return compact FP32 `[rows,H]`.
Duplicate routed IDs are permitted, route coefficients are nonnegative finite
values, and no normalization is implicit. The shared expert has coefficient1.
The final sum follows route slot order, then the shared expert.

Workspaces reject concurrent use with BUSY. Invalid routes, scales, NaNs,
input/output overlap, mismatched shapes and domains are rejected. The operator
does not reset or retry a failed NPU submission. Hot calls reuse all device
allocations and templates; CPU scheduling vectors may allocate.

## Middle-stage choice

CPU middle is the numerical default: INT32 GEMM, FP32 dequantization/SiLU/MUL,
nearest-even hidden quantization, INT32 down GEMM and FP32 weighted combine.

`config.middle=RK_NPU_MOE_MIDDLE_NPU_LUT` explicitly enables the approximate
NPU dequantization/SiLU path for I=512. It uses FP16 coefficients, FP32 main
tensors, a SiLU table on[-8,8] and a4096 output carrier. MUL/requantization and
final weighted combine remain on CPU. Unsupported half coefficients fail
instead of silently selecting CPU execution. LUT refresh is immediately before
the DPU consumer; callers must serialize competing NPU users during the call.

Library timings report **actual complete-call time including LUT refresh**,
with `lut_load_us` separately available. To reproduce the user's experimental
LUT-ready latency, subtract that field. No runtime-PM controls are changed.
`*_device_bytes` reports requested device payload bytes, excluding page
rounding, FP32 scale vectors, other host containers and thread stacks.

## Validation and integration

- `test_moe_w8` checks independent integer/float references, A/B/A weight
  rebinding, batch transitions, duplicates, zero input, nearest-even input,
  invalid arguments, domain mismatch, and fewer jobs than requested cores.
- `tools/ling_moe_probe/compare_api.cpp` compares both public input entries to
  the original real-weight experiment on two fixture slices, normal/native-A,
  and CPU/NPU middle. All checked outputs match that corresponding experiment
  bit for bit; this does not equate the approximate LUT path with CPU SiLU.
- The parent research workspace contains a MindNano adapter in
  `integrations/mindnano/`; it applies the API to the actual decoder factory
  and retains its original router. That adapter and its fixtures are not
  bundled with this standalone library.

The public API above remains W8. An experimental private mixed backend in
`src/rk_npu_moe_w4_internal.h` keeps routed weights in packed INT4 and the
shared expert in INT8; see the adapter's `LING3_OPEN_MOE_W4` option. It reuses
the same run/free functions with compatible mixed handles, requires H/I
multiples of64, CPU middle and strictly certified full K. It rejects unsupported
weights instead of relaxing bounds. `test_moe_w8 --w4` exercises this path.
Group-wise W4, split-K mixed MoE, different shared-expert widths, router kernels,
and CPU/NPU request overlap are not implemented by this backend.

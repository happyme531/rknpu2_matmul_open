# Scalar-gated DeltaNet prefill

The [GDN C API](../include/rk_npu_gdn.h) provides normalized recurrence and a
fused projected-input path. CPU Conv/SiLU/QK normalization writes the NPU's
native inputs directly. NPU TF32 matrix products compute Gram, state
prediction, state increments and block output; FP32 DPU operations merge state.
CPU code constructs decay coefficients, solves the triangular system and
collects the output.

## Tensors and recurrence

All public tensors are contiguous FP32, one sequence per call. Current shapes
are C7's 16 key heads, 32 value heads and key/value dimensions 128.

| Tensor | C-order shape | Meaning |
| --- | --- | --- |
| Q/K | `[T,16,128]` | L2-normalized, Q not prescaled |
| V/O | `[T,32,128]` | Value and recurrence output |
| g/beta | `[T,32]` | Nonpositive log decay / sigmoid beta |
| Initial/final state | `[32,128,128]` | Value head, value channel, key channel |

The key channel is contiguous in canonical state. Per token/head, with
`a=exp(g)`, the operator evaluates:

```text
S = a * S
delta = beta * (v - S @ k)
S = S + delta @ transpose(k)
o = S @ q / sqrt(128)
```

`g=-infinity` gives zero decay. Other inputs are finite; beta is in `[0,1]`.
Q scaling is applied once inside the operator. Gate activation, projections
and output gated normalization belong to the caller.

`RK_NPU_GDN_HEAD_GROUPED` maps value head `h` to key head `h/2`.
`RK_NPU_GDN_HEAD_INTERLEAVED` maps it to `h%16`. The latter matches the
interleaved broadcast in ggml's `ggml_gated_delta_net` at llama.cpp revision
`de3ebb93bd8828496faefb44ed45059ec7ed3ea5`. The API follows its explicit
Q/K/V/g/beta/initial-state operator boundary, with separate output and final
state pointers. It returns one final state rather than per-token snapshots.

## Preparation and execution

`rk_npu_gdn_config_init` selects chunk64, CPU4, NPU mask7, ordinary TF32 and
grouped heads. `rk_npu_gdn_query` validates the configuration and returns the
required host spans without opening a device. Preparation fixes the token
count, chunk, precision and scheduling configuration, allocates buffers and
builds reusable register chains. Prepare a separate workspace for a different
window shape. A workspace can serve successive layers with independent caller
states and histories.

For an existing device/domain and caller-owned tensor buffers:

```cpp
#include "rk_npu_gdn.h"

rk_npu_gdn_config config;
rk_npu_gdn_config_init(&config, tokens);
rk_npu_gdn_workspace* workspace = nullptr;
int status = rk_npu_gdn_prepare(domain, &config, &workspace);
if (status != RK_NPU_OK) return status;

status = rk_npu_gdn_run(workspace, q, k, v, log_decay, beta, initial_state,
                       output, final_state, nullptr);
rk_npu_gdn_workspace_free(workspace);
return status;
```

Keep the workspace between calls to reuse the CPU workers and all device
allocations. Calls include native packing, cache synchronization, blocking
submission and canonical state export; no per-run data-buffer allocation or
graph construction occurs. CPU workers inherit the caller's affinity and do
not change the calling thread's affinity.

The workspace retains its IOMMU domain; the device context must outlive it.
Simultaneous use returns `RK_NPU_ERR_BUSY`. Free must not race with a call.
Applications schedule workspaces sharing physical NPU cores.

`final_state` may equal `initial_state`. Other output/input spans must be
distinct except for the projected history alias below. Invalid arguments or
device errors preserve the input state, including the in-place case. Output
is undefined on failure. Recreate the workspace after a device failure.

## Fused CPU input producer

`rk_npu_gdn_run_projected` replaces Q/K/V with projected QKV and the K4 Conv
parameters:

| Tensor | C-order shape |
| --- | --- |
| Projected QKV | `[T,8192]`, Q2048 then K2048 then V4096 per token |
| Conv weights | `[8192,4]`, oldest-to-current taps, no bias |
| History/final history | `[8192,3]`, original projected inputs, oldest first |

It applies causal Conv, SiLU and Q/K L2 with epsilon `1e-6`. No global Conv
output or normalized Q/K intermediate is materialized. V is kept token-major;
K and scaled Q are written directly to the Gram/prediction native buffers.
The Conv history advances by the real token count and uses raw projected
inputs. `final_history` may equal `history`.

```cpp
status = rk_npu_gdn_run_projected(workspace, projected, conv_weights, history,
    log_decay, beta, initial_state, output, final_state, final_history, &timings);
```

The caller owns weights, state and history. History and state are exported
only after the recurrence succeeds. They can be passed to an existing CPU
decode implementation without retaining the NPU workspace.

## Precision and internal state

Ordinary `RK_NPU_GDN_TF32` uses one TF32 product per logical GEMM. FP32 input
containers retain their bits; the tested NPU multiplier truncates 13 low
significand bits. `RK_NPU_GDN_COMPENSATED_TF32` uses high/high, high/residual
and residual/high products with FP32 sums. Neither mode narrows state storage.

Internally state is native `[head,key_channel/4,value_channel,4]`. A scalar
per head represents canonical `S=scale*Z`, avoiding repeated matrix decay.
State stays on the device between chunks. Export restores the scalar and
canonical layout. Preparation and execution reuse the shared floating
register generator, device allocator and multicore submission helpers.

## Validation

Build `test_gdn` and `test_gdn_header_c` on the board. `test_gdn` checks the
CPU-only configuration contract; `test_gdn --device` checks independent FP64
recurrence, both head mappings and precisions, supported physical core masks,
CPU thread configurations, replay, in-place state/history, BUSY and retained
IOMMU-domain ownership. Existing MM tests cover the shared floating backend.

`tools/validate_gdn.py` accepts captured ggml NPZ fixtures and the preceding
kernel executable. Optional Conv assets and producer executable add paired
real-weight/trace validation. It compares complete output and final state
bitwise, keeps numerical/latency summaries and deletes temporary FP32 tensors.
`--benchmark` uses two ABBA rounds, two warmups and seven measured calls per
segment. Both versions retain state import/export in the measured interval.

The 2026-10-18 board integration passed 142 bitwise comparisons with the
preceding kernel: 52 normalized cases and 90 real Conv weight/trace cases.
Shared and static library tests passed the independent FP64 recurrence,
2048-token windows, in-place state/history, both head mappings, CPU1..4,
all supported NPU masks, BUSY and nonzero retained IOMMU domains. Existing
FP16/BF16/TF32 MM board regressions and C-header compilation also passed.

Paired complete-call means, four A76s at 2256 MHz, NPU at 1 GHz, CPU affinity
`0xf0`, DDR uncontrolled, GCC 12.2.0, `-O3 -ffp-contract=off`, chunk64,
ordinary TF32:

| Interval | Tokens | Preceding kernel / ms | Library / ms |
| --- | ---: | ---: | ---: |
| Normalized recurrence | 128 | 7.182 | 6.904 |
| Normalized recurrence | 512 | 22.909 | 23.391 |
| Conv/SiLU/L2 + recurrence | 128 | 8.025 | 7.997 |
| Conv/SiLU/L2 + recurrence | 512 | 26.207 | 27.147 |

Normalized calls have 28 samples/version/window. Projected calls rotate three
real layer inputs, 84 samples/version/window. All samples are retained with no
filtering. The T512 projected library call is 3.59% slower; cumulative CPU time
is 61.416 versus 62.182 core-ms. The probe pins individual workers; library
workers inherit the caller mask. These are component replays; long projected
inputs cycle 21 saved QKV rows, with synthetic gates and initial GDN state.

[Validation summary](evidence/gdn-20261018.json) records counts, numerical
errors, paired statistics and clock restoration. `tools/summarize_gdn.py`
also retains full numerical records and unfiltered samples in the local result
directory's `verification.json`.

## Limits

- One sequence, scalar gate, Hk16/Hv32 and Dk128/Dv128.
- Fixed windows of 1..2048 tokens; chunk16/32/64/128. The interface is optimized
  for prefill. Single-token decode remains a separate CPU operation.
- Gram chains are limited to 4095 tasks. Compensated chunk16 accepts at most 816
  tokens and chunk32 at most 1632; chunk64/128 support the full 2048-token range.
- CPU threads 1..4 and NPU masks 1/2/4, 3 or 7. Caller controls clocks, CPU
  affinity and scheduling. No hidden tuning or environment-driven precision.
- Ordinary TF32 has numerical drift. Compensated mode is checked against the
  strict CPU fixture tolerance. Full-model/PPL validation belongs to runtime
  integration; component timings are not model throughput.

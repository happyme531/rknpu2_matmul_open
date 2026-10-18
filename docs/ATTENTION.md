# Experimental FP16 Attention

The public [C API](../include/rk_npu_attention_f16.h) is part of the shared and static
`rk_npu_matmul` libraries. It borrows the caller's device context and IOMMU domain,
retains domain references, and accepts one selected NPU core (mask 1, 2 or 4),
two cores (mask 3), or three cores (mask 7).
Existing matmul APIs and default execution choices are unchanged.

Set `config.flags` to `RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL` for the
fixed-shift approximation, or `RK_NPU_ATTENTION_F16_STABLE_SOFTMAX` for stable
row-max Attention. Stable mode defaults to NPU LUT exponential with a CPU
softmax fallback for PV magnitude and task limits. Fixed-shift numerical range
handling remains the caller's responsibility. No RKNN runtime, compiler, model or `.tasks` file is
needed at build or run time. The DPU-RDMA mask registers are generated from dimensions,
using tested native Softmax layout rules.

## Supported shapes and mask semantics

- One batch; compact FP16 bit-pattern tensors. Head dimension is a positive
  multiple of 32. Query heads are a positive integer multiple of KV heads,
  covering MHA, GQA and MQA; 1..32 KV heads and 1..128 logical query rows per call.
- `softmax_scale` sets a positive finite query scale; zero selects
  `1/sqrt(head_dim)`. Default geometry remains Hq/Hkv=32/8, D128.
- Large head dimensions or query/KV ratios are handled by internal query-row
  splitting. `run` accepts logical rows and restores compact output order;
  `query` reports peak segment buffers, total output bytes and QK/PV tasks;
  LUT execution adds DPU tasks at run time.
- KV capacity is aligned to 32, grows up to an explicit limit of at most 32768.
  Each internal graph must fit within 4095 NPU tasks; `query` rejects configurations
  that cannot fit even after query-row splitting.
- KV tile sizes: 256/512/1024/2048/4096. Default 512 preserves the prefill choice.
- The library uses at most one selected core per KV head and reports the actual
  selection in `sizes.active_core_mask`. Masks 5/6 remain unsupported by the driver.
- Compact Q/O are `[query_heads,query_rows,head_dim]`, K/V are `[kv_heads,length,head_dim]`.
  Output is FP32. No dropout or arbitrary additive attention bias is implemented.

`mask_mode` selects:

| Mode | Meaning |
|---|---|
| `CAUSAL` | Row r attends through absolute position `query_start+r`. Earlier prefill chunks can run after the complete KV tensor is loaded. |
| `NO_MASK` | Every query attends to all valid KV, without a mask tensor/RDMA read. Query rows may exceed KV length. |
| `BOOLEAN_MASK` | Nonzero byte / True means visible; no implicit causal restriction. Head, query and key dimensions can independently broadcast from size 1. |

Boolean semantics match [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).
They are opposite to PyTorch `MultiheadAttention.key_padding_mask`.
HF-style `[1,H,Q,K]` and `[Q,K]` masks are described by byte strides; batch size is one.
HF token padding masks can use `heads=1,query_rows=1`. Convert non-boolean HF tensors
to byte boolean data before calling this interface. For a causal boolean tensor,
compose its causal restriction in the supplied mask. Fully hidden rows return zero.

`boolean_mask_init` describes contiguous data. Edit strides for a view, and set
`query_offset` to select a block from a larger query-mask tensor. `bytes` bounds
all accesses. `version=0` refreshes every call; a nonzero version may reuse an
unchanged mask. Increment that version after modifying the source. Changing the
valid KV length still updates previously padded columns.

## Cache, workspace and active length

Create one cache per layer and share one workspace among sequential calls with
matching head geometry, tile size, context and domain. Objects return `ERR_BUSY`
on concurrent use. Schedule separate workspaces explicitly when sharing a core.
The caller's `ctx` must outlive all handles; freeing a caller domain reference
does not invalidate handles that retain it.

Cache capacity and computation are separate. A cache reserved for 8192 tokens
with valid length 4096 computes 4096, not 8192. Causal prefill computes only
`align32(query_start+query_rows)`; no-mask/boolean use `align32(cache.length)`.
Native KV is kept in DDR GEM buffers, with no programmable NPU cache assumed.

Multi-token `cache_append` packs only the newly supplied rows: native K uses
contiguous channel copies and native V uses N16/K32 block transposes, including
partial stripes. Earlier tokens, padding and denominator validity retain the
existing layout. Single-token append uses the existing scatter path. This also
applies to the private KV cache shared by MLA; public inputs remain compact FP16.
Tail staging and synchronization retain their original policies. The partial-B
update boundary cases are part of the regular `test_cpu_kernels` suite.

Fixed-clock RK3588 comparisons with Mixed Qwen3-4B (logical batch512,
ubatch64, CPU4-7/4 threads, NPU mask7; CPU/NPU/DDR at 2.4/1/2.112 GHz)
found modest complete-model gains: about 2% for pp512 and an uneven 1.2% for
pp2048. The pp4096 comparison was effectively unchanged, with opposite signs
in the two paired runs and about 2.5% baseline drift. No meaningful decode
gain was observed. These measurements do not establish a long-context speedup.

`cache_reserve` copies existing native K and V transactionally. `cache_append`
grows geometrically when needed, bounded by `max_capacity`. New data and the
denominator-validity channel are synchronized by ranges. Padding has zero V and
zero denominator validity, so no-mask does not accidentally normalize over
physical padding. Invalid arguments/capacity limits are checked before writes.

Full PV blocks read the native cache directly. A partial block uses a persistent
per-cache tail buffer; only newly appended columns are refreshed within a fixed
tail geometry. Workspace geometry changes at 32-token boundaries and when query
row counts change. Graph preparation, tail staging and reserve costs are reported
separately and included in run/step `total_us`. `prepare_query` can explicitly warm
a known current query without submitting it.

Sync/submit failure poisons the affected state. Reload a cache before reuse after
a write failure; recreate a workspace after a failed submit. The library does not
silently retry/fallback. These handles are not a framework-level model integration.

### Optional dirty stripe synchronization

`RK_NPU_ATTN_DIRTY_SYNC_AFTER=N` enables narrower V append and incremental-tail
sync only when the valid KV length is strictly greater than N. Append uses the
post-append length, not aligned span or allocated capacity. Threshold-only selects
one N16/K32 stripe per sync. `RK_NPU_ATTN_DIRTY_SYNC_GROUPS=1/2/4/span` selects the
granularity; explicit `0` disables it even with a threshold. GROUPS alone retains
always-on behavior; neither variable set keeps the original full-range path.
Both controls are captured at cache creation. Invalid threshold/group values
conservatively disable the optimization. Thresholds are nonnegative decimal token
counts up to 2147483647; 0 enables for nonempty caches. Load, reserve and full tail
repacking retain full synchronization; reloading a shorter cache reapplies the gate.

The default tile512 is unchanged. A length gate does not choose a tile or imply a
performance win: measured stripe gains mainly concern large PV tiles. See the
experiment and validation.

## Multi-core execution

Set `cfg.core_mask=3` or `7` when preparing the workspace. Cache layout does not
depend on the core count; single/multi-core workspaces can share the same cache.
The existing default remains one core, with no hidden core-count selection.

Each core owns complete QK -> fused exp/mask -> PV chains for contiguous KV heads.
Eight KV heads split 4/4 on two cores, or 3/3/2 on three. Query heads remain grouped
four per KV head, and output buffers retain their existing native layout.
Core partitions write disjoint head ranges; there are no cross-core intermediates
or device barriers. All partitions are submitted together in one blocking ioctl;
CPU output synchronization and FP32 accumulation/normalization happen once.
`rk_npu_attention_f16_run_half` uses the same computation and writes the final
normalized result directly as compact FP16, without a full FP32 output tensor.
The ordinary `run` entry point continues to return FP32. The `query` size's
`output_bytes` describes that FP32 result; callers of `run_half` allocate two
bytes per logical output element.

Each core's PC tail is terminated separately. Driver subcore slots follow the
existing board-validated ABI: 0/1 for dual core, 2/3/4 for triple core, and the
physical core slot for a single core. The builder exports single-core GEMM bodies,
then partitions the mixed Attention graph by head. Splitting that graph by raw
task count would violate its QK/PV dependencies.

The multi-core suite compares all output elements bitwise against a single-core
workspace, in addition to independent FP32 exact-exp references. It exercises
causal/no-mask/boolean masks, prefill row counts, short tails, cache rebinding,
reserve/append and INT8 Linear tasks on every selected physical core. Numerical
approximation and range behavior are unchanged.

## Example

```cpp
#include "rk_npu_attention_f16.h"

rk_npu_ctx* ctx = rk_npu_open(nullptr);
rk_npu_iommu_domain* domain = rk_npu_iommu_domain_create(ctx, 0);
rk_npu_attention_f16_config cfg;
rk_npu_attention_f16_config_init(&cfg);
cfg.flags = RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL;
cfg.initial_capacity = 8192;
cfg.max_capacity = 32768;
cfg.core_mask = 1;

rk_npu_attention_f16_cache* kv = nullptr;
rk_npu_attention_f16_workspace* workspace = nullptr;
// Check every return code; all *_create/prepare operations return RK_NPU_ERR_*.
rk_npu_attention_f16_cache_create(domain, &cfg, &kv);
rk_npu_attention_f16_prepare(domain, &cfg, &workspace);
rk_npu_attention_f16_cache_load(ctx, kv, k_fp16, v_fp16, 4096);

// Any causal prefill block, not only the final one: Q[32,32,128].
rk_npu_attention_f16_run(ctx, workspace, kv, q_block, 32, 64,
                         output_fp32, nullptr, nullptr);
// Append K/V[8,128], then run Q[32,128] at position 4096.
rk_npu_attention_f16_timings t;
rk_npu_attention_f16_decode_step(ctx, workspace, kv, q_token, new_k, new_v,
                                 output_fp32, &t, nullptr);

rk_npu_attention_f16_workspace_free(workspace);
rk_npu_attention_f16_cache_free(kv);
rk_npu_iommu_domain_free(domain);
rk_npu_close(ctx);
```

To use no mask, set `cfg.mask_mode=RK_NPU_ATTENTION_F16_NO_MASK` before preparing
the workspace. The same compatible cache can serve workspaces of different mask
modes. To use a boolean mask:

```cpp
cfg.mask_mode = RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
rk_npu_attention_f16_boolean_mask mask;
rk_npu_attention_f16_boolean_mask_init(&mask, keep_bytes, keep_byte_count,
                                      1, 1, valid_kv_length);
mask.version = 1; // Source stays unchanged until this version changes.
// Use a workspace prepared with the BOOLEAN_MASK configuration.
rk_npu_attention_f16_run(ctx, boolean_workspace, kv, q_token, 1, 4095,
                         output_fp32, &t, &mask);
```

## Measurements and validation

Fixed NPU 1 GHz, CPU4 2.4 GHz, DDR 2.112 GHz, core mask 1. All modes use identical
FP16 operands and all-visible masks for the decode comparison. Reserved KV capacity
8192; plan/mask warm-up is outside repeated-run timing, five warmups and 50 samples.

| Valid KV | Causal (ms) | No mask (ms) | Cached boolean (ms) | Boolean refreshed every call (ms) |
|---:|---:|---:|---:|---:|
| 4096 | 1.790 | 1.792 | 1.791 | 1.882 |
| 4097 | 1.806 | 1.807 | 1.806 | 1.891 |

No-mask saves memory and mask preparation, but no measurable NPU latency gain
appeared in this single-core decode comparison. Refreshing a broadcast
boolean mask adds about 80–82 us. Reusing an immutable mask avoids that conversion.

Continuous append+run from 4095 through 4160: median about 1.885 ms no-mask,
1.905 ms causal, 1.888 ms cached boolean, 1.955 ms refreshed boolean, with K512.
These medians include KV writes/sync and tail staging. At 32-token boundaries,
graph rebuilds raise calls to roughly 3–4 ms; this cost is included, not removed
from the report. Reserve/allocation spikes are distinct from these graph updates.

The board suite checks core masks 1/2/4, KV preservation across reserve/automatic
growth, arbitrary prefill positions and row counts, shared workspace A/B/A,
strided/broadcast/versioned masks, fully hidden rows, domain rejection, and INT8
Linear/Attention alternation. KV head counts 1/8/32 and query length greater than
KV length are covered. C header, host shared/static builds and CPU sizing checks pass.
The exp approximation remains the existing one; model PPL/TPOT is not measured.

The later multi-core study measures
identical operands with masks 1/3/7 interleaved. With K512, 4K cached decode takes
1.790/0.941/0.763 ms; append+decode takes 1.892/1.030/0.847 ms. A 32-query prefill
chunk against 4K KV takes 3.847/3.077/2.676 ms. These are complete calls, including
synchronization and CPU finish; prefill is a chunk measurement, not full 4K TTFT.
Multi-core/single-core outputs are bitwise identical for the same tile. K1024
slightly improves this growing-decode scenario; the default K512 remains unchanged.
Lowering DDR from 2112 to 1068 MHz makes triple-core decode roughly 73% slower,
while NPU 1000 to 600 MHz makes it about 9% slower. This supports a shared-memory
limit on scaling, without claiming measured DDR saturation. Triple-core no-mask
prefill saves about 9% here, despite the earlier single-core decode result.

Sources, binary hashes, clocks, raw samples and checks:
public API report.
Reproduce on the board after building `test_attention_f16` and `bench_attention_f16`:

```bash
./build/test_attention_f16 --board 1
./build/test_attention_f16 --board 7 --long 512
./build/bench_attention_f16 --length 4096 --capacity 8192 --core-mask 7 --key-tile 512 --loops 50 --warmup 5
```

Record actual board frequencies and restore externally changed governors or limits after testing.

## Workspace 几何更新（2026-10-04）

GQA 与 MLA 共用的私有 graph builder 现在直接生成 FP16寄存器 body，不再为导出
body 创建临时设备 plan；与旧 prepared-plan 导出结果有逐位对照验证。几何更新
复用设备 buffers 并在需要时增长，NPU 完全覆写的 score/partial 不再由 CPU预清零。
首次 mask 完整初始化、前移增量更新；输入与输出同步按当前逻辑范围执行。
公开 GQA的形状、causal有效前缀规则和C ABI保持原定义；五种core mask回归通过。
Workspace可能保留历史较大buffer，query()给出当前形状需要的最小大小。
几何重建失败后需重建workspace，以免继续使用已交接的 buffers。

## 长上下文 QK 分块

MLA 和 GQA 原共用的 graph 把完整上下文当作一个 QK 输出 cube；有效 KV 超过
8192 时，DPU `0x403c` 和 mask RDMA `0x5014` 的 13-bit channel 字段越界。
现在 QK 每段最多 4096 tokens，并分别设置 K 权重、score 与 mask 地址；
PV 仍使用配置中的 `kv_tile`，不改变默认值 512。`query()` 包含全部 QK 分块任务，
4095-task 限制依然有效，不保证所有 head/tile/容量组合均可执行。

4096 是实测的安全配方。普通 native FP16 `M4/K128/N8192` 即使输出正确，
仍可让后续 INT8 超时，而 N4096 的混跑通过；这一内部状态原因未证实。
本修改只约束 Attention 私有 QK，不改变公开 FP16 GEMM 的默认 N 分块策略。
末尾 Q1 causal decode 内部可跳过冗余 mask RDMA；较早的 causal query 仍保留 mask。
回归覆盖长上下文、三种 mask、Q1/5/32、单/多核、完整输出 oracle、guard、
增长/rebind 和每个选定物理核的 INT8 交错。详细证据及 decode 测量见
长上下文修复记录。

## Stable LUT exponential

`RK_NPU_ATTENTION_F16_STABLE_SOFTMAX` 与 MLA 共用稳定路径：原始 QK 写回后，CPU
排除无效 keys、以 FP32 减行最大值，NPU LUT 执行 exponential 并通过 PINGPONG
连接 PV；CPU 以 FP32 去掉正偏置并归一化。stable 无需 native mask tensor，
`mask_bytes=0`，boolean 的逻辑 ABI 和 GQA causal 前缀规则保留。

启动进程前设置 `RK_NPU_ATTN_LUT_EXP=0` 恢复 CPU exponential 以及原有 KV
更新和截断行为。大 score 在 V 幅值允许时通过 PV denominator 延后归一化，
短 score/大 V 保留 CPU 归一化概率。若 `max(abs(V))*kv_tile*1.001>32752`，
或加入 exponential 后总任务数超过4095，自动使用 CPU softmax。

LUT 分别覆盖 `[-16,-8]` 和 `[-8,0]`。CPU 维护 V 的 FP32 前缀和以修正 LUT
正偏置；每32个 KV token、每个 V channel 保存4 bytes，另有当前累加和。
workspace 增加一个与 score 等大的 FP16 exponential 缓冲区。截断保留原有
分配和前缀，并清除被丢弃的 V/denominator 行。

LUT 使用公共增益 `g=255/256` 和正偏置 `delta=16/32768`，近似
`P'=g*exp(x)+delta`。增益为正偏置保留 INT16 表项的峰值空间；CPU 修正
`P'@V-delta*sum(V)` 和 `P'@1-delta*key_count` 后相除，公共增益自动抵消。
该配方不增加任务或提交次数。板端测得 `x=0` 在去偏置并除以增益后为1。

Qwen3-4B-W4A8-Mixed-LMHeadW8，ubatch128、CPU4-7/4 threads、NPU mask7、
CPU/NPU/DDR 2.4/1/2.112 GHz、Clang22/O2 的完整模型 ABBA 对照中，prefill512
吞吐增加1.68%，decode128基本持平，整次累计进程 CPU 时间减少3.83%。
该性能结果使用 `g=1`；`g=255/256` 没有单独测量完整模型吞吐。

WikiText-2 raw test 前32个独立512-token窗口、每窗重置 KV，评分 target257–511，
共8160个目标，三种模式使用同一可执行文件与 FP64 NLL 累加：

| Exponential | PPL | 相对 CPU |
| --- | ---: | ---: |
| CPU FP32 | 14.885053 | — |
| LUT `g=1` | 14.972756 | +0.5892% |
| LUT `g=255/256` | 14.953214 | +0.4579% |

两种 LUT 的相对 PPL 变化为−0.1305%，按32个配对窗口 NLL 差值计算的近似
95%区间为[−1.45%, +1.20%]。该语料前缀不足以确认整体质量提升，不代表其他
模型或下游任务。固定的36层真实 QKV 回放中，`g=255/256` 相对 CPU softmax 输出
的 relative L2 均值/最大值为0.04449%/0.05114%，32层低于 `g=1`。

`g=1` 的稳定路径通过8种通用 GQA 几何、query 行数1/3/31/32/64/128、三种 mask、
KV grow/truncate/替换后缀/reset 回归。公共增益改变 LUT 量化与 FP16 中间舍入；
上述36层回放和完整模型 PPL 覆盖 `g=255/256`。

# Ling3 Tiny FP16 Gated MLA

Historical measurement notes named below belong to the parent research workspace;
raw reports and helper scripts are not bundled with this standalone library.
The figures describe those experiments, not a fresh release benchmark.

接口：[rk_npu_mla_f16.h](../include/rk_npu_mla_f16.h)。这是直接用户态驱动上的实验性
Attention 核心，不需要 RKNN runtime、转换器或模型文件。固定维度针对
[Ling-3.0-tiny 官方配置](https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/405ba96/config.json)：
H=16，Q/K=192（内容 128 + RoPE 64），V=128，KV rank=512，Q rank=256。
MLA 出现在零起始层 3/7/11/15/19/23；其余 Attention 层是 KDA。
代码依据与原文件 SHA 见 source.json (historical parent-workspace note: `source.json`)。

## 输入输出与门控位置

`q[Q,16,192]` 已经过 Q 投影及 RoPE；`latent[L,512]` 已经过
`kv_a_layernorm`；`rope_key[L,64]` 已经过 RoPE。所有 FP16 数据通过 `uint16_t`
位模式传递。`kv_b[16,256,512]` 直接对应 HF `kv_b_proj.weight[4096,512]`，
每个 head 的前 128 行是 K 权重，后 128 行是 V 权重。

输出是 FP32 `out[Q,16,128]`。可选 FP32 `gate_logits[Q,16]` 是 `g_proj`
的结果；kernel 在 Attention 后乘以 FP32 sigmoid，再交给调用方做输出投影。
传 NULL 表示普通、不带 gate 的 MLA。门控按 head 广播到 128 个 V 分量，
没有 RMSNorm 或 SiLU；这与 KDA 的输出门控不同。

调用方负责 Q/KV 的降维投影、两次 RMSNorm、Q 上投影、RoPE、gate 投影和最终
`o_proj/dense`。本接口完成 KV 上投影或权重吸收、Attention、必要的 V 上投影及 gate。
官方当前 MLA 类使用压缩 rank 上的 RMSNorm，没有在每个 Q/K head 上再加 QK Norm；
不要仅从配置里的 `use_qk_norm` 字段额外插入操作。

## 两种显式缓存模式

设 `Wk[h]` 与 `Wv[h]` 均为 `[128,512]`，`c` 为归一化后的 latent。

**EXPANDED：** `K[h]=concat(c @ Wk[h]^T,rope_key)`，`V[h]=c @ Wv[h]^T`。
缓存加载/追加时由共享输入的 NPU FP16 GEMM 展开，每批至多 128 tokens；
`[M,512]@[512,4096]` 按 256 个输出列分块。权重与 native 输出布局兼容原来的
16-head BMM，输入无需复制16份。Attention 直接计算
192 维 QK 和 128 维 PV，适合计算量较大的 prefill 查询块。

**ABSORBED：** `q_abs[h]=q_nope[h] @ Wk[h]`，
`score=(q_abs @ c^T + q_rope @ rope_key^T)/sqrt(192)`，
`u=softmax(score) @ c`，`out[h]=u @ Wv[h]^T`。
cache 只有一个共享的 KV head，16 个 Q head 都读取它；两个投影由 NPU BMM 完成。
QK 实际是 576 维，但缩放仍为 **1/sqrt(192)**。

两条路径代数等价，但 FP16 投影发生的位置不同，输出不要求逐位相同。
同一模式、同一 tile 的单/双/三核输出应逐位相同。
API 不自动切换模式；cache 与 workspace 必须使用相同模式、tile、权重对象及 domain。

压缩模式的逻辑 cache 是每 token `512+64` 个 FP16，即 1152 bytes。
当前硬件 QK/PV 要求不同的 native 布局，因此物理 cache 保存两份 latent：
K 为 576 channels，PV 为 512+32 channels（包括 denominator 与 padding）。
在容量 4096、tile512 时，两种物理缓存分别为：

| 模式 | K | V/latent PV | 合计 |
|---|---:|---:|---:|
| expanded | 24 MiB | 20 MiB | 44 MiB |
| absorbed | 4.5 MiB | 4.25 MiB | 8.75 MiB |

这是实际 native 存储，不是逻辑 latent 大小。尾块、临时 workspace 及投影权重另外计；
共用 packed weights 为 8 MiB。benchmark 预留容量 8192，所以实际固定 K/V 是
88 MiB / 17.5 MiB，不能把 4096 容量的数字当作 benchmark 分配值。
`cache_get_info` 可查询当前实际分配。

两种模式也支持 `RK_NPU_ATTN_DIRTY_SYNC_AFTER=N`：追加后的有效 KV 长度严格
大于 N 才启用 V/增量 tail 的脏 stripe 同步。仅设阈值默认 stripe1，
`RK_NPU_ATTN_DIRTY_SYNC_GROUPS=0` 强制关闭；粒度和阈值在 cache 创建时固定。
两个变量均未设时仍用原同步，默认 tile512 不变。长上下文不保证小 tile 有收益；
参数优先级、边界验证及合成性能见 脏同步实验 (historical parent-workspace note: `DIRTY_SYNC.md`)。

## 调用示例

```c
rk_npu_mla_f16_config cfg;
rk_npu_mla_f16_config_init(&cfg);
cfg.flags = RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL;
cfg.mode = RK_NPU_MLA_F16_EXPANDED; /* 或 ABSORBED */
cfg.max_query_rows = cfg.mode == RK_NPU_MLA_F16_EXPANDED ? 128 : 32;
cfg.core_mask = 7;
cfg.initial_capacity = 4096;
cfg.max_capacity = 8192;

rk_npu_mla_f16_weights *weights = NULL;
rk_npu_mla_f16_cache *cache = NULL;
rk_npu_mla_f16_workspace *work = NULL;
/* ctx/domain 已打开；所有 int 返回值均需要检查。 */
int rc = rk_npu_mla_f16_weights_create(domain, kv_b_fp16, &weights);
/* rc == 0 后继续 */
rc = rk_npu_mla_f16_cache_create(domain, &cfg, weights, &cache);
rc = rk_npu_mla_f16_prepare(domain, &cfg, weights, &work);
rc = rk_npu_mla_f16_cache_load(ctx, cache, latent, rotated_key, length);
rc = rk_npu_mla_f16_run(ctx, work, cache, q, gate_logits,
    query_rows, query_start, output, &timings, NULL);
/* decode_step 一并 append 一个 token，再执行 query，统计包括整个过程。 */
rk_npu_mla_f16_workspace_free(work);
rk_npu_mla_f16_cache_free(cache);
rk_npu_mla_f16_weights_free(weights);
/* 所有 handles 销毁后才能关闭 ctx。 */
```

weights 为不可变共享对象；cache/workspace 会 retain 它，因此调用方可以提前释放
自己的 weights handle。不同层使用不同 weights/cache；workspace 绑定该层 weights。
所有对象非重入，竞争返回 `ERR_BUSY`；不同 workspaces 的物理 core 竞争由调用方调度。
同模式 cache 可以跨不同 core mask 的 workspace 使用。

## mask、动态长度、调度与数值边界

batch=1，expanded Q=1..128、absorbed Q=1..32；默认 max_query_rows 仍为32。
KV capacity 为 32 的倍数且不超过 32768；tile 支持
256/512/1024/2048/4096；core mask 支持 0/1/2/4/3/7，0 表示 1。
不支持 additive bias、dropout、自动 fallback 或自动重试。

长上下文已改为独立的 QK N 分块，每段最多 4096 tokens；这与 `kv_tile`
控制的 PV K 分块不同。原实现把完整上下文写成一个 QK 输出 cube，超过 8192
便超出 DPU/RDMA 的 13-bit channel 字段；GQA 也复现了同一问题。
分块同时修正 K 权重、score 输出、mask RDMA 地址与 channel，以及 `query()` task 数。
选用 4096 而非字段最大值 8192，还避免了板测发现的宽 FP16 任务之后 INT8 超时；
这一状态切换的硬件内部原因未证实，不改公共 GEMM 的默认策略。
32K、三种 mask、多 query、cache 增长与 INT8 交错的结果及复现命令见
长上下文修复记录 (historical parent-workspace note: `LONG_CONTEXT.md`)。

末尾的单 query causal decode 看见所有有效 KV，内部跳过冗余 mask RDMA；
padding 的 V 和 denominator 为零，输出保持逐位一致。预加载未来 KV 后的较早
query 仍使用 causal mask，并在 masked/unmasked graph 切换时重建配置。

mask 复用 [Attention 的 boolean ABI](ATTENTION.md)：非零=True=可见，
支持 HF/SDPA `[1,H,Q,K]`、二维 `[Q,K]`、token 广播、非连续 byte stride、version。
causal 按 `query_start+row`；no-mask/boolean 读所有有效 KV；padding 永远不进 denominator。
全隐藏行输出零。NULL gate 支持无门控。mask 无效时返回参数错误，保留 output/cache。

Attention graph 使用 private shape 参数复用既有 GQA 的 native cache、fused exp/mask
和单次多核 submit。expanded 的三核分配为 6/5/5 heads。
absorbed 的相同 KV 在 GEMM 的 M 行内被多个 Q head 复用，不依赖可编程 GPU cache。
Q1 的分组随核数变化：单核一次16 heads，双核各8 heads，三核6/5/5 heads。
三核每组使用 M6，末两组补一行零；减少相同 KV 的重复读取，并避免旧4-head组的
2/1/1不均衡。Q2..16 每组4 heads，Q17..32 每组2 heads，M补到4。
K576 时仍保证 native A 在四个 CBUF banks 内。两个投影 BMM 仍是16 heads；
score/partial 输出互不重叠，没有跨核依赖，只返回真实 Q/head。

decode 的长度按有效 causal 前缀 align32；reserve 不增加 compute length。
fixed-shift MLA 的 Q>1 causal 查询使用整个有效 KV 长度的固定 graph，用 mask 屏蔽未来 token。
整段预先加载 KV 的 prefill 因而可以复用 graph，并增量更新新可见的 mask 窄带，
避免每 32 个 queries 重新分配和构建 graph。代价是早期查询也计算后续有效 KV 的
masked 分数，测试以整段耗时衡量收益。扩容的空闲 capacity 不参与计算。
原有公开 GQA API 仍使用有效 causal 前缀。
graph 变形时直接在 CPU 生成与 prepared FP16 plan 逐位相同的寄存器 body，
复用 query/score/mask/partial/regcmd/descriptor buffers；容量不足才按几何方式增长。
完全由 NPU 覆写的 score/partial 不再由 CPU 清零和 TO_DEVICE 同步。
mask 首次使用完整初始化，连续 causal 前移只改新可见窄带；同步限制到实际有效区域。
workspace 会保留历史最大 buffer，query() 按传入 KV 长度报告几何大小；
stable causal prefill 的实际裁剪长度见 timings.attention.compute_length。
cache load/append/reserve 包含真实 packing/sync；Q1 跨32-token边界仍需更新 graph，
run/step 包含这些成本。执行或几何重建失败后重建 workspace，cache 失败后 reload。

当前沿用显式 opt-in 的 fixed-shift exp 位近似，不做每行减最大值。
投影与 PV 写回为 FP16，跨 tile 归约、除 denominator 和 gate 为 FP32。
仍需真实权重及模型质量验证；合成数据通过不代表 BF16 模型的 PPL/长上下文质量通过。

`RK_NPU_MLA_F16_STABLE_SOFTMAX` 可替代 fixed-shift flag：第一次 submit 做原始
QK，CPU 在 native block-8 score 上排除 causal/boolean/padding，再以 FP32
减行最大值和 exp，第二次 submit 做 PV。长 score 且 `max(abs(V))*kv_tile<=32752`
的 head 使用未归一化 e，利用已有 `[V,1]` 的 PV denominator 在最后做 FP32 除法；
短 score 和较大 V 保留 CPU 归一化，极大 V 使用可抵消的1/2公共倍率。
常见 causal/no-mask 路径使用四线程 NEON FP32 exp 多项式，boolean 路径使用
标量 exp。两次 score 同步及 CPU softmax 包含在完整调用计时里，不能称为全
NPU fused softmax。此模式忽略 `exp_shift`，保留 FP16 QK/PV 和中间 e/p 的舍入。

stable 多 query causal prefill 按可见前缀裁剪并尽量对齐完整 PV tile；无需 native
mask tensor，`query().mask_bytes=0`。缓存格式、默认 tile512 和公开 C ABI 保留。
范围推导、32K及极值回归、同一稳定补丁的反序对照和整模型验证见
稳定路径性能记录 (historical parent-workspace note: `STABLE_OPTIMIZATION.md`)。

expanded cache 支持 `cache_load_expanded` / `cache_append_expanded`：调用方传
head-major FP16 `K[16,count,192]`、`V[16,count,128]`，保留既有投影/量化策略。
只有 expanded 的 create/prepare 可以传 NULL weights；此时非空 latent
load/append 被拒绝。MindNano 用这两个入口维护六层 native KV 镜像，完整状态
仍保存原有 BF16 K/V。接入与真实模型基准见
MindNano MLA 记录 (historical parent-workspace note: `README.md`)。

验证与板端测量见 tools/mla_probe (historical parent-workspace note: `README.md`)及
优化后的交错对照 (historical parent-workspace note: `OPTIMIZATION.md`)。
与 MindNano 已有实现的实际逐 token 比较见
decode 曲线与范围 (historical parent-workspace note: `MINDNANO_DECODE.md`)；SDK/raw-driver 混跑的
timeout 仍待集成验证，目前只完成两个独立连续轨迹的性能/输出对照。

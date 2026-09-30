# rk_npu_matmul 环境变量

库在 `src/` 里用 `std::getenv` 读取的变量都列在这里。扫描范围是 `rk_npu_matmul/src`、公开头文件和对应测试。设备节点不是环境变量：`rk_npu_open(NULL)` 打开 `/dev/dri/card1`，其它节点把路径传给 `rk_npu_open`。

`RK_NPU_W4A8_REDUCE_NPU` 也不是环境变量。它是 `rk_npu_w4a8_options.reduce_backend` 的枚举，由 `rk_npu_w4a8_workspace_create_ex` 的 options 选择。

布尔开关的规则以各节为准。W8A8 的两个开关只认精确 `"1"`，其余值为关；新 MM 和 MoE mixed 的默认与错误处理不同。

| 变量 | 默认 | 读取点 | 作用 |
| --- | --- | --- | --- |
| `RK_NPU_I8_NPU_REDUCE` | 关 | W8A8 workspace 查询、创建、autotune、cache key | INT32 split-K 累加改到 NPU DPU |
| `RK_NPU_W8A8_NPU_DEQUANT` | 关 | 同上；只对浮点算子生效 | NPU 上用 FP16 系数反量化，并强制打开上一行 |
| `RK_NPU_W4A8_K_TILE_MULTIPLIER` | `1` | `k_tile=0` 的 weight 创建，以及 autotune 的 `k_tile_multiplier==0` | 放宽 INT4 K tile 的安全上界 |
| `RK_NPU_MOE_TRACE` | 关 | 每次 MoE `ioctl` submit 前后 | 往 stderr 打阶段日志 |
| `RK_NPU_MOE_W4_MIXED` | 开 | 仅 `rk_npu_moe_w4_workspace_create` | 混合 W4/W8 MoE 是否把 INT4 和 INT8 GEMM 拼进同一条 PC chain |
| `RK_NPU_MM_SPLIT_K` | 开 | 新 `rk_npu_mm_plan_create`，options.split_k=-1 时 | 允许 FP16/BF16/TF32 MM/BMM 自动 split-K；`0` 关闭，`1` 开启，其它值报错；不影响旧接口 |
| `XDG_CACHE_HOME`、`HOME` | 见下文 | cache path 参数为 NULL 时 | autotune 结果目录 |

改环境变量不会改已经建好的 workspace 或 weight。W8A8 模式在 `plan_config()` 里读一次并写进 plan；W4A8 倍数写进 `rk_npu_w4a8_weight_info`；`RK_NPU_MOE_W4_MIXED` 写进 workspace 的 `mixed_chain`。`RK_NPU_MOE_TRACE` 是每次 submit 现读的。

`setenv` 和正在进行的 `getenv` 没有同步。进程起来之前设好，或者至少在查询、调优、创建 handle 之前设好，创建期间不要再改。

## 新 MM/BMM：`RK_NPU_MM_SPLIT_K`

未设置或 `1` 默认允许 split-K；`0` 禁止；其它字符串（包括空串、`true`、`01`）
返回 `RK_NPU_ERR_PARAM`。自动策略在 `K>=2048` 且 batch 数小于可用核数时按核数拆分，
否则保持一份。partial 由 CPU 用 FP32 归约，最终输出 c_type，可能改变中间舍入/溢出。
TF32 partial/输出均为 FP32，其 K 分块按 16 对齐；FP16/BF16 按 32 对齐。
只在创建 plan 时读取并固化，运行时不重新读取；options.split_k 的显式 0/1/2/3
覆盖环境变量。现有 FP16、INT8、INT4 和量化接口均不读取此变量。
完整语义、预打包 B 兼容条件和例子见 [MM.md](MM.md)。

## W8A8：`RK_NPU_I8_NPU_REDUCE` 和 `RK_NPU_W8A8_NPU_DEQUANT`

实现在 `src/rk_npu_quant_matmul.cpp` 的 `execution_mode()`。两个变量都要精确等于 `"1"`。

| `RK_NPU_I8_NPU_REDUCE` | `RK_NPU_W8A8_NPU_DEQUANT` | `i8i8i32` | `f16i8f16` / `f32i8f32` |
| --- | --- | --- | --- |
| 关 | 关 | CPU 累加 INT32 partial | CPU 累加，再用原始 FP32 scale 反量化 |
| `1` | 关 | NPU DPU 做 INT32 split-K 累加 | 同上，反量化仍在 CPU，scale 仍是 FP32 |
| 关或 `1` | `1` | 反量化开关被忽略；只有 REDUCE=`1` 才走 NPU 累加 | NPU 累加，再在 NPU 上用 FP16 舍入后的 scale 反量化 |

`RK_NPU_W8A8_NPU_DEQUANT=1` 对浮点算子隐含打开 NPU reduce。对 `RK_NPU_MATMUL_I8I8I32` 这个变量不产生任何位。

读取发生在：

- `rk_npu_matmul_workspace_memory_query`
- `rk_npu_matmul_workspace_create`
- 各算子的 autotune / family autotune
- `rk_npu_matmul_autotune_cached` 和显式 cache load/save 的 key

因此查询、调优、创建必须看到同一组值。cache key 把模式或进 `tuning_revision` 的高 16 位：CPU 是 0，只开 NPU reduce 是 1，浮点 NPU 反量化是 3。三种后端的调优结果互不命中。显式传 cache 路径也一样，路径相同而模式不同仍然是 miss。

NPU 路径的几何限制在 `query_i8_kn_memory()`：

- `M <= 128`
- `K <= 131071`（按每项 `-128 * -128` 也不会溢出有符号 INT32）
- 权重必须是未压缩的 W8。DCOMP 权重在 `run` 里直接返回 `RK_NPU_ERR_PARAM`，不会退回 CPU 累加
- 不满足 `M`/`K` 时，memory query 返回 `RK_NPU_ERR_PARAM`，workspace 创建返回 NULL。autotune 同样失败，不会偷偷改回 CPU 后端

NPU reduce 把各 K wave 的 INT32 partial 留在 workspace 里，用同一组 NPU core 上的 offline DPU 加完。原有 A/C layout（normal、native、panel8、panel16）仍然可用。这条链占用额外的 command/task，以及每个 wave 一份 input slot；反量化再加系数 buffer 和 FP32 结果 buffer。

NPU 反量化把每个 channel 的 weight scale 和每个 token 的 activation scale 收成 FP16 再送进 DPU。这是明确的精度交换，和 CPU 路径的 FP32 scale 不是 bit-exact。非零 scale 舍入成 0，或者变成 Inf，`half_bits()` 失败，本次 run 返回 `RK_NPU_ERR_PARAM`，不会退回 CPU 反量化。输出先以 FP32 放在设备 buffer，FP16 算子拷回 host 时再做一次转换。

## W4A8：`RK_NPU_W4A8_K_TILE_MULTIPLIER`

只在自动选择 K tile 时读取。

- `rk_npu_w4a8_weights_create`：`cfg->k_tile == 0` 时读。显式正的 `k_tile` 忽略环境变量，并按 INT16 安全界严格检查；过不上返回 NULL。
- autotune：`rk_npu_w4a8_autotune_config.k_tile_multiplier == 0` 时读。`rk_npu_w4a8_autotune_config_init` 把这个字段设成 0，所以默认调优会读环境变量。字段 `>= 1` 时用字段，不再读环境变量。
- `rk_npu_w4a8_weights_create_tuned` / `_ex` 使用 strategy 里已经记下的 multiplier，不再读环境变量。

未设置时因子是 1。字符串必须被 `strtod` 完整吃掉，结果有限且 `>= 1`。`0`、负数、`nan`、`inf`、空串、尾部多余字符（例如 `1.5oops`）都无效：weight 创建返回 NULL，autotune 的 profile 返回 `RK_NPU_ERR_PARAM`。

选中的 tile 是

```text
floor(min(cap, safe_k_tile * multiplier) / 32) * 32
cap = min(2048, align_up(K, 32))
```

`safe_k_tile` 是按这份权重认证过的最大固定 tile。`multiplier > 1` 时 `bound_relaxed=1`：INT16 partial 可能饱和，输出可以是错的，运行时不再检测，也不会重试。`multiplier == 1` 且选中 tile 没有超过 `safe_k_tile` 时，整数重建仍按认证路径处理。

因子会进 W4A8 cache key（`multiplier` 的原始位，以及由此算出的 tile 上界）。换因子之后旧 cache 不会命中。等待策略不在这个 key 里，见下文 OpenMP。

## MoE

### `RK_NPU_MOE_TRACE`

任意已设置的值都打开，包括空串和 `"0"`。未设置才关闭。每次 MoE submit 在 `ioctl` 前打一行，返回后再打一行，都到 stderr：

```text
submit <stage> cores=<n> tasks=<count>
done <stage>
```

stage 来自调用点：`gather`、`gate`、`gate4`、`lut`、`middle`、`down`、`down4`。公开 W8 和私有混合 W4 走同一个 `Commands::run`。`lut` / `middle` 只在 `config.middle=RK_NPU_MOE_MIDDLE_NPU_LUT` 时出现。混合 chain 打开时，INT8 的 `gate` / `down` 不再单独 submit。

### `RK_NPU_MOE_W4_MIXED`

只在 `rk_npu_moe_w4_workspace_create`（`src/rk_npu_moe_w4_internal.h`，实验接口）里读取。公开的 `rk_npu_moe_w8_workspace_create` 不读它。

混合后端要求 routed expert 是打包 INT4、shared expert 是 INT8，H/I 是 64 的倍数，CPU middle，`native_input=0`，并且整段 K 通过严格 INT16 界。

| 值 | workspace |
| --- | --- |
| 未设置或 `"1"` | `mixed_chain=1`。每个 stage 把 INT4 GEMM 和 INT8 GEMM 接到同一条 PC chain，一次 submit。shared expert 可以分到任意 core |
| `"0"` | 分开 submit。INT8 shared expert 固定在 core 0，INT4 job 仍按负载分 core |
| 其它字符串 | 创建失败，返回 NULL |

创建成功后，后续 `run_f32` / `run_quantized` 用当时记下的模式。

## Autotune 缓存目录：`XDG_CACHE_HOME` 和 `HOME`

`src/rk_npu_autotune_cache.cpp` 在调用方传入的 cache path / directory 为 NULL 时决定目录：

1. `XDG_CACHE_HOME` 非空，就用它
2. 否则用 `$HOME/.cache`
3. 两个都没有，默认路径为空。load 是 cache miss，save 返回 `RK_NPU_ERR_IO`；调优本身已经成功的结果仍会返回给调用方

子目录和文件名：

| 接口 | 默认目录 | 文件 |
| --- | --- | --- |
| `rk_npu_matmul_autotune_cached` 及 family 的默认路径 | `$BASE/rk_npu_matmul` | `<16 位十六进制 hash>.tune` |
| W4A8 `*_cached`，`cache_path == NULL` | `$BASE/rk_npu_w4a8` | 同上 |

`$BASE` 是上面的 `XDG_CACHE_HOME` 或 `$HOME/.cache`。缺父目录时以权限 `0700` 创建。文件是带版本的文本，先写临时文件再 `rename`。显式传入的 path 或 directory 不再看这两个变量；显式 directory 就用作目录本身，不会再追加 `rk_npu_matmul` 或 `rk_npu_w4a8`。

W8A8 的 hash 含算子、形状、CPU affinity、NPU mask、测量参数、driver version、格式版本、tuning revision，以及上面的执行模式。W4A8 的 hash 另外含权重指纹、安全 tile 和 multiplier。

## OpenMP 运行时

这些变量由 OpenMP / libgomp 读取。本库不调用 `setenv` 去改它们，也不把它们放进 cache key。

W4A8、W4A4、FlatQuant、MoE 和 W8A8 的策略执行都在 pragma 上写了 `num_threads(...)`，线程数来自 config 或 strategy 的 `cpu_threads`。`OMP_NUM_THREADS` 不决定这些路径的 worker 数。

`src/rk_npu_cpu_kernels.cpp` 里大量 `parallel for` 没有写 `num_threads`。legacy INT8、FP16 pack/unpack 使用进程当前的 OpenMP 团队大小，因此 `OMP_NUM_THREADS` 会影响它们。typed W8A8 的 worker 启动时会 `omp_set_num_threads` 到 strategy 的 `cpu_threads`（`pin_openmp_team`），这个设置是进程级的，之后那些没写 `num_threads` 的区域，包括 `M > 1` 时的 NPU 反量化拷回，跟着这个值走。autotune 测量前也会做同样的设置。

延迟敏感、又希望空闲时让出 CPU 时，在启动进程前设置：

```bash
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ./your_program
```

`OMP_WAIT_POLICY=PASSIVE` 选择被动等待。GNU libgomp 的 `GOMP_SPINCOUNT=0` 关掉一开始的自旋。它们改的是等待方式，不改计算线程数。同一策略要同时用于调优和推理；W4A8 换了等待策略之后用 `tune_w4a8 --refresh`，因为 cache 不会区分它们。

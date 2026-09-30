# 原始 MM/BMM 接口

新增入口为 [`include/rk_npu_mm.h`](include/rk_npu_mm.h)，实现位于
[`src/rk_npu_mm.cpp`](src/rk_npu_mm.cpp)。旧头文件、函数和默认行为保持不变。

实现 FP16 × FP16 → FP16、BF16 × BF16 → BF16、TF32 × TF32 → FP32，
统一计算 `C[b] = op(A[b]) @ op(B[b])`。
`batch_count=1` 即 MM；BMM 的每个矩阵形状相同。接口不包含量化、scale、bias
或激活。完整 FP32 输入计算、INT8、INT4 以及其它混合类型在新入口暂时返回
`RK_NPU_MM_ERR_UNSUPPORTED`，不会转换精度或退回 CPU GEMM。

独立 `compute` 字段及其枚举已删除，由 A/B/C dtype 组合决定执行路径。
这次调整改变了新 `rk_npu_mm_desc` 的布局，使用新接口的调用方必须与库一起重新编译，
包括 ctypes 等绑定；不能用旧 descriptor 二进制布局调用新库。旧 FP16 专用 C ABI 不变。

## 描述与生命周期

1. `rk_npu_mm_desc_init` 初始化形状、FP16 类型和连续的独立 batch；
   设置转置、行跨度和 batch 跨度。
2. `rk_npu_mm_plan_create` 在 CPU 上验证描述、确定策略和设备 buffer 大小。
   `rk_npu_mm_plan_get_info` 返回归一化描述、实际 split 数、核 mask 和内存需求。
3. `rk_npu_mm_workspace_create` 在指定 IOMMU domain 上准备命令和运行资源。
4. 重复调用相应的 `run`；释放 workspace、packed B，最后关闭 ctx。

plan 是不可变的 CPU 对象。workspace 和 packed B 都复制 plan 并持有 domain
引用，因此创建后可释放原始 plan/domain handle；ctx 必须活到所有对象释放之后。
同一个 workspace 不支持并发执行，冲突返回 `RK_NPU_ERR_BUSY`；释放对象不能与使用并发。
创建函数返回错误码，失败时把输出 handle 设为 NULL。

### 数据布局

- FP16/BF16 均使用 `uint16_t` 位模式。所有 stride 以元素为单位；所有 buffer size 以字节为单位。
- TF32 输入与 FP32 输出使用 `float`，设置 `a_type=b_type=RK_NPU_MM_TF32`、
  `c_type=RK_NPU_MM_F32`。TF32 输入对齐到 K16，FP16/BF16 为 K32。
- 默认类型仍为 FP16。BF16 调用方须同时设置 `a_type=b_type=c_type=RK_NPU_MM_BF16`，
  并提供真实 BF16 位模式；库不把 FP32 或 FP16 输入偷偷转换成 BF16。
- `lda/ldb` 是转置前矩阵的行跨度，零表示物理行宽。`ldc=0` 表示 N。
- A 的物理形状是 `[M,K]` 或 `[K,M]`；B 是 `[K,N]` 或 `[N,K]`。
- 输入 batch stride 为零表示广播；非零的 batch 不允许重叠。
- C 的 batch stride 为零表示 `M*ldc`，输出 batch 不允许重叠。
- 正 stride 和行间 padding 可用；负 stride、grouped GEMM 和输出转置未实现。
- 修改初始化后的形状或行跨度时，需同步修改 batch stride。只切换 transpose
  且保持紧凑布局时，原 batch stride 仍然正确。
- 新 packer 直接从带 stride/转置的主机数据写设备布局；不创建完整连续副本。
  连续、未转置、未 split 的路径复用旧的优化 batch pack/unpack。
- 主机 C 的行间/batch 间 padding 保持不变。主机 C 不允许与 A/B 的跨度重叠。

## 三种 workspace

| 模式 | 执行入口 | 资源与同步 |
| --- | --- | --- |
| `RK_NPU_MM_HOST_DYNAMIC` | `rk_npu_mm_run` | 持有 A/B/C；每次打包 A/B、同步、提交、解包/归约 |
| `RK_NPU_MM_HOST_PACKED_B` | `rk_npu_mm_run_packed_b` | 持有 A/C；B 由 `rk_npu_mm_packed_b_create` 打包并同步一次 |
| `RK_NPU_MM_DEVICE_ONLY` | `rk_npu_mm_run_device` | 只持有命令/task；调用方提供三个设备 buffer 并管理同步 |

`run_device` 也可使用其它模式的 workspace，但始终只提交设备计算。
其输出在 split 时是对应 c_type 的 partial，须经 `rk_npu_mm_unpack_c` 才得到最终矩阵。
CPU-only 的 `pack_a/pack_b/unpack_c` 可接受普通主机存储构成的 `rk_npu_mem`，
本身不做设备同步。设备执行要求相同 ctx/domain、足够容量、16 字节地址对齐和
不重叠的输入/输出；外部 dmabuf 的 ownership 协议仍由外部 owner 负责。

packed B 的兼容条件为同一 ctx/domain、dtype、K、N、batch 数、split 数和 part K。
同形状的 FP16/BF16 packed B 不能互换，错误类型在提交前返回 `RK_NPU_ERR_PARAM`。
TF32 的 packed B 也与其它 dtype 隔离。
M、输入/输出布局、N tile 和核 mask 可以不同，只要上述打包契约仍匹配。
源 B 在创建结束后即可释放。广播 B 当前在设备侧为每个 batch 保留副本，
尚未做共享设备地址优化。

`info.device` 列出所有 packed buffer 和控制内存的逻辑大小，不包含分配器页对齐
以及 C++ 元数据。dynamic workspace 数据量为 input+weight+output；packed-B
workspace 为 input+output，B 对象另占 weight；device-only 不分配数据 buffer。

## 多核与默认 split-K

新接口默认允许三核，`allowed_npu_core_mask` 支持 1/2/4/3/7；0 也表示 7。
实际 mask 可因任务不足缩小。内部沿用旧 FP16 batch 执行器，把 `[batch][split]`
展开成独立 GEMM，再将 M/N/batch/split 任务分配给各核。

未 split 时每个任务计算完整 K：batch 分组、M 分行或 N 分列可独立并行，
无跨核求和。任务按 batch、N tile、M tile 顺序生成，然后按任务数均分给各核。
默认 `n_tile=0` 不主动切 N；若 batch=1 且 M 只生成一个 tile，就只有一个任务，
实际使用一个核。设置正的 `n_tile` 可增加 N 方向任务，目前尚无自动 N 切块搜索。

`RK_NPU_MM_SPLIT_K` 在 **plan 创建时**读取：

| 值 | 行为 |
| --- | --- |
| 未设置或 `1` | 默认允许自动 split-K |
| `0` | 禁止自动 split-K |
| 其它，包括空串、`true`、`01` | 创建失败，返回 `RK_NPU_ERR_PARAM` |

自动规则：`K>=2048` 且 `batch_count<允许的核数` 时，split 数等于允许的核数；
否则为 1。每份 K 按输入类型向上对齐（FP16/BF16 为 32，TF32 为 16），尾部补零。
这是初始确定性规则，不是性能调优结论。

options 中 `split_k=-1` 使用环境变量；`0/1` 显式关闭/允许自动选择；`2/3`
强制相应分块数并忽略环境变量。强制分块要求可用核数足够且
`K>=输入K对齐粒度*split_count`。
创建之后改变环境变量不影响 plan、workspace、packed B。应在创建前设置环境，
避免 `setenv/getenv` 并发。该变量不影响任何旧接口。

NPU 写回 c_type 的 partial，CPU 按 K 顺序用 FP32 求和；FP16/BF16 最后再 RNE
舍入到对应类型，TF32 的 FP32 输出不做最后的低精度缩窄。
这不等价于完整 FP32 累加，也可能因 partial 溢出影响最终结果，尤其是大值抵消。
未 split 时直接保留 NPU 输出位模式。normal/native 与 N tile 均可和 split
组合使用；没有融合后处理或 CPU GEMM fallback。

BF16 复用同一 16 位布局与私有浮点执行器，在命令生成时选择 CNA/CORE/DPU 精度。
没有在运行时扫描修改命令，也没有经过 FP16 数值中转。NPU 内部累加顺序/精度
不构成严格 FP32 ABI；参考计算在个别舍入边界可能与实机相差一个 BF16 相邻值。
NaN payload/quiet 位、signed-zero 不保证保留。研究及数值边界见父工作区记录
`docs/bf16_mm_probe_2026-09-29.md`。

### TF32 数值契约

主机提供原始 FP32 数据，packing 只搬运位模式；本板测试表明硬件对有限 normal
输入截去低 13 位尾数，保留 10 位 fraction，随后输出 FP32。
它与本板 RNE 输入参考不等价，不承诺与其它厂商的 TF32 舍入语义一致。
如果调用方需要预先 RNE 到 10 位 fraction，可在输入本库前显式完成，库不会隐式代做。
非有限值、subnormal、NaN payload 尚未做完整行为认证；内部累加顺序不保证
与某一种 CPU FP32 顺序逐位一致。

设备 feature 为 K4/M4，weight 为 N16/K16，output 为 N4/M4；normal 布局也支持。
每元素四字节，大小查询、CBUF 切块、N/M/batch 地址和 split-K 归约均使用该宽度。

当前 TF32 后端要求每个 K 分片对齐后 `part_k<=4096`。不满足时在 plan 创建阶段
返回 `RK_NPU_MM_ERR_UNSUPPORTED`，不会提交 NPU，也不会隐式覆盖 split-K 设置。
大 K 可显式使用 2/3 路 split-K，或由调用方自行分块；本限制属于当前已验证配置，
不代表硬件 K 上限。初次 K=5120、不分 K 的 native 用例曾发生 submit timeout，
该配置已从正式接口支持范围中排除，进一步 CBUF 调优暂缓。

## 使用示例

以下假设 ctx/domain 已创建，A/B/C 是足够大的 FP16 主机数组；每个返回码都需检查。

```c
rk_npu_mm_desc desc;
rk_npu_mm_desc_init(&desc, M, N, K, batch_count);
/* 若 A/B/C 存的是 BF16 位模式，设置：
 * desc.a_type = desc.b_type = desc.c_type = RK_NPU_MM_BF16; */
desc.trans_b = 1;             /* B 的物理形状为 [batch,N,K] */
desc.batch_stride_b = 0;      /* 可选：所有 batch 共享同一个 B */

rk_npu_mm_plan *plan = NULL;
int rc = rk_npu_mm_plan_create(&desc, NULL, &plan);
if (rc != RK_NPU_OK) return rc;
rk_npu_mm_info info;
rc = rk_npu_mm_plan_get_info(plan, &info);
rk_npu_mm_workspace *workspace = NULL;
if (rc == RK_NPU_OK)
    rc = rk_npu_mm_workspace_create(domain, plan, RK_NPU_MM_HOST_DYNAMIC, &workspace);
if (rc == RK_NPU_OK)
    rc = rk_npu_mm_run(workspace, A, info.host_a_bytes,
                      B, info.host_b_bytes, C, info.host_c_bytes);
rk_npu_mm_workspace_free(workspace);
rk_npu_mm_plan_free(plan);
return rc;
```

关闭默认 split-K：`RK_NPU_MM_SPLIT_K=0 ./your_program`。
需要固定 packed B 格式跨多个 M 复用时，可显式指定相同 split 数，再查询兼容性。

TF32 使用同一流程，类型改为：

```c
desc.a_type = desc.b_type = RK_NPU_MM_TF32;
desc.c_type = RK_NPU_MM_F32;
/* A/B/C 均为 float 数组；没有 compute 参数。 */
```

## 验证

- `test_mm --cpu`：旧 pack/unpack 作为布局对照，覆盖转置、padding、广播、
  normal/native、N tile、split 1/2/3、最小输出 buffer、容量/跨度检查及环境快照。
- `test_mm --board`：真实 FP16 MM/BMM 与独立 FP64 参考比较，覆盖多核、
  split-K、动态/预打包 B、一致性、跨 M 复用、重复调用缓存更新、设备入口和生命周期。
- `test_mm_header_c`：新头文件与旧头文件共同通过 C 编译。

板端测试打印的首次完整调用耗时只是诊断值，没有定频/充分预热，不能作为性能比较。
当前无 autotuner，整数新后端尚未接入；旧接口继续提供原来的能力。

### 2026-09-29 初始 FP16 验证记录

- 主机 GCC 16.1.1 Debug 构建：12 项 CTest 全部通过（含新接口 55 组布局用例
  与策略/参数检查）；所有公开 C 头文件检查通过。
- 新实现与测试通过 `-Wall -Wextra -Werror` 语法检查。
- RK3588 板端 GCC 10.2.1 Release：`test_mm --board` 全部通过。测试中的
  有限 FP16 输入对照 FP64 参考，最大绝对误差约 0.000646；这不代表全值域或模型精度保证。
- 同一新库重链接旧 `test_batch_f16`，运行 `test_batch_f16 0`，8 个回归用例全部通过。
- 板端设置 `OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0`。
  split 开/关都执行了真实 NPU submit；未进行定频性能对比。

### BF16 接入

新增 BF16 转换/归约 CPU 测试，覆盖全部 65536 种 BF16 位模式、RNE 中点、
NaN/Inf 和超 FP16 范围的 partial。FP16/BF16 共用的 110 组布局用例覆盖转置、
stride、广播、normal/native、N tile 和 split 1/2/3。

板端测试对两种类型运行同一套 MM/BMM、动态/预打包 B、跨 M 复用、设备入口和
生命周期用例；另检查错误 dtype 的 packed B、BF16 超 FP16 范围的 split/unsplit
输出，以及同一 ctx 内 BF16 后切回 FP16。官方 canary 对照脚本为父工作区
`tools/bf16_probe/verify_public_api.py`，直接加载正式共享库，通过公开 ABI 调用。

接入后的主机 12 项 CTest、公开 C 头文件检查全部通过；板端上述 FP16/BF16
公共接口用例、大/小值 split/unsplit 及同 ctx 类型切换全部通过。正式库的四种
normal/native A/C 布局各自与官方 SDK 的 256 个 BF16 输出逐位一致。
旧 FP16 BMM 8 项回归、旧 MM/融合 ADD/MUL 的 30 次执行全部通过。
报告在父工作区 `tools/bf16_probe/results/2026-09-29/integration/report.json`
和 `public_api.json`，包含正式共享库 SHA256。尚无定频性能比较。

### TF32 与 descriptor 调整后的验证

13 项 CTest、公开 C 头文件检查通过；TF32 CPU 部分新增 53 组布局用例和
dtype/旧 descriptor 大小/K 分片上限检查。板端 TF32、FP16/BF16、旧 BMM、
旧融合算子及更新 ctypes 后的 BF16 官方结果对照均通过。
TF32 覆盖 K=4096 不分片、K=5120/11008 自动分片，以及截断边界和数值范围。
证据见父工作区 `docs/tf32_mm_2026-09-29.md` 和
`tools/tf32_probe/results/2026-09-29/integration/report.json`。

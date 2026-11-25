# BW1000（伯温 BMZ / DCU · gfx936）硬件参数 —— 优化速查

> 数据来源：`HPC_APP/dcu-benchmark-suite` 项目的 `L0_hardware/`（静态规格 + `docs/bw1000.md`）、
> `L1_microbench/`（实测带宽/延迟/算力）、`L3_prof/hipprof/`（`pmc_info.md` + `wave_occupancy.py`）。
> 提取日期：2026-08-16。来源标注沿用原项目约定：`[实测]`=本机 benchmark、`[BMZ]`=伯温官方规格、
> `[用户]`=平台专家、`[假定]`=推断未核实。
>
> 本文件只收录**与 GPU kernel 优化直接相关**的参数，通信/多卡项见文末简述。

---

## 一、架构核心

| 参数 | 值 | 来源 |
|---|---|---|
| SKU | 伯温 BMZ（单封装大芯片，无 Die 间互联） | [BMZ] |
| 架构 | DCU（C-3000 GPGPU），arch 代号 `gfx936` | [BMZ] |
| CU 数量 | **80 / 卡** | [实测][BMZ] |
| SIMD / CU | **4** | [用户] |
| Wavefront（波前） | **64 线程** | [用户] |
| 核心时钟 sclk | 最高 **1600 MHz**（实测有效 ~1.5 GHz） | [实测] |
| 向量位宽 | 256 bit（待核实） | [假定] |
| 硬件 wave 上限 | **40 wave/CU**（10/SIMD × 4 SIMD） | [用户] |

---

## 二、寄存器与 occupancy（kernel 优化的第一杠杆）

### 2.1 寄存器容量

| 参数 | 值 | 来源 |
|---|---|---|
| VGPR 文件 | **768 KB/CU = 196608 个 32-bit**（per-SIMD 分配，192KB/SIMD） | [BMZ] |
| 单线程最大 VGPR | **256** | [BMZ] |
| VGPR 分配粒度 | **4**（`ceil_to(vgpr, 4)`） | [用户] |
| SGPR | **800 / SIMD**（分配粒度 4） | [用户] |
| LDS（共享内存） | **64 KB/CU**（单 workgroup 上限假定 = 64KB） | [BMZ] |

> ⚠️ **关键认知**：BW1000 的 VGPR 是 **per-SIMD 分配**的（768KB/CU ÷ 4 SIMD = 192KB/SIMD）。
> 一个 wavefront 绑定到某个 SIMD，只能用该 SIMD 的 192KB 寄存器，余数**不能**跨 SIMD 共享。
> 因此 occupancy 的 VGPR 维度必须按 `每 SIMD 192KB` 各算再乘 4，而不是 per-CU 768KB 整体算
> （后者会**高估**：vgpr=216 时 per-SIMD 只有 3/SIMD = 12/CU，per-CU 整体会错算成 14）。
> 注：`L3_prof/hipprof/wave_occupancy.py` 的 `vgpr_kb_per_cu: 768` 整体模型与此不符，使用前需按 per-SIMD 修正。

### 2.2 occupancy 模型（`wave_occupancy.py` 官方实现）

```
max_waves/CU = min(
    硬件调度上限          40,
    VGPR 限制   = 4 × ⌊(768KB / 4 SIMD) / (ceil_to(vgpr,4) × 64 × 4B)⌋
              = 4 × ⌊49152 / (ceil_to(vgpr,4) × 64)⌋,   # 每 SIMD 192KB
    SGPR 限制   = (800 / ceil_to(sgpr,4)) × 4 SIMD,
    LDS  限制   = (64KB / lds_per_wg) × ceil(wgr/64)
)
```

### 2.3 VGPR → wave/CU 换算表（per-CU 768KB 池）

| vgpr/线程 | 每 SIMD wave | 每 CU wave | occupancy(/40) |
|---|---:|---:|---:|
| ≤256（单线程上限） | 3 | 12 | 30% |
| **216（TorsionTeam 当前）** | **3** | **12** | **30%** |
| ≤192 | 4 | 16 | 40% |
| ≤152 | 5 | 20 | 50% |
| ≤128 | 6 | 24 | 60% |
| ≤108 | 7 | 28 | 70% |
| ≤96 | 8 | 32 | 80% |

> 每 SIMD wave = `⌊49152 / (ceil_to(vgpr,4) × 64)⌋`，每 CU = ×4。vgpr 按 4 对齐（153→156、170→172）。
> 要跨到下一个整档，关键阈值：**≤192 → 16 wave（40%）**、**≤152 → 20 wave（50%）**、
> **≤128 → 24 wave（60%）**。vgpr 从 216 降到 192 仍同属 3/SIMD 档，真正的跳变点在 192 / 152 / 128。

### 2.4 SGPR 也是隐藏限制

SGPR 800/SIMD × 4 = 3200/CU。若 kernel 的 sgpr 高（复杂 kernel 常见 80–100），
SGPR 维会先于 VGPR 触顶。**看 PMC 时务必同时看 sgpr**，不能只盯 vgpr。

---

## 三、内存层级

```
VGPR 768KB/CU ── LDS 64KB/CU
      │             │
      └── L1 16KB/CU ┘
            │
            ↓
        L2 8MB（全卡共享）
            │
            ↓
     HBM2e 64GB（4 颗，4096 bit，理论 1.8 TB/s）
```

| 层级 | 容量 | 说明 |
|---|---|---|
| LDS | 64 KB/CU | 超过即 spill 到全局 |
| L1 | **16 KB/CU** | 小于常见 GCN 的 32KB，小 tile 更依赖 L2 |
| L2 | 8 MB（全卡共享） | 8MB 容量边界清晰，>8MB 带宽腰斩 |
| HBM2e | 64 GB（可分配 65.3GB） | 4096 bit 位宽，理论 1.8 TB/s |

---

## 四、实测带宽（`L1_microbench/memory/BW1000_results.md`）

### 4.1 HBM 流式（256MB >> 8MB L2，float4）

| 模式 | 带宽 (GB/s) | 达成率(/1800) |
|---|---:|---:|
| read | 1216.6 | 67.6% |
| write | 1216.6 | 67.6% |
| copy | 1325.9 | 73.7% |
| scale | **1335.3** | **74.2%**（峰值） |
| add | 1276.8 | 70.9% |
| triad | 1277.6 | 71.0% |

> **HBM 实测峰值 ≈ 1.34 TB/s**（scale float4），理论 1.8T 的 ~74%。
> roofline 分析建议用 **1.2T** 作带宽峰值（`pmc_info.md` 采用值）。
> float4 对纯 read/write 无增益（访存指令已饱和），对 compute-heavy（scale/copy）有 ~2.7% 提升。

### 4.2 L2 带宽扫频

| 工作集 | 512KB | 1MB | 2MB | 4MB | 8MB | 16MB | 32MB | 64MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| read (GB/s) | 5948 | **5968** | 5964 | 5938 | 5477 | 2516 | 1316 | 1201 |
| write (GB/s) | 2174 | 2181 | 2182 | **2250** | 2226 | 1559 | 1139 | 1089 |

> **L2 read 平台 ~6.0 TB/s、write ~2.25 TB/s**；工作集 >8MB 后跌落（16MB 起 read 腰斩），
> 64MB 收敛到 HBM。**8MB 是 L2 容量硬边界**。

### 4.3 LDS 带宽（320 blocks = 80CU×4，256 threads）

| 模式 | 带宽 (GB/s) | 达成率 |
|---|---:|---:|
| read（stride 1，无冲突） | **15453** | 94.3%（/16.38T 理论） |
| write（stride 1） | 11797 | 72.0% |
| rw（双向计数） | 30437 | >100% 正常 |
| read（stride 32，max bank conflict） | **2455** | **仅 15.9%** |

> LDS 理论 16.38 TB/s = 128B/cycle/CU × 80 CU × 1.6 GHz。
> **stride 32 全 wavefront 命中同 bank → 带宽退化 84%**，是 LDS 密集 kernel 的关键约束。

### 4.4 随机访问（128B 粒度）

| 访问 | 带宽 (GB/s) | 相对顺序 read |
|---|---:|---:|
| random read | 154.8 | **12.7%** |
| random write | 154.8 | 12.7% |

> 随机跳转 = 顺序的 **12.7%**，落在 cache-line 浪费模型（16B/128B = 12.5%）附近，
> DRAM row-hit 率大幅下降。**torsion 旧 kernel 的 L1→L2 随机 gather 瓶颈即源于此。**

### 4.5 host↔device（PCIe Gen5）

H2D 峰值 **48.5** GB/s、D2H 峰值 **47.0** GB/s（理论 ~63 GB/s 的 ~76%）。D2H 饱和更快（8MB），H2D 要到 128MB。

---

## 五、实测延迟（`L1_microbench/latency/BW1000_results.md`）

### 5.1 指令依赖链（单 wavefront 单链，有效时钟 ~1.5 GHz）

| 指令 | FP32 (cycles) | FP32 (ns) | FP64 (cycles) | FP64 (ns) |
|---|---:|---:|---:|---:|
| add | 3.5 | 2.33 | 4.5 | 3.00 |
| mul | 3.5 | 2.33 | 4.5 | 3.00 |
| **fma** | **8.5** | **5.67** | **4.5** | **3.00** |
| div | 57.5 | 38.34 | 61.5 | 41.00 |
| sqrt | 140.0 | 93.34 | 103.5 | 69.00 |

> ⚠ **fp32 fma（8.5c）明显长于 fp64 fma（4.5c）**——DCU 硬件特性，
> **不要把关键依赖链放在 fp32 fma 上**。fma 吞吐：chains=4 时 2.125 cycles/op（多独立链重叠）。
> div/sqrt 是长延迟指令（sqrt 达 104–140c），在 ReaxFF 的 sqrt/trig 密集路径里是关键成本。

### 5.2 内存层级延迟（指针追逐）

| 层级 | 延迟 |
|---|---|
| LDS | **42.67 ns** |
| L2（≤8MB） | ~140–157 ns |
| HBM（≥32MB） | ~260–277 ns（64MB 单点 277.5ns） |

> LDS 比全局快 ~6.5×；L2 比 HBM 快 ~2×；8MB L2 容量边界与带宽扫频结论一致。

---

## 六、算力峰值（`L1_microbench/compute/BW1000_results.md`）

### 6.1 理论峰值（[BMZ]）

| 精度 | Vector | Tensor |
|---|---:|---:|
| FP64 | 15.4 TFLOPS | 30.8 TFLOPS |
| FP32 | 30.8 TFLOPS | 61.4 TFLOPS |
| TF32 | — | 245.7 TFLOPS |
| FP16 / BF16 | 30.8 TFLOPS | 491.5 TFLOPS |
| INT8 | 61.4 TOPS | 884.7 TOPS |

> 规律：Vector FP32:FP64 = 2:1；Tensor 同精度 = Vector ×2。**不支持 FP8/FP6/FP4**。

### 6.2 实测达成率（真实 FMA）

| 精度 | 理论 | 实测 | 达成率 |
|---|---:|---:|---:|
| Vector FP64 | 15.4T | 15.003 | **97.4%** |
| Vector FP32 | 30.8T | 28.363 | **92.1%** |
| Vector FP16 | 30.8T | 29.887 | 97.0% |
| Tensor FP32 | 61.4T | 57.169 | 93.1% |
| Tensor FP16/BF16 | 491.5T | 442.0 | 89.9% |
| Tensor INT8 | 884.7T | 883.5 | **99.9%** |

> ⚠ **FP32 测法极敏感**：累加器形式 `acc=acc*a+c` 才测到 ~28T；常量操作数形式
> `a=a*b+c` 被编译器插入 ~200 条 `v_mov` 稀释，掉到 ~8.6T。**写 kernel 时保持真 FMA 累加链。**

---

## 七、Roofline ridge point

```
ridge = 算力峰值 / 带宽峰值（带宽用实测 1.2 TB/s）
```

| 精度 | ridge point |
|---|---:|
| FP64 | 15T / 1.2T = **12.5 FLOP/B** |
| FP32 | 30.8T / 1.2T = **25.7 FLOP/B** |

> kernel 计算访存比 I > ridge → 算力受限；I < ridge → 带宽受限。
> 计算访存比的分母用 **DRAM 实际字节**（FETCH_SIZE + WRITE_SIZE），不是名义 load 量。

---

## 八、PMC 指标诊断速查（`L3_prof/hipprof/pmc_info.md`）

### 8.1 资源字段（wave 并发估算输入）

`vgpr`（向量寄存器/线程）、`sgpr`（标量寄存器/线程）、`lds`（LDS 字节）、`wgr`（work-group size）、
`scr`（scratch 字节，>0 表示 spilling 到 local memory）、`fbar`（fbarrier 数）。

### 8.2 性能字段含义

| 指标 | 含义 |
|---|---|
| `kernel time` | `GRBM_GUI_ACTIVE / (max_engine_clk_fcompute × 1e6)` |
| `processed ALU` | 计算单元负载率 = `100 × SQ_ACTIVE_INST_VALU × 4 / SIMD_NUM / GRBM_GUI_ACTIVE` |
| `L1 cache unit is stalled` | L1 等待占比 = `100 × TCP_TCP_TA_DATA_STALL_CYCLES_MAX / GRBM_GUI_ACTIVE / SE_NUM` |
| `L2 cache hit rate` | `100 × TCC_HIT_SUM / (TCC_HIT_SUM + TCC_MISS_SUM)` |

### 8.3 诊断速查

| 现象 | 指标特征 | 诊断 |
|---|---|---|
| ALU 利用率高 + ktime 长 | `processed ALU`≈100% | 计算受限 |
| L2 hit 低 | <50% 且 ktime 长 | 访存受限 / 工作集过大 |
| LDS bank conflict 高 | >10% | 共享内存冲突，改 stride |
| L1/L2 stall 高但 ALU 空闲 | stall 高、ALU 低 | 访存延迟未隐藏，wave 并发不足 |
| `size of L2 cache read` << 名义读取量 | 远小于代码 load 量 | L2 命中率高，DRAM 流量小 |

### 8.4 实测 vs 理论 wave

```
平均并发 wave = SQ_ACCUM_PREV_HIRES / GRBM_GUI_ACTIVE        （全 SIMD）
每 SIMD 平均  = 上式 / SIMD_NUM
```

实测远低于理论估算 → 调度/依赖/资源分配问题。

> ⚠ **BW1000 (gfx936) 无指令细分计数器**：`flops1/2/3` 分组仅 BW1100 支持。
> BW1000 只能用 `SQ_INSTS_VALU × 64 × 2` 估算 FLOP（把全部 VALU 当 FMA 的上界）。

---

## 九、通信 / 多卡（简述，单卡 kernel 优化可跳过）

- 卡间互联 Hylink 1.0（7 个 xHCL）：单向 224 / 双向 448 GB/s（标称）。
- 单机 P2P 实测：单向 176 GB/s（78.7%）、双向 316 GB/s（70.5%）；真往返延迟 ~33µs。
- 单机集合通信 bus_bw 收敛到 ~117–121 GB/s（大包）；alltoall 跨节点显著变慢。
- 节点可见 8 HCU，全连接 HSW 拓扑。

---

## 附：来源文件清单

| 主题 | 来源路径（`HPC_APP/dcu-benchmark-suite/`） |
|---|---|
| 静态规格 | `L0_hardware/specs/bw1000.yaml` |
| 硬件知识 | `L0_hardware/docs/bw1000.md` |
| 带宽实测 | `L1_microbench/memory/BW1000_results.md` |
| 延迟实测 | `L1_microbench/latency/BW1000_results.md` |
| 算力实测 | `L1_microbench/compute/BW1000_results.md` |
| 通信实测 | `L1_microbench/comm/BW1000_results.md` |
| PMC 手册 | `L3_prof/hipprof/pmc_info.md` |
| occupancy 脚本 | `L3_prof/hipprof/wave_occupancy.py` |

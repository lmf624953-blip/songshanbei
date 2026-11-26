# Q128 packed 资源与调度实验

正式推理使用仓库 `optimization/rebuild-v2.sh --long-attention-q128-packed --fused-qknorm-mrope` 构建的三镜像方案。当前实验基准来自该方案的 packed 分支，长度动态，实验主机只用于每卡七个 head、dim128 和步长 896/896/2688/896 的回放。不要将提取出的单镜像目录当作通用正式推理配置。

## 重建实验基准

激活已适配的运行环境，设置 `H3_AUTOTUNE_CODE_ROOT` 为仓库绝对路径。先通过 `H3_AOT_SAVE_IR=1` 在新的构建目录生成正式三镜像，也可复用已经保存 IR 的正式构建。令 `production_attention` 指向其中的长注意力目录，`reference` 指向尚不存在的实验目录。

```bash
python prepare-q128-packed-reference.py "$production_attention" "$reference"
mkdir "$candidates"
cp -a "$reference" "$candidates/v0"
python build-q128-backend.py "$reference" "$candidates/v-lse-rounding" lse-rounding
python screen.py "$reference" "$candidates" "$results"
```

`screen.py` 先比较对照可加载 ELF 区段，再覆盖普通/放大 QK、完整输出、有限值和末尾哨兵。数值通过后使用九组交替顺序计时，两个生产长度均至少节省 0.5% 才列入 `winners`。正式默认的替换另需动态长度、八卡真实回放、官方完整生成、独立质量与多用例稳态。

## 已验证的取舍

- 提前整段 LSE、只提前概率 sum、V 预取和对应寄存器/调度约束可缩短 live range，但编译器额外融合缩放与减法，产生少量 BF16 输出差异。
- 名称含 `rounding` 的候选使用没有数学指令的编译边界保留原 FP32 缩放舍入，恢复逐位一致。当前全部无额外可靠收益。VGPR192 的 LSE 版本接近持平，8–20 字节私有存储的代价需按实际性能判断。
- K 的 phase/maxPhase/vec 六种变体均逐位一致，当前全部变慢，保留正式布局。
- P 直接布局转换逐位一致，但回退约 1%。8 warps 逐位一致，但回退约 8%–9%。

证据位于 [舍入与调度](../../docs/q128-rounding-1218234.json)、[K LDS](../../docs/q128-k-layout-1218234.json)、[P 转换](../../docs/q128-p-convert-1218234.json) 和 [不同长度](../../docs/q128-lengths-1218234.json)。资源下降不等同于已验证的性能收益。所有候选都保留全部 KV 遍历、BF16 矩阵输入与 FP32 累加。


## Q128 packed 的 PV 操作数打包候选

`build-pv-wide-operand.py` 把 PV 的 `kWidth=4` 改为8，保留相同数量的BF16 MMAC指令、全部KV遍历及FP32缩放舍入边界，研究宽LDS读取是否更有利。新增P向量布局、V布局及P直接转换对照。编译已完成；两种V布局出现8/24B private segment，会跳过GPU微测；其余镜像仍无scratch。P直接转换版本为213VGPR、10处barrier，单凭这些资源变化不宣布更快。

`screen-pv-wide-operand.py` 先核对三镜像分派、哈希和ABI，排除相同`.text`与spills，再在实际packed stride的27570/22016长度交替计时，并检查普通、放大输入及末尾哨兵。只有两个长度均逐位一致且收益至少0.5%才进入真实去噪验证。本轮微测等待compact归约与row SwiGLU的七次正式生成退出，不与正式计时并行。

## Q128 packed 的 K 预取

`build-k-lookahead.py` 在完整 K 块主循环中携带下一块 K，比较 QK 后和 PV 前两种预取位置。初始块和下一块均带完整块边界掩码，尾块保留原处理。使用同一个正式三镜像构建目录作为基准，候选只替换 packed 镜像。

```bash
mkdir "$builds"
python build-k-lookahead.py "$reference" "$builds/after-qk" after-qk
python build-k-lookahead.py "$reference" "$builds/before-pv" before-pv
python screen-pv-wide-operand.py "$reference" "$builds" "$results"
hipprof --pmc --pmc-type 3 --kernel-name dense_attention_aot_kernel -o "$pmc" \
  python "$H3_AUTOTUNE_CODE_ROOT/optimization/profile-attention-pmc.py" "$build" --sequence 27570 22016 --iterations 3
```

长度 27570 和 22016 上，两种预取均通过普通与放大 QK 的逐位比较和输出尾部哨兵检查，分别回退 4.19%–4.23% 和 6.12%–6.31%。独立 PMC 的硬件分配从 216 VGPR 增至 232/236，LDS 均为 8192 B，scratch 为零。平均活跃 wave/CU 约为 11.32/10.59，L2 命中率约为 97.8%/97.5%。这些指标无法将性能回退归因于占用率下降。两种候选均未进入八卡回放。见 [配对计时](../../docs/q128-k-lookahead-1218234-micro.json) 与 [PMC 对照](../../docs/q128-k-lookahead-1218234-pmc.json)。

## Q128 packed 的 CTA 发射顺序

`build-q128-cta-order.py --query-group 1/4/8` 以 query 分组、head 和组内 query 依次枚举 CTA，保留原始主机 ABI 和动态序列长度。`--query-group 0` 是原顺序对照。可用 `--rounding-fence` 重现额外舍入边界的实验，再用 `screen-pv-wide-operand.py` 对比。

额外舍入边界令原顺序本身变慢约 2.45%–2.51%。保留原舍入方式时，对照的 `.text` 与正式镜像一致。三种重排均通过普通与放大 QK 的逐位比较，配对收益为 0.12%–0.25%，未达到八卡回放门槛。4 块分组在长度 27570/22016 上的 TCC 命中率由约 97.78%/97.48% 降至 90.29%/90.53%，两者硬件资源均为 216 VGPR、8192 B LDS、零 scratch。见 [计时与 PMC 记录](../../docs/q128-cta-order-1218234.json)。

## Q128 packed 的 P/V LDS 时序

`build-pv-lds-order.py` 比较 `swap`、`after-qk`、`before-probability-cast` 和 `prefetch-only`。它只调整 V 的加载与 LDS 时序，保留原点积和归约表达式。对提前读取的变体，使用 `--rounding-fence` 可保留原 FP32 缩放舍入。

仅交换 P/V 的 LDS 顺序时，27570/22016 两个长度的普通和放大输入均逐位一致，耗时增加 2.60%/2.65%。提前 V 全局读取后，两种 LDS 时序在原舍入优化下均出现 1 个 BF16 ULP 的差异。加舍入边界后输出恢复逐位一致，单独预取慢约 1.9%–2.0%，更早的 LDS 读写慢约 2.9%–6.1%。独立 PMC 中，交换顺序的 LDS 指令数和 bank conflict 计数与对照相同，LDS 等待计数增加约 20%。本轮无候选进入八卡回放。见 [配对计时与 PMC](../../docs/q128-pv-lds-order-1218234.json)。

## Q128 packed 的 P LDS 连续轴

对正式 Q128 三镜像构建目录运行 `build-p-lds-axis.py`，仅在 packed 镜像中修改 P 的 LDS swizzle，保留 BF16 数据、矩阵操作数布局、K 分块和累加顺序。以下命令构建目前微测与八卡回放最快的配置。

```bash
python build-p-lds-axis.py "$reference" "$builds/row-v4-phase2" row-v4-phase2
python screen-pv-wide-operand.py "$reference" "$builds" "$results"
bash "$H3_AUTOTUNE_CODE_ROOT/optimization/run-optimized.sh" --config "$baseline_config" --replay-run \
  --model "$model" --captures "$captures" --out "$replay" --steps 1,25,49 --iterations 5 \
  --attention-ablation "$builds/row-v4-phase2"
```

`row-v4-phase2` 使用 `vec=4, perPhase=2, maxPhase=8, order=[1,0]`。两个生产长度的单算子配对中位收益为 1.16% 和 1.05%，20 个边界长度均逐位一致。八卡第 1、25、49 次去噪输入上，音视频张量和各算子调用数一致，三步中位耗时之和缩短约 0.69%。独立 PMC 中，LDS 指令计数约下降 27%，bank conflict 计数下降 40%，L2 命中率基本持平。见 [测量记录](../../docs/q128-p-lds-axis-1218234.json)。

普通官方入口 A 完整执行了 49 次 Transformer，输出规格和独立参考质量通过，MP4 与现用版逐字节一致。独立进程用时 261.734 秒，包含模型加载和首次编译。常驻序列完成两次正式 A 后依用户要求停止，端到端分别为 165.866 和 164.816 秒，去噪分别为 145.738 和 145.737 秒，成片均逐字节一致。两条的去噪比现用版三次均值约快 0.67%。样本数不同，这组数据仅作为两条观测，不作为正式稳态成绩。候选保留在实验框架里，正式构建开关已撤除。
## Q128 packed 的 K 加载线程与 LDS 轴

`build-k-producer.py` 只调整 K 的加载线程或 LDS 排布，保留 QK/PV 的矩阵操作数和归约布局。用正式 Q128 三镜像目录作为 `reference`，构建 `d4-n1`、`d8-n2`、`d16-n1`、`d8-n2-wide`，再以 `original --row-lds` 构建 LDS 转置对照。然后用 `screen-pv-wide-operand.py` 测两个生产长度。

五种配置都通过普通和放大 QK 输入的逐位比较。最接近对照的 `d8-n2` 在长度 27570 慢 0.10%，在长度 22016 快 0.02%。K 的 LDS 转置慢约 21%。独立 PMC 表明转置使 LDS 指令数增至约 2.35 倍、bank conflict 计数增至 3.8 倍，而 VMEM 读取指令数相同。所有候选均归档，本轮没有进入八卡验证的配置。见 [配对计时与 PMC](../../docs/k-producer-a14r1n11.json)。

## 在线 softmax 累加器缩放时序

用 `build-softmax-rescale-order.py REFERENCE OUTPUT PLACEMENT` 构建 `chain-after-p`、`multiply-before-pv` 和 `chain-before-pv`，再用 `screen-pv-wide-operand.py` 做逐位比较和配对计时。变体移动 FP32 缩放乘法或其整条依赖链，保留原始的在线 softmax 计算与 QK/PV 点积。

三种时序在长度 27570 和 22016 上都逐位一致。最接近基准的 `chain-after-p` 慢 0.17% 和 0.10%；`chain-before-pv` 慢 2.21% 和 2.00%。独立 PMC 中，后者的可分配 VGPR 从 216 增至 224，LDS 等待计数增加约 18%，LDS 指令和 bank conflict 计数相同。本轮归档三种时序。见 [逐位校验、配对计时与 PMC](../../docs/softmax-rescale-a14r1n11.json)。

## 完整 KV 块读取掩码

用 `build-full-kv-unmasked.py REFERENCE OUTPUT {k,v,kv}` 构建候选，使用 `screen-pv-wide-operand.py` 校验和配对计时。构建脚本校验原始 IR 哈希。现用主机以七个 head 和 `ceil(rows/128)` 个 Q 块发射，完整 KV 循环处理 `floor(rows/32)` 个 32 token 块。脚本只移除该循环中的读取掩码，保留尾块和输出掩码。

K-only 在两个生产长度逐位一致，配对收益为 0.25% 和 0.47%。V-only 和 KV 联合版本在原编译方式下有一个 BF16 ULP 的差异。加入 `--rounding-fence` 后恢复逐位一致，分别回退约 7% 和 5%。独立 PMC 中，K-only 的全局读取、LDS 指令和 bank conflict 计数一致，LDS 等待计数增加约 3%。本轮收益低于晋级门槛，保留实验源码和数据。见 [实测数据](../../docs/full-kv-unmasked-a14r1n11.json)。

## P 和 V 的 LDS 同时分配

用 `build-pv-lds-overlap.py REFERENCE OUTPUT` 同时发出 P、V 的 LDS 分配与写入，再读回两个操作数。默认覆盖完整循环和尾块。`--full-only` 仅改完整循环，`--v-first` 调整两个操作数的读回次序。脚本校验原始 packed IR 的摘要和局部依赖。LDS 地址分配和同步由编译器完成。

三种候选在长度 27570、22016 的普通及放大输入上均逐位一致。LDS 从 8192 B 增到 16384 B，静态 barrier 数从 12 降到 8 或 10。配对计时回退 0.63%–1.30%。独立 PMC 中，完整循环与尾块都改的版本保持与对照相同的全局读取指令、LDS 指令和 bank conflict 计数，LDS 等待计数增加约 6%。本轮保留实验记录。见 [配对计时与 PMC](../../docs/pv-lds-overlap-a14r1n11.json)。

## P 的 wave 内寄存器交换

`build-p-wave-exchange.py REFERENCE OUTPUT` 根据冻结的 Q128 packed LLVM IR 替换 P 的 LDS 写读。脚本验证源文件摘要，并枚举 4096 个位置，核对源寄存器及读写双方的 wave 归属。使用 `--full-only` 可只替换完整块。两个镜像均保留 BF16 MMAC 指令和 K、V 的同步。

在长度 27570 和 22016 上，两种方案的普通和放大输入均逐位一致，配对计时回退 4.21%–4.28%。同时改完整块和尾块时，静态 barrier 从 12 处降至 8 处，增加 32 条 `ds_bpermute_b32`。独立 PMC 显示，LDS 指令计数减少约 31%，LDS 等待计数减少约 36%，VALU 指令计数增加约 5%。候选未进入八卡验收。见 [配对计时与 PMC](../../docs/p-wave-exchange-a14r1n11.json)。

加 `--quad-dpp` 可用 DPP quad shuffle 执行前两级交换，保留最终的跨 lane 寄存器映射。两种范围均通过逐位比较，配对回退 3.69%–3.83%。完整替换的静态 `ds_bpermute_b32` 由 32 条减到 16 条。相对现用镜像，独立 PMC 的 LDS 指令计数下降约 46%，LDS 等待计数下降约 51%，VALU 指令计数增加约 6%。本轮只归档。见 [DPP 配对计时与 PMC](../../docs/p-wave-dpp-a14r1n11.json)。

在 `--quad-dpp` 上加 `--rounding-mode paired` 可在软件舍入后直接拼接两个 BF16 高半字。`--rounding-mode unpacked` 在交换后才提取高半字，每次交换传递一个已舍入的 32 位整数。两种方式都复用原 RNE 舍入和 `0x7fff` NaN 编码。DTK 26.04 汇编器在 `gfx936` 上拒绝 `v_cvt_pk_bf16_f32`，接受 `v_perm_b32`。

两种生产长度的普通与放大输入均通过逐位校验。直接打包回退 3.72%–3.77%，独立 PMC 的全局读取、LDS 和 VALU 指令计数与上一轮 DPP 相同。延迟截取回退 6.18%–6.24%。本轮只归档。见 [配对计时与 PMC](../../docs/p-round-pack-a14r1n11.json)。

## P 的 LDS 宽访问与 wave 内等待

`build-p-wave-lds.py REFERENCE OUTPUT` 以 `s_waitcnt lgkmcnt(0)` 代替 P 写后读前的 workgroup barrier。基准映射的全部 4096 个 BF16 存取位置均已逐一核对，每个元素由同一 wave 写入并读回。加 `--dedicated` 把 P 放进单独的 LDS 区域，省去 K 读取和 P 写入之间的 workgroup barrier。V 写前仍保留原 barrier，保护共享区里跨 wave 的 K 读取。

通过 `--packed linear|swizzled` 将相同读写 lane 的两个或四个 BF16 合成一次访问，`--items 4` 选择四个。`--packed padded --items 4` 使用每槽 65 个 64 位字的布局，使编译器生成 `ds_write2_b64` 和 `ds_read2_b64`。`--packed wide-read --items 4` 将槽距调整为 66 个字，在读回端合并成 128 位。脚本枚举地址，按访问宽度核对连续线程的 bank 分布，并验证读回对齐。`--workgroup-publication` 保留 P 读写之间的原 barrier。各模式均由现用 Q128 packed LLVM IR 的固定 SHA-256 开始构建。

17 个变体在长度 27570 和 22016 的普通和放大输入上均逐位一致，复建的可执行 `.text` 与实测镜像一致。最快的 128 位读回版保留原 barrier，配对耗时分别缩短 0.17% 和 0.09%。独立 PMC 的 LDS 指令下降约 54%，LDS 等待计数下降约 60%，全局读取指令相同，VALU 指令增加约 1%。只换 wave 内等待的版本慢约 1.7%。本轮没有进入八卡或官方成片回归的候选。见 [配对计时、复建与 PMC](../../docs/p-wave-lds-a14r1n11.json)。

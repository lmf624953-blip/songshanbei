# Q128 packed 资源与调度实验

2026-11-27 当前正式配置为 P 行 LDS + V 预排布 + 内部 P payload/ROR4 精简，已完成真实八卡、普通官方 A、三组稳态 A 配对和 B/C/返回 A 验收。官方 A 完整请求均值由本轮 137.001 降为 132.825 秒，下降 3.049%，全部成片与对应已验收版本逐字节一致。此前 137.344 秒为上一阶段版本，保留在本文件的历史实验中。稳定 max 单项仅投影约 2.608% 完整请求收益，暂不选用。 [当前统一结果](../../docs/p-nan-ror4-f12r2n09-public-a.json)

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

## 同一线程块内的双 Q128 K/V 复用

`build-dual-q128.py REFERENCE OUTPUT` 将两个同一 head 的相邻 Q128 tile 放进同一个 512 线程块。每组仍按原四个 wave 计算 QK、在线 softmax 和 PV，Q 尾部掩码覆盖第二组无有效行的情况。`--shared-kv` 只让前四个 wave 读取并向 LDS 发布 K/V，两个 Q tile 各自读回 K/V，P 保留各自的 LDS 地址范围。K/V 前后同步仍由整个线程块参与。实验主机库只改变 packed 镜像的网格和线程数，保留另两个镜像的原有启动参数。

用 `screen-dual-q128.py REFERENCE BUILDS OUTPUT` 复现时，`BUILDS` 下两个目录分别命名为 `independent-kv` 和 `shared-kv`。两个生产长度和 12 个边界长度的普通输入、放大输入及输出哨兵检查均通过。两个生产长度的三组对照各做 11 组无 profiler 交替配对。27570 token 下，单 Q128 为约 35.10 ms，独立双 Q128 为 115.45 ms，共享双 Q128 为 99.56 ms。共享版相对独立版节省约 13.86%。22016 token 的同类配对节省约 13.40%。

独立 PMC 确认两种双 Q128 均分配 128 VGPR、16 KiB LDS，并分别使用 344 B、320 B scratch。共享版使总 VMEM 读指令数比独立版少约 15%，但与无 scratch 的单 Q128 相比仍约为 10.6 倍。此计数涵盖编译器生成的 scratch 读写，仅用于解释当前实现的整体访存变化。本轮没有候选进入八卡回放或官方成片验收。见 [配对、边界、复建和 PMC](../../docs/dual-q128-a14r1n11.json)。

## Softmax wave 内归约的 lane 交换

`build-softmax-xor.py REFERENCE OUTPUT` 在原归约树中以 `row_ror:8` 替换异或 8 的两条带 bank mask 的 DPP。`--scope max|sum` 分别限制到最大值或求和。`--row4 ds` 和 `--row8 ds` 可比较 `ds_swizzle_b32` 的地址映射。脚本核对 64 个 lane 的映射、完整与尾块的变更位置，并保留原 FP32 算术顺序和 BF16 MMAC。

六个镜像在两个生产长度的普通与放大输入上均逐位一致。最大值和求和同时用 `row_ror:8` 时，27570/22016 token 配对分别缩短 1.69%/2.01%。静态 `v_mov_b32_dpp` 从 192 条降至 160 条，`s_nop` 从 42 条降至 15 条。独立 PMC 中 VALU 指令减少约 2.6%，LDS 和全局读指令计数相同，硬件仍分配 216 VGPR、8192 B LDS 和 0 B scratch。DS 路径均回退。`--mov-dpp` 与 `--dpp-combine` 的三种组合生成了和最优旋转镜像相同的可加载区段，未重复计时。本轮收益低于端到端优先线，继续归档。见 [配对、编译器对照与 PMC](../../docs/softmax-xor-a14r1n11.json)。

## Softmax DPP 交换与算术融合

`build-softmax-dpp-arithmetic.py REFERENCE OUTPUT {row,quad-last,quad,all}` 在行旋转版的归约树上分别融合不同级别的交换与 `v_max_f32_dpp`、`v_add_f32_dpp`。每个内联 DPP 前显式等待 VGPR 数据，非最后一级在输出后也保留等待。异或 4 的两条掩码交换、FP32 归约顺序及 BF16 MMAC 保留。四个候选的普通和放大输入均通过逐位比对。

可直接从九齿重新生成基准。构建时保留 LLVM IR，融合脚本在校验摘要时只忽略 `DIFile` 的目录字段。以下命令假定已加载项目运行环境，`code` 指向交付仓库，`control` 和 `candidate` 是两个新的空路径。

```bash
H3_AOT_SAVE_IR=1 python "$code/optimization/adaln/build-attention-aot.py" "$control" \
  --fp32-scale --aligned-strides --query-block 128 --key-block 32 --warps 4 --stages 1 --kpack 2 \
  --optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops \
  --coalesce-pv-value --direct-attention-query --packed-attention-strides
python build-softmax-dpp-arithmetic.py "$control" "$candidate" all
```

`all` 相对上一版镜像在 27570/22016 token 配对分别缩短 7.10%/7.00%，20 个额外长度逐位一致。独立 PMC 中 VALU 指令减少约 15.6%，LDS 和全局读取指令数相同，scratch 为零。八卡第 1、25、49 次去噪的输出逐位一致，三个配对中位数合计缩短 4.13%。设备镜像从重新生成的九齿 IR 构建，三个镜像的可加载区段均与实测版一致。见 [单卡配对、八卡回放、复建与 PMC](../../docs/softmax-dpp-arithmetic-a14r1n11.json)。

普通官方 A 通过规格检查，成片与上一验收版逐字节一致。独立常驻实例做三组 A 交替配对，端到端均值由 167.695 秒降至 161.959 秒，节省 3.420%；Transformer 均值由 146.748 秒降至 140.700 秒，节省 4.121%。三组端到端降幅分别为 3.954%、2.528%、3.769%。常驻 11 条请求包含 B、C 和返回 A，每条均执行 49 次 Transformer，成片分别与已验收结果逐字节一致。用 `summarize-softmax-dpp-steady.py` 从两次验收的原始记录重算配对与哈希检查。已将本地 `selected-best.env` 指向按上述步骤重建的 `attention-q128-softmax-dpp-a14r1n11`。见 [成片与常驻验收](../../docs/softmax-dpp-arithmetic-a14r1n11-qualified.json)。

## Softmax 异或 4 与多行交错调度

在原 `all` 的基础上，`--xor4 max|sum|both` 将异或 4 及其算术融合为两条使用互补 bank mask 的 DPP。`--bundle max|sum|both` 合并每行四级依赖，和 `--bundle-width 2|4` 一起使用时，以其他行的 DPP 填充同一行的等待间隔。单行归约树和原 FP32 算术顺序保留。

```bash
python build-softmax-dpp-arithmetic.py "$control" "$candidate" all --xor4 both --bundle both --bundle-width 4
```

在 `a14r1n11` 测试了 12 个配置，全部在两个生产长度的普通和放大输入上逐位一致。四路交错同时覆盖最大值和求和时最快，静态 `s_nop` 由 224 条降为 8 条，动态 VALU 指令减少约 9.2%，全局读取与 LDS 指令数相同，scratch 为零。增补 20 个长度的边界测试通过。公开 A 长度相对现用 DPP 融合版节省 9.29%，八卡三步去噪中位数合计减少约 5.34%，音频、视频输出逐位一致。重建的设备镜像及主机库与实测文件逐字节相同。见 [单卡、八卡、复建和 PMC](../../docs/softmax-dpp-xor4-a14r1n11.json)。

普通官方 A 与 11 条常驻请求均通过规格、成片哈希和独立参考比对。A 保持 768×928、50 steps、124 帧、seed 42、49 次 Transformer 调用。三组正式 A 同进程配对中，端到端均值由 161.337 秒降至 153.933 秒，减少 4.589%，三组分别减少 4.785%、4.186%、4.790%。Transformer 均值由 140.733 秒降至 133.213 秒，减少 5.344%。B、C 和返回 A 的成片分别与已验收结果逐字节一致。现用本地配置指向重建产物 `attention-q128-softmax-dpp-w4-a14r1n11`。计时结论针对已测的运行环境和生产长度。见 [完整验收和默认加载复核](../../docs/softmax-dpp-xor4-a14r1n11-qualified.json)。

## QK/PV 等待调度和 FP32 打包乘法

`build-mmac-issue.py` 校验上述四行 DPP 版的固定汇编摘要，再直接重汇编 packed 镜像。等待调度覆盖完整块与尾块，可按 2、4 或 8 次 LDS 读取合并等待，也可先发射读数已就绪的 PV 指令。所有变体保留各累加器的原有 BF16 MMAC 序列。完整块的 PV 累加器有连续的缩放因子，可将 64 条标量 FP32 乘法合成 32 条 `v_pk_mul_f32`。完整块的 QK 分数也可每两个打包，缩放因子经两个已结束使用的 VGPR 传递。两项打包均保留尾块的原乘法。gfx936 工具链拒绝用 `op_sel_hi` 直接广播 QK 的缩放因子。

```bash
python build-mmac-issue.py "$candidate" "$packed" \
  --qk-wait-width 2 --pv-wait-width 4 --pack-pv-rescale --pack-qk-scale
```

重新汇编对照得到相同 packed 镜像。17 个候选均通过两个生产长度的普通与放大输入逐位校验。等待调度组合最多节省 0.89%/0.82%，PV 打包单独节省 3.46%/3.53%，QK 打包单独节省 1.12%/1.09%。最佳组合在 27570/22016 token 上节省 4.37%/4.31%。独立重建的整目录与已测候选逐文件一致，并通过另外 20 个长度的位模式和哨兵检查。独立 PMC 显示最佳组合的 VALU 指令数减少约 8%，LDS 指令、bank conflict 和全局读取指令数相同。硬件仍分配 220 VGPR、8192 B LDS、0 B scratch。按当前公开 A 的注意力占比估算，组合的端到端收益约为 2.1%，先归档以便后续叠加。本轮保持已验收默认。见 [配对计时、复建与 PMC](../../docs/mmac-issue-a14r1n11.json)。

## V 读取提前与受掩码清零

`--prefetch-v copy|direct` 将完整块的四条 V 全局读移到 QK 缩放后、softmax 最大值归约前，复用此时空闲的 v175–v182。两条路径保留原加载条件和 VMEM 等待。`copy` 在等待后搬回四对寄存器，`direct` 只搬一项以释放 P 读取所用的 v182，其余七项直接用于 V 的 LDS 打包。`--kv-zero packed|miss` 分别合并完整块中 K 和 V 的清零指令，或让加载受屏蔽时才执行这些指令。`--elide-v pack` 删去 V 打包时对同一目标、掩码和源重复执行的第二条 `v_bfi_b32`。QK、在线 softmax 和 PV 的计算顺序保留。新增三项改动仅针对 packed 镜像的完整 K/V 块。

```bash
python build-mmac-issue.py "$candidate" "$prefetched" \
  --elide-v pack --kv-zero miss --prefetch-v direct \
  --qk-wait-width 2 --pv-wait-width 4 --pack-pv-rescale --pack-qk-scale
```

11 个候选在 27570/22016 token 的普通与放大输入上逐位一致。直接预取单项节省 0.68%/0.70%，受掩码清零单项节省 1.94%/1.94%。最佳组合节省 6.24%/6.26%，复用 K 加载的比较结果会削弱收益。最佳组合独立重建到相同产物，再做 20 个长度的逐位和哨兵检查。独立 PMC 的 VALU 指令数降低 11.4%，全局读取和 LDS 指令计数持平，硬件分配 220 VGPR、8192 B LDS、0 B scratch。见 [配对计时、独立 PMC 与八卡回放](../../docs/v-prefetch-a14r1n11.json)。

在同一八卡实例回放第 1、25、49 次去噪输入，各取五次无 profiler 同步墙钟计时的中位数。三次分别节省 92.44、93.85、93.87 毫秒，中位数合计节省 3.44%，两侧音视频张量哈希和算子调用计数一致。按 49 次去噪外推为 4.58 秒，占已验收 A 稳态端到端 153.93 秒的 2.97%。本轮按端到端 3% 优先线归档该组合，正式版本沿用四行 DPP。

## K、V 联合发射

`--joint-kv-wait 0|2|4` 让完整块的两条 K 全局读与四条 V 全局读共用原来的掩码分支，先发射 K，再发射 V。V 暂存于 v198–v205，在最大值归约后等待并转入 V 打包已用的寄存器。尾块及每个累加器的 BF16 MMAC 序列保留。`vmcnt(4)` 让 QK 所需的 K 到达后即可继续，V 的等待推迟到实际转存前。

```bash
python build-mmac-issue.py "$candidate" "$joined" \
  --joint-kv-wait 4 --elide-v pack --kv-zero miss --prefetch-v direct \
  --qk-wait-width 2 --pv-wait-width 4 --pack-pv-rescale --pack-qk-scale
```

`vmcnt(4)` 在 27570/22016 token 的无 profiler 配对相对正式四行 DPP 版节省 7.63%/7.45%，相对上节预取组合节省 1.44%/1.38%。扩大 K 消费前的等待范围至 `vmcnt(2)` 或 `vmcnt(0)` 均慢于上节预取组合。各变体在两种生产长度的普通与放大输入下逐位一致，最佳镜像独立重建逐字节一致，另外 20 个长度及尾部哨兵校验通过。独立 PMC 显示，各版的 LDS 指令、全局读取和 bank conflict 计数相同，最佳版仍分配 220 VGPR、8192 B LDS、0 B scratch。见 [双对照、复建与独立 PMC](../../docs/joint-kv-a14r1n11.json)。

在同一八卡实例回放第 1、25、49 次去噪输入，每侧各取五次无 profiler 同步计时。三次中位数分别节省 110.69、112.10、112.30 毫秒，中位数合计节省 4.11%。两侧音视频张量哈希和算子调用计数一致。按 49 次调用外推，公开 A 端到端预计节省 5.47 秒（3.56%）。使用 `qualify-attention.sh RUN CODE BASE_CONFIG ACCEPTED BUILD` 依次测普通入口与同一进程的三组 A 配对、B/C 和返回 A，并复算规格、哈希与端到端收益。

普通官方 A 与 11 条常驻请求通过完整验收，A 保持 768×928、50 steps、124 帧、seed 42 和 49 次 Transformer 调用。三组 A 的稳态端到端均值由 153.717 秒降至 147.552 秒，下降 4.011%，三组降幅分别为 3.953%、4.429%、3.651%。Transformer 均值由 133.144 秒降至 127.649 秒，下降 4.127%。A、B、C 及返回 A 的成片与验收版逐字节一致。本地默认镜像为 `attention-q128-joint-kv-a14r1n11`。见 [官方及常驻验收记录](../../docs/joint-kv-a14r1n11-qualified.json)。

## P 的 LDS 高半字写入

`build-p-half-lds.py REFERENCE OUTPUT store --scope both` 从已验收联合发射镜像的固定汇编摘要开始。它保留 FP32 至 BF16 的 RNE 和 NaN 编码，把舍入值留在 VGPR 高半字，使用 `ds_write_b16_d16_hi` 写原 LDS 地址。完整块和尾块各省去 16 条右移，全部 P 读写地址和 PV 累加顺序保留。

27570/22016 token 的无 profiler 配对分别节省 1.29%/1.42%。生产长度和另外 20 个长度均逐位一致，独立构建的三份设备镜像、汇编和主机库与实测对应文件逐字节一致。独立 PMC 显示 VALU 指令减少约 3.8%，LDS、全局读取及 bank conflict 计数持平，LDS 等待计数增加约 8%–10%。本轮只归档。

低、高半字分别读到同一 VGPR 的方案、加入中间等待和交换半字的对照均出现数值差异。可用 `hipcc --offload-arch=gfx936 -x hip probe-lds-d16.cc -o probe-lds-d16` 在 DTK 26.04 和 BW1000 上复现。探针从初值 `0xaabbccdd` 分别得到 `0x00003344` 和 `0x77880000`。同一寄存器先读低半字、等待，再读高半字后为 `0x77880000`。高半字写入将原 LDS 字 `0x11223344` 更新为 `0x112299aa`。见 [配对、边界校验、探针与 PMC](../../docs/p-half-lds-a14r1n11.json)。

## 联合发射版的 P/V LDS 分区

`build-p-half-lds.py REFERENCE OUTPUT store --overlap p-first --p-pending 7` 在联合发射汇编上把 V 的 LDS 地址移到第二个 8 KiB 区域。先写 P，再写 V，共用一次线程块同步后读回。先发射 P 的读请求，在 V 发射前允许至多七条 LDS 操作未完成。PV 消费前的等待仍覆盖两个操作数。完整块与尾块的累加顺序保留。用 `unchanged` 代替 `store` 可测保留标量右移的分区版。

九个候选在 27570/22016 token 的普通与放大输入均逐位一致。单独分区时，先读 P 回退 0.8%–0.9%，先读 V 回退约 3.4%。组合高半字写入和 `lgkmcnt(7)` 后，相对正式版分别节省 1.98%/2.01%，相对单独高半字写入的 19 组配对分别节省 0.71%/0.63%。组合版独立重建逐文件一致，20 个额外长度及哨兵均通过。设备汇编的 barrier 从 12 处减到 8 处，独立 PMC 测得 LDS 从 8 KiB 增为 16 KiB，LDS 等待计数较正式版少约 8%–9%，LDS 指令、bank conflict 和 220 VGPR 保持一致。收益位于单 kernel 范围，先归档。见 [逐位校验、双对照和 PMC](../../docs/joint-pv-overlap-a14r1n11.json)。

## 联合发射版的跨块预取

`build-k-cross-block.py REFERENCE OUTPUT {backedge,after-pv,before-dpp}` 在完整块的回边、当前 PV 点积结束后或求和 DPP 前发射下一块 K。复用 PV 后的 v166–v173，下一轮从这组寄存器直接写 LDS。用 `--include-v` 同时预取 V 到 v206–v213。最后一个完整块继续按原分支进入尾块，加载条件和所有累加顺序保留。`--pair-with-lds` 叠加上一节的高半字写入、16 KiB 分区和 `lgkmcnt(7)`。

八个候选在 27570/22016 token 的普通与放大输入均逐位一致。紧接 PV 的 K 单项节省 2.14%/2.29%，K/V 同时预取的最佳位置在求和 DPP 前，节省 2.45%/2.53%。最佳组合用 `after-pv --pair-with-lds`，节省 4.02%/4.63%。该组合的独立重建产物逐字节一致，另外 20 个长度和哨兵均通过。三个独立 PMC 对照均测得 220 VGPR 和 0 B scratch，K 单项和 K/V 单项保留 8192 B LDS，最佳组合使用 16384 B LDS。对 27570 token，K/V 单项的 LDS 等待计数降低约 23%，实际耗时收益约 2.45%。本轮按单 kernel 收益归档。见 [配对、边界验证与独立 PMC](../../docs/k-cross-block-a14r1n11.json)。

## 完整块的标量掩码与地址递推

`--loop-mask` 将完整块中不随迭代变化的掩码提到循环外。packed 主机入口检查序列长度和有符号 32 位字节跨度，完整块索引小于 `floor(rows/32)`，因此块号上界和非负偏移条件在该路径成立，仍保留原 lane 掩码。`--address-induction k|v|both` 逐块递推相应的 64 位字节地址，保留进位、原内存读取与尾块。DTK 要求 64 位标量复制的源寄存器对齐，基址初始化因此采用两条 32 位复制。

```bash
python build-k-cross-block.py "$candidate" "$induction" after-pv --pair-with-lds --address-induction k
```

十个候选在 27570/22016 token 均通过普通和放大输入的逐位校验。仅递推 K 地址叠加 K 预取和 LDS 分区后，相对正式版节省 4.88%/5.09%，19 组复测相对上一轮组合再省 0.87%/0.47%。双地址递推相对正式版节省 4.56%/5.17%，说明最佳选择随长度变化。掩码外提在本轮对照中回退。K 单独递推还通过另外 20 个长度和尾部哨兵检查，重建产物逐文件一致。独立 PMC 测得 220 VGPR、16 KiB LDS、0 B scratch，LDS 等待计数相对上一轮最快组合减少约 11%。按单 kernel 收益归档。见 [配对计时与独立 PMC](../../docs/k-loop-scalar-a14r1n11.json)。

## 在线 softmax 的逐行最大值更新

`build-softmax-schedule.py REFERENCE OUTPUT` 在固定的联合发射版和 K 地址递推组合上重排逐行指令，保留 RAW、WAR、WAW 与 `vcc` 依赖。五种纯重排逐位一致，叠加后没有额外收益。

`--fold-max only` 折叠每行更新之前的两个 `max(x,x)`，完整块和尾块各减少 16 条指令。该变换锁定两个输入镜像的汇编摘要，两侧操作数源于前级最大值或初始负无穷。DTK 26.04 的 BW1000 探针覆盖 324 对取值，其中 80 对出现差异，均涉及 signaling NaN。排除 signaling NaN 的 225 对逐位一致，两种 max 生产者在探针输入上均未输出 signaling NaN。用 `hipcc --offload-arch=gfx936 -x hip probe-max-self.cc -o probe-max-self` 可复现。

```bash
python build-softmax-schedule.py "$induction" "$folded" --fold-max only
python screen-softmax-special.py "$candidate" "$folded"
```

折叠直接应用到正式联合发射版时节省 1.35%/1.45%，叠加 K 地址递推和 LDS 组合后节省 6.90%/7.08%，19 组对上一轮组合的配对分别再省 2.10%/2.11%。最佳组合的独立重建逐文件一致，额外 20 个长度、尾部哨兵和 24 组特殊 BF16 输入逐位一致。独立 PMC 测得 220 VGPR、16 KiB LDS、0 B scratch。见 [配对、特殊值探针与八卡回放](../../docs/softmax-schedule-a14r1n11.json)。

普通官方 A 和包含 B、C、返回 A 的 11 条常驻请求均通过规格与独立质量检查，成片与各自已验收结果逐字节一致。三组 A 同进程配对的端到端均值由 147.507 秒降至 143.627 秒，节省 2.630%，单组降幅为 2.890%、2.951%、2.049%。Transformer 均值由 127.706 秒降至 123.079 秒，节省 3.623%。端到端低于 3% 门槛，归档候选。见 [官方入口和常驻验收](../../docs/softmax-schedule-a14r1n11-qualified.json)。

## 在线求和回写与 DPP 交错

`build-softmax-schedule.py REFERENCE OUTPUT --row-sum direct` 让完整块的八条 FMA 直接回写下一轮使用的求和寄存器。普通回边的八条搬运移到完整块退出边，尾块入口保留原寄存器布局。`--row-sum interleave` 把前四行的 FMA 放到两组 DPP 中间，并省去第二组原有的 `s_nop 1`。`interleave-nop` 保留这条等待作为对照。每行的归约树及 FMA 舍入顺序保留。

```bash
python build-softmax-schedule.py "$folded" "$interleaved" --row-sum interleave
python screen-softmax-special.py "$candidate" "$interleaved"
```

四个候选在 27570/22016 token 的普通和放大输入均逐位一致。最快组合相对正式联合发射版节省 8.47%/8.59%，19 组配对相对最大值折叠版再省 1.70%/1.67%。重建产物逐文件一致，20 个额外长度、尾部哨兵和 24 个特殊 BF16 输入通过。独立 PMC 测得 220 VGPR、16 KiB LDS、0 B scratch。见 [配对、特殊值和八卡回放](../../docs/softmax-row-sum-a14r1n11.json)。

从完整推理的标准 Q128 packed 构建重建候选时，为 `rebuild-v2.sh` 设置 `H3_AOT_SAVE_IR=1` 并启用 `--long-attention-q128-packed`。令 `build` 指向该构建目录，`release` 为新的输出目录。生成的 `candidate.env` 可通过 `run-optimized.sh --config` 加载。

```bash
bash build-row-sum-release.sh "$build/attention" "$build/candidate.env" "$release"
```

在 `a14r1n11` 的三组八卡 A 配对中，端到端均值由 147.80 秒降至 142.89 秒，节省 3.32%。Transformer 均值由 127.67 秒降至 121.97 秒。普通官方 A 与常驻 A、B 已完成的成片均和联合发射版逐字节一致。该轮计时全部来自同一个常驻进程，C 和返回 A 回归待完成。 [已完成的端到端配对与成片检查](../../docs/softmax-row-sum-a14r1n11-steady-partial.json)

## 独立 K 分区与 P 写入流水

`build-disjoint-k-lds.py` 在求和交错镜像的 16 KiB 基础上为 K 增加 8 KiB。K 分区的读写使用原偏移加 16384 B，保留每次发布后的读取顺序。独立分区允许分别移除覆盖 K 前及覆盖 P 前的线程块同步。用 `--allocation-only` 保持设备指令一致，只调整发射时的 LDS 分配。

`build-p-lds-pipeline.py` 保持 16 KiB。它将完整块的 P 写入按寄存器最后一次赋值提前，保持原 LDS 地址和写入顺序。可独立测量同步前移、每次发射一条或四条、移除已有屏障后的等待。用 `--k-in-v-region` 比较 K/V 共用分区、P 独立的布局，P 可在 K/V 覆盖同步之前发射。

```bash
ref="$release/attention"
mkdir "$builds"
python build-disjoint-k-lds.py "$ref" "$builds/drop-both" --remove-k-fence --remove-p-fence
python build-p-lds-pipeline.py "$ref" "$builds/early-four" --move-p-fence --store-batch 4
python build-p-lds-pipeline.py "$ref" "$builds/alias-pending" \
  --k-in-v-region --store-batch 4 --keep-p-pending
python screen-pv-wide-operand.py "$ref" "$builds" "$results"
```

在 `f12r2n09` 的 27570/22016 token 上，13 个布局和时序候选均通过正常输入、放大输入及输出哨兵校验。独立 K 分区同时移除两处同步后，相对只换布局快约 2.2%–2.6%，相对 16 KiB 基准仍回退 10.01%/9.40%。机器码一致的 24 KiB 分配对照回退 12.57%/12.36%，采样的平均活跃 wave 从约 11.32/10.59 降至 7.75/7.30 每 CU。独立 PMC 中，三个分区变体采样的 LDS 指令、bank conflict、全局读取指令数相同。移除同步降低 LDS 等待计数约 30%，但额外分配的代价更大。移除两处同步的镜像通过另外 20 个长度与 24 组特殊 BF16 输入。

16 KiB 组的 `early-four` 在 19 组复测中节省 0.67%/0.66%，对应独立 PMC 的 LDS 等待计数下降约 10%。K/V 复用分区的最快组合节省 0.39%/0.35%。代表性构建逐文件复建一致。本轮收益低于进入八卡验证的优先线，保留实验版本和原默认。 [配对、边界及 PMC 记录](../../docs/kp-lds-f12r2n09.json)

## 尾块 P 寄存器复用与分配量对照

`build-tail-p-vgpr.py REFERENCE OUTPUT` 将尾块唯一使用的 `v216` 替换为同一基本块内随后会覆盖的 `v109`，保持 LDS 读取、打包、MMAC 及尾部掩码顺序。用 `--reserve 224|232|256` 保留基准设备指令，仅改变 VGPR 分配描述。四种候选仍使用 16 KiB LDS。

四种候选在 27570 和 22016 token 上均通过普通和放大输入逐位校验，11 组无 profiler 配对差异均小于 0.04%。三种分配量对照的 `.text` 与基准一致。独立 PMC 确认基准、尾块复用、256 分配对照分别使用 220、216、256 VGPR，均为 0 B scratch。两个长度的平均活跃 wave 约为 11.32 和 10.59 每 CU，在三个对照间基本持平。尾块复用另通过 20 个长度及 24 组特殊 BF16 输入。候选及对照重建产物逐文件一致。本轮只归档，在现有 16 KiB 的分配条件下没有观察到仅调整 VGPR 数可带来性能收益。 [配对、边界及 PMC 记录](../../docs/tail-p-vgpr-f12r2n09.json)

对双 Q128 共享 K/V 的冻结 LLVM IR 做独立后端探针。保持 512 线程组，分别增加 `amdgpu-num-vgpr=216`、`amdgpu-num-vgpr=256`，或声明统一线程组。两种 VGPR 指定的汇编与历史对照逐字节一致，三个改动都仍使用 128 VGPR 和 320 B scratch。本轮保留 CPU 编译结果，后续若继续双 Q128，需要改变活跃值或线程分布。

## V 等待位置与在线 softmax 交错

`build-v-wait-pipeline.py` 将完整块 V 的寄存器改到下一次 P 的 LDS 读取前尚未使用的区间，删除五条搬运。分别测最大值归约后、P 覆盖同步点、V 打包开始前三个等待位置。`--preload-next` 在发射下一块 K 时一并发射 V，保留完整块结束条件和受掩码清零。地址递推、P 写入前移和 QK 交错各有独立配对。

`build-qk-online-order.py` 在两组最大值 DPP 之间调度 QK 缩放和第一组在线指数，保持原舍入和各行的归约顺序。`build-pv-online-sum.py` 将四项、八项求和或第一组求和 DPP 移到最后一批 PV MMAC 附近，每条 MMAC 累加链保持原顺序。

```bash
ref="$release/attention"
mkdir "$builds"
python build-v-wait-pipeline.py "$ref" "$builds/fence" --wait-at p-fence --register-base 208
python build-qk-online-order.py "$ref" "$builds/qk" --split-scale --drop-second-nop --online interleaved
python build-pv-online-sum.py "$ref" "$builds/pv" dpp-split
python screen-pv-wide-operand.py "$ref" "$builds" "$results"
```

四组 19 个变体在 27570 和 22016 token 的普通输入、放大输入和输出哨兵检查中均通过。V 在 P 同步点等待的两个寄存器布局用 19 组复测，分别节省约 1.19%–1.20% 和 1.12%–1.13%。提到最大值归约后即等待、推到 V 打包才等待，以及下一块 K/V 联合发射均无正收益。P 前移或 QK 交错叠加到 V 版本后，收益小于单独延后等待。

单独 QK 交错最好约 0.10%/0.08%。PV 提前处理求和回退 0.39%–2.72%。独立 PMC 保留在原始 TXT，三种代表候选均为 220 VGPR、16 KiB LDS、0 B scratch。V 版本在 27570 token 的 VALU 指令数约下降 1.30%，VMEM 读取指令数持平。四个代表镜像逐文件复建一致，分别通过 20 个额外长度和哨兵检查，三个代表镜像各通过 24 组特殊 BF16 输入。本轮验证范围为单卡。 [配对、逐位与 PMC 记录](../../docs/online-pipeline-f12r2n09.json)

## 完整 K 块的全局内存直达 LDS

DTK 26.04 和 gfx936 上，`probe-global-to-lds.cc` 验证每个 lane 的四个 dword 按 lane 顺序写入 LDS。对 32×128 BF16 K tile 使用 896 元素的 packed 行步长，通过 `tid ^ (tid >> 4)` 选择源 lane，继续使用原 QK 的 LDS 地址。分别用 `buffer_load_dwordx4 ... lds` 与 `global_load_dwordx4 ... lds` 传输，两者的 2048 个 dword 均与原 swizzle 一致。

```bash
hipcc --offload-arch=gfx936 -O2 -std=c++17 -x hip probe-global-to-lds.cc -o "$probe"
"$probe" k-tile-buffer
"$probe" k-tile-global
python build-k-direct-lds.py "$release/attention" "$builds/early-v" --prefetch --v-wait-fence
```

`build-k-direct-lds.py` 保留原 K 读取地址、MMAC 顺序、LDS 分配和尾部路径。完整块正常路径以直写代替暂存 VGPR 及两条显式 LDS 写入，受掩码路径在原地址写零。默认在原覆盖同步点后发射，`--prefetch` 将覆盖同步与发射提到循环回边。四个变体对两个生产长度均逐位一致。首轮配对中，消费前才发射的两个版本回退 3.42%–3.82%，跨块预取的两个版本收益为 0.30%–0.66%。

跨块版叠加 V 延后等待后，19 组复测相对求和交错基准节省 0.38%/0.70%，相对只做 V 延后等待回退 0.86%/0.46%。独立 PMC 中，直写的两个代表镜像均减少约 3.84% 的 LDS 指令，bank conflict 与 VMEM 读取指令数持平。组合版通过 20 个额外长度及 24 组特殊 BF16 输入，两个代表镜像逐文件复建一致。 [设备探针、配对、逐位校验及独立 PMC](../../docs/k-direct-lds-f12r2n09.json)

## V 的位模式打包与双读取

原 V LDS 每个 dword 存储同一特征列、相邻两行的 BF16，两个全局元素相隔 2688 个 BF16。`probe-v-direct-lds.cc` 在 gfx936 上验证 `buffer_load_ushort ... lds` 会零扩展成整个 dword，并将 `m0+2` 的目标对齐到原 dword。两次半字读取会覆盖前一次值，32×128 V tile 的原布局拼接产生 2048 个 dword 差异。该探针按目标拼接的差异返回 1，保留完整输出。其结论覆盖本轮测试的指令和寻址方式。

`build-v-pack-permute.py` 直接用八条 `v_perm_b32` 构造原 LDS 数据，保留全局读取与共享内存地址。完整块少七条位运算，另有三种版本比较 V 等待位置。原等待版在 27570/22016 token 的 19 组配对中节省 1.41%/1.35%，相对上一轮只延后 V 等待的版本再省 0.20%/0.22%。叠加延后等待后，相对求和交错基准节省 1.29%/1.27%。

`build-v-two-loads.py` 改为每线程两行、每行八列，使用两条 `global_load_dwordx4`。K 消费前的等待同步改为 `vmcnt(2)`，与其后的两条 V 请求对应。首版沿用 `vmcnt(4)` 时出现逐位差异，原始失败记录归档。修正等待计数后的四个镜像均通过两个长度逐位检查。最快组合在 19 组配对中回退 1.73%/1.95%。独立 PMC 测得 VMEM 读取指令数减少约 33.3%，LDS bank conflict 计数增加约 79.9%，LDS 等待计数增加约 33%。

```bash
python build-v-pack-permute.py "$release/attention" "$builds/compact"
python build-v-two-loads.py "$release/attention" "$builds/two-early" --wait-at early
python verify-v-pack-layout.py "$release/attention" "$builds/compact" "$builds/two-early"
python screen-pv-wide-operand.py "$release/attention" "$builds" "$results"
```

`verify-v-pack-layout.py` 解释实际打包汇编，分别使用元素索引和随机 16 位载荷，检查全部 4096 个 BF16 的最终 LDS 地址。基准与八个候选均通过。三个代表镜像各通过 20 个额外长度、24 组特殊 BF16 输入和逐文件重建校验。原等待的八置换版 VALU 指令约减少 1.83%，LDS、VMEM 读取与 bank conflict 计数持平。本轮按单卡结果归档。 [布局、设备探针、配对及独立 PMC](../../docs/v-supply-f12r2n09.json)

## P 的配对 dword 读取

`build-p-read2.py` 将完整块 16 条 `ds_read_u16` 替换为 8 条 `ds_read2_b32`。读取基址对齐到 dword，按 lane 选择低半字或高半字，再用原字节置换构造 PV 操作数。两组 4 KiB 偏移分别使用独立地址，满足 read2 的 8 位偏移限制。置换按源值生命周期调度，避免新寄存器分配覆盖尚未消费的数据。用 `--p-pending 3|7` 对比 V 读回之前的等待。

两个版本在 27570/22016 token 均逐位一致。等待三条的版本经 19 组配对，回退 1.12%/0.74%。独立 PMC 显示，LDS 指令数下降约 15.4%，bank conflict 与 VMEM 读取计数持平，VALU 指令数增加约 0.78%，LDS 等待计数下降约 3%–4%。硬件仍分配 220 VGPR、16 KiB LDS、0 B scratch。代表镜像通过 20 个额外长度、24 组特殊 BF16 输入和逐文件重建校验。本轮归档。 [配对、逐位及独立 PMC](../../docs/p-read2-f12r2n09.json)

## V 预排布与 Q tile 复用

`v_prepack_kernel.py` 每次注意力调用重新整理 V 的完整 32-token 块，输出按 head、K/V 块、原 LDS 位布局连续存放。所有 Q tile 读取同一份整理结果。尾块继续从原 V 读取。预处理只有 BF16 数据搬运，完整点积、在线归约及 MMAC 累加次序沿用求和交错镜像。

`build-v-prepack-aot.py` 把九齿预处理编译为 AOT。`build-prepacked-v.py` 让完整块直接读取预排布数据到 V 的 LDS 区域。实验借用原保留指针参数的 72 B 位置传递工作区，需要 `prepacked_attention_host.cc` 的组合发射入口。`load_attention` 会在装载该实验镜像前提示工作区依赖。`screen-prepacked-v.py --prepack` 由同一个主机调用连续发射预处理和注意力，每次重新计算 V。小于 32 token 或行步长不匹配时使用原注意力。工作区按设备、长度和 stream 分开持有。

```bash
mkdir "$builds"
python build-v-prepack-aot.py "$prepack" --tile 2048 --warps 8
python build-prepacked-v.py "$release/attention" "$builds/after-k-write" --position after-k-write
python screen-prepacked-v.py "$release/attention" "$builds" "$results" --prepack "$prepack"
hipprof --pmc --pmc-type 2 --kernel-name dense_attention_aot_kernel -o "$pmc" \
  python profile-prepacked-v.py "$release/attention" "$builds/after-k-write" "$prepack"
```

在 `f12r2n09`，六个 V 发射位置和六组九齿预处理分块均做无 profiler 对照，两个生产长度的普通及放大输入逐位一致。最新 19 组配对中，27570 token 从 24.9977 ms 降为 23.5625 ms，22016 token 从 16.8874 ms 降为 15.9482 ms，组合耗时分别节省 5.73%/5.54%。预处理单独约 0.237/0.189 ms，工作区分别为 49,373,184/39,452,672 B。前期 Python 发射预处理的计时记录单独保留。

主注意力的独立 PMC 测得 bank conflict 计数下降约 59.9%–60.0%，VMEM 读取指令下降约 33.3%，LDS 指令下降约 3.84%，仍分配 220 VGPR、16 KiB LDS、0 B scratch。预处理为 24 VGPR、0 B LDS、0 B scratch，无 LDS bank conflict。[主注意力原始 PMC TXT](../../docs/prepacked-v-f12r2n09-attention-pmc.txt) · [预处理原始 PMC TXT](../../docs/prepacked-v-f12r2n09-prepack-pmc.txt) · [基准原始 PMC TXT](../../docs/prepacked-v-f12r2n09-reference-pmc.txt)

预排布位模式、20 个额外长度、24 组特殊 BF16 输入、连续更新 V 和不同步长回退均通过。主注意力及主机库独立重建逐字节一致，预处理重建的设备指令一致。实验首版误把 Q tile 索引用于 head 寻址，修正为原 head 偏移后开始有效计时，失败记录保留。按历史 A 的每卡 2450 次调用粗估，新增节省约 3.5 s，约占历史 142.89 s 的 2.46%。该估算跨节点，实际八卡端到端尚待验证。本轮按低于 3% 的优先线归档。 [完整配对、位模式校验及 PMC 记录](../../docs/prepacked-v-f12r2n09.json)

## P 行布局与 V 预排布组合

已有通过 `build-row-sum-release.sh` 构建的求和交错发布目录时，可使用下面的固定配置构建入口。它核对两份基准设备镜像，构建九齿 V 位复制、P/V 组合及正式主机 ABI，输出可直接传给 `run-optimized.sh --config` 的 `candidate.env`，并记录构建源码和全部设备/主机产物摘要。

```bash
bash build-p-row-prepacked-release.sh "$row_sum_release" "$new_release"
bash "$H3_AUTOTUNE_CODE_ROOT/optimization/benchmark-public-a.sh" \
  "$new_release" "$model" "$new_benchmark_output"
```

该入口已在 `HIP_VISIBLE_DEVICES=` 的 CPU 构建环境跑通，六个产物的可加载 ELF 区段均与本轮验收镜像相同，其中五个文件逐字节相同，V 预处理文件仅非加载内容不同。`release.json` 的 `qualified=false` 表示重建入口不会替代新环境的实际规格、质量和计时检查。

统一 `benchmark-public-a.sh` 根据发布元数据选择求和交错基准和新组合，仍执行同一八卡常驻进程中的两次预热和三组 A 配对。汇总时同时核对主注意力、V 预排布设备指令、主机库摘要、回退镜像及运行快照；报告中的 `comparison` 区分两组已冻结的对照组合。`summarize-public-a.py report` 也接受已完整结束的 A×8/B/C/A 资格验收目录，只统计前八条 A，明确保留原完整序列，便于直接复用本轮证据。

`build-p-row-lds.py` 从已冻结的求和交错汇编重建完整 KV 循环的 P LDS 映射。构建前枚举 256 线程的 4096 个 BF16 写入位置，验证新布局为双射，并逐项核对 PV 的八个打包寄存器。保留 16 次原高 BF16 写入，将 16 次半字读取和 8 条置换换为 4 次 64 位读取。QK、在线 softmax、PV 的浮点算术、舍入和累加顺序不变，尾 KV 块保持原实现。

`build-prepacked-v.py --p-row-layout` 将两种供数方式组合。以下命令中的 `reference` 为求和交错正式三镜像目录，`prepack` 由上一节的九齿构建命令生成。所有输出路径需要尚不存在。

```bash
python build-p-row-lds.py "$reference" "$plain"
python build-prepacked-v.py "$reference" "$combined" --position after-k-write --p-row-layout
python build-prepacked-dispatch.py "$reference" "$combined" "$prepack" "$integrated"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python verify-prepacked-dispatch.py \
  "$reference" "$integrated" "$native_results"
hipprof --pmc --kernel-name dense_attention_aot_kernel -o "$pmc" \
  python profile-prepacked-v.py "$reference" "$combined" "$prepack"
```

正式主机 ABI 的组合入口保持原来的三个分派镜像作为回退，针对不小于 4096 的 packed 输入发射 V 位复制和新的长注意力。工作区按 device/stream 管理，仅复用分配空间，每次重新计算 V；其余步长及短长度走原三镜像。动态长度增大时按需重新分配，缩小时复用空间。`aot_runtime.load_attention` 验证新主机库、预处理镜像和回退镜像摘要，尚未集成正式主机的实验镜像仍被拒绝。

`f12r2n09` 的 19 组无 profiler 配对中，单改 P 节省约 3.58% 单算子耗时；组合的正式主机调用在 27570 token 下由 24.9961 降为 22.9014 ms，在 22016 token 下由 16.8878 降为 15.4809 ms，配对收益约 8.39%/8.33%，均包含 V 预排布。组合与上一轮仅预排布 V 相比，实验主机调用再节省约 2.69%/2.91%。

实验入口的 53 条记录覆盖三组 19 对计时、20 个边界长度、零值/放大输入/零 scale 和连续更新 V；正式主机的 40 条记录覆盖生产长度配对、八张卡、短长度回退、Q/K/V/O 步长回退、双 stream 及工作区增长/缩小。数值逐位比较、有限值和输出尾部哨兵全部通过。独立 `hipprof --pmc` 原始 TXT 的 `shared memory bank conflict` 为 27570 token 约 0.0178%、22016 token 0%，分配资源仍为 220 VGPR、16 KiB LDS、零 scratch。

真实八卡第 1/25/49 步各做五次正式计时，中位数分别由 2.489466/2.488783/2.488489 s 降至 2.384881/2.384666/2.384024 s，三处中位数合计节省约 4.19%，音视频张量逐位相同、算子调用数相同。普通官方默认入口 A 已通过规格及独立参考比较，完整执行 49 次 Transformer，MP4 与已验收版逐字节一致，SSIM 93.832013%。本节点首次进程墙钟 883.30 s，包含约 630.32 s 模型创建加载；生成管线 198.561 s，Transformer 119.350 s，单次启动与稳态计时分别保留。

11 请求完整验收已经结束，三组正式 A 请求由 142.320005±0.646517 s 降至 137.344178±0.934113 s，耗时下降 3.496225%；Transformer 由 121.994201 降至 116.859849 s，下降 4.208686%。三个配对耗时下降分别为 4.189987%、2.727497%、3.568075%。所有 A/B/C/返回 A 成片均与对应已验收视频逐字节相同，独立参考 SSIM 为 93.832013%/99.404123%/99.630047%。本地正式配置已切换到该组合，旧配置有备份，新增产物复制后逐字节核对。整卡显存五秒采样最高读数 31960 MiB；同进程保留双方分配，不作为独立基准显存差的证据。 [统一计时结果](../../docs/p-row-prepacked-f12r2n09-public-a.json) · [完整验收](../../docs/p-row-prepacked-f12r2n09-qualified.json) · [配对与构建记录](../../docs/p-row-prepacked-f12r2n09.json) · [原始 PMC TXT](../../docs/p-row-prepacked-f12r2n09-basic-pmc.txt)

## MMAC 依赖诊断

新组合的完整 QK 段含 32 条 BF16 MMAC，分布在 4 组累加器，每组按原序更新 8 次；PV 段同为 32 条，分布在 16 组累加器，每组更新 2 次。静态差异不足以证明 QK 存在可利用的依赖停顿，需要先验证现有 4 组是否已足够隐藏实际指令延迟。

`build-mmac-dependency-probe.py` 从已验证组合的内核描述符构建独立诊断镜像，固定线程块、VGPR/SGPR 分配和动态 LDS，逐一使用 1/2/4/8/16 组累加链。每轮执行相同的 64 条原生 BF16 MMAC，矩阵输入固定为 1；循环结束后将结果写回，测量程序会先检查全部线程的精确结果再计时。镜像不包含真实 QK/softmax/PV，不能用于替换模型注意力。

```bash
python build-mmac-dependency-probe.py "$integrated" "$probe" --iterations 2048
# 等完整模型性能验收退出后，另行运行以下命令。
HIP_VISIBLE_DEVICES=0 python bench-mmac-dependency-probe.py "$probe" "$probe_result"
```

初版初始化后立即使用 MMAC，在独立进程中出现期望 2097152、实际部分为 2097144 的反例。补入初始化后的 `s_nop15`，保留相同的循环、运算数、寄存器和 LDS，三次独立进程覆盖 80/320 两档线程块，所有 660 次逐次输出校验通过。数值检查位于计时事件之外，初版失败数据保留。

每个进程在 80 个线程块下，1/2/4/8/16 链均约 0.771 ms；320 个线程块下均约 2.865 ms。单靠增加独立累加链未展示可靠吞吐收益，本轮归档。此为不含访存、softmax 的裸 MMAC 循环，不能推导完整注意力不存在其他供数或依赖瓶颈。 [初始化反例、全部逐次校验及计时](../../docs/mmac-dependency-f12r2n09.json)

## Q 常驻 VGPR 位置

`build-query-register-placement.py` 在 P 行布局与 V 预排布组合上，将原 `v65:v96` 的 Q 常驻寄存器放到 VGPR 文件末端，并将原 `v97:v218` 紧凑移到 `v65:v186`。提供 187/188/189/190 四个 Q 起始位置，检查操作数寄存器位置是否影响 MMAC。

所有寄存器映射均为双射，硬件入口的 `v0` 不变。构建脚本逐一确认所有多寄存器操作数仍连续，并用逆映射恢复整份原汇编逐字节比较，再更新 VGPR 资源描述。四个版本均保留 128 条静态 BF16 MMAC，数学指令、浮点顺序及其余控制流不变，没有插入运行时拷贝。VGPR 静态需求分别为 219/220/221/222，实际分配粒度和性能需要运行后确认。

```bash
python build-query-register-placement.py "$integrated" "$register_candidate" --query-register 188
python screen-pv-wide-operand.py "$integrated" "$register_builds" "$register_result"
```

四个版本均通过两个生产长度的普通/放大输入逐位比较、有限值和输出哨兵检查。11 组无 profiler 配对的变化在约 ±0.02% 范围，没有可靠收益。独立 PMC 中 q188 分配 220 VGPR，q190 为 224 VGPR，均为 64 SGPR、16 KiB LDS、零 scratch，bank conflict 保持约 0.0178%/0%。本轮归档，未进入新的整模型复跑。 [计时与原始 PMC](../../docs/query-register-placement-f12r2n09.json)

## 内部 P 的 NaN payload 与 xor4 归约

在已验收的 137.344178 秒组合上，`build-p-nan-payload.py` 删除完整 KV 块内部 P 格式化的 16 条有序比较、16 条 NaN 掩码和一条常量。所有 FP32 数学、RNE 舍入、MMAC、尾块和最终输出 BF16 NaN 规范化保留；16 个原始 FP32 概率继续进入在线分母。脚本检查 VCC 读者与下一次完整定义、临时寄存器覆盖和最终代码不变。有限概率的舍入结果相同，非有限输入由不变的分母及最终规范化处理；覆盖异常输入的最终输出逐位测试仍是必要门槛，不以通常模型输入有限作为前提。

`build-softmax-ror4.py` 利用前一级 xor8 归约的对称性，把完整块 xor4 的两条 bank-mask DPP 改成一条 `row_ror:4`，删除 8 条 max 和 8 条 sum 指令。CPU 枚举 64 个 lane 的来源，逐项检查前驱、读写依赖及尾部不变。有限 FP32 操作数保持原有加法层级；NaN payload 和带符号零仍另做最终输出逐位比较。

```bash
python build-p-nan-payload.py "$integrated" "$nan_only"
python build-softmax-ror4.py "$integrated" "$ror_only"
python build-softmax-ror4.py "$nan_only" "$combined"
python verify-p-nan-payload.py "$integrated" "$combined" "$verification_jsonl"
```

三个版本分别通过 95 个边界、普通/零值/溢出/非有限输入测试，以及两个生产长度的 19 组无 profiler 配对。长度小于 4096 时调用原回退，不能计作新 shader 的覆盖。27570/22016 token 的单项收益分别为：NaN payload 6.51%/6.40%，ROR4 2.78%/2.76%；两项单独预计均不足 3% 完整请求收益，单项不选版。组合由 22.9032/15.4837 ms 降至 20.8552/14.1163 ms，下降 8.947%/8.829%。首次内联扩展测试的 36 条记录覆盖两种符号的全部 254 个 BF16 NaN payload、带符号零和最大有限 Q/K，输出逐位相同。独立 `verify-p-nan-payload-edges.py` 已在完整验收退出后执行，36 条结果及尾部保护全部通过。

真实八卡第 1/25/49 步的音视频输出与本轮正式基准逐位一致、调用数相同，各五次均值为 2.385210→2.283258、2.384882→2.282501、2.384840→2.281983 s，下降 4.274%/4.293%/4.313%。历史 capture 来自早期版本，本轮两侧均与其不同；逐位结论仅指本轮配对双方。独立重建的六个产物及汇编与已测试候选逐字节一致。独立 PMC 保持 220 VGPR、64 SGPR、16 KiB LDS、零 scratch，27570 token 的 bank conflict 约 0.01955%，22016 为 0%。PMC 的 `processed ALU instructions` 不是芯片峰值算力利用率，不能据此反推计算峰值。

预计 49 次 Transformer 可节省约 5.017 秒，即约 3.65% 先前完整请求墙钟；这是回放投影，不是新的端到端实测。普通官方默认入口 A 及 11 条连续请求均已通过。三组完整 A 请求由 137.001043±0.366754 s 降为 132.824526±0.905221 s，下降 3.048529%；Transformer 116.891928→111.878162 s，下降 4.289232%。三组分别下降 3.329599%/2.557911%/3.260003%，中位数 3.260003%，全部为正；八条 A 和 B/C/返回 A 均与对应已验收 MP4 逐字节一致，SSIM 为 93.832013%/99.404123%/99.630047%。新进程墙钟 880.90 s、模型组件创建加载约 635.48 s、首次生成管线 191.064 s、Transformer 114.281 s，单列而不混作稳态成绩。正式配置已切换到该组合，旧配置已备份。五秒显存采样最高读数 33784 MiB，未覆盖前面的加载且可能漏瞬时峰值，混合实例不作独立显存差依据。一次误用单卡可见环境的回放在加载 Transformer 前失败，原始失败日志保留；改为显式八卡后上述真实回放成功，不归因于硬件故障。 [逐位校验、无 profiler 配对、独立重建与真实八卡记录](../../docs/p-nan-ror4-f12r2n09.json) · [原始 PMC TXT](../../docs/p-nan-ror4-f12r2n09-basic-pmc.txt)

## 精简 softmax 候选的完整发布重建

`build-softmax-compact-release.sh` 从上一节已冻结的 P 行布局 + V 预排布发布目录，依次执行内部 P payload 与 ROR4 变换，生成完整主机 ABI 的候选目录及配置。其余五个产物直接保留，短长度、其他步长和预排布回退均不变。该版本已完成完整验收并选为正式配置；重建操作本身仍不替代新环境的质量与性能验收。

```bash
bash build-softmax-compact-release.sh "$current_prepacked_release" "$softmax_release"
bash "$code/optimization/benchmark-public-a.sh" \
  "$softmax_release" "$model" "$new_benchmark_output"
```

计时仍是双方各预热一次后的三组完整 A 配对，`protocol` 和 `wall.candidate.mean` 的定义不变；`comparison` 为 `p-row-prepacked-v-vs-softmax-compact`，基准是此前约 137.344 秒的组合。当前 `reproduce-public-a.sh build` 已串接本节发布重建；其 `build-work/prepacked` 保留约 137 秒基准，`build-work/release` 是本轮约 132.825 秒正式版本。

CPU 独立发布重建中，六个产物的可加载区段及映射均与实际八卡回放/普通官方 A 所用候选相同，五个逐字节相同，V 预排布镜像只在调试信息上不同。`verify-softmax-release-rebuild.py` 同时核对上游发布清单、七份构建相关源码及六个产物摘要；传入旧序列和旧 JSON 时，按 JSON 协议核对旧计时全部字段。新的汇总程序实际重算 137.344 秒结果后全字段相等，三阶段的冻结镜像检查通过，错误基准、未冻结的后续 shader 及未完成运行均被拒绝。基准快照里的构建 JSON 是内嵌数据，按结构比较，其余产物按 SHA-256 比较。 [重建及计时协议回归记录](../../docs/softmax-compact-release-f12r2n09.json)

## 稳定最大值时的恒等重缩放，单项收益不足 3% 归档

`build-stable-max-rescale.py` 已在 CPU 上构建三种对照，均以本轮待验收的 NaN payload + ROR4 组合为直接参考：`deferred` 只延后 8 条 alpha exp，`guard-scale` 加 wave 一致判断但保留 exp，只省去恒等缩放，`guarded` 同时省去 exp 与恒等缩放。这样区分 SFU 调度变化、条件分支代价和省去实际指令的收益；准备阶段没有 GPU 结论，实际结果见本节末尾。

判定使用 8 个原始 `old_max-new_max` 的 FP32 位模式，按位 OR 并忽略符号位，只有 wave 全部活跃 lane 的八项都是 ±0 时才走快路径。NaN、Inf、最小非零 FP32 均走原路径，不按输入类型或样例分支。快速路径为后续原有分母 FMA 精确写入 alpha=1；MMAC、加法层级、尾块、barrier 与输出格式化保持原样。原描述符为 FP32 denorm mode 3、IEEE mode 1；`v206` 只作为本段死寄存器，后续 tail 先完整重定义，VCC 和 SCC 也检查后续原定义。其收益取决于真实数据中 max 稳定的比例以及分支开销，需要实测，不能由指令数直接报成绩。

```bash
for mode in deferred guard-scale guarded; do
  python build-stable-max-rescale.py "$combined" "$guard_builds/$mode" --mode "$mode"
done
# 下列GPU操作必须等完整八卡验收实际退出后串行执行。
python verify-stable-max-values.py "$combined" "$guard_builds" "$value_edges"
```

冻结脚本的静态断言通过，生成汇编与三份 CPU 构建逐字节一致，8192 组整数谓词覆盖通过。`verify-stable-max-values.py` 另准备 90 条专用检查，用零 Q/K 强制从第二完整块起进入稳定 max，覆盖最小 BF16 subnormal、极小正规值、抵消、带符号零、最大量级和非有限 V；准备阶段尚未执行这些 GPU 检查。原 `run-stable-max-followup.sh` 等待完整验收进程实际退出并核对源码后才启动，12 条专用记录通过后遇到后述输入构造错误。修正后的独立重试扩为 126 条，随后完成通用校验及 19 组配对，全程未与完整八卡测速重叠。 [准备记录](../../docs/stable-max-rescale-f12r2n09-preparation.json)

稳定 max 的后续 GPU 测试已完成。首次脚本在 12 条结果已通过后，将 Python 大整数 `2**120` 交给 BF16 `fill_` 时发生类型转换溢出，未进入该用例的 kernel 比较；原始失败和源码快照保留。修正为显式浮点数，并加入加权极小 V 后，126 条专用最终 BF16 位模式及尾部保护检查、每变体 95 条通用检查均通过。部分加权极小 V 的参考输出为零，记录保留非零和 BF16 subnormal 数量，不宣称观测到非零内部 FP32 subnormal。

19 组无 profiler 配对中，`deferred` 在 27570/22016 token 节省 0.705%/0.638%，`guard-scale` 为 3.341%/3.114%，`guarded` 为 6.776%/6.264%。最佳主长度由 20.856625 降至 19.442707 ms；按每次完整 A 的 2450 次长注意力调用，仅投影约 3.464 秒、2.608% 完整请求收益。三个单项均先归档，未单独采用，也未新跑整模型验收；投影不是新的端到端实测。下一步研究与其他未选用的小项组合超过 3%。 [修正输入构造、全部逐位及配对原始记录](../../docs/stable-max-rescale-f12r2n09.json)

## P 成对减法与负最大值表示

`build-p-paired-sub.py` 为完整 KV 块的 8 个 new_max 连续放置寄存器，保留纯重命名、负最大值标量、成对 P 算术及其 stable-max 组合的对照。普通重命名通过逆映射逐字节恢复原汇编；新位置只占用原 tail 的死寄存器，退出完整循环时恢复原值，`.LBB0_20` 后代码完全不变。实际 gfx936 汇编器拒绝 packed add 的 neg/opsel 修饰和直接 `v_pk_sub_f32`，因此通过单项 min 的输入取负生成负 new_max，再使用无修饰的 `v_pk_add_f32`。不是直接采用其他架构的指令语法。

初版保留正向 Phi，每次循环用 XOR 恢复它；另有负向 Phi 只在完整循环首次入口转换、退出恢复的对照。所有方式保持原 SFU 数量、RNE、归约树和 MMAC 累加顺序。10 个变体各 95 条通用最终输出位模式检查，以及合计 420 条稳定 max 专用边界检查通过；小于 4096 的长度仍为旧回退覆盖。胜出微测候选另通过 12 条全部 BF16 NaN payload/零/最大有限值检查。六个产物与汇编的独立重建逐字节一致。

首批 GPU0 的后半计时出现明显扰动，本任务退出后驱动仍显示其他进程占用约 58 GB、利用率 100%；整批性能仅保留，不用于选版，数值检查结果不因此失效。移到空闲物理 GPU1，两批 39 组配对中基准 CV 均小于 0.15%，首尾自身配对接近零。仅 P 成对算术约 1.76% 单 kernel 收益；与尚未选用 stable-max 组合由 20.861586→19.187025 ms、14.117272→13.062872 ms，配对下降 8.0295%/7.4775%。负向 Phi 版本没有足以承担额外复杂度的独立收益，优先较简单的正向 Phi 组合。

按 2450 次长注意力调用，该组合仅投影约 3.089% 完整请求收益，不是新的端到端实测。GPU0 的外部任务退出后，2026-11-27 16:46 在八卡空闲条件下启动真实回放。第 1/25/49 次去噪中双方视频和音频输出逐位一致，单次 Transformer 平均下降 4.1325%，对应完整请求投影约 3.4808%。普通官方 A 和完整 11 请求正在收尾验证，尚未选用；当前正式配置仍为已验收的 132.824526 秒。独立 `hipprof --pmc` 在物理 GPU1 完成，原始 TXT 标记 gpu id1，220 VGPR、64 SGPR、16 KiB LDS、零 scratch；27570 token 的 bank conflict 约 0.0213%，22016 为零。 [全部原始配对、扰动证据、数值检查与取舍](../../docs/p-paired-sub-f12r2n09.json) · [原始 PMC TXT](../../docs/p-paired-sub-f12r2n09-basic-pmc.txt)

额外四个负 logit 变体均完成 95 条通用逐位检查、168 条专用边界及独立的 GPU1 配对。仅负 logit 标量约退化 0.31%，packed 约改善 1.37%；与 guard 组合约 7.96%/7.42%，没有胜过正 Phi 的 `guard-packed`，不选用。负 logit 的缩放系数只在完整循环入口和出口翻转，尾部保持原样；gfx936 packed FMA 只接受实际探测可用的向量操作数。

`build-stable-paired-release.sh SOFTMAX_RELEASE NEW_OUTPUT_DIR` 将已冻结的精简 softmax 发布目录依次经过 stable-max guarded 和正 Phi packed 变换，提供完整主机 ABI、六个产物及来源清单。独立 CPU 发布重建的可加载区段和 ELF 映射均与真实回放候选相同；V 预排布镜像只有调试信息差异，其余五个产物逐字节相同。重建不会自动把候选设为正式版本。

`build-running-max-fold.py` 的 max3/min3 方向仅完成 CPU 构建，未执行 GPU 数值或性能验证。用户要求本轮收尾整合后暂停，因此不启动该方向后续测试，不能将其计入有效性能或已验收优化。

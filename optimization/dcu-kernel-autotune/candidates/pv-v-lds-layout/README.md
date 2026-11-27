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

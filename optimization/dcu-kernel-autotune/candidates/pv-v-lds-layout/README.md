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

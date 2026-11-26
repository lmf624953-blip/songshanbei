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


## 后续探索入口（2026-11-26）

`build-q256.py REFERENCE OUTPUT` 通过九齿重新生成 Q256/K32，而后应用独立微测的 V 合并加载、Q 直接加载及同长度 packed stride 约束，编译 4/8 warps、P maxPhase16/32。保留完整 KV 上界掩码，未放宽生产源码的 hash guard。四个初始变体均出现 private segment：4 warps 为 VGPR256/424B，8 warps 为 VGPR128/348B；是否还有收益以串行实测为准。这里的 fallback 使用同一固定步长镜像，**禁止将本实验目录作为正式推理配置**。

`build-cache-policy.py REFERENCE OUTPUT` 仅给已验收 Q128 IR 中的两处 K 和两处 V 加载增加缓存属性，筛 K.cg、V.cg、K/V.cg、K/V.cv、K/V.ca。动态形状、尾部掩码、softmax 和矩阵点积表达式保持原样。CPU 编译时五个变体均为 VGPR215、0 private segment；ISA 已显示对应加载属性生效，数值与性能仍需微测通过。

两种构建均生成 `v0` 原镜像对照，可复用 `screen.py REFERENCE OUTPUT RESULT` 做两个目标长度的普通/幅度放大数值检查、尾部边界和无 profiler 交替计时。正式采用前仍须通用守卫、真实八卡和多用例验证。

## Q256 与缓存属性实测结论

Q256 四个配置在普通、放大输入与尾部检查上逐位通过，但两个长度较 Q128 慢约 126%–263%，同时出现 348–424B private segment；本轮淘汰，不能把更大分块当作收益。见 [Q256 结果](../../docs/q256-1218234.json)。

缓存属性候选的 K.cg、V.cg、K/V.cg 均只有约 ±0.1% 波动，K/V.cv 慢约 13%–14%，没有稳定胜出项。K/V.ca 与对照的 ELF 加载区段相同，是控制项；筛选器已排除这类相同代码，避免把计时噪声当作优化。见 [缓存实测](../../docs/q128-cache-policy-1218234.json) 和 [ELF 证明](../../docs/q128-cache-policy-1218234-elf-equivalence.json)。默认仍保持已验收 Q128。


## Q128 packed 的 PV 操作数打包候选

`build-pv-wide-operand.py` 把 PV 的 `kWidth=4` 改为8，保留相同数量的BF16 MMAC指令、全部KV遍历及FP32缩放舍入边界，研究宽LDS读取是否更有利。新增P向量布局、V布局及P直接转换对照。编译已完成；两种V布局出现8/24B private segment，会跳过GPU微测；其余镜像仍无scratch。P直接转换版本为213VGPR、10处barrier，单凭这些资源变化不宣布更快。

`screen-pv-wide-operand.py` 先核对三镜像分派、哈希和ABI，排除相同`.text`与spills，再在实际packed stride的27570/22016长度交替计时，并检查普通、放大输入及末尾哨兵。只有两个长度均逐位一致且收益至少0.5%才进入真实去噪验证。本轮微测等待compact归约与row SwiGLU的七次正式生成退出，不与正式计时并行。

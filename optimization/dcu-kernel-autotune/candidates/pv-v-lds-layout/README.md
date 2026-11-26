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

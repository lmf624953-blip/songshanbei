# K 预排布与直接矩阵操作数加载

本实验研究先把 K 排布为适合 QK 矩阵操作数的存储方式，减少每个 Q 块重复的 K LDS 暂存。九齿原始计算及其 BF16 MMAC 数量、FP32 累加顺序、完整 KV 遍历均保留。候选只用于本目录的诊断入口，K 指针表示预排布缓冲区。禁止把这些镜像直接放入正式推理的配置。

`build.py --unit 0` 构建 head、dim、padded-sequence 的整轴转置。`--unit 8` 或 `--unit 16` 先按 32 个 token 分块，再以 8/16 个 BF16 为连续单元排布。主机 ABI 仍使用 Q128 packed 的三个镜像，只有 packed 镜像更改读取地址。

```bash
export H3_AUTOTUNE_CODE_ROOT=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
python build.py "$reference" "$build" --unit 8
python bench.py "$reference" "$build" --unit 8 --output "$results"
hipprof --pmc --pmc-type 3 --kernel-name dense_attention_aot_kernel -o "$pmc" \
  python bench.py "$reference" "$build" --unit 8 --profile-candidate --output "$profile"
```

`bench.py` 对两种生产长度的普通与放大 QK 输入检查逐位一致、有限值及末尾哨兵。无 profiler 测量使用 11 组交替顺序，并单列原注意力、候选注意力、排布和排布加注意力。诊断排布采用 PyTorch copy。正式集成需要九齿排布，但本轮候选注意力即使不计排布成本也已经慢于原版，因此没有投入正式排布实现。

2026-11-26 在新节点 a14r1n11 上测得，所有配置均通过数值校验。长度 27570/22016 的原注意力约为 35.10/23.70 ms。整轴转置导致标量读取和 236 B scratch，排布加注意力分别约为 127.86/87.03 ms。8 元素方案恢复向量读取和零 scratch，barrier 从 12 处降到 8 处，注意力本身仍约为 36.39/24.56 ms，计入排布后慢 4.49%/4.62%。16 元素方案计入排布后慢 7.78%/7.91%。

该结果淘汰本轮三种排布，正式默认继续使用已有验收版。不同的跨 lane/warp 排布仍需单独证据，不能把本轮有限搜索称为全部预排布方案的最优结论。

独立 `hipprof --pmc --pmc-type 3` 在两个生产长度各采集 4 次注意力调用。8 元素版本的 VMEM 读取指令计数由 31,314,920/19,918,976 增至 62,599,712/39,799,424，LDS 指令约减少 19.2%，L2 命中率基本一致。该版本的 LDS bank conflict 计数与基准相同。源码改为复用已有 `k_resources.py` 后再次构建，可加载区段、映射和目标与被测镜像一致。见 [配对数据和 PMC](../../docs/k-pretranspose-a14r1n11.json)。

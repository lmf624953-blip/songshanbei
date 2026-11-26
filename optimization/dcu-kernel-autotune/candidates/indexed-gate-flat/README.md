# 索引门控的扁平读取

候选保留 BF16 乘法舍入和随后残差加法，比较扁平寻址、对齐提示及按索引分段的实现。使用 `build.py` 为不同 tile 和 warp 数编译镜像，再用 `bench.py` 对照现用 AOT 索引门控。

```bash
python bench.py "$builds" --reference "$indexed_build" --skip-buffer --output "$results"
```

本轮使用 `--skip-buffer` 排除包含未降解 LLVM 内部符号的实验镜像。实际运行的 12 个镜像共通过 72 项逐位比较、有限值和输出尾部哨兵检查。对分组索引，长度 27570 上对齐版本的配对中位收益为 1.50%，在 22016 上回退 1.29%。长度 22016 上最快的分段版本与对照持平，在 27570 上回退 1.16%。没有候选在两个生产长度上同时提速。见 [无 profiler 配对结果](../../docs/indexed-gate-flat-valid-images-1218234-micro.json)。

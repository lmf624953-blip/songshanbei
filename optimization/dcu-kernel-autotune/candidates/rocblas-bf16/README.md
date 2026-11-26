# 去噪 BF16 GEMM 选核筛选

在 `a14r1n11` 上以现用配置回放公开 A 第 25 次去噪输入。加载 `selected-best.env` 后设置 `ROCBLAS_LAYER=3`、`ROCBLAS_LOG_BENCH_PATH` 和 `ROCBLAS_LOG_TRACE_PATH`，使用 `optimization/adaln/launch.py --replay-run --steps 25 --iterations 3 --candidate-only`。回放的 video/audio 哈希与历史现用配置的同一输入一致。带日志的回放仅用于调用参数采集。

原日志有 8512 条，四个长序列 BF16 投影各有 1600 条，对应 50 层、8 卡和四次回放调用。`h3-bf16-a-projections.txt` 保存完整的 Layer3 命令，`shapes.txt` 保存同一组尺寸的 DTK 选核输入。四个调用均为 TN、batch=1、alpha=1、beta=0、BF16 输入与输出、FP32 累加。`shapes.txt` 的列为 `M N K batch`。设备微测沿用 DTK 工具对 batch=1 展开自然 stride 的行为，前导维度与原始调用一致。

在私有 DTK 26.04 `auto_select_tools` 中配置 `mode=user`、`source=layer3`、`hardware_type=BW`、`gpu_ids=0`、`update_target_config=0`。全量 1420 个候选在 420 秒时限内未产出结果。随后复用已编译的两个工具，筛选 16 个显式 MMAC 和 49 个 256×256 tile 候选，目录提供比对所用清单。两组都固定 `gfx936`、`GSU1` 和同一张卡。各组默认运行位于强制选核之前与之后，均使用正式 HIP 环境中的同一库。默认组预热 5 次、计时 20 次，筛选组预热 2 次、计时 6 次。

```bash
bench="$private_tool/source-test/bin/bench-test"
tune="$private_tool/source-all/bin/tune-all"
unset ROCBLAS_LAYER ROCBLAS_TENSILE_LIBPATH TENSILE_DB KERNEL_TEST ROCBLAS_UNSELECT_CHECK
"$bench" "$out/default-before.txt" 5 20 TN bf16 0
ROCBLAS_UNSELECT_CHECK=1 "$tune" "$out/tile256.txt" 2 6 BW TN bf16 "$candidate/tile256-kernels.txt" 0
"$bench" "$out/default-after.txt" 5 20 TN bf16 0
```

以上三个输入尺寸文件均由 `shapes.txt` 复制。单卡筛选使用 `tune-all` 报告的吞吐量换算耗时，两个基准的计时来自独立 `bench-test`。两组候选的四个形状均无胜出项。全量搜索超时不构成性能结论。见 [采集与微测结果](../../docs/rocblas-bf16-a14r1n11.json)。

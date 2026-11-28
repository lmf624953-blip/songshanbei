# 当前 rocBLAS 候选覆盖与逐位筛选

本实验从实际加载的 DTK Tensile MessagePack 目录提取 BF16 TN 的候选，不修改公共 DTK，也不改变正式推理分派。展开的 `name` 包含 `AMAS3` 等字段，自动调优使用的 `info.SolutionNameMin` 是另一种短名称；应通过目录映射比较，不能直接按字符串是否含 `AMAS3` 判断搜索覆盖。

`f13r1n16` 的实际库有 1446 个候选，历史全清单的 1420 个短名称全部能映射，其中 26 个不在旧清单。历史 1420 全搜索超时，完成的是 16+49 个子集，因此也不能把旧清单当作全部实测证据。本轮选取这 26 个及 macroTile 两维均不小于64的16个RK1候选，覆盖8种形状，共336组合。

```bash
python extract-candidates.py "$actual_tensile_dat" "$previous_kernel_list" "$new_extraction"

# 激活正式推理使用的 DTK、GCC 环境后编译；仅C++主机程序。
"$CXX" -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -I"$DTK_ROOT/include" \
  screen-solutions.cc -L"$DTK_ROOT/lib" -lrocblas -lamdhip64 -o "$new_binary"

# shapes.tsv每行为M N K；candidates.tsv每行为库index与短名称。
# 关闭采集变量及外部选择配置，保持同一卡空闲。
unset KERNEL_TEST ROCBLAS_TENSILE_LIBPATH TENSILE_DB
unset ROCBLAS_LAYER ROCBLAS_LOG_BENCH_PATH ROCBLAS_LOG_SPLIT_BY_HANDLE
"$new_binary" "$shapes" "$new_extraction/targeted-candidates.tsv" 2 4 5 "$new_results"
```

库的 `index` 是0-based；调用 `rocblas_gemm_algo_solution_index` 时传入index+1。本轮用独立hipprof核对实际展开名称，确认没有默默回退默认路径。对于其他库版本必须重新验证。

输入为确定性原始BF16有限位模式，覆盖正负及不同指数，生产TN布局、alpha1/beta0、batch1、FP32累加。每种形状先保存默认结果，再逐元素比较完整BF16位；存在位差异时不进入计时选择。通过后以GPU event记录交替顺序配对；候选库初次加载与分配在计时外。它是单GEMM筛选，不能替代真实去噪、成片质量及完整公开入口资格。

336组合的rocBLAS调用均成功，128个逐位一致，208个有差异；16个RK1在8个形状均一致，其余26个在所测输入均不一致。较快候选与默认路径对照再进行39组配对，主要形状收益仍只有约1%–2.7% GEMM耗时。按各主要投影每层一次、50层、49次Transformer估算，完整A约0.32%投影收益，不是完整请求成绩，按低于3%单项先归档。 [原始样本、映射与独立trace证据](../../docs/gemm-coverage-f13r1n16.json)

## 补齐当前库的候选覆盖

`prepare-unmeasured.py` 复用历史清单和实际目录。先筛 75 个未测的大 tile 单 split 解，随后排除这批并遍历其余 1264 个解。历史 65 个子集补充完整位模式检查，历史 42 个定向候选的有效记录直接复用。四种公开 A 投影共核对 5784 个形状与 index 组合。

实验构建可以启用 `H3_SCREEN_DEVICE_COMPARE`，在设备上比较完整 BF16 位模式，并将每种形状的前三个结果和 CPU 完整数组计数交叉核对。该 HIP 检查只存在于独立筛选程序中。未定义时沿用原 CPU 比较，原七个参数的入口保持可用。

~~~bash
candidate="$H3_AUTOTUNE_CODE_ROOT/optimization/dcu-kernel-autotune/candidates/rocblas-coverage"
library="$DTK_ROOT/lib/rocblas/library_gpu5/TensileLibrary_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx936.dat"
# out 指向已创建的独立实验目录，以下输出子目录及结果文件须尚不存在。
python "$candidate/prepare-unmeasured.py" "$library" "$out/large"
python "$candidate/prepare-unmeasured.py" "$library" "$out/remaining" \
    --scope remaining --exclude-selection "$out/large/selection.json"
hipcc -O2 -std=c++17 --offload-arch=gfx936 -DH3_SCREEN_DEVICE_COMPARE=1 -x hip \
    "$candidate/screen-solutions.cc" -lrocblas -o "$out/screen"
printf '2688 27570 5376\n3584 27570 5376\n5376 27570 1792\n5376 27570 896\n' > "$out/shapes.tsv"
HIP_VISIBLE_DEVICES=0 "$out/screen" "$out/shapes.tsv" "$out/large/candidates.tsv" \
    2 2 9 "$out/large.jsonl"
HIP_VISIBLE_DEVICES=0 "$out/screen" "$out/shapes.tsv" "$out/remaining/candidates.tsv" \
    2 2 9 "$out/remaining.jsonl"
~~~

直接读取 MessagePack 目录需要实验 Python 环境提供 `msgpack`。已有目录导出时可传 `--catalog-json EXPORTED_JSON`，脚本仍核对原库 SHA 及候选数。旧 65 个子集从已提交 `rocblas-bf16-a14r1n11.json` 的 `subsets.*.kernels` 映射 index，固定库的完整记录随报告保留。

独立 PMC 用一个形状与三个候选，按控制、位差候选、逐位通过但较慢的候选排序。每个检查完成后，`--profile-check` 仅在默认和候选各一次发射期间打开采样，停止前排空两秒。该模式不输出性能配对。

~~~bash
HIP_VISIBLE_DEVICES=0 hipprof --pmc --pmc-off --devices 0 -o "$out/pmc" \
    "$out/screen" "$out/profile-shapes.tsv" "$out/profile-candidates.tsv" \
    2 2 9 "$out/pmc-check.jsonl" --profile-check
~~~

本轮实验目录内保存各阶段冻结源码、二进制和输入 TSV，初始筛选源码还保存于 `snapshots/a02r3n04/`。当前主实现增加了独立采样参数，七参数数值筛选与快照执行相同运算。`summarize-unmeasured.py EXPERIMENT OUTPUT_DOCS --label a02r3n04` 核对原始阶段退出码、5784 个无重复组合、控制样本和六条 PMC 的实际符号，并生成统一 JSONL、摘要和原始 PMC 副本。[本轮结果与限制](../../docs/GEMM目录补全-a02r3n04.md)

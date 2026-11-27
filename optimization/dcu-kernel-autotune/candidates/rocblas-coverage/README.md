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

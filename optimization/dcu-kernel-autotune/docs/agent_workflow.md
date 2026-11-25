# Agent 优化约定

每个 agent 任务只处理一个假设，例如降低 VGPR、缩短 live range、改变并行
映射或拆分热点 kernel。不要在同一轮混入多个无法归因的改动。

1. `context ROUND` 获取当前热点和历史结果。
2. 阅读 profile 算法对应的 Kokkos 源码，并只修改 `source_delta` 中的文件。
3. 运行 `run ROUND`。
4. 只有 correctness 为 `PASS` 且性能指标改善时，才把该轮作为下一轮基础。
5. 对失败或回退轮次保留报告，方便 agent 避免重复尝试。

`source_delta` 的目录结构必须与 `SOURCE_ROOT` 相对路径一致。例如：

```text
rounds/round_001/source_delta/src/KOKKOS/pair_sw_kokkos.h
```

如果源码树不是 LAMMPS stable 版本，先更新 profile 的 `SNAPSHOT_FILES`，再执行
`init`。框架不会猜测源码文件，也不会自动修改未列入快照的文件。

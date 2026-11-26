# 保持原库归约顺序的 RMSNorm/AdaLN 融合

这是 `wide-norm-numerics` 的后续候选。对每行保持原库 1024 线程分工、wave 内求和树和 wave 0..15 的 FP32 顺序累加，再执行原有 BF16 RMSNorm 与索引 AdaLN 舍入。源码由九齿表达，C++ 主机桥复用已有 guarded native ABI。

`build.py NEW_OUTPUT --mode original-serial --warps 4` 构建独立微测镜像，4/8/16 warps 均已测。使用 `../wide-norm-indexed/bench.py BUILD_ROOT --output NEW_RESULT` 对照当前原始 RMSNorm+已验收 indexed AdaLN，核对完整输出、有限值和末段边界，按同一 InfiniCore stream 的 event 交替计时。

1/33/22016/27570 行全部逐位一致。4-warps 在两种目标行数减少约 33.75% / 34.00%，8-warps 减少约 14.17% / 14.78%，16-warps 反而慢约 48%，故只选择 4-warps 进入真实八卡验证。这里对照是原两段算子；不能将百分比当作整体去噪或端到端收益。详见 [单算子实测](../../docs/wide-norm-original-serial-1218234-micro.json)。

可复用的生产候选构建位于 `optimization/adaln/build-wide-norm-indexed.py NEW_OUTPUT --mode original-serial --warps 4`。它自动保留编译 IR，检查 `sizePerThread=1` 布局守卫，生成同一动态 rows/table_rows ABI 和主机桥。完整重建可显式增加 `--fused-wide-norm-indexed-original-serial`，它同时启用必要的 Q/K RMSNorm/MRoPE 原生补丁。模式默认仍为 `naive`，新模式没有自动成为正式默认；通过 `H3_WIDE_NORM_INDEXED_BUILD` 显式装载并配合已重建 guarded native。长度按运行时传入，不按提示词、seed 或输入内容分派。

独立微测与从交付源码重建的设备 ELF 加载区段相同，主机桥也另做动态 rows/table_rows 对照。真实八卡消融、普通官方 A 和 A,A,A,A,B,C,A 顺序回归正在执行；通过完整质量与正式计时前不修改 `selected-best.env`。

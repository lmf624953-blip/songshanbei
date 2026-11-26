# 宽 RMSNorm 数值诊断

本目录只诊断归约顺序，不作为性能计时或正式生产配置。`reference_probe.hip` 是原算法的中间量探针；新增生产计算仍须通过九齿表达。官方步数、帧数、画布与 seed 不变。

已部署 `libinfiniop.so` 的 BF16/1024 线程 RMSNorm ISA 显示：各线程以 1024 步长顺序累加平方，wave 内采用 shuffle-down 1/2/4/8/16/32，最后 thread 0 顺序累加 wave 0..15。当前 DTK hipCUB 的跨 wave 树状归约会产生少量差异。`extract-reference.py LIBRARY NEW_OUTPUT --objdump PATH` 可独立提取并记录实际原库 SHA-256 与设备 ELF 来源。

`build.py NEW_OUTPUT` 编译 default hipCUB、fast hipCUB、原库 serial 三个 HIP 探针，以及 naive、cub-warp、original-serial 三个九齿诊断镜像。`bench.py BUILD --output NEW_RESULT` 对同一 InfiniCore stream 的原始 RMSNorm 核对 BF16 输出、FP32 平方和及 inverse RMS。环境使用当前运行 skill；设置 `H3_AUTOTUNE_CODE_ROOT` 指向已适配源码和 `H3_AOT_SAVE_IR=1`。GPU 微测只用一张卡，与其他 GPU 性能实验串行。

2026-11-26 的 1/33/4096 行、幅度 1/8/1 测试中，serial HIP 探针与实际原库输出逐位一致；original-serial 九齿版本与该探针的平方和、inverse RMS 和最终 BF16 输出全部逐位一致，有限值和尾部检查通过。naive 与当前 hipCUB 版本仍存在少量差异。证据见 [完整诊断](../../docs/wide-norm-numerics-serial-1218234.json)。这只证明已测输入；尚需融合完整调制、目标大行数、实际模型和性能验证。

九齿目前不支持在这一调度中将归约域手动 reshape 后输出另一域，当前源码采用 16 个按 wave 掩码的归约，再顺序累加，保留明确的 FP32 顺序。布局阶段改为 `sizePerThread=1`，避免多元素线程局部求和改变原库归约次序。后续性能优化须继续核对中间量。

历史失败保留在本地：首次 HIP 编译遗漏 C++17，随后探针遗漏 `__launch_bounds__(1024)`；后者的默认最大 workgroup 为 256，与 1024 线程发射不一致，结果无效，不能用于归因。加入明确 launch bounds 后重新验证。两次多域归约写法分别因 SSA 不支持 `partial.reshape`、不同归约域被拒绝，未进入正式模型。以上均是诊断工具构建问题，不归为原模型或显卡错误。

# Packed BF16 SwiGLU 优化

实际 `GateUpParallelLinear::forward_split` 输出 BF16 `[1,N,3584]`，两个视图各宽 1792、行步长 3584。原库先将 `1/(1+exp(-gate))` 舍入到 BF16，再以 FP32 依次乘 gate、up，最后写 BF16。候选保留这些顺序及原库快速倒数，不合并或改变模型算子。九齿生成设备计算，C++ 仅装载、发射和检查形状。N 来自运行时，既适用于官方 A，也接受其他满足布局的长度。

`build.py` 的 `--math fast-rcp --tile 1024 --warps 4` 为当前生产候选。`bench.py BUILD_ROOT --output RESULT` 在同一 InfiniCore stream 与原库交替计时，覆盖全部 65280 个有限 BF16 gate 编码、行数 1/33/1024/22016/27570、随机输入、64 倍幅值和末段保护值。全部逐位一致；27570 行原库/候选中位耗时约 1.072/0.346 ms，减少约 67.7%。仅使用 `accurate` 或 `fast` 而不用原库倒数的实现，会在极小数值上不同，故不作为正式候选。

生产入口位于 `optimization/adaln/build-swiglu.py NEW_OUTPUT`；整套重建可追加 `--fused-swiglu`。原生补丁只接受 HYGON、BF16、三维 batch=1、宽1792、两侧 packed 行步长3584、列步长1、同设备、16-byte对齐、int32安全范围且非图录制的输入，其余回退原库。`H3_SWIGLU_BUILD` 装载通过 SHA256 和 ABI 检查的 AOT 与主机桥；`load-candidate-config.sh` 清除上次配置。

`check-host.py PRODUCTION_BUILD --output RESULT` 已验证生产桥的全部同类数值与尾部边界。真实八卡通过 `--swiglu-ablation` 在第 1/25/49 步单独开关该算子，其他已验收优化保持启用；再执行普通官方 A 与完整七请求验证。当前仍在真实模型验收，不把单算子收益写作端到端收益，不修改已选默认。

`build-row.py` 是后续微测方向：按 packed 行组织输出 tile，以减少线性索引除余。只构建微测镜像，未接入正式配置；GPU 微测与正式模型验收串行。

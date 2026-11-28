# 嵩山杯 MiniMax-H3 FL2VA

当前完成完整本地验收的最优组合为 **`peer-sharded-norm-f13r1n16`**，采用九齿长短注意力、归一化与门控融合，以及按完整行分片的归一化和跨卡发布。公开 A 在 `f13r1n16` 的八张 BW1000 上，常驻预热后的完整请求三次均值为 **116.975680 秒**，包含 MP4 写出。相同实例的前序 peer 对照为 **122.526417 秒**，耗时下降 **4.530237%**。

新进程从启动到退出的既有记录为 **875.75 秒**，包含模型加载、首次生成和清理。常驻请求与新进程时间分别记录，复测时以同一计时口径比较。

- [当前最优版本、完整构建链和规则核查](optimization/当前最优版本.md)
- [本次核查的文件身份和证据](optimization/当前最优版本核查.json)
- [统一复测命令和计时字段](optimization/公开A耗时复测说明.md)
- [模型级实现、适用范围及资格验证](optimization/dcu-kernel-autotune/candidates/peer-sharded-norm-model/README.md)
- [环境准备与优化记录](optimization/README.md)

正式公开 A 为 **768×928、50 steps、124 帧、seed 42**，原调度器执行 **49 次 Transformer**。本版本已完成普通默认入口、真实八卡第 1/25/49 次去噪及 A×8/B/C/返回 A 回归。所有成片与对应已验收版本逐字节相同。A/B/C 对独立参考的 SSIM 分别为 **93.832013%、99.404123%、99.630047%**。

完整适配框架源码位于 [h3](h3/README.md)，InfiniCore、InfiniLM、九齿及其他依赖源码位于 `h3/frameworks/`，上游分支与 commit 记录在 [UPSTREAM.json](h3/frameworks/UPSTREAM.json)。使用完整 Git 历史克隆 `xsh-opt`，构建依赖历史基线提交。运行环境、比赛权重和本地稳定发布二进制按文档准备。

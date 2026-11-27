# 连续分片计算与 AllGather 发布

保留已验收 `peer-gate-nt-f13r1n16/peer` 的九齿设备镜像、ABI、每元素 BF16 累加及门控顺序。只将独立 HIP 远端写发布替换为分组 `infinicclAllGather`。AllGather 复制每卡连续的 `count/8` 个 BF16 元素，不执行求和，因此没有 ReduceScatter 改变 BF16 加法顺序的问题。

全部输入就绪的依赖保留；九齿计算与 AllGather 在各自同一 stream 排队，AllGather 完成后同 stream 的后继才能消费。主机桥检查输出与所有输入、残差地址区间不重叠。校准不符仍执行原 InfiniCCL 求和与九齿门控。此实现仍依赖 H3 在 TP rank 间复制 residual、gate、indices 的不变量。

本目录只用于独立实验。`build.py` 复用 `../rccl-sequence-parallel/peer_transport.hip` 的输入依赖与校准支持，编译时记录依赖文件和产物 SHA；旧的 HIP 发布函数虽然留在该依赖库中，但快路径调用的是 AllGather。

```bash
# 激活运行环境并设置 H3_AUTOTUNE_CODE_ROOT；输出目录必须不存在。
python build.py "$accepted_release/peer" "$new_runtime"
```

使用 `../peer-direct-publish/bench-direct.py` 的 `--bridge-build` 和 `--reference-peer-build` 比较同一 communicator 上的现用两阶段与 AllGather 版本。两边都使用独立输出。先做带哨兵和 GPU 消费者的正确性诊断，再去掉 `--consumer-copy`，做交替配对计时。记录 InfiniCCL 实际校准是否进入快路径；无 profiler 独立算子成绩不能直接宣称为模型端到端成绩。

结果与取舍见 [两阶段与 AllGather 发布实验](../../docs/两阶段与AllGather发布-f13r1n16.md)。未经八卡模型、完整公开入口及多用例验收，本目录不会替换正式默认。

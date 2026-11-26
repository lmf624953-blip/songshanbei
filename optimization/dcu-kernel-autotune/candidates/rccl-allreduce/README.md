# 八卡 BF16 AllReduce 通道筛选

从现用 Q128 组合的八卡去噪 trace 提取了两个原地 BF16 Sum AllReduce 尺寸。大消息为 148216320 元素（296432640 字节），三个采样窗口全八卡共 4800 次。小消息为 3806208 元素（7612416 字节），共 192 次。

加载竞赛运行环境，为脚本提供 InfiniCCL 依赖的同一 DTK `librccl.so.1`，并保持八卡可见。每个配置使用独立进程。脚本在计时外恢复相同输入，记录八卡最大设备事件时间和主机同步时间，逐卡哈希完整输出字节。实验按默认、4、8、16、32 通道、默认的顺序执行。固定通道时同时选用 Ring/Simple。

```bash
python bench-allreduce.py "$DTK_ROOT/lib/librccl.so.1"
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 \
  python bench-allreduce.py "$DTK_ROOT/lib/librccl.so.1"
```

默认前后两次的 296 MB 消息中位数分别为 5.669 和 5.665 ms。固定 4 和 8 通道变慢且两个尺寸的输出字节均改变。16 和 32 通道分别为 5.645 和 5.640 ms，两个尺寸的输出哈希均与默认一致。小消息偶有收益，在实际 trace 中调用频率低。固定通道配置只归档，未进入正式去噪回放。结果仅适用于该节点、该 RCCL 版本及所测尺寸和输入。见 [原始样本](../../docs/rccl-allreduce-a14r1n11.json)。

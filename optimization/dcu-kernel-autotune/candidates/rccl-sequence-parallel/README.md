# 分片消费者前的集合数值顺序检查

现用去噪中，AllReduce后还有归一化与门控工作。一个潜在方向是先ReduceScatter，只在各卡负责的序列分片执行逐元素工作，再AllGather，以减少重复计算；另一方向是分块AllReduce与消费者重叠。两者都必须首先证明符合现用BF16结果要求，不能只依据数学上的求和等价。

`probe-collective-order.py` 使用与现用InfiniCCL相同的DTK RCCL，在八张本机卡构造相同输入，比较重复整段AllReduce、显式RS/AG及2/4/8个完整行分块。原形状为27570×5376，另测补齐至27576行以满足完整行八等分；补齐版本与同尺寸整段AllReduce比较，不能当作与未补齐正式输出已经一致。输入包含确定性普通BF16值和不同rank尺度，比较所有原始有效元素的原始位模式。

```bash
# 激活正式推理环境并使八卡可见。
python probe-collective-order.py "$DTK_ROOT/lib/librccl.so.1" "$new_output" --pairs 5
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 \
  python probe-collective-order.py "$DTK_ROOT/lib/librccl.so.1" "$new_ring16_output" --pairs 5
```

本轮默认与固定Ring/Simple16通道均发现位差异：原尺寸显式RS/AG约7524万个元素不同，2/4/8分块约3805万/7596万/7179万，八卡一致。重复整段对照用于排除测量流程错误。不同配置在本轮只与各自整段对照比较，不把配置之间的位结果当作已经相同。没有接入分片消费者，也没有新的模型性能或质量结论。

本轮拒绝的是这些直接用现成集合API拆分的形式；没有证明保持原始逐元素加法顺序的自定义分片实现不可行。后续如继续，需要先建立当前RCCL每个输出位置的实际rank累加顺序，再验证分片实现是否严格保留，不能用FP32临时求和掩盖改变。[完整结果](../../docs/gemm-coverage-f13r1n16.json)

## 累加链诊断的初步证据

`infer-reduction-order.py LIBRARY NEW_OUTPUT` 使用每rank32种BF16探针重复填充原尺寸，采样106902组位置，并与全部40320种逐步BF16 RNE串行排列比较。当前8种输出signature均能匹配，表现为一个rank环的循环移位，第一对加法交换等价；部分signature仍有多个等价匹配，不能宣称已唯一反推出实际指令序。全部采样通过也不证明未采样位置或其他输入、尺寸已经通过。原始紧凑NPZ留在validation路径。下一轮须增加独立探针、建立块与尾部的索引映射，并以随机完整数组核对自定义实现，才可讨论实用候选。

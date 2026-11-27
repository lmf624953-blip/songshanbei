# 保持 BF16 累加顺序的 peer 分片与门控融合实验

2026-11-27，在 `f13r1n16` 上实现新的八卡 peer 原型。最终候选把每个输出元素按现用 RCCL rank 顺序逐步 BF16 相加，在该卡负责的分片上执行原门控残差计算，再推送结果到八卡。与 **现用 RCCL + 九齿 indexed_gate** 的39组无profiler配对：公开A对应消息约 **6.370 → 4.920 ms**，两个其他长度同样约节省22.5%–22.8%，完整BF16输出和尾部检查全部通过。按49次Transformer、每次100个相关投影，仅作完整A粗略投影约 **5.59%**；尚未接入模型，没有新的完整请求成绩。正式默认仍为已验收127.831秒。

## 为什么先检查集合顺序

直接ReduceScatter + AllGather及2/4/8个完整行分块AllReduce，在默认配置和固定Ring/Simple16通道下均改变BF16位结果，重复整段AllReduce对照通过。原尺寸27570×5376共148216320元素，直接拆分每卡约3805万至7596万个元素不同，因此这些形式未接入消费者。补齐到27576行的版本也失败；它与同尺寸整段对照比较，不能当作与未补齐正式结果一致。[前一轮完整记录](../../docs/gemm-coverage-f13r1n16.json)

新的 `infer-reduction-order.py` 用各rank的重复BF16探针识别采样位置。独立种子验证与RCCL DEBUG环信息相符，当前rank环为 `[0,1,3,2,4,6,5,7]`。活动16通道、每步524288个元素的索引模型，在原长度上无法区分256/512/1024尾块对齐；新增24577 token探针后，只有1024对齐通过所测边界。`validate-order-model.py` 按每个采样元素核对，保留失败模型。采样证明不代替完整随机数组验证。

## 实现及实测取舍

`peer_reduce.hip` 的第一阶段各卡处理一个连续输出分片，通过peer读取八个输入，按位置对应的原rank顺序逐步BF16 RNE相加。第二阶段比较远端读取与主动推送；推送版在消费者之前增加全部发布完成事件依赖。原地输入直到所有分片计算完成才被覆盖。

| 方案 | 296 MB消息的代表耗时 | 取舍 |
| --- | ---: | --- |
| peer读取结果，320 blocks | 约10.4 ms | 明显退化 |
| peer读取结果，增大网格 | 最好约5.37 ms | 单项完整请求投影不足3% |
| 主动推送结果 | 约4.95 ms，RCCL对照约5.72 ms | 单项投影约2.97%，暂不单独合入 |
| 按owner展开分支 | 约5.03 ms | 没有额外可靠收益 |
| packed rank查找 | 约4.93 ms | 不将不同进程的小差别宣布为收益 |
| 分片内门控后推送 | 约4.92 ms，RCCL+现用九齿门控约6.37 ms | 达到后续真实模型验证的候选门槛 |

融合门控保持 `(gate * activation).to(BF16)` 再与 residual 相加并转BF16的顺序。特殊值检查最初发现NaN编码不同：RCCL求和保留其原语义，九齿门控转换使用`0x7fff`。现在两种转换分别实现；修正后普通、NaN/Inf/带符号零、极小及大幅值输入全部通过。不能用普通输入通过代替这个检查。

最终候选在27570、22016、24577 token上各做39组配对，全部八卡、全部有效元素和末尾哨兵通过。独立CPU重建主机库及设备打包内容逐字节一致。独立PMC采集24个分片门控kernel，覆盖八卡、每卡三次；44 VGPR、48 SGPR、0 LDS、0 scratch，原始TXT保留。独立trace只用于分解阶段，带profiler的RCCL时间不用于正式收益。

## 复用命令

先激活正式推理使用的DTK/GCC/venv并使八卡可见。所有输出文件使用新路径。

```bash
python infer-reduction-order.py "$DTK_ROOT/lib/librccl.so.1" "$new_probe" --seed 82372
python validate-order-model.py "$new_model_report" "$new_probe"
hipcc -O3 -std=c++17 -ffp-contract=off -DPEER_PACKED_RING \
  --offload-arch=gfx936 -shared -fPIC peer_reduce.hip -o "$new_peer_library"
python bench-peer-reduce.py "$DTK_ROOT/lib/librccl.so.1" "$new_peer_library" "$new_results" \
  --gather write --vectors 1 --blocks 5120 --pairs 39 \
  --counts 148216320 118358016 132125952 --seed 99991 \
  --indexed-build "$current_indexed_build"
```

省略`--indexed-build`只比较AllReduce；`--case special|tiny|wide`可检查特殊值。`--fixed-order`只用于性能可行性，不能当作逐位合格。完整结果必须读取JSON的`exact`，不能只依据程序退出码，因为实验记录也保留不通过的候选。

`ring_order_map.py` 和源代码中的rank环目前描述本节点已经验证的实验条件；换节点、通信参数或尺寸后需重新验证。模型接入还需运行时校准及回退，不能直接将这个实验库替换正式通信库。[模型接入状态与约束](模型接入方案.md) · [完整原始记录](../../docs/peer-reduce-f13r1n16.json) · [原始PMC TXT](../../docs/peer-reduce-f13r1n16-basic-pmc.txt)

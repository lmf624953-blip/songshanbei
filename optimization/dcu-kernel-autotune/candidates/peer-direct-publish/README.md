# 九齿分片门控直接发布实验

2026-11-28，`f13r1n16`。该目录为独立实验，正式配置仍为已验收的120.698043秒版本。本轮朴素直接发布及固定owner直接发布均通过数值检查，但未找到达到3%完整请求门槛的可靠收益，因此未启动新的八卡模型回放和完整生成，也未改动正式配置。

## 表达、事件及适用前提

`direct_kernel.py`按原位置owner和已校准rank环逐步BF16累加、门控/残差后，直接把分片写到八个最终输出，取消中间partial读写和独立发布kernel。`direct_bridge.cc`要求输入/残差与所有输出不别名，并保留原输入就绪和全部生产者发布完成依赖。原地调用被显式拒绝。该实验沿用H3复制残差、相同索引与调制表的前提；新输出由模型单独分配，独立微测必须传`--separate-output`。

`owner_kernel.py`进一步按原RCCL位置owner分工，主机预排输入指针为该owner的原累加顺序，去掉逐元素动态选择。`--channel-rotation 2`可按rank轮转channel访问顺序，改变地址同时热点而不改变每元素的累加顺序；A有约1.52%流水线微测收益，B/C不稳定，未达到采用门槛。快路径只支持已校准的16 channels、chunk524288、alignment1024，其余情况保留原通信加九齿门控回退。输出保留16×每通道长度的源布局，但逻辑迭代裁剪至每owner容量；当前九齿后端使用一维平坦网格，host发射`16*ceil(capacity/tile)`个块。所有channel、尾块和padding都受掩码保护。

`build-direct-bridge.py`区分普通/owner和目的卡轮转。几何变化可用`--reuse-runtime`复用主机库；它核对生成C++源码、transport及ABI完全相同，避免重复编译，不改变模型编译缓存策略。

## 无profiler结果

对照为当前已验收的两阶段peer路径，双方使用同一InfiniCCL communicator、相同数据、相同八线程桥；不是拿更慢的原RCCL路径作为性能分母。下表为流水线微测，其他长度栏依次为B/C实际消息长度。除8-wave的A为9组外，其余每个长度均为39组交替配对。

| 方案 | A中位数参考→候选 | A配对收益 | B / C配对收益 |
| --- | ---: | ---: | ---: |
| direct-1024-clean | 5.3549 → 5.4830 ms | -1.931% | -1.996% / -1.790% |
| screen-256 | 5.2739 → 5.3837 ms | -2.024% | -0.978% / -1.446% |
| screen-512 | 5.3115 → 5.4120 ms | -1.878% | -0.099% / -0.407% |
| screen-2048 | 5.2526 → 5.3757 ms | -2.248% | +1.051% / -0.502% |
| screen-1024-rotate | 5.3106 → 5.4278 ms | -2.117% | -0.363% / -1.742% |
| screen-1024-w8 | 5.4338 → 5.4584 ms | -0.437% | 未测 / 未测 |
| owner-clean | 5.4272 → 5.4848 ms | -0.910% | -2.567% / +0.504% |
| owner-channel2 | 5.4962 → 5.4333 ms | +1.520% | -1.814% / +0.948% |

三种实际count为148216320、123105024、129357312。全部八卡、全部有效元素及尾部哨兵通过。计时有小幅波动，微小正值不作为采用理由；这些独立算子结果不代表完整请求耗时，也不宣称所有直接发布方案均无优化空间。

## 数值与独立诊断

`--consumer-copy`在每轮执行前填入输出哨兵，并在发布后立即由同一GPU stream读取所有输出，逐轮比较，防止相同输入的旧结果掩盖事件/可见性问题。普通、NaN/Inf、极小和大幅值均通过；owner版本在三个实际长度也逐位通过。消费者检查的计时包含额外读取，只用于验证，不混入干净流水线计时。拒绝原地别名的负向测试通过。

CPU独立区间覆盖及原RCCL owner索引公式交叉核对覆盖三个实际长度、24577 token、4097 token和最小长度，证明区间完整且无重叠；它不代替GPU完整数值检查。

独立PMC的朴素直接发布kernel为60 VGPR、64 SGPR、0 LDS、0 scratch、0% bank conflict；`L2 cache write unit is stalled`约68.7%–80.6%，`L1 cache unit is stalled`约22.0%–26.7%。原始字段见[PMC TXT](../../docs/peer-direct-f13r1n16-basic-pmc.txt)。此工作量包括发布，不能仅与现用求和kernel的PMC作因果比较；现用独立发布kernel对照也已采集，`L2 cache write unit is stalled`约92.7%–93.6%，高于直接发布但整体反而更快，因此不按单项计数推导因果。[现用发布原始PMC](../../docs/peer-direct-current-publish-f13r1n16-pmc.txt)

复用已有时间线，独立微测的八卡求和起始偏差中位数为122.097µs，真实模型为22.9205µs。这里只定位微测与模型环境差异，不把profiler中的绝对时间当成绩。

## 复用

先激活正式DTK/GCC/venv，设置`H3_AUTOTUNE_CODE_ROOT`并独占八卡。所有输出目录使用新路径。

```bash
python build-direct-nt.py "$new_nt" --tile 1024 --warps 4
python build-direct-bridge.py "$new_nt" "$new_runtime"
python build-owner-nt.py "$new_owner_nt" --tile 1024 --warps 4
python build-direct-bridge.py "$new_owner_nt" "$new_owner_runtime" --owner-partition --rotate-outputs
python bench-direct.py "$INFINI_ROOT/lib/libinfiniccl.so" "$new_runtime/peer-gate-bridge.so" "$new_result"   --bridge-build "$new_runtime" --reference-peer-build "$current_peer"   --indexed-build "$current_indexed" --separate-output --gather write   --vectors 1 --blocks 5120 --pairs 39 --counts 148216320 123105024 129357312
```

数值压力检查追加`--consumer-copy --case random|special|tiny|wide`。必须检查JSON的`exact`，程序也会保留不通过的实验，不能只看退出码。

同进程带标记trace中，最后一组GPU0的现用求和+发布合计4.700ms，直接发布4.913ms；主机空档也随发射次数变化。该采集只定位构成，不能把带profiler的请求时间用于选版。

## 后续

下一步保留阶段分工：固定owner的九齿求和/门控写紧凑partial，再由独立发布kernel执行通信。对“保留全部reduced等待”和“独立输出下移除该全卡等待”分别计时；始终保留所有输入就绪和发布完成依赖。输出仍不得与输入/残差别名，避免任一卡发布覆盖其他卡尚未读取的数据。

实现约束：紧凑partial采用`(16, capacity)`布局，每rank预留至少`N/8+16*1024`个BF16元素，shape1/stride0为capacity；源输入仍为全量N。发布kernel可采用`(320,16)`网格，以channel标量和chunk位移恢复原全量输出位置，减少逐线程动态除法。对同一stream上的NT→publish保留自然顺序，最后仍记录八卡发布事件并让所有消费stream等待。先以不变的原BF16次序验证，再判断净收益。未实现或未实测的方向不能视为收益。

[完整原始记录](../../docs/peer-direct-f13r1n16.json)

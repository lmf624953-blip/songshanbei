# 保持 BF16 累加顺序的 peer 分片与门控融合实验

2026-11-28，`f13r1n16` 上九齿分片门控已通过普通官方默认A、同实例完整A×8/B/C/返回A、真实八卡1/25/49逐位、稳定重建目录装载、独立PMC/trace和Tree回退检查。正式完整A从128.436449降到120.698043秒，下降6.025086%；Transformer从107.217758降到101.138358秒，下降5.670143%。所有成片与此前对应验收版逐字节一致，默认配置已切换至`validated-build/peer-gate-nt-f13r1n16`。 [完整报告](../../docs/分片门控融合-f13r1n16.md) · [原始记录](../../docs/peer-gate-nt-f13r1n16.json)

先前`peer_reduce.hip`是可行性研究原型，39组配对的6.370→4.920ms及5.59%完整请求粗略投影仅适用于该HIP原型。正式候选的九齿tile1024版本，独立算子6.358→5.231ms；接入实际InfiniCCL、八个主机worker与动态校准后，独立算子6.392→5.406ms。两者均逐位通过，数值与真实模型回放分开记录，不把HIP原型成绩当九齿版本成绩。

## 九齿计算与模型桥

`peer_nt_kernel.py`表达8份输入按原循环rank顺序累加、每步BF16 RNE、原indexed门控和残差；`nt_bitcast_compat.py`在构建进程内补齐当前九齿lowering的`bitcast=True`语义，保存前端和发射器来源哈希，不修改已安装包。`build-peer-nt.py`保存九齿表达、lowered源码、IR、HSACO和运行时ABI。tile256/1024/2048已比较，选1024、4 warps、0 LDS。

`peer_transport.hip`是输入就绪依赖、分片发布与完成依赖；`peer_gate_bridge.cc`在八个原有`RankWorker`上汇合后调度九齿镜像，保留原设备和stream。每个新count/communicator用两组独立有限BF16探针校准实际InfiniCCL rank环、通道、块和尾部，并在所有位置及八卡验证；不通过则执行原AllReduce和现用九齿门控。探针和工作区不替换真实模型输入，不减少模型层数或投影计算。

`prepare-native-hook.py`给已验收H3源码添加回调。`build-peer-native.py`从该版本源码包重建私有overlay，`build-peer-bridge.py`固定镜像、桥和transport的哈希。`peer_gate_runtime.py`装载并提供实验对照开关。开关用于同实例消融和安全回退；本轮完整资格验收已通过，`selected-best.env`已选择稳定发布目录。

在已经激活正式DTK/GCC/venv、设置`H3_AUTOTUNE_CODE_ROOT`后，以全新目录构建：

```bash
here="$H3_AUTOTUNE_CODE_ROOT/optimization/dcu-kernel-autotune/candidates/rccl-sequence-parallel"
# 统一入口：冻结前序配置并依次构建设备镜像、通信桥和native。
# 若使用私有源码版xmake，先设置XMAKE_PROGRAM_DIR为其Lua脚本目录。
bash "$here/build-peer-release.sh" "$accepted_config" "$new_release" "$xmake_binary"
```

统一入口已在新目录完整执行，构建退出码0；生成的native、主机桥和transport逐字节一致，九齿镜像加载区段及映射一致。`reference.env`保存求值后的前序配置，不依赖以后可能改变的`selected-best.env`。`candidate.env`是新发布目录的候选入口；`release.json`初始`qualified=false`，仍须完整模型验收后才能选为正式版本。

新发布目录接入现有统一复测命令，普通入口和常驻性能口径保持分开：

```bash
bash "$H3_AUTOTUNE_CODE_ROOT/optimization/reproduce-public-a.sh" resident \
  "$new_release" "$model" "$new_benchmark_output"
bash "$H3_AUTOTUNE_CODE_ROOT/optimization/reproduce-public-a.sh" single \
  "$new_release" "$model" "$new_single_output"
```

`resident`只运行8个完整A，顺序为`off,on,off,on,on,off,off,on`，排除双方各一次预热，读取`benchmark.json`的`wall.candidate.mean`；它不代替B/C资格验收。`single`使用新进程运行一次完整A，保留模型加载、首次生成和进程总墙钟。`peer-image-identity.json`固定设备加载内容和主机库身份，允许HSACO调试路径变化；复测前会核对原生库、九齿镜像及通信桥。`report-peer-benchmark.py`也支持直接汇总完整11请求的前8条，避免重复生成A。新汇总入口已直接复用本轮11请求的前8条，端到端及全部阶段统计与完整验收完全一致。

需要分步研究时可使用以下命令；不要把统一入口与分步入口的输出指向同一目录：

```bash
python "$here/build-peer-nt.py" "$new_nt" --tile 1024 --warps 4
python "$here/build-peer-bridge.py" "$new_nt" "$new_peer_runtime"
# xmake也可使用源码附带工具的私有构建，并设置其XMAKE_PROGRAM_DIR。
python "$here/build-peer-native.py" "$accepted_native" "$new_native" --xmake "$xmake_binary" --jobs 8
python "$here/bench-peer-reduce.py" "$INFINI_ROOT/lib/libinfiniccl.so" \
  "$new_peer_runtime/peer-gate-bridge.so" "$new_micro_json" \
  --bridge-build "$new_peer_runtime" --indexed-build "$current_indexed_build" \
  --gather write --vectors 1 --blocks 5120 --pairs 39 --counts 148216320
```

候选配置先加载当前已验收配置，再覆盖`H3_ADALN_BUILD`和`H3_PEER_GATE_BUILD`为新构建，其他计算保持不变。`qualify-peer-gate.sh CANDIDATE_CONFIG NEW_OUTPUT ACCEPTED_QUALIFICATION`先验证普通官方默认入口，再执行11个完整请求，对照顺序`off,on,off,on,on,off,off,on,on,on,on`，排除前两个预热，正式配对为`(2,3)/(5,4)/(6,7)`，记录整卡显存。`summarize-peer-gate.py`核对49次Transformer、所有门控/投影计数、实际库和镜像哈希、独立参考质量，以及A/B/C成片与已有验收版逐字节相同。主字段仍为`wall.candidate.mean`，协议名区分为`minimax-h3-public-a-peer-gate-paired-v1`；新成绩必须来自此完整测量。

CPU独立重建已确认：九齿HSACO因调试路径造成整文件哈希差异，可加载区段及映射完全一致；主机桥和transport逐字节一致。InfiniLM的130个编译对象全部逐字节一致，xmake的遍历顺序会改变链接布局，因此保存`native-link-order.json`并用`link-peer-native.py`逐对象校验后按固定顺序链接，最终native也逐字节一致。构建脚本已调用这一固定链接步骤，拒绝把未知编译对象宣称为同一构建。

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

`ring_order_map.py`和HIP原型源代码中的rank环描述该原型在本节点验证的条件，不能直接将该实验库替换正式通信库。当前九齿模型桥已经增加运行时校准及回退，仍需完整请求资格验收。[模型接入状态与约束](模型接入方案.md) · [HIP原型完整记录](../../docs/peer-reduce-f13r1n16.json) · [HIP原型原始PMC TXT](../../docs/peer-reduce-f13r1n16-basic-pmc.txt)

独立九齿PMC与时间线使用`profile-peer-gate.sh CONFIG NEW_OUTPUT`，在完整请求计时结束后单独运行；脚本保留原始`pmc.txt`、`trace_stream.json`及对应数值结果。该九齿采集已通过，原始TXT为`docs/peer-gate-nt-f13r1n16-basic-pmc.txt`，覆盖八卡各三次；不使用先前HIP原型的PMC代替。

# 模型级行分片归一化候选

本目录在独立原型基础上接入真实 H3 Transformer。当前状态：独立 Tensor 桥闸门、回退、异常唤醒、独立重建、真实八卡第 1/25/49 次去噪、普通官方默认 A 及完整 A×8/B/C/返回 A 均已通过。正式完整 A 配对 **122.526417 → 116.975680 秒，下降 4.530237%**，Transformer **101.170642 → 97.202297 秒**。全部成片与对应验收版逐字节一致。**稳定发布目录实际加载及独立采样正在执行，正式默认尚未改变**。[完整资格报告](../../docs/行分片归一化模型-f13r1n16.md) · [原始资格 JSON](../../docs/peer-sharded-norm-model-f13r1n16-qualified.json)

## 模型边界

`model_bridge.cc` 的 gate 回调返回行分片 hidden，norm 回调返回各 rank 完整有效的归一化输出，分别注册到原 native 的两个调用位置。末层 FFN 后、`norm_out` 之前显式物化完整 hidden。块内 norm 不支持快路径时，也先物化再走原计算。每次正常 forward 退出均不保留 pending hidden；异常时永久标记失败并唤醒等待 rank，保留在途 Tensor 生命周期，拒绝继续使用该进程状态。

当前分片通过 Tensor 的持有引用、数据地址、元素数、rank、communicator、stream 和阶段状态匹配。归一化后，原 hidden 的本卡有效行继续用于下一次门控残差。任何不同身份的残差、阶段不一致或不合法的关闭切换都会拒绝。

首次经过每个 count/communicator/weight 组合时，完整比较八卡 RMSNorm weight、逻辑 scale/shift 及 indices 的字节。检查同步放在预热内；复制前提不满足时，物化 hidden 并执行各 rank 的完整 norm。缓存持有 weight Tensor 引用，防止已释放参数地址被复用；调制表和索引的持续复制依赖当前 H3 源码的 AllGather 和相同输入不变量，不宣称支持任意 rank 独立调制表。

## 对照与计时口径

配置同时加载已验收 peer 桥和模型候选桥。验证中的 `off` 表示已验收 peer，`on` 表示行分片归一化；两侧使用同一个新 native。关闭候选时直接注册原已验收回调，保留原计算。`model_runtime.py` 把比较标识记为 `accepted-peer-vs-sharded-norm`，避免将旧 AllReduce 对照误当作本轮参考。

完整请求双方应各有 39200 个 rank 门控回调、4900 个 peer 批次。开启时应有 4851 个分片 norm、49 次末层物化及 392 个首层完整 norm 回调；关闭时上述分片计数应为零。两侧保留全部注意力、GEMM、归约、SwiGLU 和 49 次 Transformer。

`qualify.sh` 先执行不覆盖 steps/frames/seed/height/width 的官方默认 A，检查成片与已验收版本一致，再进行同实例 A×8/B/C/返回 A。稳态顺序为 `off,on,off,on,on,off,off,on,on,on,on`，双方各一次预热，正式配对为 `(2,3),(5,4),(6,7)`。主指标仍为 `benchmark.json` 的 `wall.candidate.mean`，包含 MP4 写出。普通入口的进程加载时间单独报告；未启用编译缓存优化。

## 构建与独立闸门

环境沿用运行 skill。`reference` 是已验收 peer 发布目录；所有新构建目录必须不存在。

```bash
python build-native.py "$reference/native" "$new_native" --xmake "$xmake_binary"
python ../peer-sharded-norm/build-row-gate.py "$new_nt" --tile 1024 --warps 4
python ../peer-sharded-norm/build.py "$new_nt" "$new_prototype" --split-arrivals
python build-bridge.py "$new_prototype" "$new_model_bridge"
```

native 从冻结的已应用补丁源码包构建，只有 H3 Transformer 对象变化，其余 129 个对象保持一致。`prepare-native.py` 拒绝未知的参考源码 SHA，新增的物化/失败回调为空时保持原路径。主机桥复用已验证行分片九齿镜像，计算仍由九齿执行，HIP 仅用于发布与依赖。

`check-model.py` 在八个持久 worker 上验证真实 InfiniCore Tensor 接口。它执行开启/关闭/开启，每种连续三次残差链，检查完整 norm 输出、局部 hidden、未拥有区域、尾部和最终物化。`--mismatch-weight` 验证复制前提不满足时的物化回退；`--test-abort` 只让七个 rank 进入 gate，再中断，验证等待唤醒与失败后拒绝复用。Tree 回退另开进程设置 `NCCL_ALGO=Tree`。

## 真实八卡回放

候选配置在 `reference/candidate.env` 基础上仅覆盖：

```bash
export H3_ADALN_BUILD=/absolute/new_native
export H3_PEER_SHARDED_NORM_BUILD=/absolute/new_model_bridge
```

通过统一入口运行：

```bash
bash optimization/run-optimized.sh --config "$candidate_config" --replay-run \
    --model "$model" --captures "$captures" --out "$new_replay" \
    --steps 1,25,49 --iterations 5 --peer-gate-ablation
```

当前回放的每组包含一次独立预热和五次正式 Transformer 调用。三步中位数为 2.064661→1.982741、2.064398→1.983129、2.064086→1.981949 秒，合计降幅 3.961250%。输出原始字节 SHA 同时与同实例参考及此前验收版匹配。此前完整 A 投影约 3.319847%；后续完整请求已经实测下降 4.530237%，仅 Transformer 差额折算为参考完整请求的 3.238767%。两类指标分别保留，不将投影当作正式成绩。

`summarize-replay.py` 核对实际 native/设备镜像、原始哈希和数学调用计数。捕获文件的旧版输出仅用于诊断，回放质量结论使用当前同实例及已验收 peer 输出。

## 统一构建、复测与发布

从仓库根目录可统一构建本候选；`accepted_peer_config` 必须是行分片归一化之前、已验收的 peer 配置。`build` 子命令仅构建更早的注意力前序，不能用它代替下面的 `build-sharded`。

```bash
bash optimization/reproduce-public-a.sh build-sharded \
  "$accepted_peer_config" "$new_work" "$xmake_binary"
```

若使用私有源码版 xmake，预先设置 `XMAKE_PROGRAM_DIR` 为配套 Lua 脚本目录。此入口从冻结前序源码构建 native、九齿行分片镜像、发布 transport 和模型桥，把所有活动运行组件复制到 `new_work/release`，核对加载区段及固定主机库身份。构建不自动通过完整资格，也不改默认配置。

```bash
bash optimization/reproduce-public-a.sh resident "$release" "$model" "$new_output"
bash optimization/reproduce-public-a.sh single "$release" "$model" "$other_new_output"
```

两种入口均识别 `h3-peer-sharded-norm-v1`。`resident` 生成八次完整 A，汇总三组配对；`single` 在新进程运行一次 A，另记加载与首次生成的进程总墙钟。完整 11 请求已经完成时，也可以直接运行 `report-benchmark.py QUALIFICATION/steady-paired NEW_JSON`，汇总相同的前八条记录，避免重跑。

`package-runtime.py NEW_RELEASE --qualification QUALIFICATION` 只接受已正常完成、质量一致且完整 A 配对均值收益至少 3% 的资格记录。所有活动 `H3_*_BUILD` 运行组件复制到发布目录，native overlay 路径随之更新；原实验路径仅保留为来源记录。发布仍需验证实际加载，单独切换 `selected-best.env`。运行仍依赖账号中已准备的 `runtime`、权重及同版本 DTK，不宣称为独立容器。

## 独立采样

正式无 profiler 计时结束后，以新目录分别运行，不能重叠：

```bash
bash profile-model.sh trace "$candidate_config" "$new_trace"
bash profile-model.sh pmc-gate "$candidate_config" "$new_gate_pmc"
bash profile-model.sh pmc-norm "$candidate_config" "$new_norm_pmc"
```

`trace` 在权重加载和一次预热后，仅采集真实模型第 25 次去噪的一次 Transformer；停止前两秒用于排空 profiler，不能算入计算时间。输出 `hipprof_stream.json` 与回放数值结果。

两项 PMC 使用相同主机桥和设备镜像，在独立八卡 Tensor 桥检查中采集官方 A 长度，保留原始 `pmc.txt`；这不是完整模型 PMC。检查仍按开启/关闭/开启运行，各三次残差链，每步执行当前 peer 参考；使用 `--pmc-off` 配合 `--profile-candidate-once`，只在首组开启候选的第三次 gate/norm 期间启动采集，避免参考同名 kernel 混入。每份 TXT 应仅有八卡各一次、共八条目标 kernel 记录；必须核对实际计数，不能仅以文件非空判断通过。profiler 耗时不作为正式模型性能。

`summarize-profile.py MODE OUTPUT` 在采集正常退出后验证：真实 trace 的两个输出原始 SHA、预热加正式调用数、ROI 内八卡各 100 次 gate/publish/norm、末层物化状态、实际加载版本；PMC 的八条目标记录、逐 rank 行分片网格、寄存器、LDS/scratch 及完整独立输出检查。输出 `validation-summary.json`，保留所有原始字段。两类诊断不能代替无 profiler 的完整请求配对。

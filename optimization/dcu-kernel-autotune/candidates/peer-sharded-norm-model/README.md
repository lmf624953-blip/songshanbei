# 模型级行分片归一化候选

本目录在独立原型基础上接入真实 H3 Transformer。当前状态：独立 Tensor 桥闸门、回退、异常唤醒、独立重建及真实八卡第 1/25/49 次去噪已通过，**完整官方入口与稳态资格尚待验证，正式默认未改变**。已验收最佳仍为 `peer-gate-nt-f13r1n16`，已有官方 A 稳态完整请求均值 120.698043 秒。

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

当前回放的每组包含一次独立预热和五次正式 Transformer 调用。三步中位数为 2.064661→1.982741、2.064398→1.983129、2.064086→1.981949 秒，合计降幅 3.961250%。输出原始字节 SHA 同时与同实例参考及此前验收版匹配。完整 A 投影约 3.319847%，仍需完整请求实测证明。

`summarize-replay.py` 核对实际 native/设备镜像、原始哈希和数学调用计数。捕获文件的旧版输出仅用于诊断，回放质量结论使用当前同实例及已验收 peer 输出。

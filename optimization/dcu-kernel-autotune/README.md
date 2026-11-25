# MiniMax-H3 长注意力自动调优

公共轮次驱动来自 `MD/matpl/dcu-kernel-autotune` 的工作树，基准提交为 `667b45b`。三个公共脚本的复制前 SHA-256 见 [framework.sha256](docs/framework.sha256)。本地适配支持 DTK 26.04 的 PMC CSV，并修正报告在 PMC 匹配数为零时的退出问题。源码、轮次和结果均为本次 H3 调优所用。

## 运行

当前本机工作区已初始化，可直接复用已有 baseline 和轮次。新克隆的仓库中先准备 `runtime`、完成 v3 构建，再初始化一次隔离源码。

```bash
export H3_AUTOTUNE_BASE_CONFIG=/absolute/path/to/v3-build/candidate.env
bash adapters/init-h3-source.sh
```

初始化只复制 `templates/h3-source` 中冻结的 v3 AOT 适配源码并创建快照，拒绝覆盖已有工作区。仓库本身既可位于项目根目录，也可使用本机的 `submission-xsh` 工作树布局。模型路径可用 `H3_AUTOTUNE_MODEL` 指定，真实去噪输入目录可用 `H3_AUTOTUNE_CAPTURES` 指定。

```bash
cd /public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune
export PROFILE=h3-long-attention
./scripts/autotune.sh check
./scripts/autotune.sh status
./scripts/autotune.sh baseline
./scripts/autotune.sh new --name "调整 AOT wave 并发约束"
# 新轮次的编号由上一步输出，这里以 round_001 为例。
printf 'WAVES_PER_EU=2\nSCHEDULE_HINT=none\n' > rounds/h3-long-attention/round_001/source_delta/flags.conf
./scripts/autotune.sh run round_001
```

- `baseline` 复制当前 `selected-best.env` 指向的长注意力设备产物到独立安装目录，并记录哈希。候选在同一进程中与这份快照通过相同的 native launcher 比较。
- `work/source` 是隔离源码树。每轮的候选作用于该树，框架在实验后恢复快照。H3 正式代码在 `submission-xsh`。默认用当前节点第一张可见卡，可按需设置 `H3_AUTOTUNE_DEVICE`。
- 无 profiler 计时使用 GPU event，覆盖 27570 与 22016 token、每卡 7 头、head dim 128、BF16 和生产的 packed stride。每个形状有 11 组交替顺序的 v3/候选配对样本；百分比使用每对样本的耗时节省比例的中位数。表中的毫秒来自单算子回放。
- 普通和放大 Q/K 的输入分别检查有限值及 BF16 位模式。通过正确性后框架分别采集 hipprof trace 和目标 kernel 的 PMC。每轮保存源码改动、设备对象、编译 IR、数值和计时日志。
- 单算子收益明确的候选进入真实八卡 Transformer 回放，再按公开 A 的 768×928、50 steps、124 帧和 seed 42 做完整生成与官方质量验证。

框架状态 `PASS` 表示通过数值校验并执行剖析。每种长度的实际收益和硬件资源见 按长度对照（本机 `reports/h3-long-attention/按长度对照.md`），用 `python3 scripts/report-h3-attention.py` 更新。也可查看各轮的 调优目录（本机 `rounds/h3-long-attention/`） 和 框架汇总（本机 `reports/h3-long-attention/summary.md`）。

## 当前进展

第 1–10 轮排除了强制增加 wave 并发、调度提示、尾部 mask 分支、提前归约 LSE、Q 类型转换、PV 转置及单次累加物化 Q 等写法。其中第 9 轮数值失败，框架拦截后没有执行 trace/PMC；其余轮次数值通过但没有可靠收益。

第 11 轮在九齿生成代码上做循环不变加载提升，将 Q 的读取和布局转换移出 KV 循环。27570、22016 token 的单算子配对计时分别节省约 7.95%、7.96%，普通与放大 Q/K 输入均逐位一致，Scratch 为 0。27570 token 的全局读取指令减少 33.29%，LDS 指令减少 7.68%，bank conflict 原始计数没有减少。完整记录见 第 11 轮（本机 `rounds/h3-long-attention/round_011/report.md`）。

真实八卡 Transformer 回放使用以下入口，输出到尚不存在的 `validation/round_011/`。它在同一个模型实例中比较正式 v3 与候选，保存第 1、25、49 次真实去噪输入的输出哈希及每侧三次计时。比较输出哈希时按相同 step 和 video/audio 分别匹配；`against_capture` 是与历史采集输出比较，不能直接当作候选相对于本轮 v3 的误差。

```bash
bash adapters/replay-h3-round.sh round_011
python3 adapters/summarize-h3-replay.py validation/round_011
```

第 11 轮的真实八卡回放已通过，第 1、25、49 次 Transformer 调用的 video/audio 输出均与同轮 v3 哈希一致，三个中位数合计节省 5.80%。见 八卡对照（本机 `validation/round_011/Transformer回放.md`）。

第 14 轮叠加 PV 的 LDS swizzle，单算子降至 49.956 / 32.150 ms，普通和放大 Q/K 输入仍逐位一致。PMC 的 LDS bank conflict 原始计数减少 75%，184 VGPR、8192 B LDS、0 B Scratch。该轮是单算子候选，后续拆分 P/V 布局并测量不同向量宽度，最终候选重新执行八卡回放。

第 12 轮的 `num_stages=1, local prefetch=1` 组合在编译阶段被拒绝，不属于运行故障；第 13 轮是实验 AOT 适配器遗漏 TTIR 导出回退，修正在第 14 轮。框架已修复构建失败返回和退出时的源码恢复，并用构建命令失败、命令成功但产物缺失两种情况验证，见 验证记录（本机 `reports/构建失败恢复验证.json`）。正式 `selected-best.env` 仅在完整规格、质量和正式测速通过后更新。

第 18–20 轮的计时接近，选择 VGPR 最低的第 19 轮进行集成。它使用 P 的 `vec=8, perPhase=2, maxPhase=4` 和 V 的 `vec=8, perPhase=1, maxPhase=16`。27570 / 22016 token 分别约为 48.997 / 31.394 ms，同卡配对节省约 19%，LDS bank conflict 原始计数减少 93.75%。实际资源为 180 VGPR、8192 B LDS、0 B Scratch。第 21 轮将 Q 读取提升与双阶段 local prefetch 组合，耗时增至 91.858 / 58.807 ms，已淘汰。

第 19 轮已集成到 `submission-xsh/optimization/adaln/attention_codegen.py`，由 `build-attention-aot.py --optimize-long-attention` 显式启用。集成源码的主机库及设备可加载区段与实验产物一致，关闭新开关与正式 v3 一致，15 个分块边界检查全部逐位一致。加入全局写回及原子操作检查后再次重建，设备产物仍一致。八卡真实早、中、晚期回放的 video/audio 输出全部与同轮 v3 哈希一致，Transformer 三个中位数合计节省 **12.81%**，见 真实回放（本机 `validation/round_019/Transformer回放.md`）。完整公开 A 的普通入口验收保存在 `validation/round_019-official/`。

## 复现已测候选

`candidates/long-attention-lds/source_delta` 保存第 19 轮的完整增量源码。创建新轮次后，将该目录内容复制到实际输出的轮次编号下即可复测。以下编号仅用于说明，新克隆中通常为 `round_001`，已有工作区使用 `new` 返回的编号。

```bash
export PROFILE=h3-long-attention
./scripts/autotune.sh new --name "Q 读取提升与 PV LDS swizzle"
cp -a candidates/long-attention-lds/source_delta/. rounds/h3-long-attention/round_001/source_delta/
./scripts/autotune.sh run round_001
```

框架源码与精简报告可以进入仓库。`work`、`source_snapshots`、`rounds`、`validation`、`validated-build`、`reports` 由本机实验生成；原始日志、trace、PMC 和设备产物留在运行工作区，避免把运行数据随源码反复复制。

## 分块后续搜索

第 22 轮 Q32/K32/4 warps 比 v3 慢约 22%。第 23 轮 Q64/K32/4 warps 降到 **42.953 / 27.977 ms**，普通和放大 Q/K 仍逐位一致，保留为新单算子候选，真实八卡回放待完成。第 24 轮 Q128/K32/8 warps 出现 116 B Scratch，耗时回退至 104.228 / 66.214 ms，已淘汰。

`adapters/stage-h3-weights.py SOURCE /tmp/新目录 --manifest work/weights.json` 可将完整模型复制到节点本地盘并逐文件核对 SHA-256，拒绝覆盖已有目标且检查剩余空间。只用于减少后续加载等待，正式推理参数及计时口径不受此步骤影响。

仓库内冻结实测与证明见 [长注意力实测](docs/H3长注意力实测.md) 和 [按长度对照](docs/按长度对照.md)。

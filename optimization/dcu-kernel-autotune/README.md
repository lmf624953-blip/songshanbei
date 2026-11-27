# MiniMax-H3 长注意力自动调优

2026-11-27 当前完整验收版为 stable-max guarded / 成对 P 算术组合，官方 A 常驻预热后三次均值为 **127.831001 秒**，同轮基准 132.556449 秒。普通官方 A 及完整 11 请求均通过；用户已恢复持续优化；QK拆分、两种中间布局和融合预取对照已完成，见 [本轮报告](docs/qk-split-f13r1n16.json)。当前复测使用 [统一入口](../reproduce-public-a.sh) 和 [计时说明](../公开A耗时复测说明.md)，取 `wall.candidate.mean`。完整取舍见 [暂停分析](../本轮最优整合与暂停分析.md)。下文保留早期 v3 轮次和框架用法。

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

## 历史进展

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

## Q64 的真实回放与后续搜索

第 25 轮保留 P 的 `vec=8, perPhase=2, maxPhase=8`，VGPR 为 176，两个生产长度为 43.026 / 28.047 ms。真实八卡 Transformer 三个阶段中位数约由 4.49 s 降至 3.61 s，合计节省 19.60%，video/audio 哈希均与同轮 v3 一致。集成重建与实验产物的加载区段、ELF 映射和主机库一致，15 个边界长度全部通过。普通公开 A 已通过本地检查，SSIM 93.832013%，成片哈希与已验收 v3 一致；正式默认配置在稳态和多用例回归前尚未切换。

冻结增量位于 `candidates/long-attention-q64/source_delta`。完整实验目录在本机 `rounds/h3-long-attention`；第 26–28 轮测试了提前 LSE、提前 V 和 Q128，均未胜出。第 29 轮 MFMA32 未进入当前 gfx936 的预期 BF16 路径，被布局检查拒绝；第 30–31 轮扩大 PV kWidth 或修改 MFMA warp 分布造成浮点差异且更慢，在数值检查后停止。第 32–36 轮继续检查流水线、kpack、调度及 QK 的 LDS 暂存。

## 构建边界与重复实验

H3 候选构建默认限制 180 s，使用 `H3_AUTOTUNE_BUILD_TIMEOUT_SECONDS` 调整。第 34 轮发现当前 DTK 的 IGLP 调度组构造持续增长，保存调用栈后终止，未形成 GPU 性能结论；IGLP1 的第 38 轮也在 60 s 时限停止。第 39 轮去掉 IGLP 后与第 25 轮的加载区段、映射和主机库完全相同，标记 `equivalent_binary` 并跳过 GPU 重测。框架汇总同时保留这些状态。

第 37 轮修正双阶段预取的 P/V 别名后数值一致，但增加到 208 VGPR、20480 B LDS，耗时约 50.272 / 32.766 ms，无额外收益。第 40 轮继续检查完整 KV 块与尾块分离。正式候选仍为已通过完整公开 A 的 Q64 第 25 轮。

第 40 轮的直接配对确认相对第 25 轮再节省 1.89% / 2.52%，15 个边界长度通过。真实八卡三时刻输出与同轮 v3 哈希一致，Transformer 中位数合计节省 **20.51%**。尚需本轮的正式集成、完整质量、多用例与稳态验证；不能套用第 25 轮完整 A 的成片证明。

仓库内冻结证据见 [实测报告](docs/H3长注意力实测.md) 与 [逐长度对照](docs/按长度对照.md)。


## 完整稳态与第41–50轮

第40轮已完成11次完整请求及独立参考比较。相同节点、同一实例仅切换长注意力，双方各一次预热后，三组完整A配对：端到端240.533937→195.138460秒（下降18.872795%），完整Transformer220.292703→175.072892秒（下降20.527149%）。A/B/C成片均与对应已验收v3哈希相同，独立参考SSIM分别93.832013% / 99.404123% / 99.630047%；切换后返回A仍一致。资源采样属于同一实例，不能推断独立实例显存节省。见 [完整稳态证据](docs/第40轮完整稳态验收.md)、[原始样本摘要](docs/round40-steady-final.json)。

第41轮在完整KV分块中省去可证明冗余的K/V加载上界，两个生产长度40.879/26.625ms，与第40轮直接11组配对又节省3.1847% / 2.7239%，普通和放大输入逐位一致。正式可选开关 `--elide-full-kv-load-mask` 必须配合 `--optimize-long-attention --split-kv-tail`；全量重建为 `--long-attention-q64-full-load`。生成源码SHA、完整循环和加载数量守卫拒绝未知编译输出。重建加载区段、映射、主机库与实验产物一致，15个边界通过；更多动态长度及八卡/完整生成按实际完成状态单独报告。

第42–50轮记录在 [筛选原始样本](docs/round41-50-search.json)。循环展开和减少warps更慢；提前归约、K64产生浮点差异且更慢；buffer寻址数值一致并略快于第40轮，但慢于41，未来只作独立有字节范围限制的组合候选；异步预取触发当前LLVM编译器段错误。编译失败不作为GPU性能结论。

`adapters/check-h3-lengths.py` 接受任意正序列长度，默认4096、8192、16384、24577、27570、32768、65535，保留配对原始样本和普通/放大输入误差。`H3_AUTOTUNE_USE_ENTRY_DEFAULTS=1` 使普通入口检查省略steps/帧数/seed/尺寸覆盖参数，直接核对原入口默认值50/124/42并记录源码SHA。构建适配器仅清理私有实验源码的旧Python字节码，避免同秒同尺寸候选误用缓存。

## 2026-11-27 融合流水线与现用热点

用户提出的 QK 分阶段拆分已实测退化，新的无 spill 双 Q128 共享 K/V 也未胜出。融合内 guard 与 P 同步只有不足1%单 kernel收益，按完整请求3%门槛归档；正式最佳仍为已验收127.831秒。现用版本另做一次真实八卡诊断，注意力/AllReduce/GEMM约占42.8%/25.7%/20.4%，通信包含等待，诊断不是新成绩。 [可重建源码、对照及原始 PMC](docs/融合流水线与双Q复用-f13r1n16.md)

## 当前 GEMM 覆盖与集合顺序闸门

2026-11-27 在 `f13r1n16` 以实际Tensile目录审核短名称映射，补测26个旧目录外候选及16个主要RK1，共336形状组合，128个逐位一致；最快候选又做39组配对，完整A粗略收益投影仅约0.324%，不合入。八卡直接RS/AG及2/4/8分块在默认与Ring/Simple16下均不保持原BF16位结果，重复整段对照通过。独立rank链探针的106902采样组均匹配8类串行舍入链，尚未证明完整分片映射，继续研究保持原序的自定义分片。 [GEMM源码与用法](candidates/rocblas-coverage/README.md) · [集合源码与后续限制](candidates/rccl-sequence-parallel/README.md) · [完整原始样本](docs/gemm-coverage-f13r1n16.json)

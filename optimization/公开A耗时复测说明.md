# 官方公开 A 耗时复测说明

## 当前正式默认：行分片归一化 116.976 秒

2026-11-28 当前本地默认为 `peer-sharded-norm-f13r1n16`。同一八卡 BW1000 实例的三组完整官方 A 为 **122.526417 → 116.975680 秒，下降 4.530237%**；Transformer 为 **101.170642 → 97.202297 秒，下降 3.922428%**。普通默认 A、A×8/B/C/返回 A、质量、独立重建、实际稳定目录加载、真实模型时间线和独立模型桥 PMC 均通过。全部成片与对应验收版逐字节一致。[完整报告](dcu-kernel-autotune/docs/行分片归一化模型-f13r1n16.md) · [原始资格](dcu-kernel-autotune/docs/peer-sharded-norm-model-f13r1n16-qualified.json) · [选版证明](dcu-kernel-autotune/docs/peer-sharded-norm-model-f13r1n16-selection.json)

```bash
repo=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
release=/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune/validated-build/peer-sharded-norm-f13r1n16
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
# 先定义两个尚不存在的绝对输出目录；两条命令串行执行。
bash "$repo/optimization/reproduce-public-a.sh" resident "$release" "$model" "$new_resident_output"
bash "$repo/optimization/reproduce-public-a.sh" single "$release" "$model" "$new_single_output"
```

`resident` 模型只加载一次，顺序 `off,on,off,on,on,off,off,on`，排除双方各一次预热，正式配对 `(2,3)/(5,4)/(6,7)`。`off` 使用当前 peer 前序，`on` 使用行分片归一化。读取 `benchmark.json` 的 `wall.candidate.mean`；协议为 `minimax-h3-public-a-sharded-norm-paired-v1`，比较为 `accepted-peer-vs-sharded-norm`。它包含输入准备、完整生成、MP4 写出与同步。不能与前序版本的协议字段混用。

`single/process-time.txt` 单列新进程从启动到退出，包含环境、模型加载、首次生成与清理；本轮普通默认 A 为 **875.75 秒**，不是 116.975680 秒。官方 Python `total.end_to_end=823.827 s`，首次生成管线 182.571 s。VAE 阶段也有波动；仅 Transformer 差额折算为参考完整请求的 **3.238767%**，不将全部端到端改善归因于 kernel。

已有完整 11 请求时，可直接运行 `peer-sharded-norm-model/report-benchmark.py QUALIFICATION/steady-paired NEW_JSON` 汇总同一批前八条记录；已确认完整验收与统一复测的 `wall`、全部 `stages` 完全相等，无需为汇总重跑 A。正式输入仍为 768×928、50 steps、124 帧、seed 42、49 次 Transformer；未启用跨进程编译缓存优化。

新账号先准备相同 runtime 与权重，再构建下文注意力前序、用 peer 的 `build-peer-release.sh` 构建 peer 参考，最后 `reproduce-public-a.sh build-sharded ACCEPTED_PEER_CONFIG NEW_WORK XMAKE_BINARY`。详细构建与各入口见[模型 README](dcu-kernel-autotune/candidates/peer-sharded-norm-model/README.md)。构建不自动通过新环境资格，也不会修改默认。

## 前序 peer 版本：120.698 秒的记录与入口

2026-11-28 前序已验收版是九齿分片求和/门控发布版 `peer-gate-nt-f13r1n16`，在 `f13r1n16` 同实例三组完整 A 配对中为 **128.436449 → 120.698043 秒**，下降 **6.025086%**。协议为 `minimax-h3-public-a-peer-gate-paired-v1`，比较为 `selected-stable-paired-vs-peer-gate`，主字段仍为 `wall.candidate.mean`。阶段、质量及原始记录见[完整报告](dcu-kernel-autotune/docs/分片门控融合-f13r1n16.md)。

当前账号可以直接复用已经验收的发布目录，无需重建：

```bash
repo=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
release=/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune/validated-build/peer-gate-nt-f13r1n16
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
# 输出目录必须尚不存在。
bash "$repo/optimization/reproduce-public-a.sh" resident "$release" "$model" "$new_resident_output"
bash "$repo/optimization/reproduce-public-a.sh" single "$release" "$model" "$new_single_output"
```

两种模式分别启动任务，应串行执行。`resident` 是同一模型实例中的八个完整 A，排除双方各一次预热，读取后三组配对；`single` 是新进程从启动到退出，包含加载和首次生成。本轮后者为 **878.74 秒**，不能与 120.698043 秒直接比较。阶段存在嵌套，不能将 `generation.pipeline_total` 与其中的 Transformer、VAE 再相加。

新账号或新克隆需要先准备相同运行环境；当前统一脚本识别 peer 发布目录。源码重建步骤见[peer 构建说明](dcu-kernel-autotune/candidates/rccl-sequence-parallel/README.md)：使用其 `build-peer-release.sh REFERENCE_CONFIG NEW_RELEASE XMAKE_BINARY`，前序配置可以由下文的注意力构建流程生成。`reproduce-public-a.sh build` 仅生成下文的注意力前序，不会自动接入 peer 或行分片归一化，不能把其输出当作最新组合。

该 peer 版本现在作为行分片归一化的已验收参考保留；最新默认及新协议见本文首节。

## 历史注意力版本：127.831 秒的固定源码与复测记录

以下第 1–5 节完整保留 `f12r2n09` 注意力版本的原始构建、样本与协议。“本轮”“当前 fast”均指该历史版本。其对照和成绩字段不能套用至上节的 peer 或行分片归一化版本。

当前完整验收版为 **P 行 LDS + V 预排布 + 内部 P/ROR4 精简 + stable-max guarded / 成对 P 算术**。在 `f12r2n09` 的八卡 BW1000 上，模型常驻、双方各一次完整预热后三次完整 A 请求均值为 **127.831000722 秒**。同轮基准为上一版精简 softmax，均值 132.556449158 秒；下降 3.564857%。

统一入口：[reproduce-public-a.sh](reproduce-public-a.sh)。主协议固定为 `minimax-h3-public-a-resident-paired-v1`，当前比较为 `softmax-compact-vs-stable-paired`，主字段为 `wall.candidate.mean`。代码、输入和统计口径可以固定；节点负载、依赖、频率和写盘速度会影响绝对秒数。

## 1. 代码与环境

本轮默认源码构建固定在提交 `782a321940c01438c768f29b87e9c7ac7bf2a20b`。后续文档提交可包含它；第三方要复现本轮时固定该提交。

```bash
git clone --branch xsh-opt git@github.com:lmf624953-blip/songshanbei.git songshanbei-retest
cd songshanbei-retest
git checkout --detach 782a321940c01438c768f29b87e9c7ac7bf2a20b
repo=$(pwd)
git rev-parse HEAD
module load compiler/dtk/26.04 compiler/gcc/11.4.0
```

不要使用 shallow clone，原生构建需要历史基线 `45b2c3094d8d189e030f8984d7c07dead4db690a`。需要独占同节点八张 BW1000（gfx936）、CPython 3.11、海光 Torch 2.9、比赛离线依赖及相同模型权重。仓库不含权重和这些软件环境。新账号按 [准备运行环境](README.md#准备运行环境) 建立包含注意力主机桥的 `runtime`。当前账号的新克隆可复用：

```bash
ln -s /public/home/acvduipmem/xsh/songshanbei/runtime "$repo/runtime"
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
test -x "$repo/runtime/.venv/bin/python"
test -r "$repo/runtime/artifacts/attention-bridge-build.json"
test -r "$model/model_index.json"
hy-smi
```

双方使用同一权重目录，不在配对之间清页缓存或更换权重位置。权重一致性由交付方核对，脚本不自动哈希全部权重。使用干净 Bash，同批八卡只运行这一项性能任务，不附加 hipprof/PMC、张量抓取或有限值诊断。

## 2. 一次构建，按固定协议计时

```bash
mkdir -p "$repo/artifacts"
job=$(mktemp -d "$repo/artifacts/public-a-retest-XXXXXXXX")
bash "$repo/optimization/reproduce-public-a.sh" build "$job/build-work"
release="$job/build-work/release"
bash "$repo/optimization/reproduce-public-a.sh" resident \
  "$release" "$model" "$job/resident"
```

输出目录须尚不存在。`build` 依次生成 Q128/K32 packed 基础版、求和交错版、P 行布局/V 预排布版、内部 P/ROR4 精简版（`build-work/softmax`），最后通过 [build-stable-paired-release.sh](dcu-kernel-autotune/candidates/pv-v-lds-layout/build-stable-paired-release.sh) 生成当前正式组合（`build-work/release`）。构建不运行模型、不计入成绩；已有对应发布目录时可直接复用，无需每次重建。

发布目录记录配置、产物及来源哈希。`qualified=false` 表示单纯重建不代替新环境验收。脚本不依赖本机 `selected-best.env`，不会自动采用未验收的 max3/min3 等实验。

`resident` 等价于 `bash optimization/benchmark-public-a.sh RELEASE MODEL NEW_OUT`，两者选一条。模型只加载一次，依次生成八次完整 A：

| 请求编号 | 注意力版本 | 统计用途 |
| --- | --- | --- |
| 00 | base，精简 softmax | 完整预热，排除 |
| 01 | fast，当前 stable/paired 组合 | 完整预热，排除 |
| 02 / 03 | base / fast | 正式配对 1 |
| 04 / 05 | fast / base | 正式配对 2，反转顺序 |
| 06 / 07 | base / fast | 正式配对 3 |

正式样本是 base 的 02/05/06 和 fast 的 03/04/07，不能选最小值、混入预热或将八次总耗时除以八。

固定官方小狗首帧、`A dog falls asleep`、768×928、50 steps、124 帧、seed 42、49 次 Transformer。首帧 SHA-256 为 `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。显式设置八卡、`OMP_NUM_THREADS=4`、编码器 TP=4、初始容量 512、VAE 设备 0/1及确定性首帧编码，每次请求重新执行完整模型计算。

`wall_seconds` 在 [run-sequence.py](run-sequence.py) 中用 `time.perf_counter()`，从输入准备前到 MP4 写出及设备同步后结束。包含条件编码、首帧 VAE、完整去噪、音视频解码和媒体写出。模型加载在循环前；媒体规格验证在本次计时后。`generation.pipeline_total` 包含内部组件，不与 Transformer/VAE 分项重复相加。

每个进程创建新的九齿/Triton 缓存目录，预热后在同一进程内自然复用；未启用跨进程 `H3_BENCH_JIT_CACHE` 优化。脚本核对冻结设备指令、主机库、回退镜像、实际运行快照、输入、49 次调用及视频规格，并要求八条 MP4 逐字节一致。失败退出，不把不完整记录当作成绩；性能协议不代替独立参考质量验收。

## 3. 唯一主字段与本轮样本

```bash
python3 - "$job/resident/benchmark.json" <<'RESULT'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["protocol"] == "minimax-h3-public-a-resident-paired-v1"
assert r["comparison"] == "softmax-compact-vs-stable-paired"
w = r["wall"]
for name in ("reference", "candidate"):
    print(name, "samples =", w[name]["samples"])
    print(name, "mean =", w[name]["mean"], "s; stdev =", w[name]["stdev"], "s")
print("time reduction =", w["mean_time_reduction_percent"], "%")
print("paired reductions =", w["paired_reduction_percent"]["samples"])
print("node =", r["runtime"]["hostname"])
print("Transformer =", r["stages"]["compute.h3_transformer"])
RESULT
```

| 指标 | 本轮 base | 当前 fast |
| --- | ---: | ---: |
| 样本 1 | 133.002252798 s | 127.596812735 s |
| 样本 2 | 131.826594966 s | 127.413284532 s |
| 样本 3 | 132.840499711 s | 128.482904901 s |
| 完整请求均值 | 132.556449158 s | **127.831000722 s** |
| 样本标准差 | 0.637226 s | 0.571975 s |
| Transformer 合计均值 | 111.882041016 s | 107.287698452 s |

完整请求下降 3.564857%，三组分别下降 4.064172%、3.347815%、3.280321%；Transformer 下降 4.106416%。阶段波动也保留：视频 VAE 14.319830→14.104838 秒，条件编码 0.310535→0.480583 秒，媒体编码 1.560515→1.506217 秒。跨节点先核对同节点配对收益，再看绝对秒数。[原始成绩](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09-public-a.json)

## 4. 普通入口首次启动另测

```bash
bash "$repo/optimization/reproduce-public-a.sh" single \
  "$release" "$model" "$job/single"
cat "$job/single/process-time.txt"
rg 'model\.|generation.pipeline_total|compute.h3_transformer|total.end_to_end' "$job/single/run.log"
```

`single/process-time.txt` 的 `process.wall` 从启动到退出，包含环境、导入、模型加载、首次生成和清理；`total.end_to_end` 是官方 Python 主流程，包括模型加载；`generation.pipeline_total` 是加载后的生成管线，首次包含 JIT，且不含 MP4 写出。三者均不能直接与常驻 `wall.candidate.mean` 比较。`resident/process-time.txt` 则覆盖整个八请求进程，也不能除以八作为正式均值。

本轮普通默认入口 `process.wall=900.36 s`、模型组件创建加载约 659.023 s、首次生成管线 184.202 s、49 次 Transformer 110.060 s、`total.end_to_end=845.521 s`，另保留于验收记录。首次进程时间受共享盘读取影响。别人报告 280 秒时，先核对其字段和命令；仅凭秒数不能判断原因。

## 5. 交付与已有记录复核

保留整个 `$job`，至少包括构建日志、`source-commit.txt`、`source-status.txt`、`candidate.env`、`release.json`，以及 `resident/benchmark.json`、命令、设备信息、运行日志、进程时间、退出码、逐请求 `record.json` 和视频。另保留 `sequence/source/manifest.json`、`attention-overrides.json` 和 `complete.json`。

本轮普通官方 A 与 A×8/B/C/返回 A 的完整 11 请求全部通过，所有成片与对应已验收版本逐字节一致；A/B/C 独立参考 SSIM 为 93.832013%/99.404123%/99.630047%。B/C 是兼容性回归，未作为本轮配对加速比。[完整验收](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09-qualified.json) · [本轮来源和原始 PMC](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09.json)

已有完整八次配对或完整 11 请求时，可直接重新汇总，无需重跑：

```bash
"$repo/runtime/.venv/bin/python" \
  "$repo/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/summarize-public-a.py" \
  report /absolute/path/to/sequence /absolute/path/to/new-benchmark.json
```

新 JSON 必须尚不存在，原构建仍需可读以核对运行快照。历史 132.824526 秒版固定提交为 `34c842dad5165fee238df1a3137667a604f9460b`，比较名为 `p-row-prepacked-v-vs-softmax-compact`；历史 137.344178 秒版为 `e9d6670286283245554d1e33e81919a880c8e3f6`。不能用新默认构建对应旧成绩字段，也不将历史不同节点的 167 秒等作为本轮加速比基准。

# 官方公开 A 耗时复测说明

当前通过完整验收的版本为 **P 行 LDS 布局 + V 预排布**。正式成绩 **137.344178 秒** 指：八卡 BW1000、模型常驻、双方各一次完整预热后，三次完整 A 请求的墙钟均值。协议为 `minimax-h3-public-a-resident-paired-v1`。

统一入口：[reproduce-public-a.sh](reproduce-public-a.sh)；底层计时：[benchmark-public-a.sh](benchmark-public-a.sh)；原始结果：[p-row-prepacked-f12r2n09-public-a.json](dcu-kernel-autotune/docs/p-row-prepacked-f12r2n09-public-a.json)。可以固定版本、输入、预热和统计边界，换节点后不能保证绝对秒数完全一致。

## 1. 获取正确版本及运行环境

验收源码锚点为 `067b267bfec81e95952f8c9dae9abcc1428d624a`；本文与统一入口在包含它的后续 `xsh-opt` 提交中。旧提交 `0ba8bb17` 是此前 142.89 秒版本，不能用于复测 137.344 秒。

```bash
git clone --branch xsh-opt git@github.com:lmf624953-blip/songshanbei.git songshanbei-retest
cd songshanbei-retest
repo=$(pwd)
git rev-parse HEAD
git merge-base --is-ancestor 067b267bfec81e95952f8c9dae9abcc1428d624a HEAD
test -f "$repo/optimization/reproduce-public-a.sh"
```

不要 shallow clone，原生构建需要历史基线 `45b2c3094d8d189e030f8984d7c07dead4db690a`。需要独占同节点八张 BW1000、`gfx936`、DTK 26.04、GCC 11.4、CPython 3.11、海光 Torch 2.9 及比赛离线依赖。仓库不含模型权重和这些软件环境。

新账号按 [README](README.md#准备运行环境) 准备 `runtime`，包括注意力主机桥；当前账号可复用：

```bash
module load compiler/dtk/26.04 compiler/gcc/11.4.0
# 新克隆目录尚无 runtime 时执行；其他账号改为自己的运行目录。
ln -s /public/home/acvduipmem/xsh/songshanbei/runtime "$repo/runtime"
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
test -x "$repo/runtime/.venv/bin/python"
test -r "$repo/runtime/artifacts/attention-bridge-build.json"
test -r "$model/model_index.json"
hy-smi
```

使用干净 Bash；不附加 hipprof/PMC、张量抓取或有限值检查，同批八卡只运行这一项性能任务。双方使用相同权重目录，不在配对之间切换权重位置或清理页缓存。权重一致性由交付方核对，入口不自动哈希所有权重。

## 2. 从源码构建当前验收版

```bash
mkdir -p "$repo/artifacts"
job=$(mktemp -d "$repo/artifacts/public-a-retest-XXXXXXXX")
bash "$repo/optimization/reproduce-public-a.sh" build "$job/build-work"
release="$job/build-work/release"
```

`build-work` 必须尚不存在。脚本依次构建 Q128/K32 packed 基础版、求和交错版、P 行布局 + V 预排布组合。配置为 `$release/candidate.env`；配对基准为 `$job/build-work/row-sum/attention`。脚本不依赖 `selected-best.env`，不会自动加入未验收的实验 kernel。

构建不运行模型，不计入推理成绩。保留 `build-work/{build,row-sum,release}.log`、`build-exit-code`、提交及工作区状态。`release.json` 记录源码和产物 SHA-256，`qualified=false` 表示构建操作不代替新环境质量验收。

## 3. 常驻配对计时

```bash
bash "$repo/optimization/reproduce-public-a.sh" resident \
  "$release" "$model" "$job/resident"
```

等价底层命令为 `bash "$repo/optimization/benchmark-public-a.sh" "$release" "$model" "$job/resident"`，两条选一条；输出目录必须尚不存在。八次完整生成通常需要二十分钟左右，另加启动和模型加载。

固定 `fl2va`、官方小狗首帧、提示词 `A dog falls asleep`、768×928、50 steps、124 帧、seed 42，每次请求 49 次 Transformer。首帧 SHA-256 为 `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。

脚本显式设置八卡 `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `OMP_NUM_THREADS=4`；编码器 TP=4、初始容量 512、VAE 设备 0/1、确定性首帧编码。模型只加载一次，每次请求重新执行完整计算。

| 请求编号 | 注意力版本 | 统计用途 |
| --- | --- | --- |
| 00 | base，求和交错版 | 完整预热，不计均值 |
| 01 | fast，当前组合版 | 完整预热，不计均值 |
| 02 / 03 | base / fast | 正式配对 1 |
| 04 / 05 | fast / base | 正式配对 2，反转执行顺序 |
| 06 / 07 | base / fast | 正式配对 3 |

正式样本固定为 base 的 02/05/06、fast 的 03/04/07。不能选最小值、加入预热，或用八次总耗时除以八。

`wall_seconds` 使用 `time.perf_counter()`，从请求 `prepare_generation_inputs()` 前开始，到 MP4 写出且相关设备同步后结束。包含输入准备、条件编码、首帧 VAE 编码、49 次 Transformer、视频和音频 VAE 解码、媒体编码和写出。模型加载在请求循环前，媒体规格验证在本次计时后。实现见 [run-sequence.py](run-sequence.py)。

每个进程创建新的九齿/Triton 缓存目录，同进程预热后自然复用代码。未启用 `H3_BENCH_JIT_CACHE` 或跨进程编译缓存优化。

脚本核对设备指令冻结哈希、主机库及回退镜像、运行快照、官方输入、49 次调用和成片规格，要求八条 MP4 逐字节一致；失败退出，不把部分记录当作有效成绩。该性能协议不代替独立参考质量验收。

## 4. 唯一主字段及原始样本

```bash
python3 - "$job/resident/benchmark.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1]))
assert r["protocol"] == "minimax-h3-public-a-resident-paired-v1"
assert r["comparison"] == "row-sum-vs-p-row-prepacked-v"
w = r["wall"]
for name in ("reference", "candidate"):
    x = w[name]
    print(name, "samples =", x["samples"])
    print(name, "mean =", x["mean"], "s; stdev =", x["stdev"], "s")
print("time reduction =", w["mean_time_reduction_percent"], "%")
print("paired reductions =", w["paired_reduction_percent"]["samples"])
for name, x in r["stages"].items():
    print(name, x["reference"]["mean"], "->", x["candidate"]["mean"], "s")
print("model_load_seconds =", r["model_load_seconds"])
print("node =", r["runtime"]["hostname"])
print("images =", r["images"])
PY
```

`f12r2n09` 上的正式结果，单位秒：

| 指标 | base，求和交错版 | fast，当前组合版 |
| --- | --- | --- |
| 样本 1 | 142.983865004 | 136.992860135 |
| 样本 2 | 142.283798470 | 138.403012385 |
| 样本 3 | 141.692351870 | 136.636661931 |
| 请求墙钟均值 | 142.320005115 | **137.344178150** |
| 样本标准差 | 0.646517 | 0.934113 |
| Transformer 合计均值 | 121.994201339 | 116.859849044 |

同轮请求耗时下降 3.496225%，Transformer 下降 4.208686%。不同节点先对照同节点配对收益，再检查绝对时间。167 秒、142.89 秒属于其他版本或节点，不混作当前基准。

主成绩只读取 `wall.candidate.mean`，注明“八卡 BW1000、模型常驻、双方各一次预热后三组完整 A 请求均值”。`generation.pipeline_total` 包含其内部组件，不与 Transformer/VAE 分项重复相加。

## 5. 新进程普通入口计时，排查 280 秒

```bash
bash "$repo/optimization/reproduce-public-a.sh" single \
  "$release" "$model" "$job/single"
cat "$job/single/process-time.txt"
rg 'model\.|generation.pipeline_total|compute.h3_transformer|output.video_vae_decode|output.media_encode|total.end_to_end' \
  "$job/single/run.log"
```

展开的普通入口如下，与 `single` 二选一。脚本另外保存配置、产物清单、命令、设备信息、退出码及日志，镜像检查在被计时进程启动前完成。

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
/usr/bin/time -f 'process.wall %e s' \
  bash "$repo/optimization/run-optimized.sh" --config "$release/candidate.env" fl2va \
  --model-path "$model" --first-frame "$repo/runtime/unified-dog-input/first_frame.png" \
  --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7 \
  --steps 50 --num-frames 124 --seed 42 --output "$job/single-manual.mp4"
```

| 字段 | 边界 | 与 137.344 秒直接比较 |
| --- | --- | --- |
| `resident/benchmark.json: wall.candidate.mean` | 预热后三次完整 A 请求均值，含 MP4 写出 | 可以 |
| 单次 `generation.pipeline_total` | 模型加载后生成管线，含首次 JIT，不含 MP4 编码 | 不可以 |
| 单次 `total.end_to_end` | 官方 Python `main()` 开始到生成、写出、同步、原生引擎清理，含模型加载 | 不可以 |
| `single/process-time.txt: process.wall` | 普通入口启动到退出，含环境、导入、加载、首次生成和清理 | 不可以 |
| `resident/process-time.txt: process.wall` | 八次请求及整个常驻进程启动、退出 | 不可以 |

当前验收轮普通新进程曾测得 `process.wall=883.30 s`，其中模型创建加载约 630.324 s、首次 `generation.pipeline_total=198.561 s`、`total.end_to_end=831.044 s`。与 137.344 秒同时保留，后者不是首次启动总耗时。这些单次数字仅描述该次共享盘读取和节点状态。

280 秒若来自 `process.wall` 或 `total.end_to_end`，应按普通入口比较；若来自同协议预热后的 `wall_seconds`，再核对版本、产物、DTK/Torch、节点负载及分项。未看到对方命令和日志前，不能断定原因。

## 6. 交付清单及已有记录复核

交付整个 `$job`，至少包含：

- `build-work/source-commit.txt`、`source-status.txt`、构建日志、配置、`release.json`。
- `resident/benchmark.json`、`command.log`、`devices.log`、`run.log`、`process-time.txt`、`process-exit-code`。
- `resident/sequence/source/manifest.json`、`attention-overrides.json`、`complete.json`、00–07 的 `record.json` 和 MP4。
- 测了普通入口时保留 `single/` 的配置、清单、日志、命令、退出码、进程时间及视频。
- 模型路径及双方核对的权重清单。移动构建前保留原配置和运行快照。

已有完整八次配对或 A×8/B/C/A，可只重算汇总：

```bash
"$repo/runtime/.venv/bin/python" \
  "$repo/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/summarize-public-a.py" \
  report /absolute/path/to/sequence /absolute/path/to/new-benchmark.json
```

末尾 JSON 必须尚不存在，原构建仍须可读以核对运行快照。完整 11 请求验收见 [qualified.json](dcu-kernel-autotune/docs/p-row-prepacked-f12r2n09-qualified.json)，所有成片与对应已验收版本逐字节一致，A/B/C 独立参考 SSIM 为 93.832013%/99.404123%/99.630047%。

历史 142.89 秒证据保留于 [softmax-row-sum-a14r1n11-steady-partial.json](dcu-kernel-autotune/docs/softmax-row-sum-a14r1n11-steady-partial.json)，对应 `0ba8bb17`、`joint-kv-vs-row-sum` 比较。

# 官方公开 A 耗时复测说明

本文固定已公布 142.89 秒成绩的版本、输入和计时方法。复测主入口是仓库已有的 `optimization/benchmark-public-a.sh`。它在同一八卡常驻进程内完成双方预热和三组配对，输出可比较的 `benchmark.json`。

142.89 秒是模型已加载并完成预热后的完整请求墙钟均值。普通入口第一次运行、模型加载和整个进程退出的耗时需要分别报告。统一协议能够保证测量定义一致，绝对秒数仍受节点、软件版本和存储状态影响。

## 1. 固定源码和运行条件

已在 GitHub `xsh-opt` 发布的复测入口对应提交 `0ba8bb171b2d63d1aefb0c8a45a15bd16e1407f9`。以下克隆命令面向第三方的新目录，不需要切换已有实验工作区。

```bash
git clone --branch xsh-opt git@github.com:lmf624953-blip/songshanbei.git songshanbei-retest
cd songshanbei-retest
git checkout --detach 0ba8bb171b2d63d1aefb0c8a45a15bd16e1407f9
repo=$(pwd)
```

需要独占同一节点的八张 BW1000，设备架构为 `gfx936`。使用 DTK 26.04、GCC 11.4、CPython 3.11、海光 Torch 2.9 和相同的比赛离线依赖。模型目录直接包含 `model_index.json`。同批八卡同时只运行这一项性能任务，性能测量不附加 hipprof、PMC、张量抓取或有限值诊断。

已有当前账号的适配环境时，在新克隆的仓库根目录运行下列命令。其他账号将 `runtime` 的链接目标改为自己的已适配运行目录，准备方法见 [optimization/README.md](README.md) 的“准备运行环境”。源码仓库不包含模型权重、DTK、Torch 和离线依赖，单独克隆仓库后仍须完成该准备。

```bash
module load compiler/dtk/26.04 compiler/gcc/11.4.0
ln -s /public/home/acvduipmem/xsh/songshanbei/runtime "$repo/runtime"
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
test -x "$repo/runtime/.venv/bin/python"
test -r "$model/model_index.json"
hy-smi
```

若仓库已有正确的 `runtime`，跳过 `ln -s`。新账号的模型路径也需要改为自己的路径。两侧配对使用同一模型目录，不清理系统页缓存，不在配对之间切换共享盘和节点本地盘。

## 2. 从源码构建被测版本

以下所有路径均为绝对路径，构建目录必须是新目录。构建时间不计入推理成绩。

```bash
mkdir -p "$repo/artifacts"
job=$(mktemp -d "$repo/artifacts/public-a-retest-XXXXXXXX")

H3_BUILD_JOBS=8 H3_AOT_SAVE_IR=1 \
  bash "$repo/optimization/rebuild-v2.sh" "$job/build" \
  --short-attention --deterministic-keyframe --long-attention-q128-packed \
  --fused-wide-norm-indexed-original-serial-compact --fused-swiglu-row \
  > "$job/build.log" 2>&1

bash "$repo/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/build-row-sum-release.sh" \
  "$job/build/attention" "$job/build/candidate.env" "$job/release" \
  > "$job/release.log" 2>&1
```

`release/joint-kv` 是本轮约 147.80 秒的对照镜像，`release/attention` 是约 142.89 秒的求和交错候选。两者保留其余优化，配对中的 `base` 指联合发射版。历史固定原版和早期 v3 的秒数属于其他版本或节点。

最新 kernel 探索的预排布 V、P 布局等实验不属于这组 142.89 秒成绩。该候选的公开 A 配对和普通官方 A 已完成，历史 A/B/C/返回 A 连续验收未全部完成，当前正式默认配置仍采用联合发射版。复测时显式指定本节生成的配置。

## 3. 复测约 142.89 秒的常驻请求指标

```bash
bash "$repo/optimization/benchmark-public-a.sh" \
  "$job/release" "$model" "$job/resident"
```

脚本固定以下工作量，每次重新执行完整生成。

| 项目 | 固定值 |
| --- | --- |
| 任务 | `fl2va` |
| 首帧 | `runtime/unified-dog-input/first_frame.png` |
| 首帧 SHA-256 | `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54` |
| 提示词 | `A dog falls asleep` |
| 视频画布 | 768×928，由官方首帧和原入口解析 |
| steps / 帧数 / seed | 50 / 124 / 42 |
| Transformer 调用数 | 每次完整请求 49 次 |
| 卡数 | 8，`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| 主机线程 | `OMP_NUM_THREADS=4` |
| 编码器 / VAE 配置 | 编码器 TP=4、初始容量 512、VAE 设备 0/1、确定性首帧编码 |

请求顺序如下，索引从 0 开始。

| 请求 | 注意力镜像 | 用途 |
| --- | --- | --- |
| 00 | base | 基准完整预热 |
| 01 | fast | 候选完整预热 |
| 02 / 03 | base / fast | 正式配对 1 |
| 04 / 05 | fast / base | 正式配对 2，反转执行顺序 |
| 06 / 07 | base / fast | 正式配对 3 |

正式统计仅使用 base 的 02/05/06 和 fast 的 03/04/07。脚本完成八次完整 A 生成，时间通常在二十多分钟量级，另加模型加载和启动开销。

`wall_seconds` 使用 `time.perf_counter()`，从请求输入准备开始，到 MP4 写出且全部相关设备同步结束。包含输入准备、条件编码、VAE 首帧编码、49 次 Transformer、视频及音频 VAE 解码、媒体编码和写出。模型创建加载在请求循环前，媒体规格检查在本次请求计时后。阶段计时中的 `generation.pipeline_total` 包含其内部组件，不能与组件耗时重复相加。

标准启动脚本为每个进程创建新的九齿/Triton 缓存目录。预热后同一进程自然复用已编译代码，未设置 `H3_BENCH_JIT_CACHE`，也未启用另行实验的跨进程缓存优化。

## 4. 唯一对照字段与历史样本

```bash
python3 - "$job/resident/benchmark.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
for key in ("wall",):
    x = d[key]
    print("base samples", x["reference"]["samples"])
    print("fast samples", x["candidate"]["samples"])
    print("base mean", x["reference"]["mean"], "s")
    print("fast mean", x["candidate"]["mean"], "s")
    print("time reduction", x["mean_time_reduction_percent"], "%")
for stage in ("compute.h3_transformer", "output.video_vae_decode", "generation.pipeline_total"):
    x = d["stages"][stage]
    print(stage, x["reference"]["mean"], "->", x["candidate"]["mean"], "s")
print("model_load_seconds", d["model_load_seconds"])
print("runtime", d["runtime"])
print("images", d["images"])
PY
```

历史证据在 `dcu-kernel-autotune/docs/softmax-row-sum-a14r1n11-steady-partial.json`，原节点为 `a14r1n11`。

| 指标 | base | fast |
| --- | --- | --- |
| 请求墙钟样本，秒 | 147.039450 / 148.692147 / 147.671007 | 143.245521 / 143.024396 / 142.391238 |
| 请求墙钟均值，秒 | 147.800868 | 142.887052 |
| 请求墙钟样本标准差，秒 | 0.833967 | 0.443393 |
| Transformer 合计均值，秒 | 127.667392 | 121.966777 |

本轮平均请求耗时下降 3.324619%，Transformer 耗时下降 4.465208%。第三方复测同时提交绝对秒数和同节点配对收益。更换节点后，不将两节点的绝对秒数之比当作优化收益。

脚本检查两套 packed 镜像的设备 `.text` 哈希、首帧、参数、49 次 Transformer 调用、媒体规格和算子调用数，并要求八条成片逐字节一致。任一检查失败时该轮不生成有效汇总。`output_matches_reference_sha256` 额外显示是否与历史优化成片哈希一致，官方参考相似度仍以独立比较为准。

## 5. 普通入口单次计时，用于排查 280 秒

如果别人执行的是普通推理入口，按以下命令另测同一个候选。新进程包含模型加载和首次编译等开销，完整保留这条结果，连同常驻结果一起交付。

```bash
mkdir "$job/single"
(
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export OMP_NUM_THREADS=4
  /usr/bin/time -o "$job/single/process-time.txt" -f 'process.wall %e s' \
    bash "$repo/optimization/run-optimized.sh" \
    --config "$job/release/candidate.env" fl2va \
    --model-path "$model" \
    --first-frame "$repo/runtime/unified-dog-input/first_frame.png" \
    --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7 \
    --steps 50 --num-frames 124 --seed 42 \
    --output "$job/single/video.mp4" \
    > "$job/single/run.log" 2>&1
)
```

建议从独立 Bash 会话执行，以避免手动设置的历史环境变量残留。此命令显式选定重建配置；不依赖本机的 `selected-best.env`。

读取所有计时字段。

```bash
cat "$job/single/process-time.txt"
rg 'model\.|generation.pipeline_total|compute.h3_transformer|output.video_vae_decode|output.media_encode|total.end_to_end' \
  "$job/single/run.log"
```

| 字段 | 边界和用途 |
| --- | --- |
| 常驻 `benchmark.json` 的 `wall.candidate.mean` | 三次预热后完整请求均值，对照 142.89 秒 |
| 单次日志 `generation.pipeline_total` | 生成管线，包含首次调用触发的编译，模型已加载，不包含 MP4 媒体编码 |
| 单次日志 `total.end_to_end` | 官方 Python `main()` 解析参数后开始，到媒体写出、设备同步和原生引擎清理后结束，包含模型加载；不包含此前导入和宿主启动的全部时间 |
| `single/process-time.txt` 的 `process.wall` | 从 Bash 启动普通入口到退出，包含环境初始化、Python 导入、模型加载、首次生成、写出和退出清理 |
| `resident/process-time.txt` 的 `process.wall` | 整个常驻进程和八次请求的总墙钟，需要按该含义报告 |

如果 280 秒来自单次 `process.wall` 或 `total.end_to_end`，应与本节单次结果比较。如果它来自三次正式常驻 `wall_seconds`，则需要比较源码及镜像、Torch/DTK、节点状态与阶段耗时。尚未拿到对方的命令和日志，不能认定 280 秒完全由启动或首次编译造成。

## 6. 交付结果

交付整个 `$job`，至少保留以下文件。

- `build.log`、`release.log`、两份 `candidate.env` 和构建元数据。
- `resident/benchmark.json`、`command.log`、`devices.log`、`run.log`、`process-time.txt` 和 `process-exit-code`。
- `resident/sequence/source/manifest.json`、`attention-overrides.json`、`complete.json`，以及 00–07 每条请求的 `record.json` 和视频。
- 若测了普通入口，保留 `single/run.log`、`single/process-time.txt` 和 `single/video.mp4`。
- 仓库提交、实际运行目录和模型路径。完整权重一致性以双方的权重文件清单及 SHA-256 核对，入口脚本不自动对所有模型权重做哈希。

公开成绩始终注明“八卡 BW1000、常驻预热后三次完整 A 请求均值”或“新进程普通入口单次总耗时”，并附原始字段和样本。后续优化继续沿用这两种明确区分的协议。

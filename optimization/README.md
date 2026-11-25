# MiniMax-H3 决赛 v3

v3 在八张海光 BW1000 上运行，包含 InfiniLM 优化补丁、九齿 AOT 内核、优化推理入口、连续请求验收和剖析脚本。原始框架以提交 `45b2c3094d8d189e030f8984d7c07dead4db690a` 为构建基线。克隆时保留这个提交的历史。

## 准备运行环境

需要 DTK 26.04、GCC 11.4、CPython 3.11、海光 Torch 2.9、比赛离线依赖、官方首帧和 MiniMax-H3 权重。仓库根目录为 `project`，先将 `base_py`、`cache`、`model` 指向当前账号可读的绝对路径。

```bash
project=$(pwd)
module load compiler/dtk/26.04 compiler/gcc/11.4.0

# 已有适配好的运行目录时，可在仓库根目录将 runtime 链接过去。
# 从头准备时，执行以下命令。
"$base_py" "$project/reproduce-local.py" prepare --cache "$cache" --model "$model"
"$base_py" "$project/reproduce-local.py" build --cache "$cache" --model "$model"
(
  export IFAB_AFS_HOME="$project/runtime"
  source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
  python "$IFAB_AFS_HOME/prepare_runtime_layout.py" \
    --first-frame "$IFAB_AFS_HOME/unified-dog-input/first_frame.png"
  python "$IFAB_AFS_HOME/scripts/build_attention_bridge.py"
)
```

模型根目录需要直接包含 `model_index.json`。准备独立 PyTorch 参考时使用同一组离线依赖执行 `reproduce-local.py reference`，然后执行 `reproduce-local.py generate`，两个阶段均传入对应的 `--cache` 和 `--model`。

## 构建 v3 和验收公开样例

```bash
mkdir -p "$project/artifacts"
job=$(mktemp -d "$project/artifacts/v3-XXXXXXXX")
build="$job/build"
first_frame="$project/runtime/unified-dog-input/first_frame.png"
H3_BUILD_JOBS=8 bash "$project/optimization/rebuild-v2.sh" "$build" \
  --short-attention --deterministic-keyframe

out="$job/official"
mkdir "$out"
bash "$project/optimization/run-optimized.sh" --config "$build/candidate.env" fl2va \
  --model-path "$model" --first-frame "$first_frame" \
  --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7 \
  --steps 50 --num-frames 124 --seed 42 --output "$out/video.mp4" \
  > "$out/run.log" 2>&1
```

官方入口按首帧计算画布。公开样例应输出 768×928、124 帧、24 fps 的视频，50 steps 的原调度器实际调用 Transformer 49 次。将 `reference_mp4` 指向独立 PyTorch 参考视频，通过下列命令检查规格及官方视频相似度。

```bash
(
  export IFAB_AFS_HOME="$project/runtime"
  source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
  python "$project/optimization/validate-official-output.py" "$out" "$reference_mp4" "$first_frame"
)
```

使用 `run-resident-measurement.sh` 和 `validate-sequence.py` 运行并核验 A、B、C 连续请求。正式 v3 的八卡 A 请求墙钟均值为 241.306 秒，固定原版为 378.450 秒，A、B、C 的 SSIM 分别为 93.832%、99.404%、99.630%。这些数值来自此前同节点正式测量。

## 长注意力优化入口

- `adaln/build-attention-aot.py` 配置九齿 AOT 代码生成，`adaln/infinilm-short-attention.patch` 接入 InfiniLM 的长短注意力分派。
- v3 的长序列配置使用 32×32、2 warps、1 stage、`kpack=2`。长度小于 4096 且满足原生分派约束时，短序列走 64×64 AOT。输入不满足原生实现的设备、数据类型、步长或执行模式约束时，使用既有 Python 注意力桥。
- `profile-attention-pmc.py` 直接调用 v3 的长注意力二进制。公开 A 和方形 B 的回放长度分别为 27570、22016，构造的 BF16 输入保留每卡 7 个 head 和实际 packed stride。

```bash
pmc="$job/pmc"
mkdir "$pmc"
(
  export IFAB_AFS_HOME="$project/runtime"
  source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
  hipprof --pmc --pmc-type 3 --kernel-name dense_attention_aot_kernel \
    -o "$pmc/attention" python -u "$project/optimization/profile-attention-pmc.py" \
    "$build/attention" --sequence 27570 22016 --iterations 3
)
```

真实 Transformer 回放使用连续请求入口的 `--capture-denoise` 生成 `step-01.pt`、`step-25.pt`、`step-49.pt`。将 `captures` 设为该张量目录的绝对路径。八卡时间线记录同步窗口，窗口结束后等待 2 秒写出剩余事件。

```bash
trace="$job/trace"
mkdir "$trace"
(
  export IFAB_AFS_HOME="$project/runtime"
  source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
  source "$project/optimization/load-candidate-config.sh" "$build/candidate.env"
  export PYTHONPATH="$(cat "$H3_ADALN_BUILD/native-overlay.txt"):$PYTHONPATH"
  hipprof --hip-trace --rccl-trace --devices 0,1,2,3,4,5,6,7 \
    --group-stream --flush-interval 100 --buffer-size 10000 --trace-off \
    --output-type 0 -o "$trace/hipprof" \
    python -u "$IFAB_AFS_HOME/scripts/launch_h3.py" "$project/optimization/adaln/launch.py" \
    --replay-run --model "$model" --captures "$captures" --out "$trace/replay" \
    --steps 1,25,49 --profile-once --profile-drain-seconds 2
) > "$trace/replay.log" 2>&1
```

导出 hipprof 的 stream JSON 后，用 `analyze-hipprof-timeline.py REPLAY_LOG TRACE_JSON --output-dir OUTPUT_DIR` 生成逐卡 kernel 统计和同步窗口的空泡分布。将实验产物写到自己的 `artifacts/` 目录。

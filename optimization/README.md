# MiniMax-H3 决赛推理

v3 在八张海光 BW1000 上运行，包含 InfiniLM 优化补丁、九齿 AOT 内核、优化推理入口、连续请求验收和剖析脚本。原始框架以提交 `45b2c3094d8d189e030f8984d7c07dead4db690a` 为构建基线。克隆时保留这个提交的历史。

当前选用第 51 轮长注意力与 Q/K RMSNorm + MRoPE 融合。组合版已通过官方默认 A 和完整常驻 A/B/C 回归。三次稳态 A 请求均值 188.414 秒，Transformer 均值 167.297 秒。见 [完整验收记录](dcu-kernel-autotune/docs/combination-1218234-qualified.json)。

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

## 构建当前版本和验收公开样例

```bash
mkdir -p "$project/artifacts"
job=$(mktemp -d "$project/artifacts/selected-XXXXXXXX")
build="$job/build"
first_frame="$project/runtime/unified-dog-input/first_frame.png"
H3_BUILD_JOBS=8 bash "$project/optimization/rebuild-v2.sh" "$build" \
  --short-attention --deterministic-keyframe --long-attention-q64-buffer --fused-qknorm-mrope

out="$job/official"
mkdir "$out"
bash "$project/optimization/run-optimized.sh" --config "$build/candidate.env" fl2va \
  --model-path "$model" --first-frame "$first_frame" \
  --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7 \
  --output "$out/video.mp4" \
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

## v3 基础上的长注意力编译器候选

`adaln/attention_codegen.py` 为九齿生成代码增加两个可选变换，将循环不变的只读 Q 加载提升到 KV 循环前，并为 PV 的 LDS 暂存加入 swizzle。模型计算继续由九齿表达并经其编译链生成，变换保留点积、softmax、精度及完整 KV 遍历。

新开关默认关闭。它只接受已经测量的 BF16 PV、FP32 scale、aligned stride、K32、Q32/2 warps 或 Q64/4 warps、1 stage、`kpack=2`、`matrix_instr_nonkdim=0` 组合；其他参数组合会在创建产物目录前退出。生成代码中不能确认循环依赖或 LDS 布局时构建失败。短注意力保持单独构建。

仅重建长注意力时使用新的输出路径。

```bash
(
  export IFAB_AFS_HOME="$project/runtime"
  source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
  H3_AOT_SAVE_IR=1 python "$project/optimization/adaln/build-attention-aot.py" \
    "$project/artifacts/long-attention-lds" --fp32-scale --aligned-strides \
    --query-block 32 --key-block 32 --warps 2 --stages 1 --kpack 2 \
    --optimize-long-attention
)
```

全量重建时给 `rebuild-v2.sh` 追加 `--long-attention-lds`，生成的 `candidate.env` 会指向新长注意力产物。AOT 元数据记录启用的变换、P/V 布局参数、变换源码哈希和最终九齿生成源码哈希。

本机自动调优目录为 `/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune`。第 19 轮在 27570、22016 token 上分别约为 49.0、31.4 ms，同卡配对节省约 19%。27570 token 的全局读取指令减少 33.29%、LDS bank conflict 原始计数减少 93.75%，180 VGPR、8192 B LDS、0 B Scratch。该结果是单算子计时，完整生成收益单独验收。

集成源码重建后，设备可加载区段与第 19 轮相同；关闭新开关后与正式 v3 相同，两侧主机库均文件一致。15 个短长度和分块边界的普通与放大 Q/K 输入检查均逐位一致。结果保存在自动调优目录的 `validated-build/`，真实八卡回放保存在 `validation/round_019/`。`selected-best.env` 的更新以完整规格、质量及正式计时通过为准。

第 25 轮在相同计算上扩大到 Q64/K32、4 warps，P 的 LDS 布局使用 `vec=8, perPhase=2, maxPhase=8`。单算子约 43.026 / 28.047 ms，176 VGPR、16384 B LDS、0 B Scratch；真实八卡三个去噪时刻由约 4.49 s 降到 3.61 s，合计节省 19.60%，video/audio 输出哈希均与同轮 v3 一致。

Q64 构建时将上例改为 `--query-block 64 --warps 4`；全量构建使用 `--long-attention-q64`。集成重建的加载区段、ELF 映射和主机库与第 25 轮一致，15 个边界长度检查全部通过。普通公开 A 已通过本地检查，成片哈希与已验收 v3 完全一致；正式默认在稳态和多用例回归完成前仍为 v3。详见 [实测报告](长注意力编译优化.md)。


## 最新完整稳态结论

在当前 `f12r1n15` 节点，以冻结v3为对照，仅替换第40轮长注意力，同一常驻实例完成双方预热、三组完整A配对、B、C、返回A，共11请求。端到端240.533937→195.138460秒，下降18.872795%；完整去噪220.292703→175.072892秒，下降20.527149%。A/B/C成片哈希均与对应v3相同，规格及官方参考比较通过。历史241.306秒来自另一节点，不参与本次加速比。见 [最新完整报告](长注意力编译优化.md) 和 [完整稳态样本](dcu-kernel-autotune/docs/round40-steady-final.json)。

第41轮可选全量重建开关为 `--long-attention-q64-full-load`，对应完整KV块加载边界消除，强制生成源码守卫；默认构建不变。它正在完成后续验证，以结果文件区分单算子、八卡回放和完整生成，不能互相代替。默认配置的最终切换将随完整验证和最佳候选取舍一起更新。

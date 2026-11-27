# MiniMax-H3 决赛推理

当前完整验收最优版为 **九齿分片 BF16 求和/门控 + 按完整行分片归一化后发布**，沿用已验收长注意力。`f13r1n16` 同一八卡实例三组完整官方 A 为 **122.526417→116.975680 秒，下降 4.530237%**；49 次 Transformer 为 **101.170642→97.202297 秒，下降 3.922428%**。普通默认入口、A×8/B/C/返回 A、独立重建、稳定发布实际加载、真实模型 trace 与独立模型桥 PMC 全部通过，成片逐字节一致，本地默认已切换。未启用跨进程编译缓存。[完整报告与适用范围](dcu-kernel-autotune/docs/行分片归一化模型-f13r1n16.md) · [选版记录](dcu-kernel-autotune/docs/peer-sharded-norm-model-f13r1n16-selection.json)

已有稳定发布目录可直接用 `reproduce-public-a.sh resident RELEASE MODEL NEW_OUTPUT`，读取 `benchmark.json` 的 `wall.candidate.mean`；`single` 单列新进程加载与首次生成，本轮普通入口 875.75 秒。新构建先生成下文注意力前序，再用 `rccl-sequence-parallel/build-peer-release.sh` 构建 peer 参考，最后 `reproduce-public-a.sh build-sharded ACCEPTED_PEER_CONFIG NEW_WORK XMAKE_BINARY`。完整命令与计时口径见[复测说明](公开A耗时复测说明.md)及[模型构建说明](dcu-kernel-autotune/candidates/peer-sharded-norm-model/README.md)。前序 peer 版 120.698043 秒保留作为参考，不将跨轮绝对时间差当本轮加速比。

上一阶段完整验收版为 Q128/K32 packed 的 **P 行 LDS + V 预排布 + 内部 P/ROR4 精简 + stable-max guarded / 成对 P 算术**。在 `f12r2n09` 的同一八卡常驻进程内，双方各一次完整预热后三组官方 A 的完整请求均值为 **132.556449→127.831001 秒，下降 3.564857%**；Transformer 为 **111.882041→107.287698 秒，下降 4.106416%**。候选样本为 127.596813、127.413285、128.482905 秒，标准差 0.571975 秒，三组请求分别下降 4.064172%、3.347815%、3.280321%。

普通官方默认 A 和完整 A×8/B/C/返回 A 均通过，所有成片与对应已验收版本逐字节一致；A/B/C 参考 SSIM 为 93.832013%/99.404123%/99.630047%。官方 A 保持 768×928、50 steps、124 帧、seed 42 和 49 次 Transformer。没有跨进程编译缓存优化。用户已恢复持续优化。新节点实测 QK/在线softmax+PV 拆分，降低QK到120VGPR并优化合并访存后仍退化，未采用；融合内K预取有小收益但不足3%完整请求，先归档。[本轮拆分与预取证据](dcu-kernel-autotune/docs/qk-split-f13r1n16.json)

[统一成绩](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09-public-a.json) · [完整验收](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09-qualified.json) · [来源与原始 PMC](dcu-kernel-autotune/docs/p-paired-sub-f12r2n09.json) · [本轮整合与暂停分析](本轮最优整合与暂停分析.md)

## 构建或复测前序 127.831 秒注意力版本

准备好下文的 `runtime` 和注意力主机桥后，在仓库根目录运行。构建只需一次，已有对应发布目录可直接复用；输出目录必须尚不存在。

```bash
repo=$(pwd)
model=/path/to/MiniMax-H3
mkdir -p "$repo/artifacts"
job=$(mktemp -d "$repo/artifacts/public-a-retest-XXXXXXXX")
bash "$repo/optimization/reproduce-public-a.sh" build "$job/build-work"
bash "$repo/optimization/reproduce-public-a.sh" resident \
  "$job/build-work/release" "$model" "$job/resident"
# 需要另测首次进程启动到退出的耗时时，在常驻配对结束后执行：
bash "$repo/optimization/reproduce-public-a.sh" single \
  "$job/build-work/release" "$model" "$job/single"
```

正式成绩读取 `resident/benchmark.json` 的 `wall.candidate.mean`，比较名为 `softmax-compact-vs-stable-paired`。包含输入准备、条件编码、完整去噪、音视频解码和 MP4 写出，排除模型加载及双方各一次完整预热。`single/process-time.txt` 则是新进程从启动到退出，不能混作常驻成绩；本轮为 900.36 秒，模型组件创建加载约 659.02 秒，首次生成管线 184.202 秒。完整流程和计时边界见 [公开 A 耗时复测说明](公开A耗时复测说明.md)。

默认 `build` 依次经过基础构建、求和交错、P 行/V 预排布、精简 softmax、稳定 max 与成对 P 算术。`build-work/softmax` 保留本轮约 132 秒基准，`build-work/release` 为该阶段正式版；实际设备代码按冻结指令哈希核对。其他 Q/K RMSNorm/MRoPE、原库归约顺序 compact 宽 RMSNorm/索引 AdaLN 和按行 BF16 SwiGLU 均保持原样。

整卡显存五秒采样最高为 33784 MiB，起于普通 A 条件编码器加载中段，可能漏瞬时峰值；混合实例保留两侧工作区，不作为独立显存差证明。历史 132.824526 秒版固定于 `34c842da`，137.344178 秒版固定于 `e9d66702`。本轮只与同轮 132.556449 秒基准比较，下文保留历史构建和剖析用法。

## 复测 142.89 秒的公开 A 常驻配对

在独占八张 BW1000 的单节点上使用 DTK 26.04、相同的 MiniMax-H3 权重和官方首帧。准备好下文的运行环境后，从仓库根目录运行。将 `model` 指向包含 `model_index.json` 的权重目录。

```bash
repo=$(pwd)
model=/path/to/MiniMax-H3
mkdir -p "$repo/artifacts"
job=$(mktemp -d "$repo/artifacts/public-a-XXXXXXXX")
H3_BUILD_JOBS=8 H3_AOT_SAVE_IR=1 bash "$repo/optimization/rebuild-v2.sh" "$job/build" \
  --short-attention --deterministic-keyframe --long-attention-q128-packed \
  --fused-wide-norm-indexed-original-serial-compact --fused-swiglu-row

bash "$repo/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/build-row-sum-release.sh" \
  "$job/build/attention" "$job/build/candidate.env" "$job/release"

bash "$repo/optimization/benchmark-public-a.sh" "$job/release" "$model" "$job/bench"
```

使用 `benchmark.json` 中的 `wall.reference.mean`、`wall.candidate.mean` 和 `wall.mean_time_reduction_percent` 对照公布成绩。脚本在同一模型进程内先分别预热基准和候选一次，随后用请求 02→03、05→04、06→07 做三组配对。请求墙钟包含输入准备、条件编码、49 次 Transformer 调用、音视频解码和 MP4 写出，计时在设备同步后结束。模型加载单列于 `model_load_seconds`，整个常驻进程的墙钟在 `process-time.txt`。阶段耗时、逐请求样本、机器环境和镜像哈希随结果保存。

脚本核对首帧、两套 packed 镜像的设备指令、官方参数和每次请求的算子调用数，并要求八条请求生成的 MP4 逐字节一致。`output_matches_reference_sha256` 记录成片是否也与历史参考视频逐字节一致。

公布成绩来自 `a14r1n11` 上的八卡同进程配对。复测时提供完整 `benchmark.json`，用 `stages["compute.h3_transformer"]` 和 `stages["output.video_vae_decode"]` 定位节点间的耗时差异。绝对秒数随节点状态、依赖版本和视频写盘性能变化。 [原始参考成绩](dcu-kernel-autotune/docs/softmax-row-sum-a14r1n11-steady-partial.json)

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

## 构建 Q128 packed 基础版并运行公开样例

```bash
mkdir -p "$project/artifacts"
job=$(mktemp -d "$project/artifacts/selected-XXXXXXXX")
build="$job/build"
first_frame="$project/runtime/unified-dog-input/first_frame.png"
H3_BUILD_JOBS=8 bash "$project/optimization/rebuild-v2.sh" "$build" \
  --short-attention --deterministic-keyframe --long-attention-q128-packed --fused-wide-norm-indexed-original-serial-compact --fused-swiglu-row

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

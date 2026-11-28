# 视频 VAE 解码分段与 D64 短序列注意力

对照使用两卡时间窗调度、现有 BF16 权重、原时空窗口与解码器。四个入口分别采集设备时间线、检查更大时间窗 batch、比较 D64 注意力和采集单卡 PMC。完整请求当前采用已验收的两卡时间窗实现。实验使用固定种子的独立诊断 latent。

```bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4
unset H3_VAE_CACHE_REPLICAS
candidate="$code/optimization/dcu-kernel-autotune/candidates/vae-decoder-profile"
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/vae-decoder-XXXXXXXX")
bash "$candidate/profile.sh" "$model" "$out/trace"
python "$candidate/summarize.py" "$out/trace" "$out/trace-summary.json"
python "$candidate/bench-batch.py" "$model" "$out/batch.json" --batches 2 3 4 --pairs 3
HIP_VISIBLE_DEVICES=0 python "$candidate/bench-head64.py" "$out/head64"
HIP_VISIBLE_DEVICES=0 hipprof --pmc --pmc-off --devices 0 --kernel-name application_kernel \
    -o "$out/d64-pmc" python "$candidate/profile-head64.py" 128 4 "$out/d64-validation.json"
```

时间线在一次已预热的请求上采集，辅助同步与 profiler 影响绝对耗时。两卡的累计事件可能重叠。时间窗 batch 先验完整输出位模式，仅通过筛选的候选进入配对。D64 候选沿用 K64、原 online softmax 和 BF16 边界，在四种形状上进行逐位与有限值检查。1797-token 使用 31 组无 profiler 设备事件配对。

[结果与取舍](../../docs/视频VAE时间线与D64-f12r1n15.md)

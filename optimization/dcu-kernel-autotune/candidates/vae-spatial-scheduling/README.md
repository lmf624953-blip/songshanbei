# 视频 VAE 空间 tile 调度

复用当前 VAE 的时间窗、空间 tile 坐标、卷积、重叠区混合和输出裁剪。原方案把完整时间窗分给两张卡，本实验将每个时间窗内独立的空间 tile 分给至多八张卡。各卡调用原 `post_quant_conv` 和 `decoder`，结果按原二维索引收集，再由原 `_stitch_tiles` 拼接。复用原调度器的时间窗拼接代码。

```bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
unset H3_VAE_CACHE_REPLICAS
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/vae-spatial-XXXXXXXX")
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
python "$code/optimization/dcu-kernel-autotune/candidates/vae-spatial-scheduling/bench.py" \
    --model "$model" --output "$out/a-paired.json" --devices 8 --pairs 3
```

每侧一次预热，三组交替计时。使用同一模型实例和固定种子的独立诊断 latent，测时包含解码、复制、拼接和设备同步。完整像素检查与显存读数在测时边界以外。历史公开 A 的端到端成绩由完整入口测量。

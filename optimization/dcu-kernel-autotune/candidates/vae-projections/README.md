# 视频 VAE 投影、逐元素融合与时间窗发射对照

在独立诊断入口比较真实 VAE 权重的 QKV 投影、RoPE、SwiGLU 及原时间窗发射。输入由固定种子生成。逐位检查比较 BF16 位模式，配对交替发射。微测输出不作为成片验收结果。

```bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/vae-projections"
model=/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/vae-projections-XXXXXXXX")
python "$candidate/bench-qkv.py" "$model" "$out/qkv.json"
python "$candidate/bench-elementwise.py" "$out/elementwise"
python "$candidate/bench-rope-flat.py" "$out/rope-flat"
hipprof --pmc --pmc-off --devices 0 --kernel-name apply_rope_flat_kernel \
    -o "$out/rope-flat-pmc" python "$candidate/bench-rope-flat.py" \
    "$out/rope-flat-pmc-run" --tiles 2048 --pmc
python "$candidate/build-rope-flat.py" "$out/rope-aot-build"
python "$candidate/bench-rope-flat.py" "$out/rope-aot" \
    --aot "$out/rope-aot-build" --tiles 512 2048
hipprof --pmc --pmc-off --devices 0 --kernel-name apply_rope_flat_kernel \
    -o "$out/rope-aot-pmc" python "$candidate/bench-rope-flat.py" \
    "$out/rope-aot-pmc-run" --tiles 512 --aot "$out/rope-aot-build" --pmc
python "$candidate/bench-clip-graph.py" "$model" "$out/clip-graph.json"
```

`bench-elementwise.py` 保留首版 RoPE 的错误索引以复现实验，`silu_fast` 普通输入的计时也仅作诊断。单独的平面 RoPE 入口会先完成七组输入检查，每个通过的 tile 再运行 31 组无 profiler 配对。AOT 对照沿用相同的运算，固定已知的连续布局 stride，并替换通用 JIT 发射器。时间窗入口用新 latent 覆盖静态输入，验证捕获后的每次回放重新计算。

[结果与取舍](../../docs/视频VAE投影与逐元素-f12r1n15.md)

# 行并行 GEMM 直接写到归约卡

使用公开 A 行并行投影的 `N=27570, M=5376, K=896/1792`。八张卡分别持有本 rank 原始输入和权重。对照一次原 rocBLAS GEMM 写全部输出。两个候选按归约 owner 的完整行区间发射八次 GEMM，并分别写本卡或目标卡。各目标卡内每个来源 rank 使用独立缓冲平面，保留非整除八卡的行尾。

测试会在八卡逐位比较完整投影输出和目标尾段哨兵。校验中的远端读取不计时。每个合格候选和全尺寸对照交替计时 39 组，包含主机发射和同步。此实验只衡量投影，后继求和、门控和普通入口均未纳入。

```bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$project/optimization/selected-best.env"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/row-projection-scatter"
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/row-projection-scatter-XXXXXXXX")
hipcc -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC "$candidate/projection.hip" \
    -lrocblas -o "$out/projection.so"
for k in 896 1792; do
    python "$candidate/bench.py" "$out/projection.so" "$out/A-k$k.json" --k "$k"
done
hipprof --pmc --pmc-off --devices 0,1,2,3,4,5,6,7 -o "$out/direct-pmc" \
    python "$candidate/bench.py" "$out/projection.so" "$out/direct-pmc.json" --k 896 --profile
```

[实测数据与取舍](../../docs/行并行投影直接写归约卡-f12r1n15.md)

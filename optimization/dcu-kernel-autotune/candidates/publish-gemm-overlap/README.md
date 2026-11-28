# 八卡归一化发布与投影重叠

诊断使用现用九齿宽归一化镜像和 rocBLAS 的 BF16 输入、FP32 累加投影。八张卡各负责原行区间，所有归一化行均发布到每张卡。测量从归一化发射到八卡投影全部完成，包含主机发射与同步。

| 模式 | 发布与 GEMM 安排 |
| --- | --- |
| `full` | 每发布者发射一次原标量行复制，每消费者等待全部行后发射完整 GEMM |
| `split_serial` | 分两段发布，等两段齐备后串行发射两次 GEMM |
| `split_overlap` | 分两段发布，消费者仅等待当前段的发布事件 |
| `vector32` / `vector64` / `vector128` | 保留完整 GEMM，仅把发布读写改成相应位宽 |
| `peer_copy` | 用现有发布流对各目的卡提交整行块的 HIP 复制，发布事件覆盖全部复制 |
| `push_local` | 九齿先写本卡完整输入的所属行，再以标量推送到其余七卡 |
| `push_links` | 直接写所属行，128 位复制时以不同 block 负责各对端 |
| `pull_links` | 直接写所属行，目的卡等待各源卡归一化完成后，用 128 位读取其余七卡的行 |

行切分只改变 GEMM 的 N 方向分组。库可能因此选择其他 kernel，程序在性能采样前核对所有 rank 的完整归一化输入及投影输出位模式，检查归一化、发布缓冲和投影输出的末端哨兵。样本使用同一批张量和同一运行进程，按均衡顺序交替。

```bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$project/optimization/selected-best.env"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/publish-gemm-overlap"
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/publish-gemm-XXXXXXXX")
hipcc -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC "$candidate/pipeline.hip" \
    -lrocblas -o "$out/pipeline.so"
python "$candidate/bench.py" "$out/pipeline.so" "$H3_WIDE_NORM_INDEXED_BUILD" \
    "$out/split-qkv.json" --rows 27570 --m 2688
python "$candidate/bench.py" "$out/pipeline.so" "$H3_WIDE_NORM_INDEXED_BUILD" \
    "$out/vector-ffn.json" --rows 27570 --m 3584 --modes full vector32 vector64 vector128
python "$candidate/bench.py" "$out/pipeline.so" "$H3_WIDE_NORM_INDEXED_BUILD" \
    "$out/peer-copy-qkv.json" --rows 27570 --m 2688 --modes full peer_copy
```

独立 PMC 使用 `hipprof --pmc --pmc-off` 启动同一入口，参数增加 `--profile-mode` 并指定其中一个模式。编译、逐位校验、预热和 profiler 采样均在正式配对计时以外。实验用随机 BF16 张量和生产投影形状，跨节点或整模型收益需分别测量。

## 按链路分配 block

直接写本卡所属行的三个候选共享同一分片、原九齿归一化镜像和原全尺寸 GEMM。各卡发布的行互不重叠。推送消费者等八张卡的发布事件，拉取消费者只等本卡拉取完成，拉取流先等其他七卡归一化事件。边界检查还要求旧 compact partial 的全部位置保留哨兵。

```bash
for projection in 2688 3584; do
    python "$candidate/bench.py" "$out/pipeline.so" "$H3_WIDE_NORM_INDEXED_BUILD" \
        "$out/links-m$projection.json" --rows 27570 --m "$projection" \
        --modes full push_local push_links pull_links
done
for mode in full push_links pull_links; do
    hipprof --pmc --pmc-off --devices 0,1,2,3,4,5,6,7 -o "$out/links-$mode" \
        python "$candidate/bench.py" "$out/pipeline.so" "$H3_WIDE_NORM_INDEXED_BUILD" \
        "$out/links-pmc-$mode.json" --rows 27570 --m 2688 \
        --modes full push_links pull_links --profile-mode "$mode"
done
```

[实测数据和取舍](../../docs/跨卡行发布链路-f12r1n15.md)

# 长注意力 Q 分段供数与资源门槛

本候选用于检验融合 kernel 的寄存器压力是否值得通过重载 Q 来降低。它是实验代码，**未采用**，不能替换正式发布目录。`2026-11-28` 在 `f13r1n16` 完成三个对照、逐位校验、39 次无 profiler 配对及独立默认 `hipprof --pmc`。

此前 QK→FP32 分数缓冲→在线 softmax/PV 的真实拆分已经退化；本候选保留融合内部的数据交换，只缩短 Q 的常驻区间。原始 32 个 Q VGPR 改为每次加载 8 个，保留正常块和尾块每条 dot 的累加顺序、原 softmax/PV、RNE、scale 和最终写回。完整复制正式 ABI、预排布和回退镜像，仅替换 packed 注意力镜像。

三个模式：`resource-control` 只改变资源配额，`reload-full` 增加分段 Q 读取但保留高编号寄存器，`reload-compact` 将已腾出的 Q 寄存器用于高编号临时值。静态断言检查冻结镜像、mask、各 QK 操作数的逆向映射、128 条 MMAC、尾部写回和连续寄存器组。断言是构造检查；数值与性能结论仍以 GPU 实测为准。

```bash
code=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
framework=/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune
reference="$framework/validated-build/peer-sharded-norm-f13r1n16/components/attention-aot"
out=$(mktemp -d "$framework/validation/query-reload-XXXXXXXX")
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/attention-query-reload"
for mode in resource-control reload-full reload-compact; do
    python "$candidate/build.py" "$reference" "$out/$mode" --mode "$mode"
done
python "$candidate/screen.py" "$reference" "$out" "$out/screen.jsonl" --pairs 39
```

必须先完成所有编译再计时，不与其他 GPU 任务并行。`screen.py` 使用正式加载器，计时包含每次重新执行 V 预排布及完整注意力；12 个长度覆盖单/双半 Q 尾块、完整块和非整 KV 尾块，并检查原始输入、放大 Q/K、输出 guard 的 BF16 位模式。

`resource-probe.py build REFERENCE NEW_DIRECTORY` 构造独立符号、只有 `s_endpgm` 的资源描述探针；`query` 仅调用 HIP occupancy API，**不发射任何探针 kernel**。该探针不是注意力实现，不可用于性能或数值比较。它显示本节点 256 线程、16 KiB LDS 下，192 VGPR 可报告 4 blocks/CU，196 及以上只报告 3；改成 24/32 KiB LDS 后，本轮查询均最多 2。SGPR 两档结论相同。

最终 packed 资源为 219→198 VGPR（硬件取整 220→200），仍为 3 blocks/CU。27570-token 注意力约 19.20→41.02 ms，22016-token 约 13.07→25.84 ms，全部逐位通过但没有收益。详细样本、原生 PMC 与取舍见 [阶段拆分与Q供数实验](../../docs/阶段拆分与Q供数-f13r1n16.md)。

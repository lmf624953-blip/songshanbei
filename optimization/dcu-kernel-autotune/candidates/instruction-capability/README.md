# gfx936 指令能力与 LDS 矩阵读取映射

本目录是目标能力及内存布局诊断工具，**没有新的正式模型优化或端到端成绩**。保留 DTK 26.04 的真实编译拒绝与 DCU 地址标签输出，避免把 LLVM 中存在某个 intrinsic 当成当前硬件支持。

- `probe-assembler.py NEW_DIRECTORY`：20 项 CPU 编码检查。现用 BF16 MMAC 和两条 LDS 矩阵读取通过；已测大形状 MMAC、MFMA/WMMA、原生 BF16 转换和部分修饰符被当前目标拒绝。不发射 GPU。
- `probe-intrinsics.py NEW_DIRECTORY`：验证 HCU intrinsic 的实际目标支持。普通四种 `lit/lts` 组合能被 `llc` 打印，但 gfx936 汇编器拒绝这两个标志；CZ intrinsic 被代码生成器拒绝。工具记录实际 0/−6/1 返回码，不强行修补位编码，也不发射 GPU。
- `build-matrix-map.py REFERENCE NEW_BUILD`：复用冻结九齿 AOT 描述符，构造 13 个独立内存标签探针。仅做内存复制、地址生成和待检查的 LDS 读取，不作为模型镜像。
- `run-matrix-map.py BUILD NEW_JSONL`：真实 DCU 上每个探针执行两次，检查输出 guard、重复一致性、全部标签范围、四个 wave 一致性、普通 `ds_read_b128` 对照和输入未改变。64 KiB LDS 用于覆盖地址标签，不是正式注意力资源。
- `analyze-matrix-map.py CAPTURE NEW_JSON --reference REFERENCE`：核对固定 packed 镜像及地址生成指令，重建两条矩阵读取的 source-lane/元素偏移模型，验证五种捕获模式，给出当前固定 K/V/P 布局下不能直接替换的见证。模型与结论有明确范围，不宣称所有布局不可行。
- `summarize.py EXPERIMENT NEW_JSON`：只读已完成本轮实验；四个 wave 已实测相同，版本记录保存第一 wave，完整数据和哈希留在实验目录。

```bash
code=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
framework=/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune
candidate="$code/optimization/dcu-kernel-autotune/candidates/instruction-capability"
out=$(mktemp -d "$framework/validation/instruction-capability-XXXXXXXX")
python "$candidate/probe-assembler.py" "$out/assembler"
python "$candidate/probe-intrinsics.py" "$out/intrinsics"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
reference="$framework/validated-build/peer-sharded-norm-f13r1n16/components/attention-aot"
python "$candidate/build-matrix-map.py" "$reference" "$out/matrix-map"
python "$candidate/run-matrix-map.py" "$out/matrix-map" "$out/matrix-map.jsonl"
python "$candidate/analyze-matrix-map.py" "$out/matrix-map-summary.json" "$out/analysis.json" --reference "$reference"
```

真实 GPU 任务与其他性能任务串行。探针没有性能比较；不将汇编通过、地址标签通过或资源数减少当作注意力加速。现有普通入口、官方规格及正式配置不变。完整取舍见 [指令能力与矩阵读取报告](../../docs/指令能力与矩阵读取-f13r1n16.md)。

# K/PV 分段供数与阶段寄存器复用

本目录是 `2026-11-28 f13r1n16` 的隔离实验，**未选用**。它继续验证“大融合 kernel 的资源压力是否值得按阶段调整”，保留原注意力的每条 MMAC、在线 max、分母归约和 BF16 舍入顺序，不增加分数矩阵的全局写回。

`build.py` 以 SHA-256 固定的正式 packed 镜像为输入，完整复制正式主机 ABI、预排布和回退镜像，只替换实验 packed 代码：

- 基础版本将 PV 的 V32 分成两批 V16，复用同一组 VGPR，读取总量不变；尾块先将 P16 半字打包成 P8 VGPR，再读取 V。
- `--early-sum` 将原 alpha/C 缩放和原序分母归约移到 P/V 读取前，释放指数结果；全部浮点指令及其各依赖链保持。
- `--staged-k --key-registers 8|16` 将 QK 的 K 分批读入原片上数据，LDS 读取和 MMAC 总数不变，保留每个 dot 的累加次序。更严格的等待有性能代价。
- `--remat-addresses` 将两个常驻 P LDS 地址改为在每个完整 KV 块写 P 前重算两条整数 XOR，释放跨块的常驻 VGPR；它增加少量动态地址计算，未增加模型浮点工作。

`compact.py` 按 header、完整块 QK、完整块 softmax/PV、尾块 QK、尾块 softmax/PV 五个 CFG 区域重新分配物理 VGPR。所有跨区域活跃值保留原接口，所有向量操作数保持连续，保留部分 EXEC 和部分 DPP bank 写入的旧值依赖。复杂输出区保持原指令和寄存器，边界保守保留 `v0:113`。

构造器中的 CFG/着色检查不能代替 GPU 数值检查。第一次着色遗漏“本区域没有显式使用、但仍需传到后继区域”的活跃寄存器，出现地址错误；已修复 `used` 集合及完整性检查。失败镜像、退出 134、源码快照和日志保留在实验目录。当前源码已包含修复；v2 通过全部验证，但没有性能收益。

```bash
code=/public/home/acvduipmem/xsh/songshanbei/submission-xsh
framework=/public/home/acvduipmem/xsh/songshanbei/optimization/dcu-kernel-autotune
reference="$framework/validated-build/peer-sharded-norm-f13r1n16/components/attention-aot"
out=$(mktemp -d "$framework/validation/pv-stream-XXXXXXXX")
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/attention-pv-stream"
python "$candidate/build.py" "$reference" "$out/k16" \
    --early-sum --staged-k --key-registers 16 --remat-addresses
python "$candidate/compact.py" "$out/k16" "$out/k16-compact" --limit 192
# 先完成全部编译，再串行验证和无profiler配对；不能与其他GPU任务并行。
python "$code/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/verify-p-nan-payload.py" \
    "$reference" "$out/k16-compact" "$out/verify-k16-compact.jsonl"
hipprof --pmc --kernel-name dense_attention_aot_kernel -o "$out/pmc-k16-compact" \
    python "$code/optimization/profile-attention-pmc.py" "$out/k16-compact" --sequence 27570 --iterations 1
```

`verify-extra-tails.py` 用于实验目录内六个已构建变体的额外尾块检查；单独重建一个变体时不要直接调用该批量入口。`summarize.py` 只读取本轮完整实验，检查 570 条最终位模式记录、12 项正式配对和 30 个额外尾块记录，不发射 GPU。

192 VGPR 确实得到 HIP 上限查询的 4 blocks/CU，独立 PMC 为 192 VGPR、64 SGPR、16 KiB LDS、0 scratch。但最佳 K16/192 仍比正式实现慢约 1.14%/1.30%。不进行负候选的整模型资格，不改变正式发布。详见 [实测报告](../../docs/KPV分段供数与192VGPR-f13r1n16.md)。

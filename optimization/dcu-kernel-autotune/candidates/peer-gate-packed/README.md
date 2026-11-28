# 双 BF16 分片门控

候选将两个相邻 BF16 打包为 uint32。每对数据共享 owner 与 rank 选择，输入输出使用 32 位读写。七次求和中，两路仍分别执行原 FP32 加法和 BF16 RNE，保留门控、残差的舍入边界。owner 按每八个原元素定义，打包索引使用 `index // 4`。gate 表沿用原非紧凑行间距，两个 BF16 一并读取。

`kernel.py` 是九齿表达，`build.py` 经现有九齿编译链产生镜像。这里的 ABI 将原 BF16 指针解释为 uint32，长度和分片偏移按两个元素折半。它只用于本目录的独立实验，不能直接放进现用模型桥。

~~~bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/peer-gate-packed"
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/peer-gate-packed-XXXXXXXX")
reference="$project/optimization/dcu-kernel-autotune/validated-build/peer-sharded-norm-f13r1n16/components/model"
for tile in 256 512 1024; do
    python "$candidate/build.py" "$out/packed-v2-$tile" --tile "$tile"
done
python "$candidate/bench.py" "$reference" "$out" "$out/gate-paired.json" \
    --variants packed-v2-256 packed-v2-512 packed-v2-1024
hipprof --pmc --pmc-off --devices 0,1,2,3,4,5,6,7 --kernel-name h3_peer_gate_kernel \
    -o "$out/packed-pmc" python "$candidate/bench.py" "$reference" "$out" \
    "$out/packed-pmc.json" --variants packed-v2-256 --profile-variant packed-v2-256
~~~

复用 `peer-gate-metadata/bench.py` 的八卡独立 gate 检查、固定种子 rank 排列和计时协议。九组输入、三个候选和八卡共 216 项位模式与尾部哨兵检查通过，三种生产行数分别完成 39 组交替配对，共 351 组。输入为合成数据，未校准当前真实 communicator、未接入后继 norm/publish 或完整模型。

首次计时中，候选在事件区间内额外构造位视图，增加了发射开销。修正后的两侧采用相同指针与布局键检查，位视图在预热中准备，仅缓存一组参数视图，不缓存输入计算结果。正式结论使用 `gate-paired.json`。初稿还因有符号与无符号类型提升未通过编译，显式 uint32 转换后构建成功。两个旧稿保存在 `snapshots/a02r3n04/`，相应日志保留在本地实验目录。

三个变体未产生可靠收益，未进入正式选版。独立 PMC 和重建核对已完成。[逐对样本、原始 PMC 和取舍](../../docs/双BF16分片门控-a02r3n04.md)

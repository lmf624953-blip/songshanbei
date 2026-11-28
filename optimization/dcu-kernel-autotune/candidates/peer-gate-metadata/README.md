# 分片求和与门控的 owner 元数据复用

原九齿 kernel 每个输出元素读取八个 rank 的 BF16 输入，按运行时 ringbits 和逐位置 owner 选择原顺序，逐步 BF16 RNE 后完成门控与残差。这里保留相同数学计算，独立比较预旋转 rank 顺序、块内共享 owner 及两者组合。

`build.py` 读取并核对原表达的 SHA，通过九齿构建入口生成目标代码。`reduction_lowering.py` 为本候选声明的块内 owner min/max 补充隐式 lane 归约，同时修正 rank 循环状态的标量类型。混合 owner 块执行原逐元素路径。编译变换仅在本构建进程启用，两个共享变体使用 16 字节 LDS。

~~~bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
candidate="$code/optimization/dcu-kernel-autotune/candidates/peer-gate-metadata"
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/peer-gate-metadata-XXXXXXXX")
reference="$project/optimization/dcu-kernel-autotune/validated-build/peer-sharded-norm-f13r1n16/components/model"
python "$candidate/build.py" "$out/rotate" --mode rotate
python "$candidate/build.py" "$out/shared-v4" --mode shared
python "$candidate/build.py" "$out/shared-rotate-v4" --mode shared-rotate
python "$candidate/bench.py" "$reference" "$out" "$out/gate-micro.json"
hipprof --pmc --pmc-off --devices 0,1,2,3,4,5,6,7 --kernel-name h3_peer_gate_kernel \
    -o "$out/shared-pmc" python "$candidate/bench.py" "$reference" "$out" \
    "$out/shared-pmc.json" --variants shared-v4 --profile-variant shared-v4
~~~

计时前完成九组输入、三个候选、八卡共 216 项全数组位模式和边界检查，包括混合 owner、端点相同而内部不同的反例、极小值、大幅值及 NaN/Inf。通过后在三个生产行数上分别做 39 组无 profiler 交替配对。

输入为合成张量及固定种子的运行时 rank 排列。该入口只执行八卡独立 gate，后继 norm/publish、真实 communicator 校准及完整模型未接入。原型加载器按元数据传递 LDS，接入其他主机桥时需要同步检查这一要求。

[本轮结果、编译失败与适用范围](../../docs/分片门控元数据-a02r3n04.md)

# VAE 多进程时间窗分派实验

对照现用两卡 Python 线程调度与按设备运行的独立进程，检查主机发射是否限制更多设备处理原独立时间窗。每个进程加载同一 VAE 权重，执行原 `_decode_clip`，不更改网络、batch、时空 tile 或注意力表达。父进程继续调用原 `decode_parallel` 的 padding、窗口顺序、重叠混合与最终裁剪。

当前入口使用官方 A 尺寸的固定种子诊断 latent，输出要求 `[1, 3, 124, 928, 768]`、有限且与现用两卡结果逐位一致。输入经过 CPU 共享存储交给子进程，子进程返回当前请求的完整窗口结果，父进程搬回主设备后执行原拼接。正式配对计时包含每请求的输入输出搬运、共享存储准备、解码、同步和拼接，工作进程的模型创建与预热单独记录。

~~~bash
project=/public/home/acvduipmem/xsh/songshanbei
code="$project/submission-xsh"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
export H3_PY_ATTENTION_FP32_SCALE=1
unset H3_BENCH_JIT_CACHE H3_VAE_CACHE_REPLICAS
candidate="$code/optimization/dcu-kernel-autotune/candidates/vae-process-scheduling"
out=$(mktemp -d "$project/optimization/dcu-kernel-autotune/validation/vae-process-XXXXXXXX")
python -u "$candidate/bench.py" \
    /work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3 \
    "$out/run" --workers 4 7 --pairs 3 > "$out/bench.log" 2>&1
~~~

`run/result.json` 保存完整预热、逐位检查、交替配对、子进程耗时和分配器峰值。首轮两卡进程用于区分进程机制本身的代价，随后使用正式配置的 FP32 缩放 attention，比较四、七进程。每侧先完成完整预热，再执行三组交替配对。输出未逐位一致的变体不进入性能比较。

独立诊断后已完成11次完整模型常驻配对、普通默认入口、多用例及进程生命周期验证。同节点完整请求118.966930→107.276659秒，下降9.826488%；11个成片与已验收结果逐字节一致。进一步52条原始解码数值检查及32次资源稳态通过，详见[完整模型报告](../../docs/VAE七进程完整模型-a02r3n04.md)。候选仍显式启用；有效独立时间线/PMC与统一发布整合尚待完成，没有替换默认选版。独立实验中的进程分配器数字不包含完整模型资源，不能直接作为赛事显存指标。

## 完整模型候选接入

`process_runtime.py` 使用独立 subprocess 入口。这样不会触发 Python spawn 对现有无 main guard 的模型启动器进行重放。CPU tensor 共享存储的认证信息通过单次文件描述符传递，不写入环境变量或命令行。`check-ipc.py` 用实际 worker 引导入口验证两进程、三次输入更新的 CPU 共享存储往返和正常退出，它不构成模型数值验证。

宿主给原 `decode_parallel` 函数绑定独立的 `schedule_clips`，其拼接字节码和计算保持原样。worker 调用同一 `_decode_clip`，按继承的 `H3_PY_ATTENTION_FP32_SCALE` 加载匹配九齿 attention。每请求释放不再需要的 GPU 输出引用，并在响应前执行 Python GC，开销计入完整请求。记录 GC 前后的当前显存，用于核查长时间常驻时的对象生命周期。

完整模型候选由 `H3_VAE_PROCESS_WORKERS=7` 与 `H3_VAE_PROCESS_LOG_DIR=绝对路径` 显式启用，沿用已验收 `candidate.env` 的其他配置。`run-sequence.py --vae-process-variants` 只切换 VAE 调度，固定 Transformer 为当前已验收组合。`qualify.sh` 检查 IPC 后，执行同实例 A×8/B/C/返回 A、独立参考比较及汇总，普通公开入口另行验证。现阶段不自动修改 `selected-best.env`。

初始独立微测没有显式配置 FP32 attention，作为调度诊断保存。第二轮对齐正式路径的四、七进程均完成全部逐位与配对检查，原始程序冻结在 `snapshots/a02r3n04/`。后续完整模型候选还包含 subprocess 引导与 GC，因此最终性能以完整模型配对为准。

首次CPU IPC预检的六次数据往返通过，但3秒退出等待导致进程被提前终止，预检判为失败。改为统一20秒退出期限后，六次检查和两进程正常退出均通过。原始失败日志保留在本地独立实验目录。[独立实测及资格范围](../../docs/VAE多进程时间窗-a02r3n04.md)

# 固定 owner 的两阶段计算与发布

本目录只用于独立实验，不修改正式默认。基准为 `peer-gate-nt-f13r1n16`：九齿计算连续分片，独立 HIP kernel 发布到八卡。本实验先固定 RCCL 的逻辑 owner，让九齿只做原序 BF16 求和与门控，再由独立通信 kernel 发布紧凑 partial。计算始终由九齿表达和编译，HIP 只进行内存交换与事件管理。

## 要隔离的问题

- 固定 owner 是否能省掉每个元素的动态累加顺序选择。
- 紧凑 partial 加独立发布，是否比固定 owner 直接向八个输出写回更快。
- 输出与所有输入、残差均不重叠时，全卡 `reduced` 等待是否有必要，以及删除它的实际收益。
- channel 顺序按 `2*rank` 轮转，是否能改善同时访存；它不改变每个元素的累加顺序。

两个发布版本都等待全部输入就绪。`wait` 在发布前额外等待全部九齿 kernel；`async` 仅利用同 stream 上的计算→发布顺序。两者都在消费前等待全部发布完成。主机桥对输出与所有输入、残差的地址区间作重叠检查，拒绝不满足条件的调用。

## 布局与适用范围

快路径仅接受运行时校准确认的 `16 channels / chunk 524288 / alignment 1024`；其他校准条件回退原 InfiniCCL 求和与已选九齿门控。H3 的 residual、gate、indices 在 TP rank 间复制，因此任意 owner 计算后发布的结果一致；不能将此接口直接推广到各 rank 使用不同 residual 的场景。

令 `L=count/16`、`full=L//4194304`、`tail=ceil(ceil((L%4194304)/8)/1024)*1024`，则 `capacity=full*524288+tail`。每卡 partial 以 `[16,capacity]` 紧凑存储，保留 padding；publisher 屏蔽末尾无效位置。分配的 `count/8+16384` 个 BF16 元素覆盖该容量。九齿使用一维 program grid `16*ceil(capacity/tile)`；HIP 发布使用二维 grid `(blocks/16,16)`，二者的网格不能混用。

`--derived-iteration` 将九齿迭代域显式裁剪为从输入 count 推导的 capacity，用于对照符号索引开销。它仍写同一 compact partial，publisher 使用同一容量公式。

`--derived-iteration --grid-2d` 在保存原始九齿 lowered 源码后，将展平坐标的商、余数恢复为 `tl.program_id(1)`、`tl.program_id(0)`，同步将主机发射改为 `(ceil(capacity/1024),16,1)`。模型算术表达不变。该后端变换只审核过 tile 1024，其他 tile 会报错；带有 `grid_2d` 元数据的镜像必须使用对应构建器生成的二维主机桥，不能搭配旧一维桥。构建目录同时保留修改前后的 lowered 源码、SHA 和替换次数；原 ABI JSON 的通用展平 launch 描述不是本实验的实际发射参数。

## 构建与验证

按运行 skill 激活环境，设置 `H3_AUTOTUNE_CODE_ROOT`。每个构建输出目录必须不存在。

```bash
python build-partial-nt.py "$nt" --tile 1024 --warps 4
python build-partial-runtime.py "$nt" "$wait"
python build-partial-runtime.py "$nt" "$async" --async-publish --reuse-host "$wait"

python build-partial-nt.py "$nt2" --tile 1024 --warps 4 \
    --derived-iteration --channel-rotation 2
python build-partial-runtime.py "$nt2" "$async2" --async-publish --reuse-host "$wait"

python build-partial-nt.py "$nt_grid2d" --tile 1024 --warps 4 \
    --derived-iteration --grid-2d
python build-partial-runtime.py "$nt_grid2d" "$async_grid2d" --async-publish
```

复用 `../peer-direct-publish/bench-direct.py`，使用 `--bridge-build`、`--reference-peer-build`、`--indexed-build`、`--separate-output --gather write --vectors 1 --blocks 5120`。`--consumer-copy` 在输出和消费者缓冲区写入哨兵后，紧接着由 GPU 在同一 stream 读取输出；这种模式用于检查可见性和依赖，耗时包含额外读取，不能当作正式无 profiler 性能。

完整结果、原始配对和取舍见 [两阶段与 AllGather 发布实验](../../docs/两阶段与AllGather发布-f13r1n16.md)。通过独立微测不等于通过真实八卡模型与完整官方入口；未通过这些回归的候选不进入默认配置。

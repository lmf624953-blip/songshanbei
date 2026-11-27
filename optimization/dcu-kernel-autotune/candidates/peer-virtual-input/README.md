# 行分片输入的 peer 虚拟映射可行性检查

状态：在全部正式计时与串行采样结束后完成能力检查，八卡均支持 HIP VMM，最小粒度均为 4096 B。8 owner 共 32768 B 的映射由全部八卡复制并逐字通过，`memory-copy-pass`。随后生产A的六项39组配对已完成，当前VMM按需供数流水线退化约83%–101%，不采用；B跨形状序列还有未解决位差。没有新的模型成绩，正式116.975680秒配置保持不变。[生产实测与取舍](../../docs/VMM分片供数-f13r1n16.md)。[原始能力记录](../../docs/peer-virtual-input-f13r1n16-capability.json)

现有行分片归一化仍会在后续列并行 GEMM 之前发布完整输入。待研究的问题是：能否把不同 rank 拥有的页映射到同一连续虚拟地址区间，让原 GEMM 在读取时访问各 owner 的数据，从而避免独立发布，并将供数与计算重叠。它保留原 GEMM 的计算、形状和累加次序，不用量化或跳过层。

这不是已经成立的优化。需要先确认当前驱动支持八卡 HIP VMM、跨 owner 映射及设备 kernel 读取，再测原 GEMM 的真实输入重用行为。远程重复读可能比主动发布更慢；如果原 GEMM 的输入缓存不能抵消远程流量，该方向应归档。不得因为省掉一个发布 kernel 就推断模型变快。

`probe-vmm.cc` 只创建小型八 owner 页区间，授予八卡访问权限，分别由每张卡的通信复制 kernel 读取完整区间，与主机生成的完整原始字节比较。它不加载模型、不实现矩阵运算，也不替代模型验收。`unsupported-attribute` 是设备能力结论，不是性能通过；其他错误应保留阶段与 HIP 错误码，不直接宣称 VMM 不可用。

```bash
hipcc -O3 -std=c++17 --offload-arch=gfx936 probe-vmm.cc -o "$new_probe_binary"
"$new_probe_binary" > "$new_probe_json" 2> "$new_probe_errors"
```

后续若能力检查通过，仍需先完成以下实验，才考虑接入模型：

1. 按实际分配粒度确定完整行边界。每行 `5376*2` 字节，不能把页对齐的填充当成有效 token，也不能改变正式 GEMM 的 leading dimension。
2. 以生产 A/B/C 的实际 GEMM 尺寸，对比完整本地输入与分片映射输入；所有八卡均验证 BF16 输出位模式。通过独立消费者检查和无 profiler 配对，计时必须包含被替代的 norm/发布链，避免只看 GEMM。
3. 检查后继层写入、残差 Tensor 生命周期、流事件依赖及输入别名；相同虚拟指针不能代替每 rank 的拥有关系证明。
4. 只有完整请求收益预算有可靠超过 3% 的可能，才继续真实八卡与完整公开入口资格。驱动不支持或性能不足时保留证据并归档。

代码和计划中的 API 来自当前 DTK 26.04 的 `hip_runtime_api.h` / `hip_runtime_defines.h`；头文件声明不代表当前驱动已实现能力。

## 生产流水线复现

`vmm_gemm.hip` 只负责内存、发布、依赖并调用原库；归一化仍用当前九齿镜像，GEMM调用原rocBLAS。以新目录构建，正式性能任务期间不能并发运行：

```bash
hipcc -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC vmm_gemm.hip -lrocblas -o "$new_library"
python bench-vmm-gemm.py "$new_library" "$accepted_norm_build" "$new_result" \
  --rows 27570 --m 2688 3584 --pairs 39 --warmup 5 --map-granularity 2097152
```

通过`--map-granularity`比较4096/65536/2097152。逻辑norm行仍使用floor边界，物理区间等长页对齐，末尾映射填充不计作有效token。默认A/B/C连续运行会暴露上述B闸门限制，不能把默认命令退出1误读为模型故障或合格性能。`--failure-recheck`仅用于位差诊断，保留局部/候选重复性字段，不放宽比较。

独立采样使用单形状及`--profile-mode local|vmm`，在预热和数值验证后只采集一个流水线；配合hipprof的`--pmc --pmc-off`或`--hip-trace --trace-off --output-type 0`，两种采样单独运行。输出中`diagnostic_profiler=true`不能用作正式成绩。

`probe-residency.py LIBRARY`核对1024/27570行下的句柄device id及各卡显存记账；`summarize-experiment.py EXPERIMENT CODE_ROOT`从原始文件核验39组配对、逐位结果、实际编译HIP源与计时函数AST、八卡kernel数及独立窗口，生成summary.json。旧初始化和封装错误均保留，不重复完整模型基线。

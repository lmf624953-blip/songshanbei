# 宽 RMSNorm 与索引 AdaLN 融合实验

当前实验把 MiniMax-H3 每层的两次 5376 维 RMSNorm 与后续索引 AdaLN 合为一个九齿 kernel，减少一次 kernel 发射和中间张量读写。保留 RMSNorm 的 epsilon、BF16 中间舍入、scale/shift 索引及调制顺序；最终层 RMSNorm 不替换。九齿使用 `ntl.sum` 归约，FP32 加法顺序与原 CUB 不同，因此输出不承诺逐位一致，采用真实八卡输出诊断和独立参考视频质量验证。

截至 2026-11-26，原型已完成同输入八卡第 1、25、49 次 Transformer 配对测试，三步中位数合计 9.445303→9.077796 秒，减少 3.890889%。完整公开 A 保留默认 768×928、50 steps、124 帧、seed 42，执行 49 次 Transformer 和 39,200 次融合调用；平均 SSIM 95.134275%，原已验收 Q128 组合为 93.832013%。这是单次完整生成的结果，**尚不能代替多用例与正式稳态验收，也尚未切换默认**。

## 可复用入口

生产候选源码位于 `optimization/adaln/wide_norm_indexed.py`、`wide_norm_indexed_host.cc` 和 `infinilm-wide-norm-indexed.patch`。新计算由九齿表达，C++ 仅负责检查、分配和发射。重建选项：

```bash
H3_BUILD_JOBS=8 bash optimization/rebuild-v2.sh /绝对路径/全新构建目录 \
  --short-attention --deterministic-keyframe \
  --long-attention-q128-packed --fused-qknorm-mrope --fused-wide-norm-indexed
```

`--fused-wide-norm-indexed` 自动保留 Q/K RMSNorm/MRoPE 融合所需的原生补丁。也可分别构建 native 和设备镜像，使用 `H3_ADALN_BUILD`、`H3_WIDE_NORM_INDEXED_BUILD` 装载。该开关是显式实验选项，普通重建不启用它。

原生 dispatch 仅在 HYGON、BF16、`[1,N,5376]` 连续输入、`eps=1e-5`、I64 连续索引和符合原模型布局的 scale/shift 表上启用；检查设备、形状、stride、16 字节指针对齐和 int32 索引范围。其他情况回退原路径，图捕获期间也回退。长度和调制表行数通过运行时 ABI 传入；不按提示词、首帧内容或 seed 分派。

单算子探索脚本 `build.py`、`build-reduction-layout.py`、`bench.py` 和 `kernel.py` 保留朴素归约、CUB 顺序及线程布局变体。`check-host.py` 对照已测直接发射镜像验证新主机桥的动态行数、表行数、有限值和末段写入边界。重建出的设备 ELF 加载区段与已测试原型一致，证据见 `../../docs/wide-norm-indexed-1218234-source-rebuild.json`。

## 正式验收边界

`--replay-run --widenorm-indexed-ablation` 在同一真实输入上只开关本融合，保留 Q128 注意力及其他优化。未融合分支须与已验收版本输出哈希相同；融合分支检查有限值并记录误差，不把与早期采集结果的差异当作本次融合误差。

进入默认前仍需带完整 guard 的重建等价检查、独立常驻 A,A,A,A,B,C,A、原官方比较函数、三次正式 A 计时、全部请求 49 次调用和 B/C/返回 A 质量核对。首次预热、profiler 和诊断不进入正式平均。详细记录在 `../../docs/wide-norm-indexed-1218234-*.json`，原始视频及大日志留本地。

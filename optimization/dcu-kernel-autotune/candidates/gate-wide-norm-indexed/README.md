# 门控残差、宽 RMSNorm 与索引 AdaLN 融合探索

这是宽 RMSNorm/AdaLN 融合之后的独立探索，尚未接入正式模型。每层第一次门控残差的输出随即进入第二次 RMSNorm/AdaLN，因此尝试在一个九齿 kernel 中计算门控残差，保留残差输出供后续跳连使用，同时生成归一化和调制后的输出。

两处输出和所有 BF16 中间舍入均保留；不省略任何层、不减少去噪调用，也不复用生成结果。微测必须同时比较残差输出和归一化调制输出，检查动态行数、调制表行数、有限值和尾部边界，再用同 stream 交替 GPU event 测量相对“原索引门控 + 已测宽 RMSNorm/AdaLN 融合”的收益。

```bash
export H3_AUTOTUNE_CODE_ROOT=/绝对路径/songshanbei源码
# 先加载已经适配的运行环境和 H3_INDEXED_BUILD。
python candidates/gate-wide-norm-indexed/build.py /全新目录/w4 --warps 4
python candidates/gate-wide-norm-indexed/build.py /全新目录/w8 --warps 8
python candidates/gate-wide-norm-indexed/bench.py /全新目录 \
  --reference /宽RMSNorm融合直接发射参考目录 --output /结果.json
```

参考目录需要宽 RMSNorm 融合微测的 `build.json` 和设备镜像。原型用双输出九齿 ABI 直接发射，正式采用前须增加原生形状守卫、实际模型消融、官方默认 A、多用例和正式稳态检查。线程数量变化也可能改变归约舍入，逐位差异必须明确记录。

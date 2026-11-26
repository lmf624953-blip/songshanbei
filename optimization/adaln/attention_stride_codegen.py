"""对已验证的 packed stride 生成九齿长注意力专门化镜像。"""

import hashlib
import re


PACKED_STRIDES = (896, 896, 2688, 896)


def specialize_attention_strides(source):
    normalized = re.sub(r'loc\("[^"]*/dense_attention_aot\.triton\.py":',
                        'loc("<generated>":', source)
    if hashlib.sha256(normalized.encode()).hexdigest() not in {
            "d9ece80ba7d813f3f2667549538bd84bbcf1a58d34bc47a9262b2792bb9f6119",
            "8d66fa9618b94ed6317c6d7cfb24811ac7ad2cd14ac5ba82a2e0cfe80eecd819"}:
        raise RuntimeError("packed stride 专门化要求已核对的 Q 直接加载与 V 协同布局 IR")
    lines = source.splitlines()
    signatures = [i for i, line in enumerate(lines)
                  if line.startswith("  tt.func public @dense_attention_aot_kernel(")]
    assert len(signatures) == 1
    start = signatures[0]
    sizes = [f"%ninetoothed_ninetoothed_tensor_{i}_size_2" for i in range(4)]
    strides = [f"%ninetoothed_ninetoothed_tensor_{i}_stride_2" for i in range(4)]
    assert all(name + ": i32 " in lines[start] for name in sizes + strides)
    redirects = {name: sizes[0] for name in sizes[1:]}
    constants = []
    for i, (name, stride) in enumerate(zip(strides, PACKED_STRIDES, strict=True)):
        label = f"%h3_stride{i}"
        redirects[name] = label
        constants.append(f"    {label} = arith.constant {stride} : i32")
    variable = re.compile(r"%[-.\w]+")
    body = [variable.sub(lambda m: redirects.get(m[0], m[0]), line)
            for line in lines[start + 1:]]
    return "\n".join(lines[:start + 1] + constants + body) + "\n"

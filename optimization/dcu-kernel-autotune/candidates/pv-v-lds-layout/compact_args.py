"""核对主机 ABI 的同长度约束和已知步长能否缩短地址计算。"""

import re


def apply(source, strides):
    lines = source.splitlines()
    match = [n for n, line in enumerate(lines) if line.startswith("  tt.func public @dense_attention_aot_kernel(")]
    assert len(match) == 1
    head = lines[match[0]]
    assert head.rstrip().endswith("{")
    names = [f"%ninetoothed_ninetoothed_tensor_{i}_size_2" for i in range(4)]
    stride = [f"%ninetoothed_ninetoothed_tensor_{i}_stride_2" for i in range(4)]
    assert all(name+": i32 " in head for name in names+stride)
    redirects = {name:names[0] for name in names[1:]}
    constants = []
    if strides:
        # 仅用于确定为此步长的微测。正式使用前需要四指针精确步长守卫。
        for i, (name, value) in enumerate(zip(stride, [896, 896, 2688, 896], strict=True)):
            label = f"%h3_stride{i}"
            redirects[name] = label
            constants.append(f"    {label} = arith.constant {value} : i32")
    variable = re.compile(r"%[-.\w]+")
    body = [variable.sub(lambda m: redirects.get(m[0],m[0]), line) for line in lines[match[0]+1:]]
    return "\n".join(lines[:match[0]+1]+constants+body)+"\n"

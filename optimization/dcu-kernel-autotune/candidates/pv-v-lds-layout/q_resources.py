"""验证 hoist 之后的 Q 能否直接进入矩阵寄存器布局，降低 LDS 峰值。"""

import re

from producer_layout import ALLOWED, LINE, SSA

TENSOR_LAYOUT = re.compile(r"#(?:blocked|mma)\b")
DOT = "#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>"
Q_TYPE = f"tensor<64x128xbf16, {DOT}>"
Q_SMEM = "!ttg.memdesc<64x128xbf16, #shared, #smem>"


def apply(source, method):
    lines = source.splitlines()
    alloc = [i for i, line in enumerate(lines)
             if "ttg.local_alloc" in line and line.find(Q_SMEM) >= 0]
    loads = [i for i, line in enumerate(lines)
             if "ttg.local_load" in line and Q_SMEM in line and Q_TYPE in line]
    assert len(alloc) == len(loads) == 1 and loads[0] == alloc[0] + 1
    start, end = alloc[0], loads[0]
    alloc_value = LINE.fullmatch(lines[start])
    shared_load = LINE.fullmatch(lines[end])
    assert alloc_value and shared_load
    loaded_value = SSA.search(alloc_value[3])[0]
    assert SSA.search(shared_load[3])[0] == alloc_value[2]
    if method == "convert":
        lines[start:end+1] = [
            f"    {shared_load[2]} = ttg.convert_layout {loaded_value} : "
            f"tensor<64x128xbf16, #blocked> -> {Q_TYPE} loc(#loc369)"
        ]
        return "\n".join(lines) + "\n"

    assert method == "direct"
    definitions = {}
    for i, line in enumerate(lines[:start]):
        m = LINE.fullmatch(line)
        if m:
            definitions[m[2]] = (i, m[3])
    load_index, rhs = definitions[loaded_value]
    assert rhs.startswith("tt.load ") and load_index + 1 == start
    changes = {}
    clones = []

    def clone(value):
        if value in changes:
            return changes[value]
        definition = definitions.get(value)
        if definition is None:
            return value
        index, rhs = definition
        if not TENSOR_LAYOUT.search(rhs):
            return value
        assert index <= load_index
        op = rhs.split(" ", 1)[0]
        if op == "ttg.convert_layout":
            changes[value] = clone(SSA.search(rhs)[0])
            return changes[value]
        if op not in ALLOWED or (op == "tt.load" and value != loaded_value):
            raise RuntimeError(f"无法搬移 Q 的地址/掩码表达式：{op} {value}")
        rhs = SSA.sub(lambda m: clone(m[0]), rhs)
        name = shared_load[2] if value == loaded_value else f"%h3_q_{len(clones)}"
        changes[value] = name
        clones.append(f"    {name} = " + TENSOR_LAYOUT.sub("#h3_q_dot", rhs))
        return name

    assert clone(loaded_value) == shared_load[2]
    assert 10 < len(clones) < 120
    lines[load_index:end+1] = clones
    return f"#h3_q_dot = {DOT}\n" + "\n".join(lines) + "\n"

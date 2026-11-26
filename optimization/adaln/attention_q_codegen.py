"""验证 hoist 之后的 Q 能否直接进入矩阵寄存器布局，降低 LDS 峰值。"""

import hashlib
import re

from attention_v_codegen import ALLOWED, LINE, SSA

TENSOR_LAYOUT = re.compile(r"#(?:blocked|mma)\b")
DOT = "#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>"


def load_attention_query_direct(source, *, query_block=64):
    """保持 Q 索引及掩码，直接按 QK 的矩阵操作数布局加载。"""
    normalized = re.sub(r'loc\("[^"]*/dense_attention_aot\.triton\.py":', 'loc("<generated>":', source)
    expected = {
        64: "3a2233acee520416b0e465be48f1e3d9c9032080fba573723e485f3c30f66ac6",
        128: "e0bc50687b95af04e11633ace298283afaff2e5ffc723d2089b8fbff134db7ca",
    }
    if hashlib.sha256(normalized.encode()).hexdigest() != expected.get(query_block):
        raise RuntimeError("Q 直接加载要求已核对的 V 协同布局和九齿生成 IR")
    q_type = f"tensor<{query_block}x128xbf16, {DOT}>"
    q_smem = f"!ttg.memdesc<{query_block}x128xbf16, #shared, #smem>"
    lines = source.splitlines()
    alloc = [i for i, line in enumerate(lines)
             if "ttg.local_alloc" in line and q_smem in line]
    loads = [i for i, line in enumerate(lines)
             if "ttg.local_load" in line and q_smem in line and q_type in line]
    assert len(alloc) == len(loads) == 1 and loads[0] == alloc[0] + 1
    start, end = alloc[0], loads[0]
    alloc_value = LINE.fullmatch(lines[start])
    shared_load = LINE.fullmatch(lines[end])
    assert alloc_value and shared_load
    loaded_value = SSA.search(alloc_value[3])[0]
    assert SSA.search(shared_load[3])[0] == alloc_value[2]
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
    index = next(i for i, line in enumerate(lines) if line.startswith("#mma = "))
    lines.insert(index + 1, f"#h3_q_dot = {DOT}")
    return "\n".join(lines) + "\n"

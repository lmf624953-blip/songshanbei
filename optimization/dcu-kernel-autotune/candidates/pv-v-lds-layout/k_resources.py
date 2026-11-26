"""验证 K 是否也能绕过 LDS，将每块 K 直接加载到 QK 使用的布局。"""

import re

from producer_layout import ALLOWED, LINE, SSA

LAYOUT = re.compile(r"#(?:blocked1|blocked|mma)\b")
DOT = "#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>"
K_TYPE = f"tensor<128x32xbf16, {DOT}>"
K_SMEM = "!ttg.memdesc<128x32xbf16, #shared1, #smem>"


def apply(source, tail):
    lines = source.splitlines()
    definitions = {}
    for i, line in enumerate(lines):
        parsed = LINE.fullmatch(line)
        if parsed:
            definitions[parsed[2]] = (i, parsed[3])
    allocs = [i for i, line in enumerate(lines) if "ttg.local_alloc" in line and K_SMEM in line]
    reads = [i for i, line in enumerate(lines) if "ttg.local_load" in line and K_SMEM in line and K_TYPE in line]
    assert len(allocs) == len(reads) == 2
    assert all(a+1 == b for a,b in zip(allocs, reads, strict=True))
    count = 2 if tail else 1
    for index in reversed(range(count)):
        alloc, read = allocs[index], reads[index]
        shared, local = LINE.fullmatch(lines[alloc]), LINE.fullmatch(lines[read])
        assert shared and local
        k_value = SSA.search(shared[3])[0]
        load, rhs = definitions[k_value]
        assert rhs.startswith("tt.load ") and SSA.search(local[3])[0] == shared[2]
        assert load+1 == alloc
        changes = {}
        clones = []

        def clone(value):
            if value in changes:
                return changes[value]
            definition = definitions.get(value)
            if definition is None:
                return value
            i, rhs = definition
            if not LAYOUT.search(rhs):
                return value
            assert i <= load
            op = rhs.split(" ", 1)[0]
            if op == "ttg.convert_layout":
                changes[value] = clone(SSA.search(rhs)[0])
                return changes[value]
            if op not in ALLOWED or (op == "tt.load" and value != k_value):
                raise RuntimeError(f"不能移动 K 的张量定义: {op} {value}")
            rhs = SSA.sub(lambda m: clone(m[0]), rhs)
            name = local[2] if value == k_value else f"%h3_k{index}_{len(clones)}"
            changes[value] = name
            clones.append(f"      {name} = " + LAYOUT.sub("#h3_k_dot", rhs))
            return name

        assert clone(k_value) == local[2] and 10 < len(clones) < 120
        lines[load:read+1] = clones
    index = next(i for i, line in enumerate(lines) if line.startswith("#mma = "))
    lines.insert(index+1, f"#h3_k_dot = {DOT}")
    return "\n".join(lines)+"\n"

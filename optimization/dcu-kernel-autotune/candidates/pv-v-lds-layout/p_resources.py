"""仅把 PV 概率从矩阵输出布局转换为操作数布局，测试编译器能否省去 LDS。"""

import re

from producer_layout import LINE, SSA

def apply(source, tail, *, query_block=64):
    assert query_block in (64, 128)
    p_smem = f"!ttg.memdesc<{query_block}x32xbf16, #shared2, #smem>"
    p_type = f"tensor<{query_block}x32xbf16, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 4}}>>"
    lines = source.splitlines()
    allocs = [i for i, line in enumerate(lines) if "ttg.local_alloc" in line and p_smem in line]
    reads = [i for i, line in enumerate(lines) if "ttg.local_load" in line and p_smem in line and p_type in line]
    assert len(allocs) == len(reads) == 2
    assert all(a + 1 == b for a, b in zip(allocs, reads, strict=True))
    count = 2 if tail else 1
    for index in reversed(range(count)):
        alloc, read = allocs[index], reads[index]
        shared, local = LINE.fullmatch(lines[alloc]), LINE.fullmatch(lines[read])
        assert shared and local and SSA.search(local[3])[0] == shared[2]
        value = SSA.search(shared[3])[0]
        source_location = re.search(r"loc\(#loc\d+\)$", local[3])[0]
        lines[alloc:read+1] = [
            f"      {local[2]} = ttg.convert_layout {value} : "
            f"tensor<{query_block}x32xbf16, #mma> -> {p_type} {source_location}"
        ]
    return "\n".join(lines) + "\n"

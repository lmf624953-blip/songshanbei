"""仅把 PV 概率从矩阵输出布局转换为操作数布局，测试编译器能否省去 LDS。"""

import re

from producer_layout import LINE, SSA

P_SMEM = "!ttg.memdesc<64x32xbf16, #shared2, #smem>"
P_TYPE = "tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>"


def apply(source, tail):
    lines = source.splitlines()
    allocs = [i for i, line in enumerate(lines) if "ttg.local_alloc" in line and P_SMEM in line]
    reads = [i for i, line in enumerate(lines) if "ttg.local_load" in line and P_SMEM in line and P_TYPE in line]
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
            f"tensor<64x32xbf16, #mma> -> {P_TYPE} {source_location}"
        ]
    return "\n".join(lines) + "\n"

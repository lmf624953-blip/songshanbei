"""仅重新分派 V 的加载线程，保留九齿生成的地址、掩码和点积。"""

import re

LAYOUTS = {
    "k2-n4": ((2, 4), (1, 64), (4, 1), (1, 0)),
    "k4-n2": ((4, 2), (1, 64), (4, 1), (1, 0)),
    "k4-n4": ((4, 4), (2, 32), (4, 1), (1, 0)),
    "k4-n2-kmajor": ((4, 2), (1, 64), (4, 1), (0, 1)),
    "k8-n2": ((8, 2), (1, 64), (4, 1), (1, 0)),
    "k8-n2-kmajor": ((8, 2), (1, 64), (4, 1), (0, 1)),
}
SSA = re.compile(r"%[-.\w]+")
LINE = re.compile(r"^(\s*)(%[-.\w]+) = (.*)$")
BLOCKED = re.compile(r"#blocked\b")
PRODUCER = "#h3_v_producer"
ALLOWED = {
    "tt.load", "tt.make_range", "tt.expand_dims", "tt.broadcast", "tt.splat", "tt.addptr",
    "arith.andi", "arith.cmpi", "arith.addi", "arith.muli", "arith.select", "arith.constant",
    "arith.remsi", "arith.divsi", "arith.ori", "arith.subi", "arith.extsi", "arith.trunci",
}


def apply(source, preset):
    lines = source.splitlines()
    definitions = {}
    for i, line in enumerate(lines):
        match = LINE.fullmatch(line)
        if match is not None:
            definitions[match[2]] = (i, match[3])
    locations = []
    for i, line in enumerate(lines):
        match = re.search(r"ttg\.local_alloc (%[-.\w]+) : \(tensor<32x128xbf16, #blocked>\)"
                          r" -> !ttg.memdesc<32x128xbf16, #shared3, #smem>", line)
        if match is not None:
            locations.append((i, match[1]))
    assert len(locations) == 2
    for index, (alloc_index, load_value) in reversed(list(enumerate(locations))):
        load_index, load_rhs = definitions[load_value]
        assert load_index < alloc_index and load_rhs.startswith("tt.load ")
        changed = {}
        clones = []

        def clone(value):
            if value in changed:
                return changed[value]
            definition = definitions.get(value)
            if definition is None:
                return value
            line_index, rhs = definition
            if not BLOCKED.search(rhs):
                return value
            assert line_index <= load_index
            op = rhs.split(" ", 1)[0]
            if op not in ALLOWED or (op == "tt.load" and value != load_value):
                raise ValueError(f"不支持移动 V 的张量定义 {op} {value}")
            rewritten = SSA.sub(lambda m: clone(m[0]), rhs)
            new_value = f"%h3_v{index}_{len(clones)}"
            changed[value] = new_value
            clones.append("      " + new_value + " = " + BLOCKED.sub(PRODUCER, rewritten))
            return new_value

        v = clone(load_value)
        assert v != load_value and 10 < len(clones) < 120
        lines[alloc_index] = BLOCKED.sub(PRODUCER, lines[alloc_index].replace(load_value, v))
        lines[load_index:load_index+1] = clones
    per_thread, lanes, warps, order = map(list, LAYOUTS[preset])
    header = (f"{PRODUCER} = #ttg.blocked<{{sizePerThread = {per_thread}, threadsPerWarp = {lanes}, "
              f"warpsPerCTA = {warps}, order = {order}}}>")
    return header + "\n" + "\n".join(lines) + "\n"

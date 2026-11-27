"""仅重新分派 V 的加载线程，保留九齿生成的地址、掩码和点积。"""

import hashlib
import re

SSA = re.compile(r"%[-.\w]+")
LINE = re.compile(r"^(\s*)(%[-.\w]+) = (.*)$")
BLOCKED = re.compile(r"#blocked\b")
PRODUCER = "#h3_v_producer"
ALLOWED = {
    "tt.load", "tt.make_range", "tt.expand_dims", "tt.broadcast", "tt.splat", "tt.addptr",
    "arith.andi", "arith.cmpi", "arith.addi", "arith.muli", "arith.select", "arith.constant",
    "arith.remsi", "arith.divsi", "arith.ori", "arith.subi", "arith.extsi", "arith.trunci",
}


def attention_layout_digest(source):
    """校验包含行号的 IR，忽略生成目录和 Triton 安装目录。"""
    normalized = re.sub(r'loc\("[^"]*/dense_attention_aot\.triton\.py":', 'loc("<generated>":', source)
    normalized = re.sub(r'loc\("[^"]*/triton/language/standard\.py":', 'loc("<triton-standard>":', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()


def coalesce_pv_value(source):
    """仅改已核验的 Q64/K32 或 Q128/K32 生成 IR。动态长度、尾部掩码与归约保持原表达式。"""
    if attention_layout_digest(source) not in {
            "10edb79f779de408e2b1ec95b3960e8cbc0451d32300e3473f64be4b5810607c",
            "0eba84d416a67fce3f589a97e732195f5af804d4e7d9fd0bdad05d4c981cd7f6"}:
        raise RuntimeError("V 的 LDS 布局优化要求已核对的九齿 gfx936 生成 IR")
    original = "#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>"
    updated = "#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>"
    if source.count(original) != 1:
        raise RuntimeError("V 片上布局定义发生变化")
    source = source.replace(original, updated)
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
    per_thread, lanes, warps, order = [4, 4], [2, 32], [4, 1], [1, 0]
    header = (f"{PRODUCER} = #ttg.blocked<{{sizePerThread = {per_thread}, threadsPerWarp = {lanes}, "
              f"warpsPerCTA = {warps}, order = {order}}}>")
    return header + "\n" + "\n".join(lines) + "\n"

"""九齿长注意力生成代码的只读加载提升与 gfx936 LDS 布局变换。"""

import ast
import hashlib

import triton


LONG_ATTENTION_LDS_SWIZZLE = {
    "P": {"vec": 8, "perPhase": 2, "maxPhase": 4, "order": [0, 1]},
    "V": {"vec": 8, "perPhase": 1, "maxPhase": 16, "order": [1, 0]},
}
LONG_ATTENTION_LDS_SWIZZLES = {
    32: LONG_ATTENTION_LDS_SWIZZLE,
    64: {
        "P": {"vec": 8, "perPhase": 2, "maxPhase": 8, "order": [0, 1]},
        "V": {"vec": 8, "perPhase": 1, "maxPhase": 16, "order": [1, 0]},
    },
    128: {
        "P": {"vec": 8, "perPhase": 2, "maxPhase": 16, "order": [0, 1]},
        "V": {"vec": 8, "perPhase": 1, "maxPhase": 16, "order": [1, 0]},
    },
}

Q128_PACKED_P_ROW_LDS = "#shared2 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>"


def swizzle_q128_packed_probability(source):
    original = "#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>"
    descriptor = "!ttg.memdesc<128x32xbf16, #shared2, #smem>"
    if source.count(original) != 1 or source.count(descriptor) != 4:
        raise RuntimeError("Q128 packed P LDS layout does not match the verified full and tail tiles")
    return source.replace(original, Q128_PACKED_P_ROW_LDS)


def hoist_readonly_query(source, *, expected_loads=1):
    """将九齿生成代码中不依赖 KV 循环变量的只读 Q 加载提升到循环前。"""
    tree = ast.parse(source.read_text())
    count = 0

    class Hoist(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal count
            # 只处理没有嵌套循环的 KV 循环；外层 head 循环保持原样。
            if any(isinstance(child, ast.For) for statement in node.body
                   for child in ast.walk(statement)):
                return self.generic_visit(node)
            # 当前 KV 循环只读全局内存。未来若加入写回或原子操作，拒绝提升，
            # 避免在未做指针别名分析的情况下跨越内存副作用。
            if any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                   and isinstance(child.func.value, ast.Name) and child.func.value.id == "tl"
                   and (child.func.attr == "store" or child.func.attr.startswith("atomic_"))
                   for statement in node.body for child in ast.walk(statement)):
                return node
            bound = {child.id for child in ast.walk(node.target) if isinstance(child, ast.Name)}
            bound.update(child.id for statement in node.body for child in ast.walk(statement)
                         if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store))
            loads = {}
            assignments = []

            class Replace(ast.NodeTransformer):
                def visit_Call(self, call):
                    nonlocal count
                    if (isinstance(call.func, ast.Attribute) and call.func.attr == "load"
                            and isinstance(call.func.value, ast.Name) and call.func.value.id == "tl"
                            and call.args):
                        pointer_names = {child.id for child in ast.walk(call.args[0])
                                         if isinstance(child, ast.Name)}
                        names = {child.id for child in ast.walk(call) if isinstance(child, ast.Name)}
                        if "query" in pointer_names and not names.intersection(bound):
                            key = ast.dump(call, include_attributes=False)
                            if key not in loads:
                                name = f"h3_invariant_query_{count}"
                                count += 1
                                loads[key] = name
                                assignments.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=call))
                            return ast.copy_location(ast.Name(id=loads[key], ctx=ast.Load()), call)
                    return self.generic_visit(call)

            replace = Replace()
            node.body = [replace.visit(statement) for statement in node.body]
            return [*assignments, node] if assignments else node

    tree = Hoist().visit(tree)
    if count != expected_loads:
        raise RuntimeError(f"预期 {expected_loads} 个只读 Q 分块，实际提升 {count} 个")
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")



def elide_full_kv_load_mask(source):
    """仅对已核对的 Q64/K32 或 Q128/K32 九齿 AST 删除完整分块的 K/V 上界比较。

    原生 ABI 对 Q/K/V/output 传入同一个 rows，完整循环 j < rows//32。
    固定生成源码摘要同时约束索引、分块大小、范围和其余副作用；编译器输出
    变化时拒绝此变换，必须重新证明和验证，不能静默删除未知掩码。
    """
    tree = ast.parse(source.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest not in {
            "44e8620a409b0927fbad5854965da27b0b0df6da5af8f5154968af776b0a57f2",
            "9250db51ded148e269687a820e0e59710543993ed2525510c050d44f2eea29c5"}:
        raise RuntimeError("完整 KV 加载掩码的九齿源码与已验证 Q64/K32 或 Q128/K32 版本不符")
    removed = 0
    full_loops = 0

    def transform_loop(loop):
        class RemoveFullKVMask(ast.NodeTransformer):
            def visit_Call(self, node):
                nonlocal removed
                if not (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "tl" and node.func.attr == "load" and node.args):
                    return self.generic_visit(node)
                pointers = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}
                if not pointers.intersection({"key", "value"}):
                    return node
                bound = ("ninetoothed_ninetoothed_tensor_1_size_2" if "key" in pointers
                         else "ninetoothed_ninetoothed_tensor_2_size_2")
                class RemoveUpper(ast.NodeTransformer):
                    def visit_Compare(self, comparison):
                        nonlocal removed
                        if (len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.Lt)
                                and isinstance(comparison.comparators[0], ast.Name)
                                and comparison.comparators[0].id == bound
                                and loop.target.id in {n.id for n in ast.walk(comparison.left)
                                                       if isinstance(n, ast.Name)}):
                            removed += 1
                            return ast.copy_location(ast.Constant(value=True), comparison)
                        return comparison
                for keyword in node.keywords:
                    if keyword.arg == "mask":
                        keyword.value = RemoveUpper().visit(keyword.value)
                return node
        loop.body = [RemoveFullKVMask().visit(statement) for statement in loop.body]

    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or any(isinstance(child, ast.For)
                for statement in loop.body for child in ast.walk(statement)):
            continue
        call = loop.iter
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"
                and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == 0):
            full_loops += 1
            transform_loop(loop)
    if full_loops != 1 or removed != 3:
        raise RuntimeError(f"预期一个完整 KV 循环中三个 K/V 上界，实际 {full_loops}/{removed}")
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")

def swizzle_attention_pv(compiled, output, target, options, *, query_block=32, coalesce_value=False,
                         direct_query=False, packed_strides=False, packed_p_row_lds=False):
    """变更九齿输出的共享内存布局，保留点积与 softmax 的计算。"""
    text = compiled.asm["ttgir"]
    if query_block not in LONG_ATTENTION_LDS_SWIZZLES:
        raise ValueError(f"未经验证的 query 分块：{query_block}")
    for label, alias, shape in (("P", "#shared2", f"{query_block}x32"), ("V", "#shared3", "32x128")):
        config = LONG_ATTENTION_LDS_SWIZZLES[query_block][label]
        order = config["order"]
        original = (f"{alias} = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, "
                    f"maxPhase = 1, order = {order}}}>")
        replacement = (f"{alias} = #ttg.swizzled_shared<{{vec = {config['vec']}, "
                       f"perPhase = {config['perPhase']}, maxPhase = {config['maxPhase']}, "
                       f"order = {order}}}>")
        if text.count(original) != 1 or f"!ttg.memdesc<{shape}xbf16, {alias}, #smem>" not in text:
            raise RuntimeError(f"九齿 PV 布局与已验证的 gfx936 编译前提不符: {label}")
        text = text.replace(original, replacement)
    if coalesce_value:
        from attention_v_codegen import coalesce_pv_value
        text = coalesce_pv_value(text)
    if direct_query:
        from attention_q_codegen import load_attention_query_direct
        text = load_attention_query_direct(text, query_block=query_block)
    if packed_strides:
        from attention_stride_codegen import specialize_attention_strides
        text = specialize_attention_strides(text)
    if packed_p_row_lds:
        if not packed_strides or query_block != 128:
            raise ValueError("P row LDS requires the Q128 packed image")
        text = swizzle_q128_packed_probability(text)
    modified = output.with_name(
        "pv-swizzle.ttgir" if output.name == "dense-attention-aot.hsaco"
        else f"{output.stem}-pv-swizzle.ttgir")
    modified.write_text(text)
    return triton.compile(str(modified), target=target, options=options)

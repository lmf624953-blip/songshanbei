"""九齿长注意力生成代码的只读加载提升与 gfx936 LDS 布局变换。"""

import ast

import triton


LONG_ATTENTION_LDS_SWIZZLE = {
    "P": {"vec": 8, "perPhase": 2, "maxPhase": 4, "order": [0, 1]},
    "V": {"vec": 8, "perPhase": 1, "maxPhase": 16, "order": [1, 0]},
}


def hoist_readonly_query(source):
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
    if count != 1:
        raise RuntimeError(f"预期一个只读 Q 分块，实际提升 {count} 个")
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")


def swizzle_attention_pv(compiled, output, target, options):
    """变更九齿输出的共享内存布局，保留点积与 softmax 的计算。"""
    text = compiled.asm["ttgir"]
    for label, alias, shape in (("P", "#shared2", "32x32"), ("V", "#shared3", "32x128")):
        config = LONG_ATTENTION_LDS_SWIZZLE[label]
        order = config["order"]
        original = (f"{alias} = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, "
                    f"maxPhase = 1, order = {order}}}>")
        replacement = (f"{alias} = #ttg.swizzled_shared<{{vec = {config['vec']}, "
                       f"perPhase = {config['perPhase']}, maxPhase = {config['maxPhase']}, "
                       f"order = {order}}}>")
        if text.count(original) != 1 or f"!ttg.memdesc<{shape}xbf16, {alias}, #smem>" not in text:
            raise RuntimeError(f"九齿 PV 布局与已验证的 gfx936 编译前提不符: {label}")
        text = text.replace(original, replacement)
    modified = output.with_name("pv-swizzle.ttgir")
    modified.write_text(text)
    return triton.compile(str(modified), target=target, options=options)

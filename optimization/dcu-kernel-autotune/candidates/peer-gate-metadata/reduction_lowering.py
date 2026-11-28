"""为九齿隐式逐元素域补充本候选显式声明的块内 owner 整数归约。"""

import ast
import hashlib
from pathlib import Path


def restore_owner_reductions(path, tile):
    path = Path(path)
    original = path.read_text()
    tree = ast.parse(original)
    kernel, = (node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "h3_peer_gate_kernel")
    definitions = {node.targets[0].id: node for node in kernel.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    branch, = (node for node in kernel.body if isinstance(node, ast.If))
    condition = definitions[branch.test.id].value
    assert isinstance(condition, ast.Compare) and len(condition.ops) == 1
    assert isinstance(condition.ops[0], ast.Eq)
    low = definitions[condition.left.id]
    high = definitions[condition.comparators[0].id]
    assert ast.dump(low.value) == ast.dump(high.value)
    assert isinstance(low.value, ast.Call) and isinstance(low.value.func, ast.Attribute)
    assert low.value.func.attr == "to" and ast.unparse(low.value.args[0]) == "tl.int32"
    owner_cast = definitions[low.value.func.value.id].value
    assert isinstance(owner_cast, ast.Call) and owner_cast.func.attr == "to"
    owner_load = definitions[owner_cast.func.value.id].value
    assert isinstance(owner_load, ast.Call) and ast.unparse(owner_load.func) == "tl.load"
    assert any(isinstance(node, ast.Name) and node.id == "owners" for node in ast.walk(owner_load.args[0]))
    # 两个值在DSL中被声明为min/max。当前前端按源标量处理，后端在
    # BLOCK个隐式lane的同一owner向量上补充归约，混合块继续走原逐元素分支。
    for operation, assignment in (("min", low), ("max", high)):
        assignment.value = ast.Call(
            func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr=operation, ctx=ast.Load()),
            args=[assignment.value, ast.Constant(0)], keywords=[])
    def scalar(node, environment, visiting=()):
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            if node.id in visiting:
                return False
            return node.id not in environment or scalar(environment[node.id].value, environment,
                                                         (*visiting, node.id))
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            return all(scalar(child, environment, visiting) for child in ast.iter_child_nodes(node)
                       if isinstance(child, ast.expr))
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("tl.min", "tl.max"):
                return True
            if name == "tl.cast":
                return scalar(node.args[0], environment, visiting)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                return scalar(node.func.value, environment, visiting)
            if name in ("tl.where", "tl.minimum", "tl.maximum"):
                return all(scalar(value, environment, visiting) for value in node.args)
        return False

    # 前端把原逐元素r循环状态初始化为BLOCK向量。块归约以后，
    # r成为标量，需要同步消除其初始化广播，acc及混合块仍保留向量。
    scalar_states = []
    for statements in (branch.body, branch.orelse):
        environment = dict(definitions)
        for statement in statements:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = statement.value
            if (target.id.startswith("vr_") and isinstance(value, ast.Call)
                    and ast.unparse(value.func) == "tl.broadcast_to"
                    and scalar(value.args[0], environment)):
                statement.value = value.args[0]
                scalar_states.append(target.id)
            environment[target.id] = statement
    assert len(scalar_states) == 1, scalar_states
    rewritten = ast.fix_missing_locations(tree)
    result = ast.unparse(rewritten) + "\n"
    path.with_suffix(".original.py").write_text(original)
    path.write_text(result)
    return dict(integer_block_reductions=["min", "max"], scalar_loop_states=scalar_states, tile=tile,
                original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                lowered_sha256=hashlib.sha256(result.encode()).hexdigest())

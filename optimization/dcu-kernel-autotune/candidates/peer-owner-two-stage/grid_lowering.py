"""将九齿固定owner的展平二维迭代域恢复为二维program grid，不改计算表达。"""
import ast
from collections import Counter
import hashlib


def restore_grid(source, tile):
    if tile != 1024:
        raise ValueError("本实验仅审核过tile=1024")
    original = source.read_text()
    tree = ast.parse(original)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "h3_peer_gate_kernel")
    assigns = {n.targets[0].id: n.value for n in kernel.body
               if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    channel = assigns["v18"]
    assert isinstance(channel, ast.BinOp) and isinstance(channel.op, ast.Add) and isinstance(channel.right, ast.Constant) and channel.right.value == 0
    channel = channel.left
    assert isinstance(channel, ast.BinOp) and isinstance(channel.op, ast.FloorDiv)
    assert ast.unparse(channel.left) == "index // 1024"
    columns = ast.unparse(channel.right)
    assert "ninetoothed_ninetoothed_tensor_0_size_0" in columns
    patterns = {
        ast.dump(channel): "tl.program_id(1)",
        ast.dump(ast.parse(f"(index // 1024) % ({columns})", mode="eval").body): "tl.program_id(0)",
    }
    replaced = Counter()

    class Rewriter(ast.NodeTransformer):
        def visit_BinOp(self, node):
            key = ast.dump(node)
            if key in patterns:
                replaced[patterns[key]] += 1
                return ast.copy_location(ast.parse(patterns[key], mode="eval").body, node)
            return self.generic_visit(node)

    Rewriter().visit(kernel)
    assert replaced["tl.program_id(0)"] > 1 and replaced["tl.program_id(1)"] > 1
    # 保留原全局mask的线性index；所有二维坐标表达已逐项替换。
    offsets = next(n for n in kernel.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "offsets")
    offsets.value = ast.parse(f"(tl.program_id(1)*({columns})+tl.program_id(0))*BLOCK+tl.arange(0,BLOCK)", mode="eval").body
    ast.fix_missing_locations(tree)
    lowered = ast.unparse(tree) + "\n"
    source.with_suffix(".flat.py").write_text(original)
    source.write_text(lowered)
    return dict(mode="owner-channel-y-local-x", replacements=dict(replaced), columns=columns,
                flat_sha256=hashlib.sha256(original.encode()).hexdigest(),
                grid2d_sha256=hashlib.sha256(lowered.encode()).hexdigest())

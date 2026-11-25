"""Lower NineToothed and check the HIP ABI before writing an HSACO."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

from ninetoothed.compiler.driver import CompileRequest, DEFAULT_COMPILER
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from songshan_h3.compiler_compat import configure_hygon_compiler

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[1] / "runtime").resolve()
WIDTH = 5376
TILE = 256
WARPS = 4



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
    if count != 2:
        raise RuntimeError(f"预期完整循环和尾块循环共两个只读 Q 分块，实际提升 {count} 个")
    removed = 0
    full_loops = 0
    class RemoveFullKVMask(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal removed
            if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tl" and node.func.attr == "load" and node.args):
                return self.generic_visit(node)
            pointers = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}
            if not pointers.intersection({"key", "value"}):
                return node
            bound = "ninetoothed_ninetoothed_tensor_1_size_2" if "key" in pointers else "ninetoothed_ninetoothed_tensor_2_size_2"
            class RemoveUpper(ast.NodeTransformer):
                def visit_Compare(self, comparison):
                    nonlocal removed
                    if (len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.Lt)
                            and isinstance(comparison.comparators[0], ast.Name)
                            and comparison.comparators[0].id == bound
                            and loop.target.id in {n.id for n in ast.walk(comparison.left) if isinstance(n, ast.Name)}):
                        removed += 1
                        return ast.copy_location(ast.Constant(value=True), comparison)
                    return comparison
            for keyword in node.keywords:
                if keyword.arg == "mask":
                    keyword.value = RemoveUpper().visit(keyword.value)
            return node
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or any(isinstance(child, ast.For) for statement in loop.body for child in ast.walk(statement)):
            continue
        call = loop.iter
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"
                and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == 0):
            full_loops += 1
            loop.body = [RemoveFullKVMask().visit(statement) for statement in loop.body]
    if full_loops != 1 or removed != 3:
        raise RuntimeError(f"预期一个完整 KV 循环中三个 K/V 长度上界，实际 {full_loops}/{removed}")
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")


def lower_kernel(build, arrangement, application, tensors, name, warps=WARPS):
    configure_hygon_compiler()
    compilation = DEFAULT_COMPILER.compile(CompileRequest(
        arrangement=arrangement, application=application, tensors=tensors, kernel_name=name,
        backend="triton", platform="hygon-bw1000", num_warps=warps, num_stages=1, max_num_configs=1,
    ))
    compilation.artifact.write_to(build)
    source = build / f"{name}.triton.py"
    if name == "dense_attention_aot":
        hoist_readonly_query(source)
    spec = importlib.util.spec_from_file_location(f"songshan_{name}_lowered", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source, getattr(module, f"{name}_kernel"), compilation.launch_abi.kernel_args


def compile_image(output, kernel, signature, constants, runtime_signature, warps=WARPS,
                  *, block=TILE, allow_shared=False, aligned_scalars=(), num_stages=1,
                  matrix_instr_nonkdim=0, kpack=1, waves_per_eu=1, schedule_hint="none"):
    signature["BLOCK"] = "constexpr"
    if list(signature) != kernel.arg_names or [t for t in signature.values() if t != "constexpr"] != runtime_signature:
        raise RuntimeError(f"Unexpected {kernel.__name__} launcher signature: {signature}")
    attrs = {(i,): [["tt.divisibility", 16]] for i, name in enumerate(signature)
             if signature[name].startswith("*") or name in aligned_scalars}
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs={"BLOCK": block, **constants}, attrs=attrs),
        target=GPUTarget("hip", "gfx936", 64),
        options={"num_warps": warps, "num_stages": num_stages,
                 "matrix_instr_nonkdim": matrix_instr_nonkdim, "kpack": kpack,
                 "waves_per_eu": waves_per_eu, "schedule_hint": schedule_hint},
    )
    initial_asm = dict(compiled.asm)
    # 九齿生成的计算不变，仅替换 PV 暂存所用的共享内存地址布局。
    if output.name == "dense-attention-aot.hsaco":
        text = compiled.asm["ttgir"]
        substitutions = (
            ("#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>",
             "#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>"),
            ("#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>",
             "#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>"),
        )
        for original, replacement in substitutions:
            if text.count(original) != 1:
                raise RuntimeError(f"PV LDS 布局与实验前提不符: {original}")
            text = text.replace(original, replacement)
        modified = output.with_name("pv-swizzle.ttgir")
        modified.write_text(text)
        compiled = triton.compile(str(modified), target=GPUTarget("hip", "gfx936", 64),
                                  options={"num_warps": warps, "num_stages": num_stages,
                                           "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                           "kpack": kpack, "waves_per_eu": waves_per_eu,
                                           "schedule_hint": schedule_hint})
    if os.environ.get("H3_AOT_SAVE_IR") == "1":
        for kind in ("ttir", "ttgir", "llir", "amdgcn"):
            output.with_suffix("." + kind).write_text(compiled.asm[kind] if kind in compiled.asm else initial_asm[kind])
    params = re.search(rf"define amdgpu_kernel void @{kernel.__name__}\(([^\n]*)\)", compiled.asm["llir"])
    if params is None:
        raise RuntimeError(f"The device object does not export {kernel.__name__}")
    types = re.findall(r"(ptr addrspace\(1\)|i32|float) inreg", params[1])
    expected = ["ptr addrspace(1)" if t.startswith("*") else {"fp32": "float"}.get(t, t)
                for t in runtime_signature]
    if types != expected + ["ptr addrspace(1)"] * 2:
        raise RuntimeError(f"Unexpected device ABI: {params[1]}")
    if (compiled.metadata.shared and not allow_shared) or compiled.metadata.profile_scratch_size:
        raise RuntimeError("Unsupported shared memory or profile scratch requirement")
    image = compiled.asm["hsaco"]
    output.write_bytes(image)
    return hashlib.sha256(image).hexdigest(), compiled.metadata.shared


def compile_host(output, sources, *, defines=()):
    template = json.loads((RUNTIME / "artifacts/attention-bridge-build.json").read_text())["command"]
    dtk = Path(os.environ["DTK_ROOT"])
    command = list(template)
    source_index = command.index(str(RUNTIME / "songshan_h3/attention_bridge.cc"))
    command[source_index:source_index + 1] = [
        "-D__HIP_PLATFORM_AMD__", f"-I{dtk / 'include'}", *(str(HERE / s) for s in sources),
        f"-L{dtk / 'lib'}", "-lamdhip64",
    ]
    command[command.index("-o") + 1] = str(output)
    command.extend(f"-D{name}={value}" for name, value in defines)
    subprocess.run(command, check=True)
    return command

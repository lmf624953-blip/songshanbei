"""Lower NineToothed and check the HIP ABI before writing an HSACO."""
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

if __package__:
    from .elf_validation import verify_lowered_hsaco
else:
    from elf_validation import verify_lowered_hsaco

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[1] / "runtime").resolve()
WIDTH = 5376
TILE = 256
WARPS = 4


def lower_kernel(build, arrangement, application, tensors, name, warps=WARPS, *, hoist_query=False, query_loads=1, elide_full_kv_mask=False):
    configure_hygon_compiler()
    compilation = DEFAULT_COMPILER.compile(CompileRequest(
        arrangement=arrangement, application=application, tensors=tensors, kernel_name=name,
        backend="triton", platform="hygon-bw1000", num_warps=warps, num_stages=1, max_num_configs=1,
    ))
    compilation.artifact.write_to(build)
    source = build / f"{name}.triton.py"
    if hoist_query:
        from attention_codegen import hoist_readonly_query
        hoist_readonly_query(source, expected_loads=query_loads)
    if elide_full_kv_mask:
        from attention_codegen import elide_full_kv_load_mask
        elide_full_kv_load_mask(source)
    spec = importlib.util.spec_from_file_location(f"songshan_{name}_lowered", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source, getattr(module, f"{name}_kernel"), compilation.launch_abi.kernel_args


def compile_image(output, kernel, signature, constants, runtime_signature, warps=WARPS,
                  *, block=TILE, allow_shared=False, aligned_scalars=(), num_stages=1,
                  matrix_instr_nonkdim=0, kpack=1, attention_lds_swizzle=False,
                  attention_query_block=32, buffer_addressing=False, coalesce_pv_value=False,
                  direct_attention_query=False, packed_attention_strides=False, packed_p_row_lds=False):
    signature["BLOCK"] = "constexpr"
    if list(signature) != kernel.arg_names or [t for t in signature.values() if t != "constexpr"] != runtime_signature:
        raise RuntimeError(f"Unexpected {kernel.__name__} launcher signature: {signature}")
    attrs = {(i,): [["tt.divisibility", 16]] for i, name in enumerate(signature)
             if signature[name].startswith("*") or name in aligned_scalars}
    if buffer_addressing:
        for i, name in enumerate(signature):
            if signature[name].startswith("*"):
                attrs[(i,)].append(["tt.pointer_range", 32])
    target = GPUTarget("hip", "gfx936", 64)
    options = {"num_warps": warps, "num_stages": num_stages,
               "matrix_instr_nonkdim": matrix_instr_nonkdim, "kpack": kpack}
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs={"BLOCK": block, **constants}, attrs=attrs),
        target=target, options=options,
    )
    initial_asm = dict(compiled.asm)
    if attention_lds_swizzle:
        from attention_codegen import swizzle_attention_pv
        compiled = swizzle_attention_pv(compiled, output, target, options,
                                        query_block=attention_query_block, coalesce_value=coalesce_pv_value,
                                        direct_query=direct_attention_query, packed_strides=packed_attention_strides,
                                        packed_p_row_lds=packed_p_row_lds)
    if os.environ.get("H3_AOT_SAVE_IR") == "1":
        for kind in ("ttir", "ttgir", "llir", "amdgcn"):
            output.with_suffix("." + kind).write_text(
                compiled.asm[kind] if kind in compiled.asm else initial_asm[kind])
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
    verify_lowered_hsaco(image)
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

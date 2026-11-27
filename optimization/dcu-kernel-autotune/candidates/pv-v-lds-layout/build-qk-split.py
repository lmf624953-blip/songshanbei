#!/usr/bin/env python3
"""由同一九齿生成汇编拆出 QK producer 和保持在线递推的 softmax/PV consumer。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def clean(source):
    prefix = source[:source.index("\t.section\t.debug_abbrev")]
    metadata = source[source.index("\t.amdgpu_metadata"):source.index("\t.end_amdgpu_metadata")]
    text = prefix + metadata + "\t.end_amdgpu_metadata\n"
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith((".loc", ".file", ".cfi"))) + "\n"


def base(source):
    text = clean(source)
    # 原保留指针 o14 用作 QK 缓冲；新增一个 Q tile 批次的起始 grid id。
    assert not re.search(r"\bs(?:5[4-9]|6\d)\b|s\[(?:5[4-9]|6\d):", text)
    start = text.index("\ts_load_dword s33,")
    setup = ("\ts_load_dwordx2 s[54:55], s[4:5], 0x50\n"
             "\ts_load_dword s56, s[4:5], 0x58\n"
             "\ts_mov_b32 s57, s6\n\ts_waitcnt lgkmcnt(0)\n"
             "\ts_add_u32 s6, s6, s56\n")
    text = text[:start] + setup + text[start:]
    text = text.replace(".amdhsa_kernarg_size 88", ".amdhsa_kernarg_size 96")
    text = text.replace(".kernarg_segment_size: 88", ".kernarg_segment_size: 96")
    text = text.replace(".amdhsa_next_free_sgpr 54", ".amdhsa_next_free_sgpr 60")
    text = text.replace(".sgpr_count:     58", ".sgpr_count:     64")
    marker = "    .fp64_status:"
    extra = ("      - .name:           split_grid_start\n"
             "        .offset:         88\n        .size:           4\n"
             "        .value_kind:     by_value\n")
    text = text.replace(marker, extra + marker, 1)
    return text


def address(soa=False):
    return ("\ts_mul_i32 s58, s57, s41\n\ts_add_u32 s58, s58, s46\n"
            "\ts_mov_b32 s59, 0\n\ts_lshl_b64 s[58:59], s[58:59], 14\n"
            "\ts_add_u32 s58, s54, s58\n\ts_addc_u32 s59, s55, s59\n"
            f"\tv_lshlrev_b32_e32 v206, {4 if soa else 6}, v0\n")


def producer(source, soa=False):
    text = base(source)
    stop = text.index(".Ltmp0:")
    suffix = text[text.index("\t.section\t.rodata,#alloc"):]
    text = text[:stop]
    begin = text.index("\ts_andn2_b64 vcc, exec, s[26:27]\n", text.index(".LBB0_16:"))
    end = text.index(".Lh3_prepacked_v_done:", begin) + len(".Lh3_prepacked_v_done:\n")
    text = text[:begin] + text[end:]
    text += address(soa)
    for offset, reg in enumerate(range(158, 174, 4)):
        if soa and offset:
            text += "\tv_add_u32_e32 v206, 4096, v206\n"
        text += f"\tglobal_store_dwordx4 v206, v[{reg}:{reg+3}], s[58:59] offset:{0 if soa else 16*offset}\n"
    text += ("\ts_add_i32 s46, s46, 1\n\ts_cmp_eq_u32 s41, s46\n"
             "\ts_cbranch_scc1 .Lh3_qk_exit\n"
             "\ts_add_u32 s22, s22, 0xe000\n\ts_addc_u32 s23, s23, 0\n"
             "\tglobal_load_dwordx4 v[166:169], v137, s[22:23]\n"
             "\tglobal_load_dwordx4 v[170:173], v136, s[22:23]\n"
             "\ts_branch .Lh3_v_guard\n.Lh3_qk_exit:\n\ts_waitcnt vmcnt(0)\n\ts_endpgm\n")
    assert text.count("v_mmac_f32_16x16x16_bf16") == 32
    text = text.replace("s_cbranch_scc1 .LBB0_20", "s_cbranch_scc1 .Lh3_qk_exit")
    return (text + suffix).replace("dense_attention_aot_kernel", "h3_split_qk_kernel")


def consumer(source, soa=False):
    text = base(source)
    # 仅省去完整 KV 块的 QK；尾块沿用原实现。
    first = text.index("\tglobal_load_dwordx4 v[166:169], v137, s[22:23]", text.index(".Lh3_first_k_load:"))
    last = text.index(".Lh3_v_guard:", first)
    text = text[:first] + text[last:]
    for line in ("\tds_write_b128 v138, v[166:169]\n", "\tds_write_b128 v138, v[170:173] offset:4096\n"):
        assert text.count(line) == 1
        text = text.replace(line, "", 1)
    first = text.index("\tds_read_b128 v[97:100], v101", text.index(".Lh3_prepacked_v_done:"))
    last = text.index(".Ltmp0:", first)
    loads = address(soa)
    for offset, reg in enumerate(range(158, 174, 4)):
        if soa and offset:
            loads += "\tv_add_u32_e32 v206, 4096, v206\n"
        loads += f"\tglobal_load_dwordx4 v[{reg}:{reg+3}], v206, s[58:59] offset:{0 if soa else 16*offset}\n"
    loads += "\ts_waitcnt vmcnt(0)\n"
    text = text[:first] + loads + text[last:]
    first = text.index(".Lh3_next_k_load:")
    last = text.index(".Lh3_next_k_ready:", first)
    text = text[:first] + ".Lh3_next_k_load:\n" + text[last:]
    assert text[text.index(".LBB0_20:"):text.index("\t.section\t.rodata,#alloc")] == base(source)[base(source).index(".LBB0_20:"):base(source).index("\t.section\t.rodata,#alloc")]
    assert text.count("v_mmac_f32_16x16x16_bf16") == source.count("v_mmac_f32_16x16x16_bf16") - 32
    return text.replace("dense_attention_aot_kernel", "h3_split_softmax_pv_kernel")


def compact_producer(source):
    """对实际CFG做保守VGPR死定义消除，再按四寄存器组压缩编号。"""
    registers = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))["registers"]
    lines = source.splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("\t.section\t.rodata"))
    prefix, suffix = lines[:end], lines[end:]
    removed = 0
    while True:
        labels = {line[:-1]: i for i, line in enumerate(prefix) if line.endswith(":")}
        reads, writes, successors, removable = [], [], [], []
        for i, line in enumerate(prefix):
            refs = registers(line)
            match = re.match(r"\t(?:v_\S+|ds_read_\S+|global_load_\S+) (v(?:\d+|\[\d+:\d+\])),", line)
            pure = bool(match and "vcc" not in line and not line.startswith("\tv_cmp"))
            target = registers(match[1]) if pure else set()
            operands = line[match.end():] if pure else line
            use = registers(operands)
            # masked global loads可能只改写部分lane，保留旧值定义。
            if line.startswith("\tglobal_load"):
                use |= target
            writes.append(target); reads.append(use); removable.append(pure)
            branch = re.search(r"\bs_(?:cbranch_\S+|branch) (\.\w+)$", line)
            if branch:
                edges = [labels[branch[1]]]
                if not line.startswith("\ts_branch ") and i + 1 < len(prefix):
                    edges.append(i + 1)
            else:
                edges = [] if line.startswith("\ts_endpgm") or i + 1 == len(prefix) else [i + 1]
            successors.append(edges)
        live_in = [set() for _ in prefix]; live_out = [set() for _ in prefix]
        while True:
            changed = False
            for i in range(len(prefix) - 1, -1, -1):
                out = set().union(*(live_in[j] for j in successors[i]))
                incoming = reads[i] | (out - writes[i])
                if incoming != live_in[i] or out != live_out[i]:
                    live_in[i], live_out[i], changed = incoming, out, True
            if not changed:
                break
        drops = {i for i in range(len(prefix)) if removable[i] and writes[i].isdisjoint(live_out[i])}
        if not drops:
            break
        removed += len(drops)
        prefix = [line for i,line in enumerate(prefix) if i not in drops]
    used = set().union(*(registers(line) for line in prefix))
    quads = sorted({r // 4 for r in used})
    mapping = {4*q + k: 4*i + k for i,q in enumerate(quads) for k in range(4)}
    assert mapping[0] == 0
    for line in prefix:
        for lo, hi in re.findall(r"v\[(\d+):(\d+)\]", line):
            assert mapping[int(hi)] - mapping[int(lo)] == int(hi) - int(lo), line
    def replace(line):
        line = re.sub(r"\bv(\d+)\b", lambda m: f"v{mapping[int(m[1])]}", line)
        return re.sub(r"v\[(\d+):(\d+)\]", lambda m: f"v[{mapping[int(m[1])]}:{mapping[int(m[2])]}]", line)
    prefix = [replace(line) for line in prefix]
    count = 4 * len(quads)
    result = "\n".join(prefix + suffix) + "\n"
    result = re.sub(r"\.amdhsa_next_free_vgpr \d+", f".amdhsa_next_free_vgpr {count}", result)
    result = re.sub(r"\.vgpr_count:\s+\d+", f".vgpr_count:     {count}", result)
    assert result.count("v_mmac_f32_16x16x16_bf16") == 32
    return result, dict(removed_vector_definitions=removed, next_free_vgpr=count)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--compact-producer", action="store_true")
    cli.add_argument("--score-layout", choices=("aos", "soa4"), default="aos")
    args = cli.parse_args()
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    meta = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert hashlib.sha256((args.reference / "dense-attention-packed.hsaco").read_bytes()).hexdigest() == meta["packed_strides"]["hsaco_sha256"]
    args.output.mkdir()
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    records = {}
    for name, transform in (("qk", producer), ("softmax-pv", consumer)):
        assembly = transform(source, args.score_layout == "soa4")
        compaction = None
        if name == "qk" and args.compact_producer:
            assembly, compaction = compact_producer(assembly)
        (args.output / (name + ".amdgcn")).write_text(assembly)
        image = backend.make_hsaco(assembly, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
        (args.output / (name + ".hsaco")).write_bytes(image)
        records[name] = dict(sha256=hashlib.sha256(image).hexdigest(), static_mmac=assembly.count("v_mmac_f32_16x16x16_bf16"), compaction=compaction)
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"
    sys.path.insert(0, str(code))
    from aot_codegen import compile_host
    host = Path(__file__).with_name("qk_split_host.cc")
    command = compile_host(args.output / "split-host.so", (host,))
    record = dict(reference=str(args.reference.resolve()), reference_packed_sha256=meta["packed_strides"]["hsaco_sha256"],
                  scope="Q tile批次有界FP32 QK中间缓冲，完整KV块拆分；consumer原在线softmax/PV和尾块保留",
                  score_layout=args.score_layout, images=records, host_command=command, host_source_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),
                  correctness="尚未验证", selected=False)
    (args.output / "split-build.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

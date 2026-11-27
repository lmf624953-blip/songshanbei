#!/usr/bin/env python3
"""按真实 CFG 对两段区域做保守寄存器合并，保留边界接口与连续操作数。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import time


VGPR = re.compile(r"\bv(?:(\d+)\b|\[(\d+):(\d+)\])")


def registers(text):
    result = 0
    for m in VGPR.finditer(text):
        for r in ([int(m[1])] if m[1] else range(int(m[2]), int(m[3])+1)):
            result |= 1 << r
    return result


def members(bits):
    while bits:
        bit = bits & -bits
        yield bit.bit_length()-1
        bits ^= bit


def liveness(lines, return_successors=False):
    labels = {line[:-1]:i for i,line in enumerate(lines) if line.endswith(":")}
    reads, writes, successors = [], [], []
    masks = []
    for i,line in enumerate(lines):
        partial = bool(masks)
        save = re.fullmatch(r"\ts_and_saveexec_b64 (s\[\d+:\d+\]), .*", line)
        restore = re.fullmatch(r"\ts_or_b64 exec, exec, (s\[\d+:\d+\])", line)
        if save:
            masks.append(save[1])
        elif restore:
            assert masks and masks[-1] == restore[1], (i,line,masks)
            masks.pop()
        elif re.match(r"\ts_\S+ exec,", line):
            raise ValueError(f"未建模的EXEC写入：{line}")
        use, definition = registers(line), 0
        if re.match(r"\t(?:v_|ds_read|global_load)", line) and ", " in line:
            first, rest = line.split(", ", 1)
            definition, use = registers(first), registers(rest)
            if line.startswith("\tglobal_load") and " lds" in rest:
                definition = 0  # 此类指令直接写LDS，VDST仅为汇编器形式参数。
            if partial or line.startswith("\tv_fmac") or ("_dpp " in line and "bank_mask:0xf" not in line):
                use |= definition
        reads.append(use); writes.append(definition)
        branch = re.search(r"\bs_(?:cbranch_\S+|branch) (\.\w+)$", line)
        if branch:
            edges = [labels[branch[1]]]
            if not line.startswith("\ts_branch "):
                edges.append(i+1)
        else:
            edges = [] if line.startswith("\ts_endpgm") or i+1 == len(lines) else [i+1]
        successors.append(edges)
    incoming, outgoing = [0]*len(lines), [0]*len(lines)
    while True:
        changed = False
        for i in range(len(lines)-1, -1, -1):
            after = 0
            for target in successors[i]:
                after |= incoming[target]
            before = reads[i] | (after & ~writes[i])
            if incoming[i] != before or outgoing[i] != after:
                changed = True
                incoming[i], outgoing[i] = before, after
        if not changed:
            break
    result = incoming, outgoing, reads, writes
    return (*result,successors) if return_successors else result


def color(lines, incoming, outgoing, reads, writes, start, end, anchors, limit, seconds):
    edges = [0]*219
    used = 0
    parent = list(range(219))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(start,end):
        # 当前区域没有显式读写、却要穿过区域传给后继的值，也必须占据其物理槽。
        used |= registers(lines[i]) | incoming[i] | outgoing[i]
        for a,b in re.findall(r"v\[(\d+):(\d+)\]", lines[i]):
            for r in range(int(a)+1,int(b)+1):
                parent[find(r)] = find(int(a))
        # 同时存活/同指令读取的不同原始值不能合并；死定义也不能破坏活值。
        for live in (incoming[i], outgoing[i] | writes[i]):
            for r in members(live):
                edges[r] |= live & ~(1<<r)
    groups = {}
    for r in members(used):
        groups.setdefault(find(r),[]).append(r)
    groups = [tuple(sorted(g)) for g in groups.values()]
    assert all(list(g) == list(range(g[0],g[-1]+1)) for g in groups)
    occupied = [0]*limit
    mapping = {}
    def valid(group, base):
        return all(not (edges[r] & occupied[base+r-group[0]]) for r in group)
    def assign(group, base, add):
        for r in group:
            target = base+r-group[0]
            if add:
                occupied[target] |= 1<<r; mapping[r] = target
            else:
                occupied[target] &= ~(1<<r); del mapping[r]
    pending = []
    for group in groups:
        if any(r in anchors for r in group):
            if group[-1] >= limit or not valid(group,group[0]):
                return None
            assign(group,group[0],True)
        else:
            pending.append(group)
    deadline = time.monotonic()+seconds
    nodes = 0
    def search(todo):
        nonlocal nodes
        nodes += 1
        if time.monotonic() > deadline:
            raise TimeoutError
        if not todo:
            return True
        choices = []
        for group in todo:
            domain = [base for base in range(limit-len(group)+1) if valid(group,base)]
            if not domain:
                return False
            choices.append((len(domain),-sum(edges[r].bit_count() for r in group),group,domain))
        _,_,group,domain = min(choices)
        remaining = [g for g in todo if g != group]
        domain.sort(key=lambda base:(base != group[0],abs(base-group[0]),base))
        for base in domain:
            assign(group,base,True)
            if search(remaining):
                return True
            assign(group,base,False)
        return False
    try:
        found = search(pending)
    except TimeoutError:
        return None
    if not found:
        return None
    assert set(mapping) == set(members(used))
    assert all(mapping.get(r,r) == r for r in anchors)
    for r in mapping:
        for other in members(edges[r]):
            if other in mapping:
                assert mapping[r] != mapping[other]
    return mapping, dict(search_nodes=nodes, groups=len(groups),
        anchors=sorted(anchors), peak_live=max(incoming[i].bit_count() for i in range(start,end)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--limit",type=int,default=192)
    cli.add_argument("--seconds",type=float,default=12)
    args = cli.parse_args()
    source = (args.reference/"dense-attention-packed.amdgcn").read_text()
    all_lines = source.splitlines()
    stop = all_lines.index("\tv_lshrrev_b32_e32 v65, 4, v112")
    # 输出区有多层控制流和EXEC合并，保持原区全部指令/寄存器不动。
    # 在分析边界保守保留其全部v0:113值，避免把mask不活动lane误判为死值。
    lines = all_lines[:stop] + ["h3_boundary_read " + " ".join(f"v{r}" for r in range(114))]
    incoming,outgoing,reads,writes,successors = liveness(lines,True)
    bounds=[0,lines.index(".LBB0_14:"),lines.index(".Ltmp0:"),
            lines.index(".LBB0_20:"),lines.index(".Ltmp4:"),len(lines)]
    region = {i:j for j,(a,b) in enumerate(zip(bounds,bounds[1:])) for i in range(a,b)}
    interface=set()
    for i,targets in enumerate(successors):
        for target in targets:
            if region[i]!=region[target]:
                interface.update(members(incoming[target]))
    anchors = interface | set(range(114))
    maps,info = [],[]
    for begin,end in zip(bounds,bounds[1:]):
        print(json.dumps(dict(region=[begin,end],peak_live=max(incoming[i].bit_count() for i in range(begin,end)),
                              interface=sorted(interface)),ensure_ascii=False),flush=True)
        result = color(lines,incoming,outgoing,reads,writes,begin,end,anchors,args.limit,args.seconds)
        if result is None:
            raise RuntimeError(f"{args.limit}VGPR约束在本轮搜索预算内没有找到解；不能声称不可解")
        maps.append(result[0]);info.append(result[1])
    helper = runpy.run_path(str(Path(__file__).parents[1]/"attention-query-reload/build.py"))
    rename = helper["rename"]
    changed = [rename(line,maps[region[i]]) for i,line in enumerate(lines[:-1])]
    text = "\n".join(changed+all_lines[stop:])+"\n"
    text = text.replace(".amdhsa_next_free_vgpr 219",f".amdhsa_next_free_vgpr {args.limit}")
    text = text.replace(".vgpr_count:     219",f".vgpr_count:     {args.limit}")
    from songshan_h3.compiler_compat import configure_hygon_compiler
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip","gfx936",64))
    image = backend.make_hsaco(text,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    shutil.copytree(args.reference,args.output)
    (args.output/"dense-attention-packed.amdgcn").write_text(text)
    (args.output/"dense-attention-packed.hsaco").write_bytes(image)
    meta = json.loads((args.reference/"dense-attention-aot-build.json").read_text())
    meta["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    meta["experiment"]["resources"]["vgpr_count"] = args.limit
    meta["experiment"]["pv_stream_compact"] = dict(limit=args.limit,regions=info,
        mappings=[{str(k):v for k,v in m.items()} for m in maps],
        reference_packed_sha256=hashlib.sha256((args.reference/"dense-attention-packed.hsaco").read_bytes()).hexdigest(),
        numerical_verification="尚未验证；静态分析和搜索不代替逐位GPU验证")
    (args.output/"dense-attention-aot-build.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()

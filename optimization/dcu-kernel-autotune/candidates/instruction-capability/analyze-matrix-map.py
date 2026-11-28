#!/usr/bin/env python3
"""用地址标签重建collective映射，并给出当前K/V/P不能直接替换的约束见证。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('capture',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--reference',type=Path,required=True)
    args=cli.parse_args();capture=json.loads(args.capture.read_text())
    packed=args.reference/'dense-attention-packed.hsaco'
    assert hashlib.sha256(packed.read_bytes()).hexdigest()=='a00cc85245e936216531503c3c1dfb57293d00d907dd517b4b1b6c8932f8a3cd'
    assembly=(args.reference/'dense-attention-packed.amdgcn').read_text()
    required=["v_and_b32_e32 v111, 15, v0", "v_and_b32_e32 v112, 48, v0",
        "v_mul_u32_u24_e32 v4, 0x110, v111", "v_xor_b32_e32 v110, v4, v112",
        "v_lshl_or_b32 v14, v111, 6, v14", "v_xor_b32_e32 v135, v14, v106",
        "v_lshlrev_b32_e32 v133, 6, v133", "v_or_b32_e32 v133, v133, v147",
        "v_xor_b32_e32 v147, 32, v133"]
    assert all('\t'+line+'\n' in assembly for line in required)
    data={(x['opcode'],x['pattern']):x['values'][:64] for x in capture['records']}
    result=[]
    for opcode in ['ds_read_m32x16_b16','ds_read_m32x16_b16_alt']:
        zero=data[(opcode,'zero')];wide=data[(opcode,'stride512')]
        source=[]
        for z,w in zip(zero,wide):
            assert all((b-a)%256==0 for a,b in zip(z,w))
            source.append([(b-a)//256 for a,b in zip(z,w)])
        for pattern in ['stride16','stride512-plus64','stride512-xor256']:
            for lane in range(64):
                for item in range(8):
                    index=source[lane][item]
                    address=index*16 if pattern=='stride16' else index*512+64 if pattern=='stride512-plus64' else index*512^256
                    assert data[(opcode,pattern)][lane][item]==zero[lane][item]+address//2
        # 对0、1两lane，所有输出槽均从相同source lane取地址，但只相差1或2个BF16列。
        # 当前K/V/P读回在这两个lane的目标差异远大于该值，任一统一槽置换也不能消除。
        assert source[0]==source[1]
        differences=[b-a for a,b in zip(zero[0],zero[1])]
        assert len(set(differences))==1
        witnesses=[]
        for operand,old_delta in [('K-b128',136),('V-read2st64-b64',32),('P-two-b64',32)]:
            assert old_delta!=differences[0]
            witnesses.append(dict(operand=operand,lanes=[0,1],old_target_delta_in_BF16_elements=old_delta,
                native_delta_in_BF16_elements=differences[0],same_address_source=True,
                uniform_output_slot_permutation_cannot_fix=True))
        result.append(dict(opcode=opcode,source_lanes=source,element_offsets=zero,
            all_five_captured_patterns_match_model=True,direct_replacement_witnesses=witnesses))
    summary=dict(scope='对已捕获地址模式和当前固定lane布局的约束；不是所有重布局方法不可能',
        reference_packed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),
        verified_address_instructions=required,
        model='输出BF16标签 = source_lanes指定lane提供的字节地址/2 + element_offsets',
        records=result,decision='保留现有K/V/P生产者布局时不能直接替换；需要生产者重布局或额外lane交换，收益需另证')
    with args.output.open('x')as stream:json.dump(summary,stream,ensure_ascii=False,indent=2);stream.write('\n')


if __name__=='__main__':main()

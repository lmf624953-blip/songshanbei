#!/usr/bin/env python3
"""读取实际Tensile MessagePack目录，以短名称审核旧搜索覆盖并提取定向候选。"""
import argparse
import hashlib
import json
from pathlib import Path
import msgpack


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('library',type=Path)
    cli.add_argument('old_list',type=Path)
    cli.add_argument('output',type=Path,help='尚不存在的输出目录')
    args=cli.parse_args()
    data=msgpack.unpackb(args.library.read_bytes(),strict_map_key=False)
    solutions=data['solutions']
    assert all(s['problemType']['operationIdentifier']=='Contraction_l_Alik_Bljk_Cijk_Dijk' and
               s['problemType']['aType']=='BFloat16' and s['problemType']['highPrecisionAccumulate'] for s in solutions)
    short={s['info']['SolutionNameMin']:s for s in solutions}
    assert len(short)==len(solutions), '短名称重复，需要重新审计映射'
    old=set(args.old_list.read_text().splitlines())
    novel=[s for name,s in short.items() if name not in old]
    rk1=[s for s in solutions if '_RK1_' in s['name'] and min(s['sizeMapping']['macroTile'][:2])>=64]
    chosen={s['index']:s for s in novel+rk1}
    args.output.mkdir()
    (args.output/'targeted-candidates.tsv').write_text(''.join(f"{i} {s['info']['SolutionNameMin']}\n" for i,s in sorted(chosen.items())))
    record=dict(library=str(args.library.resolve()),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                old_list=str(args.old_list.resolve()),old_list_sha256=hashlib.sha256(args.old_list.read_bytes()).hexdigest(),
                actual_solutions=len(solutions),old_solutions=len(old),old_short_name_intersection=len(old&short.keys()),
                novel_count=len(novel),rk1_large_count=len(rk1),selected_count=len(chosen),
                selection_scope='仅新增候选及两个macroTile维度均>=64的RK1；不是全库搜索',
                index_convention='库index为0-based；rocblas_gemm_algo_solution_index传入index+1，需独立trace核对实际发射',
                selected=[dict(index=i,short=s['info']['SolutionNameMin'],actual=s['name'],size_mapping=s['sizeMapping']) for i,s in sorted(chosen.items())])
    (args.output/'coverage.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('selected',)},ensure_ascii=False))


if __name__=='__main__':main()

import ast,hashlib,json,re,runpy,sys
from pathlib import Path
from statistics import median,mean,stdev
root=Path(sys.argv[1]);code=Path(sys.argv[2]);candidate=code/'optimization/dcu-kernel-autotune/candidates/peer-virtual-input'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (root/'remaining-exit-code').read_text().strip()=='0'
assert (root/'pmc-exit-code').read_text().strip()=='0'
assert (root/'trace-vmm-resume-exit-code').read_text().strip()=='0'
assert sha(candidate/'vmm_gemm.hip')==sha(root/'vmm_gemm-granularity-v3.hip')
def run_ast(p):return ast.dump(next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='run'),include_attributes=False)
assert run_ast(candidate/'bench-vmm-gemm.py')==run_ast(root/'bench-vmm-gemm-v3.py')
measurements={};brief=[]
for gran,name in [(4096,'production-v3-4096.json'),(65536,'production-A-65536.json'),(2097152,'production-A-2097152.json')]:
 data=read(root/name);assert len(data['records'])==2;measurements[str(gran)]=data
 for r in data['records']:
  assert r['rows']==27570 and r['m'] in [2688,3584] and len(r['pairs'])==39 and len(r['validation'])==8
  assert sorted(v['rank'] for v in r['validation'])==list(range(8))
  assert all(all(v[k] for k in ['input_exact','output_exact','output_tail','norm_tail']) for v in r['validation'])
  a=[p['local']['wall_ms'] for p in r['pairs']];b=[p['vmm']['wall_ms'] for p in r['pairs']]
  assert all(p['order']==([0,1] if i%2==0 else [1,0]) for i,p in enumerate(r['pairs']))
  assert abs(median(a)-r['local_median_ms'])<1e-10 and abs(median(b)-r['vmm_median_ms'])<1e-10
  brief.append(dict(granularity=gran,m=r['m'],local_ms=median(a),vmm_ms=median(b),paired_saved_ms=median(x-y for x,y in zip(a,b)),
   paired_reduction_percent=median(100*(1-y/x) for x,y in zip(a,b)),local_cv_percent=stdev(a)/mean(a)*100,vmm_cv_percent=stdev(b)/mean(b)*100,
   local_max_device_median_ms=median(max(p['local']['device_ms']) for p in r['pairs']),vmm_max_device_median_ms=median(max(p['vmm']['device_ms']) for p in r['pairs'])))
pmc={}
for mode in ['local','vmm']:
 records=[]
 for block in re.split(r'(?=kernel-name:)',(root/f'pmc-{mode}.txt').read_text()):
  if not block.strip():continue
  lines=block.splitlines();fields={}
  for line in lines[1:]:
   if not line.strip():continue
   m=re.fullmatch(r'\s*(.*?)\s{2,}(\S+)\s*',line);assert m,line;fields[m[1]]=m[2]
  records.append(dict(kernel=lines[0].split(':',1)[1].strip('"'),fields=fields))
 gemm=[r for r in records if r['kernel'].startswith('Cijk_')];assert len(gemm)==8
 assert sorted(int(r['fields']['gpu id']) for r in gemm)==list(range(8))
 assert all(r['fields']['vgpr count']=='256' and r['fields']['shared memory size']=='65536' and r['fields']['scratch size']=='0' for r in gemm)
 pmc[mode]=dict(txt_sha256=sha(root/f'pmc-{mode}.txt'),records=records)
assert {r['kernel'] for r in pmc['local']['records'] if r['kernel'].startswith('Cijk_')}=={r['kernel'] for r in pmc['vmm']['records'] if r['kernel'].startswith('Cijk_')}
analyze=runpy.run_path(str(code/'optimization/analyze-hipprof-timeline.py'))['summarize_window'];phase_windows=runpy.run_path(str(code/'optimization/summarize-profile.py'))['phase_windows'];timeline={}
for mode in ['local','vmm']:
 sample=read(root/f'trace-profile-{mode}.json');assert sample['diagnostic_profiler'] and len(sample['validation'])==8
 assert all(all(v[k] for k in ['input_exact','output_exact','output_tail','norm_tail']) for v in sample['validation'])
 phases=phase_windows(root/f'trace-{mode}.log');assert len(phases)==1;start,end,name,_=phases[0]
 trace=read(root/f'trace-{mode}_stream.json');events=trace['traceEvents'] if isinstance(trace,dict) else trace
 value,rows=analyze(events,start,end,name,list(range(8)));parts={}
 for rank in range(8):
  rank_rows=[r for r in rows if r['device']==rank]
  assert sum(r['calls'] for r in rank_rows if r['name'].startswith('Cijk_'))==1
  assert sum(r['calls'] for r in rank_rows if 'publish_rows' in r['name'])==(1 if mode=='local' else 0)
  assert sum(r['calls'] for r in rank_rows if r['name']=='wide_norm_indexed_kernel')==1
  parts[str(rank)]={key:sum(r['total_ms'] for r in rank_rows if match(r['name'])) for key,match in [('norm',lambda n:n=='wide_norm_indexed_kernel'),('publish',lambda n:'publish_rows' in n),('gemm',lambda n:n.startswith('Cijk_'))]}
 timeline[mode]=dict(trace_sha256=sha(root/f'trace-{mode}_stream.json'),timeline=value,per_device_kernel_ms=parts,sample=sample)
fresh=read(root/'fresh-B-4096.json');assert len(fresh['records'])==1 and all(v['input_exact'] and v['output_exact'] for v in fresh['records'][0]['validation'])
residency=read(root/'residency.json')
assert (root/'residency-exit-code').read_text().strip()=='0'
for check in residency['records']:
 assert len(set(check['used_delta_bytes']))==1
 assert all(x['retain_status']==x['properties_status']==0 and x['returned_location_type']==1 and x['returned_location_id']==x['requested_owner'] for x in check['properties'])
report=dict(node='f13r1n16',scope='八卡独立九齿norm+发布/就绪+原rocBLAS GEMM微测；不含共同gate，不是模型成绩',decision='当前VMM按需读取方案不采用，保留116.975680秒本地默认',experiment=str(root),
 source_sha256={p.name:sha(p) for p in candidate.iterdir() if p.is_file()},compiled_HIP_source_identical=True,timed_run_AST_identical=True,residency=residency,summary=brief,paired_records=measurements,
 setup_initial_failure=(root/'setup-diagnostic.json').read_text(),setup_v3=[read(root/f'setup-granularity-v3-{g}.json') for g in [4096,65536,2097152]],
 shape_change_failure=dict(sequence='A两种投影后B',observed=(root/'production-v3-4096.log').read_text(),fresh_B_independent_passed=True,root_cause='未定位，不能宣称完整跨形状逐位通过'),fresh_B=fresh,
 pmc=pmc,timeline=timeline,limits=['合成BF16输入，真实生产尺寸，未接入模型；所有A变体逐位通过后39组配对','更换物理页粒度未取得正收益','PMC总体L2命中率不是每操作数命中率，不能据此定量归因远程链路','B跨形状序列存在未解决位差；因A性能明显退化，不继续模型资格'],
 trace_wrapper_check_failure='首个封装脚本文件名变量缺花括号，local采样完成保留，随后仅补vmm采样，原退出码1保留',global_optimality_claimed=False)
(root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(summary=brief,timeline={k:v['per_device_kernel_ms']['0'] for k,v in timeline.items()}),ensure_ascii=False,indent=2))

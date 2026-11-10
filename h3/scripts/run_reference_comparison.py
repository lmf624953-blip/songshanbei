import os,sys,json,time,hashlib,subprocess,traceback,importlib.util,shutil
from pathlib import Path
r=Path(os.environ['IFAB_AFS_HOME']);sys.path.insert(0,str(r));out=r/'outputs/comparison'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True);(r/'artifacts/comparison-current-path.txt').write_text(str(out))
prior=json.loads((r/'outputs/public_fl2va/20261107_165708_direct_infinilm/run.json').read_text());original=prior['command'];existing=Path(next(x.split('=',1)[1] for x in original if x.startswith('--output=')));assert prior['generation_succeeded']
assert hashlib.sha256(existing.read_bytes()).hexdigest()=='5cca4e4ed222413fd348edb30a5aba518060bf4bab4c5a975d1613a0cb79b1e0'
cmd=[original[0]]+original[2:];cmd=[('--output='+str(out/'reference.mp4')) if x.startswith('--output=') else x for x in cmd];cmd+=['--legacy','--legacy-memory-fraction=0.4']
rec=dict(phase='reference_generation',reference_command=cmd,infinilm_command=original,comparison_complete=False,official_pass_claimed=False)
def save(): (out/'comparison-run.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2))
save();print('REFERENCE_RUN_DIR',out,flush=True);start=time.monotonic()
try:
 import torch
 assert torch.cuda.device_count()==8
 rec['hardware']=[dict(device=i,name=torch.cuda.get_device_properties(i).name,arch=torch.cuda.get_device_properties(i).gcnArchName,memory=torch.cuda.get_device_properties(i).total_memory) for i in range(8)];save()
 with (out/'reference.log').open('w') as f: ret=subprocess.run(cmd,cwd=r/'h3-runtime/sources/InfiniLM',stdout=f,stderr=subprocess.STDOUT)
 rec.update(reference_returncode=ret.returncode,reference_wall_seconds=time.monotonic()-start);save();assert ret.returncode==0,ret.returncode
 from songshan_h3.output_validation import validate_generated_media
 rec['reference_media']=validate_generated_media(out/'reference.mp4',dict(num_frames=124,fps=24,height=768,width=1344));assert rec['reference_media']['video_specs_passed']
 rec['phase']='comparison';save();compare=r/'h3-runtime/sources/InfiniLM/examples/minimax_h3_compare.py';rec['comparison_script_sha256']=hashlib.sha256(compare.read_bytes()).hexdigest()
 cc=[sys.executable,str(compare),'--tasks=fl2va','--model-path=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','--reference-fl2va-video='+str(out/'reference.mp4'),'--infinilm-fl2va-video='+str(existing),'--output-dir='+str(out/'official_compare')]
 with (out/'official-compare.log').open('w') as f: cr=subprocess.run(cc,stdout=f,stderr=subprocess.STDOUT)
 rec['official_compare_returncode']=cr.returncode;assert cr.returncode==0,cr.returncode
 spec=importlib.util.spec_from_file_location('official_h3_compare',compare);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 metrics=mod.compare_video_streams(out/'reference.mp4',existing);(out/'similarity.json').write_text(json.dumps(metrics,indent=2));rec.update(phase='complete',comparison_complete=True,metrics=metrics,reference_sha256=hashlib.sha256((out/'reference.mp4').read_bytes()).hexdigest());save();print('SIMILARITY_COMPLETE',json.dumps(metrics),flush=True)
except BaseException:
 rec.update(phase='failed',error=traceback.format_exc());save();print(rec['error'],flush=True)
finally:
 dest=Path(os.environ['H3_EXPORT_DIR'])
 for name in ('comparison-run.json','reference.log','official-compare.log','similarity.json'):
  if (out/name).exists():shutil.copyfile(out/name,dest/('h3-'+name))
 print('FINAL_PHASE',rec['phase'],flush=True)
sys.exit(0 if rec['comparison_complete'] else 1)

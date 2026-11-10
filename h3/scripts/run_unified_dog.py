import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
R=Path(os.environ['IFAB_AFS_HOME']);sys.path.insert(0,str(R))
O=R/'outputs/unified_fl2va'/time.strftime('%Y%m%d_%H%M%S');O.mkdir(parents=True,exist_ok=False)
(R/'artifacts/unified-dog-current-path.txt').write_text(str(O)+'\n')
D=dict(phase='preflight',complete=False,official_pass_claimed=False,prompt='A dog falls asleep',output_directory=str(O))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
 (O/'run.json').write_text(json.dumps(D,indent=2)+'\n');print('PHASE',D['phase'],flush=True)
def run(cmd,name,env,limit):
 D['phase']=name;D[name+'_command']=cmd;save();start=time.monotonic()
 with (O/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
 D[name+'_returncode']=p.returncode;D[name+'_wall_seconds']=time.monotonic()-start;save()
 if p.returncode:raise RuntimeError(name+' failed; see '+str(O/(name+'.log')))
def main():
 try:
  image=R/'unified-dog-input/first_frame.png';D['input_sha256']=sha(image)
  assert D['input_sha256']=='b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54'
  import torch
  assert torch.cuda.is_available() and torch.cuda.device_count()==8
  D['hardware']=[]
  for i in range(8):
   p=torch.cuda.get_device_properties(i);assert 'gfx936' in p.gcnArchName
   D['hardware'].append(dict(device=i,name=p.name,arch=p.gcnArchName,memory=p.total_memory))
  from PIL import Image
  from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
  with Image.open(image) as im:size=im.size
  h,w=resolve_canvas_size(*size,32,768,768*1344)
  config=dict(height=h,width=w,num_frames=124,fps=24,steps=50,seed=42,gpus=list(range(8)))
  D['input_size']=size;D['configuration']=config
  entry=R/'h3-runtime/sources/InfiniLM/examples/minimax_h3.py';compare=entry.with_name('minimax_h3_compare.py')
  D['comparison_script_sha256']=sha(compare);assert D['comparison_script_sha256']=='e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590'
  D['entry_sha256']=sha(entry)
  D['adaptation_hashes']={str(p.relative_to(R)):sha(p) for p in (R/'songshan_h3').glob('*') if p.is_file() and p.suffix in ('.py','.cc','.so')}
  base=[str(R/'.venv/bin/python'),str(entry),'fl2va','--model-path=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','--first-frame='+str(image),'--text=A dog falls asleep','--gpus=0,1,2,3,4,5,6,7','--num-frames=124','--steps=50','--seed=42','--height='+str(h),'--width='+str(w)]
  env=dict(os.environ,PATH=str(R/'.venv/bin')+os.pathsep+os.environ['PATH'],PYTHONPATH=str(R),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TRITON_CACHE_MANAGER='songshan_h3.cache_compat:SameDirectoryCacheManager')
  from songshan_h3.output_validation import validate_generated_media
  run(base[:1]+[str(R/'scripts/launch_h3.py')]+base[1:]+['--output='+str(O/'infinilm.mp4')],'infinilm',env,1800)
  D['infinilm_media']=validate_generated_media(O/'infinilm.mp4',config);assert D['infinilm_media']['video_specs_passed'];D['infinilm_sha256']=sha(O/'infinilm.mp4');save()
  refenv=dict(env);refenv['PYTHONPATH']=os.pathsep.join([str(R/'.reference-runtime/python'),str(R/'.reference-runtime/flash-attn'),str(R)])
  refenv['LD_LIBRARY_PATH']=str(R/'.reference-runtime/python/torch/lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
  run(base+['--output='+str(O/'reference.mp4'),'--legacy','--legacy-memory-fraction=0.4'],'reference',refenv,1800)
  D['reference_media']=validate_generated_media(O/'reference.mp4',config);assert D['reference_media']['video_specs_passed'];D['reference_sha256']=sha(O/'reference.mp4');save()
  cmd=[str(R/'.venv/bin/python'),str(compare),'--tasks=fl2va','--model-path=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','--reference-fl2va-video='+str(O/'reference.mp4'),'--infinilm-fl2va-video='+str(O/'infinilm.mp4')]
  run(cmd,'official-compare',env,240)
  spec=importlib.util.spec_from_file_location('dog_official_compare',compare);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  metrics=module.compare_video_streams(O/'reference.mp4',O/'infinilm.mp4');(O/'similarity.json').write_text(json.dumps(metrics,indent=2)+'\n')
  D['metrics']=metrics;D['complete']=True;D['phase']='complete';save();print('UNIFIED_DOG_COMPLETE',json.dumps(metrics),flush=True)
  return 0
 except BaseException:
  D['phase']='failed';D['error']=traceback.format_exc();save();print(D['error'],flush=True);return 1
 finally:
  dest=Path(os.environ['H3_EXPORT_DIR'])
  for name in ('run.json','infinilm.log','reference.log','official-compare.log','similarity.json','infinilm.mp4','reference.mp4'):
   p=O/name
   if p.is_file():shutil.copyfile(p,dest/('dog-'+name))
if __name__=='__main__':raise SystemExit(main())

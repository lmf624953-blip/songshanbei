"""Fresh native compilation: consumes only a checksum-pinned source archive."""
import os,sys,pathlib,subprocess,tarfile,tempfile,time,json,hashlib,traceback,shutil,sysconfig
P=pathlib.Path
root=P(os.environ['IFAB_AFS_HOME']).resolve(); os.chdir(root)
assert P(sys.prefix).resolve()==root/'.venv'
assert P('/dev/kfd').exists() and P('/opt/dtk/env.sh').exists(), 'DCU compute node required'
log=(root/'artifacts/clean-native.log').open('a',buffering=1)
status={'complete':False,'started':time.time()}
def note(*x):print(time.strftime('%FT%T'),*x,file=log,flush=True)
raw=subprocess.check_output(['bash','-c','source "$IFAB_AFS_HOME/scripts/activate.sh" >/dev/null && env -0'])
env={k.decode():v.decode() for x in raw.split(bytes([0])) if b'=' in x for k,v in [x.split(b'=',1)]}
env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',PYTHONNOUSERSITE='1')
env['PATH']=str(root/'.venv/bin')+':'+env['PATH'];env.pop('PYTHONPATH',None)
scratch=P(tempfile.mkdtemp(prefix='h3-clean-native-'))
(root/'artifacts/compute-scratch-current.txt').write_text(str(scratch));status['scratch']=str(scratch)
env['TMPDIR']=str(scratch)
def run(cmd,cwd=None,timeout=2700):
 status['phase']=' '.join(map(str,cmd));note('RUN',status['phase'])
 (root/'artifacts/clean-native-status.json').write_text(json.dumps(status,indent=2))
 subprocess.run(list(map(str,cmd)),cwd=cwd or root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
 note('OK',status['phase'])
try:
 sources=scratch/'sources'
 for name in ('InfiniCore','InfiniLM','xmake'):
  origin=P(os.environ['H3_VENDORED_SOURCE_ROOT'])/name
  assert origin.is_dir(), 'Missing public vendored source: '+name
  manifest=json.loads((root/'RELEASE-MANIFEST.json').read_text())
  prefix='frameworks/'+name+'/'
  for relative,item in manifest.items():
   if not relative.startswith(prefix):continue
   source=origin/relative[len(prefix):]
   assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==item['sha256'],relative
  shutil.copytree(origin,sources/name)
  for relative,item in manifest.items():
   if relative.startswith(prefix) and item.get('executable'):
    (sources/name/relative[len(prefix):]).chmod(0o755)
 for p in sources.rglob('*.so'):raise RuntimeError('Prebuilt shared library in source: '+str(p))
 env['XMAKE_PROGRAM_DIR']=str(sources/'xmake/xmake')
 run(['bash','configure'],sources/'xmake',600);run(['make','-j96'],sources/'xmake',1200)
 xmake=sources/'xmake/build/xmake';run([xmake,'--version'],timeout=30)
 env['PATH']=str(root/'.venv/bin')+':'+str(xmake.parent)+':'+env['PATH']
 def py(code):return subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()
 abi=py('import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
 env.update(INFINILM_CXX11_ABI=abi,SONGSHAN_PYBIND11_INCLUDE=py('import pybind11;print(pybind11.get_include())'),SONGSHAN_PYTHON_INCLUDE=py('import sysconfig;print(sysconfig.get_paths()["include"])'))
 core=sources/'InfiniCore';lm=sources/'InfiniLM'
 run([xmake,'f','-y','--ccache=n','--hygon-dcu=y','--hygon-arch=gfx936','--nv-gpu=n','--ccl=y','--aten=y','--cudnn=n','--ninetoothed=n','--omp=n','--cuda='+env['CUDA_HOME'],'--cc='+env['CC'],'--cxx='+env['CXX'],'--cxflags=-D_GLIBCXX_USE_CXX11_ABI='+abi,'-m','release'],core,180)
 run([xmake,'build','-y','-j96'],core)
 # Xmake's file-copy syscall is not implemented by this AFS gateway. Install
 # into the node's writable filesystem, then materialize bytes in personal AFS.
 env['INFINI_ROOT']=str(scratch/'local-prefix-install')
 run([xmake,'install','-y'],core,600)
 for item in (scratch/'local-prefix-install').rglob('*'):
  if item.is_file():
   target=root/'.infini'/item.relative_to(scratch/'local-prefix-install')
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.is_symlink():target.unlink()
   data=item.read_bytes();target.write_bytes(data)
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(data).digest()
 env['INFINI_ROOT']=str(root/'.infini')
 env['SONGSHAN_USE_INSTALLED_CORE']='1'
 run([xmake,'build','-y','-j96','_infinicore'],core,900);run([xmake,'install','-y','_infinicore'],core,300)
 run([xmake,'f','-y','--ccache=n','-m','release','--cxx11-abi='+abi,'--cc='+env['CC'],'--cxx='+env['CXX']],lm,180)
 run([xmake,'build','-y','-j96','_infinilm'],lm,1200);run([xmake,'install','-y','_infinilm'],lm,300)
 suffix=sysconfig.get_config_var('EXT_SUFFIX')
 for name in ('InfiniCore','InfiniLM'):
  package=name.lower();src=sources/name/'python'/package
  assert (src/'lib'/('_'+package+suffix)).is_file(), 'Wrong Python extension ABI'
  target=root/'.venv/lib/python3.11/site-packages'/package
  assert not target.exists(), 'Fresh package install required'
  shutil.copytree(src,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
  (root/'artifacts'/(name+'-fresh-install.json')).write_text(json.dumps(files,indent=2))
 libdir=root/'.infini/lib'
 for p in list(libdir.rglob('*')):
  if p.is_symlink() and p.resolve().is_file():
   data=p.read_bytes();p.unlink();p.write_bytes(data)
 library_hashes={str(p.relative_to(libdir)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                 for p in libdir.rglob('*.so*') if p.is_file()}
 assert library_hashes, 'No persistent native shared libraries installed'
 (root/'artifacts/fresh-libraries.json').write_text(json.dumps(library_hashes,indent=2))
 # The runnable entry is a separately hash-verified export from the successful
 # platform run. Keep the pristine checkout and native build tree separate.
 runtime=root/'h3-runtime/sources/InfiniLM/examples'
 runtime.mkdir(parents=True,exist_ok=True)
 for name in ('minimax_h3.py','minimax_h3_compare.py'):
  source=root/'runtime_entry'/name
  target=runtime/name
  if target.exists() and target.read_bytes()!=source.read_bytes():
   raise RuntimeError('Different existing runtime entry retained: '+str(target))
  if not target.exists():shutil.copyfile(source,target)
 state=py('import torch,triton,infinicore,infinilm,ninetoothed,ntops,diffusers,json;print(json.dumps({m.__name__:m.__file__ for m in [torch,triton,infinicore,infinilm,ninetoothed,ntops,diffusers]}))')
 (root/'artifacts/fresh-module-paths.json').write_text(state)
 note('FRESH_MODULES',state);status['complete']=True;note('CLEAN_NATIVE_OK')
except BaseException as e:
 status['error']=repr(e);traceback.print_exc(file=log)
finally:
 (root/'artifacts/clean-native-status.json').write_text(json.dumps(status,indent=2));log.close()

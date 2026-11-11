import os,pathlib,venv,ensurepip,zipfile,subprocess,json,hashlib,tarfile,time,traceback
root=pathlib.Path(os.environ['IFAB_AFS_HOME']); base=pathlib.Path(os.environ['H3_INPUT_CACHE']); run=root/'artifacts'
status={'complete':False,'phase':'venv'}
env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('VIRTUAL_ENV',None)
env.update(PYTHONNOUSERSITE='1',IFAB_AFS_HOME=str(root),SONGSHAN_STORAGE=str(root/'h3-runtime'))
(root/'.tmp').mkdir(exist_ok=True);env['TMPDIR']=str(root/'.tmp')
def check(cmd):
 print('RUN',*map(str,cmd),flush=True); subprocess.run(list(map(str,cmd)),cwd=root,env=env,check=True)
try:
 venv.EnvBuilder(system_site_packages=True,with_pip=False,symlinks=True).create(root/'.venv')
 site=root/'.venv/lib/python3.11/site-packages'
 bundled=next((pathlib.Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
 with zipfile.ZipFile(bundled) as z: z.extractall(site)
 py=root/'.venv/bin/python'
 pipshim=root/'artifacts/pip_personal.py';pipshim.parent.mkdir(exist_ok=True)
 pipshim.write_text("import pathlib,sys,mimetypes\nmimetypes.knownfiles=[]\nmimetypes.init()\nassert pathlib.Path(sys.prefix).resolve()==pathlib.Path(__file__).resolve().parents[1]/'.venv'\nimport pip._internal.network.session as s\ns.user_agent=lambda:'songshan-clean-rebuild'\nfrom pip._internal.cli.main import main\nraise SystemExit(main(sys.argv[1:]))\n")
 check([py,root/'verify_source.py'])
 vendor_name='triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
 cached_vendor=base/'downloads/vendor'/vendor_name
 if cached_vendor.is_file():
  assert cached_vendor.stat().st_size==105484167
  assert hashlib.file_digest(cached_vendor.open('rb'),'sha256').hexdigest()=='e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581'
  vendor_target=root/'downloads/vendor'/vendor_name
  vendor_target.parent.mkdir(parents=True,exist_ok=True)
  with cached_vendor.open('rb') as source, vendor_target.open('wb') as target:
   while data:=source.read(1024*1024):target.write(data)
 check([py,root/'download_vendor_triton.py'])
 wheel=next((root/'downloads/vendor').glob('triton-*.whl'))
 check([py,pipshim,'install','--no-index','--no-deps','--no-compile',wheel])
 check([py,root/'apply_triton_compat_memory.py'])
 check([py,root/'scripts/prepare_triton_compat.py','--check'])
 status['phase']='python_dependencies'
 wheels=root/'downloads/runtime';wheels.mkdir(parents=True,exist_ok=True)
 archives={1:'25b8bd54fdfef867ee1b224234f1aeb97b65e1389ed56a1e837c72f40483d73d',2:'52cfefbe37c2957d5b7480b774208d38407844313deaf845726e987d2f85c3df'}
 manifest={}
 for part,sha in archives.items():
  archive=base/f'artifacts/h3-runtime-wheels-part{part}-20261107.tar'
  assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==sha
  with tarfile.open(archive) as tf:
   for m in tf:
    assert m.isfile() and pathlib.PurePosixPath(m.name).name==m.name
    data=tf.extractfile(m).read()
    if m.name=='SHA256SUMS.json':manifest.update(json.loads(data))
    else:(wheels/m.name).write_bytes(data)
 for name,sha in manifest.items():assert hashlib.sha256((wheels/name).read_bytes()).hexdigest()==sha,name
 check([py,pipshim,'install','--no-index','--no-deps','--no-compile','--ignore-installed',*[wheels/n for n in sorted(manifest)]])
 status['phase']='pure_python_sources'
 archive=base/'artifacts/h3-pinned-frameworks-20261107.tar.gz'
 assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=='6f468d855e5d9d63f1f74b1b580f1f72a79830e711d59abceefebf1a2781e0a7'
 sources=root/'h3-runtime/sources'
 with tarfile.open(archive) as tf:
  for m in tf:
   parts=pathlib.PurePosixPath(m.name).parts
   assert not m.name.startswith('/') and '..' not in parts
   if parts[0] not in ('ninetoothed','ntops','diffusers') or not m.isfile():continue
   if len(parts)>1 and parts[1] in ('docs','examples','tests','.github','benchmarks'):continue
   target=sources/m.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
 for name in ('ninetoothed','ntops','diffusers'):
  if name=='diffusers':
   out=root/'downloads/built';out.mkdir(exist_ok=True)
   subprocess.run([str(py),'setup.py','bdist_wheel','--dist-dir='+str(out)],cwd=sources/name,env=env,check=True)
   check([py,pipshim,'install','--no-index','--no-deps','--no-compile',next(out.glob('diffusers-*.whl'))])
  else:check([py,pipshim,'install','--no-index','--no-deps','--no-build-isolation','--no-compile',sources/name])
 vision=next((base/'artifacts/offline-tools').glob('torchvision-0.24.0+cpu*.whl'))
 assert hashlib.file_digest(vision.open('rb'),'sha256').hexdigest()=='51fe8d72ddcc88a50ff27c50dd4987d82fba8b0dfd591898ee1f60e2902d18d1'
 check([py,pipshim,'install','--no-index','--no-deps','--no-compile',vision])
 flash=next((base/'downloads').glob('flash_attn-2.8.3+das.opt1.dtk2604.torch290*.whl'))
 assert hashlib.file_digest(flash.open('rb'),'sha256').hexdigest()=='24d4021bf4ace04cf3c417f60affebe134128c0626a5b136230989812e66c18a'
 dest=root/'.reference-runtime/flash-attn';dest.mkdir(parents=True,exist_ok=False)
 with zipfile.ZipFile(flash) as z:
  for name in z.namelist():assert not name.startswith('/') and '..' not in pathlib.PurePosixPath(name).parts
  z.extractall(dest)
 check([py,root/'setup_reference_torch_view.py'])
 image=base/'unified-dog-input/first_frame.png'
 assert hashlib.sha256(image.read_bytes()).hexdigest()=='b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54'
 (root/'unified-dog-input').mkdir(exist_ok=True);(root/'unified-dog-input/first_frame.png').write_bytes(image.read_bytes())
 check([py,root/'scripts/check_inference_dependencies.py'])
 check([py,root/'verify_source.py'])
 status['complete']=True;print('CPU_SETUP_OK',flush=True)
except BaseException as exc:
 status['error']=repr(exc);traceback.print_exc()
finally:
 (run/'rebuild-prepare-status.json').write_text(json.dumps(status,indent=2));print('STATUS',status,flush=True)

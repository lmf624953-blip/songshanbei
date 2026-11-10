from pathlib import Path
import urllib.request,hashlib,zipfile,json
r=Path.cwd();u='https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl';p=r/'downloads'/u.rsplit('/',1)[1]
if not p.exists():
 with urllib.request.urlopen(u,timeout=30) as response,p.open('wb') as f:
  while True:
   b=response.read(8*1024*1024)
   if not b:break
   f.write(b)
print('WHEEL_DOWNLOADED',p.stat().st_size,flush=True);assert p.stat().st_size==658898543
h=hashlib.sha256(p.read_bytes()).hexdigest();dest=r/'.reference-runtime/flash-attn';dest.mkdir(parents=True,exist_ok=True,mode=0o770)
with zipfile.ZipFile(p) as z:
 for n in z.namelist():
  if n.endswith('/') or '..' in Path(n).parts:continue
  q=dest/n;q.parent.mkdir(parents=True,exist_ok=True,mode=0o770);q.write_bytes(z.read(n))
(r/'artifacts/reference-flash-provenance.json').write_text(json.dumps(dict(source_url=u,wheel_sha256=h,bytes=p.stat().st_size,reference_only_path=str(dest),formal_venv_unchanged=True),indent=2));print('REFERENCE_FLASH_READY',dest,h,flush=True)

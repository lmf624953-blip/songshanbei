#!/usr/bin/env python3
"""从实际部署的 libinfiniop.so 提取 RMSNorm 设备 ELF，保留可核验的来源。"""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('library',type=Path);cli.add_argument('output',type=Path);cli.add_argument('--objdump',type=Path,required=True);a=cli.parse_args()
    a.output.mkdir();data=a.library.read_bytes();start=0
    while True:
        offset=data.find(b'\x7fELF',start)
        if offset<0:raise RuntimeError('未找到包含 RMSNorm 的 AMDGPU ELF')
        start=offset+4
        if not offset or data[offset+18:offset+20]!=b'\xe0\x00':continue
        shoff=struct.unpack_from('<Q',data,offset+40)[0];entry,count=struct.unpack_from('<HH',data,offset+58);size=shoff+entry*count
        image=data[offset:offset+size]
        if b'_Z13rmsnormKernel' in image:break
    target=a.output/'rmsnorm.hsaco';target.write_bytes(image)
    with (a.output/'rmsnorm.s').open('w') as stream:subprocess.run([str(a.objdump),'-d','--mcpu=gfx936',str(target)],stdout=stream,check=True)
    (a.output/'source.json').write_text(json.dumps(dict(library=str(a.library.resolve()),library_sha256=hashlib.sha256(data).hexdigest(),file_offset=offset,elf_bytes=size,elf_sha256=hashlib.sha256(image).hexdigest(),scope='仅取证，不替代九齿生产算子'),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()

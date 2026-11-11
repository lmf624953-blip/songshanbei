"""Fetch checksum-pinned original rebuild inputs without installing packages."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import urllib.parse
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, required=True)
a = p.parse_args()
lock = json.loads(Path(__file__).with_name('inputs.lock.json').read_text())
base = 'https://github.com/lmf624953-blip/songshanbei/releases/download/h3-source-cache-v1/'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

for name, item in lock.items():
    target = a.destination / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            raise SystemExit('Existing input checksum mismatch; retained unchanged: '+str(target))
        print('VERIFIED_EXISTING', name, flush=True)
        continue
    partial = target.with_name(target.name+'.download')
    url = item.get('url') or base + urllib.parse.quote(name)
    request = urllib.request.Request(url, headers={'User-Agent': 'songshan-h3-rebuild'})
    try:
        with urllib.request.urlopen(request, timeout=120) as r, partial.open('wb') as f:
            while data := r.read(1024*1024):
                f.write(data)
    except OSError as exc:
        raise SystemExit(
            'Unable to fetch '+url+'\n'+str(exc)+'\n'
            'On a network-enabled machine run this downloader, upload the cache '
            'with the same relative paths, then retry without --download-inputs. '
            'Completed files are verified and reused; partial files are not accepted.')
    if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']:
        raise SystemExit('Downloaded input checksum mismatch: '+str(partial))
    os.replace(partial, target)
    print('DOWNLOADED_VERIFIED', name, flush=True)
print('RAW_INPUT_CACHE_READY', a.destination, flush=True)

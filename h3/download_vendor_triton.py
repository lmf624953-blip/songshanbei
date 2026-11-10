#!/usr/bin/env python3
"""Download and SHA256-check the exact Hygon wheel; never install packages."""
import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

FILENAME = 'triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
URL = 'https://pypi.sourcefind.cn/das_release/dtk2604-rc1/+f/e39/7b16365da2250/' + FILENAME
SHA256 = 'e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'downloads/vendor')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / FILENAME
    if target.exists():
        if digest(target) != SHA256:
            raise SystemExit('Existing wheel has wrong SHA256; retained for inspection: ' + str(target))
        print('SHA256 OK (reused):', target)
        return
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output_dir, prefix=FILENAME + '.',
                                         suffix='.part', delete=False) as output:
            temp = Path(output.name)
            with urlopen(URL, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
        actual = digest(temp)
        if actual != SHA256:
            raise SystemExit('SHA256 mismatch: expected ' + SHA256 + ', got ' + actual)
        temp.replace(target)
        print('SHA256 OK:', SHA256)
        print('Downloaded:', target, '(' + str(target.stat().st_size) + ' bytes)')
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


if __name__ == '__main__':
    main()

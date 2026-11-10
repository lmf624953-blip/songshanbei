#!/usr/bin/env python3
"""Install native NineToothed dispatch, then execute the upstream H3 entry."""
from pathlib import Path
import faulthandler
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
faulthandler.enable(all_threads=True)
from songshan_h3.attention_bridge import install
from songshan_h3.diffusers_attention import install_video_attention

if len(sys.argv) < 2:
    raise SystemExit('Usage: launch_h3.py /path/to/official/minimax_h3.py [arguments]')
entry = Path(sys.argv[1]).resolve()
sys.argv = [str(entry), *sys.argv[2:]]
install()
install_video_attention()
print('Hygon native attention dispatcher: NineToothed JIT', flush=True)
print('H3 video VAE attention: NineToothed JIT', flush=True)
runpy.run_path(str(entry), run_name='__main__')

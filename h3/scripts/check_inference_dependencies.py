#!/usr/bin/env python3
"""Check the installed H3 auxiliary stack without loading model weights."""
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import time
import traceback

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
report = dict(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'), packages={},
              dependency_errors=[], checks={}, inference_passed=False)
names = ('torch', 'triton', 'numpy', 'transformers', 'tokenizers', 'huggingface-hub',
         'diffusers', 'accelerate', 'ninetoothed', 'ntops', 'av', 'imageio',
         'imageio-ffmpeg', 'pybind11', 'hatchling', 'scipy', 'scikit-image',
         'einops', 'sentencepiece', 'ftfy', 'janus', 'xxhash', 'torchvision')
for name in names:
    try:
        dist = md.distribution(name)
        report['packages'][name] = dist.version
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            try:
                actual = md.version(req.name)
                if req.specifier and actual not in req.specifier:
                    report['dependency_errors'].append(f'{name}: {req}; installed {actual}')
            except md.PackageNotFoundError:
                report['dependency_errors'].append(f'{name}: missing {req}')
    except md.PackageNotFoundError:
        report['packages'][name] = None
        report['dependency_errors'].append(f'Missing package: {name}')

# Check the selected constraints separately from upstream package metadata.
for line in (ROOT / 'env/inference-constraints.txt').read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'):
        continue
    req = Requirement(line)
    try:
        actual = md.version(req.name)
        if actual not in req.specifier:
            report['dependency_errors'].append(f'Constraint {req}; installed {actual}')
    except md.PackageNotFoundError:
        report['dependency_errors'].append(f'Constraint {req}; package missing')


def check(name, function):
    try:
        detail = function()
        report['checks'][name] = dict(passed=True, detail=detail)
    except Exception:
        report['checks'][name] = dict(passed=False, error=traceback.format_exc())


def vendor_stack():
    import torch
    assert torch.version.__hcu_version__ == '2.9.0+das.opt1.dtk2604'
    assert md.version('torch') == '2.9.0+das.opt1.dtk2604'
    assert md.version('triton') == '3.5.1+das.opt1.dtk2604.torch290'
    return dict(torch=md.version('torch'), triton=md.version('triton'))


def hub_api():
    from huggingface_hub import resolve_revision
    from huggingface_hub.errors import RevisionResolutionError
    assert callable(resolve_revision)
    return 'resolve_revision and RevisionResolutionError available'


def qwen_api():
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    assert callable(Qwen3VLProcessor.create_mm_token_type_ids)
    return 'Qwen tokenizer, model and multimodal processor API available'


def h3_api():
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference,
    )
    from diffusers.utils.export_utils import encode_video
    return 'H3 model, pipeline, references and encode_video imports passed'


def helper_imports():
    for name in ('ninetoothed', 'ntops', 'av', 'accelerate'):
        importlib.import_module(name)
    import imageio_ffmpeg
    path = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return dict(ffmpeg=path)


def infinilm_auxiliary_api():
    import asyncio
    import janus
    import xxhash

    async def queue_check():
        queue = janus.Queue()
        queue.sync_q.put('h3-import-check')
        assert await queue.async_q.get() == 'h3-import-check'
        queue.async_q.task_done()
        await queue.aclose()

    asyncio.run(queue_check())
    assert len(xxhash.xxh3_128(b'h3-import-check').digest()) == 16
    return 'Janus sync/async queue and xxhash.xxh3_128 API available'


check('vendor_stack', vendor_stack)
check('hub_api', hub_api)
check('qwen_api', qwen_api)
check('h3_api', h3_api)
check('helper_imports', helper_imports)
check('infinilm_auxiliary_api', infinilm_auxiliary_api)
import sys
sys.path.insert(0, str(ROOT))
from songshan_h3.preprocessing import check_vision_api
check('vision_api', check_vision_api)
report['passed'] = not report['dependency_errors'] and all(
    result['passed'] for result in report['checks'].values())
output = ROOT / 'artifacts/inference-dependencies-current.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f'H3 dependency check: {"PASS" if report["passed"] else "FAIL"}; report: {output}')
raise SystemExit(0 if report['passed'] else 1)

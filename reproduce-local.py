#!/usr/bin/env python3
"""Deploy and run the pinned submission on the current cluster account."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path('/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3')
DTK_ROOT = Path('/public/software/compiler/dtk-26.04')
GCC_ROOT = Path('/public/software/compiler/gcc-11.4.0')
REFERENCE_WHEELS = {
    'iniconfig-2.1.0-py3-none-any.whl': '9deba5723312380e77435581c6bf4935c94cbfab9b1ed33ef8d238ea168eb760',
    'pytest-8.3.5-py3-none-any.whl': 'c69214aa47deac29fad6c2a4f590b9c4a9fdb16a403176fe154b79c0b4d4d820',
}


def replace_once(source, original, updated):
    if source.count(original) != 1:
        raise RuntimeError(f'Unexpected pinned source near {original!r}')
    return source.replace(original, updated, 1)


def execute(source, filename, arguments):
    sys.argv = [str(filename), *arguments]
    exec(compile(source, str(filename), 'exec'), {
        '__file__': str(filename), '__name__': '__main__',
    })


def deploy(source, runtime):
    prepare = (source / 'rebuild/prepare.py').read_text()
    start = prepare.index(' flash=next(')
    end = prepare.index(' image=base/', start)
    reference_setup = prepare[start:end]
    (runtime / 'artifacts/rebuild-prepare.py').write_text(prepare[:start] + prepare[end:])
    reference_setup = replace_once(
        reference_setup,
        " check([py,root/'setup_reference_torch_view.py'])",
        " subprocess.run([str(py), str(root/'artifacts/rebuild-reference-torch.py')],"
        " cwd=root, check=True)",
    )
    (runtime / 'artifacts/rebuild-reference.py').write_text(
        'import hashlib, os, pathlib, subprocess, sys, zipfile\n'
        'root = pathlib.Path(os.environ["IFAB_AFS_HOME"])\n'
        'base = pathlib.Path(os.environ["H3_INPUT_CACHE"])\n'
        'py = pathlib.Path(sys.executable)\n'
        + '\n'.join(line[1:] for line in reference_setup.splitlines()) + '\n'
    )
    reference_torch = (source / 'setup_reference_torch_view.py').read_text()
    reference_torch = replace_once(
        reference_torch,
        'source = Path("/usr/local/lib/python3.11/site-packages/torch")',
        'from importlib.util import find_spec\n'
        'source = Path(find_spec("torch").origin).resolve().parent',
    )
    (runtime / 'artifacts/rebuild-reference-torch.py').write_text(reference_torch)

    machine = runtime / 'env/machine.local.sh'
    machine.write_text(
        f'export DTK_ROOT={DTK_ROOT}\n'
        f'export SONGSHAN_GCC_ROOT={GCC_ROOT}\n'
        'export HYGON_ARCH=gfx936\n'
    )
    native = runtime / 'artifacts/rebuild-native.py'
    native.write_text(replace_once(
        native.read_text(), "P('/opt/dtk/env.sh')", f"P('{DTK_ROOT}/env.sh')",
    ))

    accept = runtime / 'artifacts/rebuild-accept.py'
    accept.write_text(replace_once(
        accept.read_text(),
        "    [py, root/'scripts/run_unified_dog.py'],",
        f"    [py, {str(Path(__file__).resolve())!r}, '--runtime', str(root), 'generate'],",
    ))


def prepare_reference(runtime, cache):
    # The Hygon FlashAttention package imports pytest when it initializes.
    artifacts = runtime / 'artifacts'
    ready = artifacts / 'reference-runtime-ready'
    if ready.is_file():
        return
    paths = []
    for filename, expected_hash in REFERENCE_WHEELS.items():
        wheel = cache / 'artifacts/local-wheels' / filename
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Reference dependency checksum mismatch: {wheel}')
        paths.append(wheel)
    python = runtime / '.venv/bin/python'
    view_ready = artifacts / 'reference-torch-view-ready'
    if not view_ready.is_file():
        subprocess.run([python, artifacts / 'rebuild-reference.py'], cwd=runtime, check=True)
        view_ready.touch()
    subprocess.run([
        python, artifacts / 'pip_personal.py', 'install', '--no-index', '--no-deps',
        '--no-compile', '--target', runtime / '.reference-runtime/python', *paths,
    ], cwd=runtime, check=True)
    ready.touch()


def retry_reference(source, run, reference_model):
    """Keep this run's validated InfiniLM video while restarting its reference."""
    record_path = run / 'run.json'
    record = json.loads(record_path.read_text())
    if record.get('phase') != 'failed' or record.get('infinilm_returncode') != 0:
        raise RuntimeError('Expected a failed attempt with a completed InfiniLM video')
    if (run / 'reference.mp4').exists() or (run / 'reference-shared-disk.log').exists():
        raise RuntimeError('Reference retry would overwrite an existing artifact')
    stage = json.loads((run / 'reference-weights-staging.json').read_text())
    if Path(stage['destination']) != reference_model or stage['source'] != os.environ['H3_LOCAL_MODEL']:
        raise RuntimeError('Reference staging manifest does not match the model paths')
    if not (reference_model / 'model_index.json').is_file():
        raise FileNotFoundError(reference_model / 'model_index.json')
    (run / 'reference.log').rename(run / 'reference-shared-disk.log')
    record['reference_shared_disk_attempt'] = {
        'command': record.pop('reference_command'),
        'returncode': record.pop('reference_returncode', None),
        'wall_seconds': record.pop('reference_wall_seconds', None),
        'error': record.pop('error', None),
        'log': 'reference-shared-disk.log',
    }
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    os.environ['H3_RESUME_RUN'] = str(run)
    source = replace_once(
        source,
        "O=R/'outputs/unified_fl2va'/time.strftime('%Y%m%d_%H%M%S');O.mkdir(parents=True,exist_ok=False)",
        "O=Path(os.environ['H3_RESUME_RUN'])",
    )
    source = replace_once(
        source,
        "D=dict(phase='preflight',complete=False,official_pass_claimed=False,prompt='A dog falls asleep',output_directory=str(O))",
        "D=json.loads((O/'run.json').read_text())",
    )
    return replace_once(
        source,
        "  run(base[:1]+[str(R/'scripts/launch_h3.py')]+base[1:]+['--output='+str(O/'infinilm.mp4')],'infinilm',env,1800)",
        "  assert sha(O/'infinilm.mp4') == D['infinilm_sha256']",
    )


def generate(runtime, model, reference_model, resume_run=None):
    if not (model / 'model_index.json').is_file():
        raise FileNotFoundError(f'MiniMax-H3 model_index.json missing in {model}')
    runner = runtime / 'scripts/run_unified_dog.py'
    source = runner.read_text()
    source = source.replace(
        "'--model-path=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers'",
        "'--model-path='+os.environ['H3_LOCAL_MODEL']",
    )
    source = replace_once(
        source,
        ",'--height='+str(h),'--width='+str(w)",
        '',
    )
    source = replace_once(
        source,
        "  D['input_size']=size",
        "  assert (w, h) == (768, 928), (w, h)\n  D['input_size']=size",
    )
    source = replace_once(
        source,
        "  refenv=dict(env);",
        "  deadline = time.monotonic() + 2400\n"
        "  while not (R/'artifacts/reference-runtime-ready').is_file():\n"
        "   if time.monotonic() >= deadline:\n"
        "    raise TimeoutError('Reference runtime setup did not finish')\n"
        "   time.sleep(15)\n"
        "  refenv=dict(env);",
    )
    if reference_model != model:
        source = replace_once(
            source,
            "  run(base+['--output='+str(O/'reference.mp4')",
            "  reference_base = [('--model-path='+os.environ['H3_REFERENCE_MODEL'])"
            " if arg.startswith('--model-path=') else arg for arg in base]\n"
            "  run(reference_base+['--output='+str(O/'reference.mp4')",
        )
    if resume_run is not None:
        source = retry_reference(source, resume_run, reference_model)
    execute(source, runner, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build', 'reference', 'accept', 'generate'))
    parser.add_argument('--source', type=Path, default=WORKSPACE / 'h3')
    parser.add_argument('--runtime', type=Path, default=WORKSPACE / 'runtime')
    parser.add_argument('--cache', type=Path, default=WORKSPACE / 'input-cache')
    parser.add_argument('--model', type=Path, default=None)
    parser.add_argument('--resume-run', type=Path, help='Resume the reference and comparison for an existing run')
    parser.add_argument('--reference-model', type=Path, help='Path to the same weights staged for PyTorch')
    args = parser.parse_args()
    source = args.source.resolve()
    runtime = args.runtime.resolve()
    cache = args.cache.resolve()
    model = (args.model or Path(os.environ.get('H3_LOCAL_MODEL', DEFAULT_MODEL))).resolve()
    ref_model = (args.reference_model or Path(os.environ.get('H3_REFERENCE_MODEL', model))).resolve()
    os.environ.update(IFAB_AFS_HOME=str(runtime), H3_INPUT_CACHE=str(cache), H3_LOCAL_MODEL=str(model))

    if args.reference_model is not None or args.phase in ('accept', 'generate'):
        if not (ref_model / 'model_index.json').is_file():
            parser.error(f'MiniMax-H3 model_index.json missing in {ref_model}')
        os.environ['H3_REFERENCE_MODEL'] = str(ref_model)
    if args.resume_run is not None and (args.phase != 'generate' or ref_model == model):
        parser.error('--resume-run requires the generate phase and a staged reference model')
    if args.phase == 'generate':
        generate(runtime, model, ref_model, args.resume_run)
        return
    if args.phase == 'reference':
        prepare_reference(runtime, cache)
        return

    rebuild = source / 'rebuild.py'
    rebuild_source = rebuild.read_text()
    if args.phase == 'prepare':
        libraries = os.environ.get('LD_LIBRARY_PATH')
        gcvm = str(DTK_ROOT / 'dcc/gcvm/lib')
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (gcvm, libraries)))
        rebuild_source = replace_once(
            rebuild_source,
            "    run([sys.executable, root/'artifacts/rebuild-prepare.py']",
            "    deploy(HERE, root)\n"
            "    run([sys.executable, root/'artifacts/rebuild-prepare.py']",
        )
        sys.argv = [str(rebuild), '--destination', str(runtime), '--cache-root', str(cache),
                    '--phase', args.phase]
        exec(compile(rebuild_source, str(rebuild), 'exec'), {
            '__file__': str(rebuild), '__name__': '__main__', 'deploy': deploy,
        })
    else:
        execute(rebuild_source, rebuild, [
            '--destination', str(runtime), '--cache-root', str(cache), '--phase', args.phase,
        ])


if __name__ == '__main__':
    main()

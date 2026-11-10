#!/usr/bin/env python3
"""Run the pinned H3 entry with the recorded local runtime fixes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from prepare_public_sample import validate_local

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('infinilm', 'legacy'), default='infinilm')
    args = parser.parse_args()
    run_dir = ROOT / 'outputs/public_fl2va' / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}_{args.backend}')
    run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(backend=args.backend, generation_succeeded=False,
                  content_validation_passed=None, inference_passed=False,
                  video_specs_passed=False, similarity_evaluated=False,
                  competition_validation_passed=False, phase='prepare')
    try:
        case = validate_local()
        config = dict(case['generation'])
        if (config['num_frames'], config['steps'], config['seed'], config['fps']) != (124, 50, 42, 24):
            raise ValueError('Public validation expects 124 frames, 50 steps, seed 42 and 24 fps')
        manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
        if not manifest.get('all_content_hashes_verified') or manifest['revision'] != case['model_revision']:
            raise ValueError('The completed model manifest does not match the prepared public case')
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
        config['height'], config['width'] = resolve_canvas_size(
            *case['image_size'], 32, 768, 768 * 1344)
        storage = Path(os.getenv('SONGSHAN_STORAGE', '/work1/share/sugonhpcapp01/zhanghengyu/songshanbei'))
        lm = ROOT / 'sources/InfiniLM-build'
        if not (lm / 'examples/minimax_h3.py').is_file():
            lm = storage / 'sources/InfiniLM'
        command = [sys.executable, str(lm / 'examples/minimax_h3.py'), 'fl2va',
                   '--model-path=' + manifest['model_path'],
                   '--first-frame=' + case['first_frame']['path'],
                   '--text=' + Path(case['prompt_file']['path']).read_text(),
                   '--gpus=' + ','.join(map(str, config['gpus'])),
                   '--num-frames=' + str(config['num_frames']),
                   '--steps=' + str(config['steps']), '--seed=' + str(config['seed']),
                   '--height=' + str(config['height']), '--width=' + str(config['width']),
                   '--output=' + str(run_dir / 'video.mp4')]
        if args.backend == 'legacy':
            command.append('--legacy')
        else:
            command.insert(1, str(ROOT / 'scripts/launch_h3.py'))
        report.update(command=command, configuration=config,
                      first_frame_sha256=case['first_frame']['sha256'],
                      prompt_sha256=case['prompt_file']['sha256'], model_revision=manifest['revision'])
        import torch
        report['phase'] = 'runtime_dependencies'
        # The InfiniLM distribution currently declares no runtime requirements,
        # but its package import eagerly loads its scheduler and prefix cache.
        import janus
        import xxhash
        from importlib.metadata import version
        report['runtime_dependencies'] = {name: version(name) for name in ('janus', 'xxhash')}
        report['phase'] = 'preprocessing_check'
        from songshan_h3.preprocessing import check_model_components
        report['preprocessing'] = check_model_components(manifest['model_path'], case['first_frame']['path'])
        print('CPU processor loading and first-frame preprocessing checks passed.', flush=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() <= max(config['gpus']):
            raise RuntimeError(f"Requested 8 devices; PyTorch can access {torch.cuda.device_count()} in this session")
        if args.backend == 'infinilm':
            report['phase'] = 'native_import_check'
            from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder, MiniMaxH3TransformerRunner
            report['native_imports_passed'] = True
            report['phase'] = 'native_attention_check'
            from songshan_h3.attention_bridge import install
            from check_attention_bridge import check_device
            install()
            for device_id in config['gpus']:
                report.setdefault('native_attention_checks', {})[str(device_id)] = check_device(device_id)
            report['native_attention_check_passed'] = True
            report['native_attention_implementation'] = 'NineToothed JIT via InfiniCore host dispatcher'
            report['video_vae_attention_implementation'] = 'NineToothed JIT, head dimension 64'
        report['phase'] = 'generation'
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        started = time.monotonic()
        print(f"Running {args.backend}: 124 frames, 50 steps, seed 42, 8 devices.", flush=True)
        print(f'Generation log: {run_dir / "generation.log"}', flush=True)
        with (run_dir / 'generation.log').open('w') as log:
            result = subprocess.run(command, cwd=lm, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report.update(returncode=result.returncode, wall_seconds=time.monotonic() - started)
        report['clean_exit'] = result.returncode == 0
        if (run_dir / 'video.mp4').is_file():
            report['phase'] = 'output_specs'
            from songshan_h3.output_validation import validate_generated_media
            report.update(validate_generated_media(run_dir / 'video.mp4', config))
        if result.returncode:
            report['phase'] = 'process_exit_failure'
            print('Generation log tail:', flush=True)
            print('\n'.join((run_dir / 'generation.log').read_text(errors='replace').splitlines()[-40:]), flush=True)
            raise RuntimeError(f'Official generation entry exited {result.returncode}; see generation.log')
        report['phase'] = 'output_specs'
        if not report['video_specs_passed']:
            raise ValueError('Output dimensions, frames, FPS, H.264 or AAC stereo/32 kHz specs differ')
        report['generation_succeeded'] = True
        report['phase'] = 'complete'
        print(f'Generation process exited normally; media format checks passed: {run_dir / "video.mp4"}')
        print('Video content has NOT been validated. This is not an inference-quality or competition PASS.', flush=True)
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        (run_dir / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'Run record: {run_dir / "run.json"}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

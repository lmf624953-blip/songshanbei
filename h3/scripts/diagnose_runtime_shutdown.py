#!/usr/bin/env python3
"""Test native worker teardown using tiny synthetic models, without weight loading."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def native_backtrace(command, environment, path):
    process = subprocess.Popen(
        ['gdb', '-q', '-batch', '-ex', 'set pagination off', '-ex', 'set confirm off',
         '-ex', 'run', '-ex', 'thread apply all bt 15', '--args', *command],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        # Interrupt only this diagnostic's debugger and inferior, capture the
        # blocked native stacks, then ensure no diagnostic process is left over.
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    path.write_text(output)


def worker(devices):
    import faulthandler
    import gc
    import torch
    import infinicore
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig

    faulthandler.enable(all_threads=True)
    manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
    path = str(Path(manifest['model_path']) / 'transformer')
    overrides = dict(hidden_size=128, num_layers=1, num_refiner_layers=1,
                     num_attention_heads=8, attention_head_dim=128, ffn_dim=256,
                     time_embed_hidden_dim=128, time_embed_dim=64)
    engines = []
    for index in range(2):
        print(f'Creating tiny engine {index}; no checkpoint weights will be loaded', flush=True)
        engines.append(InferEngine(path, device=infinicore.device('cuda', devices[0]),
                                  distributed_config=DistConfig(tp_device_ids=devices),
                                  hf_config_overrides=overrides))
    for index, engine in enumerate(reversed(engines)):
        print(f'Closing tiny engine {index}', flush=True)
        engine.close()
        engine.close()  # Explicit shutdown must be idempotent.
        print(f'Tiny engine {index} closed', flush=True)
    del engine
    engines.clear()
    gc.collect()
    for device in devices:
        torch.cuda.synchronize(device)
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if '/libnl-' in line})
    (ROOT / 'logs/runtime-shutdown-libraries.json').write_text(
        json.dumps(dict(pid=os.getpid(), libraries=libraries), indent=2) + '\n')
    print('Loaded libnl libraries: ' + json.dumps(libraries), flush=True)
    for stem in ('libnl-3', 'libnl-route-3'):
        matching = [path for path in libraries if Path(path).name.startswith(stem)]
        if len(matching) > 1:
            raise RuntimeError(f'Duplicate {stem} libraries; source scripts/activate.sh again: {matching}')
    print('Worker Python cleanup completed; now testing interpreter exit', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gdb-on-failure', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    devices = [int(item) for item in args.devices.split(',')]
    if args.worker:
        worker(devices)
        return 0
    import torch
    report = dict(diagnostic_only=True, checkpoint_weights_loaded=False,
                  shutdown_passed=False, inference_passed=False, devices=devices)
    try:
        if not devices or min(devices) < 0 or max(devices) >= torch.cuda.device_count():
            raise RuntimeError(f'This session exposes {torch.cuda.device_count()} accelerators')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--devices', args.devices]
        environment = {**os.environ, 'PYTHONFAULTHANDLER': '1'}
        log = ROOT / 'logs/runtime-shutdown-diagnostic.log'
        library_log = ROOT / 'logs/runtime-shutdown-libraries.json'
        library_log.unlink(missing_ok=True)
        try:
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
            log.write_text(result.stdout + result.stderr)
            print(result.stdout + result.stderr, end='', flush=True)
            report.update(returncode=result.returncode, log=str(log), shutdown_passed=result.returncode == 0)
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b'') + (error.stderr or b'')
            log.write_bytes(output)
            report.update(timed_out=True, log=str(log), error='Shutdown diagnostic exceeded 120 seconds')
            print(report['error'], flush=True)
        if library_log.is_file():
            report['mapped_netlink'] = json.loads(library_log.read_text())
        if not report['shutdown_passed'] and args.gdb_on_failure:
            gdb_log = ROOT / 'logs/runtime-shutdown-gdb.log'
            native_backtrace(command, environment, gdb_log)
            report['gdb_log'] = str(gdb_log)
            print('Native backtrace: ' + str(gdb_log), flush=True)
        print('Native shutdown diagnostic: ' + ('PASS' if report['shutdown_passed'] else 'FAIL'), flush=True)
        return 0 if report['shutdown_passed'] else 1
    except Exception as error:
        report['error'] = str(error)
        print(report['error'], flush=True)
        return 1
    finally:
        (ROOT / 'artifacts/runtime-shutdown-diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())

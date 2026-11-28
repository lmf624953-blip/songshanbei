#!/usr/bin/env python3
"""仅验证正式subprocess入口的CPU共享存储传递与退出，不运行模型。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import torch

from process_runtime import Workers


BOOTSTRAP = r'''
import runpy,sys
from pathlib import Path
entry=Path(sys.argv[1]);sys.path.insert(0,str(entry.parent))
import bench
def probe(model,rank,connection,log):
    connection.send(("ready",rank,"torch.bfloat16"))
    while True:
        command=connection.recv()
        if command[0]=="stop":break
        kind,request,jobs=command
        assert kind=="run"
        connection.send(("done",request,[(i,t.clone()) for i,t in jobs],0.0,0,{}))
    connection.close()
bench.worker=probe
sys.argv=[str(entry),*sys.argv[2:]]
runpy.run_path(str(entry),run_name="__main__")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    original = subprocess.Popen

    def start(command, **kwargs):
        return original([*command[:2], "-c", BOOTSTRAP, *command[2:]], **kwargs)

    with tempfile.TemporaryDirectory(prefix="h3-vae-ipc-check-") as temporary:
        pool = None
        try:
            with patch("process_runtime.subprocess.Popen", start):
                pool = Workers(Path("unused-model"), 2, Path(temporary))
            source = torch.arange(4096, dtype=torch.int16).view(torch.bfloat16)
            checks = []
            for trial in range(3):
                source.view(torch.int16).add_(1)
                for connection in pool.connections:
                    connection.send(("run", trial, [(0, source)]))
                for rank in range(2):
                    kind, request, outputs, *_ = pool.receive(rank)
                    assert kind == "done" and request == trial
                    assert torch.equal(outputs[0][1].view(torch.int16), source.view(torch.int16))
                    checks.append(dict(rank=rank, trial=trial, exact=True))
        finally:
            if pool is not None:
                pool.close()
        assert pool is not None and all(p.exitcode == 0 for p in pool.processes), [p.exitcode for p in pool.processes]
        result = dict(complete=True, scope="CPU IPC协议检查，不是模型数值或性能结果", checks=checks,
                      worker_exit_codes=[p.exitcode for p in pool.processes],
                      source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ("worker-entry.py", "process_runtime.py", "bench.py")})
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(dict(complete=True, cpu_checks=len(checks), worker_exit_codes=result["worker_exit_codes"])))


if __name__ == "__main__":
    main()
